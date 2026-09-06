"""Synthetic video text sequences with exact trajectory ground truth.

The tracking half of a video text spotter needs supervision that no static
dataset can give: the same instance seen across frames under motion, blur,
scale change and occlusion.  Real video-text annotation is polygon-per-frame
plus identity, which is why every public benchmark combined is only tens of
thousands of annotated frames -- far too few to train an association embedding
from random initialisation.

VimTS's answer was VTD-368k, built by propagating text through Content
Deformation Fields.  CoDeF needs a per-video optimisation and a pretrained
generative stack, so it is out of reach here.  The approach below gets the
properties that actually matter for learning association -- identity across
time, realistic motion, motion blur coupled to velocity, and occlusion events
that force long-term re-identification -- from compositing alone:

* every instance is rendered once and moved by a per-frame homography, so the
  polygon ground truth is exact by construction, not estimated;
* a global camera homography (pan / zoom / roll) is composed with per-instance
  motion, so instances move coherently the way scene text does;
* motion blur is applied along the true inter-frame displacement direction,
  which is the single most characteristic degradation of video text;
* occluders sweep across the frame, producing the disappear/reappear events
  that a short-term matcher cannot solve and a long-term one must.

The honest limitation: composited text does not inherit scene lighting or
non-rigid surface deformation.  This is pretraining, not a substitute for the
real-data fine-tuning stage.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import ImageFont

from .synth_static import (BackgroundSampler, SynthConfig, _alpha_composite,
                           _apply_homography, _poly_overlaps, discover_fonts,
                           pick_text_color, render_word_layer, sample_text)


@dataclass
class VideoSynthConfig:
    width: int = 640
    height: int = 360
    num_frames: int = 24
    fps: float = 25.0
    min_instances: int = 4
    max_instances: int = 16
    # global camera motion per frame
    cam_translate: float = 6.0        # pixels/frame
    cam_rotate: float = 0.6           # degrees/frame
    cam_zoom: float = 0.004           # relative scale change/frame
    # per-instance motion on top of the camera
    instance_translate: float = 2.0
    instance_prob_moving: float = 0.4
    # degradations
    motion_blur: bool = True
    max_blur_kernel: int = 15
    noise_std: float = 4.0
    jpeg_prob: float = 0.4
    # occlusion
    num_occluders: int = 2
    occluder_prob: float = 0.6
    # an instance is annotated only when this much of it is visible
    min_visible_ratio: float = 0.35
    static: SynthConfig = field(default_factory=lambda: SynthConfig(curve_prob=0.3))


@dataclass
class _Sprite:
    layer: np.ndarray          # RGBA
    poly: np.ndarray           # polygon in layer coordinates
    text: str
    track_id: int
    origin: Tuple[float, float]
    velocity: Tuple[float, float]


def _translation(dx: float, dy: float) -> np.ndarray:
    return np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]], np.float32)


def _camera_homography(t: int, rng_state: dict, cfg: VideoSynthConfig) -> np.ndarray:
    """Smooth camera motion at frame ``t`` (sinusoidal, so it never runs away)."""
    phase = rng_state["phase"]
    tx = rng_state["dir"][0] * cfg.cam_translate * t
    ty = rng_state["dir"][1] * cfg.cam_translate * t
    angle = cfg.cam_rotate * math.sin(0.15 * t + phase)
    scale = 1.0 + cfg.cam_zoom * t * rng_state["zoom_dir"]
    cx, cy = cfg.width / 2.0, cfg.height / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle, scale)
    H = np.vstack([M, [0, 0, 1]]).astype(np.float32)
    return _translation(tx, ty) @ H


def _motion_blur(img: np.ndarray, dx: float, dy: float, max_k: int) -> np.ndarray:
    """Directional blur along the displacement vector."""
    mag = math.hypot(dx, dy)
    if mag < 1.0:
        return img
    k = int(min(max(int(mag), 3) | 1, max_k | 1))
    kernel = np.zeros((k, k), np.float32)
    angle = math.atan2(dy, dx)
    cx = cy = k // 2
    for i in range(k):
        offset = i - cx
        x = int(round(cx + offset * math.cos(angle)))
        y = int(round(cy + offset * math.sin(angle)))
        if 0 <= x < k and 0 <= y < k:
            kernel[y, x] = 1.0
    total = kernel.sum()
    if total < 1:
        return img
    return cv2.filter2D(img, -1, kernel / total)


def _visible_ratio(poly: np.ndarray, occluders: Sequence[np.ndarray],
                   width: int, height: int) -> float:
    """Fraction of the polygon inside the frame and not covered by an occluder."""
    x1 = int(np.floor(poly[:, 0].min())); x2 = int(np.ceil(poly[:, 0].max()))
    y1 = int(np.floor(poly[:, 1].min())); y2 = int(np.ceil(poly[:, 1].max()))
    if x2 <= x1 or y2 <= y1:
        return 0.0
    w, h = x2 - x1, y2 - y1
    if w * h > 4_000_000:
        return 0.0
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, [(poly - [x1, y1]).astype(np.int32)], 1)
    total = int(mask.sum())
    if total == 0:
        return 0.0
    frame = np.zeros_like(mask)
    fx1, fy1 = max(0 - x1, 0), max(0 - y1, 0)
    fx2, fy2 = min(width - x1, w), min(height - y1, h)
    if fx2 <= fx1 or fy2 <= fy1:
        return 0.0
    frame[fy1:fy2, fx1:fx2] = 1
    vis = mask * frame
    for occ in occluders:
        cv2.fillPoly(vis, [(occ - [x1, y1]).astype(np.int32)], 0)
    return float(vis.sum()) / total


class SyntheticVideoGenerator:
    """Generate a clip of frames plus exact per-frame polygons and track ids."""

    def __init__(self, cfg: Optional[VideoSynthConfig] = None,
                 fonts: Optional[Sequence[str]] = None,
                 background_dir: Optional[str] = None, seed: Optional[int] = None):
        self.cfg = cfg or VideoSynthConfig()
        self.rng = random.Random(seed)
        self.fonts = list(fonts) if fonts else discover_fonts()
        if not self.fonts:
            raise RuntimeError("no TrueType fonts found; install fonts or pass fonts=[...]")
        self.backgrounds = BackgroundSampler(background_dir, rng=self.rng)

    def generate(self) -> Tuple[List[np.ndarray], List[List[dict]]]:
        """Returns ``(frames, per_frame_instances)``.

        Each instance dict is ``{polygon, text, track_id, visible_ratio}``.
        """
        cfg, rng = self.cfg, self.rng
        base_bg = self.backgrounds.sample(int(cfg.width * 1.6), int(cfg.height * 1.6))
        sprites = self._make_sprites()
        cam_state = {
            "phase": rng.uniform(0, math.tau),
            "dir": (rng.uniform(-1, 1), rng.uniform(-1, 1)),
            "zoom_dir": rng.choice([-1.0, 1.0]),
        }
        occluder_tracks = self._make_occluders()

        frames: List[np.ndarray] = []
        annotations: List[List[dict]] = []
        prev_centres: dict[int, Tuple[float, float]] = {}

        for t in range(cfg.num_frames):
            cam = _camera_homography(t, cam_state, cfg)
            frame = self._warp_background(base_bg, cam)
            occluders = [occ[t] for occ in occluder_tracks]
            instances: List[dict] = []

            for sprite in sprites:
                H = self._sprite_homography(sprite, cam, t)
                poly = _apply_homography(sprite.poly, H)
                centre = (float(poly[:, 0].mean()), float(poly[:, 1].mean()))
                prev = prev_centres.get(sprite.track_id, centre)
                dx, dy = centre[0] - prev[0], centre[1] - prev[1]
                prev_centres[sprite.track_id] = centre

                warped = self._warp_sprite(sprite, H)
                if warped is None:
                    continue
                if cfg.motion_blur:
                    warped = _motion_blur(warped, dx, dy, cfg.max_blur_kernel)
                _alpha_composite(frame, warped, 0, 0)

                ratio = _visible_ratio(poly, occluders, cfg.width, cfg.height)
                if ratio >= cfg.min_visible_ratio:
                    instances.append({"polygon": poly.tolist(), "text": sprite.text,
                                      "track_id": sprite.track_id,
                                      "visible_ratio": ratio})

            for occ in occluders:
                colour = (40, 40, 40)
                cv2.fillPoly(frame, [occ.astype(np.int32)], colour)

            frames.append(self._degrade(frame))
            annotations.append(instances)

        return frames, annotations

    # -- internals -------------------------------------------------------
    def _make_sprites(self) -> List[_Sprite]:
        cfg, rng = self.cfg, self.rng
        scfg = cfg.static
        sprites: List[_Sprite] = []
        placed: List[np.ndarray] = []
        target = rng.randint(cfg.min_instances, cfg.max_instances)
        attempts = 0
        while len(sprites) < target and attempts < target * 12:
            attempts += 1
            text = sample_text(rng)
            size = int(scfg.min_font_size + (scfg.max_font_size - scfg.min_font_size)
                       * rng.random() ** 2.0)
            try:
                font = ImageFont.truetype(rng.choice(self.fonts), size)
            except Exception:
                continue
            curvature = (rng.uniform(-scfg.max_curvature, scfg.max_curvature)
                         if rng.random() < scfg.curve_prob else 0.0)
            try:
                layer_img, poly = render_word_layer(
                    text, font, curvature, (255, 255, 255))
            except Exception:
                continue
            lw, lh = layer_img.size
            if lw >= cfg.width or lh >= cfg.height:
                continue
            x0 = rng.uniform(0, cfg.width - lw)
            y0 = rng.uniform(0, cfg.height - lh)
            world_poly = poly + np.array([x0, y0], np.float32)
            if any(_poly_overlaps(world_poly, p) for p in placed):
                continue

            colour = pick_text_color(
                np.full((4, 4, 3), rng.randint(0, 255), np.uint8),
                scfg.min_contrast, rng)
            layer_img, poly = render_word_layer(text, font, curvature, colour)
            layer_np = np.array(layer_img)

            moving = rng.random() < cfg.instance_prob_moving
            vel = ((rng.uniform(-1, 1) * cfg.instance_translate,
                    rng.uniform(-1, 1) * cfg.instance_translate) if moving else (0.0, 0.0))
            sprites.append(_Sprite(layer_np, poly, text, len(sprites), (x0, y0), vel))
            placed.append(world_poly)
        return sprites

    def _sprite_homography(self, sprite: _Sprite, cam: np.ndarray, t: int) -> np.ndarray:
        ox, oy = sprite.origin
        vx, vy = sprite.velocity
        return cam @ _translation(ox + vx * t, oy + vy * t)

    def _warp_sprite(self, sprite: _Sprite, H: np.ndarray) -> Optional[np.ndarray]:
        cfg = self.cfg
        poly = _apply_homography(sprite.poly, H)
        if (poly[:, 0].max() < 0 or poly[:, 1].max() < 0
                or poly[:, 0].min() >= cfg.width or poly[:, 1].min() >= cfg.height):
            return None
        return cv2.warpPerspective(sprite.layer, H, (cfg.width, cfg.height),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT,
                                   borderValue=(0, 0, 0, 0))

    def _warp_background(self, bg: np.ndarray, cam: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        return cv2.warpPerspective(bg, cam, (cfg.width, cfg.height),
                                   flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_REFLECT)

    def _make_occluders(self) -> List[List[np.ndarray]]:
        cfg, rng = self.cfg, self.rng
        tracks: List[List[np.ndarray]] = []
        for _ in range(cfg.num_occluders):
            if rng.random() > cfg.occluder_prob:
                continue
            w = rng.uniform(0.08, 0.25) * cfg.width
            h = rng.uniform(0.3, 1.2) * cfg.height
            y = rng.uniform(-0.2, 0.4) * cfg.height
            x0 = rng.uniform(-w, cfg.width)
            speed = rng.uniform(-1, 1) * cfg.width / max(cfg.num_frames, 1) * 1.5
            frames = []
            for t in range(cfg.num_frames):
                x = x0 + speed * t
                frames.append(np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]],
                                       np.float32))
            tracks.append(frames)
        return tracks

    def _degrade(self, frame: np.ndarray) -> np.ndarray:
        rng = self.rng
        out = frame.astype(np.float32)
        if self.cfg.noise_std > 0:
            out += np.random.randn(*out.shape).astype(np.float32) * self.cfg.noise_std
        out = np.clip(out, 0, 255).astype(np.uint8)
        if rng.random() < self.cfg.jpeg_prob:
            ok, enc = cv2.imencode(".jpg", out,
                                   [int(cv2.IMWRITE_JPEG_QUALITY), rng.randint(40, 90)])
            if ok:
                out = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        return out
