"""Clip-consistent augmentation.

The rule that matters: a *clip* shares one base geometric transform, with only
small per-frame jitter on top.  Sampling an independent crop per frame would
scramble the apparent motion and teach the association head that instances
teleport, which is worse than no augmentation at all.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Sequence, Tuple

import cv2
import numpy as np


@dataclass
class AugConfig:
    short_side: int = 640
    max_long_side: int = 1280
    crop_size: Tuple[int, int] = (640, 640)
    scale_range: Tuple[float, float] = (0.6, 1.8)
    rotate_deg: float = 8.0
    frame_jitter_px: float = 2.0        # per-frame translation jitter
    frame_jitter_deg: float = 0.5
    hflip_prob: float = 0.0             # off by default: text is not mirror-symmetric
    colour_prob: float = 0.5
    brightness: float = 0.3
    contrast: float = 0.3
    saturation: float = 0.3
    blur_prob: float = 0.15
    jpeg_prob: float = 0.15
    keep_ratio: bool = True
    text_crop_prob: float = 0.85   # chance the crop is anchored on an instance


def transform_points(pts: np.ndarray, H: np.ndarray) -> np.ndarray:
    p = np.concatenate([pts, np.ones((len(pts), 1), np.float32)], axis=1)
    out = (H @ p.T).T
    return (out[:, :2] / np.maximum(out[:, 2:3], 1e-6)).astype(np.float32)


def resize_homography(src_hw: Tuple[int, int], dst_hw: Tuple[int, int],
                      keep_ratio: bool = True) -> np.ndarray:
    sh, sw = src_hw
    dh, dw = dst_hw
    if keep_ratio:
        s = min(dh / sh, dw / sw)
        sx = sy = s
    else:
        sx, sy = dw / sw, dh / sh
    return np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], np.float32)


class ClipAugmentor:
    """Builds one base homography per clip plus per-frame jitter."""

    def __init__(self, cfg: AugConfig, training: bool = True,
                 rng: random.Random | None = None):
        self.cfg = cfg
        self.training = training
        self.rng = rng or random.Random()

    def base_homography(self, src_hw: Tuple[int, int],
                        keypoints: np.ndarray | None = None
                        ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Sample the clip's base transform.

        ``keypoints`` are instance centres in source coordinates.  When given,
        the crop window is biased to contain one of them.  Without this a large
        fraction of crops land on empty background: they cost a full forward and
        backward pass and contribute nothing but negative pixels, which for a
        from-scratch model is exactly the gradient it needs least.
        """
        cfg, rng = self.cfg, self.rng
        sh, sw = src_hw
        if not self.training:
            # Deterministic letterbox to a stride-32 multiple.
            scale = min(cfg.short_side / min(sh, sw), cfg.max_long_side / max(sh, sw))
            out_h = int(math.ceil(sh * scale / 32) * 32)
            out_w = int(math.ceil(sw * scale / 32) * 32)
            H = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], np.float32)
            return H, (out_h, out_w)

        ch, cw = cfg.crop_size
        base_scale = min(cfg.short_side / min(sh, sw), cfg.max_long_side / max(sh, sw))
        scale = base_scale * rng.uniform(*cfg.scale_range)
        angle = rng.uniform(-cfg.rotate_deg, cfg.rotate_deg)

        M = cv2.getRotationMatrix2D((sw / 2.0, sh / 2.0), angle, scale)
        H = np.vstack([M, [0, 0, 1]]).astype(np.float32)

        corners = transform_points(
            np.array([[0, 0], [sw, 0], [sw, sh], [0, sh]], np.float32), H)
        min_xy = corners.min(axis=0)
        max_xy = corners.max(axis=0)
        span = max_xy - min_xy

        anchor = None
        if keypoints is not None and len(keypoints) and rng.random() < self.cfg.text_crop_prob:
            kp = transform_points(np.asarray(keypoints, np.float32).reshape(-1, 2), H)
            inside = kp[(kp[:, 0] >= min_xy[0]) & (kp[:, 0] <= max_xy[0])
                        & (kp[:, 1] >= min_xy[1]) & (kp[:, 1] <= max_xy[1])]
            if len(inside):
                anchor = inside[rng.randrange(len(inside))]

        def offset(axis: int, out_size: float) -> float:
            lo, hi = float(min_xy[axis]), float(max_xy[axis])
            if hi - lo <= out_size:
                return lo - (out_size - (hi - lo)) / 2.0
            if anchor is not None:
                # keep the anchor inside the window, with a random position in it
                target = float(anchor[axis]) - rng.uniform(0.15, 0.85) * out_size
                return float(np.clip(target, lo, hi - out_size))
            return lo + rng.uniform(0, hi - lo - out_size)

        tx = offset(0, cw)
        ty = offset(1, ch)
        H = np.array([[1, 0, -tx], [0, 1, -ty], [0, 0, 1]], np.float32) @ H

        if cfg.hflip_prob > 0 and rng.random() < cfg.hflip_prob:
            F = np.array([[-1, 0, cw], [0, 1, 0], [0, 0, 1]], np.float32)
            H = F @ H
        return H, (ch, cw)

    def frame_homography(self, base: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
        if not self.training or self.cfg.frame_jitter_px <= 0:
            return base
        rng, cfg = self.rng, self.cfg
        h, w = out_hw
        angle = rng.uniform(-cfg.frame_jitter_deg, cfg.frame_jitter_deg)
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        J = np.vstack([M, [0, 0, 1]]).astype(np.float32)
        J[0, 2] += rng.uniform(-cfg.frame_jitter_px, cfg.frame_jitter_px)
        J[1, 2] += rng.uniform(-cfg.frame_jitter_px, cfg.frame_jitter_px)
        return J @ base

    def warp_frame(self, image: np.ndarray, H: np.ndarray,
                   out_hw: Tuple[int, int]) -> np.ndarray:
        h, w = out_hw
        return cv2.warpPerspective(image, H, (w, h), flags=cv2.INTER_LINEAR,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    def photometric(self, image: np.ndarray) -> np.ndarray:
        """Per-frame colour jitter -- real video has per-frame exposure changes."""
        if not self.training:
            return image
        cfg, rng = self.cfg, self.rng
        out = image.astype(np.float32)
        if rng.random() < cfg.colour_prob:
            out *= 1.0 + rng.uniform(-cfg.brightness, cfg.brightness)
            mean = out.mean()
            out = (out - mean) * (1.0 + rng.uniform(-cfg.contrast, cfg.contrast)) + mean
            grey = out.mean(axis=2, keepdims=True)
            out = grey + (out - grey) * (1.0 + rng.uniform(-cfg.saturation, cfg.saturation))
        out = np.clip(out, 0, 255).astype(np.uint8)
        if rng.random() < cfg.blur_prob:
            out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.3, 1.5))
        if rng.random() < cfg.jpeg_prob:
            ok, enc = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY),
                                                 rng.randint(40, 95)])
            if ok:
                out = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        return out


def clip_polygon_to_image(poly: np.ndarray, height: int, width: int
                          ) -> Tuple[np.ndarray, float]:
    """Clamp a polygon into the frame; returns ``(clipped, kept_area_ratio)``.

    The ratio drives the ignore decision: a polygon that is mostly outside the
    crop has an unreliable transcription (half the word is gone) and must not
    supervise the recognition head, but it *is* still text, so it becomes an
    ignore region rather than background.
    """
    from ..utils.polygon import poly_area
    before = poly_area(poly)
    clipped = poly.copy()
    clipped[:, 0] = np.clip(clipped[:, 0], 0, width - 1)
    clipped[:, 1] = np.clip(clipped[:, 1], 0, height - 1)
    after = poly_area(clipped)
    ratio = float(after / before) if before > 1e-6 else 0.0
    return clipped, ratio
