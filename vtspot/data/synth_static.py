"""Procedural synthetic scene-text image generator.

This is not a nicety -- for a model with no pretrained weights it *is* the
pretraining corpus.  ImageNet supplies a ResNet's early filters for free; with
that removed, edges, strokes and character shapes have to be learnt from
something, and hand-annotated video text (a few thousand frames in total across
every public benchmark) is nowhere near enough.  Millions of synthetic
characters are.

What is deliberately modelled here, and why:

* **Curved text.**  Rendered character-by-character along an arc, with the
  polygon built from per-character quads.  Straight-only synthetic data produces
  a model that cannot represent curved instances at all, which is >30% of
  ArTVideo.
* **Low contrast and small sizes.**  The sampler is biased towards small text,
  because that is what video benchmarks are made of and what a model trained on
  comfortable synthetic text fails on.
* **Perspective.**  Text on real surfaces is rarely fronto-parallel.

No external assets are required: backgrounds are procedural and fonts are
discovered from the system.  Point ``--backgrounds`` at a directory of real
photographs when you have one -- it measurably narrows the domain gap -- but the
pipeline runs without it.
"""

from __future__ import annotations

import math
import random
import string
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_SEARCH_DIRS = [
    "/usr/share/fonts", "/usr/local/share/fonts", "/Library/Fonts",
    "/System/Library/Fonts", str(Path.home() / ".fonts"),
]

# A small built-in lexicon keeps synthetic transcriptions word-like rather than
# uniform random noise.  Real text has structure (digraph frequencies, word
# lengths) and a recogniser trained only on uniform strings learns no useful
# prior at all.
LEXICON = """the of and to in is you that it he was for on are as with his they at
be this have from or one had by word but not what all were we when your can said
there use an each which she do how their if will up other about out many then them
these so some her would make like him into time has look two more write go see
number no way could people my than first water been call who oil its now find long
down day did get come made may part exit open closed shop store cafe hotel bank
station road street avenue market pharmacy parking entrance sale price menu coffee
pizza burger taxi bus train airport hospital school library museum theatre cinema
STOP YIELD SLOW LEFT RIGHT NORTH SOUTH EAST WEST""".split()

DIGIT_PATTERNS = ["{d}{d}", "{d}{d}{d}", "{d}{d}:{d}{d}", "{d}{d}.{d}{d}",
                  "{d}{d}{d}{d}", "A{d}{d}", "No.{d}{d}"]


def discover_fonts(extra_dirs: Sequence[str] = ()) -> List[str]:
    """All usable TTF/OTF files on this machine."""
    fonts: List[str] = []
    for d in list(FONT_SEARCH_DIRS) + list(extra_dirs):
        p = Path(d)
        if not p.is_dir():
            continue
        for ext in ("*.ttf", "*.otf", "*.TTF", "*.OTF"):
            fonts.extend(str(f) for f in p.rglob(ext))
    return sorted(set(fonts))


@dataclass
class SynthConfig:
    width: int = 640
    height: int = 640
    min_words: int = 3
    max_words: int = 12
    min_font_size: int = 10          # deliberately small -- video text is small
    max_font_size: int = 56
    curve_prob: float = 0.25         # fraction of instances rendered on an arc
    max_curvature: float = 0.9
    rotate_prob: float = 0.6
    max_rotation: float = 25.0
    perspective_prob: float = 0.3
    min_contrast: float = 60.0       # in 0-255 luma, keeps text readable
    blur_prob: float = 0.3
    noise_prob: float = 0.4
    charset: str = string.ascii_letters + string.digits
    max_place_attempts: int = 20


class BackgroundSampler:
    """Procedural backgrounds, or crops from a directory of real images."""

    def __init__(self, image_dir: Optional[str] = None, rng: Optional[random.Random] = None):
        self.rng = rng or random.Random()
        self.paths: List[Path] = []
        if image_dir:
            d = Path(image_dir)
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.PNG"):
                self.paths.extend(d.rglob(ext))

    def sample(self, width: int, height: int) -> np.ndarray:
        if self.paths:
            img = self._sample_real(width, height)
            if img is not None:
                return img
        return self._sample_procedural(width, height)

    def _sample_real(self, width: int, height: int) -> Optional[np.ndarray]:
        for _ in range(3):
            path = self.rng.choice(self.paths)
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img is None:
                continue
            h, w = img.shape[:2]
            scale = max(width / w, height / h, 1.0)
            if scale > 1.0:
                img = cv2.resize(img, (int(w * scale) + 1, int(h * scale) + 1))
                h, w = img.shape[:2]
            x = self.rng.randint(0, max(w - width, 0))
            y = self.rng.randint(0, max(h - height, 0))
            return img[y:y + height, x:x + width].copy()
        return None

    def _sample_procedural(self, width: int, height: int) -> np.ndarray:
        rng = self.rng
        kind = rng.choice(["gradient", "noise", "blocks", "stripes"])
        c1 = np.array([rng.randint(0, 255) for _ in range(3)], dtype=np.float32)
        c2 = np.array([rng.randint(0, 255) for _ in range(3)], dtype=np.float32)

        if kind == "gradient":
            angle = rng.uniform(0, math.pi)
            yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
            t = (xx * math.cos(angle) + yy * math.sin(angle))
            t = (t - t.min()) / max(t.max() - t.min(), 1e-6)
            img = c1[None, None] * (1 - t[..., None]) + c2[None, None] * t[..., None]
        elif kind == "noise":
            small = np.random.rand(max(height // 16, 2), max(width // 16, 2), 3).astype(np.float32)
            img = cv2.resize(small, (width, height), interpolation=cv2.INTER_CUBIC)
            img = c1[None, None] * (1 - img) + c2[None, None] * img
        elif kind == "blocks":
            img = np.tile(c1[None, None], (height, width, 1))
            for _ in range(rng.randint(3, 12)):
                x1, y1 = rng.randint(0, width - 1), rng.randint(0, height - 1)
                x2 = min(x1 + rng.randint(20, width // 2), width)
                y2 = min(y1 + rng.randint(20, height // 2), height)
                col = np.array([rng.randint(0, 255) for _ in range(3)], dtype=np.float32)
                img[y1:y2, x1:x2] = col
        else:  # stripes
            period = rng.randint(8, 60)
            yy, xx = np.mgrid[0:height, 0:width]
            band = (((xx + yy) // period) % 2).astype(np.float32)
            img = c1[None, None] * (1 - band[..., None]) + c2[None, None] * band[..., None]

        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.5, 3.0))
        return np.clip(img, 0, 255).astype(np.uint8)


def _luma(color: Sequence[int]) -> float:
    b, g, r = color[0], color[1], color[2]
    return 0.114 * b + 0.587 * g + 0.299 * r


def pick_text_color(background_patch: np.ndarray, min_contrast: float,
                    rng: random.Random) -> Tuple[int, int, int]:
    """A colour that is actually readable against this patch."""
    bg_luma = float(background_patch.reshape(-1, 3).mean(axis=0) @ np.array([0.114, 0.587, 0.299]))
    for _ in range(12):
        color = tuple(rng.randint(0, 255) for _ in range(3))
        if abs(_luma(color) - bg_luma) >= min_contrast:
            return color
    return (255, 255, 255) if bg_luma < 128 else (0, 0, 0)


def sample_text(rng: random.Random) -> str:
    r = rng.random()
    if r < 0.15:
        pattern = rng.choice(DIGIT_PATTERNS)
        return pattern.replace("{d}", "").join([]) or "".join(
            str(rng.randint(0, 9)) if ch == "d" else ch
            for ch in pattern.replace("{d}", "d"))
    word = rng.choice(LEXICON)
    if r < 0.35:
        word = word.upper()
    elif r < 0.45:
        word = word.capitalize()
    if r > 0.92:
        word = word + str(rng.randint(1, 99))
    return word


def render_word_layer(text: str, font: ImageFont.FreeTypeFont, curvature: float,
                      color: Tuple[int, int, int], pad_ratio: float = 0.06
                      ) -> Tuple[Image.Image, np.ndarray]:
    """Render one word into an RGBA layer and return its control polygon.

    ``curvature`` in ``[-1, 1]`` bends the baseline into an arc; 0 is straight.

    The polygon uses the *tight ink extent* of the word (``font.getbbox``), not
    the font's ascender/descender metrics.  Metrics-based boxes are ~1.7x taller
    than the glyphs for an all-caps word, and a detector trained on loose boxes
    predicts loose boxes -- which costs IoU against ICDAR annotations, which are
    drawn tightly around the ink.

    Each character is placed on the arc and the polygon takes that character's
    two top and two bottom corners, so the polygon follows the curve instead of
    boxing it.
    """
    if not text:
        raise ValueError("empty text")

    widths = [font.getlength(ch) for ch in text]
    total_w = float(sum(widths))
    bbox = font.getbbox(text)                     # (x0, y0, x1, y1) from anchor "la"
    band_top, band_bot = float(bbox[1]), float(bbox[3])
    band_h = max(band_bot - band_top, 1.0)
    pad = band_h * pad_ratio
    band_top -= pad
    band_bot += pad
    band_h = band_bot - band_top
    band_mid = (band_top + band_bot) / 2.0

    radius = 0.0
    if abs(curvature) > 1e-3:
        radius = total_w / (abs(curvature) * math.pi)
    bulge = 0.0
    if radius > 0:
        half = min(total_w / 2.0, radius)
        bulge = radius - math.sqrt(max(radius ** 2 - half ** 2, 0.0))

    margin = int(band_h + 4)
    layer_w = int(total_w + 2 * margin)
    layer_h = int(band_h + 2 * bulge + 2 * margin)
    layer = Image.new("RGBA", (max(layer_w, 1), max(layer_h, 1)), (0, 0, 0, 0))

    cx = float(margin)
    cy = float(margin + bulge)                    # y of band_mid at the word centre

    glyph_size = int(2 * band_h + 2 * margin)
    top_pts: List[List[float]] = []
    bottom_pts: List[List[float]] = []
    x_cursor = 0.0

    def place(dx: float, dy: float, angle_rad: float,
              ctr: Tuple[float, float]) -> List[float]:
        """Rotate an offset from the character centre and translate to layer coords."""
        ca, sa = math.cos(angle_rad), math.sin(angle_rad)
        return [ctr[0] + dx * ca - dy * sa, ctr[1] + dx * sa + dy * ca]

    for ch, w in zip(text, widths):
        x_mid = x_cursor + w / 2.0
        if radius > 0:
            offset = x_mid - total_w / 2.0
            theta = offset / radius
            sign = 1.0 if curvature > 0 else -1.0
            dy = sign * radius * (1.0 - math.cos(theta))
            angle = sign * theta                  # image y points down
        else:
            dy, angle = 0.0, 0.0

        centre = (cx + x_mid, cy + dy)

        glyph = Image.new("RGBA", (glyph_size, glyph_size), (0, 0, 0, 0))
        # anchor "la": ink spans y in [draw_y + bbox[1], draw_y + bbox[3]], so
        # drawing at -band_mid puts the band centre on the canvas centre.
        ImageDraw.Draw(glyph).text((glyph_size / 2.0 - w / 2.0,
                                    glyph_size / 2.0 - band_mid),
                                   ch, font=font, fill=color + (255,))
        if abs(angle) > 1e-3:
            glyph = glyph.rotate(-math.degrees(angle), resample=Image.BICUBIC,
                                 expand=False, center=(glyph_size / 2.0, glyph_size / 2.0))
        layer.alpha_composite(glyph, (int(round(centre[0] - glyph_size / 2.0)),
                                      int(round(centre[1] - glyph_size / 2.0))))

        half_h = band_h / 2.0
        top_pts.append(place(-w / 2.0, -half_h, angle, centre))
        bottom_pts.append(place(-w / 2.0, half_h, angle, centre))
        if ch is text[-1]:
            top_pts.append(place(w / 2.0, -half_h, angle, centre))
            bottom_pts.append(place(w / 2.0, half_h, angle, centre))
        x_cursor += w

    poly = np.array(top_pts + bottom_pts[::-1], dtype=np.float32)
    return layer, poly


def _apply_homography(poly: np.ndarray, H: np.ndarray) -> np.ndarray:
    pts = np.concatenate([poly, np.ones((len(poly), 1), np.float32)], axis=1)
    out = (H @ pts.T).T
    return (out[:, :2] / np.maximum(out[:, 2:3], 1e-6)).astype(np.float32)


def _random_similarity_perspective(rng: random.Random, cfg: SynthConfig,
                                   size: Tuple[int, int]) -> np.ndarray:
    """Rotation (+ optional mild perspective) about the layer centre."""
    w, h = size
    angle = rng.uniform(-cfg.max_rotation, cfg.max_rotation) if rng.random() < cfg.rotate_prob else 0.0
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    H = np.vstack([M, [0, 0, 1]]).astype(np.float32)
    if rng.random() < cfg.perspective_prob:
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        jitter = min(w, h) * 0.12
        dst = src + np.float32([[rng.uniform(-jitter, jitter) for _ in range(2)]
                                for _ in range(4)])
        H = cv2.getPerspectiveTransform(src, dst).astype(np.float32) @ H
    return H


class SyntheticImageGenerator:
    """Compose one synthetic scene-text image with polygon + text annotations."""

    def __init__(self, cfg: Optional[SynthConfig] = None, fonts: Optional[Sequence[str]] = None,
                 background_dir: Optional[str] = None, seed: Optional[int] = None):
        self.cfg = cfg or SynthConfig()
        self.rng = random.Random(seed)
        self.fonts = list(fonts) if fonts else discover_fonts()
        if not self.fonts:
            raise RuntimeError(
                "no TrueType fonts found; install fonts (e.g. fonts-dejavu) or pass fonts=[...]")
        self.backgrounds = BackgroundSampler(background_dir, rng=self.rng)

    def generate(self) -> Tuple[np.ndarray, List[np.ndarray], List[str]]:
        """Returns ``(image_bgr, polygons, transcriptions)``."""
        cfg, rng = self.cfg, self.rng
        canvas = self.backgrounds.sample(cfg.width, cfg.height)
        occupied: List[np.ndarray] = []
        polys: List[np.ndarray] = []
        texts: List[str] = []

        n_words = rng.randint(cfg.min_words, cfg.max_words)
        for _ in range(n_words):
            placed = self._place_one(canvas, occupied)
            if placed is None:
                continue
            poly, text = placed
            polys.append(poly)
            texts.append(text)
            occupied.append(poly)

        canvas = self._degrade(canvas)
        return canvas, polys, texts

    # -- internals -------------------------------------------------------
    def _place_one(self, canvas: np.ndarray,
                   occupied: Sequence[np.ndarray]) -> Optional[Tuple[np.ndarray, str]]:
        cfg, rng = self.cfg, self.rng
        for _ in range(cfg.max_place_attempts):
            text = sample_text(rng)
            # Bias towards small sizes: video text is mostly small, and a model
            # trained on comfortable 40px text collapses on 12px text.
            size = int(cfg.min_font_size + (cfg.max_font_size - cfg.min_font_size)
                       * rng.random() ** 2.0)
            try:
                font = ImageFont.truetype(rng.choice(self.fonts), size)
            except Exception:
                continue
            curvature = (rng.uniform(-cfg.max_curvature, cfg.max_curvature)
                         if rng.random() < cfg.curve_prob else 0.0)

            try:
                probe, _ = render_word_layer(text, font, curvature, (255, 255, 255))
            except Exception:
                continue
            lw, lh = probe.size
            if lw >= cfg.width or lh >= cfg.height:
                continue
            x0 = rng.randint(0, cfg.width - lw)
            y0 = rng.randint(0, cfg.height - lh)

            patch = canvas[y0:y0 + lh, x0:x0 + lw]
            if patch.size == 0:
                continue
            color = pick_text_color(patch, cfg.min_contrast, rng)

            layer, poly = render_word_layer(text, font, curvature, color)
            H = _random_similarity_perspective(rng, cfg, layer.size)
            layer_np = np.array(layer)
            warped = cv2.warpPerspective(layer_np, H, layer.size,
                                         flags=cv2.INTER_LINEAR,
                                         borderMode=cv2.BORDER_CONSTANT,
                                         borderValue=(0, 0, 0, 0))
            poly = _apply_homography(poly, H) + np.array([x0, y0], np.float32)

            if poly[:, 0].min() < 0 or poly[:, 1].min() < 0:
                continue
            if poly[:, 0].max() >= cfg.width or poly[:, 1].max() >= cfg.height:
                continue
            if any(_poly_overlaps(poly, o) for o in occupied):
                continue

            _alpha_composite(canvas, warped, x0, y0)
            return poly, text
        return None

    def _degrade(self, img: np.ndarray) -> np.ndarray:
        """Blur / noise / JPEG artefacts -- the cheap half of the domain gap."""
        rng = self.rng
        out = img.astype(np.float32)
        if rng.random() < self.cfg.blur_prob:
            out = cv2.GaussianBlur(out, (0, 0), rng.uniform(0.4, 1.6))
        if rng.random() < self.cfg.noise_prob:
            out = out + np.random.randn(*out.shape).astype(np.float32) * rng.uniform(2, 12)
        out = np.clip(out, 0, 255).astype(np.uint8)
        if rng.random() < 0.3:
            ok, enc = cv2.imencode(".jpg", out,
                                   [int(cv2.IMWRITE_JPEG_QUALITY), rng.randint(35, 92)])
            if ok:
                out = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        return out


def _alpha_composite(canvas: np.ndarray, rgba: np.ndarray, x0: int, y0: int) -> None:
    h, w = rgba.shape[:2]
    H, W = canvas.shape[:2]
    x1, y1 = min(x0 + w, W), min(y0 + h, H)
    if x1 <= x0 or y1 <= y0:
        return
    src = rgba[: y1 - y0, : x1 - x0]
    alpha = (src[:, :, 3:4].astype(np.float32) / 255.0)
    region = canvas[y0:y1, x0:x1].astype(np.float32)
    canvas[y0:y1, x0:x1] = np.clip(
        src[:, :, :3].astype(np.float32) * alpha + region * (1 - alpha), 0, 255).astype(np.uint8)


def _poly_overlaps(a: np.ndarray, b: np.ndarray, thresh: float = 0.02) -> bool:
    """Reject placements that overlap: overlapping synthetic text is unreadable
    and its transcription label becomes wrong, which poisons the CTC head."""
    from ..utils.polygon import poly_iou
    return poly_iou(a, b) > thresh
