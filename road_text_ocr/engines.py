"""Detection and recognition engines.

Two modes, and the difference is measurable -- on a 1280x720 clip with four
small tilted signs:

    mode=pipeline    3/4 read exactly, confidences 0.95-1.00
    mode=two_stage   2/4 read exactly, confidences 0.52-1.00

``pipeline`` hands the frame to the engine's own end-to-end call.  ``two_stage``
splits detection from recognition explicitly -- crop each detected polygon, read
each crop alone -- which is easier to reason about and worse at reading, because
it throws away the merging and contrast-retry logic the engine does between its
own two stages.  Use ``two_stage`` to see *which* stage is failing, then switch
back to ``pipeline`` for results.

Both modes expose ``detect()`` on its own, so stage-1 boxes can be drawn without
running recognition at all.

Engines ship pretrained weights that download on first use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np


@dataclass
class Prediction:
    """One detected-and-read piece of text in one frame."""
    poly: np.ndarray      # (N, 2) float32
    text: str
    confidence: float

    @property
    def box(self) -> tuple:
        """Axis-aligned bounds as (x1, y1, x2, y2)."""
        x, y = self.poly[:, 0], self.poly[:, 1]
        return float(x.min()), float(y.min()), float(x.max()), float(y.max())


@dataclass
class DetectParams:
    """Knobs that decide what counts as text.  Defaults match the engine's own.

    ``mag`` is the one to reach for first on road video: the detector resizes
    the frame by this factor before looking, so 2.0 gives distant signage twice
    the pixels to be found in.  It costs roughly the square in time.

    ``link`` is the one behind "it only found half the words".  The detector
    finds character regions and links neighbours into a word; a high threshold
    links less, so "SPEED 40" comes back as "SPEED" and "40".  Lower it, and
    raise ``width_ths``, to merge more.
    """
    mag: float = 1.0              # pre-detection upscale
    low_text: float = 0.4         # lower  -> fainter strokes count as text
    text_threshold: float = 0.7   # lower  -> weaker regions count as text
    link: float = 0.4             # lower  -> neighbouring words merge
    width_ths: float = 0.5        # higher -> merge boxes further apart
    min_size: int = 10            # ignore boxes smaller than this (px)
    decoder: str = "greedy"       # or 'beamsearch' -- slower, a little better
    merge_words: bool = False     # group boxes into blocks; see below


# ``merge_words`` is what recovers a phrase the detector split.  On a frame
# whose signs read PHARMACY / EXIT 24 / MAIN STREET / SPEED 40, no combination
# of link and width_ths merged the last one -- it came back as "SPEED" and "40"
# every time.  Grouping does merge it, and reads all four exactly.
#
# It costs two things, so it is opt-in rather than the default:
#
# 1. Grouping discards per-box confidence, so every reading is scored 1.0.
#    Track agreement then counts frames that read a track the same way, rather
#    than weighting them by confidence, and --min-conf stops filtering.
# 2. It over-merges.  Grouping is geometric, so two *different* signs that drift
#    close together are joined as readily as two halves of one phrase -- on the
#    same clip, "EXIT 24" and "PHARMACY" became one track reading
#    "EXIT 24 PHARMACY" once they overlapped.
#
# So: turn it on when phrases arrive split, and check the tracks table for
# merges that should not be there.


# --------------------------------------------------------------------------
# cropping (two_stage mode only)
# --------------------------------------------------------------------------

def crop_polygon(image: np.ndarray, poly: np.ndarray, target_height: int = 64,
                 pad_y: float = 0.15, pad_x: float = 0.02) -> Optional[np.ndarray]:
    """Warp a (possibly rotated) quad to an upright crop for the recogniser.

    Padding is a fraction of the text *height* in both directions, and is much
    smaller horizontally than vertically.  Scaling the quad uniformly instead --
    the obvious implementation -- pads a long word by a lot horizontally and
    almost nothing vertically, which is backwards: recognisers want vertical
    margin, and the horizontal overshoot drags in whatever frames the text.  On
    a bordered sign that showed up as ``[PHARMACY]``: the panel edge, read as
    brackets.

    Returns None when the quad is degenerate.
    """
    poly = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    if poly.shape[0] != 4:
        poly = cv2.boxPoints(cv2.minAreaRect(poly)).astype(np.float32)
    poly = _order_quad(poly)

    width = max(np.linalg.norm(poly[1] - poly[0]), np.linalg.norm(poly[2] - poly[3]))
    height = max(np.linalg.norm(poly[3] - poly[0]), np.linalg.norm(poly[2] - poly[1]))
    if width < 2 or height < 2:
        return None

    # Expand along the quad's own axes, so a tilted box stays tilted.
    x_axis = (poly[1] - poly[0]) / max(np.linalg.norm(poly[1] - poly[0]), 1e-6)
    y_axis = (poly[3] - poly[0]) / max(np.linalg.norm(poly[3] - poly[0]), 1e-6)
    dx, dy = x_axis * height * pad_x, y_axis * height * pad_y
    poly = np.array([poly[0] - dx - dy, poly[1] + dx - dy,
                     poly[2] + dx + dy, poly[3] - dx + dy], dtype=np.float32)

    out_h = int(target_height)
    out_w = max(int(round((width + 2 * height * pad_x) / (height + 2 * height * pad_y)
                          * out_h)), 8)
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
                   dtype=np.float32)
    return cv2.warpPerspective(image, cv2.getPerspectiveTransform(poly, dst),
                               (out_w, out_h), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)


def _order_quad(poly: np.ndarray) -> np.ndarray:
    """Order 4 points clockwise from the top-left corner."""
    s = poly.sum(axis=1)
    d = np.diff(poly, axis=1).ravel()
    return np.array([poly[np.argmin(s)], poly[np.argmin(d)],
                     poly[np.argmax(s)], poly[np.argmax(d)]], dtype=np.float32)


# --------------------------------------------------------------------------
# EasyOCR
# --------------------------------------------------------------------------

class EasyOCREngine:
    """CRAFT detector + CRNN recogniser (pure PyTorch, easiest to install)."""

    name = "easyocr"

    def __init__(self, langs: Sequence[str] = ("en",), gpu: bool = True,
                 params: Optional[DetectParams] = None):
        import easyocr  # lazy: --engine paddleocr should not need easyocr

        self.reader = easyocr.Reader(list(langs), gpu=gpu, verbose=False)
        self.p = params or DetectParams()

    @staticmethod
    def _rgb(image: np.ndarray) -> np.ndarray:
        """EasyOCR reads files as RGB but takes arrays as BGR, and the two give
        different results.  Feed it RGB so a frame matches the same frame saved
        to disk and passed by path."""
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def _detect_kwargs(self) -> dict:
        p = self.p
        return dict(text_threshold=p.text_threshold, low_text=p.low_text,
                    link_threshold=p.link, mag_ratio=p.mag, min_size=p.min_size,
                    width_ths=p.width_ths)

    def detect(self, image: np.ndarray) -> List[np.ndarray]:
        horizontal, free = self.reader.detect(self._rgb(image), **self._detect_kwargs())
        horizontal = horizontal[0] if horizontal else []
        free = free[0] if free else []

        polys: List[np.ndarray] = []
        for box in horizontal:
            # EasyOCR's horizontal boxes are (x_min, x_max, y_min, y_max) --
            # note the ordering, it is not the usual (x1, y1, x2, y2).
            x1, x2, y1, y2 = [float(v) for v in box[:4]]
            polys.append(np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                                  dtype=np.float32))
        for quad in free:
            polys.append(np.asarray(quad, dtype=np.float32).reshape(-1, 2))
        return polys

    def read(self, image: np.ndarray) -> List[Prediction]:
        """End-to-end: detection and recognition in the engine's own pipeline."""
        results = self.reader.readtext(
            self._rgb(image), decoder=self.p.decoder,
            paragraph=self.p.merge_words, **self._detect_kwargs())
        out = []
        for item in results:
            # Grouped results are (quad, text); ungrouped are (quad, text, conf).
            quad, text = item[0], item[1]
            conf = float(item[2]) if len(item) > 2 else 1.0
            text = str(text).strip()
            if text:
                out.append(Prediction(np.asarray(quad, dtype=np.float32).reshape(-1, 2),
                                      text, conf))
        return out

    def recognize(self, image: np.ndarray, polys: Sequence[np.ndarray]) -> List[Prediction]:
        """Read each detected polygon in isolation (two_stage mode)."""
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        out: List[Prediction] = []
        for poly in polys:
            crop = crop_polygon(grey, poly)
            if crop is None:
                continue
            try:
                # With no box lists, recognize() treats the whole crop as one line.
                result = self.reader.recognize(crop, detail=1, decoder=self.p.decoder)
            except Exception:
                continue
            if not result:
                continue
            _, text, conf = result[0]
            text = str(text).strip()
            if text:
                out.append(Prediction(np.asarray(poly, dtype=np.float32), text,
                                      float(conf)))
        return out


# --------------------------------------------------------------------------
# PaddleOCR
# --------------------------------------------------------------------------

class PaddleOCREngine:
    """DBNet detector + SVTR recogniser.  Often sharper on small scene text."""

    name = "paddleocr"

    def __init__(self, langs: Sequence[str] = ("en",), gpu: bool = True,
                 params: Optional[DetectParams] = None):
        from paddleocr import PaddleOCR  # lazy, see EasyOCREngine

        self.p = params or DetectParams()
        self.ocr = self._build(PaddleOCR, (langs or ["en"])[0], gpu)

    @staticmethod
    def _build(PaddleOCR, lang: str, gpu: bool):
        """PaddleOCR's constructor keywords changed across 2.x and 3.x, so try
        the argument sets newest-first and keep the first that constructs."""
        for kwargs in ({"lang": lang, "use_textline_orientation": True},
                       {"lang": lang, "use_angle_cls": True, "use_gpu": gpu,
                        "show_log": False},
                       {"lang": lang}):
            try:
                return PaddleOCR(**kwargs)
            except TypeError:
                continue
        raise RuntimeError("could not construct PaddleOCR with any known signature")

    def _predict(self, image: np.ndarray) -> List[dict]:
        """Return each page's result dict, across 2.x/3.x return shapes."""
        pages: List[dict] = []
        if hasattr(self.ocr, "predict"):
            for page in self.ocr.predict(image) or []:
                if isinstance(page, dict):
                    pages.append(page)
                else:
                    pages.append(getattr(page, "json", {}).get("res", {}))
            if pages:
                return pages
        result = self.ocr.ocr(image)
        for page in result or []:
            if isinstance(page, dict):
                pages.append(page)
            elif isinstance(page, list):
                polys, texts, scores = [], [], []
                for item in page:
                    try:
                        polys.append(item[0])
                        texts.append(item[1][0])
                        scores.append(float(item[1][1]))
                    except (IndexError, TypeError):
                        continue
                pages.append({"dt_polys": polys, "rec_texts": texts,
                              "rec_scores": scores})
        return pages

    def detect(self, image: np.ndarray) -> List[np.ndarray]:
        polys: List[np.ndarray] = []
        for page in self._predict(image):
            for quad in page.get("dt_polys", []) or []:
                arr = np.asarray(quad, dtype=np.float32).reshape(-1, 2)
                if arr.shape[0] >= 4:
                    polys.append(arr)
        return polys

    def read(self, image: np.ndarray) -> List[Prediction]:
        out: List[Prediction] = []
        for page in self._predict(image):
            quads = page.get("dt_polys", []) or []
            texts = page.get("rec_texts", []) or []
            scores = page.get("rec_scores", []) or []
            for quad, text, score in zip(quads, texts, scores):
                text = str(text).strip()
                if text:
                    out.append(Prediction(
                        np.asarray(quad, dtype=np.float32).reshape(-1, 2),
                        text, float(score)))
        return out

    def recognize(self, image: np.ndarray, polys: Sequence[np.ndarray]) -> List[Prediction]:
        out: List[Prediction] = []
        for poly in polys:
            crop = crop_polygon(image, poly, target_height=48)
            if crop is None:
                continue
            try:
                pages = self._predict(crop)
            except Exception:
                continue
            for page in pages:
                texts = page.get("rec_texts", []) or []
                scores = page.get("rec_scores", []) or []
                if texts:
                    text = str(texts[0]).strip()
                    if text:
                        out.append(Prediction(np.asarray(poly, dtype=np.float32),
                                              text, float(scores[0]) if scores else 1.0))
                    break
        return out


# --------------------------------------------------------------------------
# presets
# --------------------------------------------------------------------------

PRESETS: Dict[str, dict] = {
    "default": {},
    # Distant road signage: look harder, and merge words the detector split.
    "small-text": dict(mag=2.0, low_text=0.3, text_threshold=0.6, link=0.2,
                       width_ths=1.0, min_size=6),
    # Same, but glue split phrases back together ("SPEED" + "40" -> "SPEED 40").
    "small-text-merged": dict(mag=2.0, low_text=0.3, text_threshold=0.6, link=0.2,
                              width_ths=1.0, min_size=6, merge_words=True),
    # Same, pushed further -- slow, for when small-text still misses things.
    "tiny-text": dict(mag=3.0, low_text=0.25, text_threshold=0.5, link=0.1,
                      width_ths=1.5, min_size=4, decoder="beamsearch"),
    # Long clips: fewer, bigger regions only.
    "fast": dict(mag=1.0, low_text=0.5, text_threshold=0.8, min_size=20),
}

ENGINES = {"easyocr": EasyOCREngine, "paddleocr": PaddleOCREngine}


def build_engine(name: str, langs: Sequence[str], gpu: bool,
                 params: Optional[DetectParams] = None):
    try:
        cls = ENGINES[name]
    except KeyError:
        raise SystemExit(f"unknown engine {name!r}; choose from {sorted(ENGINES)}")
    try:
        return cls(langs=langs, gpu=gpu, params=params)
    except ImportError as exc:
        raise SystemExit(
            f"engine {name!r} needs a package that is not installed ({exc}).\n"
            f"  easyocr    ->  pip install easyocr\n"
            f"  paddleocr  ->  pip install paddlepaddle paddleocr"
        ) from exc
