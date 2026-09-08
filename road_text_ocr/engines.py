"""Two-stage text engines: detect where the text is, then read what it says.

Every engine exposes the same two calls, deliberately kept separate:

    polys        = engine.detect(frame_bgr)          # stage 1 -- WHERE
    predictions  = engine.recognize(frame_bgr, polys) # stage 2 -- WHAT

Keeping them separate is the point.  A one-call ``readtext()`` hides which
stage failed: a missing word is a detection failure, a garbled word is a
recognition failure, and the fixes are unrelated.  Split like this you can
render stage-1 boxes alone and see immediately which one you are looking at.

Both engines ship pretrained weights that download on first use.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import cv2
import numpy as np


@dataclass
class Prediction:
    """One detected-and-read piece of text in one frame."""
    poly: np.ndarray      # (4, 2) float32, clockwise from top-left
    text: str
    confidence: float

    @property
    def box(self) -> tuple:
        """Axis-aligned bounds as (x1, y1, x2, y2)."""
        x, y = self.poly[:, 0], self.poly[:, 1]
        return float(x.min()), float(y.min()), float(x.max()), float(y.max())


# --------------------------------------------------------------------------
# cropping
# --------------------------------------------------------------------------

def crop_polygon(image: np.ndarray, poly: np.ndarray, target_height: int = 64,
                 pad_ratio: float = 0.08) -> Optional[np.ndarray]:
    """Warp a (possibly rotated) quad to an upright crop for the recogniser.

    A plain bounding-box crop of text at 30 degrees carries as much background
    as text, and recognisers trained on upright lines degrade badly on it.  A
    perspective warp straightens the quad instead, so the recogniser sees what
    it was trained on.

    Returns None when the quad is degenerate (zero area after rounding).
    """
    poly = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    if poly.shape[0] != 4:
        rect = cv2.minAreaRect(poly)
        poly = cv2.boxPoints(rect).astype(np.float32)

    poly = _order_quad(poly)

    # Pad outwards; recognisers want a little margin around the glyphs.
    centre = poly.mean(axis=0, keepdims=True)
    poly = centre + (poly - centre) * (1.0 + pad_ratio)

    width = max(np.linalg.norm(poly[1] - poly[0]), np.linalg.norm(poly[2] - poly[3]))
    height = max(np.linalg.norm(poly[3] - poly[0]), np.linalg.norm(poly[2] - poly[1]))
    if width < 2 or height < 2:
        return None

    out_h = int(target_height)
    out_w = max(int(round(width / height * out_h)), 8)
    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]],
                   dtype=np.float32)
    matrix = cv2.getPerspectiveTransform(poly, dst)
    return cv2.warpPerspective(image, matrix, (out_w, out_h),
                               flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def _order_quad(poly: np.ndarray) -> np.ndarray:
    """Order 4 points clockwise starting from the top-left corner."""
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
                 text_threshold: float = 0.7, low_text: float = 0.4,
                 link_threshold: float = 0.4):
        import easyocr  # imported lazily so --engine paddleocr needs no easyocr

        self.reader = easyocr.Reader(list(langs), gpu=gpu, verbose=False)
        self.text_threshold = text_threshold
        self.low_text = low_text
        self.link_threshold = link_threshold

    def detect(self, image: np.ndarray) -> List[np.ndarray]:
        horizontal, free = self.reader.detect(
            image,
            text_threshold=self.text_threshold,
            low_text=self.low_text,
            link_threshold=self.link_threshold,
        )
        # detect() returns one entry per input image; we pass one image.
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

    def recognize(self, image: np.ndarray, polys: Sequence[np.ndarray]) -> List[Prediction]:
        grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        out: List[Prediction] = []
        for poly in polys:
            crop = crop_polygon(grey, poly)
            if crop is None:
                continue
            # With no box lists, recognize() treats the whole crop as one line.
            try:
                result = self.reader.recognize(crop, detail=1)
            except Exception:
                continue
            if not result:
                continue
            _, text, conf = result[0]
            text = str(text).strip()
            if text:
                out.append(Prediction(np.asarray(poly, dtype=np.float32),
                                      text, float(conf)))
        return out


# --------------------------------------------------------------------------
# PaddleOCR
# --------------------------------------------------------------------------

class PaddleOCREngine:
    """DBNet detector + SVTR/CRNN recogniser.  Usually sharper on small text."""

    name = "paddleocr"

    def __init__(self, langs: Sequence[str] = ("en",), gpu: bool = True, **_):
        from paddleocr import PaddleOCR  # lazy, see EasyOCREngine

        lang = "en" if not langs else str(langs[0])
        self.ocr = self._build(PaddleOCR, lang, gpu)

    @staticmethod
    def _build(PaddleOCR, lang: str, gpu: bool):
        """PaddleOCR's constructor keywords changed across 2.x and 3.x.

        Rather than pinning one version, try the argument sets newest-first and
        keep the first that constructs.
        """
        for kwargs in ({"lang": lang, "use_textline_orientation": True},
                       {"lang": lang, "use_angle_cls": True, "use_gpu": gpu,
                        "show_log": False},
                       {"lang": lang}):
            try:
                return PaddleOCR(**kwargs)
            except TypeError:
                continue
        raise RuntimeError("could not construct PaddleOCR with any known signature")

    def _run(self, image: np.ndarray, **kwargs):
        try:
            return self.ocr.ocr(image, **kwargs)
        except TypeError:
            # 3.x dropped the det/rec switches from ocr().
            return self.ocr.ocr(image)

    def detect(self, image: np.ndarray) -> List[np.ndarray]:
        result = self._run(image, det=True, rec=False)
        if not result:
            return []
        page = result[0] if isinstance(result[0], list) else result
        polys: List[np.ndarray] = []
        for item in page or []:
            quad = item[0] if (isinstance(item, (list, tuple)) and item
                               and isinstance(item[0], (list, tuple, np.ndarray))
                               and np.asarray(item[0]).ndim == 2) else item
            arr = np.asarray(quad, dtype=np.float32).reshape(-1, 2)
            if arr.shape[0] >= 4:
                polys.append(arr)
        return polys

    def recognize(self, image: np.ndarray, polys: Sequence[np.ndarray]) -> List[Prediction]:
        out: List[Prediction] = []
        for poly in polys:
            crop = crop_polygon(image, poly, target_height=48)
            if crop is None:
                continue
            try:
                result = self._run(crop, det=False, rec=True)
            except Exception:
                continue
            parsed = self._parse_rec(result)
            if parsed is None:
                continue
            text, conf = parsed
            if text:
                out.append(Prediction(np.asarray(poly, dtype=np.float32), text, conf))
        return out

    @staticmethod
    def _parse_rec(result) -> Optional[tuple]:
        """Pull (text, confidence) out of whichever shape this version returned."""
        if not result:
            return None
        node = result
        for _ in range(4):
            if isinstance(node, dict):
                texts = node.get("rec_texts") or node.get("rec_text")
                scores = node.get("rec_scores") or node.get("rec_score")
                if texts:
                    text = texts[0] if isinstance(texts, (list, tuple)) else texts
                    score = 1.0
                    if scores:
                        score = scores[0] if isinstance(scores, (list, tuple)) else scores
                    return str(text).strip(), float(score)
                return None
            if isinstance(node, (list, tuple)) and node:
                if len(node) == 2 and isinstance(node[0], str):
                    return str(node[0]).strip(), float(node[1])
                node = node[0]
            else:
                return None
        return None


# --------------------------------------------------------------------------

ENGINES = {"easyocr": EasyOCREngine, "paddleocr": PaddleOCREngine}


def build_engine(name: str, langs: Sequence[str], gpu: bool):
    try:
        cls = ENGINES[name]
    except KeyError:
        raise SystemExit(f"unknown engine {name!r}; choose from {sorted(ENGINES)}")
    try:
        return cls(langs=langs, gpu=gpu)
    except ImportError as exc:
        raise SystemExit(
            f"engine {name!r} needs a package that is not installed ({exc}).\n"
            f"  easyocr    ->  pip install easyocr\n"
            f"  paddleocr  ->  pip install paddlepaddle paddleocr"
        ) from exc
