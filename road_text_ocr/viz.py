"""Drawing helpers for the annotated output video."""

from __future__ import annotations

from typing import Optional, Sequence

import cv2
import numpy as np

# Distinguishable on road scenes (BGR).  Track id picks one by modulo, so the
# same sign keeps the same colour for as long as it holds its id -- which makes
# an id switch visible at a glance instead of buried in the JSON.
PALETTE = [
    (66, 133, 244), (52, 168, 83), (251, 188, 5), (234, 67, 53),
    (171, 71, 188), (0, 172, 193), (255, 112, 67), (158, 157, 36),
]


def colour_for(track_id: Optional[int]) -> tuple:
    if track_id is None:
        return (0, 215, 255)          # amber: detected but not yet tracked
    return PALETTE[track_id % len(PALETTE)]


def draw_polygon(frame: np.ndarray, poly: np.ndarray, colour: tuple,
                 thickness: int = 2) -> None:
    pts = np.asarray(poly, dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(frame, [pts], isClosed=True, color=colour, thickness=thickness,
                  lineType=cv2.LINE_AA)


def draw_label(frame: np.ndarray, poly: np.ndarray, label: str, colour: tuple,
               scale: float = 0.55) -> None:
    """Filled caption above the polygon, nudged inside the frame if it overflows."""
    if not label:
        return
    pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
    x, y = int(pts[:, 0].min()), int(pts[:, 1].min())

    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(label, font, scale, 1)
    pad = 4
    box_h = th + baseline + 2 * pad

    top = y - box_h
    if top < 0:                        # no room above -- put it below instead
        top = min(int(pts[:, 1].max()), frame.shape[0] - box_h - 1)
    top = max(top, 0)
    left = max(min(x, frame.shape[1] - tw - 2 * pad - 1), 0)

    cv2.rectangle(frame, (left, top), (left + tw + 2 * pad, top + box_h),
                  colour, thickness=-1)
    cv2.putText(frame, label, (left + pad, top + th + pad), font, scale,
                (255, 255, 255), 1, cv2.LINE_AA)


def annotate(frame: np.ndarray, items: Sequence[dict], show_conf: bool = True,
             stage: str = "both") -> np.ndarray:
    """Draw one frame's results.

    ``items`` are dicts with ``poly`` and optionally ``text``, ``confidence``
    and ``track_id``.  ``stage='detect'`` draws boxes only -- useful for seeing
    what detection did before recognition had a chance to spoil it.
    """
    out = frame.copy()
    for item in items:
        colour = colour_for(item.get("track_id"))
        draw_polygon(out, item["poly"], colour)
        if stage == "detect":
            continue
        text = item.get("text", "")
        if not text:
            continue
        parts = []
        if item.get("track_id") is not None:
            parts.append(f"#{item['track_id']}")
        parts.append(text)
        if show_conf and item.get("confidence") is not None:
            parts.append(f"{item['confidence']:.2f}")
        draw_label(out, item["poly"], " ".join(parts), colour)
    return out


def draw_hud(frame: np.ndarray, lines: Sequence[str]) -> None:
    """Small translucent status panel in the top-left corner."""
    if not lines:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, pad, gap = 0.5, 8, 20
    width = max(cv2.getTextSize(t, font, scale, 1)[0][0] for t in lines) + 2 * pad
    height = gap * len(lines) + pad

    panel = frame[0:height, 0:width]
    if panel.size:
        frame[0:height, 0:width] = cv2.addWeighted(
            panel, 0.35, np.zeros_like(panel), 0.0, 0)
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (pad, pad + gap * i + 6), font, scale,
                    (255, 255, 255), 1, cv2.LINE_AA)
