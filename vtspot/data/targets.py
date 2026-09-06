"""Build the dense DB supervision maps from polygon annotations.

Produces four full-resolution maps per image:

``shrink_map``   1 inside the shrunk polygon of every trainable instance.
``shrink_mask``  0 on ignore regions -- pixels excluded from the loss entirely.
``thresh_map``   a normalised border band, ``[thresh_min, thresh_max]``, peaking
                 on the polygon boundary itself.
``thresh_mask``  1 only inside that band, since the threshold branch is
                 supervised nowhere else.

Shrinking is what separates adjacent instances: two words 3 px apart merge into
one blob if you supervise the raw polygons, and no amount of post-processing
recovers them.  DSText, where the average frame holds ~24 instances, is entirely
about this.
"""

from __future__ import annotations

from typing import List, Sequence

import cv2
import numpy as np

from ..utils.polygon import (adaptive_unclip_distance, expand_polygon,
                             is_valid_poly, offset_polygon, poly_area,
                             shrink_polygon)


def _distance_to_segment(xs: np.ndarray, ys: np.ndarray,
                         p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Point-to-segment distance over a coordinate grid."""
    seg = p2 - p1
    seg_len_sq = float(seg[0] ** 2 + seg[1] ** 2)
    dx, dy = xs - p1[0], ys - p1[1]
    if seg_len_sq < 1e-9:
        return np.sqrt(dx * dx + dy * dy)
    t = np.clip((dx * seg[0] + dy * seg[1]) / seg_len_sq, 0.0, 1.0)
    px, py = p1[0] + t * seg[0], p1[1] + t * seg[1]
    return np.sqrt((xs - px) ** 2 + (ys - py) ** 2)


def _draw_threshold_band(thresh_map: np.ndarray, thresh_mask: np.ndarray,
                         poly: np.ndarray, shrink_ratio: float) -> None:
    h, w = thresh_map.shape
    area, perim = poly_area(poly), max(np.linalg.norm(
        poly - np.roll(poly, 1, axis=0), axis=1).sum(), 1e-6)
    distance = area * (1 - shrink_ratio ** 2) / perim
    dilated = expand_polygon(poly, ratio=(1 - shrink_ratio ** 2))
    if dilated is None:
        return

    x1 = int(np.clip(np.floor(dilated[:, 0].min()), 0, w - 1))
    x2 = int(np.clip(np.ceil(dilated[:, 0].max()), 0, w - 1))
    y1 = int(np.clip(np.floor(dilated[:, 1].min()), 0, h - 1))
    y2 = int(np.clip(np.ceil(dilated[:, 1].max()), 0, h - 1))
    if x2 <= x1 or y2 <= y1:
        return

    xs = np.arange(x1, x2 + 1, dtype=np.float32)[None, :].repeat(y2 - y1 + 1, axis=0)
    ys = np.arange(y1, y2 + 1, dtype=np.float32)[:, None].repeat(x2 - x1 + 1, axis=1)

    dist = np.full(xs.shape, np.inf, dtype=np.float32)
    for i in range(len(poly)):
        d = _distance_to_segment(xs, ys, poly[i], poly[(i + 1) % len(poly)])
        dist = np.minimum(dist, d)
    # 1 on the boundary, falling to 0 at the band edge.
    band = np.clip(1.0 - dist / max(distance, 1e-6), 0.0, 1.0)

    region = np.zeros((y2 - y1 + 1, x2 - x1 + 1), dtype=np.uint8)
    cv2.fillPoly(region, [(dilated - np.array([x1, y1])).astype(np.int32)], 1)

    sub_map = thresh_map[y1:y2 + 1, x1:x2 + 1]
    sub_mask = thresh_mask[y1:y2 + 1, x1:x2 + 1]
    np.maximum(sub_map, band * region, out=sub_map)
    np.maximum(sub_mask, region.astype(np.float32), out=sub_mask)


def build_db_targets(polygons: Sequence[np.ndarray], ignore_flags: Sequence[bool],
                     height: int, width: int, shrink_ratio: float = 0.4,
                     thresh_min: float = 0.3, thresh_max: float = 0.7,
                     min_text_size: int = 4) -> dict:
    """Rasterise polygons into the four DB supervision maps.

    Instances too small to survive shrinking are marked *ignore* rather than
    dropped: they are real text, so scoring them as background would teach the
    model to suppress exactly the small text DSText cares about.
    """
    shrink_map = np.zeros((height, width), dtype=np.float32)
    shrink_mask = np.ones((height, width), dtype=np.float32)
    thresh_map = np.zeros((height, width), dtype=np.float32)
    thresh_mask = np.zeros((height, width), dtype=np.float32)

    for poly, ignore in zip(polygons, ignore_flags):
        poly = np.asarray(poly, dtype=np.float32)
        if not is_valid_poly(poly):
            continue
        pts = poly.astype(np.int32)
        w_box = poly[:, 0].max() - poly[:, 0].min()
        h_box = poly[:, 1].max() - poly[:, 1].min()
        too_small = min(w_box, h_box) < min_text_size

        if ignore or too_small:
            cv2.fillPoly(shrink_mask, [pts], 0.0)
            continue

        shrunk = shrink_polygon(poly, shrink_ratio)
        if shrunk is None or len(shrunk) < 3:
            cv2.fillPoly(shrink_mask, [pts], 0.0)
            continue

        cv2.fillPoly(shrink_map, [shrunk.astype(np.int32)], 1.0)
        _draw_threshold_band(thresh_map, thresh_mask, poly, shrink_ratio)

    thresh_map = thresh_map * (thresh_max - thresh_min) + thresh_min
    return {"shrink_map": shrink_map[None], "shrink_mask": shrink_mask[None],
            "thresh_map": thresh_map[None], "thresh_mask": thresh_mask[None]}


def decode_db_polygons(prob_map: np.ndarray, box_thresh: float = 0.5,
                       bin_thresh: float = 0.3, unclip_ratio: float = 1.5,
                       max_candidates: int = 1000, min_size: int = 3,
                       adaptive_unclip: bool = True, shrink_ratio: float = 0.4
                       ) -> tuple[List[np.ndarray], List[float]]:
    """Inverse of :func:`build_db_targets`: probability map -> polygons + scores.

    The score is the *mean probability inside the region*, not the peak, which is
    a far better-calibrated confidence and is what the tracker's rescoring
    depends on.

    ``adaptive_unclip`` (default on) picks the expansion distance per instance
    by inverting the shrink analytically, instead of applying one global
    ``unclip_ratio``.  Measured on rectangles spanning realistic text aspect
    ratios, recovering the source polygon from its shrunk form gives a mean IoU
    of 0.89 adaptively versus 0.62 with a fixed 1.5 -- and for elongated
    instances (200x15, 300x12, 80x10) the fixed ratio lands at IoU 0.32-0.45,
    i.e. *below* the 0.5 matching threshold, so those detections are scored as a
    false positive and a false negative rather than a hit.  Long thin boxes are
    the common case for text, so this is not an edge case.

    Set ``adaptive_unclip=False`` to reproduce standard DBNet post-processing.
    """
    if prob_map.ndim == 3:
        prob_map = prob_map[0]
    binary = (prob_map > bin_thresh).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    polys: List[np.ndarray] = []
    scores: List[float] = []
    for contour in contours[:max_candidates]:
        if len(contour) < 4:
            continue
        eps = 0.002 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, eps, True).reshape(-1, 2).astype(np.float32)
        if len(approx) < 4:
            approx = cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float32)

        score = _region_score(prob_map, approx)
        if score < box_thresh:
            continue

        if adaptive_unclip:
            distance = adaptive_unclip_distance(approx, shrink_ratio)
            expanded = offset_polygon(approx, distance)
        else:
            expanded = expand_polygon(approx, unclip_ratio)
        if expanded is None or len(expanded) < 4:
            continue
        w_box = expanded[:, 0].max() - expanded[:, 0].min()
        h_box = expanded[:, 1].max() - expanded[:, 1].min()
        if min(w_box, h_box) < min_size:
            continue
        polys.append(expanded)
        scores.append(float(score))
    return polys, scores


def _region_score(prob_map: np.ndarray, poly: np.ndarray) -> float:
    h, w = prob_map.shape
    x1 = int(np.clip(np.floor(poly[:, 0].min()), 0, w - 1))
    x2 = int(np.clip(np.ceil(poly[:, 0].max()), 0, w - 1))
    y1 = int(np.clip(np.floor(poly[:, 1].min()), 0, h - 1))
    y2 = int(np.clip(np.ceil(poly[:, 1].max()), 0, h - 1))
    if x2 < x1 or y2 < y1:
        return 0.0
    mask = np.zeros((y2 - y1 + 1, x2 - x1 + 1), dtype=np.uint8)
    cv2.fillPoly(mask, [(poly - np.array([x1, y1])).astype(np.int32)], 1)
    if mask.sum() == 0:
        return 0.0
    return float(cv2.mean(prob_map[y1:y2 + 1, x1:x2 + 1], mask)[0])
