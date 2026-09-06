"""Polygon geometry helpers shared by the DB detection head and PolyAlign.

Everything here operates on ``(N, 2)`` float arrays in *image pixel* coordinates
and is deliberately numpy-only so it can run inside dataloader worker processes.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import cv2
import numpy as np
import pyclipper
from shapely.geometry import Polygon as ShapelyPolygon


def poly_area(poly: np.ndarray) -> float:
    """Shoelace area (always positive)."""
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def poly_perimeter(poly: np.ndarray) -> float:
    d = poly - np.roll(poly, 1, axis=0)
    return float(np.linalg.norm(d, axis=1).sum())


def is_valid_poly(poly: np.ndarray, min_area: float = 1.0) -> bool:
    if poly is None or len(poly) < 3:
        return False
    if not np.isfinite(poly).all():
        return False
    return poly_area(poly) >= min_area


def order_quad_clockwise(quad: np.ndarray) -> np.ndarray:
    """Order a 4-point quad as TL, TR, BR, BL.

    ICDAR annotations are *mostly* in this order but not reliably; a wrong order
    silently mirrors the RoI crop and poisons the recognition head, which is one
    of the more painful bugs to find after the fact.
    """
    quad = np.asarray(quad, dtype=np.float32).reshape(4, 2)
    centre = quad.mean(axis=0)
    angles = np.arctan2(quad[:, 1] - centre[1], quad[:, 0] - centre[0])
    quad = quad[np.argsort(angles)]           # counter-clockwise from -pi
    # rotate so the point closest to the top-left of the bbox comes first
    start = int(np.argmin(quad.sum(axis=1)))
    quad = np.roll(quad, -start, axis=0)
    # Image coordinates have y pointing down, so the visually-clockwise order
    # TL,TR,BR,BL has a *positive* shoelace area under this convention.
    if poly_signed_area(quad) < 0:
        quad = quad[[0, 3, 2, 1]]
    return quad


def poly_signed_area(poly: np.ndarray) -> float:
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def shrink_polygon(poly: np.ndarray, ratio: float) -> np.ndarray | None:
    """Vatti-clip a polygon inwards -- the DB shrink used to build the target map.

    Offset follows DBNet: ``d = A * (1 - r^2) / L``.
    """
    if not is_valid_poly(poly):
        return None
    area, perim = poly_area(poly), poly_perimeter(poly)
    if perim < 1e-6:
        return None
    distance = area * (1.0 - ratio ** 2) / perim
    pco = pyclipper.PyclipperOffset()
    pco.AddPath(poly.astype(np.int64).tolist(), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    shrunk = pco.Execute(-distance)
    if not shrunk:
        return None
    best = max(shrunk, key=lambda p: poly_area(np.asarray(p, dtype=np.float32)))
    out = np.asarray(best, dtype=np.float32)
    return out if is_valid_poly(out) else None


def expand_polygon(poly: np.ndarray, ratio: float) -> np.ndarray | None:
    """Inverse of :func:`shrink_polygon`; used to unclip predicted regions."""
    if not is_valid_poly(poly):
        return None
    area, perim = poly_area(poly), poly_perimeter(poly)
    if perim < 1e-6:
        return None
    distance = area * ratio / perim
    pco = pyclipper.PyclipperOffset()
    pco.AddPath(poly.astype(np.int64).tolist(), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    expanded = pco.Execute(distance)
    if not expanded:
        return None
    best = max(expanded, key=lambda p: poly_area(np.asarray(p, dtype=np.float32)))
    out = np.asarray(best, dtype=np.float32)
    return out if is_valid_poly(out) else None


def unclip_ratio_for_shrink(poly: np.ndarray, shrink_ratio: float) -> float:
    """The unclip ratio that exactly undoes ``shrink_polygon(poly, shrink_ratio)``.

    Shrinking offsets inwards by ``d = A * (1 - r^2) / L`` using the *original*
    polygon's area and perimeter; expanding offsets outwards by
    ``d' = A' * ratio / L'`` using the *shrunk* polygon's.  Since shrinking
    changes both A and L non-linearly, ``ratio`` is not ``1 - r^2`` or any other
    closed form in ``r`` alone -- it depends on the instance's aspect ratio.

    This is why DBNet exposes ``unclip_ratio`` as a tuned constant (1.5 by
    default) rather than deriving it.  For typical text aspect ratios the exact
    value lands around 1.5-1.7; use this function to check the setting against
    your own data rather than assuming.
    """
    poly = np.asarray(poly, dtype=np.float32)
    area, perim = poly_area(poly), poly_perimeter(poly)
    if perim < 1e-6:
        return 1.5
    distance = area * (1.0 - shrink_ratio ** 2) / perim
    shrunk = shrink_polygon(poly, shrink_ratio)
    if shrunk is None:
        return 1.5
    a2, l2 = poly_area(shrunk), poly_perimeter(shrunk)
    if a2 < 1e-6:
        return 1.5
    return float(distance * l2 / a2)


def adaptive_unclip_distance(shrunk: np.ndarray, shrink_ratio: float = 0.4) -> float:
    """Offset distance that inverts a DB shrink, from the *shrunk* polygon alone.

    Motivation: the exact inverse ratio is strongly aspect-ratio dependent.  For
    a shrink of 0.4 it is 1.45 for a near-square instance but 4.88 for a 300x12
    one -- and long thin instances are what text mostly *is*.  A single global
    ``unclip_ratio=1.5`` therefore under-expands exactly the common case, which
    shows up as detections that are systematically too tight on long words and
    costs IoU right at the 0.5 matching threshold.

    Offsetting a polygon outwards by ``d`` with round joins gives
    ``A(d) = A' + L'*d + pi*d^2`` and ``L(d) = L' + 2*pi*d``.  The shrink that
    produced this polygon used ``d = A(d) * (1 - r^2) / L(d)``, so ``d`` is the
    root of

        f(d) = d * (L' + 2*pi*d) - (1 - r^2) * (A' + L'*d + pi*d^2)

    which is a quadratic with one positive root.  Solved in closed form below.
    """
    shrunk = np.asarray(shrunk, dtype=np.float32)
    a1, l1 = poly_area(shrunk), poly_perimeter(shrunk)
    if a1 < 1e-6 or l1 < 1e-6:
        return 0.0
    k = 1.0 - float(shrink_ratio) ** 2
    # (2pi - k*pi) d^2 + (L' - k*L') d - k*A' = 0
    qa = math.pi * (2.0 - k)
    qb = l1 * (1.0 - k)
    qc = -k * a1
    if abs(qa) < 1e-9:
        return float(-qc / qb) if abs(qb) > 1e-9 else 0.0
    disc = qb * qb - 4 * qa * qc
    if disc < 0:
        return 0.0
    return float((-qb + math.sqrt(disc)) / (2 * qa))


def offset_polygon(poly: np.ndarray, distance: float) -> np.ndarray | None:
    """Offset a polygon outwards (positive) or inwards (negative) by ``distance``."""
    if not is_valid_poly(poly) or abs(distance) < 1e-6:
        return poly if is_valid_poly(poly) else None
    pco = pyclipper.PyclipperOffset()
    pco.AddPath(poly.astype(np.int64).tolist(), pyclipper.JT_ROUND,
                pyclipper.ET_CLOSEDPOLYGON)
    result = pco.Execute(distance)
    if not result:
        return None
    best = max(result, key=lambda p: poly_area(np.asarray(p, dtype=np.float32)))
    out = np.asarray(best, dtype=np.float32)
    return out if is_valid_poly(out) else None


def poly_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Polygon IoU via shapely, robust to self-intersecting annotations."""
    try:
        pa, pb = ShapelyPolygon(a), ShapelyPolygon(b)
        if not pa.is_valid:
            pa = pa.buffer(0)
        if not pb.is_valid:
            pb = pb.buffer(0)
        if pa.is_empty or pb.is_empty:
            return 0.0
        inter = pa.intersection(pb).area
        union = pa.union(pb).area
        return float(inter / union) if union > 1e-9 else 0.0
    except Exception:
        return 0.0


def poly_to_bbox(poly: np.ndarray) -> np.ndarray:
    """Axis-aligned ``[x1, y1, x2, y2]``."""
    return np.array([poly[:, 0].min(), poly[:, 1].min(),
                     poly[:, 0].max(), poly[:, 1].max()], dtype=np.float32)


def bbox_iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Vectorised IoU between ``(N,4)`` and ``(M,4)`` xyxy boxes."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    inter = wh[..., 0] * wh[..., 1]
    area_a = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    area_b = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter
    return (inter / np.maximum(union, 1e-9)).astype(np.float32)


# --------------------------------------------------------------------------
# Polygon <-> control-point conversion.
#
# The recognition head needs a *rectified* crop of the text region.  Rather than
# assuming a rotated rectangle (which fails on the curved text that makes up
# >30% of ArTVideo), every instance is represented by ``num_points`` control
# points along the top edge and the same number along the bottom edge.  A
# straight quad is just the degenerate case, so one code path serves both.
# --------------------------------------------------------------------------

def resample_polyline(points: np.ndarray, n: int) -> np.ndarray:
    """Resample an open polyline to ``n`` points at uniform arc length."""
    points = np.asarray(points, dtype=np.float32)
    if len(points) == 1:
        return np.repeat(points, n, axis=0)
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    total = cum[-1]
    if total < 1e-6:
        return np.repeat(points[:1], n, axis=0)
    targets = np.linspace(0.0, total, n)
    out = np.stack([np.interp(targets, cum, points[:, 0]),
                    np.interp(targets, cum, points[:, 1])], axis=1)
    return out.astype(np.float32)


def split_polygon_edges(poly: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Split a text polygon into its top and bottom boundaries.

    Convention (shared by CTW1500, Total-Text and ArTVideo): a ``2k``-point
    polygon lists ``k`` points along the top edge left-to-right followed by
    ``k`` points along the bottom edge right-to-left.  For a 4-point quad
    ordered TL,TR,BR,BL this reduces to top=[TL,TR], bottom=[BL,BR].
    """
    poly = np.asarray(poly, dtype=np.float32)
    n = len(poly)
    if n < 4:
        raise ValueError(f"need >= 4 points, got {n}")
    if n % 2 != 0:
        poly = poly[:-1]
        n -= 1
    k = n // 2
    top = poly[:k]
    bottom = poly[k:][::-1]          # reverse -> left-to-right
    return top, bottom


def polygon_to_control_points(poly: np.ndarray, num_points: int = 8) -> np.ndarray:
    """``(2 * num_points, 2)`` control points: top edge then bottom edge, both L->R."""
    top, bottom = split_polygon_edges(poly)
    top = resample_polyline(top, num_points)
    bottom = resample_polyline(bottom, num_points)
    return np.concatenate([top, bottom], axis=0).astype(np.float32)


def control_points_to_polygon(ctrl: np.ndarray) -> np.ndarray:
    """Inverse of :func:`polygon_to_control_points` (top L->R, bottom R->L)."""
    k = len(ctrl) // 2
    top, bottom = ctrl[:k], ctrl[k:]
    return np.concatenate([top, bottom[::-1]], axis=0).astype(np.float32)


def mask_to_polygons(mask: np.ndarray, min_area: float = 4.0,
                     max_candidates: int = 1000) -> List[np.ndarray]:
    """Contour extraction from a binary mask, largest-first."""
    mask_u8 = (mask.astype(np.uint8) > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    polys: List[np.ndarray] = []
    for cnt in contours[:max_candidates]:
        if len(cnt) < 3:
            continue
        eps = 0.002 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, eps, True).reshape(-1, 2).astype(np.float32)
        if len(approx) < 4:
            approx = cv2.boxPoints(cv2.minAreaRect(cnt)).astype(np.float32)
        if poly_area(approx) < min_area:
            continue
        polys.append(approx)
    polys.sort(key=poly_area, reverse=True)
    return polys


def order_polygon_for_text(poly: np.ndarray) -> np.ndarray:
    """Re-order an arbitrary contour into the top-edge/bottom-edge convention.

    Uses the minimum-area rectangle to establish the dominant reading direction,
    then splits the contour points by which long edge they sit closer to.  This
    is what turns a raw ``findContours`` output into something PolyAlign can
    sample in reading order.
    """
    poly = np.asarray(poly, dtype=np.float32)
    if len(poly) == 4:
        return order_quad_clockwise(poly)

    rect = cv2.minAreaRect(poly.astype(np.float32))
    box = cv2.boxPoints(rect).astype(np.float32)
    box = order_quad_clockwise(box)
    tl, tr, br, bl = box
    # long axis = reading direction
    if np.linalg.norm(tr - tl) < np.linalg.norm(bl - tl):
        tl, tr, br, bl = tr, br, bl, tl

    axis = tr - tl
    axis = axis / (np.linalg.norm(axis) + 1e-6)
    normal = np.array([-axis[1], axis[0]], dtype=np.float32)

    rel = poly - tl
    t = rel @ axis                      # position along reading direction
    s = rel @ normal                    # signed distance to the top edge
    mid = (s.max() + s.min()) / 2.0

    top = poly[s <= mid]
    bottom = poly[s > mid]
    if len(top) < 2 or len(bottom) < 2:
        return order_quad_clockwise(box)
    top = top[np.argsort(t[s <= mid])]
    bottom = bottom[np.argsort(t[s > mid])][::-1]
    return np.concatenate([top, bottom], axis=0).astype(np.float32)
