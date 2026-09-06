"""Long-Short-Term matching for video text trajectories.

Short-term matching (adjacent frames) is easy and is where almost all
associations happen.  Long-term matching exists for the cases short-term cannot
see: an instance occluded for fifteen frames, or one that leaves and re-enters
the frame.  Splitting them lets each use the evidence it can actually trust --
the short-term matcher leans on motion, which is reliable across one frame and
useless across thirty; the long-term matcher leans on appearance, which survives
the gap but cannot separate two identical words.

Both are solved as linear assignment problems (Hungarian), which is optimal for
the given cost matrix -- unlike greedy nearest-neighbour matching, which locks
in an early wrong pair and cascades.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..utils.polygon import bbox_iou_matrix


@dataclass
class MatchConfig:
    # short-term
    iou_weight: float = 0.6
    embed_weight: float = 0.4
    max_cost: float = 0.75            # reject pairs worse than this
    motion_gate: float = 4.0          # max centre displacement, in instance widths
    # long-term
    long_embed_threshold: float = 0.6  # cosine similarity, stricter than short-term
    max_lost_frames: int = 30
    long_motion_gate: float = 8.0


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between ``(N, D)`` and ``(M, D)`` unit-norm embeddings."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-6)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-6)
    return (a @ b.T).astype(np.float32)


def centre_distance_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Centre distance normalised by the mean instance width of each pair."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), np.float32)
    ca = np.stack([(boxes_a[:, 0] + boxes_a[:, 2]) / 2,
                   (boxes_a[:, 1] + boxes_a[:, 3]) / 2], axis=1)
    cb = np.stack([(boxes_b[:, 0] + boxes_b[:, 2]) / 2,
                   (boxes_b[:, 1] + boxes_b[:, 3]) / 2], axis=1)
    dist = np.linalg.norm(ca[:, None, :] - cb[None, :, :], axis=-1)
    wa = np.maximum(boxes_a[:, 2] - boxes_a[:, 0], 1.0)
    wb = np.maximum(boxes_b[:, 2] - boxes_b[:, 0], 1.0)
    scale = (wa[:, None] + wb[None, :]) / 2.0
    return (dist / scale).astype(np.float32)


def solve_assignment(cost: np.ndarray, max_cost: float
                     ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """Hungarian assignment with a cost ceiling.

    Returns ``(matches, unmatched_rows, unmatched_cols)``.
    """
    n, m = cost.shape
    if n == 0 or m == 0:
        return [], list(range(n)), list(range(m))
    rows, cols = linear_sum_assignment(cost)
    matches: List[Tuple[int, int]] = []
    matched_rows, matched_cols = set(), set()
    for r, c in zip(rows, cols):
        if cost[r, c] <= max_cost:
            matches.append((int(r), int(c)))
            matched_rows.add(int(r))
            matched_cols.add(int(c))
    return (matches,
            [i for i in range(n) if i not in matched_rows],
            [j for j in range(m) if j not in matched_cols])


class ShortTermMatcher:
    """Adjacent-frame association from motion + appearance."""

    def __init__(self, cfg: MatchConfig):
        self.cfg = cfg

    def __call__(self, track_boxes: np.ndarray, track_embeds: np.ndarray,
                 det_boxes: np.ndarray, det_embeds: np.ndarray
                 ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        cfg = self.cfg
        if len(track_boxes) == 0 or len(det_boxes) == 0:
            return [], list(range(len(track_boxes))), list(range(len(det_boxes)))
        iou = bbox_iou_matrix(track_boxes, det_boxes)
        sim = cosine_matrix(track_embeds, det_embeds)
        cost = cfg.iou_weight * (1.0 - iou) + cfg.embed_weight * (1.0 - sim)
        # Hard gate on implausible motion: without it, appearance similarity
        # alone will happily link two identical words at opposite corners.
        gate = centre_distance_matrix(track_boxes, det_boxes) > cfg.motion_gate
        cost = np.where(gate, cfg.max_cost + 1.0, cost)
        return solve_assignment(cost, cfg.max_cost)


class LongTermMatcher:
    """Re-association of lost tracks, on appearance with a motion sanity check."""

    def __init__(self, cfg: MatchConfig):
        self.cfg = cfg

    def __call__(self, lost_boxes: np.ndarray, lost_embeds: np.ndarray,
                 lost_ages: np.ndarray, det_boxes: np.ndarray, det_embeds: np.ndarray
                 ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        cfg = self.cfg
        if len(lost_boxes) == 0 or len(det_boxes) == 0:
            return [], list(range(len(lost_boxes))), list(range(len(det_boxes)))
        sim = cosine_matrix(lost_embeds, det_embeds)
        cost = 1.0 - sim
        # The gate widens with how long the track has been lost -- an instance
        # unseen for 20 frames may legitimately have travelled much further.
        gate_scale = cfg.long_motion_gate * np.maximum(lost_ages, 1)[:, None]
        gate = centre_distance_matrix(lost_boxes, det_boxes) > gate_scale
        cost = np.where(gate, 2.0, cost)
        return solve_assignment(cost, 1.0 - cfg.long_embed_threshold)
