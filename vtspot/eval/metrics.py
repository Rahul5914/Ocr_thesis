"""Tracking metrics: MOTA, MOTP, IDF1 and HOTA -- in detection *and* spotting mode.

The distinction that decides whether your numbers mean anything:

* **Tracking mode.**  A prediction matches a ground-truth instance when their
  IoU clears the threshold.  This scores detection + association only.
* **Spotting mode.**  A match additionally requires the *transcription to be
  correct*.  This is what the ICDAR video "end-to-end" tasks score.

Reporting tracking-mode MOTA and calling it video text spotting overstates the
result by a wide margin, and it is an easy mistake to make because the metric
code is identical apart from one predicate.  Both modes are exposed here and
:func:`evaluate` reports them side by side so the difference is visible rather
than assumed.

Transcriptions are compared after the ICDAR normalisation: case-folded and
restricted to the evaluated alphabet.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

from ..utils.polygon import poly_iou

# A trajectory as consumed by the evaluator: frame index -> polygon, plus text.
Trajectory = Dict[str, object]

ALPHAS = np.arange(0.05, 0.99, 0.05)


def normalise_for_match(text: str, alphabet: Optional[str] = None) -> str:
    t = text.lower()
    if alphabet is not None:
        allowed = set(alphabet.lower())
        t = "".join(ch for ch in t if ch in allowed)
    return t


def build_frame_index(trajectories: Sequence[Trajectory]
                      ) -> Dict[int, List[Tuple[int, np.ndarray]]]:
    """``frame -> [(traj_position, polygon)]``."""
    index: Dict[int, List[Tuple[int, np.ndarray]]] = defaultdict(list)
    for i, traj in enumerate(trajectories):
        for frame, poly in traj["frames"].items():
            index[int(frame)].append((i, np.asarray(poly, np.float32)))
    return index


def similarity_matrices(gt: Sequence[Trajectory], pred: Sequence[Trajectory],
                        spotting: bool, alphabet: Optional[str] = None,
                        ignore_illegible: bool = True
                        ) -> Dict[int, Tuple[np.ndarray, List[int], List[int]]]:
    """Per-frame IoU matrices between ground-truth and predicted instances.

    In spotting mode pairs whose transcriptions disagree are forced to zero
    similarity, so they can never match at any IoU threshold.
    """
    gt_index = build_frame_index(gt)
    pred_index = build_frame_index(pred)
    gt_text = [normalise_for_match(str(t.get("text", "")), alphabet) for t in gt]
    pred_text = [normalise_for_match(str(t.get("text", "")), alphabet) for t in pred]

    out: Dict[int, Tuple[np.ndarray, List[int], List[int]]] = {}
    for frame in sorted(set(gt_index) | set(pred_index)):
        g = gt_index.get(frame, [])
        p = pred_index.get(frame, [])
        sim = np.zeros((len(g), len(p)), np.float32)
        for a, (gi, gpoly) in enumerate(g):
            for b, (pi, ppoly) in enumerate(p):
                value = poly_iou(gpoly, ppoly)
                if spotting and value > 0:
                    if ignore_illegible and not gt_text[gi]:
                        value = 0.0
                    elif gt_text[gi] != pred_text[pi]:
                        value = 0.0
                sim[a, b] = value
        out[frame] = (sim, [gi for gi, _ in g], [pi for pi, _ in p])
    return out


@dataclass
class MOTResult:
    mota: float = 0.0
    motp: float = 0.0
    idf1: float = 0.0
    idp: float = 0.0
    idr: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    hota: float = 0.0
    deta: float = 0.0
    assa: float = 0.0
    num_tp: int = 0
    num_fp: int = 0
    num_fn: int = 0
    num_idsw: int = 0
    num_gt: int = 0

    def as_dict(self) -> Dict[str, float]:
        return {k: (round(float(v), 4) if isinstance(v, float) else int(v))
                for k, v in self.__dict__.items()}


def clear_mot(sims: Dict[int, Tuple[np.ndarray, List[int], List[int]]],
              iou_threshold: float = 0.5) -> Tuple[float, float, int, int, int, int, int]:
    """CLEAR MOT: returns ``(MOTA, MOTP, TP, FP, FN, IDSW, GT)``.

    Follows the standard identity-preserving rule: a pair matched in the
    previous frame is kept if it still clears the threshold, *before* the
    Hungarian pass runs on the rest.  Skipping that step invents identity
    switches whenever two overlapping instances are equally good matches, which
    silently depresses MOTA.
    """
    tp = fp = fn = idsw = gt_total = 0
    dist_sum = 0.0
    prev_matches: Dict[int, int] = {}          # gt traj index -> pred traj index

    for frame in sorted(sims):
        sim, gt_ids, pred_ids = sims[frame]
        gt_total += len(gt_ids)
        matched_gt, matched_pred = set(), set()
        matches: Dict[int, int] = {}

        # 1. carry over still-valid matches from the previous frame
        for a, gi in enumerate(gt_ids):
            pj = prev_matches.get(gi)
            if pj is None or pj not in pred_ids:
                continue
            b = pred_ids.index(pj)
            if sim[a, b] >= iou_threshold:
                matches[gi] = pj
                matched_gt.add(a); matched_pred.add(b)
                dist_sum += float(sim[a, b])

        # 2. Hungarian on what is left
        free_g = [a for a in range(len(gt_ids)) if a not in matched_gt]
        free_p = [b for b in range(len(pred_ids)) if b not in matched_pred]
        if free_g and free_p:
            sub = sim[np.ix_(free_g, free_p)]
            rows, cols = linear_sum_assignment(-sub)
            for r, c in zip(rows, cols):
                if sub[r, c] >= iou_threshold:
                    gi, pj = gt_ids[free_g[r]], pred_ids[free_p[c]]
                    matches[gi] = pj
                    matched_gt.add(free_g[r]); matched_pred.add(free_p[c])
                    dist_sum += float(sub[r, c])

        for gi, pj in matches.items():
            if gi in prev_matches and prev_matches[gi] != pj:
                idsw += 1
        tp += len(matches)
        fn += len(gt_ids) - len(matches)
        fp += len(pred_ids) - len(matches)
        # Only update identities for GT present in this frame; a GT that
        # disappears keeps its last association so a reappearance is not
        # counted as a switch.
        for gi in gt_ids:
            if gi in matches:
                prev_matches[gi] = matches[gi]

    mota = 1.0 - (fn + fp + idsw) / max(gt_total, 1)
    motp = dist_sum / max(tp, 1)
    return mota, motp, tp, fp, fn, idsw, gt_total


def identity_metrics(sims: Dict[int, Tuple[np.ndarray, List[int], List[int]]],
                     num_gt: int, num_pred: int, iou_threshold: float = 0.5
                     ) -> Tuple[float, float, float]:
    """IDF1 / IDP / IDR via one global trajectory-to-trajectory assignment."""
    if num_gt == 0 or num_pred == 0:
        return 0.0, 0.0, 0.0
    overlap = np.zeros((num_gt, num_pred), np.float64)
    gt_len = np.zeros(num_gt, np.float64)
    pred_len = np.zeros(num_pred, np.float64)

    for frame in sorted(sims):
        sim, gt_ids, pred_ids = sims[frame]
        for gi in gt_ids:
            gt_len[gi] += 1
        for pj in pred_ids:
            pred_len[pj] += 1
        for a, gi in enumerate(gt_ids):
            for b, pj in enumerate(pred_ids):
                if sim[a, b] >= iou_threshold:
                    overlap[gi, pj] += 1

    # Maximise total true positives over a global one-to-one identity mapping.
    rows, cols = linear_sum_assignment(-overlap)
    idtp = float(sum(overlap[r, c] for r, c in zip(rows, cols)))
    idfn = float(gt_len.sum() - idtp)
    idfp = float(pred_len.sum() - idtp)
    denom = 2 * idtp + idfp + idfn
    idf1 = 2 * idtp / denom if denom > 0 else 0.0
    idp = idtp / max(idtp + idfp, 1e-9)
    idr = idtp / max(idtp + idfn, 1e-9)
    return idf1, idp, idr


def hota(sims: Dict[int, Tuple[np.ndarray, List[int], List[int]]],
         num_gt: int, num_pred: int, alphas: np.ndarray = ALPHAS
         ) -> Tuple[float, float, float]:
    """HOTA, DetA and AssA, averaged over the alpha sweep.

    Implements the two-pass formulation: a first pass counts potential matches
    between every ground-truth and predicted trajectory, and the per-frame
    assignment in the second pass is biased by that global alignment score, so
    the matching prefers pairs that are consistent over the whole sequence
    rather than merely best in the current frame.
    """
    if num_gt == 0 or num_pred == 0:
        return 0.0, 0.0, 0.0

    potential = np.zeros((num_gt, num_pred), np.float64)
    gt_count = np.zeros(num_gt, np.float64)
    pred_count = np.zeros(num_pred, np.float64)
    for frame in sorted(sims):
        sim, gt_ids, pred_ids = sims[frame]
        for gi in gt_ids:
            gt_count[gi] += 1
        for pj in pred_ids:
            pred_count[pj] += 1
        for a, gi in enumerate(gt_ids):
            for b, pj in enumerate(pred_ids):
                if sim[a, b] > 0:
                    potential[gi, pj] += 1
    union = gt_count[:, None] + pred_count[None, :] - potential
    global_align = potential / np.maximum(union, 1e-9)

    hota_scores, det_scores, ass_scores = [], [], []
    for alpha in alphas:
        matched = np.zeros((num_gt, num_pred), np.float64)
        tp = fp = fn = 0
        pairs: List[Tuple[int, int]] = []
        for frame in sorted(sims):
            sim, gt_ids, pred_ids = sims[frame]
            if len(gt_ids) and len(pred_ids):
                score = sim * global_align[np.ix_(gt_ids, pred_ids)]
                rows, cols = linear_sum_assignment(-score)
                frame_pairs = [(r, c) for r, c in zip(rows, cols)
                               if sim[r, c] >= alpha]
            else:
                frame_pairs = []
            for r, c in frame_pairs:
                gi, pj = gt_ids[r], pred_ids[c]
                matched[gi, pj] += 1
                pairs.append((gi, pj))
            tp += len(frame_pairs)
            fn += len(gt_ids) - len(frame_pairs)
            fp += len(pred_ids) - len(frame_pairs)

        det_a = tp / max(tp + fn + fp, 1e-9)
        if tp == 0:
            ass_a = 0.0
        else:
            ass_union = gt_count[:, None] + pred_count[None, :] - matched
            ass_iou = matched / np.maximum(ass_union, 1e-9)
            ass_a = float(np.mean([ass_iou[gi, pj] for gi, pj in pairs]))
        det_scores.append(det_a)
        ass_scores.append(ass_a)
        hota_scores.append(float(np.sqrt(det_a * ass_a)))
    return float(np.mean(hota_scores)), float(np.mean(det_scores)), float(np.mean(ass_scores))


def evaluate(gt: Sequence[Trajectory], pred: Sequence[Trajectory],
             spotting: bool = False, iou_threshold: float = 0.5,
             alphabet: Optional[str] = None) -> MOTResult:
    """All metrics for one video."""
    sims = similarity_matrices(gt, pred, spotting=spotting, alphabet=alphabet)
    mota, motp, tp, fp, fn, idsw, gt_total = clear_mot(sims, iou_threshold)
    idf1, idp, idr = identity_metrics(sims, len(gt), len(pred), iou_threshold)
    h, d, a = hota(sims, len(gt), len(pred))
    return MOTResult(mota=mota, motp=motp, idf1=idf1, idp=idp, idr=idr,
                     precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1),
                     hota=h, deta=d, assa=a, num_tp=tp, num_fp=fp, num_fn=fn,
                     num_idsw=idsw, num_gt=gt_total)


def evaluate_dataset(per_video: Sequence[Tuple[Sequence[Trajectory], Sequence[Trajectory]]],
                     spotting: bool = False, iou_threshold: float = 0.5,
                     alphabet: Optional[str] = None) -> MOTResult:
    """Corpus-level metrics.

    MOTA is accumulated over the whole corpus rather than averaged per video --
    per-video averaging lets a two-instance clip weigh as much as a
    thousand-instance one, which is not what the benchmarks report.
    """
    tp = fp = fn = idsw = gt_total = 0
    dist_sum = 0.0
    idf1s, hotas, detas, assas = [], [], [], []
    for gt, pred in per_video:
        sims = similarity_matrices(gt, pred, spotting=spotting, alphabet=alphabet)
        _, motp, v_tp, v_fp, v_fn, v_idsw, v_gt = clear_mot(sims, iou_threshold)
        tp += v_tp; fp += v_fp; fn += v_fn; idsw += v_idsw; gt_total += v_gt
        dist_sum += motp * v_tp
        idf1s.append(identity_metrics(sims, len(gt), len(pred), iou_threshold)[0])
        h, d, a = hota(sims, len(gt), len(pred))
        hotas.append(h); detas.append(d); assas.append(a)

    return MOTResult(
        mota=1.0 - (fn + fp + idsw) / max(gt_total, 1),
        motp=dist_sum / max(tp, 1),
        idf1=float(np.mean(idf1s)) if idf1s else 0.0,
        precision=tp / max(tp + fp, 1), recall=tp / max(tp + fn, 1),
        hota=float(np.mean(hotas)) if hotas else 0.0,
        deta=float(np.mean(detas)) if detas else 0.0,
        assa=float(np.mean(assas)) if assas else 0.0,
        num_tp=tp, num_fp=fp, num_fn=fn, num_idsw=idsw, num_gt=gt_total)
