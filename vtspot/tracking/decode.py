"""CTC decoding and trajectory-level transcription aggregation."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import List, Sequence, Tuple

import numpy as np


def ctc_greedy_decode(log_probs: np.ndarray, blank: int = 0
                      ) -> Tuple[List[int], float]:
    """Best-path decode of a ``(T, C)`` log-probability matrix.

    Returns the collapsed label sequence and a length-normalised confidence
    (mean per-frame probability of the chosen path).  Length normalisation
    matters: an un-normalised path score punishes long words, so a trajectory
    vote would systematically prefer short misreadings.
    """
    if log_probs.size == 0:
        return [], 0.0
    best = log_probs.argmax(axis=-1)
    conf = float(np.exp(log_probs.max(axis=-1)).mean())
    out: List[int] = []
    prev = -1
    for idx in best:
        idx = int(idx)
        if idx != prev and idx != blank:
            out.append(idx)
        prev = idx
    return out, conf


def levenshtein(a: Sequence, b: Sequence) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def aggregate_trajectory_text(observations: Sequence[Tuple[str, float]],
                              min_confidence: float = 0.0) -> Tuple[str, float]:
    """Fuse per-frame transcriptions of one trajectory into a single answer.

    This is the cheap stand-in for a vision-language aggregator.  TraRA reaches
    trajectory-level consistency by adapting a 9B VLM with LoRA; that is out of
    reach for a model trained from scratch, and it turns out most of the win is
    available for free.  Any individual frame may be blurred, occluded or
    clipped, but the *set* of frames rarely fails the same way, so:

      1. group observations by their string,
      2. score each group by ``sum of confidences`` (so a reading seen in ten
         frames beats one lucky frame),
      3. break near-ties by character-level majority vote across the members of
         the winning cluster, where "near" is a small edit distance.

    Zero extra parameters, no extra training, and it recovers exactly the
    failure mode trajectory-level aggregation exists to fix.
    """
    obs = [(t, c) for t, c in observations if t and c >= min_confidence]
    if not obs:
        return "", 0.0

    groups: dict[str, List[float]] = defaultdict(list)
    for text, conf in obs:
        groups[text].append(conf)
    scored = sorted(groups.items(), key=lambda kv: (sum(kv[1]), len(kv[1])), reverse=True)
    best_text, best_confs = scored[0]

    # Pull in near-identical readings (<=1 edit) and vote per character; this
    # repairs single-character errors that no individual frame gets right.
    cluster = [(t, c) for t, c in obs
               if levenshtein(t, best_text) <= 1 and len(t) == len(best_text)]
    if len(cluster) >= 3:
        voted = []
        for pos in range(len(best_text)):
            counts: Counter = Counter()
            for text, conf in cluster:
                counts[text[pos]] += conf
            voted.append(counts.most_common(1)[0][0])
        best_text = "".join(voted)

    confidence = float(np.mean(best_confs))
    return best_text, confidence
