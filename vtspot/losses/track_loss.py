"""Contrastive loss for the cross-frame association embedding.

Multi-positive InfoNCE (SupCon): every other appearance of the same track inside
the clip is a positive, everything else is a negative.

The wrinkle the video-text-spotting literature underplays: *scene text repeats*.
The same word appears on two adjacent shop signs, a price appears twice on one
menu, a logo recurs across the frame.  An appearance embedding trained with
plain InfoNCE is being asked to separate pairs that are genuinely
indistinguishable from pixels alone, and it will either fail on them or distort
the rest of the space trying.  Two mitigations are built in:

* ``hard_negative_weight`` up-weights negatives that share a transcription --
  these are the pairs the embedding must learn to split using context (stroke
  detail, background, colour) rather than word identity.
* ``ambiguous_weight`` (< 1) instead *down*-weights them, for the opposite
  policy: accept that they are unresolvable by appearance and let the motion
  prior in the matcher break the tie.

Which to use is an empirical call, so both are exposed; the default (1.0) is
plain SupCon.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class ContrastiveTrackLoss(nn.Module):
    def __init__(self, temperature: float = 0.1, hard_negative_weight: float = 1.0,
                 ambiguous_weight: float = 1.0):
        super().__init__()
        self.temperature = temperature
        self.hard_negative_weight = hard_negative_weight
        self.ambiguous_weight = ambiguous_weight

    def forward(self, embeddings: torch.Tensor, track_ids: torch.Tensor,
                frame_ids: Optional[torch.Tensor] = None,
                text_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            embeddings: ``(N, D)``, L2-normalised.
            track_ids: ``(N,)`` instance identity within the clip.
            frame_ids: ``(N,)``; when given, same-frame pairs are excluded so the
                loss only rewards *temporal* consistency rather than trivially
                separating co-occurring instances.
            text_ids: ``(N,)`` transcription hash, used for the hard/ambiguous
                negative re-weighting described above.
        """
        n = embeddings.shape[0]
        if n < 2:
            return embeddings.sum() * 0.0

        sim = embeddings @ embeddings.t() / self.temperature
        eye = torch.eye(n, dtype=torch.bool, device=embeddings.device)
        sim = sim.masked_fill(eye, float("-inf"))

        same_track = track_ids[:, None] == track_ids[None, :]
        positives = same_track & ~eye
        if frame_ids is not None:
            positives = positives & (frame_ids[:, None] != frame_ids[None, :])

        valid = positives.any(dim=1)
        if not bool(valid.any()):
            return embeddings.sum() * 0.0

        # Re-weight the denominator: negatives sharing a transcription get
        # hard_negative_weight, everything else 1.
        log_w = torch.zeros_like(sim)
        if text_ids is not None and (self.hard_negative_weight != 1.0
                                     or self.ambiguous_weight != 1.0):
            same_text_diff_track = (text_ids[:, None] == text_ids[None, :]) & ~same_track
            weight = self.hard_negative_weight * self.ambiguous_weight
            log_w = log_w.masked_fill(same_text_diff_track,
                                      torch.log(torch.tensor(weight)).item())

        log_denom = torch.logsumexp(sim + log_w, dim=1)
        log_prob = sim - log_denom[:, None]

        pos_count = positives.sum(dim=1).clamp(min=1)
        mean_log_prob = (log_prob.masked_fill(~positives, 0.0).sum(dim=1) / pos_count)
        return -(mean_log_prob[valid]).mean()
