"""Recognition losses: CTC, plus the auxiliary attention CE used for GTC."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CTCRecognitionLoss(nn.Module):
    """CTC over the per-instance sequence logits.

    Guards worth keeping: CTC is undefined when the input is shorter than the
    label (``T < 2U-1`` for labels with repeats), and returns ``inf`` rather than
    erroring.  Those instances are dropped instead, and ``zero_infinity`` catches
    anything that slips through -- a single ``inf`` here poisons the whole step.
    """

    def __init__(self, blank: int = 0):
        super().__init__()
        self.blank = blank
        self.ctc = nn.CTCLoss(blank=blank, reduction="sum", zero_infinity=True)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                target_lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: ``(N, T, C)``
            targets: ``(N, L)`` padded label indices
            target_lengths: ``(N,)``
        """
        if logits.numel() == 0 or int(target_lengths.sum()) == 0:
            return logits.sum() * 0.0
        n, t, _ = logits.shape
        input_lengths = torch.full((n,), t, dtype=torch.long, device=logits.device)

        # A label of length L needs at least L + (number of adjacent repeats)
        # frames; drop instances that cannot be represented at this crop width.
        repeats = (targets[:, 1:] == targets[:, :-1]).long()
        valid_pos = (torch.arange(targets.shape[1] - 1, device=targets.device)[None, :]
                     < (target_lengths - 1)[:, None])
        min_len = target_lengths + (repeats * valid_pos).sum(dim=1)
        keep = (target_lengths > 0) & (min_len <= t)
        if not bool(keep.any()):
            return logits.sum() * 0.0

        log_probs = F.log_softmax(logits[keep], dim=-1).permute(1, 0, 2)  # (T, N, C)
        flat = torch.cat([targets[i, : target_lengths[i]] for i in range(n) if keep[i]])
        loss = self.ctc(log_probs, flat, input_lengths[keep], target_lengths[keep])
        return loss / keep.sum().clamp(min=1)


class AttentionGuidanceLoss(nn.Module):
    """Cross-entropy on the training-only attention branch.

    Ignores the PAD index so padded tail positions contribute nothing.  Label
    smoothing helps here specifically because the attention branch is a
    *teacher* for the shared encoder -- overconfident targets make it memorise
    rather than shape useful features.
    """

    def __init__(self, ignore_index: int = 0, label_smoothing: float = 0.1):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(ignore_index=ignore_index,
                                        label_smoothing=label_smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.numel() == 0:
            return logits.sum() * 0.0
        return self.loss(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
