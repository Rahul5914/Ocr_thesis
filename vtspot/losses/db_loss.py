"""Losses for the differentiable-binarization detection head."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class BalancedBCELoss(nn.Module):
    """BCE with online hard-negative mining at a fixed negative:positive ratio.

    Text occupies a few percent of pixels.  Plain BCE lets the background term
    dominate the gradient, and a from-scratch model responds by predicting
    "background everywhere" and sitting there.  Restricting the negative term to
    the ``ratio * n_pos`` hardest background pixels keeps the two terms
    comparable throughout training.
    """

    def __init__(self, ratio: float = 3.0, min_negatives: int = 32):
        super().__init__()
        self.ratio = ratio
        self.min_negatives = min_negatives

    def forward(self, logits: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        positive = (target * mask).bool()
        negative = ((1 - target) * mask).bool()
        n_pos = int(positive.sum())
        n_neg = min(int(negative.sum()),
                    max(int(n_pos * self.ratio), self.min_negatives))
        loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        if n_neg <= 0:
            return loss.sum() * 0.0
        pos_loss = loss[positive].sum() if n_pos > 0 else loss.sum() * 0.0
        neg_loss = loss[negative]
        neg_loss, _ = neg_loss.topk(n_neg)
        return (pos_loss + neg_loss.sum()) / max(n_pos + n_neg, 1)


class DiceLoss(nn.Module):
    """Soft dice on the approximate binary map; complements BCE on thin strokes."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        pred, target = pred * mask, target * mask
        inter = (pred * target).sum()
        union = pred.sum() + target.sum() + self.eps
        return 1.0 - 2.0 * inter / union


class DBLoss(nn.Module):
    """``alpha * BCE(P) + beta * L1(T) + Dice(B)`` -- the DBNet combination."""

    def __init__(self, alpha: float = 1.0, beta: float = 10.0,
                 ohem_ratio: float = 3.0):
        super().__init__()
        self.alpha, self.beta = alpha, beta
        self.bce = BalancedBCELoss(ratio=ohem_ratio)
        self.dice = DiceLoss()

    def forward(self, pred: Dict[str, torch.Tensor],
                target: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        shrink_map = target["shrink_map"]
        shrink_mask = target["shrink_mask"]
        thresh_map = target["thresh_map"]
        thresh_mask = target["thresh_mask"]

        loss_prob = self.bce(pred["prob_logit"], shrink_map, shrink_mask)
        denom = thresh_mask.sum().clamp(min=1.0)
        loss_thresh = (torch.abs(pred["thresh"] - thresh_map) * thresh_mask).sum() / denom

        losses = {
            "loss_prob": self.alpha * loss_prob,
            "loss_thresh": self.beta * loss_thresh,
        }
        if "binary" in pred:
            losses["loss_binary"] = self.dice(pred["binary"], shrink_map, shrink_mask)
        losses["loss_det"] = sum(losses.values())
        return losses
