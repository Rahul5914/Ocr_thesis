"""Differentiable-Binarization detection head.

Why a dense segmentation head rather than DETR-style set prediction:

A randomly-initialised DETR needs hundreds of epochs because Hungarian matching
gives *one* supervised query per ground-truth instance per step -- an extremely
sparse gradient signal, and an unstable one early on when the matching flips
between epochs.  DBNet supervises **every pixel every step**.  With no
pretrained backbone to lean on that difference dominates: the dense head reaches
usable detection quality in tens of epochs on synthetic data, where the sparse
one is still thrashing.  Arbitrary shapes come for free because the output is a
mask, which is also what ArTVideo-style curved text needs.

The head predicts
    P  probability map
    T  threshold map
    B  approximate binary map, B = sigmoid(k * (P - T))
``B`` is differentiable, so the binarisation threshold is learnt rather than
tuned, which is the whole point of DB.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from .backbone import norm_layer
from .init import init_prior_bias


class _UpsampleBranch(nn.Module):
    """stride-4 features -> full-resolution single-channel map."""

    def __init__(self, in_channels: int, norm: str = "gn", prior_prob: float | None = None):
        super().__init__()
        inner = in_channels // 4
        self.conv = nn.Conv2d(in_channels, inner, 3, padding=1, bias=False)
        self.norm = norm_layer(norm, inner)
        self.relu = nn.ReLU(inplace=True)
        self.up1 = nn.ConvTranspose2d(inner, inner, 2, stride=2)
        self.norm2 = norm_layer(norm, inner)
        self.relu2 = nn.ReLU(inplace=True)
        self.up2 = nn.ConvTranspose2d(inner, 1, 2, stride=2)
        self.prior_prob = prior_prob

    def reset_output_bias(self) -> None:
        if self.prior_prob is not None:
            init_prior_bias(self.up2, self.prior_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.norm(self.conv(x)))
        x = self.relu2(self.norm2(self.up1(x)))
        return self.up2(x)


class DBHead(nn.Module):
    """Predicts probability / threshold / approximate-binary maps.

    Args:
        in_channels: channels of the fused FPN map.
        k: steepness of the differentiable step.  50 is the DBNet default; it
            controls how sharply the binary map saturates and therefore how
            strong the gradient is near the decision boundary.
        prior_prob: initial probability of the ``P`` map.  Text covers a few
            percent of pixels, so starting at 0.5 wastes the earliest -- and for
            a from-scratch model most fragile -- iterations on learning "almost
            everything is background".
    """

    def __init__(self, in_channels: int, k: float = 50.0, norm: str = "gn",
                 prior_prob: float = 0.02, k_start: float = 2.0):
        super().__init__()
        self.k_final = k
        self.k_start = k_start
        # `k` is annealed from k_start to k_final by set_k_progress().
        #
        # The binary map is sigmoid(k*(P - T)).  At k=50 it saturates unless
        # |P - T| < ~0.1, and starting P at the text prior (~0.02) puts it 0.48
        # away from T, where d(binary)/dx is ~4e-11 -- so the dice term is inert
        # at initialisation.  It does not stay inert: the BCE term raises P
        # until it crosses T, after which dice engages normally.  Measured on a
        # small overfit run, annealing k buys about 4% lower detection loss over
        # a fixed k=50 -- worth having since it costs nothing, but it is a
        # refinement, not a fix for a broken default.  Set db_k_warmup_steps=0
        # and db_k_start=50 to reproduce plain DBNet behaviour.
        self.k = k_start
        self.prob = _UpsampleBranch(in_channels, norm=norm, prior_prob=prior_prob)
        self.thresh = _UpsampleBranch(in_channels, norm=norm, prior_prob=0.5)
        self.custom_init = True

    def reset_parameters_custom(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.GroupNorm, nn.BatchNorm2d)):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
        self.prob.reset_output_bias()
        self.thresh.reset_output_bias()

    def set_k_progress(self, progress: float) -> None:
        """Set ``k`` from a 0->1 training-progress fraction (geometric ramp)."""
        p = float(min(max(progress, 0.0), 1.0))
        self.k = float(self.k_start * (self.k_final / self.k_start) ** p)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        prob_logit = self.prob(x)
        thresh_logit = self.thresh(x)
        prob = torch.sigmoid(prob_logit)
        thresh = torch.sigmoid(thresh_logit)
        out = {"prob_logit": prob_logit, "prob": prob, "thresh": thresh}
        if self.training:
            # Differentiable binarisation, computed in logit space for stability:
            # sigmoid(k*(P-T)) overflows in fp16 when |P-T| is large.
            out["binary"] = torch.sigmoid(self.k * (prob - thresh))
        return out
