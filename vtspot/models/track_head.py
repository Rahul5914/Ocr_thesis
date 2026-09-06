"""Per-instance appearance embedding used for cross-frame association."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TrackEmbedHead(nn.Module):
    """Rectified instance features -> L2-normalised embedding.

    Kept deliberately small.  The embedding's job is to *disambiguate*
    candidates that the motion prior has already narrowed down, not to identify
    text globally -- scene text repeats constantly (the same shop sign, the same
    word on two adjacent posters), so an appearance-only tracker will always
    have irreducible ambiguity.  See ``vtspot/tracking/matcher.py`` for how the
    motion prior and this embedding are combined.
    """

    def __init__(self, in_channels: int, embed_dim: int = 128, hidden: int = 256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(min(32, hidden), hidden),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(min(32, hidden), hidden),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(hidden, embed_dim)
        self.embed_dim = embed_dim

    def forward(self, roi_feats: torch.Tensor) -> torch.Tensor:
        """``(N, C, H, W)`` -> ``(N, embed_dim)``, unit norm."""
        if roi_feats.numel() == 0:
            return roi_feats.new_zeros((0, self.embed_dim))
        x = self.conv(roi_feats)
        x = self.pool(x).flatten(1)
        return F.normalize(self.fc(x), dim=-1, eps=1e-6)
