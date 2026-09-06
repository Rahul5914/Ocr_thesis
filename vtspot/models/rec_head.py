"""Recognition head: a CTC branch plus a training-only attention branch (GTC).

Both branches share one sequence encoder.  At inference only the CTC linear
layer runs, so the deployed model keeps CTC's speed; the attention decoder
exists purely to shape the shared features during training, which is the
practically useful part of Guided Training of CTC.

Two from-scratch details that matter more than they look:

* **Pre-LN transformer blocks.**  Post-LN (the original "Attention is All You
  Need" arrangement) needs a learning-rate warmup to survive at all, and even
  with warmup it diverges more readily from random init because the residual
  path passes through LayerNorm.  Pre-LN keeps an unnormalised identity path and
  trains stably.  This is a common cause of "my transformer NaN'd on epoch 1"
  that the from-scratch literature rarely spells out.
* **Height is collapsed by convolution, not by pooling to 1 immediately.**  The
  8 rows out of PolyAlign carry the stroke detail that separates ``i`` from
  ``l``; collapsing them gradually preserves it.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PreLNEncoderLayer(nn.Module):
    def __init__(self, dim: int, heads: int = 4, mlp_ratio: float = 4.0,
                 dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, dim), nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.norm1(x)
        x = x + self.attn(h, h, h, key_padding_mask=key_padding_mask,
                          need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class SequenceEncoder(nn.Module):
    """``(N, C, H, W)`` rectified crops -> ``(N, W, dim)`` sequence."""

    def __init__(self, in_channels: int, dim: int = 256, in_h: int = 8,
                 layers: int = 2, heads: int = 4, dropout: float = 0.1,
                 max_len: int = 64):
        super().__init__()
        # Halve the height three times: 8 -> 4 -> 2 -> 1, width untouched.
        chans = [in_channels, dim // 2, dim, dim]
        convs = []
        for i in range(3):
            convs += [
                nn.Conv2d(chans[i], chans[i + 1], 3, stride=(2, 1), padding=1, bias=False),
                nn.GroupNorm(min(32, chans[i + 1]), chans[i + 1]),
                nn.ReLU(inplace=True),
            ]
        self.conv = nn.Sequential(*convs)
        self.proj = nn.Linear(chans[-1] * max(in_h // 8, 1), dim)
        self.pos = nn.Parameter(torch.zeros(1, max_len, dim))
        self.blocks = nn.ModuleList(
            [PreLNEncoderLayer(dim, heads, dropout=dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)
        self.dim = dim
        self.custom_init = True

    def reset_parameters_custom(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)   # correct for GELU/attention paths
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        x = self.conv(x)                       # (N, dim, H/8, W)
        x = x.permute(0, 3, 1, 2).flatten(2)   # (N, W, dim*H/8)
        x = self.proj(x)                       # (N, W, dim)
        length = x.shape[1]
        if length > self.pos.shape[1]:
            raise ValueError(f"sequence length {length} exceeds max_len {self.pos.shape[1]}")
        x = x + self.pos[:, :length]
        for blk in self.blocks:
            x = blk(x)
        return self.norm(x)


class AttentionDecoder(nn.Module):
    """Single-head additive-attention decoder, used only during training."""

    def __init__(self, dim: int, num_classes: int, max_len: int = 26):
        super().__init__()
        self.embed = nn.Embedding(num_classes, dim)
        self.cell = nn.GRUCell(dim * 2, dim)
        self.attn_q = nn.Linear(dim, dim)
        self.attn_k = nn.Linear(dim, dim)
        self.attn_v = nn.Linear(dim, 1)
        self.out = nn.Linear(dim, num_classes)
        self.dim, self.max_len = dim, max_len

    def forward(self, memory: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Teacher-forced decode.  ``targets`` is ``(N, L)``; returns ``(N, L, C)``."""
        n, t, _ = memory.shape
        keys = self.attn_k(memory)
        state = memory.new_zeros(n, self.dim)
        prev = memory.new_zeros(n, dtype=torch.long)          # [GO] == index 0
        logits = []
        for step in range(targets.shape[1]):
            score = self.attn_v(torch.tanh(keys + self.attn_q(state).unsqueeze(1)))
            weight = torch.softmax(score, dim=1)               # (N, T, 1)
            context = (weight * memory).sum(1)                 # (N, dim)
            state = self.cell(torch.cat([context, self.embed(prev)], dim=-1), state)
            logits.append(self.out(state))
            prev = targets[:, step]                            # teacher forcing
        return torch.stack(logits, dim=1)


class RecognitionHead(nn.Module):
    """CTC head (+ optional GTC attention branch)."""

    def __init__(self, in_channels: int, ctc_classes: int, attn_classes: int = 0,
                 dim: int = 256, in_h: int = 8, layers: int = 2, heads: int = 4,
                 dropout: float = 0.1, max_seq_len: int = 64, max_text_len: int = 26):
        super().__init__()
        self.encoder = SequenceEncoder(in_channels, dim=dim, in_h=in_h, layers=layers,
                                       heads=heads, dropout=dropout, max_len=max_seq_len)
        self.ctc = nn.Linear(dim, ctc_classes)
        self.attn_decoder = (AttentionDecoder(dim, attn_classes, max_text_len)
                             if attn_classes > 0 else None)
        self.ctc_classes = ctc_classes

    def forward(self, roi_feats: torch.Tensor,
                attn_targets: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        if roi_feats.numel() == 0:
            empty = roi_feats.new_zeros((0, 1, self.ctc_classes))
            return {"ctc_logits": empty, "memory": roi_feats.new_zeros((0, 1, 1))}
        memory = self.encoder(roi_feats)
        out = {"ctc_logits": self.ctc(memory), "memory": memory}
        if self.training and self.attn_decoder is not None and attn_targets is not None:
            out["attn_logits"] = self.attn_decoder(memory, attn_targets)
        return out
