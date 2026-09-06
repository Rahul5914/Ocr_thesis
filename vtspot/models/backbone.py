"""From-scratch visual backbone.

Design notes for the no-pretraining setting:

* **GroupNorm, not BatchNorm.**  Video clip training means a batch of 2-4 clips,
  so the effective per-step batch of *images* is small and BN statistics become
  noise.  GroupNorm is batch-size independent and costs nothing at inference.
* **Stem stride 4, not 4-with-maxpool-to-8.**  Video text is small (DSText
  averages ~24 instances per frame, most of them tiny).  Keeping the finest
  pyramid level at stride 4 is what makes small text recoverable at all.
* **Width is a knob.**  A from-scratch model trained on synthetic data does not
  need ResNet-50 capacity; ``width=32`` (~5M params) trains far faster and
  overfits far less than a randomly-initialised ResNet-50.
"""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .init import apply_fixup_scaling


def _gn_groups(channels: int, preferred: int = 32) -> int:
    """Largest divisor of ``channels`` that is <= ``preferred``.

    GroupNorm requires ``channels % groups == 0``; widths like 48 or 24 are
    common once ``width`` is a tunable, so pick the group count rather than
    constraining the architecture to powers of two.
    """
    for g in range(min(preferred, channels), 0, -1):
        if channels % g == 0:
            return g
    return 1


def norm_layer(kind: str, channels: int, groups: int = 32) -> nn.Module:
    if kind == "gn":
        return nn.GroupNorm(_gn_groups(channels, groups), channels)
    if kind == "bn":
        return nn.BatchNorm2d(channels)
    if kind == "none":
        return nn.Identity()
    raise ValueError(f"unknown norm {kind!r}")


class ConvNormAct(nn.Sequential):
    def __init__(self, cin: int, cout: int, k: int = 3, stride: int = 1,
                 norm: str = "gn", act: bool = True):
        layers: List[nn.Module] = [
            nn.Conv2d(cin, cout, k, stride=stride, padding=k // 2, bias=(norm == "none")),
            norm_layer(norm, cout),
        ]
        if act:
            layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)


class BasicBlock(nn.Module):
    """Standard pre-activation-free residual block with GroupNorm."""

    expansion = 1

    def __init__(self, cin: int, cout: int, stride: int = 1, norm: str = "gn"):
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.norm1 = norm_layer(norm, cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.norm2 = norm_layer(norm, cout)
        self.relu = nn.ReLU(inplace=True)
        self.downsample: nn.Module | None = None
        if stride != 1 or cin != cout:
            self.downsample = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride=stride, bias=False),
                norm_layer(norm, cout),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        return self.relu(out + identity)


class FixupBasicBlock(nn.Module):
    """Residual block with **no normalisation at all**, per Fixup.

    Carries the scalar biases (rule 3) and the scalar multiplier (rule 4) that
    the scaling rule alone cannot replace.
    """

    expansion = 1
    custom_init = True

    def __init__(self, cin: int, cout: int, stride: int = 1, **_):
        super().__init__()
        self.bias1a = nn.Parameter(torch.zeros(1))
        self.conv1 = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.bias1b = nn.Parameter(torch.zeros(1))
        self.relu = nn.ReLU(inplace=True)
        self.bias2a = nn.Parameter(torch.zeros(1))
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1, bias=False)
        self.scale = nn.Parameter(torch.ones(1))
        self.bias2b = nn.Parameter(torch.zeros(1))
        self.downsample: nn.Module | None = None
        if stride != 1 or cin != cout:
            self.downsample = nn.Conv2d(cin, cout, 1, stride=stride, bias=False)

    @property
    def conv_layers(self) -> Sequence[nn.Conv2d]:
        return (self.conv1, self.conv2)

    def reset_parameters_custom(self) -> None:
        # Scaling and the zeroed last conv are applied by apply_fixup_scaling at
        # the model level, which is the only place that knows L.
        nn.init.zeros_(self.bias1a); nn.init.zeros_(self.bias1b)
        nn.init.zeros_(self.bias2a); nn.init.zeros_(self.bias2b)
        nn.init.ones_(self.scale)
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight, mode="fan_in", nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shifted = x + self.bias1a
        identity = shifted if self.downsample is None else self.downsample(shifted)
        out = self.relu(self.conv1(shifted) + self.bias1b)
        out = self.conv2(out + self.bias2a) * self.scale + self.bias2b
        return self.relu(out + identity)


class TextBackbone(nn.Module):
    """4-stage residual backbone returning strides 4, 8, 16, 32.

    Args:
        width: channels of the first stage.  Stages widen 1x/2x/4x/8x.
        layers: blocks per stage.
        norm: ``gn`` (default), ``bn``, or ``none`` -- ``none`` switches to
            Fixup blocks, which is the only configuration where removing
            normalisation is safe.
        in_channels: 3 for RGB.
    """

    def __init__(self, width: int = 48, layers: Sequence[int] = (2, 2, 2, 2),
                 norm: str = "gn", in_channels: int = 3):
        super().__init__()
        if len(layers) != 4:
            raise ValueError("layers must have 4 entries")
        self.norm_kind = norm
        self.use_fixup = norm == "none"
        block = FixupBasicBlock if self.use_fixup else BasicBlock

        # stride-4 stem, two 3x3 convs rather than one 7x7: cheaper and better
        # conditioned from random init.
        stem_norm = "none" if self.use_fixup else norm
        self.stem = nn.Sequential(
            ConvNormAct(in_channels, width // 2, 3, stride=2, norm=stem_norm),
            ConvNormAct(width // 2, width, 3, stride=2, norm=stem_norm),
        )

        chans = [width, width * 2, width * 4, width * 8]
        strides = [1, 2, 2, 2]
        cin = width
        self.stages = nn.ModuleList()
        self._blocks: List[nn.Module] = []
        for stage_idx, (cout, n, s) in enumerate(zip(chans, layers, strides)):
            blocks = []
            for b in range(n):
                blk = block(cin, cout, stride=s if b == 0 else 1, norm=norm)
                blocks.append(blk)
                self._blocks.append(blk)
                cin = cout
            self.stages.append(nn.Sequential(*blocks))
        self.out_channels = chans
        self.out_strides = [4, 8, 16, 32]

        if self.use_fixup:
            apply_fixup_scaling(self._blocks, num_branches=len(self._blocks),
                                layers_per_branch=2)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.stem(x)
        feats: List[torch.Tensor] = []
        for stage in self.stages:
            x = stage(x)
            feats.append(x)
        return feats


class FPN(nn.Module):
    """Top-down feature pyramid fused to a single stride-4 map.

    All levels are projected to ``out_channels // 4``, upsampled to stride 4 and
    concatenated -- the DBNet arrangement.  Concatenation (rather than summation)
    keeps the high-resolution edge detail that small text depends on.
    """

    def __init__(self, in_channels: Sequence[int], out_channels: int = 256,
                 norm: str = "gn"):
        super().__init__()
        if out_channels % 4 != 0:
            raise ValueError("out_channels must be divisible by 4")
        inner = out_channels // 4
        self.lateral = nn.ModuleList(
            [nn.Conv2d(c, out_channels, 1, bias=False) for c in in_channels])
        self.smooth = nn.ModuleList(
            [ConvNormAct(out_channels, inner, 3, norm=norm) for _ in in_channels])
        self.out_channels = out_channels

    def forward(self, feats: Sequence[torch.Tensor]) -> torch.Tensor:
        laterals = [conv(f) for conv, f in zip(self.lateral, feats)]
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:], mode="nearest")
        outs = [smooth(l) for smooth, l in zip(self.smooth, laterals)]
        target = outs[0].shape[-2:]
        outs = [outs[0]] + [F.interpolate(o, size=target, mode="bilinear",
                                          align_corners=False) for o in outs[1:]]
        return torch.cat(outs, dim=1)
