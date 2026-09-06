"""PolyAlign: rectify an arbitrarily-shaped text region into a fixed grid.

The recognition head needs a horizontal, left-to-right crop of each instance.
Axis-aligned ``roi_align`` cannot provide that for rotated text, and a rotated
RoI cannot provide it for curved text -- which is >30% of ArTVideo.

PolyAlign generalises both.  An instance is described by ``2K`` control points:
``K`` along the top boundary and ``K`` along the bottom boundary, both ordered
left-to-right in reading direction.  Sampling at grid position ``(u, v)`` with
``u`` in ``[0,1]`` along the text and ``v`` in ``[0,1]`` across it is

    p(u, v) = (1 - v) * top(u) + v * bottom(u)

where ``top`` and ``bottom`` are linear interpolations of the control points.
The whole thing is a ``grid_sample`` call, so it is differentiable end-to-end,
runs on CPU and GPU, and needs no compiled extension.  A rotated rectangle is
the special case ``K = 2``.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_polyalign_grid(control_points: torch.Tensor, out_h: int, out_w: int,
                         feat_h: int, feat_w: int) -> torch.Tensor:
    """Build a ``grid_sample`` grid from control points.

    Args:
        control_points: ``(N, 2K, 2)`` in **feature-map pixel** coordinates,
            top edge (K points, left->right) followed by bottom edge
            (K points, left->right).
        out_h, out_w: size of the rectified output.
        feat_h, feat_w: size of the feature map being sampled.

    Returns:
        ``(N, out_h, out_w, 2)`` grid in ``grid_sample`` normalised coordinates.
    """
    if control_points.dim() != 3 or control_points.shape[-1] != 2:
        raise ValueError(f"expected (N, 2K, 2), got {tuple(control_points.shape)}")
    n, two_k, _ = control_points.shape
    if two_k % 2 != 0 or two_k < 4:
        raise ValueError(f"need an even number of >=4 control points, got {two_k}")
    k = two_k // 2
    device, dtype = control_points.device, control_points.dtype

    top = control_points[:, :k, :]          # (N, K, 2)
    bottom = control_points[:, k:, :]       # (N, K, 2)

    # Resample both boundaries to out_w points along the reading direction.
    # interpolate() works on (N, C, L), so move the coordinate axis to C.
    def densify(edge: torch.Tensor) -> torch.Tensor:
        e = edge.permute(0, 2, 1)                                   # (N, 2, K)
        e = F.interpolate(e, size=out_w, mode="linear", align_corners=True)
        return e.permute(0, 2, 1)                                   # (N, out_w, 2)

    top_d = densify(top)
    bottom_d = densify(bottom)

    # Linear blend across the text height.
    v = torch.linspace(0.0, 1.0, out_h, device=device, dtype=dtype).view(1, out_h, 1, 1)
    grid = (1.0 - v) * top_d.unsqueeze(1) + v * bottom_d.unsqueeze(1)  # (N,out_h,out_w,2)

    # Pixel centres -> normalised [-1, 1], matching align_corners=False.
    gx = 2.0 * (grid[..., 0] + 0.5) / max(feat_w, 1) - 1.0
    gy = 2.0 * (grid[..., 1] + 0.5) / max(feat_h, 1) - 1.0
    return torch.stack([gx, gy], dim=-1)


class PolyAlign(nn.Module):
    """Extract rectified per-instance features from a shared feature map."""

    def __init__(self, out_h: int = 8, out_w: int = 32, spatial_scale: float = 0.25):
        """
        Args:
            out_h, out_w: rectified crop size.  ``out_w`` bounds the CTC input
                length, so it must exceed the longest expected transcription
                (32 columns comfortably covers ~15 characters).
            spatial_scale: feature stride reciprocal; 0.25 for a stride-4 map.
        """
        super().__init__()
        self.out_h, self.out_w = out_h, out_w
        self.spatial_scale = spatial_scale

    def forward(self, features: torch.Tensor, control_points: torch.Tensor,
                batch_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: ``(B, C, H, W)`` shared feature map.
            control_points: ``(N, 2K, 2)`` in **image** coordinates.
            batch_index: ``(N,)`` long tensor mapping each instance to its image.

        Returns:
            ``(N, C, out_h, out_w)``
        """
        if control_points.numel() == 0:
            return features.new_zeros((0, features.shape[1], self.out_h, self.out_w))
        b, c, h, w = features.shape
        cp = control_points.to(features.dtype) * self.spatial_scale
        grid = build_polyalign_grid(cp, self.out_h, self.out_w, h, w)
        selected = features[batch_index.long()]            # (N, C, H, W)
        return F.grid_sample(selected, grid, mode="bilinear",
                             padding_mode="zeros", align_corners=False)


def quads_to_control_points(quads: torch.Tensor) -> torch.Tensor:
    """``(N, 4, 2)`` TL,TR,BR,BL quads -> ``(N, 4, 2)`` PolyAlign control points.

    Control-point layout is top=[TL,TR], bottom=[BL,BR] (both left-to-right).
    """
    if quads.shape[-2:] != (4, 2):
        raise ValueError(f"expected (N, 4, 2), got {tuple(quads.shape)}")
    tl, tr, br, bl = quads[:, 0], quads[:, 1], quads[:, 2], quads[:, 3]
    return torch.stack([tl, tr, bl, br], dim=1)


def boxes_to_control_points(boxes: torch.Tensor) -> torch.Tensor:
    """``(N, 4)`` xyxy -> control points, for the axis-aligned degenerate case."""
    x1, y1, x2, y2 = boxes.unbind(-1)
    tl = torch.stack([x1, y1], -1); tr = torch.stack([x2, y1], -1)
    bl = torch.stack([x1, y2], -1); br = torch.stack([x2, y2], -1)
    return torch.stack([tl, tr, bl, br], dim=1)
