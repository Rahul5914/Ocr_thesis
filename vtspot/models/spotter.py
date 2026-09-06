"""VideoTextSpotter: backbone -> FPN -> {detection, recognition, tracking}.

One shared feature map feeds all three heads, which is what makes this a
*spotter* rather than three models in a trench coat: the recognition and
association gradients flow back into the same features the detector uses, and
each task regularises the others.

Training uses ground-truth polygons to crop instance features, not predicted
ones.  Early in a from-scratch run the detector's polygons are noise, and
feeding noise to the recogniser teaches it to read noise.  The recognition and
tracking heads therefore see clean crops throughout; the detector is what closes
the gap at inference, and ``roi_jitter`` injects controlled polygon noise during
training so the downstream heads are not brittle to a detector that is merely
good rather than perfect.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from .backbone import FPN, TextBackbone
from .det_head import DBHead
from .init import init_weights
from .rec_head import RecognitionHead
from .roi import PolyAlign
from .track_head import TrackEmbedHead


@dataclass
class SpotterConfig:
    backbone_width: int = 48
    backbone_layers: Sequence[int] = (2, 2, 2, 2)
    norm: str = "gn"                     # "gn" | "bn" | "none" (Fixup)
    fpn_channels: int = 256
    db_k: float = 50.0
    db_k_start: float = 2.0
    prior_prob: float = 0.02
    roi_h: int = 8
    roi_w: int = 32
    rec_dim: int = 256
    rec_layers: int = 2
    rec_heads: int = 4
    rec_dropout: float = 0.1
    max_text_len: int = 25
    embed_dim: int = 128
    use_attention_branch: bool = True    # GTC guidance, training only
    roi_jitter: float = 0.05             # fraction of instance size
    ctc_classes: int = 38
    attn_classes: int = 38


class VideoTextSpotter(nn.Module):
    def __init__(self, cfg: Optional[SpotterConfig] = None):
        super().__init__()
        self.cfg = cfg or SpotterConfig()
        c = self.cfg
        self.backbone = TextBackbone(width=c.backbone_width, layers=c.backbone_layers,
                                     norm=c.norm)
        self.neck = FPN(self.backbone.out_channels, c.fpn_channels, norm=c.norm)
        self.det_head = DBHead(c.fpn_channels, k=c.db_k, norm=c.norm,
                               prior_prob=c.prior_prob, k_start=c.db_k_start)
        self.poly_align = PolyAlign(out_h=c.roi_h, out_w=c.roi_w, spatial_scale=0.25)
        self.rec_head = RecognitionHead(
            c.fpn_channels, ctc_classes=c.ctc_classes,
            attn_classes=c.attn_classes if c.use_attention_branch else 0,
            dim=c.rec_dim, in_h=c.roi_h, layers=c.rec_layers, heads=c.rec_heads,
            dropout=c.rec_dropout, max_seq_len=max(c.roi_w, 32),
            max_text_len=c.max_text_len)
        self.track_head = TrackEmbedHead(c.fpn_channels, embed_dim=c.embed_dim)
        init_weights(self)

    # -- feature extraction ---------------------------------------------
    def extract(self, images: torch.Tensor) -> torch.Tensor:
        """``(N, 3, H, W)`` -> fused stride-4 feature map."""
        return self.neck(self.backbone(images))

    def detect_maps(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.det_head(features)

    def instance_outputs(self, features: torch.Tensor, control_points: torch.Tensor,
                         frame_index: torch.Tensor,
                         attn_targets: Optional[torch.Tensor] = None
                         ) -> Dict[str, torch.Tensor]:
        """Recognition + embedding for a set of instances."""
        roi = self.poly_align(features, control_points, frame_index)
        rec = self.rec_head(roi, attn_targets)
        embed = self.track_head(roi)
        out = {"roi": roi, "embed": embed, **rec}
        return out

    # -- training path ---------------------------------------------------
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            batch: as produced by ``collate_clips``.  ``images`` is
                ``(B, T, 3, H, W)``; instance tensors are flat with
                ``inst_frame`` indexing the flattened ``B*T`` frame axis.
        """
        images = batch["images"]
        if images.dim() != 5:
            raise ValueError(f"expected (B, T, 3, H, W), got {tuple(images.shape)}")
        b, t = images.shape[:2]
        flat = images.flatten(0, 1)
        features = self.extract(flat)
        det = self.detect_maps(features)

        ctrl = batch["inst_ctrl"].to(features.dtype)
        if self.training and self.cfg.roi_jitter > 0 and ctrl.numel():
            ctrl = jitter_control_points(ctrl, self.cfg.roi_jitter)

        attn_targets = batch.get("inst_attn") if self.training else None
        inst = self.instance_outputs(features, ctrl, batch["inst_frame"], attn_targets)

        out = {f"det_{k}": v for k, v in det.items()}
        out.update(inst)
        out["features"] = features
        out["num_frames"] = torch.tensor(b * t)
        return out

    # -- inference path --------------------------------------------------
    @torch.no_grad()
    def infer_frame(self, image: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Single frame ``(1, 3, H, W)`` -> detection maps and features."""
        features = self.extract(image)
        return {"features": features, **self.det_head(features)}

    @torch.no_grad()
    def read_instances(self, features: torch.Tensor, control_points: torch.Tensor,
                       frame_index: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.instance_outputs(features, control_points, frame_index, None)

    def num_parameters(self) -> Dict[str, float]:
        def count(m: nn.Module) -> float:
            return sum(p.numel() for p in m.parameters()) / 1e6
        return {"backbone": count(self.backbone), "neck": count(self.neck),
                "det_head": count(self.det_head), "rec_head": count(self.rec_head),
                "track_head": count(self.track_head), "total": count(self)}


def jitter_control_points(ctrl: torch.Tensor, ratio: float) -> torch.Tensor:
    """Perturb control points by a fraction of each instance's size.

    Bridges the train/inference mismatch created by cropping from ground truth:
    at inference the crops come from predicted polygons, which are never exact.
    """
    if ratio <= 0 or ctrl.numel() == 0:
        return ctrl
    mins = ctrl.amin(dim=1, keepdim=True)
    maxs = ctrl.amax(dim=1, keepdim=True)
    size = (maxs - mins).clamp(min=1.0)
    noise = (torch.rand_like(ctrl) * 2.0 - 1.0) * ratio * size
    return ctrl + noise


def build_model(cfg: dict, ctc_classes: int, attn_classes: int) -> VideoTextSpotter:
    fields = {f.name for f in SpotterConfig.__dataclass_fields__.values()}
    kwargs = {k: v for k, v in cfg.items() if k in fields}
    kwargs["ctc_classes"] = ctc_classes
    kwargs["attn_classes"] = attn_classes
    return VideoTextSpotter(SpotterConfig(**kwargs))
