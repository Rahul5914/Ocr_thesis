"""Model wiring: shapes, gradient flow, and PolyAlign correctness."""
import numpy as np
import pytest
import torch

from vtspot.models.roi import (PolyAlign, boxes_to_control_points,
                               quads_to_control_points)
from vtspot.models.spotter import SpotterConfig, VideoTextSpotter, jitter_control_points
from vtspot.utils.charset import Charset


@pytest.fixture(scope="module")
def model():
    cs = Charset.from_preset("alnum")
    return VideoTextSpotter(SpotterConfig(
        backbone_width=16, backbone_layers=(1, 1, 1, 1), fpn_channels=64,
        rec_dim=64, rec_layers=1, max_text_len=10,
        ctc_classes=cs.ctc_num_classes, attn_classes=cs.attn_num_classes))


def _batch(b=2, t=2, h=128, w=128, n=5):
    return {"images": torch.randn(b, t, 3, h, w),
            "inst_ctrl": torch.rand(n, 16, 2) * 100 + 10,
            "inst_frame": torch.randint(0, b * t, (n,)),
            "inst_attn": torch.randint(0, 38, (n, 10))}


def test_forward_shapes(model):
    model.train()
    out = model(_batch())
    assert out["det_prob"].shape == (4, 1, 128, 128)
    assert out["ctc_logits"].shape[0] == 5
    assert out["embed"].shape == (5, 128)
    assert "attn_logits" in out


def test_eval_drops_attention_branch(model):
    model.eval()
    out = model(_batch())
    assert "attn_logits" not in out, "attention decoder must not run at inference"
    assert "det_binary" not in out


def test_embeddings_are_unit_norm(model):
    model.eval()
    e = model(_batch())["embed"]
    assert torch.allclose(e.norm(dim=-1), torch.ones(len(e)), atol=1e-4)


def test_gradients_reach_backbone(model):
    model.train()
    model.zero_grad()
    out = model(_batch())
    (out["ctc_logits"].sum() + out["det_prob"].sum() + out["embed"].sum()).backward()
    stem = model.backbone.stem[0][0].weight
    assert stem.grad is not None and float(stem.grad.abs().sum()) > 0


def test_zero_instances_is_safe(model):
    model.train()
    b = _batch(n=0)
    b["inst_ctrl"] = torch.zeros(0, 16, 2)
    b["inst_frame"] = torch.zeros(0, dtype=torch.long)
    b["inst_attn"] = torch.zeros(0, 10, dtype=torch.long)
    out = model(b)
    assert out["embed"].shape[0] == 0
    assert out["det_prob"].shape[0] == 4


def test_rejects_non_clip_input(model):
    with pytest.raises(ValueError):
        model({"images": torch.randn(2, 3, 64, 64), "inst_ctrl": torch.zeros(0, 16, 2),
               "inst_frame": torch.zeros(0, dtype=torch.long)})


def test_polyalign_axis_aligned_crop_is_exact():
    feat = torch.zeros(1, 1, 64, 64)
    feat[0, 0, 10:30, 8:40] = 1.0
    pa = PolyAlign(out_h=8, out_w=32, spatial_scale=1.0)
    out = pa(feat, boxes_to_control_points(torch.tensor([[8., 10., 40., 30.]])),
             torch.zeros(1, dtype=torch.long))
    assert float(out.mean()) > 0.8


def test_polyalign_beats_axis_aligned_box_on_curved_text():
    feat = torch.zeros(1, 1, 64, 80)
    for x in range(4, 72):
        y = 20 + 8 * np.sin(x / 72 * np.pi)
        feat[0, 0, int(y):int(y) + 10, x] = 1.0
    pa = PolyAlign(out_h=8, out_w=32, spatial_scale=1.0)
    xs = torch.linspace(4, 71, 7)
    ys = 20 + 8 * torch.sin(xs / 72 * np.pi)
    curved = torch.cat([torch.stack([xs, ys], 1),
                        torch.stack([xs, ys + 10], 1)], 0).unsqueeze(0)
    box = torch.tensor([[4., 20., 71., 38.]])
    idx = torch.zeros(1, dtype=torch.long)
    curved_signal = float(pa(feat, curved, idx).mean())
    box_signal = float(pa(feat, boxes_to_control_points(box), idx).mean())
    assert curved_signal > box_signal * 1.5


def test_polyalign_is_differentiable_wrt_control_points():
    feat = torch.rand(1, 4, 32, 32)
    ctrl = (torch.rand(2, 8, 2) * 20 + 5).requires_grad_(True)
    PolyAlign(spatial_scale=1.0)(feat, ctrl, torch.zeros(2, dtype=torch.long)).sum().backward()
    assert ctrl.grad is not None and float(ctrl.grad.abs().sum()) > 0


def test_polyalign_rejects_bad_shapes():
    pa = PolyAlign(spatial_scale=1.0)
    with pytest.raises(ValueError):
        pa(torch.rand(1, 2, 16, 16), torch.rand(1, 3, 2), torch.zeros(1, dtype=torch.long))


def test_quad_conversion_ordering():
    quads = torch.tensor([[[0., 0.], [10., 0.], [10., 5.], [0., 5.]]])
    ctrl = quads_to_control_points(quads)
    assert torch.allclose(ctrl[0, 0], quads[0, 0])   # top-left first
    assert torch.allclose(ctrl[0, 1], quads[0, 1])   # then top-right
    assert torch.allclose(ctrl[0, 2], quads[0, 3])   # then bottom-left
    assert torch.allclose(ctrl[0, 3], quads[0, 2])


def test_jitter_scales_with_instance_size():
    small = torch.tensor([[[0., 0.], [10., 0.], [0., 4.], [10., 4.]]])
    big = small * 20
    torch.manual_seed(0)
    ds = (jitter_control_points(small, 0.1) - small).abs().max()
    torch.manual_seed(0)
    db = (jitter_control_points(big, 0.1) - big).abs().max()
    assert db > ds * 5
