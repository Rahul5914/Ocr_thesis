"""Initialisation: the from-scratch claims, checked empirically rather than asserted."""
import torch
import torch.nn as nn

from vtspot.models.backbone import TextBackbone
from vtspot.models.det_head import DBHead
from vtspot.models.init import (activation_variance_report, apply_fixup_scaling,
                                init_prior_bias, init_weights)


def _deep_relu_stack(depth=20, ch=32):
    layers = []
    for _ in range(depth):
        layers += [nn.Conv2d(ch, ch, 3, padding=1), nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


def test_he_preserves_variance_where_xavier_collapses():
    """The claim is comparative, so assert it comparatively.

    The absolute He ratio is seed-dependent (0.03-0.50 over 12 seeds); what is
    stable across every seed is that Xavier collapses by ~six orders of
    magnitude more than He does.  Pinning the seed keeps this deterministic.
    """
    ratios = {}
    for name in ("he", "xavier"):
        torch.manual_seed(0)
        x = torch.randn(2, 32, 16, 16)
        net = _deep_relu_stack()
        if name == "he":
            init_weights(net)
        else:
            for m in net.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.xavier_normal_(m.weight)
                    nn.init.zeros_(m.bias)
        stats = activation_variance_report(net, x)
        ratios[name] = stats[-1][2] / stats[0][2]

    assert ratios["xavier"] < 1e-4, "Xavier should collapse on a deep ReLU stack"
    assert ratios["he"] > 1e-3, "He should keep a usable signal"
    assert ratios["he"] > ratios["xavier"] * 1e4


def test_kaiming_std_matches_formula():
    conv = nn.Conv2d(64, 128, 3)
    init_weights(conv)
    fan_in = 64 * 3 * 3
    expected = (2.0 / fan_in) ** 0.5
    assert abs(float(conv.weight.std()) - expected) / expected < 0.15


def test_fixup_zeroes_last_conv_of_every_branch():
    bb = TextBackbone(width=24, norm="none")
    init_weights(bb)                       # must not undo the Fixup init
    last_convs = [m.weight for n, m in bb.named_modules() if n.endswith("conv2")]
    assert last_convs, "no residual branches found"
    for w in last_convs:
        assert float(w.abs().sum()) == 0.0


def test_fixup_scaling_factor():
    class Branch(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Conv2d(16, 16, 3)
            self.b = nn.Conv2d(16, 16, 3)
            self.conv_layers = (self.a, self.b)

    branches = [Branch() for _ in range(9)]
    apply_fixup_scaling(branches, num_branches=9, layers_per_branch=2)
    expected_scale = 9.0 ** (-1.0 / 2.0)
    fan_in = 16 * 3 * 3
    expected_std = (2.0 / fan_in) ** 0.5 * expected_scale
    got = float(torch.stack([b.a.weight.flatten() for b in branches]).std())
    assert abs(got - expected_std) / expected_std < 0.15
    assert all(float(b.b.weight.abs().sum()) == 0.0 for b in branches)


def test_fixup_keeps_activations_bounded():
    bb = TextBackbone(width=24, norm="none")
    init_weights(bb)
    stats = activation_variance_report(bb, torch.randn(2, 3, 128, 128))
    assert max(s[2] for s in stats) < 100.0


def test_prior_bias_produces_requested_probability():
    head = DBHead(64, prior_prob=0.02)
    init_weights(head)
    head.train()
    out = head(torch.randn(2, 64, 16, 16))
    assert 0.005 < float(out["prob"].mean()) < 0.06
    assert 0.4 < float(out["thresh"].mean()) < 0.6


def test_prior_bias_needs_small_weights():
    """Bias alone is insufficient -- this is the bug the weight shrink fixes."""
    layer = nn.Conv2d(64, 1, 3, padding=1)
    nn.init.kaiming_normal_(layer.weight, mode="fan_in", nonlinearity="relu")
    init_prior_bias(layer, 0.02, weight_std=1.0)     # bias set, weights left large
    loose = float(torch.sigmoid(layer(torch.randn(4, 64, 16, 16))).mean())
    init_prior_bias(layer, 0.02, weight_std=0.01)    # the shipped setting
    tight = float(torch.sigmoid(layer(torch.randn(4, 64, 16, 16))).mean())
    assert abs(tight - 0.02) < abs(loose - 0.02)
    assert abs(tight - 0.02) < 0.02
