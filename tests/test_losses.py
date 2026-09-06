"""Losses: correctness on constructed cases and the guards around them."""
import math
import pytest
import torch
import torch.nn.functional as F

from vtspot.losses.db_loss import BalancedBCELoss, DBLoss, DiceLoss
from vtspot.losses.multitask import FixedWeighting, UncertaintyWeighting
from vtspot.losses.rec_loss import AttentionGuidanceLoss, CTCRecognitionLoss
from vtspot.losses.track_loss import ContrastiveTrackLoss


def test_balanced_bce_limits_negative_contribution():
    logits = torch.zeros(1, 1, 32, 32)
    target = torch.zeros(1, 1, 32, 32)
    target[0, 0, :2, :2] = 1.0                       # 4 positives out of 1024
    mask = torch.ones_like(target)
    loss = BalancedBCELoss(ratio=3.0)(logits, target, mask)
    # With 4 pos + 12 mined negatives all at logit 0, the mean must be log(2).
    assert float(loss) == pytest.approx(math.log(2), abs=1e-4)


def test_dice_is_zero_on_perfect_prediction():
    target = (torch.rand(1, 1, 16, 16) > 0.5).float()
    assert float(DiceLoss()(target, target, torch.ones_like(target))) < 1e-4


def test_db_loss_components_present():
    pred = {"prob_logit": torch.randn(2, 1, 32, 32), "thresh": torch.rand(2, 1, 32, 32),
            "binary": torch.rand(2, 1, 32, 32)}
    target = {"shrink_map": (torch.rand(2, 1, 32, 32) > 0.8).float(),
              "shrink_mask": torch.ones(2, 1, 32, 32),
              "thresh_map": torch.rand(2, 1, 32, 32),
              "thresh_mask": (torch.rand(2, 1, 32, 32) > 0.5).float()}
    out = DBLoss()(pred, target)
    assert {"loss_prob", "loss_thresh", "loss_binary", "loss_det"} <= set(out)
    assert float(out["loss_det"]) > 0


def test_ignore_mask_removes_gradient():
    logits = torch.zeros(1, 1, 8, 8, requires_grad=True)
    target = torch.ones(1, 1, 8, 8)
    mask = torch.zeros(1, 1, 8, 8)
    loss = BalancedBCELoss()(logits, target, mask)
    loss.backward()
    assert float(logits.grad.abs().sum()) == 0.0


def test_ctc_drops_labels_too_long_for_the_crop():
    """A 26-char label with repeats cannot fit 32 frames; it must be skipped,
    not returned as inf."""
    logits = torch.randn(2, 32, 38)
    targets = torch.zeros(2, 26, dtype=torch.long)
    targets[0, :3] = torch.tensor([1, 2, 3])
    targets[1, :] = 1                                # 26 identical -> needs 51 frames
    lengths = torch.tensor([3, 26])
    loss = CTCRecognitionLoss()(logits, targets, lengths)
    assert torch.isfinite(loss)
    only_valid = CTCRecognitionLoss()(logits[:1], targets[:1], lengths[:1])
    assert float(loss) == pytest.approx(float(only_valid), abs=1e-4)


def test_ctc_zero_when_nothing_to_supervise():
    logits = torch.randn(3, 32, 38)
    loss = CTCRecognitionLoss()(logits, torch.zeros(3, 10, dtype=torch.long),
                                torch.zeros(3, dtype=torch.long))
    assert float(loss) == 0.0


def test_ctc_decreases_when_logits_match_target():
    torch.manual_seed(0)
    targets = torch.tensor([[1, 2, 3] + [0] * 7])
    lengths = torch.tensor([3])
    random_logits = torch.randn(1, 32, 38)
    aligned = torch.full((1, 32, 38), -10.0)
    aligned[0, :, 0] = 5.0
    for pos, cls in zip((5, 15, 25), (1, 2, 3)):
        aligned[0, pos, cls] = 10.0
        aligned[0, pos, 0] = -10.0
    loss = CTCRecognitionLoss()
    assert float(loss(aligned, targets, lengths)) < float(loss(random_logits, targets, lengths))


def test_attention_loss_ignores_padding():
    logits = torch.randn(2, 8, 38)
    pad_only = torch.zeros(2, 8, dtype=torch.long)
    assert float(AttentionGuidanceLoss()(logits, pad_only)) == 0.0 or \
        math.isnan(float(AttentionGuidanceLoss()(logits, pad_only)))


def test_contrastive_prefers_consistent_embeddings():
    tid = torch.tensor([0, 1, 2] * 3)
    fid = torch.tensor([0] * 3 + [1] * 3 + [2] * 3)
    perfect = F.normalize(torch.eye(3)[tid].float() * 10, dim=-1)
    torch.manual_seed(0)
    noise = F.normalize(torch.randn(9, 8), dim=-1)
    loss = ContrastiveTrackLoss()
    assert float(loss(perfect, tid, fid)) < float(loss(noise, tid, fid))


def test_contrastive_zero_without_temporal_positives():
    torch.manual_seed(0)
    e = F.normalize(torch.randn(6, 8), dim=-1)
    same_frame = torch.zeros(6, dtype=torch.long)
    assert float(ContrastiveTrackLoss()(e, torch.arange(6), same_frame)) == 0.0


def test_uncertainty_weight_is_clamped():
    uw = UncertaintyWeighting(["loss_track"], s_min=-3.0, s_max=3.0)
    with torch.no_grad():
        uw.log_var["loss_track"].fill_(100.0)
    _, stats = uw({"loss_track": torch.tensor(1.0)})
    assert stats["w_loss_track"] == pytest.approx(math.exp(-3.0), abs=1e-4)


def test_weighting_rejects_unknown_tasks():
    with pytest.raises(ValueError):
        FixedWeighting({"loss_det": 1.0})({"loss_other": torch.tensor(1.0)})
