"""End-to-end: the model must actually learn, and the loop must not fall over.

These are slower than the unit tests (a minute or so on CPU) but they are the
ones that catch the failures that matter -- a loss that is wired to the wrong
tensor, a head that receives no gradient, a schedule that never warms up.
"""
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from vtspot.data.dataset import ClipConfig, SyntheticVideoDataset, collate_clips
from vtspot.data.synth_video import VideoSynthConfig
from vtspot.data.transforms import AugConfig
from vtspot.engine.scheduler import build_scheduler, param_groups, warmup_cosine
from vtspot.engine.trainer import ModelEMA, Trainer, TrainConfig
from vtspot.models.spotter import SpotterConfig, VideoTextSpotter
from vtspot.utils.charset import Charset

NO_AUG = AugConfig(crop_size=(160, 160), short_side=160, max_long_side=256,
                   scale_range=(1.0, 1.0), rotate_deg=0.0, colour_prob=0.0,
                   blur_prob=0.0, jpeg_prob=0.0, frame_jitter_px=0.0)


def _tiny_model(charset):
    return VideoTextSpotter(SpotterConfig(
        backbone_width=16, backbone_layers=(1, 1, 1, 1), fpn_channels=64,
        rec_dim=64, rec_layers=1, max_text_len=10,
        ctc_classes=charset.ctc_num_classes, attn_classes=charset.attn_num_classes))


def _fixed_loader(charset, n_clips=4, repeats=16, batch_size=2):
    ds = SyntheticVideoDataset(
        charset, length=n_clips, clip_cfg=ClipConfig(clip_len=2, max_text_len=10),
        video_cfg=VideoSynthConfig(width=256, height=160, num_frames=4,
                                   min_instances=3, max_instances=5),
        aug_cfg=NO_AUG, seed=11)
    cache = [ds[i] for i in range(n_clips)]

    class Fixed(Dataset):
        def __len__(self):
            return repeats

        def __getitem__(self, i):
            return cache[i % n_clips]

    return DataLoader(Fixed(), batch_size=batch_size, collate_fn=collate_clips)


@pytest.mark.slow
def test_all_three_heads_learn_on_a_fixed_set():
    torch.manual_seed(0)
    np.random.seed(0)
    charset = Charset.from_preset("alnum")
    loader = _fixed_loader(charset)
    trainer = Trainer(_tiny_model(charset),
                      TrainConfig(epochs=12, lr=2e-3, warmup_steps=10, amp=False,
                                  ema_decay=0.0, db_k_warmup_steps=100,
                                  ckpt_dir="/tmp/vtspot_test"),
                      device="cpu")
    trainer.scheduler = build_scheduler("cosine", trainer.optimizer, 10, 12 * len(loader))

    first = last = None
    for _ in range(12):
        totals, n = {}, 0
        for batch in loader:
            for k, v in trainer.train_step(batch).items():
                totals[k] = totals.get(k, 0.0) + v
            n += 1
        epoch = {k: v / n for k, v in totals.items()}
        first = first or epoch
        last = epoch

    assert last["loss_det"] < first["loss_det"] * 0.6, "detection head is not learning"
    assert last["loss_rec"] < first["loss_rec"] * 0.5, "recognition head is not learning"
    assert last["loss_track"] <= first["loss_track"], "tracking head is not learning"
    assert np.isfinite(last["loss_total"])


@pytest.mark.slow
def test_full_pipeline_runs_and_evaluates():
    """Train briefly, then detect -> recognise -> track -> score.  The metrics
    will be poor (a 1M-parameter model trained for seconds), but every stage
    must connect and produce well-formed output."""
    from vtspot.eval.metrics import evaluate
    from vtspot.predictor import PredictConfig, VideoTextPredictor
    from vtspot.data.synth_video import SyntheticVideoGenerator

    torch.manual_seed(0)
    np.random.seed(0)
    charset = Charset.from_preset("alnum")
    model = _tiny_model(charset)
    loader = _fixed_loader(charset, repeats=8)
    trainer = Trainer(model, TrainConfig(epochs=1, lr=2e-3, warmup_steps=5, amp=False,
                                         ema_decay=0.0, ckpt_dir="/tmp/vtspot_test"),
                      device="cpu")
    trainer.scheduler = warmup_cosine(trainer.optimizer, 5, 40)
    for _ in range(6):
        for batch in loader:
            trainer.train_step(batch)

    gen = SyntheticVideoGenerator(
        VideoSynthConfig(width=256, height=160, num_frames=4), seed=99)
    frames, anns = gen.generate()
    predictor = VideoTextPredictor(
        model, charset,
        PredictConfig(short_side=160, max_long_side=256, box_thresh=0.2,
                      bin_thresh=0.2))
    trajectories = predictor.run(frames)

    for t in trajectories:
        assert {"track_id", "text", "frames", "score"} <= set(t)
        assert all(len(p) >= 4 for p in t["frames"].values())

    gt = {}
    for f, insts in enumerate(anns):
        for i in insts:
            gt.setdefault(i["track_id"], {"text": i["text"], "frames": {}})
            gt[i["track_id"]]["frames"][f] = i["polygon"]
    result = evaluate(list(gt.values()), trajectories)
    assert 0.0 <= result.hota <= 1.0
    assert result.mota <= 1.0
    spotting = evaluate(list(gt.values()), trajectories, spotting=True)
    assert spotting.hota <= result.hota + 1e-9, \
        "spotting can never exceed tracking -- it is strictly harder"


def test_checkpoint_roundtrip(tmp_path):
    charset = Charset.from_preset("alnum")
    model = _tiny_model(charset)
    trainer = Trainer(model, TrainConfig(ckpt_dir=str(tmp_path), amp=False),
                      run_config={"charset": "alnum"})
    trainer.step = 42
    trainer.save(tmp_path / "c.pt")

    ckpt = torch.load(tmp_path / "c.pt", map_location="cpu", weights_only=False)
    assert ckpt["config"]["charset"] == "alnum", "checkpoint must record its alphabet"
    assert ckpt["config"]["ctc_classes"] == charset.ctc_num_classes

    fresh = Trainer(_tiny_model(charset), TrainConfig(ckpt_dir=str(tmp_path), amp=False))
    fresh.load(tmp_path / "c.pt")
    assert fresh.step == 42
    for a, b in zip(model.parameters(), fresh.model.parameters()):
        assert torch.allclose(a, b)


def test_non_finite_loss_is_skipped_not_propagated():
    charset = Charset.from_preset("alnum")
    model = _tiny_model(charset)
    trainer = Trainer(model, TrainConfig(amp=False, ckpt_dir="/tmp/vtspot_test"))
    before = [p.detach().clone() for p in model.parameters()]

    trainer.compute_losses = lambda b, o: {"loss_det": torch.tensor(float("nan")),
                                           "loss_rec": torch.tensor(0.0),
                                           "loss_track": torch.tensor(0.0)}
    loader = _fixed_loader(charset, n_clips=1, repeats=1)
    stats = trainer.train_step(next(iter(loader)))
    assert np.isnan(stats["loss_total"])
    for a, b in zip(before, model.parameters()):
        assert torch.allclose(a, b), "a NaN step must not modify the weights"


def test_ema_ramps_in_early():
    charset = Charset.from_preset("alnum")
    model = _tiny_model(charset)
    ema = ModelEMA(model, decay=0.999)
    key = next(iter(ema.module))
    original = ema.module[key].clone()
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    ema.update(model, step=0)
    # At step 0 the stored copy is random noise; a flat 0.999 would keep it.
    assert not torch.allclose(ema.module[key], original)


def test_param_groups_exclude_norms_from_decay():
    charset = Charset.from_preset("alnum")
    groups = param_groups(_tiny_model(charset), weight_decay=0.01)
    assert groups[0]["weight_decay"] == 0.01
    assert groups[1]["weight_decay"] == 0.0
    assert all(p.ndim > 1 for p in groups[0]["params"])


def test_warmup_then_monotone_decay():
    opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1e-3)
    sched = warmup_cosine(opt, warmup_steps=10, total_steps=100)
    lrs = []
    for _ in range(100):
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
    assert lrs[0] < lrs[9]
    assert lrs[9] == pytest.approx(1e-3)
    assert all(lrs[i] >= lrs[i + 1] - 1e-12 for i in range(10, 99))
