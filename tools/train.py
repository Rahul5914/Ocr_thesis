#!/usr/bin/env python3
"""Train the video text spotter.

Curriculum stages (see docs/TRAINING_GUIDE.md):

    stage1_static  synthetic stills          -- learn strokes, characters, detection
    stage2_video   synthetic video clips     -- learn association and blur robustness
    stage3_finetune real annotated video     -- close the domain gap

Each stage resumes from the previous stage's checkpoint via ``--init``.

    python tools/train.py --config configs/stage1_static.yaml
    python tools/train.py --config configs/stage2_video.yaml --init checkpoints/stage1/last.pt
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vtspot.data.dataset import (ClipConfig, SyntheticStaticDataset,
                                 SyntheticVideoDataset, VideoClipDataset, collate_clips)
from vtspot.data.synth_static import SynthConfig
from vtspot.data.synth_video import VideoSynthConfig
from vtspot.data.transforms import AugConfig
from vtspot.engine.trainer import Trainer, TrainConfig
from vtspot.models.spotter import build_model
from vtspot.utils.charset import Charset


def enable_tf32() -> None:
    """Turn on TF32 matmul/conv paths (Ampere and later).

    Roughly 1.5-2x on convolutions for a precision loss that is immaterial to
    training.  Off by default in PyTorch for numerical-reproducibility reasons;
    for training a detector it is simply free speed.
    """
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _sub(cls, cfg: dict, key: str):
    """Build a dataclass from the sub-dict at ``cfg[key]``, ignoring unknown keys."""
    fields = {f.name for f in cls.__dataclass_fields__.values()}
    raw = cfg.get(key, {}) or {}
    unknown = set(raw) - fields
    if unknown:
        print(f"[warn] ignoring unknown {key} keys: {sorted(unknown)}")
    return cls(**{k: v for k, v in raw.items() if k in fields})


def build_dataset(cfg: dict, charset: Charset):
    data = cfg["data"]
    kind = data["type"]
    clip_cfg = _sub(ClipConfig, data, "clip")
    aug_cfg = _sub(AugConfig, data, "aug")
    common = dict(charset=charset, clip_cfg=clip_cfg, aug_cfg=aug_cfg,
                  seed=cfg.get("seed", 0))
    if kind == "synth_static":
        return SyntheticStaticDataset(
            length=data.get("length", 100_000), synth_cfg=_sub(SynthConfig, data, "synth"),
            background_dir=data.get("background_dir"), **common)
    if kind == "synth_video":
        return SyntheticVideoDataset(
            length=data.get("length", 20_000),
            video_cfg=_sub(VideoSynthConfig, data, "synth"),
            background_dir=data.get("background_dir"), **common)
    if kind == "real_video":
        return VideoClipDataset(root=data["root"], training=True, **common)
    raise ValueError(f"unknown data.type {kind!r}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--init", default=None, help="checkpoint to initialise from")
    ap.add_argument("--resume", default=None, help="checkpoint to resume training from")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--steps-per-epoch", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--ckpt-dir", default=None)
    ap.add_argument("--no-tf32", action="store_true",
                    help="disable TF32 matmul/conv (Ampere+); slower but bit-exact")
    args = ap.parse_args()

    if not args.no_tf32:
        enable_tf32()

    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg.get("seed", 0))

    charset = Charset.build(cfg.get("charset", "alnum"),
                            case_sensitive=cfg.get("case_sensitive", False))
    print(f"charset: {charset.num_chars} chars -> {charset.ctc_num_classes} CTC classes")
    if torch.cuda.is_available() and args.device.startswith("cuda"):
        props = torch.cuda.get_device_properties(0)
        print(f"gpu: {props.name}, {props.total_memory / 1e9:.1f} GB, "
              f"bf16={'yes' if torch.cuda.is_bf16_supported() else 'no'}")

    dataset = build_dataset(cfg, charset)
    batch_size = args.batch_size or cfg["train"].get("batch_size", 2)
    workers = args.workers if args.workers is not None else cfg["train"].get("workers", 4)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=workers, collate_fn=collate_clips,
                        pin_memory=(args.device == "cuda"), drop_last=True,
                        persistent_workers=workers > 0)

    model = build_model(cfg.get("model", {}), charset.ctc_num_classes,
                        charset.attn_num_classes)
    print("params (M):", json.dumps({k: round(v, 2) for k, v in
                                     model.num_parameters().items()}))

    train_cfg = _sub(TrainConfig, cfg, "train")
    if args.epochs is not None:
        train_cfg.epochs = args.epochs
    if args.steps_per_epoch is not None:
        train_cfg.steps_per_epoch = args.steps_per_epoch
    if args.ckpt_dir is not None:
        train_cfg.ckpt_dir = args.ckpt_dir

    trainer = Trainer(model, train_cfg, device=args.device,
                      charset_blank=charset.blank,
                      run_config={"charset": cfg.get("charset", "alnum"),
                                  "case_sensitive": cfg.get("case_sensitive", False)})

    if args.init:
        ckpt = torch.load(args.init, map_location=args.device, weights_only=False)
        state = ckpt.get("ema") or ckpt["model"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"initialised from {args.init} "
              f"({len(missing)} missing, {len(unexpected)} unexpected tensors)")
    if args.resume:
        trainer.load(args.resume)
        print(f"resumed from {args.resume} at epoch {trainer.epoch}, step {trainer.step}")

    Path(train_cfg.ckpt_dir).mkdir(parents=True, exist_ok=True)
    (Path(train_cfg.ckpt_dir) / "config.yaml").write_text(yaml.safe_dump(cfg))
    summary = trainer.train(loader)
    print("final:", json.dumps({k: round(v, 4) for k, v in summary.items()
                                if isinstance(v, float)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
