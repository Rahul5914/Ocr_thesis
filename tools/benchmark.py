#!/usr/bin/env python3
"""Find out what is limiting training speed: the dataloader or the GPU.

Training throughput has exactly two possible bottlenecks, and the fix for each
is the opposite of the fix for the other:

* **Dataloader-bound** -- the GPU waits for synthetic images.  Raise workers, or
  make synthesis cheaper (smaller crops, fewer words per image).
* **GPU-bound** -- the CPU keeps up but the model is heavy.  Shrink the model or
  the crop; adding workers does nothing.

Guessing wrong wastes hours, so measure each stage in isolation first.

    python tools/benchmark.py --config configs/a4000_stage1.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from vtspot.data.dataset import collate_clips
from vtspot.engine.trainer import Trainer, TrainConfig
from vtspot.models.spotter import build_model
from vtspot.utils.charset import Charset
from train import _sub, build_dataset, enable_tf32  # noqa: E402


def time_it(fn, n, warmup=2):
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    args = ap.parse_args()

    enable_tf32()
    cfg = yaml.safe_load(Path(args.config).read_text())
    charset = Charset.build(cfg.get("charset", "alnum"))
    batch_size = args.batch_size or cfg["train"].get("batch_size", 2)
    workers = args.workers if args.workers is not None else cfg["train"].get("workers", 4)

    if args.device.startswith("cuda") and torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"gpu: {p.name}, {p.total_memory / 1e9:.1f} GB")
    print(f"config: {Path(args.config).name}  batch={batch_size}  workers={workers}")

    dataset = build_dataset(cfg, charset)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=workers, collate_fn=collate_clips,
                        drop_last=True, persistent_workers=workers > 0)
    model = build_model(cfg.get("model", {}), charset.ctc_num_classes,
                        charset.attn_num_classes)
    train_cfg = _sub(TrainConfig, cfg, "train")
    trainer = Trainer(model, train_cfg, device=args.device,
                      charset_blank=charset.blank)

    print("\n--- 1. dataloader alone (no model) ---")
    it = iter(loader)
    first = next(it)
    clip = first["images"].shape[1]
    imgs_per_step = batch_size * clip
    t_data = time_it(lambda: next(it), args.steps, warmup=2)
    print(f"    {1 / t_data:6.2f} steps/s   ({imgs_per_step / t_data:6.1f} images/s)")

    print("\n--- 2. model step alone (batch held in memory, no dataloader) ---")
    fixed = {k: v.to(trainer.device) for k, v in first.items()}
    t_gpu = time_it(lambda: trainer.train_step(fixed), args.steps, warmup=3)
    print(f"    {1 / t_gpu:6.2f} steps/s")

    print("\n--- 3. end to end ---")
    it2 = iter(loader)
    def full():
        trainer.train_step(next(it2))
    t_all = time_it(full, args.steps, warmup=2)
    print(f"    {1 / t_all:6.2f} steps/s")

    total = (cfg["train"].get("steps_per_epoch") or len(dataset) // batch_size) \
        * cfg["train"]["epochs"]
    print(f"\n--- verdict ---")
    print(f"    dataloader ceiling : {1 / t_data:6.2f} steps/s")
    print(f"    gpu ceiling        : {1 / t_gpu:6.2f} steps/s")
    print(f"    achieved           : {1 / t_all:6.2f} steps/s")
    if t_data > t_gpu * 1.3:
        head = 1 / t_gpu / (1 / t_data)
        print(f"\n    DATALOADER-BOUND. The GPU is idle waiting for images; it could run "
              f"{head:.1f}x faster.\n"
              f"    Fix, in order: raise --workers toward your core count; lower "
              f"data.synth.max_words;\n"
              f"    lower crop_size.  Adding GPU power would change nothing.")
    elif t_gpu > t_data * 1.3:
        print(f"\n    GPU-BOUND. The dataloader keeps up. More workers will not help.\n"
              f"    Fix, in order: lower aug.crop_size; lower model.backbone_width;\n"
              f"    lower model.fpn_channels.")
    else:
        print("\n    BALANCED -- both sides are near their ceiling.")
    print(f"\n    at the achieved rate, {total:,} steps = {total * t_all / 3600:.1f} h "
          f"({total * t_all / 86400:.1f} days)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
