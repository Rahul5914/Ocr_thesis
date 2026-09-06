#!/usr/bin/env python3
"""Evaluate a checkpoint on a prepared video dataset.

Reports both tracking-mode and spotting-mode metrics.  The spotting numbers are
the ones comparable to published video text *spotting* results; the tracking
numbers are comparable to video text *tracking* results.  Quoting one as the
other is the most common way these benchmarks get misreported.

    python tools/evaluate.py --checkpoint checkpoints/stage3/last.pt \
        --data data/prepared/icdar15_video_test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vtspot.data.schema import VideoAnnotation, index_dataset
from vtspot.eval.metrics import evaluate_dataset
from vtspot.models.spotter import build_model
from vtspot.predictor import PredictConfig, VideoTextPredictor
from vtspot.tracking.tracker import TrackerConfig
from vtspot.utils.charset import Charset


def gt_trajectories(ann: VideoAnnotation, include_ignore: bool = False) -> list[dict]:
    tracks: dict[int, dict] = {}
    for frame in ann.frames:
        for inst in frame.instances:
            if inst.ignore and not include_ignore:
                continue
            if inst.track_id < 0:
                continue
            t = tracks.setdefault(inst.track_id, {"text": inst.text, "frames": {}})
            t["frames"][frame.frame_idx] = inst.polygon
            if inst.text:
                t["text"] = inst.text
    return list(tracks.values())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True, help="prepared dataset root")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--short-side", type=int, default=736)
    ap.add_argument("--box-thresh", type=float, default=0.45)
    ap.add_argument("--max-videos", type=int, default=None)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--use-ema", action="store_true", default=True)
    ap.add_argument("--out", default=None, help="write per-video results here")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    cfg = ckpt.get("config", {})
    charset_spec = cfg.get("charset", "alnum")
    charset = Charset.build(charset_spec if isinstance(charset_spec, str) else "alnum")
    model = build_model(cfg, charset.ctc_num_classes, charset.attn_num_classes)
    state = ckpt.get("ema") if (args.use_ema and "ema" in ckpt) else ckpt["model"]
    model.load_state_dict(state, strict=False)

    predictor = VideoTextPredictor(
        model, charset,
        PredictConfig(short_side=args.short_side, box_thresh=args.box_thresh,
                      tracker=TrackerConfig()),
        device=args.device)

    root = Path(args.data)
    paths = index_dataset(root)[: args.max_videos]
    pairs, per_video = [], []
    for i, path in enumerate(paths):
        ann = VideoAnnotation.from_json(path)
        frames = []
        for frame in ann.frames[: args.max_frames]:
            img = cv2.imread(str(root / "frames" / ann.video_id /
                                 f"{frame.frame_idx:06d}.jpg"))
            if img is not None:
                frames.append(img)
        if not frames:
            print(f"[skip] {ann.video_id}: no frames on disk")
            continue
        pred = predictor.run(frames)
        gt = gt_trajectories(ann)
        pairs.append((gt, pred))
        per_video.append({"video_id": ann.video_id, "gt_tracks": len(gt),
                          "pred_tracks": len(pred)})
        print(f"[{i + 1}/{len(paths)}] {ann.video_id}: "
              f"{len(gt)} gt tracks, {len(pred)} predicted")

    if not pairs:
        print("no videos evaluated")
        return 1

    tracking = evaluate_dataset(pairs, spotting=False, iou_threshold=args.iou)
    spotting = evaluate_dataset(pairs, spotting=True, iou_threshold=args.iou,
                                alphabet="".join(charset.chars))
    print("\n=== TRACKING (detection + association only) ===")
    print(json.dumps(tracking.as_dict(), indent=2))
    print("\n=== SPOTTING (transcription must also be correct) ===")
    print(json.dumps(spotting.as_dict(), indent=2))

    if args.out:
        Path(args.out).write_text(json.dumps(
            {"tracking": tracking.as_dict(), "spotting": spotting.as_dict(),
             "per_video": per_video}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
