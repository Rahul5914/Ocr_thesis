#!/usr/bin/env python3
"""Run the spotter on a video file and write trajectories (+ optional overlay).

    python tools/predict_video.py --checkpoint checkpoints/stage3/last.pt \
        --video clip.mp4 --out results.json --render out.mp4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vtspot.models.spotter import build_model
from vtspot.predictor import PredictConfig, VideoTextPredictor, draw_trajectories
from vtspot.tracking.tracker import TrackerConfig
from vtspot.utils.charset import Charset


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", default="trajectories.json")
    ap.add_argument("--render", default=None, help="write an annotated mp4 here")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--short-side", type=int, default=736)
    ap.add_argument("--box-thresh", type=float, default=0.45)
    ap.add_argument("--max-frames", type=int, default=None)
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    cfg = ckpt.get("config", {})
    charset = Charset.build(cfg.get("charset", "alnum")
                            if isinstance(cfg.get("charset", "alnum"), str) else "alnum")
    model = build_model(cfg, charset.ctc_num_classes, charset.attn_num_classes)
    model.load_state_dict(ckpt.get("ema") or ckpt["model"], strict=False)

    predictor = VideoTextPredictor(
        model, charset,
        PredictConfig(short_side=args.short_side, box_thresh=args.box_thresh,
                      tracker=TrackerConfig()),
        device=args.device)

    trajectories, n_frames = predictor.run_video_file(
        args.video, max_frames=args.max_frames, progress=True)
    Path(args.out).write_text(json.dumps(trajectories, indent=2))
    print(f"{len(trajectories)} trajectories over {n_frames} frames -> {args.out}")
    for t in sorted(trajectories, key=lambda d: -len(d["frames"]))[:15]:
        print(f"  id={t['track_id']:<4} {t['text']!r:<20} "
              f"conf={t['text_confidence']:.2f} frames={len(t['frames'])} "
              f"({t['start_frame']}-{t['end_frame']})")

    if args.render:
        cap = cv2.VideoCapture(args.video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.render, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok or (args.max_frames is not None and i >= args.max_frames):
                break
            writer.write(draw_trajectories(frame, trajectories, i))
            i += 1
        cap.release(); writer.release()
        print(f"rendered {i} frames -> {args.render}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
