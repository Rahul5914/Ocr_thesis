#!/usr/bin/env python3
"""Read the text in a road-side video, frame by frame.

Two explicit stages per frame:

    stage 1  DETECT     where is the text?      -> polygons
    stage 2  RECOGNISE  what does it say?       -> transcription + confidence

then an optional third pass links detections across frames so one signboard is
reported once, with the reading its frames agreed on, instead of once per frame.

    python detect_text_video.py --video road.mp4 --render out.mp4 --out results.json

Speed: recognition dominates, and it runs once per detected box.  ``--every 3``
processes one frame in three (the output video still has every frame -- skipped
frames reuse the previous result), which is usually invisible at 25-30 fps and
roughly triples throughput.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from engines import PRESETS, DetectParams, build_engine
from tracker import TextTracker
from viz import annotate, draw_hud


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Detect and read text in a video, frame by frame.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("--video", required=True, help="input video file")
    p.add_argument("--out", default=None, help="write results here (JSON)")
    p.add_argument("--render", default=None, help="write an annotated video here")
    p.add_argument("--frames-dir", default=None,
                   help="also save annotated frames as JPGs in this directory")

    p.add_argument("--engine", default="easyocr", choices=["easyocr", "paddleocr"])
    p.add_argument("--langs", default="en",
                   help="comma-separated language codes, e.g. en or en,hi")
    p.add_argument("--cpu", action="store_true", help="force CPU even if a GPU is present")
    p.add_argument("--mode", default="pipeline", choices=["pipeline", "two_stage"],
                   help="'pipeline' is the engine's own end-to-end call and reads "
                        "better; 'two_stage' crops each detection and reads it "
                        "alone, which is worse but shows which stage failed")

    p.add_argument("--preset", default="default", choices=sorted(PRESETS),
                   help="detector tuning; 'small-text' for distant road signage")
    p.add_argument("--mag", type=float, default=None,
                   help="upscale by this before detection (the knob for small text)")
    p.add_argument("--low-text", type=float, default=None,
                   help="lower = fainter strokes count as text")
    p.add_argument("--text-threshold", type=float, default=None,
                   help="lower = weaker regions count as text")
    p.add_argument("--link", type=float, default=None,
                   help="lower = neighbouring words merge into one box")
    p.add_argument("--width-ths", type=float, default=None,
                   help="higher = merge boxes that are further apart")
    p.add_argument("--min-size", type=int, default=None,
                   help="ignore detections smaller than this many pixels")
    p.add_argument("--decoder", default=None, choices=["greedy", "beamsearch"],
                   help="beamsearch is slower and a little more accurate")
    p.add_argument("--merge-words", dest="merge_words", action="store_true",
                   default=None,
                   help="glue phrases the detector split ('SPEED'+'40' -> "
                        "'SPEED 40'); costs per-box confidence, so --min-conf "
                        "stops filtering and agreement counts frames instead")

    p.add_argument("--every", type=int, default=1,
                   help="run OCR on every Nth frame (1 = every frame)")
    p.add_argument("--max-frames", type=int, default=0,
                   help="stop after this many frames (0 = whole video)")
    p.add_argument("--resize", type=int, default=0,
                   help="resize the long side to this before OCR (0 = native). "
                        "Downscaling is a speed knob and it costs you small text")

    p.add_argument("--min-conf", type=float, default=0.30,
                   help="drop readings below this confidence")
    p.add_argument("--min-chars", type=int, default=2,
                   help="drop readings shorter than this many characters")

    p.add_argument("--no-track", action="store_true",
                   help="report every frame separately, no ids, no voting")
    p.add_argument("--iou", type=float, default=0.30, help="tracker IoU threshold")
    p.add_argument("--max-age", type=int, default=12,
                   help="frames a track survives without a detection")
    p.add_argument("--min-hits", type=int, default=2,
                   help="a track needs this many frames to be reported")

    p.add_argument("--stage", default="both", choices=["both", "detect"],
                   help="'detect' draws stage-1 boxes only, skipping recognition")
    return p.parse_args(argv)


def open_video(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {path}")
    meta = {
        "fps": cap.get(cv2.CAP_PROP_FPS) or 25.0,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    return cap, meta


def scale_for(frame: np.ndarray, long_side: int) -> float:
    """Downscale factor so OCR runs on a smaller frame; 1.0 means leave it alone."""
    if long_side <= 0:
        return 1.0
    longest = max(frame.shape[:2])
    return (long_side / longest) if longest > long_side else 1.0


def main(argv=None) -> int:
    args = parse_args(argv)

    cap, meta = open_video(args.video)
    total = meta["frames"] if args.max_frames <= 0 else min(meta["frames"], args.max_frames)
    print(f"video   : {args.video}")
    print(f"          {meta['width']}x{meta['height']}  {meta['fps']:.1f} fps  "
          f"{meta['frames']} frames")

    params = DetectParams(**PRESETS[args.preset])
    for name in ("mag", "low_text", "text_threshold", "link", "width_ths",
                 "min_size", "decoder", "merge_words"):
        value = getattr(args, name)
        if value is not None:              # an explicit flag overrides the preset
            setattr(params, name, value)

    print(f"engine  : {args.engine}  mode={args.mode}  preset={args.preset}")
    print(f"          mag={params.mag} low_text={params.low_text} "
          f"link={params.link} min_size={params.min_size}")
    print("          (first run downloads pretrained weights)")
    engine = build_engine(args.engine, [s.strip() for s in args.langs.split(",")],
                          gpu=not args.cpu, params=params)

    # Nothing to track in detect-only mode: tracks are keyed on readings.
    track_off = args.no_track or args.stage == "detect"
    tracker = None if track_off else TextTracker(
        iou_threshold=args.iou, max_age=args.max_age, min_hits=args.min_hits)

    writer = None
    if args.render:
        Path(args.render).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(args.render, cv2.VideoWriter_fourcc(*"mp4v"),
                                 meta["fps"], (meta["width"], meta["height"]))
        if not writer.isOpened():
            raise SystemExit(f"cannot write video: {args.render}")

    frames_dir = None
    if args.frames_dir:
        frames_dir = Path(args.frames_dir)
        frames_dir.mkdir(parents=True, exist_ok=True)

    per_frame: list = []
    last_items: list = []          # reused for frames we skip, so video stays smooth
    n_detected = n_read = processed = 0
    frame_index = -1
    started = time.time()

    while True:
        if args.max_frames > 0 and processed >= args.max_frames:
            break
        ok, frame = cap.read()
        if not ok:
            break
        frame_index += 1
        processed += 1

        if frame_index % args.every == 0:
            scale = scale_for(frame, args.resize)
            small = (cv2.resize(frame, None, fx=scale, fy=scale,
                                interpolation=cv2.INTER_AREA)
                     if scale != 1.0 else frame)

            if args.stage == "detect":
                # ---- stage 1 only: where is the text? -----------------------
                polys = engine.detect(small)
                preds = []
            elif args.mode == "pipeline":
                # The engine runs both stages itself and returns boxes + text.
                preds = engine.read(small)
                polys = [p.poly for p in preds]
            else:
                # ---- stage 1: where? ---- then ---- stage 2: what? ----------
                polys = engine.detect(small)
                preds = engine.recognize(small, polys)

            n_detected += len(polys)
            if preds:
                preds = [p for p in preds
                         if p.confidence >= args.min_conf
                         and len(p.text) >= args.min_chars]
                n_read += len(preds)

            # back to full-resolution coordinates
            if scale != 1.0:
                inv = 1.0 / scale
                for p in preds:
                    p.poly = p.poly * inv
                polys = [np.asarray(q, dtype=np.float32) * inv for q in polys]

            if args.stage == "detect":
                last_items = [{"poly": q, "track_id": None} for q in polys]
            elif tracker is None:
                last_items = [{"poly": p.poly, "text": p.text,
                               "confidence": p.confidence, "track_id": None}
                              for p in preds]
            else:
                tracks = tracker.update(preds, frame_index)
                last_items = [{"poly": p.poly, "text": t.text,
                               "confidence": p.confidence, "track_id": t.track_id}
                              for p, t in zip(preds, tracks)]

            per_frame.append({
                "frame": frame_index,
                "detections": [
                    {"poly": np.asarray(i["poly"]).round(1).tolist(),
                     "text": i.get("text", ""),
                     "confidence": round(float(i.get("confidence") or 0.0), 4),
                     "track_id": i.get("track_id")}
                    for i in last_items
                ],
            })

        if writer is not None or frames_dir is not None:
            shown = annotate(frame, last_items, stage=args.stage)
            draw_hud(shown, [f"frame {frame_index}",
                             f"text in view: {len(last_items)}"])
            if writer is not None:
                writer.write(shown)
            if frames_dir is not None:
                cv2.imwrite(str(frames_dir / f"frame_{frame_index:06d}.jpg"), shown)

        if total > 0 and processed % 25 == 0:
            elapsed = time.time() - started
            fps = processed / elapsed if elapsed > 0 else 0.0
            eta = (total - processed) / fps if fps > 0 else 0.0
            sys.stdout.write(f"\r  {processed}/{total} frames  {fps:5.1f} fps  "
                             f"ETA {eta/60:4.1f} min ")
            sys.stdout.flush()

    cap.release()
    if writer is not None:
        writer.release()
    print()

    elapsed = time.time() - started
    print(f"processed {processed} frames in {elapsed:.1f}s "
          f"({processed / max(elapsed, 1e-6):.1f} fps)")
    print(f"stage 1  : {n_detected} text regions detected")
    print(f"stage 2  : {n_read} readings kept (conf >= {args.min_conf})")

    results = {
        "video": str(args.video),
        "engine": args.engine,
        "meta": meta,
        "settings": {"every": args.every, "resize": args.resize,
                     "min_conf": args.min_conf, "tracking": tracker is not None},
        "frames": per_frame,
    }
    if tracker is not None:
        tracks = tracker.finished()
        results["tracks"] = [t.to_dict() for t in tracks]
        print(f"tracking : {len(tracks)} distinct pieces of text")
        for t in tracks[:25]:
            print(f"           #{t.track_id:<3} {t.text!r:<28} "
                  f"frames {t.start_frame}-{t.last_frame}  "
                  f"agreement {t.confidence:.2f}")
        if len(tracks) > 25:
            print(f"           ... and {len(tracks) - 25} more")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"wrote    : {args.out}")
    if args.render:
        print(f"wrote    : {args.render}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
