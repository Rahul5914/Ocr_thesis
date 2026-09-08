#!/usr/bin/env python3
"""Find settings that work on *your* video, by trying them and counting.

Reading nothing has a small number of causes and they need opposite fixes, so
guessing wastes runs.  This samples a few frames, tries each setting on all of
them, and reports what each found:

    python diagnose.py --video road.mp4

    setting        regions  read  mean conf  sample
    -------------------------------------------------------------
    default              6     2       0.41  'PHARMACY', 'Bu2'
    small-text          14    11       0.78  'PHARMACY', 'EXIT 24', 'MAIN ST'
    tiny-text           17    12       0.71  'PHARMACY', 'EXIT 24', 'MAIN ST'
    two_stage/small     14     8       0.55  '[PHARMACY]', 'EXIT 24'

Take the row with the most *read* at a decent mean confidence and pass its
setting to detect_text_video.py.  `--dump` writes each row's boxes as an image
so you can see what it found rather than trusting the count.

Reading the rows:

* **regions high, read low** -- detection is fine, recognition is not.  The
  text is too blurred or too small to resolve; try `--mag 3`, or accept that a
  frame that is unreadable to you is unreadable to the model too.
* **regions low everywhere** -- detection is failing.  The text is smaller than
  `min_size`, or fainter than `low_text`.  `tiny-text` addresses both.
* **words arriving split** ("SPEED", "40" instead of "SPEED 40") -- lower
  `link` and raise `width_ths`; `small-text` already does.
* **nothing works at any setting** -- crop to the region that matters, or use a
  higher-resolution source.  No setting recovers detail the video does not have.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import cv2
import numpy as np

from engines import PRESETS, DetectParams, build_engine


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Try several settings on a few frames and report what each found.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--video", required=True)
    p.add_argument("--frames", type=int, default=3, help="how many frames to sample")
    p.add_argument("--engine", default="easyocr", choices=["easyocr", "paddleocr"])
    p.add_argument("--langs", default="en")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--dump", default=None,
                   help="directory to write one annotated image per setting")
    p.add_argument("--min-conf", type=float, default=0.3)
    return p.parse_args(argv)


def sample_frames(path: str, count: int) -> List[np.ndarray]:
    """Frames spread across the clip, skipping the very start and end."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in range(count):
        pos = int(total * (i + 1) / (count + 1)) if total > 0 else i
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    if not frames:
        raise SystemExit("could not read any frames")
    return frames


def main(argv=None) -> int:
    args = parse_args(argv)
    frames = sample_frames(args.video, args.frames)
    h, w = frames[0].shape[:2]
    print(f"video   : {args.video}  ({w}x{h}, sampled {len(frames)} frames)")
    print(f"engine  : {args.engine}\n")

    langs = [s.strip() for s in args.langs.split(",")]
    trials = [(name, name, "pipeline") for name in
              ("default", "small-text", "tiny-text")]
    trials.append(("two_stage/small-text", "small-text", "two_stage"))

    dump_dir = Path(args.dump) if args.dump else None
    if dump_dir:
        dump_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for label, preset, mode in trials:
        params = DetectParams(**PRESETS[preset])
        engine = build_engine(args.engine, langs, gpu=not args.cpu, params=params)

        regions = reads = 0
        confs: List[float] = []
        texts: List[str] = []
        for i, frame in enumerate(frames):
            if mode == "pipeline":
                preds = engine.read(frame)
                polys = [p.poly for p in preds]
            else:
                polys = engine.detect(frame)
                preds = engine.recognize(frame, polys)
            kept = [p for p in preds if p.confidence >= args.min_conf]
            regions += len(polys)
            reads += len(kept)
            confs.extend(p.confidence for p in kept)
            texts.extend(p.text for p in kept)

            if dump_dir and i == 0:
                from viz import annotate
                items = [{"poly": p.poly, "text": p.text,
                          "confidence": p.confidence, "track_id": None}
                         for p in kept]
                name = label.replace("/", "_")
                cv2.imwrite(str(dump_dir / f"{name}.jpg"), annotate(frame, items))

        mean_conf = float(np.mean(confs)) if confs else 0.0
        rows.append((label, regions, reads, mean_conf, texts))

    print(f"{'setting':<22}{'regions':>8}{'read':>6}{'mean conf':>11}   sample")
    print("-" * 92)
    best = max(rows, key=lambda r: (r[2], r[3]))
    for label, regions, reads, mean_conf, texts in rows:
        uniq = list(dict.fromkeys(texts))[:4]
        mark = "  <-- best" if (label, regions, reads) == best[:3] else ""
        print(f"{label:<22}{regions:>8}{reads:>6}{mean_conf:>11.2f}   "
              + ", ".join(repr(t) for t in uniq)[:44] + mark)

    print()
    label, regions, reads, mean_conf, _ = best
    if reads == 0:
        print("Nothing was read at any setting.  Detection found "
              f"{max(r[1] for r in rows)} regions at best, so:")
        print("  - regions > 0 : the text is detected but too degraded to resolve.")
        print("                  Try --mag 4, or a higher-resolution source.")
        print("  - regions = 0 : nothing looks like text at this scale.  Check the")
        print("                  frame is what you expect (--dump) and that the")
        print("                  text is not smaller than a few pixels tall.")
        return 1

    preset = label.split("/")[-1]
    mode = "two_stage" if label.startswith("two_stage") else "pipeline"
    print(f"Best: {label} -- {reads} readings at mean confidence {mean_conf:.2f}\n")
    print("Run the video with:")
    print(f"  python detect_text_video.py --video {args.video} \\")
    print(f"      --preset {preset} --mode {mode} --render annotated.mp4 --out results.json")
    if mean_conf < 0.5:
        print("\nMean confidence is low -- expect wrong readings.  Trajectory voting")
        print("recovers some of it, so judge the tracks table rather than a frame.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
