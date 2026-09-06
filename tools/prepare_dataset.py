#!/usr/bin/env python3
"""Convert a downloaded benchmark into the unified schema and validate it.

    python tools/prepare_dataset.py --dataset icdar15_video \
        --videos raw/ICDAR15/videos --annotations raw/ICDAR15/gt \
        --out data/prepared/icdar15_video_train

The validation step is the point of this tool as much as the conversion: it
reports instances parsed, tracks found, mean instances per frame and any
polygons outside the frame.  Compare those against the dataset's published
statistics (DSText should show ~24 instances per frame, ICDAR15-video ~5.5) --
if yours are far off, the parser matched the wrong format variant and every
number you train and evaluate afterwards is meaningless.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vtspot.data.converters import icdar_video, json_video, roadtext
from vtspot.data.converters.common import extract_frames, frame_size, validate_annotation
from vtspot.data.schema import summarise

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV"}
ANN_EXT = {"icdar13_video": ".xml", "icdar15_video": ".xml", "bovtext": ".json",
           "dstext": ".json", "artvideo": ".json", "roadtext1k": ".csv"}
# Published reference statistics, for the sanity check printed at the end.
EXPECTED_DENSITY = {"icdar15_video": 5.5, "icdar13_video": 4.0, "dstext": 24.0,
                    "bovtext": 8.0, "artvideo": 13.0, "roadtext1k": 4.0}


def stem_key(name: str) -> str:
    """Match annotation files to videos despite differing prefixes/suffixes."""
    s = Path(name).stem
    s = re.sub(r"^(gt_|GT_|res_)", "", s)
    s = re.sub(r"(_GT|_gt)$", "", s)
    return s.lower()


def find_pairs(videos_dir: Optional[Path], frames_dir: Optional[Path],
               ann_dir: Path, ext: str) -> List[tuple]:
    anns = {stem_key(p.name): p for p in sorted(ann_dir.rglob(f"*{ext}"))}
    pairs = []
    if videos_dir is not None:
        for v in sorted(videos_dir.rglob("*")):
            if v.suffix in VIDEO_EXT and stem_key(v.name) in anns:
                pairs.append((v, None, anns[stem_key(v.name)]))
    if frames_dir is not None:
        for d in sorted(p for p in frames_dir.iterdir() if p.is_dir()):
            if stem_key(d.name) in anns:
                pairs.append((None, d, anns[stem_key(d.name)]))
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(ANN_EXT))
    ap.add_argument("--annotations", required=True, help="directory of annotation files")
    ap.add_argument("--videos", default=None, help="directory of video files to decode")
    ap.add_argument("--frames", default=None,
                    help="directory of already-extracted frames (<video_id>/*.jpg)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--keep-non-latin", action="store_true",
                    help="BOVText: keep Chinese instances as trainable rather than ignore")
    ap.add_argument("--frame-offset", type=int, default=None,
                    help="override the auto-detected 0/1-based frame numbering")
    args = ap.parse_args()

    if not args.videos and not args.frames:
        ap.error("pass --videos (to decode) or --frames (already extracted)")

    out = Path(args.out)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    (out / "annotations").mkdir(parents=True, exist_ok=True)

    pairs = find_pairs(Path(args.videos) if args.videos else None,
                       Path(args.frames) if args.frames else None,
                       Path(args.annotations), ANN_EXT[args.dataset])
    if not pairs:
        print(f"no (video, annotation) pairs found -- check --annotations "
              f"contains *{ANN_EXT[args.dataset]} files with matching names")
        return 1
    print(f"{len(pairs)} videos to convert")

    all_problems: List[str] = []
    for i, (video, frames_src, ann_path) in enumerate(pairs):
        video_id = stem_key((video or frames_src).name)
        dest = out / "frames" / video_id

        frames_dir = ""
        if video is not None:
            n = extract_frames(video, dest, max_frames=args.max_frames)
            probe = dest
        else:
            # Prefer a symlink so the prepared root is self-contained, but fall
            # back to recording the source path.  Windows rejects symlinks
            # without Administrator rights or Developer Mode, and copying every
            # frame of every video is not a reasonable alternative.
            dest.parent.mkdir(parents=True, exist_ok=True)
            probe = frames_src
            if not dest.exists():
                try:
                    dest.symlink_to(frames_src.resolve(), target_is_directory=True)
                    probe = dest
                except (OSError, NotImplementedError):
                    frames_dir = str(frames_src.resolve())
            else:
                probe = dest
            n = len(list(probe.glob("*.jpg")))
        width, height = frame_size(probe)

        if args.dataset in ("icdar13_video", "icdar15_video"):
            ann = icdar_video.convert(ann_path, video_id, width, height,
                                      fps=args.fps, source=args.dataset)
        elif args.dataset == "roadtext1k":
            ann = roadtext.convert(ann_path, video_id, width, height, fps=args.fps)
        else:
            ann = json_video.convert(ann_path, video_id, width, height, args.dataset,
                                     fps=args.fps, frame_offset=args.frame_offset,
                                     non_latin_as_ignore=not args.keep_non_latin)
        ann.frames_dir = frames_dir
        ann.to_json(out / "annotations" / f"{video_id}.json")

        problems = validate_annotation(ann)
        all_problems += problems
        s = ann.stats()
        flag = "  <-- CHECK" if problems else ""
        print(f"[{i + 1}/{len(pairs)}] {video_id}: {n} frames, "
              f"{int(s['instances'])} instances, {int(s['tracks'])} tracks, "
              f"{s['mean_instances_per_frame']:.1f}/frame{flag}")
        for p in problems:
            print(f"    ! {p}")

    totals = summarise(out)
    print("\n=== corpus ===")
    print(json.dumps(totals, indent=2))
    expected = EXPECTED_DENSITY.get(args.dataset)
    if expected:
        got = totals["mean_instances_per_frame"]
        ratio = got / max(expected, 1e-6)
        verdict = "plausible" if 0.5 <= ratio <= 2.0 else "SUSPICIOUS -- check the parser"
        print(f"mean instances/frame: {got:.1f} (published ~{expected}) -> {verdict}")
    if all_problems:
        print(f"\n{len(all_problems)} validation warnings -- review before training")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
