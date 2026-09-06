"""RoadText-1K -> unified schema.

CSV per video, one row per instance per frame::

    frame_id, track_id, x1, y1, x2, y2, transcription, legibility, ...

RoadText annotates *axis-aligned* boxes, which become degenerate 4-point
polygons.  That is fine for detection and tracking supervision but means the
dataset teaches nothing about orientation -- do not train on it alone.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from ..schema import VideoAnnotation
from .common import build_annotation, is_dont_care

ILLEGIBLE = {"illegible", "0", "false", "no"}


def parse_csv(path: str | Path, frame_offset: int = 0) -> Dict[int, List[dict]]:
    per_frame: Dict[int, List[dict]] = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as fh:
        sample = fh.read(4096)
        fh.seek(0)
        has_header = csv.Sniffer().has_header(sample) if sample else False
        reader = csv.reader(fh)
        if has_header:
            next(reader, None)
        for row in reader:
            if len(row) < 6:
                continue
            try:
                frame = int(float(row[0])) - frame_offset
                track = int(float(row[1]))
                x1, y1, x2, y2 = (float(v) for v in row[2:6])
            except ValueError:
                continue
            text = row[6].strip() if len(row) > 6 else ""
            legible = True
            if len(row) > 7:
                legible = row[7].strip().lower() not in ILLEGIBLE
            ignore = is_dont_care(text) or not legible or not text
            poly = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], np.float32)
            per_frame.setdefault(frame, []).append(
                {"points": poly, "text": "" if ignore else text,
                 "track_id": track, "ignore": ignore})
    return per_frame


def convert(path: str | Path, video_id: str, width: int, height: int,
            fps: float = 30.0, frame_offset: int = 0) -> VideoAnnotation:
    return build_annotation(video_id, width, height, "roadtext1k",
                            parse_csv(path, frame_offset), fps)
