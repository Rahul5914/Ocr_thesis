"""BOVText / DSText / ArTVideo -> unified schema.

These three share a JSON-per-video family::

    {"1": [{"points": [...], "ID": 3, "transcription": "SHOP", "category_id": 1,
            "language": "English"}], "2": [...]}

with per-dataset variation in key names (``ID`` vs ``tracking_id`` vs
``track_id``; ``transcription`` vs ``text``; points flat vs nested) and in
whether frame numbering starts at 0 or 1.  Rather than three near-identical
parsers, one tolerant parser handles the family and reports what it inferred, so
a format drift shows up as a message instead of an empty dataset.

BOVText is bilingual: many instances are Chinese.  Keep them as *ignore* when
training an alphanumeric model -- they are real text, so scoring them as
background teaches the detector to suppress text.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np

from ..schema import VideoAnnotation
from .common import build_annotation, is_dont_care, load_json_annotation, parse_points

TRACK_KEYS = ("ID", "id", "tracking_id", "track_id", "trackID", "instance_id")
TEXT_KEYS = ("transcription", "text", "Transcription", "label", "trans")
POINT_KEYS = ("points", "polygon", "poly", "bbox", "segmentation", "pts")
LANG_KEYS = ("language", "lang", "Language")


def _first(d: dict, keys) -> Optional[object]:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def _is_latin(text: str) -> bool:
    return bool(text) and all(ord(ch) < 128 for ch in text)


def parse_json_video(path: str | Path, frame_offset: Optional[int] = None,
                     non_latin_as_ignore: bool = True) -> Dict[int, List[dict]]:
    """Parse one JSON annotation file into ``{frame_idx: [item]}``.

    ``frame_offset`` is auto-detected from the smallest frame key (0 or 1) when
    not given; getting it wrong shifts every annotation by one frame, which
    quietly halves IoU on fast-moving text.
    """
    data = load_json_annotation(path)
    if isinstance(data, dict) and "frames" in data and isinstance(data["frames"], (list, dict)):
        data = data["frames"]

    raw: Dict[int, list] = {}
    if isinstance(data, dict):
        for key, value in data.items():
            m = re.search(r"\d+", str(key))
            if m is None or not isinstance(value, list):
                continue
            raw[int(m.group())] = value
    elif isinstance(data, list):
        for i, value in enumerate(data):
            if isinstance(value, dict) and "instances" in value:
                raw[int(value.get("frame_idx", i))] = value["instances"]
            elif isinstance(value, list):
                raw[i] = value
    if not raw:
        raise ValueError(f"{path}: no frame entries recognised; "
                         "expected {'<frame>': [instances]} or a list of frames")

    if frame_offset is None:
        frame_offset = 1 if min(raw) >= 1 else 0

    per_frame: Dict[int, List[dict]] = {}
    for frame_key, items in raw.items():
        parsed: List[dict] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            pts = parse_points(_first(item, POINT_KEYS))
            if pts is None:
                continue
            text_raw = _first(item, TEXT_KEYS)
            text = "" if text_raw is None else str(text_raw)
            lang = _first(item, LANG_KEYS)
            ignore = is_dont_care(text)
            if non_latin_as_ignore and text and not _is_latin(text):
                ignore = True
            if isinstance(lang, str) and lang.lower() in {"chinese", "cn", "zh"} \
                    and non_latin_as_ignore:
                ignore = True
            track = _first(item, TRACK_KEYS)
            try:
                track_id = int(track) if track is not None else -1
            except (TypeError, ValueError):
                track_id = -1
            parsed.append({"points": pts, "text": "" if ignore else text,
                           "track_id": track_id, "ignore": ignore})
        per_frame[frame_key - frame_offset] = parsed
    return per_frame


def convert(path: str | Path, video_id: str, width: int, height: int,
            source: str, fps: float = 30.0, frame_offset: Optional[int] = None,
            non_latin_as_ignore: bool = True) -> VideoAnnotation:
    per_frame = parse_json_video(path, frame_offset, non_latin_as_ignore)
    return build_annotation(video_id, width, height, source, per_frame, fps)
