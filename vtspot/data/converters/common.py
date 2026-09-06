"""Shared helpers for dataset converters.

Every public video-text benchmark ships a different annotation format, and
several ship *variants* of their own format between releases.  Rather than
hard-coding one layout per dataset and silently producing garbage when it does
not match, the converters here auto-detect among the known shapes and raise a
descriptive error otherwise.  A converter that quietly emits zero instances is
much more expensive to debug than one that refuses to run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from ..schema import Frame, Instance, VideoAnnotation, normalise_text

# Transcription markers meaning "text is present but must not be scored".
DONT_CARE = {"###", "##don't#care##", "##dont#care##", "#dontcare#", "###dont#care###",
             "*", "?", "###don't#care###"}


def is_dont_care(text: Optional[str]) -> bool:
    if text is None:
        return True
    return text.strip().lower() in {t.lower() for t in DONT_CARE}


def parse_points(raw) -> Optional[np.ndarray]:
    """Accept the several point encodings used across these datasets.

    Handles ``[x1,y1,x2,y2,...]``, ``[[x,y],...]`` and ``"x1,y1,x2,y2,..."``.
    """
    if raw is None:
        return None
    if isinstance(raw, str):
        parts = [p for p in raw.replace(";", ",").split(",") if p.strip()]
        try:
            raw = [float(p) for p in parts]
        except ValueError:
            return None
    arr = np.asarray(raw, dtype=np.float32).reshape(-1)
    if arr.size < 8 or arr.size % 2 != 0:
        return None
    return arr.reshape(-1, 2)


def extract_frames(video_path: str | Path, out_dir: str | Path,
                   max_frames: Optional[int] = None, quality: int = 92) -> int:
    """Decode a video file to ``out_dir/000000.jpg`` ...  Returns the frame count."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open {video_path}")
    n = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or (max_frames is not None and n >= max_frames):
                break
            cv2.imwrite(str(out_dir / f"{n:06d}.jpg"), frame,
                        [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            n += 1
    finally:
        cap.release()
    return n


def frame_size(frames_dir: str | Path) -> Tuple[int, int]:
    files = sorted(Path(frames_dir).glob("*.jpg"))
    if not files:
        return 0, 0
    img = cv2.imread(str(files[0]))
    return (img.shape[1], img.shape[0]) if img is not None else (0, 0)


def build_annotation(video_id: str, width: int, height: int, source: str,
                     per_frame: Dict[int, List[dict]], fps: float = 30.0
                     ) -> VideoAnnotation:
    """``{frame_idx: [{points, text, track_id, ignore}]}`` -> VideoAnnotation."""
    frames: List[Frame] = []
    for frame_idx in sorted(per_frame):
        instances = []
        for item in per_frame[frame_idx]:
            poly = item["points"]
            if poly is None or len(poly) < 4:
                continue
            text, auto_ignore = normalise_text(item.get("text"))
            ignore = bool(item.get("ignore", False)) or auto_ignore
            instances.append(Instance(polygon=np.asarray(poly, np.float32).tolist(),
                                      text=text, track_id=int(item.get("track_id", -1)),
                                      legible=not ignore, ignore=ignore))
        frames.append(Frame(frame_idx=frame_idx, instances=instances))
    return VideoAnnotation(video_id=video_id, width=width, height=height,
                           frames=frames, fps=fps, source=source)


def load_json_annotation(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_annotation(ann: VideoAnnotation, min_instances: int = 1) -> List[str]:
    """Sanity checks that catch the usual converter failures."""
    problems: List[str] = []
    total = sum(len(f.instances) for f in ann.frames)
    if total < min_instances:
        problems.append(f"{ann.video_id}: only {total} instances parsed -- format mismatch?")
    if ann.width <= 0 or ann.height <= 0:
        problems.append(f"{ann.video_id}: frame size unknown ({ann.width}x{ann.height})")
    out_of_bounds = 0
    for frame in ann.frames:
        for inst in frame.instances:
            p = inst.poly_array()
            if (p[:, 0].max() > ann.width * 1.5 or p[:, 1].max() > ann.height * 1.5
                    or p.min() < -ann.width * 0.5):
                out_of_bounds += 1
    if out_of_bounds and out_of_bounds >= max(total * 0.02, 2):
        problems.append(f"{ann.video_id}: {out_of_bounds}/{total} polygons out of bounds "
                        "-- coordinate order or scaling is probably wrong")
    tracked = sum(1 for f in ann.frames for i in f.instances if i.track_id >= 0)
    if total and tracked == 0:
        problems.append(f"{ann.video_id}: no track ids -- tracking supervision will be absent")
    return problems
