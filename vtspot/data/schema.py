"""One annotation schema that every dataset converter targets.

ICDAR15-video ships XML, DSText ships its own JSON, BOVText another, RoadText-1K
CSV.  Normalising once at conversion time means the training code never learns
about any of them, and adding a dataset is a converter file rather than a change
to the model.

On-disk layout produced by ``tools/prepare_dataset.py``::

    <root>/
      frames/<video_id>/<frame_idx:06d>.jpg
      annotations/<video_id>.json

Annotation JSON::

    {
      "video_id": "Video_10_1_2",
      "width": 1280, "height": 720, "fps": 30.0,
      "source": "icdar15_video",
      "frames": [
        {"frame_idx": 0,
         "instances": [
            {"track_id": 3,
             "polygon": [[x,y], ...],      # >= 4 points, top edge then bottom edge
             "text": "SHOP",
             "legible": true,
             "ignore": false}
         ]}
      ]
    }

``ignore`` marks regions excluded from the loss *and* from evaluation -- the
"###" / "don't care" convention.  Getting this wrong inflates the false-positive
count and makes MOTA look far worse than the model deserves.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

IGNORE_TOKENS = {"###", "#", "", "*", "null", "NULL"}


@dataclass
class Instance:
    polygon: List[List[float]]
    text: str = ""
    track_id: int = -1
    legible: bool = True
    ignore: bool = False

    def poly_array(self) -> np.ndarray:
        return np.asarray(self.polygon, dtype=np.float32)

    @classmethod
    def from_dict(cls, d: dict) -> "Instance":
        return cls(polygon=[[float(x), float(y)] for x, y in d["polygon"]],
                   text=d.get("text", ""),
                   track_id=int(d.get("track_id", -1)),
                   legible=bool(d.get("legible", True)),
                   ignore=bool(d.get("ignore", False)))


@dataclass
class Frame:
    frame_idx: int
    instances: List[Instance] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "Frame":
        return cls(frame_idx=int(d["frame_idx"]),
                   instances=[Instance.from_dict(i) for i in d.get("instances", [])])


@dataclass
class VideoAnnotation:
    video_id: str
    width: int
    height: int
    frames: List[Frame] = field(default_factory=list)
    fps: float = 30.0
    source: str = "unknown"
    frames_dir: str = ""      # absolute path to frames, when not under <root>/frames

    def to_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self)), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "VideoAnnotation":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(video_id=d["video_id"], width=int(d["width"]), height=int(d["height"]),
                   fps=float(d.get("fps", 30.0)), source=d.get("source", "unknown"),
                   frames_dir=d.get("frames_dir", ""),
                   frames=[Frame.from_dict(f) for f in d.get("frames", [])])

    def num_tracks(self) -> int:
        return len({i.track_id for f in self.frames for i in f.instances
                    if i.track_id >= 0 and not i.ignore})

    def stats(self) -> Dict[str, float]:
        counts = [len([i for i in f.instances if not i.ignore]) for f in self.frames]
        return {"frames": len(self.frames), "tracks": self.num_tracks(),
                "instances": int(sum(counts)),
                "mean_instances_per_frame": float(np.mean(counts)) if counts else 0.0}


def normalise_text(text: Optional[str]) -> tuple[str, bool]:
    """Return ``(text, ignore)``, applying the don't-care convention."""
    if text is None:
        return "", True
    t = text.strip()
    if t in IGNORE_TOKENS:
        return "", True
    return t, False


def index_dataset(root: str | Path) -> List[Path]:
    """All annotation files under a prepared dataset root."""
    ann_dir = Path(root) / "annotations"
    if not ann_dir.is_dir():
        raise FileNotFoundError(f"{ann_dir} not found -- run tools/prepare_dataset.py first")
    return sorted(ann_dir.glob("*.json"))


def summarise(root: str | Path) -> Dict[str, float]:
    """Corpus-level statistics -- how you check a converter actually worked."""
    totals = {"videos": 0, "frames": 0, "tracks": 0, "instances": 0}
    per_frame: List[float] = []
    for path in index_dataset(root):
        ann = VideoAnnotation.from_json(path)
        s = ann.stats()
        totals["videos"] += 1
        totals["frames"] += int(s["frames"])
        totals["tracks"] += int(s["tracks"])
        totals["instances"] += int(s["instances"])
        per_frame.append(s["mean_instances_per_frame"])
    totals["mean_instances_per_frame"] = float(np.mean(per_frame)) if per_frame else 0.0
    return totals
