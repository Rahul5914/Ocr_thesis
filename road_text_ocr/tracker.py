"""Link per-frame detections into tracks, and vote on the transcription.

Per-frame OCR alone gives you the same signboard as 40 unrelated results, each
read slightly differently.  Two cheap steps fix that:

* **IoU association** -- a box that overlaps last frame's box is the same sign,
  so it keeps its id.  A short `max_age` keeps the id alive across a few frames
  of occlusion or missed detection.
* **Confidence voting** -- across every frame a track was seen, the reading with
  the highest summed confidence wins.  A word misread in three frames and read
  correctly in twenty comes out correct, which no single frame guarantees.

This is a deliberately simple appearance-free tracker: it uses geometry only.
It is enough for road-side video, where text moves smoothly and is rarely
duplicated within one frame.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    """Intersection over union of two (x1, y1, x2, y2) boxes."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(ix2 - ix1, 0.0), max(iy2 - iy1, 0.0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(a[2] - a[0], 0.0) * max(a[3] - a[1], 0.0)
    area_b = max(b[2] - b[0], 0.0) * max(b[3] - b[1], 0.0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Track:
    track_id: int
    box: tuple
    poly: np.ndarray
    last_frame: int
    start_frame: int
    frames: Dict[int, List[List[float]]] = field(default_factory=dict)
    votes: Dict[str, float] = field(default_factory=lambda: defaultdict(float))
    hits: int = 0

    @property
    def text(self) -> str:
        """The reading with the highest total confidence over the track."""
        return max(self.votes.items(), key=lambda kv: kv[1])[0] if self.votes else ""

    @property
    def confidence(self) -> float:
        """Share of the track's total confidence that the winning reading holds.

        1.0 means every frame agreed; 0.4 means the reading is contested and the
        transcription should not be trusted much.
        """
        total = sum(self.votes.values())
        return (max(self.votes.values()) / total) if total > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "track_id": self.track_id,
            "text": self.text,
            "text_confidence": round(self.confidence, 4),
            "start_frame": self.start_frame,
            "end_frame": self.last_frame,
            "num_frames": len(self.frames),
            "frames": {str(k): v for k, v in sorted(self.frames.items())},
        }


class TextTracker:
    """Greedy IoU tracker over per-frame text predictions."""

    def __init__(self, iou_threshold: float = 0.3, max_age: int = 12,
                 min_hits: int = 2):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.min_hits = min_hits
        self.tracks: List[Track] = []
        self._next_id = 1

    def update(self, predictions, frame_index: int) -> List[Track]:
        """Assign ids to this frame's predictions; returns the matched tracks."""
        live = [t for t in self.tracks if frame_index - t.last_frame <= self.max_age]

        pairs = sorted(
            ((iou(p.box, t.box), pi, ti)
             for pi, p in enumerate(predictions)
             for ti, t in enumerate(live)),
            key=lambda x: -x[0],
        )

        used_p, used_t, assigned = set(), set(), {}
        for score, pi, ti in pairs:
            if score < self.iou_threshold:
                break
            if pi in used_p or ti in used_t:
                continue
            used_p.add(pi)
            used_t.add(ti)
            assigned[pi] = live[ti]

        touched: List[Track] = []
        for pi, pred in enumerate(predictions):
            track = assigned.get(pi)
            if track is None:
                track = Track(track_id=self._next_id, box=pred.box, poly=pred.poly,
                              last_frame=frame_index, start_frame=frame_index)
                self._next_id += 1
                self.tracks.append(track)
            track.box = pred.box
            track.poly = pred.poly
            track.last_frame = frame_index
            track.hits += 1
            track.frames[frame_index] = np.asarray(pred.poly).round(1).tolist()
            track.votes[pred.text] += float(pred.confidence)
            touched.append(track)
        return touched

    def finished(self) -> List[Track]:
        """Tracks worth reporting, oldest first.

        Tracks seen fewer than ``min_hits`` times are dropped -- a box that
        appears in one frame and never again is almost always a false positive
        on a texture, not a real sign.
        """
        keep = [t for t in self.tracks if t.hits >= self.min_hits]
        return sorted(keep, key=lambda t: t.start_frame)
