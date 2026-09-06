"""Online video text tracker: detections in, trajectories with transcriptions out."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..utils.polygon import poly_to_bbox
from .decode import aggregate_trajectory_text
from .matcher import LongTermMatcher, MatchConfig, ShortTermMatcher


@dataclass
class Detection:
    polygon: np.ndarray
    score: float
    embedding: np.ndarray
    text: str = ""
    text_confidence: float = 0.0

    @property
    def bbox(self) -> np.ndarray:
        return poly_to_bbox(self.polygon)


@dataclass
class Track:
    track_id: int
    polygon: np.ndarray
    bbox: np.ndarray
    embedding: np.ndarray
    score: float
    start_frame: int
    last_frame: int
    hits: int = 1
    age: int = 0                                     # frames since last seen
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, np.float32))
    observations: List[Tuple[str, float]] = field(default_factory=list)
    history: Dict[int, np.ndarray] = field(default_factory=dict)
    scores: List[float] = field(default_factory=list)
    confirmed: bool = False

    def predict(self) -> np.ndarray:
        """Constant-velocity extrapolation of the box for gating.

        A full Kalman filter buys little here: text is rigid, mostly moves with
        the camera, and the association already has an appearance term.  What
        matters is not extrapolating a *stale* velocity, so it decays with age.
        """
        decay = 0.9 ** self.age
        shift = np.concatenate([self.velocity, self.velocity]) * decay
        return self.bbox + shift


@dataclass
class TrackerConfig:
    match: MatchConfig = field(default_factory=MatchConfig)
    det_threshold: float = 0.3
    high_threshold: float = 0.5      # detections above this can start a track
    min_hits: int = 2                # hits before a track is reported
    max_age: int = 30                # frames a lost track stays re-associable
    embed_momentum: float = 0.8      # EMA on the track's appearance
    use_recognition_rescore: bool = True
    rescore_weight: float = 0.5


class VideoTextTracker:
    """Frame-by-frame association with long-term recovery.

    ``use_recognition_rescore`` implements the domain-gap fix that GoMatching++
    addresses with a trained rescoring head, but for free: a genuine text region
    decodes to a *confident string*, whereas a background false positive decodes
    to blanks or garbage.  Combining the detector's score with the recogniser's
    confidence therefore separates the two using a signal the model already
    produces.  Video frames are degraded enough that the detector alone is
    badly calibrated on them, and a fixed threshold on a mis-calibrated score is
    exactly where recall is lost.
    """

    def __init__(self, cfg: Optional[TrackerConfig] = None):
        self.cfg = cfg or TrackerConfig()
        self.short = ShortTermMatcher(self.cfg.match)
        self.long = LongTermMatcher(self.cfg.match)
        self.reset()

    def reset(self) -> None:
        self.tracks: List[Track] = []
        self.lost: List[Track] = []
        self._next_id = 1
        self._frame = -1

    # -- scoring ---------------------------------------------------------
    def _combined_score(self, det: Detection) -> float:
        if not self.cfg.use_recognition_rescore or det.text_confidence <= 0:
            return det.score
        w = self.cfg.rescore_weight
        # Geometric mean, so either signal being near zero vetoes the instance.
        return float((det.score ** (1 - w)) * (det.text_confidence ** w))

    # -- main step -------------------------------------------------------
    def update(self, frame_idx: int, detections: Sequence[Detection]) -> List[Track]:
        cfg = self.cfg
        self._frame = frame_idx
        dets = [d for d in detections if self._combined_score(d) >= cfg.det_threshold]

        det_boxes = (np.stack([d.bbox for d in dets]) if dets
                     else np.zeros((0, 4), np.float32))
        det_embeds = (np.stack([d.embedding for d in dets]) if dets
                      else np.zeros((0, 1), np.float32))

        track_boxes = (np.stack([t.predict() for t in self.tracks]) if self.tracks
                       else np.zeros((0, 4), np.float32))
        track_embeds = (np.stack([t.embedding for t in self.tracks]) if self.tracks
                        else np.zeros((0, det_embeds.shape[1]), np.float32))

        matches, unmatched_tracks, unmatched_dets = self.short(
            track_boxes, track_embeds, det_boxes, det_embeds)
        for ti, di in matches:
            self._update_track(self.tracks[ti], dets[di], frame_idx)

        # Second pass: what is left goes to the memory bank of lost tracks.
        if unmatched_dets and self.lost:
            lost_boxes = np.stack([t.bbox for t in self.lost])
            lost_embeds = np.stack([t.embedding for t in self.lost])
            lost_ages = np.array([t.age for t in self.lost], np.float32)
            sub_boxes = det_boxes[unmatched_dets]
            sub_embeds = det_embeds[unmatched_dets]
            lm, _, still_unmatched = self.long(lost_boxes, lost_embeds, lost_ages,
                                               sub_boxes, sub_embeds)
            revived = []
            for li, sub_di in lm:
                track = self.lost[li]
                self._update_track(track, dets[unmatched_dets[sub_di]], frame_idx)
                self.tracks.append(track)
                revived.append(li)
            self.lost = [t for i, t in enumerate(self.lost) if i not in set(revived)]
            unmatched_dets = [unmatched_dets[i] for i in still_unmatched]

        # Age out unmatched tracks.
        surviving = []
        for ti in unmatched_tracks:
            track = self.tracks[ti]
            track.age += 1
            if track.age <= cfg.max_age:
                self.lost.append(track)
            surviving.append(ti)
        self.tracks = [t for i, t in enumerate(self.tracks) if i not in set(surviving)]
        self.lost = [t for t in self.lost if t.age <= cfg.max_age]
        for t in self.lost:
            t.age += 0  # ages are incremented when they leave `tracks`

        # New tracks from confident leftovers.
        for di in unmatched_dets:
            det = dets[di]
            if self._combined_score(det) >= cfg.high_threshold:
                self.tracks.append(self._new_track(det, frame_idx))

        for t in self.lost:
            t.age = max(t.age, frame_idx - t.last_frame)

        return [t for t in self.tracks if t.hits >= cfg.min_hits]

    # -- helpers ---------------------------------------------------------
    def _new_track(self, det: Detection, frame_idx: int) -> Track:
        track = Track(track_id=self._next_id, polygon=det.polygon.copy(),
                      bbox=det.bbox, embedding=det.embedding.copy(),
                      score=self._combined_score(det), start_frame=frame_idx,
                      last_frame=frame_idx)
        track.history[frame_idx] = det.polygon.copy()
        track.scores.append(track.score)
        if det.text:
            track.observations.append((det.text, det.text_confidence))
        self._next_id += 1
        return track

    def _update_track(self, track: Track, det: Detection, frame_idx: int) -> None:
        m = self.cfg.embed_momentum
        new_box = det.bbox
        old_centre = np.array([(track.bbox[0] + track.bbox[2]) / 2,
                               (track.bbox[1] + track.bbox[3]) / 2], np.float32)
        new_centre = np.array([(new_box[0] + new_box[2]) / 2,
                               (new_box[1] + new_box[3]) / 2], np.float32)
        span = max(frame_idx - track.last_frame, 1)
        track.velocity = (new_centre - old_centre) / span

        track.polygon = det.polygon.copy()
        track.bbox = new_box
        # EMA on appearance: a single blurred frame should nudge the template,
        # not replace it.
        track.embedding = m * track.embedding + (1 - m) * det.embedding
        norm = np.linalg.norm(track.embedding)
        if norm > 1e-6:
            track.embedding = track.embedding / norm
        track.score = self._combined_score(det)
        track.scores.append(track.score)
        track.hits += 1
        track.age = 0
        track.last_frame = frame_idx
        track.history[frame_idx] = det.polygon.copy()
        if det.text:
            track.observations.append((det.text, det.text_confidence))
        if track.hits >= self.cfg.min_hits:
            track.confirmed = True

    # -- output ----------------------------------------------------------
    def finalise(self, min_length: int = 2) -> List[dict]:
        """All trajectories with an aggregated transcription per track."""
        out = []
        for track in self.tracks + self.lost:
            if len(track.history) < min_length:
                continue
            text, conf = aggregate_trajectory_text(track.observations)
            out.append({
                "track_id": track.track_id,
                "text": text,
                "text_confidence": conf,
                "score": float(np.mean(track.scores)) if track.scores else 0.0,
                "start_frame": track.start_frame,
                "end_frame": track.last_frame,
                "frames": {int(f): p.tolist() for f, p in sorted(track.history.items())},
            })
        return sorted(out, key=lambda d: d["track_id"])
