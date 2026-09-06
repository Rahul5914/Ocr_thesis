"""End-to-end video inference: frames in, text trajectories out."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .data.dataset import normalise_image
from .data.targets import decode_db_polygons
from .models.spotter import VideoTextSpotter
from .tracking.decode import ctc_greedy_decode
from .tracking.tracker import Detection, TrackerConfig, VideoTextTracker
from .utils.charset import Charset
from .utils.polygon import order_polygon_for_text, polygon_to_control_points


@dataclass
class PredictConfig:
    short_side: int = 736
    max_long_side: int = 1280
    bin_thresh: float = 0.3
    box_thresh: float = 0.45
    unclip_ratio: float = 1.7
    max_instances: int = 200
    num_control_points: int = 8
    tracker: TrackerConfig = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.tracker is None:
            self.tracker = TrackerConfig()


class VideoTextPredictor:
    """Runs the spotter over a video and returns trajectories.

    Detection, recognition and association all read from one forward pass of the
    shared backbone: the polygons come from the detection maps, and the crops
    for recognition and the embedding are sampled from the *same* feature map
    with PolyAlign.  There is no second pass over the image.
    """

    def __init__(self, model: VideoTextSpotter, charset: Charset,
                 cfg: Optional[PredictConfig] = None, device: str = "cpu"):
        self.model = model.eval().to(device)
        self.charset = charset
        self.cfg = cfg or PredictConfig()
        self.device = torch.device(device)

    # -- preprocessing ---------------------------------------------------
    def _prepare(self, frame: np.ndarray) -> Tuple[torch.Tensor, float]:
        cfg = self.cfg
        h, w = frame.shape[:2]
        scale = min(cfg.short_side / min(h, w), cfg.max_long_side / max(h, w))
        out_h = int(math.ceil(h * scale / 32) * 32)
        out_w = int(math.ceil(w * scale / 32) * 32)
        resized = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(normalise_image(resized)).unsqueeze(0)
        return tensor.to(self.device), out_w / w   # uniform enough after rounding

    # -- one frame -------------------------------------------------------
    @torch.no_grad()
    def detect_frame(self, frame: np.ndarray) -> List[Detection]:
        cfg = self.cfg
        tensor, _ = self._prepare(frame)
        h, w = frame.shape[:2]
        out = self.model.infer_frame(tensor)
        prob = out["prob"][0, 0].float().cpu().numpy()

        polys, scores = decode_db_polygons(
            prob, box_thresh=cfg.box_thresh, bin_thresh=cfg.bin_thresh,
            unclip_ratio=cfg.unclip_ratio, max_candidates=cfg.max_instances)
        if not polys:
            return []

        sy = h / prob.shape[0]
        sx = w / prob.shape[1]
        ordered: List[np.ndarray] = []
        keep_scores: List[float] = []
        ctrl_list: List[np.ndarray] = []
        for poly, score in zip(polys, scores):
            try:
                op = order_polygon_for_text(poly)
            except Exception:
                continue
            ctrl_list.append(polygon_to_control_points(op, cfg.num_control_points))
            scaled = op.copy()
            scaled[:, 0] *= sx
            scaled[:, 1] *= sy
            ordered.append(scaled)
            keep_scores.append(score)
        if not ctrl_list:
            return []

        # Control points are in detection-map (== input tensor) coordinates,
        # which is what PolyAlign expects; the returned polygons are rescaled to
        # the original frame for the caller.
        ctrl = torch.from_numpy(np.stack(ctrl_list)).to(self.device)
        idx = torch.zeros(len(ctrl), dtype=torch.long, device=self.device)
        inst = self.model.read_instances(out["features"], ctrl, idx)

        log_probs = F.log_softmax(inst["ctc_logits"].float(), dim=-1).cpu().numpy()
        embeds = inst["embed"].float().cpu().numpy()

        detections: List[Detection] = []
        for i, (poly, score) in enumerate(zip(ordered, keep_scores)):
            labels, conf = ctc_greedy_decode(log_probs[i], blank=self.charset.blank)
            detections.append(Detection(polygon=poly, score=float(score),
                                        embedding=embeds[i],
                                        text=self.charset.decode(labels),
                                        text_confidence=float(conf)))
        return detections

    # -- whole video -----------------------------------------------------
    def run(self, frames: Iterable[np.ndarray], reset: bool = True,
            progress: bool = False) -> List[dict]:
        tracker = VideoTextTracker(self.cfg.tracker)
        if reset:
            tracker.reset()
        for i, frame in enumerate(frames):
            tracker.update(i, self.detect_frame(frame))
            if progress and i % 25 == 0:
                print(f"  frame {i}", flush=True)
        return tracker.finalise()

    def run_video_file(self, path: str | Path, max_frames: Optional[int] = None,
                       progress: bool = False) -> Tuple[List[dict], int]:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise FileNotFoundError(f"cannot open video {path}")
        tracker = VideoTextTracker(self.cfg.tracker)
        count = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok or (max_frames is not None and count >= max_frames):
                    break
                tracker.update(count, self.detect_frame(frame))
                if progress and count % 25 == 0:
                    print(f"  frame {count}", flush=True)
                count += 1
        finally:
            cap.release()
        return tracker.finalise(), count


def draw_trajectories(frame: np.ndarray, trajectories: Sequence[dict], frame_idx: int,
                      show_text: bool = True) -> np.ndarray:
    """Overlay the trajectories active at ``frame_idx``."""
    out = frame.copy()
    for traj in trajectories:
        poly = traj["frames"].get(frame_idx) or traj["frames"].get(str(frame_idx))
        if poly is None:
            continue
        pts = np.asarray(poly, np.float32).astype(np.int32)
        tid = int(traj["track_id"])
        colour = ((tid * 67) % 255, (tid * 131) % 255, (tid * 197) % 255)
        cv2.polylines(out, [pts], True, colour, 2)
        if show_text and traj.get("text"):
            org = (int(pts[:, 0].min()), max(int(pts[:, 1].min()) - 4, 10))
            label = f"{tid}:{traj['text']}"
            cv2.putText(out, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
            cv2.putText(out, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 1)
    return out


def trajectories_to_frames(trajectories: Sequence[dict]) -> Dict[int, List[dict]]:
    """Invert the trajectory representation into per-frame instances."""
    per_frame: Dict[int, List[dict]] = {}
    for traj in trajectories:
        for frame, poly in traj["frames"].items():
            per_frame.setdefault(int(frame), []).append(
                {"polygon": poly, "text": traj.get("text", ""),
                 "track_id": traj["track_id"], "score": traj.get("score", 1.0)})
    return per_frame
