"""Clip datasets: real (prepared) videos, synthetic video, and synthetic stills."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from ..utils.charset import Charset
from ..utils.polygon import (is_valid_poly, order_polygon_for_text, poly_area,
                             polygon_to_control_points)
from .schema import VideoAnnotation, index_dataset
from .synth_static import SyntheticImageGenerator, SynthConfig
from .synth_video import SyntheticVideoGenerator, VideoSynthConfig
from .targets import build_db_targets
from .transforms import AugConfig, ClipAugmentor, clip_polygon_to_image, transform_points

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], np.float32) * 255.0
IMAGENET_STD = np.array([0.229, 0.224, 0.225], np.float32) * 255.0


def normalise_image(img: np.ndarray) -> np.ndarray:
    """BGR uint8 HWC -> normalised float CHW.

    The ImageNet statistics are kept even though no ImageNet weights are used:
    they are simply a reasonable centring for natural images, and matching them
    costs nothing.  Any fixed mean/std works as long as it is consistent between
    training and inference.
    """
    rgb = img[:, :, ::-1].astype(np.float32)
    return ((rgb - IMAGENET_MEAN) / IMAGENET_STD).transpose(2, 0, 1)


@dataclass
class ClipConfig:
    clip_len: int = 4
    frame_stride: int = 1              # sample every Nth frame
    random_stride: bool = True         # jitter the stride to vary apparent speed
    max_instances_per_frame: int = 64
    max_text_len: int = 25
    num_control_points: int = 8        # per boundary, so 2K = 16 points total
    min_kept_area: float = 0.55        # below this, an instance becomes ignore


class _ClipBuilder:
    """Shared logic: warp frames, build targets, pack instance tensors."""

    def __init__(self, clip_cfg: ClipConfig, aug_cfg: AugConfig, charset: Charset,
                 training: bool):
        self.clip_cfg = clip_cfg
        self.aug_cfg = aug_cfg
        self.charset = charset
        self.training = training

    def build(self, frames: Sequence[np.ndarray],
              per_frame_instances: Sequence[Sequence[dict]],
              rng: random.Random) -> Dict[str, torch.Tensor]:
        cc = self.clip_cfg
        aug = ClipAugmentor(self.aug_cfg, training=self.training, rng=rng)

        src_h, src_w = frames[0].shape[:2]
        centres = [np.asarray(i["polygon"], np.float32).mean(axis=0)
                   for inst in per_frame_instances for i in inst if not i.get("ignore")]
        keypoints = np.stack(centres) if centres else None
        base_H, out_hw = aug.base_homography((src_h, src_w), keypoints)
        out_h, out_w = out_hw

        images: List[np.ndarray] = []
        det_targets: List[Dict[str, np.ndarray]] = []
        inst_ctrl: List[np.ndarray] = []
        inst_frame: List[int] = []
        inst_track: List[int] = []
        inst_labels: List[List[int]] = []
        inst_lengths: List[int] = []
        inst_attn: List[List[int]] = []
        inst_text_hash: List[int] = []
        inst_has_text: List[bool] = []

        for t, (frame, instances) in enumerate(zip(frames, per_frame_instances)):
            H = aug.frame_homography(base_H, out_hw)
            warped = aug.warp_frame(frame, H, out_hw)
            warped = aug.photometric(warped)
            images.append(normalise_image(warped))

            polys: List[np.ndarray] = []
            ignores: List[bool] = []
            for inst in instances:
                poly = np.asarray(inst["polygon"], np.float32)
                if not is_valid_poly(poly):
                    continue
                poly = transform_points(poly, H)
                clipped, kept = clip_polygon_to_image(poly, out_h, out_w)
                if kept <= 0.02 or poly_area(clipped) < 4.0:
                    continue
                ignore = bool(inst.get("ignore", False)) or kept < cc.min_kept_area
                polys.append(clipped)
                ignores.append(ignore)

                if ignore or len(inst_ctrl) >= cc.max_instances_per_frame * len(frames):
                    continue
                try:
                    ordered = order_polygon_for_text(clipped)
                    ctrl = polygon_to_control_points(ordered, cc.num_control_points)
                except Exception:
                    continue

                text = str(inst.get("text", ""))
                encodable = bool(text) and self.charset.is_encodable(text)
                labels = self.charset.encode(text)[: cc.max_text_len] if encodable else []
                inst_ctrl.append(ctrl)
                inst_frame.append(t)
                inst_track.append(int(inst.get("track_id", -1)))
                inst_labels.append(labels + [0] * (cc.max_text_len - len(labels)))
                inst_lengths.append(len(labels))
                inst_attn.append(self.charset.encode_attn(text, cc.max_text_len)
                                 if encodable else [0] * cc.max_text_len)
                inst_text_hash.append(hash(self.charset.normalise(text)) % (2 ** 31)
                                      if text else -1 - len(inst_text_hash))
                inst_has_text.append(encodable)

            det_targets.append(build_db_targets(polys, ignores, out_h, out_w))

        def stack_map(key: str) -> torch.Tensor:
            return torch.from_numpy(np.stack([d[key] for d in det_targets]))

        n = len(inst_ctrl)
        return {
            "images": torch.from_numpy(np.stack(images)),
            "shrink_map": stack_map("shrink_map"),
            "shrink_mask": stack_map("shrink_mask"),
            "thresh_map": stack_map("thresh_map"),
            "thresh_mask": stack_map("thresh_mask"),
            "inst_ctrl": (torch.from_numpy(np.stack(inst_ctrl)) if n else
                          torch.zeros((0, 2 * cc.num_control_points, 2))),
            "inst_frame": torch.tensor(inst_frame, dtype=torch.long),
            "inst_track": torch.tensor(inst_track, dtype=torch.long),
            "inst_labels": (torch.tensor(inst_labels, dtype=torch.long) if n else
                            torch.zeros((0, cc.max_text_len), dtype=torch.long)),
            "inst_lengths": torch.tensor(inst_lengths, dtype=torch.long),
            "inst_attn": (torch.tensor(inst_attn, dtype=torch.long) if n else
                          torch.zeros((0, cc.max_text_len), dtype=torch.long)),
            "inst_text_hash": torch.tensor(inst_text_hash, dtype=torch.long),
            "inst_has_text": torch.tensor(inst_has_text, dtype=torch.bool),
        }


class VideoClipDataset(Dataset):
    """Clips sampled from a prepared dataset root (see ``data/schema.py``)."""

    def __init__(self, root: str | Path, charset: Charset,
                 clip_cfg: Optional[ClipConfig] = None,
                 aug_cfg: Optional[AugConfig] = None, training: bool = True,
                 seed: int = 0):
        self.root = Path(root)
        self.clip_cfg = clip_cfg or ClipConfig()
        self.builder = _ClipBuilder(self.clip_cfg, aug_cfg or AugConfig(), charset, training)
        self.training = training
        self.seed = seed
        self.videos = [VideoAnnotation.from_json(p) for p in index_dataset(root)]
        if not self.videos:
            raise RuntimeError(f"no annotations under {root}")
        # Index of (video, start_frame) clip origins.
        self.index: List[Tuple[int, int]] = []
        span = self.clip_cfg.clip_len * self.clip_cfg.frame_stride
        for vi, video in enumerate(self.videos):
            n = len(video.frames)
            if n == 0:
                continue
            step = 1 if training else max(span, 1)
            for start in range(0, max(n - span + 1, 1), step):
                self.index.append((vi, start))

    def __len__(self) -> int:
        return len(self.index)

    def _frame_path(self, video: VideoAnnotation, frame_idx: int) -> Path:
        return self.root / "frames" / video.video_id / f"{frame_idx:06d}.jpg"

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        vi, start = self.index[i]
        video = self.videos[vi]
        rng = random.Random((self.seed * 1_000_003 + i) if self.training else i)
        cc = self.clip_cfg
        stride = cc.frame_stride
        if self.training and cc.random_stride and stride > 0:
            stride = rng.randint(1, max(stride, 1) * 2)

        idxs = [min(start + k * stride, len(video.frames) - 1) for k in range(cc.clip_len)]
        frames, instances = [], []
        for fi in idxs:
            frame_ann = video.frames[fi]
            img = cv2.imread(str(self._frame_path(video, frame_ann.frame_idx)), cv2.IMREAD_COLOR)
            if img is None:
                img = np.zeros((video.height, video.width, 3), np.uint8)
            frames.append(img)
            instances.append([{"polygon": inst.polygon, "text": inst.text,
                               "track_id": inst.track_id, "ignore": inst.ignore}
                              for inst in frame_ann.instances])
        return self.builder.build(frames, instances, rng)


class SyntheticVideoDataset(Dataset):
    """On-the-fly synthetic clips -- stage 2 of the curriculum, no downloads."""

    def __init__(self, charset: Charset, length: int = 10_000,
                 video_cfg: Optional[VideoSynthConfig] = None,
                 clip_cfg: Optional[ClipConfig] = None,
                 aug_cfg: Optional[AugConfig] = None,
                 background_dir: Optional[str] = None, seed: int = 0,
                 training: bool = True):
        self.length = length
        self.clip_cfg = clip_cfg or ClipConfig()
        self.video_cfg = video_cfg or VideoSynthConfig(
            num_frames=self.clip_cfg.clip_len)
        self.video_cfg.num_frames = max(self.video_cfg.num_frames, self.clip_cfg.clip_len)
        self.builder = _ClipBuilder(self.clip_cfg, aug_cfg or AugConfig(), charset, training)
        self.background_dir = background_dir
        self.seed = seed

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        seed = self.seed * 1_000_003 + i
        gen = SyntheticVideoGenerator(self.video_cfg, background_dir=self.background_dir,
                                      seed=seed)
        frames, anns = gen.generate()
        rng = random.Random(seed)
        start = rng.randint(0, max(len(frames) - self.clip_cfg.clip_len, 0))
        sl = slice(start, start + self.clip_cfg.clip_len)
        return self.builder.build(frames[sl], anns[sl], rng)


class SyntheticStaticDataset(Dataset):
    """Single-frame 'clips' of synthetic stills -- stage 1 of the curriculum.

    Treating a still as a length-1 clip means stage 1 and stage 2 share one
    training loop; only the tracking loss is inactive (a length-1 clip has no
    cross-frame positives, and ``ContrastiveTrackLoss`` returns zero for it).
    """

    def __init__(self, charset: Charset, length: int = 100_000,
                 synth_cfg: Optional[SynthConfig] = None,
                 clip_cfg: Optional[ClipConfig] = None,
                 aug_cfg: Optional[AugConfig] = None,
                 background_dir: Optional[str] = None, seed: int = 0,
                 training: bool = True):
        self.length = length
        self.synth_cfg = synth_cfg or SynthConfig()
        cc = clip_cfg or ClipConfig()
        cc.clip_len = 1
        self.clip_cfg = cc
        self.builder = _ClipBuilder(cc, aug_cfg or AugConfig(), charset, training)
        self.background_dir = background_dir
        self.seed = seed

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        seed = self.seed * 1_000_003 + i
        gen = SyntheticImageGenerator(self.synth_cfg, background_dir=self.background_dir,
                                      seed=seed)
        img, polys, texts = gen.generate()
        instances = [{"polygon": p.tolist(), "text": t, "track_id": k, "ignore": False}
                     for k, (p, t) in enumerate(zip(polys, texts))]
        return self.builder.build([img], [instances], random.Random(seed))


def collate_clips(batch: Sequence[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Batch clips, remapping frame and track indices into the flat batch.

    ``inst_frame`` becomes an index into the flattened ``(B*T)`` frame axis and
    ``inst_track`` is offset per clip, so the contrastive loss never pairs
    instances from different clips (they are genuinely different identities and
    treating them as negatives is correct, but treating them as *positives*
    because two clips happened to reuse track id 3 would be a silent disaster).
    """
    images = torch.stack([b["images"] for b in batch])          # (B, T, 3, H, W)
    out: Dict[str, torch.Tensor] = {"images": images}
    for key in ("shrink_map", "shrink_mask", "thresh_map", "thresh_mask"):
        out[key] = torch.stack([b[key] for b in batch])          # (B, T, 1, H, W)

    frame_offsets, track_offsets = [], []
    frame_cursor = 0
    track_cursor = 0
    for b in batch:
        t = b["images"].shape[0]
        frame_offsets.append(frame_cursor)
        track_offsets.append(track_cursor)
        frame_cursor += t
        local = b["inst_track"]
        track_cursor += int(local.max()) + 1 if local.numel() else 0

    for key in ("inst_ctrl", "inst_labels", "inst_attn", "inst_lengths",
                "inst_text_hash", "inst_has_text"):
        out[key] = torch.cat([b[key] for b in batch], dim=0)
    out["inst_frame"] = torch.cat([b["inst_frame"] + off
                                   for b, off in zip(batch, frame_offsets)], dim=0)
    out["inst_track"] = torch.cat([
        torch.where(b["inst_track"] >= 0, b["inst_track"] + off, b["inst_track"])
        for b, off in zip(batch, track_offsets)], dim=0)
    out["clip_index"] = torch.cat([
        torch.full((b["inst_frame"].numel(),), i, dtype=torch.long)
        for i, b in enumerate(batch)], dim=0)
    return out
