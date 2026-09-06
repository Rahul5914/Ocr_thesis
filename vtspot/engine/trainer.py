"""Training loop for the three-task spotter."""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from ..losses.db_loss import DBLoss
from ..losses.multitask import build_weighting
from ..losses.rec_loss import AttentionGuidanceLoss, CTCRecognitionLoss
from ..losses.track_loss import ContrastiveTrackLoss
from ..models.spotter import VideoTextSpotter
from .scheduler import build_scheduler, param_groups


@dataclass
class TrainConfig:
    epochs: int = 10
    steps_per_epoch: Optional[int] = None
    lr: float = 1e-3
    weight_decay: float = 0.01
    warmup_steps: int = 1000
    scheduler: str = "cosine"
    grad_clip: float = 1.0
    amp: bool = True
    amp_dtype: str = "bf16"         # "bf16" on Ampere+ (A4000, A100, RTX 30/40), else "fp16"
    ema_decay: float = 0.999
    log_every: int = 20
    ckpt_dir: str = "checkpoints"
    save_every: int = 1
    keep_last_n: int = 3            # prune older epoch checkpoints; 0 keeps all
    attn_weight: float = 0.5
    db_k_warmup_steps: int = 2000   # steps to ramp the DB steepness k to its final value
    weighting: dict = field(default_factory=lambda: {
        "type": "fixed",
        "weights": {"loss_det": 1.0, "loss_rec": 1.0, "loss_track": 0.5}})
    resume: Optional[str] = None


class ModelEMA:
    """Exponential moving average of the weights.

    Worth more than usual here: from-scratch training on synthetic data is noisy
    step to step, and the averaged weights are consistently better than the last
    iterate -- often by a couple of points of detection F1 for free.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.module = {k: v.detach().clone().float()
                       for k, v in model.state_dict().items()
                       if v.dtype.is_floating_point}
        self.buffers = {k: v.detach().clone() for k, v in model.state_dict().items()
                        if not v.dtype.is_floating_point}

    @torch.no_grad()
    def update(self, model: nn.Module, step: int) -> None:
        # Ramp the decay in: at step 0 the stored copy is random noise, and a
        # 0.999 decay would keep that noise around for thousands of steps.
        d = min(self.decay, (1 + step) / (10 + step))
        for k, v in model.state_dict().items():
            if k in self.module:
                self.module[k].mul_(d).add_(v.detach().float(), alpha=1 - d)
            else:
                self.buffers[k] = v.detach().clone()

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {**self.module, **self.buffers}


class Trainer:
    def __init__(self, model: VideoTextSpotter, cfg: TrainConfig, device: str = "cpu",
                 charset_blank: int = 0, run_config: Optional[dict] = None):
        self.model = model.to(device)
        self.cfg = cfg
        # Stored verbatim in every checkpoint.  A checkpoint that does not
        # record its charset cannot be evaluated: the class indices it emits are
        # meaningless without the alphabet that produced them, and falling back
        # to a default alphabet decodes confident nonsense.
        self.run_config = dict(run_config or {})
        self.device = torch.device(device)
        self.det_loss = DBLoss()
        self.rec_loss = CTCRecognitionLoss(blank=charset_blank)
        self.attn_loss = AttentionGuidanceLoss()
        self.track_loss = ContrastiveTrackLoss()
        self.weighting = build_weighting(cfg.weighting).to(device)

        params = param_groups(self.model, cfg.weight_decay)
        params += [{"params": list(self.weighting.parameters()), "weight_decay": 0.0}]
        self.optimizer = torch.optim.AdamW(params, lr=cfg.lr, betas=(0.9, 0.999))
        self.scheduler = None
        # bfloat16 has fp32's exponent range, so it cannot overflow the way
        # fp16 does and needs no loss scaling.  On Ampere and later (A4000,
        # A100, RTX 30/40) it is the better default: same speed as fp16, none of
        # the "loss went inf at step 300" failure mode -- which matters more
        # than usual here, since a from-scratch run has large early gradients.
        self.amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16,
                          "fp32": torch.float32}.get(cfg.amp_dtype, torch.bfloat16)
        if (self.amp_dtype is torch.bfloat16 and self.device.type == "cuda"
                and not torch.cuda.is_bf16_supported()):
            print("[warn] bf16 unsupported on this GPU, falling back to fp16")
            self.amp_dtype = torch.float16
        # GradScaler is only meaningful for fp16.
        self.scaler = torch.amp.GradScaler(
            self.device.type,
            enabled=cfg.amp and self.device.type == "cuda"
            and self.amp_dtype is torch.float16)
        self.ema = ModelEMA(self.model, cfg.ema_decay) if cfg.ema_decay > 0 else None
        self.step = 0
        self.epoch = 0

    # -- loss ------------------------------------------------------------
    def compute_losses(self, batch: Dict[str, torch.Tensor],
                       out: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # The model already flattened (B, T) into the frame axis; flatten the
        # targets the same way so the two line up.
        det_pred = {"prob_logit": out["det_prob_logit"], "prob": out["det_prob"],
                    "thresh": out["det_thresh"]}
        if "det_binary" in out:
            det_pred["binary"] = out["det_binary"]
        det_target = {k: batch[k].flatten(0, 1) for k in
                      ("shrink_map", "shrink_mask", "thresh_map", "thresh_mask")}
        losses = self.det_loss(det_pred, det_target)

        has_text = batch["inst_has_text"]
        if int(has_text.sum()) > 0:
            losses["loss_rec"] = self.rec_loss(
                out["ctc_logits"][has_text], batch["inst_labels"][has_text],
                batch["inst_lengths"][has_text])
            if "attn_logits" in out:
                losses["loss_attn"] = self.cfg.attn_weight * self.attn_loss(
                    out["attn_logits"][has_text], batch["inst_attn"][has_text])
                losses["loss_rec"] = losses["loss_rec"] + losses["loss_attn"]
        else:
            losses["loss_rec"] = out["ctc_logits"].sum() * 0.0

        tracked = batch["inst_track"] >= 0
        if int(tracked.sum()) > 1:
            losses["loss_track"] = self.track_loss(
                out["embed"][tracked], batch["inst_track"][tracked],
                batch["inst_frame"][tracked], batch["inst_text_hash"][tracked])
        else:
            losses["loss_track"] = out["embed"].sum() * 0.0
        return losses

    # -- loop ------------------------------------------------------------
    def train(self, loader: DataLoader, val_fn=None) -> Dict[str, float]:
        cfg = self.cfg
        steps_per_epoch = cfg.steps_per_epoch or len(loader)
        total_steps = steps_per_epoch * cfg.epochs
        if self.scheduler is None:
            self.scheduler = build_scheduler(cfg.scheduler, self.optimizer,
                                             cfg.warmup_steps, total_steps)
        ckpt_dir = Path(cfg.ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        history = []

        for epoch in range(self.epoch, cfg.epochs):
            self.epoch = epoch
            self.model.train()
            running: Dict[str, float] = {}
            seen = 0
            t0 = time.time()
            for i, batch in enumerate(loader):
                if i >= steps_per_epoch:
                    break
                stats = self.train_step(batch)
                for k, v in stats.items():
                    running[k] = running.get(k, 0.0) + v
                seen += 1
                if self.step % cfg.log_every == 0:
                    msg = " ".join(f"{k}={v / seen:.4f}" for k, v in sorted(running.items())
                                   if k.startswith("loss"))
                    lr = self.optimizer.param_groups[0]["lr"]
                    print(f"[e{epoch} s{self.step}] lr={lr:.2e} {msg}", flush=True)

            summary = {k: v / max(seen, 1) for k, v in running.items()}
            summary["epoch"] = epoch
            summary["seconds"] = time.time() - t0
            if val_fn is not None:
                summary.update(val_fn(self))
            history.append(summary)
            (ckpt_dir / "history.json").write_text(json.dumps(history, indent=2))
            if (epoch + 1) % cfg.save_every == 0:
                self.save(ckpt_dir / f"epoch_{epoch:04d}.pt")
                self._prune_checkpoints(ckpt_dir)
            self.save(ckpt_dir / "last.pt")
        return history[-1] if history else {}

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
        if self.cfg.db_k_warmup_steps > 0:
            self.model.det_head.set_k_progress(self.step / self.cfg.db_k_warmup_steps)
        self.optimizer.zero_grad(set_to_none=True)
        amp_enabled = (self.cfg.amp and self.device.type == "cuda"
                       and self.amp_dtype is not torch.float32)
        with torch.amp.autocast(self.device.type, dtype=self.amp_dtype,
                                enabled=amp_enabled):
            out = self.model(batch)
            losses = self.compute_losses(batch, out)
            total, weights = self.weighting(losses)

        if not torch.isfinite(total):
            # Skip rather than poison every parameter with NaN.  Losing one step
            # is nothing; a NaN that reaches the optimizer ends the run.
            print(f"[warn] non-finite loss at step {self.step}, skipping", flush=True)
            self.optimizer.zero_grad(set_to_none=True)
            self.step += 1
            return {"loss_total": float("nan")}

        self.scaler.scale(total).backward()
        if self.cfg.grad_clip > 0:
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                                       self.cfg.grad_clip)
        else:
            grad_norm = torch.tensor(0.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if self.scheduler is not None:
            self.scheduler.step()
        if self.ema is not None:
            self.ema.update(self.model, self.step)
        self.step += 1

        stats = {k: float(v.detach()) for k, v in losses.items()}
        stats["loss_total"] = float(total.detach())
        stats["grad_norm"] = float(grad_norm)
        stats["db_k"] = float(self.model.det_head.k)
        stats.update(weights)
        return stats

    # -- checkpoints -----------------------------------------------------
    def _prune_checkpoints(self, ckpt_dir: Path) -> None:
        """Keep only the newest ``keep_last_n`` epoch checkpoints.

        Each one is ~195 MB (weights + optimizer state + EMA).  A 20-epoch stage
        writes 3.9 GB, which overruns a free Google Drive and Kaggle's working
        directory quota.  ``last.pt`` is never pruned, so --resume always works.
        """
        if self.cfg.keep_last_n <= 0:
            return
        epochs = sorted(ckpt_dir.glob("epoch_*.pt"))
        for old in epochs[: max(len(epochs) - self.cfg.keep_last_n, 0)]:
            try:
                old.unlink()
            except OSError:
                pass

    def save(self, path: str | Path) -> None:
        payload = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "weighting": self.weighting.state_dict(),
            "step": self.step, "epoch": self.epoch,
            "model_config": dict(self.model.cfg.__dict__),
            "config": {**self.run_config, **dict(self.model.cfg.__dict__)},
        }
        if self.ema is not None:
            payload["ema"] = self.ema.state_dict()
        torch.save(payload, str(path))

    def load(self, path: str | Path, strict: bool = True) -> None:
        ckpt = torch.load(str(path), map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model"], strict=strict)
        if "optimizer" in ckpt and ckpt["optimizer"]:
            self.optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("weighting"):
            self.weighting.load_state_dict(ckpt["weighting"])
        self.step = int(ckpt.get("step", 0))
        self.epoch = int(ckpt.get("epoch", 0))
        if self.ema is not None and "ema" in ckpt:
            for k, v in ckpt["ema"].items():
                if k in self.ema.module:
                    self.ema.module[k] = v.float()
