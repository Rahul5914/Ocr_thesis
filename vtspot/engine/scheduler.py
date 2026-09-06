"""Learning-rate schedules for from-scratch training.

Linear warmup then cosine decay.  Warmup is not optional here: with random
weights the first gradients are large and poorly conditioned, and a full-rate
step on them moves the parameters somewhere the model never recovers from.  The
symptom is a loss that spikes and then plateaus at a high value for the rest of
the run, which is easy to misread as "the architecture is wrong".
"""

from __future__ import annotations

import math
from typing import List

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def warmup_cosine(optimizer: Optimizer, warmup_steps: int, total_steps: int,
                  min_lr_ratio: float = 0.01, last_epoch: int = -1) -> LambdaLR:
    warmup_steps = max(int(warmup_steps), 1)
    total_steps = max(int(total_steps), warmup_steps + 1)

    def fn(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return LambdaLR(optimizer, fn, last_epoch=last_epoch)


def warmup_poly(optimizer: Optimizer, warmup_steps: int, total_steps: int,
                power: float = 0.9, min_lr_ratio: float = 0.0) -> LambdaLR:
    """Polynomial decay -- the DBNet convention, kept for comparability."""
    warmup_steps = max(int(warmup_steps), 1)
    total_steps = max(int(total_steps), warmup_steps + 1)

    def fn(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max((1.0 - min(progress, 1.0)) ** power, min_lr_ratio)

    return LambdaLR(optimizer, fn)


def build_scheduler(name: str, optimizer: Optimizer, warmup_steps: int,
                    total_steps: int, **kwargs) -> LambdaLR:
    if name == "cosine":
        return warmup_cosine(optimizer, warmup_steps, total_steps, **kwargs)
    if name == "poly":
        return warmup_poly(optimizer, warmup_steps, total_steps, **kwargs)
    raise ValueError(f"unknown scheduler {name!r}")


def param_groups(model, weight_decay: float = 0.01) -> List[dict]:
    """Exclude norms, biases and positional embeddings from weight decay.

    Decaying a LayerNorm gain or a bias shrinks it towards zero for no benefit
    and measurably hurts small models -- which this is.
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".pos") or "log_var" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]
