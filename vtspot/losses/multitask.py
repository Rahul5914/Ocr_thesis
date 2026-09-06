"""Multi-task weighting for detection + recognition + tracking."""

from __future__ import annotations

from typing import Dict, Iterable

import torch
import torch.nn as nn


class UncertaintyWeighting(nn.Module):
    """Kendall et al. homoscedastic uncertainty weighting.

    ``L = sum_i ( exp(-s_i) * L_i + 0.5 * s_i )`` with ``s_i = log(sigma_i^2)``
    learnt.

    The failure mode this guards against: nothing stops the optimiser from
    driving ``s_i`` up without bound for a task whose loss stays high, which
    silently switches that task off -- and tracking, the hardest task here, is
    exactly the one it will pick.  You discover this after a long run when
    IDF1 is at zero.  ``s`` is therefore clamped to a range, which caps the
    weight ratio between the easiest and hardest task at ``exp(2*s_max)``.
    """

    def __init__(self, task_names: Iterable[str], init_log_var: float = 0.0,
                 s_min: float = -3.0, s_max: float = 3.0):
        super().__init__()
        self.task_names = list(task_names)
        self.log_var = nn.ParameterDict(
            {name: nn.Parameter(torch.tensor(float(init_log_var)))
             for name in self.task_names})
        self.s_min, self.s_max = s_min, s_max

    def forward(self, losses: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, Dict[str, float]]:
        total = None
        stats: Dict[str, float] = {}
        for name in self.task_names:
            if name not in losses:
                continue
            s = self.log_var[name].clamp(self.s_min, self.s_max)
            term = torch.exp(-s) * losses[name] + 0.5 * s
            total = term if total is None else total + term
            stats[f"w_{name}"] = float(torch.exp(-s).detach())
        if total is None:
            raise ValueError(f"no known task in {sorted(losses)}; expected {self.task_names}")
        return total, stats


class FixedWeighting(nn.Module):
    """Static ``lambda`` weights -- the safer default for the first runs.

    Learnt weighting is only meaningful once each individual loss is already
    descending; starting a from-scratch run with it makes a divergence
    impossible to attribute.
    """

    def __init__(self, weights: Dict[str, float]):
        super().__init__()
        self.weights = dict(weights)

    def forward(self, losses: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, Dict[str, float]]:
        total = None
        for name, w in self.weights.items():
            if name not in losses:
                continue
            term = w * losses[name]
            total = term if total is None else total + term
        if total is None:
            raise ValueError(f"no known task in {sorted(losses)}; expected {sorted(self.weights)}")
        return total, {f"w_{k}": v for k, v in self.weights.items()}


def build_weighting(cfg: dict) -> nn.Module:
    kind = cfg.get("type", "fixed")
    if kind == "fixed":
        return FixedWeighting(cfg.get("weights", {"loss_det": 1.0, "loss_rec": 1.0,
                                                  "loss_track": 1.0}))
    if kind == "uncertainty":
        return UncertaintyWeighting(cfg.get("tasks", ["loss_det", "loss_rec", "loss_track"]),
                                    s_min=cfg.get("s_min", -3.0),
                                    s_max=cfg.get("s_max", 3.0))
    raise ValueError(f"unknown weighting type {kind!r}")
