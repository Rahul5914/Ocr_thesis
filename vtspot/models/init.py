"""Weight initialisation for training without any pretrained weights.

Two schemes are provided.

``kaiming``
    ``Var(W) = 2 / fan_in`` i.e. ``std = sqrt(2 / fan_in)``.  Note the variance /
    standard-deviation distinction: ``N(0, sigma^2)`` is parameterised by the
    *variance*, and PyTorch's ``normal_(mean, std)`` takes the *standard
    deviation*.  Writing ``W ~ N(0, sqrt(2/fan_in))`` conflates the two and gives
    a distribution whose variance is ``sqrt(2/fan_in)`` -- for a 3x3x256 layer
    that is 2304x too wide.

``fixup``
    Fixup (Zhang et al., ICLR 2019) removes normalisation from residual nets.
    It is *four* rules, not one, and the scaling rule alone diverges:

      1. initialise the classifier and the **last** layer of every residual
         branch to zero, so each branch starts as the identity;
      2. initialise every other layer with He, then scale the weights inside
         residual branches by ``L ** (-1 / (2m - 2))`` where ``L`` is the number
         of residual branches and ``m`` the number of layers per branch;
      3. add a scalar bias before every conv / linear / elementwise op;
      4. add a scalar multiplier before every conv / linear.

    Fixup also loses batch-norm's regularisation effect, so it needs stronger
    augmentation or mixup to reach the same accuracy.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn


def kaiming_init_module(module: nn.Module, nonlinearity: str = "relu",
                        a: float = 0.0) -> None:
    """He-initialise a single conv/linear layer in fan_in mode."""
    if isinstance(module, (nn.Conv2d, nn.Conv1d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight, a=a, mode="fan_in",
                                nonlinearity=nonlinearity)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def init_weights(model: nn.Module, nonlinearity: str = "relu", a: float = 0.0) -> nn.Module:
    """Apply He init to every layer of ``model``.

    Modules exposing ``custom_init`` (the Fixup blocks, the DB head's biased
    output layer, transformer blocks) initialise themselves and are skipped.
    """
    custom_roots = [name for name, m in model.named_modules()
                    if getattr(m, "custom_init", False)]

    def under_custom_root(name: str) -> bool:
        # A module that initialises itself owns its *whole subtree*: re-running
        # He on its children would silently undo Fixup's zeroed last conv and
        # its branch scaling.
        return any(name == r or name.startswith(r + ".") for r in custom_roots)

    for name, m in model.named_modules():
        if under_custom_root(name):
            continue
        kaiming_init_module(m, nonlinearity=nonlinearity, a=a)
    for name, m in model.named_modules():
        if getattr(m, "custom_init", False) and hasattr(m, "reset_parameters_custom"):
            m.reset_parameters_custom()
    return model


def apply_fixup_scaling(branches: Iterable[nn.Module], num_branches: int,
                        layers_per_branch: int = 2) -> None:
    """Rules 1 and 2 of Fixup, applied to a list of residual branches.

    Every branch must expose ``conv_layers`` -- its weight layers in forward
    order.  The last one is zeroed; the rest are scaled by
    ``L ** (-1 / (2m - 2))``.
    """
    if layers_per_branch < 2:
        raise ValueError("Fixup scaling is undefined for m < 2 (division by zero)")
    scale = float(num_branches) ** (-1.0 / (2.0 * layers_per_branch - 2.0))
    for branch in branches:
        convs = list(branch.conv_layers)
        for conv in convs[:-1]:
            nn.init.kaiming_normal_(conv.weight, mode="fan_in", nonlinearity="relu")
            conv.weight.data.mul_(scale)
        nn.init.zeros_(convs[-1].weight)     # branch starts as the identity


def init_prior_bias(layer: nn.Module, prior_prob: float = 0.01,
                    weight_std: float = 0.01) -> None:
    """Bias a sigmoid output so it starts at ``prior_prob``.

    Text pixels are a small fraction of the image.  Starting the probability map
    at 0.5 makes the first thousand iterations pure background suppression,
    which is exactly the phase where a from-scratch model is most fragile.

    The bias alone is not enough: a He-initialised output layer contributes
    enough variance to swamp it (empirically the map still starts near 0.13 for
    a nominal prior of 0.02).  Shrinking the output weights so the bias
    dominates is what actually realises the prior -- the same pairing RetinaNet
    uses for its focal-loss prior.
    """
    bias = -math.log((1.0 - prior_prob) / prior_prob)
    if getattr(layer, "weight", None) is not None:
        nn.init.normal_(layer.weight, mean=0.0, std=weight_std)
    if getattr(layer, "bias", None) is not None:
        nn.init.constant_(layer.bias, bias)


@torch.no_grad()
def activation_variance_report(model: nn.Module, sample: torch.Tensor,
                               max_layers: int = 40) -> list[tuple[str, float, float]]:
    """Record per-layer activation ``(mean, var)`` for one forward pass.

    This is the cheapest possible sanity check that initialisation is sane: with
    correct He init the variance should stay within roughly an order of
    magnitude across depth.  A geometric decay towards zero means the signal is
    vanishing (Xavier on a ReLU net); a blow-up means the scale is too large.
    """
    stats: list[tuple[str, float, float]] = []
    hooks = []

    def make_hook(name):
        def hook(_m, _inp, out):
            if isinstance(out, torch.Tensor) and out.is_floating_point():
                stats.append((name, float(out.mean()), float(out.var())))
        return hook

    targets = [(n, m) for n, m in model.named_modules()
               if isinstance(m, (nn.Conv2d, nn.Linear))][:max_layers]
    for name, module in targets:
        hooks.append(module.register_forward_hook(make_hook(name)))
    was_training = model.training
    model.eval()
    try:
        model(sample)
    finally:
        for h in hooks:
            h.remove()
        model.train(was_training)
    return stats
