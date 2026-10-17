"""Time schedules.

* `InterpolantSchedule`: monotone beta_t with beta_0 = 0, beta_1 = 1, used as
  the target-channel weight of the uniform interpolant and as the reveal
  probability a_t of the masked interpolant.
* `NoiseSchedule`: rate scale gamma(t) with its time integral, used for the
  reference CTMC of ASBS-style methods.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


# ----------------------------------------------------------------------
class InterpolantSchedule:
    """beta_t on [0,1] with beta(0)=0, beta(1)=1 and derivative."""

    def __init__(self, kind: str = "quadratic"):
        self.kind = kind

    def __call__(self, t: Tensor) -> Tensor:
        if self.kind == "linear":
            return t
        if self.kind == "quadratic":
            return t**2
        if self.kind == "cosine":
            return 0.5 * (1.0 - torch.cos(math.pi * t))
        if self.kind == "sqrt":
            return torch.sqrt(t.clamp_min(0))
        raise ValueError(self.kind)

    def deriv(self, t: Tensor) -> Tensor:
        if self.kind == "linear":
            return torch.ones_like(t)
        if self.kind == "quadratic":
            return 2.0 * t
        if self.kind == "cosine":
            return 0.5 * math.pi * torch.sin(math.pi * t)
        if self.kind == "sqrt":
            return 0.5 / torch.sqrt(t.clamp_min(1e-8))
        raise ValueError(self.kind)


# ----------------------------------------------------------------------
class NoiseSchedule:
    """gamma(t) >= 0 and its integral int_s^t gamma(u) du."""

    def __init__(self, kind: str = "const", gamma: float = 1.0, alpha: float = 0.5):
        self.kind = kind
        self.gamma = gamma
        self.alpha = alpha

    def __call__(self, t: Tensor) -> Tensor:
        if self.kind == "const":
            return torch.full_like(t, self.gamma)
        if self.kind == "log_linear":
            return self.gamma / (self.alpha + t)
        raise ValueError(self.kind)

    def integral(self, s: Tensor, t: Tensor) -> Tensor:
        if self.kind == "const":
            return self.gamma * (t - s)
        if self.kind == "log_linear":
            return self.gamma * torch.log((self.alpha + t) / (self.alpha + s))
        raise ValueError(self.kind)
