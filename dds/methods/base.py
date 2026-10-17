"""Common interface for all sampling methods.

Every method exposes:

* step()                    — one training step; returns a metrics dict
* sample(n)                 — n approximate target samples, shape (n, D)
* sample_with_weights(n)    — (x, log_w) with E[exp(log_w)] = Z whenever the
                              method admits an unbiased estimator (see each
                              method's docstring for caveats)
* state_dict()/load_state_dict()

Methods handle their own inner scheduling (buffer refreshes, phases, ...)
inside step(), keyed on an internal step counter, so the outer training
loop is method-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor

from dds.targets.base import DiscreteTarget


class Method(ABC):
    #: whether sample_with_weights returns exactly unbiased log-weights
    unbiased_log_Z: bool = True

    def __init__(self, target: DiscreteTarget, device: str | torch.device = "cpu"):
        self.target = target
        self.device = torch.device(device)
        self.step_count = 0

    @abstractmethod
    def step(self) -> dict:
        ...

    @abstractmethod
    def sample_with_weights(self, n: int) -> tuple[Tensor, Tensor]:
        ...

    @torch.no_grad()
    def sample(self, n: int) -> Tensor:
        return self.sample_with_weights(n)[0]

    # ------------------------------------------------------------------
    def state_dict(self) -> dict:
        return {"step_count": self.step_count}

    def load_state_dict(self, state: dict):
        self.step_count = state["step_count"]

    # ------------------------------------------------------------------
    def anneal_scale(self) -> float:
        """Inverse-temperature ramp: scale in (0, 1] multiplying beta."""
        n = getattr(self, "anneal_steps", 0)
        if n and self.step_count < n:
            start = getattr(self, "anneal_start", 0.5)
            return start + (1.0 - start) * self.step_count / n
        return 1.0

    # ------------------------------------------------------------------
    def supports_smc(self) -> bool:
        """Can this sampler be resampled during generation?

        Two things are needed: a sequential sampler, and a target that can
        score the part of a configuration that has been built.  Subclasses
        whose own reveal order adds a condition override this.

        The evaluation asks this once rather than calling and catching, so a
        run that cannot be resampled stays quiet instead of raising the same
        message at every evaluation for the length of the run.
        """
        if not hasattr(self, "sample_with_weights_smc"):
            return False
        return (hasattr(self.target, "partial_energy")
                or hasattr(self.target, "partial_energy_indices"))
