"""FIFO replay buffer for terminal states (and optional per-sample extras)."""

from __future__ import annotations

import torch
from torch import Tensor


class ReplayBuffer:
    """Fixed-capacity FIFO buffer holding states (n, D) plus optional extras."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.states: Tensor | None = None
        self.extras: dict[str, Tensor] = {}

    def __len__(self) -> int:
        return 0 if self.states is None else self.states.shape[0]

    def add(self, x: Tensor, **extras: Tensor):
        x = x.detach()
        extras = {k: v.detach() for k, v in extras.items()}
        if self.states is None:
            self.states = x
            self.extras = extras
        else:
            self.states = torch.cat([self.states, x], dim=0)
            for k in self.extras:
                self.extras[k] = torch.cat([self.extras[k], extras[k]], dim=0)
        if self.states.shape[0] > self.capacity:
            keep = self.states.shape[0] - self.capacity
            self.states = self.states[keep:]
            for k in self.extras:
                self.extras[k] = self.extras[k][keep:]

    def sample(self, n: int, generator=None) -> tuple[Tensor, dict[str, Tensor]]:
        assert self.states is not None and len(self) > 0, "buffer is empty"
        idx = torch.randint(
            0, self.states.shape[0], (n,), device=self.states.device, generator=generator
        )
        return self.states[idx], {k: v[idx] for k, v in self.extras.items()}
