"""Exponential moving average of model parameters."""

from __future__ import annotations

import torch


class EMA:
    """Shadow-parameter EMA with store/copy_to/restore, à la pytorch_ema."""

    def __init__(self, parameters, decay: float = 0.999, use_num_updates: bool = True):
        self.decay = decay
        self.num_updates = 0 if use_num_updates else None
        self.shadow = [p.detach().clone() for p in parameters]
        self._backup: list[torch.Tensor] | None = None

    def update(self, parameters):
        decay = self.decay
        if self.num_updates is not None:
            self.num_updates += 1
            decay = min(decay, (1 + self.num_updates) / (10 + self.num_updates))
        with torch.no_grad():
            for s, p in zip(self.shadow, parameters):
                s.mul_(decay).add_(p.detach(), alpha=1 - decay)

    def copy_to(self, parameters):
        with torch.no_grad():
            for s, p in zip(self.shadow, parameters):
                p.copy_(s)

    def store(self, parameters):
        self._backup = [p.detach().clone() for p in parameters]

    def restore(self, parameters):
        assert self._backup is not None, "call store() before restore()"
        with torch.no_grad():
            for b, p in zip(self._backup, parameters):
                p.copy_(b)
        self._backup = None

    def state_dict(self):
        return {"decay": self.decay, "num_updates": self.num_updates, "shadow": self.shadow}

    def load_state_dict(self, state):
        self.decay = state["decay"]
        self.num_updates = state["num_updates"]
        self.shadow = [s.clone() for s in state["shadow"]]
