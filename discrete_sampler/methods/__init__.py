"""Method registry."""

from __future__ import annotations

from discrete_sampler.methods.asbs import ASBS
from discrete_sampler.methods.base import Method
from discrete_sampler.methods.dig_in import DIGIN
from discrete_sampler.methods.frogs import FrOGS
from discrete_sampler.methods.leaps import LEAPS

_REGISTRY = {
    "dig_in": DIGIN,
    "leaps": LEAPS,
    "frogs": FrOGS,
    "asbs": ASBS,
}


def build_method(name: str, target, cfg: dict, device: str = "cpu") -> Method:
    """cfg: keyword arguments of the method class (name key removed)."""
    cfg = dict(cfg)
    cfg.pop("name", None)
    if name not in _REGISTRY:
        raise ValueError(f"unknown method '{name}'; available: {list(_REGISTRY)}")
    return _REGISTRY[name](target, device=device, **cfg)


__all__ = ["Method", "DIGIN", "LEAPS", "FrOGS", "ASBS", "build_method"]
