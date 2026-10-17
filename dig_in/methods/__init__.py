"""Method registry."""

from __future__ import annotations

from dig_in.methods.asbs import ASBS
from dig_in.methods.base import Method
from dig_in.methods.dsi import DSI
from dig_in.methods.frogs import FrOGS
from dig_in.methods.leaps import LEAPS

_REGISTRY = {
    "dsi": DSI,
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


__all__ = ["Method", "DSI", "LEAPS", "FrOGS", "ASBS", "build_method"]
