"""Method registry."""

from __future__ import annotations

from dds.methods.asbs import ASBS
from dds.methods.base import Method
from dds.methods.dsi import DSI
from dds.methods.frogs import FrOGS
from dds.methods.leaps import LEAPS

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
