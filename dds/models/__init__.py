"""Network architectures.

All models share the interface

    forward(x: LongTensor (B, D), t: Tensor (B,) | None) -> (B, D, out_channels)

where D = prod(lattice_shape) and x takes values in [0, vocab_size).
`vocab_size` may exceed the target alphabet (e.g. +1 for a mask token).
Models built with `require_time=False` ignore/forbid the time argument.
"""

from __future__ import annotations

from dds.models.pixelcnn import PixelCNN
from dds.models.rescnn import PeriodicResCNN


def build_model(
    name: str,
    lattice_shape: tuple[int, ...],
    vocab_size: int,
    out_channels: int,
    require_time: bool = True,
    **kwargs,
):
    """Factory. `name` selects the architecture: `rescnn` is the backbone of
    every method that takes the model group, `pixelcnn` the autoregressive
    prior of FrOGS."""
    lattice_shape = tuple(lattice_shape)
    common = dict(
        lattice_shape=lattice_shape,
        vocab_size=vocab_size,
        out_channels=out_channels,
        require_time=require_time,
    )
    if name == "rescnn":
        return PeriodicResCNN(**common, **kwargs)
    if name == "pixelcnn":
        return PixelCNN(**common, **kwargs)
    raise ValueError(f"unknown model '{name}'")
