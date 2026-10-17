"""Generic blocked heat-bath (Gibbs) sweeps for lattice targets.

Uses only target.log_score, so it works for any DiscreteTarget with a
`lattice_shape` attribute and nearest-neighbour interactions; `scale`
rescales the inverse temperature (annealing).  Sites are updated one colour
class at a time (`dds.utils.lattice.independent_groups`): two colours on an
all-even torus, three when a side is odd -- the checkerboard alone would be
wrong there, see that module.
"""

from __future__ import annotations

import torch
from torch import Tensor

from dds.utils.lattice import independent_groups

_masks_cache: dict = {}


def colour_masks(lattice_shape: tuple[int, ...], device) -> list[Tensor]:
    """One boolean (D,) mask per colour class of the torus, cached."""
    key = (tuple(lattice_shape), str(device))
    if key not in _masks_cache:
        colour = torch.from_numpy(independent_groups(lattice_shape).reshape(-1)).to(device)
        _masks_cache[key] = [colour == c for c in range(int(colour.max().item()) + 1)]
    return _masks_cache[key]


@torch.no_grad()
def heat_bath_sweeps(target, x: Tensor, n_sweeps: int = 1, scale: float = 1.0,
                     generator=None, cond: Tensor | None = None) -> Tensor:
    """Exact Gibbs sweeps of nu^scale via single-site conditionals.

    Sites of one colour class are updated together, which is exact when they
    are conditionally independent given the rest.  A target for which that is
    false -- a fluid, an embedded-atom alloy -- declares
    ``colour_independent = False`` and is routed to its own sequential sweep
    instead.  Updating a whole colour class of a fluid from one-at-a-time
    conditionals is not a Gibbs sweep and does not leave the target invariant,
    so the choice is made by the target rather than left to the caller.
    """
    if not getattr(target, "colour_independent", True):
        from dds.targets import get_mcmc

        mc = get_mcmc(target, device=x.device)
        if not hasattr(mc, "sweep"):
            raise NotImplementedError(
                f"{type(target).__name__} declares colour_independent=False "
                f"but its sampler {type(mc).__name__} has no `sweep`")
        for _ in range(n_sweeps):
            x = mc.sweep(x, scale=scale, cond=cond, generator=generator)
        return x
    masks = colour_masks(tuple(target.lattice_shape), x.device)
    for _ in range(n_sweeps):
        for m in masks:
            logits = scale * target.log_score(x, cond)  # (B, D, N)
            new = torch.distributions.Categorical(logits=logits).sample()
            x = torch.where(m.unsqueeze(0), new, x)
    return x
