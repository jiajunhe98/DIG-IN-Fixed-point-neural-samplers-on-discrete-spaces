"""Observable-free Fisher acquisition over the conditioning box.

An amortized sampler has to spend a finite exploration budget across a
continuous family of targets.  Drawing the conditioning uniformly spends most
of it where the target barely changes; the contested region -- the transition
line -- is narrower than the grid spacing and gets almost nothing.

The acquisition scores each cell of a coarse grid over the conditioning box by

    a_k = Tr Cov_{x ~ q(. | theta), theta in C_k} [ grad_theta U(x; theta) ]

which is the Fisher information of the family in its own parameters.  No
observable is chosen by hand: for an exponential family the gradient of the
energy in the natural parameters *is* the vector of sufficient statistics, so
the specific-heat and susceptibility channels appear on their own.

Two refinements, both load-bearing:

* the derivatives are taken in the NORMALISED coordinates of the sampling box.
  Tr Cov is not invariant under reparametrisation, so the raw trace would
  depend on whether temperature is measured in kelvin or its inverse rather
  than on the family being amortized over.  The caller supplies gradients
  already scaled by the box widths.
* each channel is standardised by its across-cell median before summing, and
  the sum is raised to a power below one.  The variance of the composition at
  coexistence exceeds its disordered value by several decades, and the raw
  trace would put essentially all adaptive mass in one or two cells.

Half of each batch of conditioning is drawn uniformly regardless, which keeps
coverage and keeps the acquisition statistics themselves unbiased, and a
uniform floor is mixed into the adaptive half so no cell ever starves.

Ported from the `fixed-point-sigma-h-fisher` branch of this repository and
generalised from the Ising (sigma, h) box to any conditioning box.
"""

from __future__ import annotations

import torch
from torch import Tensor


class FisherAcquisition:
    """Binned Fisher-information map over a 2-D conditioning box.

    `ranges` are the (lo, hi) of the *sampled* conditioning variables, in the
    same units the caller passes to `update` and gets back from `sample`.
    """

    def __init__(
        self,
        ranges: list[tuple[float, float]],
        bins: tuple[int, int] = (16, 16),
        ema: float = 0.9,
        floor: float = 0.05,
        power: float = 0.25,
        normalize: bool = True,
        device=None,
    ):
        assert len(ranges) == 2, "the acquisition map is over a 2-D box"
        self.ranges = [(float(a), float(b)) for a, b in ranges]
        self.bins = (int(bins[0]), int(bins[1]))
        self.ema = float(ema)
        self.floor = float(floor)
        self.power = float(power)
        self.normalize = bool(normalize)
        self.device = torch.device(device or "cpu")
        nb = self.bins[0] * self.bins[1]
        self.m1 = torch.zeros(nb, 2, device=self.device)
        self.m2 = torch.zeros(nb, 2, device=self.device)
        self.seen = torch.zeros(nb, device=self.device)

    # ------------------------------------------------------------------
    def _flat_bin(self, cond: Tensor) -> Tensor:
        cond = cond.view(-1, 2)
        idx = []
        for i, (lo, hi) in enumerate(self.ranges):
            u = (cond[:, i] - lo) / max(hi - lo, 1e-30)
            idx.append((u * self.bins[i]).long().clamp(0, self.bins[i] - 1))
        return (idx[0] * self.bins[1] + idx[1]).long()

    @torch.no_grad()
    def update(self, cond: Tensor, grad: Tensor):
        """Fold one batch in.  `grad` is (B, 2), d U / d theta in the
        normalised coordinates of the box."""
        cond = cond.to(self.device).view(-1, 2)
        g = grad.to(self.device).view(-1, 2).float()
        flat = self._flat_bin(cond)
        nb = self.m1.shape[0]
        ones = torch.ones(g.shape[0], device=self.device)
        cnt = torch.zeros(nb, device=self.device).scatter_add_(0, flat, ones)
        idx2 = flat[:, None].expand(-1, 2)
        s1 = torch.zeros(nb, 2, device=self.device).scatter_add_(0, idx2, g)
        s2 = torch.zeros(nb, 2, device=self.device).scatter_add_(0, idx2, g * g)
        got = cnt > 0
        c = cnt.clamp_min(1.0)[:, None]
        self.m1 = torch.where(got[:, None], self.ema * self.m1 + (1 - self.ema) * (s1 / c), self.m1)
        self.m2 = torch.where(got[:, None], self.ema * self.m2 + (1 - self.ema) * (s2 / c), self.m2)
        self.seen = torch.where(got, self.seen + 1, self.seen)

    @torch.no_grad()
    def map(self) -> Tensor:
        """The acquisition score per cell, shape `bins`."""
        var = (self.m2 - self.m1 ** 2).clamp_min(0.0)
        if self.normalize:
            live = self.seen > 0
            for d in range(var.shape[1]):
                v = var[live, d]
                med = v.median() if v.numel() else torch.tensor(1.0, device=self.device)
                var[:, d] = var[:, d] / med.clamp_min(1e-30)
        return var.sum(-1).view(*self.bins)

    @torch.no_grad()
    def weights(self) -> Tensor:
        a = self.map().clamp_min(0.0) ** self.power
        total = a.sum()
        if not torch.isfinite(total) or total <= 0:
            return torch.full_like(a, 1.0 / a.numel())
        a = a / total
        u = torch.full_like(a, 1.0 / a.numel())
        w = (1 - self.floor) * a + self.floor * u
        return w / w.sum()

    @torch.no_grad()
    def sample(self, n: int, generator=None) -> Tensor:
        """n conditioning draws from the acquisition map, (n, 2).

        A cell is drawn with probability proportional to its weight and the
        value is jittered uniformly inside it, so the draw stays continuous:
        training on a discrete ladder yields a lookup table that is
        confidently wrong between rungs.
        """
        w = self.weights().flatten()
        k = torch.multinomial(w, n, replacement=True, generator=generator)
        i0 = (k // self.bins[1]).float()
        i1 = (k % self.bins[1]).float()
        out = []
        for i, (lo, hi) in enumerate(self.ranges):
            step = (hi - lo) / self.bins[i]
            cell = i0 if i == 0 else i1
            u = torch.rand(n, device=self.device, generator=generator)
            out.append(lo + (cell + u) * step)
        return torch.stack(out, dim=-1)

    @torch.no_grad()
    def snapshot(self) -> dict:
        """The map and the sampling weights, small enough to write every
        checkpoint (a 16x16 box is 256 numbers per panel).

        `map` is Tr Cov[grad_theta U] per cell after the per-channel median
        standardisation; `weights` is what `sample` actually draws from, i.e.
        the map raised to `power`, normalised, and mixed with the uniform
        floor.  `seen` counts how many updates touched each cell, which is how
        an empty region is told apart from a genuinely flat one.
        """
        return {
            "bins": list(self.bins),
            "ranges": [list(r) for r in self.ranges],
            "power": self.power,
            "floor": self.floor,
            "map": self.map().cpu().tolist(),
            "weights": self.weights().cpu().tolist(),
            "seen": self.seen.view(*self.bins).cpu().tolist(),
        }

    # ------------------------------------------------------------------
    def state_dict(self) -> dict:
        return {"m1": self.m1, "m2": self.m2, "seen": self.seen}

    def load_state_dict(self, st: dict):
        if st and "m1" in st and st["m1"].shape == self.m1.shape:
            self.m1 = st["m1"].to(self.device)
            self.m2 = st["m2"].to(self.device)
            self.seen = st["seen"].to(self.device)


def mixed_cond(
    target,
    n: int,
    acq: FisherAcquisition | None,
    frac_uniform: float = 0.5,
    device=None,
    generator=None,
    **kwargs,
) -> Tensor:
    """`frac_uniform` of the draws from the family's own uniform sampler and
    the rest from the acquisition map.

    With no acquisition (or before it has seen anything) this is exactly the
    family's uniform draw, so switching the acquisition off recovers the
    previous behaviour bit for bit.
    """
    if acq is None or frac_uniform >= 1.0:
        return target.sample_cond(n, device=device, generator=generator, **kwargs)
    n_u = int(round(frac_uniform * n))
    parts = []
    if n_u > 0:
        parts.append(target.sample_cond(n_u, device=device, generator=generator, **kwargs))
    n_a = n - n_u
    if n_a > 0:
        dev = parts[0].device if parts else (device or "cpu")
        drawn = acq.sample(n_a, generator=generator).to(dev)
        if hasattr(target, "acq_to_cond"):
            drawn = target.acq_to_cond(drawn)
        parts.append(drawn)
    out = torch.cat(parts, dim=0)
    return out[torch.randperm(out.shape[0], device=out.device, generator=generator)]
