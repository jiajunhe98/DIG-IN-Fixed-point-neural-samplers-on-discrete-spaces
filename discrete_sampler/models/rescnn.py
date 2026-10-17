"""Periodic residual CNN with a receptive field that covers the torus.

Ported from the fixed-point masked-diffusion sampler (`fixed_point/`) and
generalized to the model interface of this package: any
2D lattice shape, any vocabulary, any number of output channels, with or
without time conditioning.

The design choices that matter, and why:

* **Circular padding everywhere**, so the network is *exactly* translation
  equivariant on the torus, matching the PBC of the lattice targets.  This
  is a symmetry of the target the network gets for free instead of having
  to learn it.
* **The receptive field must reach L//2**, the maximum distance on the
  torus, or the denoiser cannot see across the lattice -- which is exactly
  what it has to do at and below the critical point, where correlations
  span the system.  `cover_torus=True` turns that requirement into a
  construction-time assertion instead of a silent failure.
* **Dilations buy that radius far more cheaply than depth**: L=50 needs
  radius 25, which is 26 plain 3x3 layers but only ~7 dilated blocks.
  `dilations="auto"` sizes the ladder from the lattice shape.
* **Pre-norm residual blocks with FiLM conditioning** (GroupNorm -> conv ->
  FiLM -> GroupNorm -> conv, added back onto the input).
* **Low-frequency Fourier features** for the conditioning scalar; see
  `fourier_features` for the aliasing failure this avoids.
* **Zero-initialised head**, so the network starts as the uniform
  predictor -- the high-entropy initialisation the fixed-point loop wants,
  since it must not begin already committed to a mode.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def fourier_features(v: Tensor, n: int = 8, max_scale: float = 8.0) -> Tensor:
    """Fourier features of a scalar in [0, 1]: (B,) -> (B, 2n).

    Frequencies are geometric in [pi, max_scale * pi].  Keep `max_scale`
    LOW for any conditioning variable trained on a discrete ladder of
    values: with frequencies up to 8*pi and a ladder spacing of 0.05 the
    top component advances ~2 full cycles between neighbouring rungs, so
    adjacent rungs receive near-orthogonal codes, the network degenerates
    into a lookup table, and it returns a confidently wrong answer at every
    value in between.  A continuously sampled t in [0, 1] does not have
    that problem, which is why the default here is 8.0.
    """
    freqs = torch.exp(
        torch.linspace(0.0, math.log(max_scale), n, device=v.device)
    ) * math.pi
    a = v[:, None] * freqs[None, :]
    return torch.cat([torch.cos(a), torch.sin(a)], dim=-1)


def _num_groups(ch: int) -> int:
    """Largest GroupNorm group count <= 8 that divides `ch`."""
    return max(g for g in (8, 4, 2, 1) if ch % g == 0)


def _axis_dilation(d: int, k: int, size: int) -> int:
    """Largest dilation <= d whose circular padding fits the axis.

    Circular padding requires pad < axis length, and pad = dilation * (k // 2).
    """
    if k <= 1 or size <= 1:
        return 1
    return max(1, min(d, (size - 1) // (k // 2)))


def auto_dilations(
    n_blocks: int, kernel: tuple[int, int], need: int, shape: tuple[int, int]
) -> list[int]:
    """Smallest doubling dilation ladder whose receptive field reaches `need`.

    Interior blocks grow first and the last block stays at 1, which keeps a
    plain local convolution at each end of the stack; the ladder falls back
    to growing every block when there are too few to have an interior.
    """
    r = max(k // 2 for k in kernel)
    dil = [1] * n_blocks
    if r == 0:
        return dil
    max_d = max(
        1,
        min(
            (size - 1) // (k // 2)
            for k, size in zip(kernel, shape)
            if k > 1 and size > 1
        ),
    )
    order = list(range(1, n_blocks - 1)) or list(range(n_blocks))
    d = 1
    for i in order:
        if r * (1 + 2 * sum(dil)) >= need:
            break
        d = min(max_d, 2 * d)
        dil[i] = d
    return dil


#: circular-padded convolution per lattice dimensionality (1D chains, 2D
#: lattices, 3D cells such as the fcc supercells of the alloy families)
_CONV = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}


class _ResBlock(nn.Module):
    def __init__(
        self,
        ch: int,
        cond_dim: int | None,
        kernel: tuple[int, ...],
        dilation: tuple[int, ...],
    ):
        super().__init__()
        self.ndim = len(kernel)
        conv = _CONV[self.ndim]
        pad = tuple(d * (k // 2) for k, d in zip(kernel, dilation))
        g = _num_groups(ch)
        self.n1 = nn.GroupNorm(g, ch)
        self.c1 = conv(
            ch, ch, kernel, padding=pad, dilation=dilation, padding_mode="circular"
        )
        self.film = nn.Linear(cond_dim, 2 * ch) if cond_dim else None
        self.n2 = nn.GroupNorm(g, ch)
        self.c2 = conv(
            ch, ch, kernel, padding=pad, dilation=dilation, padding_mode="circular"
        )

    def forward(self, h: Tensor, c: Tensor | None) -> Tensor:
        r = self.c1(F.silu(self.n1(h)))
        if self.film is not None and c is not None:
            f = self.film(c).view(c.shape[0], -1, *([1] * self.ndim))
            scale, shift = f.chunk(2, dim=1)
            r = r * (1.0 + scale) + shift
        r = self.c2(F.silu(self.n2(r)))
        return h + r


class PeriodicResCNN(nn.Module):
    """Circular-padded residual CNN, `(B, D) -> (B, D, out_channels)`.

    `dilations` is either an explicit per-block list, `"auto"` (size the
    ladder from the lattice so the receptive field covers the torus), or
    `None` / `"none"` for plain undilated convolutions.
    """

    def __init__(
        self,
        lattice_shape: tuple[int, ...],
        vocab_size: int,
        out_channels: int,
        require_time: bool = True,
        hidden_size: int = 96,
        n_blocks: int = 7,
        kernel_size: int = 3,
        dilations: str | list[int] | None = "auto",
        cond_dim: int | None = None,
        n_time_freq: int = 8,
        time_max_scale: float = 8.0,
        cond_ranges: list | None = None,
        n_cond_freq: int = 6,
        cond_max_scale: float = 2.0,
        global_feat: bool = False,
        zero_init_head: bool = True,
        cover_torus: bool = True,
    ):
        super().__init__()
        assert 1 <= len(lattice_shape) <= 3, (
            f"PeriodicResCNN handles 1D, 2D and 3D lattices, got {lattice_shape}"
        )
        self.lattice_shape = tuple(lattice_shape)
        self.ndim = len(self.lattice_shape)
        conv = _CONV[self.ndim]
        self.vocab_size = vocab_size
        self.out_channels = out_channels
        self.require_time = require_time
        self.global_feat = global_feat

        # a length-1 axis has no room for a wider kernel (and circular padding
        # on it is undefined); collapse the kernel there instead of failing
        kernel = tuple(kernel_size if s > 1 else 1 for s in self.lattice_shape)
        self.kernel = kernel
        need = max(s // 2 for s in self.lattice_shape)

        if dilations is None or dilations == "none":
            dil = [1] * n_blocks
        elif isinstance(dilations, str):
            assert dilations == "auto", f"unknown dilations spec {dilations!r}"
            dil = auto_dilations(n_blocks, kernel, need, self.lattice_shape)
        else:
            dil = [int(d) for d in dilations]
            assert len(dil) == n_blocks, "one dilation per block"
        self.dilations = dil
        per_axis = [
            tuple(_axis_dilation(d, k, s) for k, s in zip(kernel, self.lattice_shape))
            for d in dil
        ]
        # receptive-field radius per axis: the input conv contributes k//2 and
        # each block two dilated convs, i.e. (k//2) * (1 + 2 * sum(dilations))
        self.radius = tuple(
            (k // 2) * (1 + 2 * sum(pa[ax] for pa in per_axis))
            for ax, k in enumerate(kernel)
        )
        if cover_torus:
            for ax, s in enumerate(self.lattice_shape):
                assert self.radius[ax] >= s // 2, (
                    f"receptive field {self.radius[ax]} does not reach L//2 = "
                    f"{s // 2} on axis {ax}: the denoiser cannot see across the "
                    f"torus.  Add blocks, widen the kernel, or pass explicit "
                    f"dilations (cover_torus=false to override)."
                )

        cond_dim = cond_dim or hidden_size
        self.inp = conv(
            vocab_size,
            hidden_size,
            kernel,
            padding=tuple(k // 2 for k in kernel),
            padding_mode="circular",
        )
        self.n_time_freq = n_time_freq
        self.time_max_scale = time_max_scale
        self.n_cond_freq = n_cond_freq
        self.cond_max_scale = cond_max_scale
        self.cond_ranges = [tuple(float(v) for v in r) for r in (cond_ranges or [])]
        self.n_cond = len(self.cond_ranges)
        # width of the conditioning vector: the time features plus, per
        # amortized variable, its own low-frequency features and raw value
        n_in = (2 * n_time_freq + 1) * int(require_time)
        n_in += self.n_cond * (2 * n_cond_freq + 1)
        self.cond = None
        if n_in:
            self.cond = nn.Sequential(
                nn.Linear(n_in, cond_dim),
                nn.SiLU(),
                nn.Linear(cond_dim, cond_dim),
                nn.SiLU(),
            )
        self.blocks = nn.ModuleList(
            [
                _ResBlock(hidden_size, cond_dim if n_in else None, kernel, pa)
                for pa in per_axis
            ]
        )
        if global_feat:
            self.global_proj = nn.ModuleList(
                [nn.Linear(hidden_size, hidden_size) for _ in range(n_blocks)]
            )
        self.norm = nn.GroupNorm(_num_groups(hidden_size), hidden_size)
        self.head = conv(hidden_size, out_channels, 1)
        if zero_init_head:
            nn.init.zeros_(self.head.weight)
            nn.init.zeros_(self.head.bias)

    def forward(
        self, x: Tensor, t: Tensor | None = None, cond: Tensor | None = None
    ) -> Tensor:
        B = x.shape[0]
        h = (
            F.one_hot(x, self.vocab_size)
            .to(self.inp.weight.dtype)
            .view(B, *self.lattice_shape, self.vocab_size)
            .movedim(-1, 1)
        )
        c = None
        if self.cond is not None:
            parts = []
            if self.require_time:
                assert t is not None, "model was built with time conditioning"
                parts += [
                    fourier_features(t, self.n_time_freq, self.time_max_scale),
                    t[:, None],
                ]
            if self.n_cond:
                assert cond is not None, "model was built with amortized conditioning"
                cond = cond.view(B, self.n_cond).to(h.dtype)
                for i, (lo, hi) in enumerate(self.cond_ranges):
                    # normalise to [0, 1] over the variable's own range, then
                    # encode with LOW frequencies -- see fourier_features for
                    # the aliasing failure that high ones cause here
                    v = (cond[:, i] - lo) / (hi - lo)
                    parts += [
                        fourier_features(v, self.n_cond_freq, self.cond_max_scale),
                        v[:, None],
                    ]
            c = self.cond(torch.cat(parts, dim=-1))
        h = self.inp(h)
        for i, blk in enumerate(self.blocks):
            h = blk(h, c)
            if self.global_feat:
                g = self.global_proj[i](h.mean(dim=tuple(range(2, 2 + self.ndim))))
                h = h + g.view(B, -1, *([1] * self.ndim))
        out = self.head(F.silu(self.norm(h)))  # (B, C, *lattice_shape)
        return out.flatten(2).transpose(1, 2)  # (B, D, C)
