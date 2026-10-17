"""Causal masked convolutional network (PixelCNN, van den Oord et al. 2016).

This is the backbone the published alloy autoregressive sampler uses -- SEGAL,
Damewood et al. 2022 -- and the prior FrOGS puts in front of its transport.
It factorises p(x) = prod_d p(x_d | x_<d) in the lattice's raster order, the
same order `x.view(B, *lattice_shape)` already implies everywhere else in this
codebase, so a draw from it indexes like any other configuration.

Causality comes from the kernel mask.  For a kernel offset o relative to the
centre, a tap is allowed only if o precedes the centre in raster order; mask
"A", used once at the input, excludes the centre itself so site d never sees
x_d, and mask "B" admits it for every later layer, which is what lets depth
accumulate context without leaking the answer.

Padding is ZERO, not circular.  Circular padding would wrap the last sites of
an axis onto the first and let a later site condition an earlier one, which
destroys the factorisation silently -- the model would train happily and its
log-densities would not be a distribution.  The lattice is periodic in the
physics; the autoregressive order over it is not.

Conditioning is FiLM per block, so one network covers a (T, delta_mu) box.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

_CONV = {1: nn.Conv1d, 2: nn.Conv2d, 3: nn.Conv3d}


def _causal_mask(kernel_size: int, ndim: int, include_centre: bool) -> Tensor:
    """1 where the tap precedes the centre in raster order."""
    k, c = kernel_size, kernel_size // 2
    grid = torch.stack(torch.meshgrid(
        *[torch.arange(k) for _ in range(ndim)], indexing="ij"), dim=-1)
    off = grid - c                                    # (k,)*ndim x ndim
    # raster order: earlier on the first axis wins; ties fall to the next
    before = torch.zeros(off.shape[:-1], dtype=torch.bool)
    equal = torch.ones(off.shape[:-1], dtype=torch.bool)
    for d in range(ndim):
        before |= equal & (off[..., d] < 0)
        equal &= off[..., d] == 0
    return (before | (equal if include_centre else False)).to(torch.float32)


class MaskedConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, ndim: int,
                 include_centre: bool):
        super().__init__()
        self.conv = _CONV[ndim](in_ch, out_ch, kernel_size,
                                padding=kernel_size // 2)
        m = _causal_mask(kernel_size, ndim, include_centre)
        self.register_buffer("mask", m.view(1, 1, *m.shape))

    def forward(self, x: Tensor) -> Tensor:
        return self.conv._conv_forward(x, self.conv.weight * self.mask,
                                       self.conv.bias)


class _Block(nn.Module):
    def __init__(self, ch: int, kernel_size: int, ndim: int, cond_dim: int):
        super().__init__()
        self.ndim = ndim
        self.conv = MaskedConv(ch, ch, kernel_size, ndim, include_centre=True)
        self.film = nn.Linear(cond_dim, 2 * ch) if cond_dim else None
        if self.film is not None:
            nn.init.zeros_(self.film.weight)
            nn.init.zeros_(self.film.bias)

    def forward(self, h: Tensor, cond: Tensor | None) -> Tensor:
        y = self.conv(F.silu(h))
        if self.film is not None and cond is not None:
            g, b = self.film(cond).chunk(2, dim=-1)
            shape = (h.shape[0], -1) + (1,) * self.ndim
            y = y * (1.0 + g.view(*shape)) + b.view(*shape)
        return h + y


class PixelCNN(nn.Module):
    """Autoregressive lattice model with a causal convolutional backbone."""

    #: the ARM decoder branches on this; a conv net has no rotary embedding
    use_rope = False

    def __init__(
        self,
        lattice_shape: tuple[int, ...],
        vocab_size: int,
        out_channels: int,
        require_time: bool = False,
        channels: int = 96,
        n_blocks: int = 6,
        kernel_size: int = 3,
        cond_ranges: list | None = None,
        logit_cap: float = 40.0,
        **_ignored,
    ):
        super().__init__()
        self.lattice_shape = tuple(lattice_shape)
        self.ndim = len(self.lattice_shape)
        assert self.ndim in (1, 2, 3), "PixelCNN expects a 1/2/3-D lattice"
        self.D = int(torch.tensor(self.lattice_shape).prod())
        self.vocab_size = vocab_size
        self.out_channels = out_channels
        self.require_time = require_time
        self.cond_ranges = list(cond_ranges or [])
        cond_dim = len(self.cond_ranges)
        self.cond_dim = cond_dim
        #: Soft bound on the logits, `cap * tanh(logit / cap)`.  Without it a
        #: reverse-KL run walks the head off to |logit| ~ 3e5: measured on a
        #: finished SEGAL checkpoint, whose conditional entropy was 0.008 nats
        #: of a possible 88.7.  Everything downstream then degenerates -- the
        #: softmax is a delta, log q is exactly 0 in fp32, the advantage and
        #: its gradient are exactly 0, and training stops without erroring.
        #:
        #: 40 is chosen to sit above the physics and far below the pathology.
        #: One site's heat-bath logit gap is beta*dE, at most about 29 on
        #: Cu-Au at 200 K, so the cap never binds on a conditional the target
        #: actually asks for; it binds four orders of magnitude before the
        #: blow-up.  0 disables it.
        self.logit_cap = float(logit_cap)

        self.emb = nn.Embedding(vocab_size + 1, channels)  # +1 = "not yet set"
        self.input = MaskedConv(channels, channels, kernel_size, self.ndim,
                                include_centre=False)
        self.blocks = nn.ModuleList(
            [_Block(channels, kernel_size, self.ndim, cond_dim)
             for _ in range(n_blocks)])
        self.head = nn.Sequential(
            nn.SiLU(), _CONV[self.ndim](channels, out_channels, 1))

    def _cond_vec(self, cond: Tensor | None, batch: int, device) -> Tensor | None:
        if not self.cond_dim:
            return None
        if cond is None:
            return torch.zeros(batch, self.cond_dim, device=device)
        c = cond.view(batch, -1).float()
        lo = torch.tensor([r[0] for r in self.cond_ranges], device=c.device)
        hi = torch.tensor([r[1] for r in self.cond_ranges], device=c.device)
        return (c - lo) / (hi - lo).clamp_min(1e-12)

    def forward(self, x: Tensor, t: Tensor | None = None,
                cond: Tensor | None = None) -> Tensor:
        """x (B, D) long -> (B, D, out_channels); row d conditions on x_<d."""
        B = x.shape[0]
        c = self._cond_vec(cond, B, x.device)
        h = self.emb(x.view(B, *self.lattice_shape)).movedim(-1, 1)
        h = self.input(h)
        for blk in self.blocks:
            h = blk(h, c)
        out = self.head(h)                      # (B, out, *S)
        if self.logit_cap > 0:
            # Straight-through clamp: the forward value is bounded, the
            # gradient is NOT.  A smooth `cap * tanh(x / cap)` bounds both --
            # its derivative is sech^2, which vanishes once the
            # pre-activation drifts past the cap, and the head then freezes
            # exactly as it did uncapped.  Passing the gradient through means
            # a drifting pre-activation can still be pulled back.
            capped = out.clamp(-self.logit_cap, self.logit_cap)
            # value is exactly `capped`, gradient is exactly identity.  Written
            # this way rather than `out + (capped - out).detach()` because that
            # form adds two numbers of order 1e5 to get one of order 10, and
            # loses the answer to float32 cancellation: measured 64.0 out of a
            # cap of 40.
            out = capped.detach() + (out - out.detach())
        return out.flatten(2).transpose(1, 2)   # (B, D, out)

    # -- the sampling interface ARM expects -----------------------------
    # A conv net has no key/value cache: the "cache" is the partially filled
    # lattice, and each step is a full masked pass.  That is D passes for a
    # draw against the transformer's D cached steps -- more arithmetic, but
    # it is what makes the backbone the published one.

    def new_kv_caches(self, batch: int, device=None, dtype=None) -> list[dict]:
        device = device or next(self.parameters()).device
        return [{"x": torch.full((batch, self.D), self.vocab_size,
                                 dtype=torch.long, device=device)}]

    def decode_step(self, prev: Tensor | None, pos: Tensor, caches: list[dict],
                    batch: int, cond: Tensor | None = None) -> Tensor:
        """Logits for site `pos`, given the sites already written."""
        state = caches[0]["x"]
        p = int(pos)
        if prev is not None and p > 0:
            state[:, p - 1] = prev.view(-1)
        return self.forward(state, None, cond)[:, p]
