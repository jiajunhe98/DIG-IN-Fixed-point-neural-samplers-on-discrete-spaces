"""Locally equivariant convolutional rate network (LEAPS, Holderrieth et al.).

The network outputs per-site logits R_d(x, a) with the built-in constraint
R_d(x, x^d) = 0 and the antisymmetry R_d(x^{d<-a}, x^d) = -R_d(x, a),
obtained by (i) making the feature at site d independent of x^d (masked-
center circular convs; state-dependent kernels with a zeroed center tap)
and (ii) subtracting the logit of the current category.  This lets the
incoming rates Q(x^{d<-a} -> x) = relu(-R_d(x, a)) be read off the same
forward pass as the outgoing rates relu(R_d(x, a)).

We use the categorical (Potts-style) construction for every alphabet size,
including binary.

The lattice may be 2-D or 3-D.  The original implementation was 2-D only,
which is why LEAPS never ran on the alloy families: those are 5x5x5, 8x4x4,
8x8x8 and 12x10x10 cells and the net asserted its way out.  FrOGS builds its
transport on this network, so 3-D is a prerequisite for it rather than a
convenience.

Both the rate net and the free-energy head optionally take a conditioning
vector, which is what lets a single model cover a (T, delta_mu) box.  It
enters where the time already does: the state-dependent kernels and the
scalar head both read [t, c] instead of t alone.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

_CONV = {2: nn.Conv2d, 3: nn.Conv3d}


def unfold_circular(x: Tensor, kernel_size: int) -> Tensor:
    """(B, C, *S) -> (B, C, K**ndim, *S) with circular padding on every axis."""
    ndim = x.dim() - 2
    pad = kernel_size // 2
    xp = F.pad(x, (pad, pad) * ndim, mode="circular")
    for d in range(ndim):
        xp = xp.unfold(2 + d, kernel_size, 1)
    # (B, C, *S, *K) -> (B, C, prod(K), *S)
    spatial = tuple(range(2, 2 + ndim))
    kernel = tuple(range(2 + ndim, 2 + 2 * ndim))
    xp = xp.permute(0, 1, *kernel, *spatial).contiguous()
    return xp.flatten(2, 1 + ndim)


class MaskedCenterConv(nn.Module):
    """Circular conv whose kernel ignores the center site."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, ndim: int = 2):
        super().__init__()
        self.conv = _CONV[ndim](
            in_ch, out_ch, kernel_size, padding=kernel_size // 2,
            padding_mode="circular", bias=False,
        )
        mask = torch.ones_like(self.conv.weight)
        centre = (Ellipsis,) + (kernel_size // 2,) * ndim
        mask[centre] = 0.0
        self.register_buffer("mask", mask)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv._conv_forward(x, self.conv.weight * self.mask,
                                       self.conv.bias)


class StateDependentConv(nn.Module):
    """Depthwise conv whose kernel is generated per site from the previous
    layer's activation (independent of the center) plus a time/condition MLP;
    the center tap is zeroed."""

    def __init__(self, channels: int, kernel_size: int, ndim: int = 2,
                 t_hidden: int = 64, cond_dim: int = 0):
        super().__init__()
        KD = kernel_size ** ndim
        self.kernel_size = kernel_size
        self.ndim = ndim
        self.mid_idx = KD // 2
        self.cond_dim = int(cond_dim)
        self.A = nn.Parameter(0.002 * torch.rand(channels, KD, channels) / KD**0.5)
        self.b1 = nn.Parameter(0.002 * torch.rand(channels, KD))
        self.b2 = nn.Parameter(0.002 * torch.rand(channels, KD))
        self.mlp = nn.Sequential(
            nn.Linear(1 + self.cond_dim, t_hidden), nn.SiLU(),
            nn.Linear(t_hidden, t_hidden), nn.SiLU(),
            nn.Linear(t_hidden, KD),
        )

    def forward(self, x_emb: Tensor, t: Tensor, prev: Tensor,
                cond: Tensor | None = None) -> Tensor:
        xu = unfold_circular(x_emb, self.kernel_size)      # (B, C, KD, *S)
        sp = (None,) * self.ndim
        a_prev = torch.einsum("bc...,lic->bli...", prev, self.A)
        k_x = F.silu(a_prev + self.b1[(None, slice(None), slice(None)) + sp])
        inp = t.view(-1, 1)
        if self.cond_dim:
            inp = torch.cat([inp, cond.view(t.shape[0], -1)], dim=-1)
        t_emb = self.mlp(inp)                              # (B, KD)
        k_x = k_x + self.b2[(None, slice(None), slice(None)) + sp]
        k_x = k_x + t_emb[(slice(None), None, slice(None)) + sp]
        k_x = k_x.clone()
        k_x[:, :, self.mid_idx] = 0.0
        return (k_x * xu).sum(dim=2)                       # (B, C, *S)


class LocEquivNet(nn.Module):
    def __init__(
        self,
        lattice_shape: tuple[int, ...],
        vocab_size: int,
        channels: int = 20,
        kernel_sizes: tuple[int, ...] = (3, 5, 7, 9),
        cond_dim: int = 0,
    ):
        super().__init__()
        ndim = len(lattice_shape)
        assert ndim in (2, 3), "LEAPS net expects a 2-D or 3-D lattice"
        # a kernel wider than the cell wraps onto itself through the circular
        # padding and stops being a neighbourhood; clamp to the shortest axis
        short = min(lattice_shape)
        kernel_sizes = tuple(min(k, short - (1 - short % 2)) for k in kernel_sizes)
        kernel_sizes = tuple(k if k % 2 else k - 1 for k in kernel_sizes)
        kernel_sizes = tuple(max(k, 3) for k in kernel_sizes)
        self.ndim = ndim
        self.lattice_shape = tuple(lattice_shape)
        self.N = vocab_size
        self.cond_dim = int(cond_dim)
        self.emb = nn.Embedding(vocab_size, channels, max_norm=1.0)
        nn.init.normal_(self.emb.weight, std=1.0 / channels**0.5)
        self.initial = MaskedCenterConv(channels, channels, kernel_sizes[0], ndim)
        self.convs = nn.ModuleList(
            [StateDependentConv(channels, k, ndim, cond_dim=self.cond_dim)
             for k in kernel_sizes[1:]]
        )
        self.linear = nn.Parameter(
            0.02 * torch.randn(vocab_size, channels) / channels**0.5)

    def forward(self, x: Tensor, t: Tensor,
                cond: Tensor | None = None) -> Tensor:
        """x (B, D) long, t (B,) -> logits R (B, D, N) with R[d, x^d] = 0."""
        B = x.shape[0]
        z = x.view(B, *self.lattice_shape)
        e = self.emb(z).movedim(-1, 1)                     # (B, C, *S)
        sp = (None,) * self.ndim
        e_t = (1.0 + t)[(slice(None), None) + sp] * e
        pre = self.initial(e_t)
        for conv in self.convs:
            pre = conv(e, t, F.silu(pre), cond)
        proj = torch.einsum("nc,bc...->bn...", self.linear, pre)
        offset = torch.gather(proj, 1, z.unsqueeze(1))
        R = proj - offset
        return R.flatten(2).transpose(1, 2)                # (B, D, N)


class FNet(nn.Module):
    """Scalar free-energy surrogate F(t) or F(t, c); only dF/dt enters the loss.

    With conditioning this is the per-(s, c) baseline the FrOGS derivation
    calls for: a minibatch mixing state points whose free energies differ by
    many kT cannot be centred by its own mean, and the learned head is what
    repairs that.
    """

    def __init__(self, hidden: int = 200, n_layers: int = 4, cond_dim: int = 0):
        super().__init__()
        self.cond_dim = int(cond_dim)
        layers: list[nn.Module] = [nn.Linear(1 + self.cond_dim, hidden), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def gradt(self, t: Tensor, cond: Tensor | None = None) -> Tensor:
        t = t.view(-1, 1).detach().requires_grad_(True)
        inp = t if not self.cond_dim else torch.cat(
            [t, cond.view(t.shape[0], -1).detach()], dim=-1)
        f = self.net(inp).sum()
        (df,) = torch.autograd.grad(f, t, create_graph=True)
        return df.view(-1)
