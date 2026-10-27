"""Bidirectional transformer denoiser over D binned coordinates.

Input: tokens (B, D) in {0, ..., L-1} or MASK = L.  Output: logits (B, D, L),
the any-order head q^d(. | revealed) at every position (only masked positions
are used).  No time input: the exact masked head depends on neither t nor
the schedule (DIG-IN / MDNS-style time-free denoiser).

A revealed coordinate is embedded by its own (position, bin) table plus
Fourier features of the bin centre on its box, so neighbouring bins start out
close.  The output head is per position and zero-initialised: the first
rollout is uniform on the box, i.e. DIG-IN's warm start.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class AnyOrderTransformer(nn.Module):
    def __init__(self, D: int, L: int, d_model: int = 256, n_layers: int = 6, n_heads: int = 8,
                 ff_mult: int = 4, n_fourier: int = 6, kinds: list[str] | None = None):
        super().__init__()
        self.D, self.L = D, L
        self.mask_token = L
        self.tok = nn.Parameter(torch.randn(D, L, d_model) * 0.02)
        self.mask_emb = nn.Parameter(torch.randn(D, d_model) * 0.02)
        self.pos = nn.Parameter(torch.randn(D, d_model) * 0.02)
        kinds = kinds or ["x"] * D
        names = sorted(set(kinds))
        self.register_buffer("kind_id", torch.tensor([names.index(k) for k in kinds]))
        self.kind_emb = nn.Embedding(len(names), d_model)
        u = (torch.arange(L) + 0.5) / L * 2.0 - 1.0                       # bin centre on [-1, 1]
        k = torch.arange(1, n_fourier + 1)
        feat = torch.cat([u.view(-1, 1), torch.cos(math.pi * u.view(-1, 1) * k),
                          torch.sin(math.pi * u.view(-1, 1) * k)], dim=-1)  # (L, 1 + 2K)
        self.register_buffer("feat", feat)
        self.feat_proj = nn.Linear(feat.shape[-1], d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, ff_mult * d_model, dropout=0.0,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head_w = nn.Parameter(torch.zeros(D, d_model, L))
        self.head_b = nn.Parameter(torch.zeros(D, L))

    def forward(self, x: Tensor) -> Tensor:
        B, D = x.shape
        masked = x == self.mask_token
        b = x.clamp(max=self.L - 1)
        ar = torch.arange(D, device=x.device)
        tok = self.tok[ar.view(1, -1), b]                                  # (B, D, d)
        tok = tok + self.feat_proj(self.feat[b])
        h = torch.where(masked.unsqueeze(-1), self.mask_emb.unsqueeze(0).expand(B, -1, -1), tok)
        h = h + self.pos.unsqueeze(0) + self.kind_emb(self.kind_id).unsqueeze(0)
        h = self.norm(self.encoder(h))
        return torch.einsum("bdh,dhl->bdl", h, self.head_w) + self.head_b


@torch.no_grad()
def any_order_sample(net: nn.Module, n: int, D: int, L: int, device, generator=None):
    """One coordinate per network call in a uniformly random order.

    Returns (bins (n, D), log q(bins | order) (n,)): the exact proposal
    log-probability of the draw given its order.
    """
    x = torch.full((n, D), L, dtype=torch.long, device=device)
    log_q = torch.zeros(n, dtype=torch.float64, device=device)
    order = torch.argsort(torch.rand(n, D, device=device, generator=generator), dim=1)
    ar = torch.arange(n, device=device)
    for k in range(D):
        d = order[:, k]
        lg = net(x)[ar, d].float().log_softmax(-1)                          # (n, L)
        u = torch.rand(n, 1, device=device, generator=generator)
        tok = (lg.exp().cumsum(-1) < u).sum(-1).clamp(max=L - 1)
        log_q += lg[ar, tok].double()
        x[ar, d] = tok
    return x, log_q
