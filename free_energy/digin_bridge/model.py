"""Hollow two-head transformer over the 60 binned internal coordinates.

Output at coordinate d depends on x^{-d} and t only (Sun et al. 2022's hollow
transformer):

  * a forward causal stream reads [BOS, x_0, ..., x_{D-2}], so its output at d
    summarizes x_{<d}; a backward causal stream does the same for x_{>d};
  * h_d = Linear([F_d, B_d]) starts the readout, and every readout block lets
    h_d attend to F_p (p <= d) and B_p (p >= d) only, never to another h.

Both heads are per coordinate and zero-initialised: q^d(. | x^{-d}, t) starts
uniform and phi^d(. | x^{-d}, t) at zero, so the untrained CTMC never jumps and
the first rollouts are the identity transport (exact Zwanzig weights).
A coordinate is embedded by its own (position, bin) table plus Fourier
features of the bin centre on its box.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def time_features(t: Tensor, n: int = 8) -> Tensor:
    k = torch.arange(1, n + 1, device=t.device, dtype=t.dtype)
    a = math.pi * t.view(-1, 1) * k
    return torch.cat([t.view(-1, 1), torch.cos(a), torch.sin(a)], dim=-1)


class ReadoutBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_mult: int):
        super().__init__()
        self.nq, self.nkv = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.nff = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, ff_mult * d_model), nn.GELU(), nn.Linear(ff_mult * d_model, d_model))

    def forward(self, h: Tensor, kv: Tensor, block: Tensor) -> Tensor:
        k = self.nkv(kv)
        h = h + self.attn(self.nq(h), k, k, attn_mask=block, need_weights=False)[0]
        return h + self.ff(self.nff(h))


class HollowTransformer(nn.Module):
    def __init__(self, D: int, L: int, d_model: int = 256, n_layers: int = 4, n_readout: int = 2,
                 n_heads: int = 8, ff_mult: int = 4, n_fourier: int = 6, kinds: list[str] | None = None):
        super().__init__()
        self.D, self.L = D, L
        self.tok = nn.Parameter(torch.randn(D, L, d_model) * 0.02)
        self.pos = nn.Parameter(torch.randn(D, d_model) * 0.02)
        kinds = kinds or ["x"] * D
        names = sorted(set(kinds))
        self.register_buffer("kind_id", torch.tensor([names.index(k) for k in kinds]))
        self.kind_emb = nn.Embedding(len(names), d_model)
        u = (torch.arange(L) + 0.5) / L * 2.0 - 1.0
        k = torch.arange(1, n_fourier + 1)
        feat = torch.cat([u.view(-1, 1), torch.cos(math.pi * u.view(-1, 1) * k),
                          torch.sin(math.pi * u.view(-1, 1) * k)], dim=-1)
        self.register_buffer("feat", feat)
        self.feat_proj = nn.Linear(feat.shape[-1], d_model)
        self.time = nn.Sequential(nn.Linear(time_features(torch.zeros(1)).shape[-1], d_model), nn.SiLU(),
                                  nn.Linear(d_model, d_model))
        self.bos = nn.Parameter(torch.randn(2, d_model) * 0.02)          # forward / backward stream
        self.query_pos = nn.Parameter(torch.randn(2, D, d_model) * 0.02)  # which coordinate a stream slot serves

        def stream():
            layer = nn.TransformerEncoderLayer(d_model, n_heads, ff_mult * d_model, dropout=0.0,
                                               activation="gelu", batch_first=True, norm_first=True)
            return nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.fwd, self.bwd = stream(), stream()
        self.register_buffer("causal", nn.Transformer.generate_square_subsequent_mask(D), persistent=False)
        p = torch.arange(D)
        # readout: query d may see F_p for p <= d and B_p for p >= d (True = blocked)
        self.register_buffer("block", torch.cat([p.view(1, -1) > p.view(-1, 1), p.view(1, -1) < p.view(-1, 1)], 1),
                             persistent=False)
        self.mix = nn.Linear(2 * d_model, d_model)
        self.readout = nn.ModuleList([ReadoutBlock(d_model, n_heads, ff_mult) for _ in range(n_readout)])
        self.norm = nn.LayerNorm(d_model)
        self.q_w = nn.Parameter(torch.zeros(D, d_model, L))
        self.q_b = nn.Parameter(torch.zeros(D, L))
        self.phi_w = nn.Parameter(torch.zeros(D, d_model, L))
        self.phi_b = nn.Parameter(torch.zeros(D, L))

    def forward(self, x: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        """x: (B, D) bins, t: (B,) -> (q logits, phi), each (B, D, L)."""
        B, D = x.shape
        ar = torch.arange(D, device=x.device)
        e = self.tok[ar.view(1, -1), x] + self.feat_proj(self.feat[x])
        e = e + self.pos + self.kind_emb(self.kind_id)                       # (B, D, d)
        te = self.time(time_features(t.to(e.dtype))).unsqueeze(1)          # (B, 1, d)
        bos = self.bos.unsqueeze(0).expand(B, -1, -1)
        f_in = torch.cat([bos[:, :1], e[:, :-1]], 1) + self.query_pos[0] + te
        b_in = torch.cat([bos[:, 1:], e.flip(1)[:, :-1]], 1) + self.query_pos[1].flip(0) + te
        F = self.fwd(f_in, mask=self.causal, is_causal=True)                # F[:, d] sees x_{<d}
        Bk = self.bwd(b_in, mask=self.causal, is_causal=True).flip(1)       # Bk[:, d] sees x_{>d}
        h = self.mix(torch.cat([F, Bk], -1)) + te
        kv = torch.cat([F, Bk], 1)
        for blk in self.readout:
            h = blk(h, kv, self.block)
        h = self.norm(h)
        q = torch.einsum("bdh,dhl->bdl", h, self.q_w) + self.q_b
        phi = torch.einsum("bdh,dhl->bdl", h, self.phi_w) + self.phi_b
        return q, phi


@torch.no_grad()
def hollow_check(net: HollowTransformer, n: int = 4, seed: int = 0) -> tuple[float, float]:
    """Change x^d and measure both heads: (max change at d, mean change elsewhere).

    Run on a randomised copy: the zero-initialised heads would pass trivially."""
    g = torch.Generator().manual_seed(seed)
    dev = next(net.parameters()).device
    x = torch.randint(0, net.L, (n, net.D), generator=g).to(dev)
    t = torch.rand(n, generator=g).to(dev)
    q0, p0 = net(x, t)
    worst, other = 0.0, []
    for d in range(net.D):
        x2 = x.clone()
        x2[:, d] = (x2[:, d] + 1 + torch.randint(0, net.L - 1, (n,), generator=g).to(dev)) % net.L
        q1, p1 = net(x2, t)
        dq, dp = (q1 - q0).abs(), (p1 - p0).abs()
        worst = max(worst, float(dq[:, d].max()), float(dp[:, d].max()))
        keep = torch.ones(net.D, dtype=torch.bool); keep[d] = False
        other.append(float(dq[:, keep].mean() + dp[:, keep].mean()))
    return worst, sum(other) / len(other)
