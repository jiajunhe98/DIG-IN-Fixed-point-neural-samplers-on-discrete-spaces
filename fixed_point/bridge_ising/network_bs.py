"""Blind-spot ("hollow") CNN for the two-head bridge objective on an L x L torus.

Output at site d must depend on X_t^{-d} ONLY.  Construction follows Laine et al. 2019
(arXiv:1901.10277): a shared HALF-PLANE branch whose receptive field is strictly above the
current row, applied to the four 90-degree rotations of the input; rotating back and
concatenating gives a receptive field = {rows above} U {rows below} U {cols left} U {cols right}
= everything except the centre.

Half-plane branch on a torus: roll the input down by one row (so row i holds row i-1), then every
conv is "causal upward": roll by one more row before a same-padded 3x3 conv, so its output at row i
sees input rows i-2..i.  After the initial roll and n causal convs the branch sees row offsets
1..(1+2n) above -- strictly above provided 1+2n < L, which caps n at 11 for L=25.  Circular padding
is then safe: offsets never reach 0 mod L.

Two things that would silently break the blind spot, and are therefore absent:
  * GroupNorm/BatchNorm -- their statistics mix ALL positions, leaking the centre.  Only a
    per-position channel norm is used.
  * any full-plane conv after the branch -- the head is 1x1 only.

Heads (both hollow, per site, over the two candidate values c in {0,1}):
  * score  q_theta^d(c | x^{-d}, t)   -- logits
  * phi    phi_theta^d(c | x^{-d}, t) -- raw
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

V = 2


def fourier(v, n=6, max_scale=2.0):
    freqs = torch.exp(torch.linspace(0.0, math.log(max_scale), n, device=v.device)) * math.pi
    a = v[:, None] * freqs[None, :]
    return torch.cat([torch.cos(a), torch.sin(a)], -1)


class ChannelNorm(nn.Module):
    """LayerNorm over channels at each position -- mixes nothing across space."""

    def __init__(self, ch):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, ch, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, ch, 1, 1))

    def forward(self, x):
        m = x.mean(1, keepdim=True)
        v = x.var(1, keepdim=True, unbiased=False)
        return (x - m) * torch.rsqrt(v + 1e-5) * self.g + self.b


class CausalConv(nn.Module):
    """3x3 circular conv whose output at row i sees input rows i-2, i-1, i only."""

    def __init__(self, cin, cout):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, padding=1, padding_mode="circular")

    def forward(self, x):
        return self.conv(torch.roll(x, 1, dims=2))


class CausalBlock(nn.Module):
    def __init__(self, ch, cond_dim):
        super().__init__()
        self.n1, self.c1 = ChannelNorm(ch), CausalConv(ch, ch)
        self.film = nn.Linear(cond_dim, 2 * ch)
        self.n2, self.c2 = ChannelNorm(ch), CausalConv(ch, ch)

    def forward(self, h, c):
        r = self.c1(F.silu(self.n1(h)))
        scale, shift = self.film(c)[:, :, None, None].chunk(2, dim=1)
        r = r * (1.0 + scale) + shift
        r = self.c2(F.silu(self.n2(r)))
        return h + r


class BlindSpotNet(nn.Module):
    def __init__(self, L=25, hidden=96, n_blocks=5, cond_dim=128, head_hidden=192):
        super().__init__()
        self.L, self.D = L, L * L
        n_causal = 1 + 2 * n_blocks
        assert 1 + 2 * n_causal < L, f"vertical reach {1+2*n_causal} must stay below L={L}"
        self.reach_rows, self.reach_cols = 1 + 2 * n_causal, n_causal
        self.cond = nn.Sequential(nn.Linear(2 * 6, cond_dim), nn.SiLU(),
                                  nn.Linear(cond_dim, cond_dim), nn.SiLU())
        self.inp = CausalConv(V, hidden)
        self.blocks = nn.ModuleList([CausalBlock(hidden, cond_dim) for _ in range(n_blocks)])
        self.norm = ChannelNorm(hidden)
        # 1x1 head over the four un-rotated branches
        self.head = nn.Sequential(nn.Conv2d(4 * hidden, head_hidden, 1), nn.SiLU(),
                                  nn.Conv2d(head_hidden, head_hidden, 1), nn.SiLU())
        self.head_film = nn.Linear(cond_dim, 2 * head_hidden)
        self.out_q = nn.Conv2d(head_hidden, V, 1)
        self.out_phi = nn.Conv2d(head_hidden, V, 1)
        nn.init.zeros_(self.out_q.weight); nn.init.zeros_(self.out_q.bias)
        nn.init.zeros_(self.out_phi.weight); nn.init.zeros_(self.out_phi.bias)

    def branch(self, x, c):
        h = self.inp(torch.roll(x, 1, dims=2))          # initial shift: row i now holds row i-1
        for blk in self.blocks:
            h = blk(h, c)
        return F.silu(self.norm(h))

    def forward(self, x, t):
        """x: [B, D] tokens in {0,1}; t: [B].  Returns (q_logits [B,D,2], phi [B,D,2])."""
        B = x.shape[0]
        z = F.one_hot(x, V).float().view(B, self.L, self.L, V).permute(0, 3, 1, 2)
        c = self.cond(fourier(t))
        rots = torch.cat([torch.rot90(z, k, dims=(2, 3)) for k in range(4)], 0)
        feats = self.branch(rots, c.repeat(4, 1))
        parts = feats.chunk(4, 0)
        unrot = [torch.rot90(p, -k, dims=(2, 3)) for k, p in enumerate(parts)]
        h = self.head(torch.cat(unrot, 1))
        scale, shift = self.head_film(c)[:, :, None, None].chunk(2, dim=1)
        h = h * (1.0 + scale) + shift
        q = self.out_q(h).permute(0, 2, 3, 1).reshape(B, self.D, V)
        phi = self.out_phi(h).permute(0, 2, 3, 1).reshape(B, self.D, V)
        return q, phi


@torch.no_grad()
def hollow_check(model, L, device, n_sites=12, seed=0):
    """Numerical proof of the blind spot: flipping input site d must leave BOTH heads at site d
    unchanged (and change them elsewhere).  Returns (max |change at d|, mean |change elsewhere|)."""
    g = torch.Generator(device=device).manual_seed(seed)
    x = (torch.rand(1, L * L, device=device, generator=g) < 0.5).long()
    t = torch.full((1,), 0.5, device=device)
    q0, p0 = model(x, t)
    worst, elsewhere = 0.0, []
    for d in torch.randint(0, L * L, (n_sites,), generator=g, device=device).tolist():
        x2 = x.clone(); x2[0, d] = 1 - x2[0, d]
        q1, p1 = model(x2, t)
        dq = (q1 - q0).abs(); dp = (p1 - p0).abs()
        worst = max(worst, float(dq[0, d].max()), float(dp[0, d].max()))
        elsewhere.append(float(dq[0].max()))
    return worst, sum(elsewhere) / len(elsewhere)
