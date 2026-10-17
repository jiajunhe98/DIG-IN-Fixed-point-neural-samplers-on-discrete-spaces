"""Masked-diffusion denoiser for Ising with field, amortised over (beta, h), MDNS convention.

The validated Ising backbone with one more FiLM scalar: h, normalised by h_norm to [-1, 1]
and encoded with the aliasing-safe low-frequency recipe (fourier + raw value), exactly like
beta.  Everything else unchanged: circular padding (exact PBC), zero-init head, fully conv.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

MASK = 2
VOCAB = 3
V = 2


def fourier(v, n=6, max_scale=2.0):
    freqs = torch.exp(torch.linspace(0.0, math.log(max_scale), n, device=v.device)) * math.pi
    a = v[:, None] * freqs[None, :]
    return torch.cat([torch.cos(a), torch.sin(a)], -1)


class Block(nn.Module):
    def __init__(self, ch, cond_dim, kernel, dilation=1):
        super().__init__()
        pad = dilation * (kernel // 2)
        self.n1 = nn.GroupNorm(8, ch)
        self.c1 = nn.Conv2d(ch, ch, kernel, padding=pad, dilation=dilation, padding_mode="circular")
        self.film = nn.Linear(cond_dim, 2 * ch)
        self.n2 = nn.GroupNorm(8, ch)
        self.c2 = nn.Conv2d(ch, ch, kernel, padding=pad, dilation=dilation, padding_mode="circular")

    def forward(self, h, c):
        r = self.c1(F.silu(self.n1(h)))
        scale, shift = self.film(c)[:, :, None, None].chunk(2, dim=1)
        r = r * (1.0 + scale) + shift
        r = self.c2(F.silu(self.n2(r)))
        return h + r


class FieldDenoiser(nn.Module):
    def __init__(self, L=25, hidden=96, n_blocks=7, kernel=3, cond_dim=128,
                 beta_range=(0.25, 0.65), h_norm=0.05, n_freq=6, max_scale=2.0,
                 dilations=None):
        super().__init__()
        self.L, self.D = L, L * L
        self.beta_range = beta_range
        self.h_norm = h_norm
        self.n_freq, self.max_scale = n_freq, max_scale
        self.inp = nn.Conv2d(VOCAB, hidden, kernel, padding=kernel // 2, padding_mode="circular")
        # cond = fourier(t) ++ fourier(beta)+raw ++ fourier(h)+raw
        self.cond = nn.Sequential(
            nn.Linear(2 * 6 + (2 * n_freq + 1) * 2, cond_dim), nn.SiLU(),
            nn.Linear(cond_dim, cond_dim), nn.SiLU())
        # The receptive field must reach L//2 to wrap the torus. Buying that with depth alone
        # costs 25 plain 3x3 blocks at L=50; dilations get there in 7.
        dil = list(dilations) if dilations else [1] * n_blocks
        assert len(dil) == n_blocks, (len(dil), n_blocks)
        assert max(dil) * (kernel // 2) < L, "circular padding must be smaller than the lattice"
        self.dilations = dil
        self.blocks = nn.ModuleList([Block(hidden, cond_dim, kernel, d) for d in dil])
        self.radius = (kernel // 2) * (1 + 2 * sum(dil))
        self.norm = nn.GroupNorm(8, hidden)
        self.out = nn.Conv2d(hidden, V, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, t, beta, h):
        """x: [B,D] tokens {0,1,MASK}; t, beta, h: [B].  -> [B,D,2] logits.

        beta and h are in the MDNS convention (see energy_mdns.py): the target is
        pi ~ exp(beta * (sum_<ij> s_i s_j + h * sum_i s_i)).
        """
        B = x.shape[0]
        z = F.one_hot(x, VOCAB).to(self.inp.weight.dtype).view(B, self.L, self.L, VOCAB).permute(0, 3, 1, 2)
        s0, s1 = self.beta_range
        sn = (beta - s0) / (s1 - s0)
        hn = h / self.h_norm
        c = self.cond(torch.cat([fourier(t), fourier(sn, self.n_freq, self.max_scale), sn[:, None],
                                 fourier(hn, self.n_freq, self.max_scale), hn[:, None]], -1))
        z = self.inp(z)
        for blk in self.blocks:
            z = blk(z, c)
        return self.out(F.silu(self.norm(z))).permute(0, 2, 3, 1).reshape(B, self.D, V)


def reveal_edges(D, n_steps, n_single=0):
    n_single = min(n_single, n_steps - 1, D - 1)
    if n_single <= 0:
        edges = torch.linspace(0, D, n_steps + 1).round().long().tolist()
    else:
        rest = torch.linspace(n_single, D, n_steps - n_single + 1).round().long().tolist()
        edges = list(range(n_single)) + rest
    assert len(edges) == n_steps + 1 and edges[0] == 0 and edges[-1] == D
    return edges


@torch.no_grad()
def unmask_sample(model, beta, h, n, n_steps, device, D=625, n_single=0, autocast=True,
                  return_logq=False):
    """Random-order unmasking.  With return_logq, also accumulates the path log-density
    log q_path(x) = sum over reveals of log P(sampled token) -- for absorbing masking the
    uniform order probability cancels against the target's, so this is the full proposal
    density needed by the importance weight log w = log pi_tilde - log q_path."""
    x = torch.full((n, D), MASK, dtype=torch.long, device=device)
    order = torch.argsort(torch.rand(n, D, device=device), dim=1)
    bet = (beta if torch.is_tensor(beta) else torch.full((n,), float(beta), device=device)).float().expand(n)
    hh = (h if torch.is_tensor(h) else torch.full((n,), float(h), device=device)).float().expand(n)
    edges = reveal_edges(D, n_steps, n_single)
    logq = torch.zeros(n, device=device, dtype=torch.float64)
    for k in range(n_steps):
        lo, hi = edges[k], edges[k + 1]
        if hi == lo:
            continue
        t = torch.full((n,), lo / D, device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            logits = model(x, t, bet, hh).float()
        idx = order[:, lo:hi]
        sel = torch.gather(logits, 1, idx[:, :, None].expand(-1, -1, V))
        p1 = torch.softmax(sel, -1)[..., 1]
        tok = (torch.rand_like(p1) < p1).long()
        if return_logq:
            p_tok = torch.where(tok > 0, p1, 1.0 - p1).clamp_min(1e-12)
            logq += p_tok.log().sum(-1).double()
        x.scatter_(1, idx, tok)
    return (x, logq) if return_logq else x


@torch.no_grad()
def md4_sample(model, beta, h, n, n_steps, device, D=2500, autocast=True, return_logq=False,
               t_start=1.0, t_end=0.0):
    """MD4 / MDLM ancestral sampling from the absorbing (masked) forward process.

    The forward process masks each site independently with probability t, so the reverse step
    from t to s < t unmasks each STILL-MASKED site independently with probability

        p_unmask = (t - s) / t ,

    drawing its value from the model's predicted categorical at that site.  Sites already
    revealed carry over unchanged.  This is the sampler the MD4 and MDLM papers use, and it is
    exactly the posterior of the forward process, so the number revealed per step is BINOMIAL
    rather than fixed.

    Difference from `unmask_sample` in this file: that one fixes the reveal count per step to
    its mean and draws a single uniform order upfront, which conditioned on the count gives the
    same distribution over revealed subsets, but lets the caller spend a chosen number of
    single-site reveals at high mask fraction (`n_single`) where factorised chunk draws hurt
    most.  MD4 spreads its reveals uniformly in time.  Neither reduces to the other: at
    n_steps = D `unmask_sample` becomes exact sequential unmasking, while MD4 still reveals a
    random count per step and only approaches exactness as n_steps -> infinity.

    return_logq: accumulate log q_path(x_1) = sum over reveals of log P(drawn value).  The
    unmask pattern is drawn from a model-independent law, exactly as the reveal order is in
    `unmask_sample`, so it cancels between proposal and target on the extended space and
    log w = log pi~(x_1) - log q_path is a valid importance log-weight for this sampler too.
    """
    x = torch.full((n, D), MASK, dtype=torch.long, device=device)
    bet = (beta if torch.is_tensor(beta) else torch.full((n,), float(beta), device=device)).float().expand(n)
    hh = (h if torch.is_tensor(h) else torch.full((n,), float(h), device=device)).float().expand(n)
    ts = torch.linspace(t_start, t_end, n_steps + 1, device=device)
    logq = torch.zeros(n, dtype=torch.float64, device=device)
    for k in range(n_steps):
        t, s = float(ts[k]), float(ts[k + 1])
        masked = x.eq(MASK)
        if not bool(masked.any()):
            break
        # the model conditions on the FRACTION REVEALED, matching how it was trained
        frac = (~masked).float().mean(-1)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast and x.is_cuda):
            logits = model(x, frac, bet, hh).float()
        p_un = 1.0 if t <= 0 else max(0.0, min(1.0, (t - s) / t))
        pick = masked & (torch.rand(n, D, device=device) < p_un)
        if not bool(pick.any()):
            continue
        p = torch.softmax(logits, -1)
        u = torch.rand(n, D, 1, device=device)
        v = (p.cumsum(-1) < u).sum(-1).clamp_max(V - 1)
        x = torch.where(pick, v, x)
        if return_logq:
            pv = p.gather(-1, v[..., None]).squeeze(-1).double().clamp_min(1e-300)
            logq += (torch.log(pv) * pick.double()).sum(-1)
    # the final step must leave nothing masked; if the schedule did not reach t=0, finish it
    masked = x.eq(MASK)
    if bool(masked.any()):
        frac = (~masked).float().mean(-1)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast and x.is_cuda):
            logits = model(x, frac, bet, hh).float()
        p = torch.softmax(logits, -1)
        u = torch.rand(n, D, 1, device=device)
        v = (p.cumsum(-1) < u).sum(-1).clamp_max(V - 1)
        x = torch.where(masked, v, x)
        if return_logq:
            pv = p.gather(-1, v[..., None]).squeeze(-1).double().clamp_min(1e-300)
            logq += (torch.log(pv) * masked.double()).sum(-1)
    return (x, logq) if return_logq else x
