"""The DIG-IN bridge between two Boltzmann distributions (paper Sec. 3.3, App. C, F.1).

Interpolant, per coordinate, with independent endpoints X_0 ~ p_0, X_1 ~ p_1:

    kappa_t(c | x0, x1) = (1 - t) 1[c = x0] + t 1[c = x1],   d_t kappa = 1[c = x1] - 1[c = x0]

Two hollow heads, both functions of (x^{-d}, t):

    q^d(c)    -> p_t(x^d = c | x^{-d})              soft cross-entropy against kbar     (eq. 125)
    phi^d(c)  -> d_t log p_t(x^d = c | x^{-d})      weighted quadratic, no division      (eq. 126)

Fixed point: X_1 comes from the current sampler, and coordinate d's endpoint
value is replaced by the target's heat-bath conditional rho_d(a | X_1^{-d})
(eq. 27).  Since the head never sees X_t^d, and X_t^{-d} does not depend on
X_1^d, the substitution is consistent:

    kbar^d(c) = (1 - t) 1[c = X_0^d] + t rho_d(c),      dkbar^d(c) = rho_d(c) - 1[c = X_0^d]

Sampler: the minimum-action CTMC of the two heads (eqs. 23-24),

    forward   x^d -> c   at rate  q^d(c) ([phi^d(c) - phi^d(x^d)]_+ + g)
    backward  y^d -> c   at rate  q^d(c) ([phi^d(y^d) - phi^d(c)]_+ + g)

tau-leaped on a uniform grid, every coordinate stepping independently given
the current state.  The path weight (eq. 34) uses the forward kernel actually
sampled and the backward kernel at the next state and time; the heads are
hollow, so the call at (x_{k+1}, t_{k+1}) serves both the backward kernel of
step k and the forward kernel of step k+1.

The Gibbs rate g (the Ising code's `gibbs_rate`) is the reversible part of the
generator: heat-bath moves under q^d, which leave p_t unchanged.  It is needed
for E[w] = Z_1 / Z_0.  With g = 0 a forward rate is exactly zero for every
downhill jump, while the backward kernel, evaluated at another state and
time, can still propose its reverse; those paths are never sampled and their
mass is missing from E[w].  On the enumerable toy of tests/test_bridge.py the
g = 0 estimate is 16 standard errors low, g = 0.3 is within one.

This follows DIG-IN/fixed_point/bridge_ising/train_bridge.py, the Ising
bridge of App. F.1, with L = 64 categories instead of 2.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

LOG_FLOOR = math.log(1e-300)


def _heads(net, x: Tensor, t: float):
    tt = torch.full((x.shape[0],), t, device=x.device)
    ql, phi = net(x, tt)
    return ql.float().softmax(-1), phi.float()


def _step_logprob(R: Tensor, src: Tensor, dst: Tensor, dt: float) -> Tensor:
    """log P(src -> dst) of one tau-leap step per sample: R (n, D, L) are the
    rates out of src (zero at src itself), every coordinate moves independently."""
    R = R.double()
    lam = R.sum(-1)
    moved = dst != src
    r_dst = R.gather(-1, dst.unsqueeze(-1)).squeeze(-1)
    jump = torch.log(-torch.expm1(-lam * dt)) + torch.log(r_dst) - torch.log(lam)
    lp = torch.where(moved, torch.nan_to_num(jump, nan=LOG_FLOOR, neginf=LOG_FLOOR).clamp_min(LOG_FLOOR), -lam * dt)
    return lp.sum(-1)


@torch.no_grad()
def ctmc_sample(net, x0: Tensor, K: int, path_weight: bool = False, gibbs_rate: float = 0.0):
    """tau-leap from x0 (n, D) at t = 0 to t = 1 in K steps.

    Returns (x1, log B - log F of the path or None, number of coordinate jumps per sample)."""
    x = x0.clone()
    n = x.shape[0]
    dt = 1.0 / K
    log_ratio = torch.zeros(n, dtype=torch.float64, device=x.device) if path_weight else None
    jumps = torch.zeros(n, device=x.device)
    prev = None
    for k in range(K + 1):
        q, phi = _heads(net, x, k / K)
        here = x.unsqueeze(-1)
        phi_x = phi.gather(-1, here)
        if path_weight and prev is not None:
            Rb = (q * ((phi_x - phi).clamp_min(0.0) + gibbs_rate)).scatter(-1, here, 0.0)   # x_k -> x_{k-1}
            log_ratio += _step_logprob(Rb, x, prev, dt)
        if k == K:
            break
        R = (q * ((phi - phi_x).clamp_min(0.0) + gibbs_rate)).scatter(-1, here, 0.0)
        lam = R.sum(-1)
        move = torch.rand_like(lam) < -torch.expm1(-lam * dt)
        u = torch.rand_like(lam).unsqueeze(-1) * lam.unsqueeze(-1)
        c = (R.cumsum(-1) < u).sum(-1).clamp(max=R.shape[-1] - 1)
        nxt = torch.where(move, c, x)
        if path_weight:
            log_ratio -= _step_logprob(R, x, nxt, dt)
        jumps += (nxt != x).float().sum(-1)
        prev, x = x, nxt
    return x, log_ratio, jumps


def bridge_labels(x0: Tensor, rho: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
    """x0: (B, D) bins, rho: (B, D, L) target conditionals at X_1, t: (B,) -> kbar, dkbar (B, D, L)."""
    one = torch.zeros_like(rho).scatter_(-1, x0.unsqueeze(-1), 1.0)
    tt = t.view(-1, 1, 1)
    return (1.0 - tt) * one + tt * rho, rho - one


def interpolate(x0: Tensor, x1: Tensor, t: Tensor) -> Tensor:
    """X_t^d = X_1^d with probability t, else X_0^d, independently per coordinate."""
    return torch.where(torch.rand(x0.shape, device=x0.device) < t.view(-1, 1), x1, x0)


def losses(net, teacher, xt: Tensor, t: Tensor, kbar: Tensor, dk: Tensor, w: Tensor, eta: float,
           lam_phi: float = 1.0):
    """Weighted fit of both heads, damped towards the teacher (EMA) with weight 1 - eta.

    w: (B,) importance weights with mean 1; they weigh the fit terms only, the
    proximal terms tie the iterate to its predecessor."""
    ql, phi = net(xt, t)
    logq = ql.float().log_softmax(-1)
    phi = phi.float()
    ce_fit = -(kbar * logq).sum(-1).mean(-1)                               # (B,)
    wq_fit = (0.5 * kbar * phi * phi - dk * phi).sum(-1).mean(-1)          # (B,)
    if teacher is not None and eta < 1.0:
        with torch.no_grad():
            ql_t, phi_t = teacher(xt, t)
            q_t = ql_t.float().softmax(-1)
            phi_t = phi_t.float()
        ce_prox = -(q_t * logq).sum(-1).mean(-1)
        wq_prox = 0.5 * (kbar * (phi - phi_t) ** 2).sum(-1).mean(-1)
        L_q = (eta * w * ce_fit + (1.0 - eta) * ce_prox).mean()
        L_phi = (eta * w * wq_fit + (1.0 - eta) * wq_prox).mean()
    else:
        L_q = (w * ce_fit).mean()
        L_phi = (w * wq_fit).mean()
    return L_q + lam_phi * L_phi, float(L_q.detach()), float(L_phi.detach())
