"""Tau-leaping simulation of the zeta-family CTMC, with forward-backward
Radon-Nikodym weights and optional predictor-corrector refinement.

The learned fields enter through `fields_fn(x, t) -> (v, vb, s)`, each of
shape (B, D, N):

    v  — forward velocity rates for moves d -> n  (0 on the diagonal)
    vb — backward (time-reversed) velocity rates  (0 on the diagonal)
    s  — score ratios rho_t(x^{d<-n}) / rho_t(x)  (1 on the diagonal)

The simulated rate is u = v + zeta_t * Q * (1 + s) with uniform mobility
Q = 1/N; the reverse-time rate is ub = vb + zeta_t * Q * (1 + s).
Each tau-leap step is an explicit product-categorical Markov kernel, so the
pathwise weight  log w = log nu_hat(x_M) - log mu(x_0) + sum log(bwd/fwd)
satisfies E[exp(log w)] = Z exactly (Prop. iw), for any step count.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

_EPS = 1e-12


def balancing(name: str):
    if name == "sum":
        return lambda a: 1.0 + a
    if name == "mh":
        return lambda a: torch.clamp(a, max=1.0)
    if name == "barker":
        return lambda a: a / (1.0 + a)
    if name == "sqrt":
        return lambda a: torch.sqrt(a)
    raise ValueError(name)


def _step_probs(rates: Tensor, x: Tensor, h: float, cap: float) -> Tensor:
    """Rates (B, D, N) with zero diagonal -> full per-site transition
    probabilities (B, D, N) including the stay probability at n = x^d."""
    p = rates * h
    row = p.sum(-1, keepdim=True)
    p = p * (cap / row.clamp_min(_EPS)).clamp(max=1.0)
    stay = (1.0 - p.sum(-1, keepdim=True)).clamp_min(_EPS)
    return p.scatter(2, x.unsqueeze(-1), stay)


def _log_trans_prob(probs: Tensor, y: Tensor) -> Tensor:
    """log prob of the realized per-site transitions, summed over sites."""
    return (
        torch.gather(probs.clamp_min(_EPS).log(), 2, y.unsqueeze(-1))
        .squeeze(-1)
        .sum(dim=1)
    )


def _zero_diag(r: Tensor, x: Tensor) -> Tensor:
    return r.scatter(2, x.unsqueeze(-1), torch.zeros_like(x, dtype=r.dtype).unsqueeze(-1))


@torch.no_grad()
def tau_leap(
    fields_fn,
    x0: Tensor,
    n_steps: int,
    zeta,
    num_states: int,
    cap: float = 0.95,
    with_rnd: bool = False,
    corrector_steps: int = 0,
    corrector_delta: float = 0.5,
    corrector_g: str = "mh",
    generator=None,
) -> tuple[Tensor, Tensor]:
    """Simulate from t=0 to t=1 on a uniform grid of n_steps steps.

    Returns (x_1, log_fb) where log_fb accumulates
    sum_i [log P_bwd(x_i | x_{i+1}) - log P_fwd(x_{i+1} | x_i)]
    (zeros if with_rnd=False).
    """
    N = num_states
    zf = zeta if callable(zeta) else (lambda _t: zeta)
    g = balancing(corrector_g)
    h = 1.0 / n_steps
    B = x0.shape[0]
    dev = x0.device

    x = x0
    t = torch.zeros(B, device=dev)
    v, vb, s = fields_fn(x, t)
    log_fb = torch.zeros(B, device=dev)

    for i in range(n_steps):
        t_next = (i + 1) * h
        # ---- predictor ----
        u = _zero_diag(v + zf(i * h) / N * (1.0 + s), x)
        probs = _step_probs(u, x, h, cap)
        x_new = (
            torch.multinomial(probs.view(B * x.shape[1], N), 1, generator=generator)
            .view(B, -1)
        )
        tv = torch.full((B,), t_next, device=dev)
        v2, vb2, s2 = fields_fn(x_new, tv)
        if with_rnd:
            log_pf = _log_trans_prob(probs, x_new)
            ub = _zero_diag(vb2 + zf(t_next) / N * (1.0 + s2), x_new)
            probs_b = _step_probs(ub, x_new, h, cap)
            log_pb = _log_trans_prob(probs_b, x)
            log_fb = log_fb + log_pb - log_pf
        x, v, vb, s = x_new, v2, vb2, s2

        # ---- corrector (frozen time t_next) ----
        for _ in range(corrector_steps):
            c = _zero_diag(g(s) / N, x)
            probs = _step_probs(c, x, corrector_delta, cap)
            x_new = (
                torch.multinomial(probs.view(B * x.shape[1], N), 1, generator=generator)
                .view(B, -1)
            )
            v2, vb2, s2 = fields_fn(x_new, tv)
            if with_rnd:
                log_pf = _log_trans_prob(probs, x_new)
                cb = _zero_diag(g(s2) / N, x_new)
                probs_b = _step_probs(cb, x_new, corrector_delta, cap)
                log_pb = _log_trans_prob(probs_b, x)
                log_fb = log_fb + log_pb - log_pf
            x, v, vb, s = x_new, v2, vb2, s2

    return x, log_fb


def log_weights_uniform_prior(
    target_log_prob: Tensor, log_fb: Tensor, dim: int, num_states: int
) -> Tensor:
    """log w = log nu_hat(x1) - log mu(x0) + log_fb with uniform mu.
    E[exp(log w)] = Z."""
    return target_log_prob + dim * math.log(num_states) + log_fb
