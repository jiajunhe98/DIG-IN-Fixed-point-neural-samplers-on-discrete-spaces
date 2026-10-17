"""Uniform reference CTMC for ASBS: generator gamma(t) (1/N 11^T - I).

Closed-form transition kernel between times s < t with
gbar = int_s^t gamma(u) du:

    p^r_{t|s}(y | x) = B^D * C^{d_H(x,y)},
    B = (1 + (N-1) e^{-gbar}) / N        (per-site stay probability),
    B*C = (1 - e^{-gbar}) / N            (per-site probability of any
                                          specific different value).
"""

from __future__ import annotations

import torch
from torch import Tensor

from dds.utils.schedules import NoiseSchedule


def _bc(noise: NoiseSchedule, s: Tensor, t: Tensor, N: int) -> tuple[Tensor, Tensor]:
    e = torch.exp(-noise.integral(s, t))
    B = (1 + (N - 1) * e) / N
    C = (1 - e) / (1 + (N - 1) * e)
    return B, C


def ref_trans_log_p(
    noise: NoiseSchedule, s: Tensor, t: Tensor, x: Tensor, y: Tensor, N: int
) -> Tensor:
    """log p^r_{t|s}(y|x); s, t (B,), x, y (B, D).  Returns (B,)."""
    B_, C_ = _bc(noise, s, t, N)
    ham = (x != y).sum(dim=-1)
    return x.shape[-1] * B_.log() + ham * C_.clamp_min(1e-30).log()


def ref_trans_log_p_diff(
    noise: NoiseSchedule, s: Tensor, t: Tensor, x: Tensor, y: Tensor,
    N: int, forward: bool,
) -> Tensor:
    """Per-site log-ratio tables of the reference kernel, shape (B, D, N).

    forward=True :  log p^r_{t|s}(y^{d<-n} | x) - log p^r_{t|s}(y | x)
    forward=False:  log p^r_{t|s}(y | x^{d<-n}) - log p^r_{t|s}(y | x)

    Both equal (1{n != z^d} - 1{x^d != y^d}) * log C with z = x (forward)
    or z = y (backward).
    """
    _, C_ = _bc(noise, s, t, N)
    log_c = C_.clamp_min(1e-30).log().view(-1, 1, 1)
    z = x if forward else y
    ind1 = z.unsqueeze(-1) != torch.arange(N, device=x.device).view(1, 1, N)
    ind2 = (x != y).unsqueeze(-1)
    return (ind1.float() - ind2.float()) * log_c


def sample_ref_bridge(
    noise: NoiseSchedule, t: Tensor, x0: Tensor, x1: Tensor, N: int, generator=None
) -> Tensor:
    """x_t ~ p^r_{t|0,1}(. | x0, x1), independently per site.  t (B,)."""
    zeros = torch.zeros_like(t)
    ones = torch.ones_like(t)
    e0t = torch.exp(-noise.integral(zeros, t))
    et1 = torch.exp(-noise.integral(t, ones))
    e01 = torch.exp(-noise.integral(zeros, ones))
    A0t = ((1 - e0t) / N).view(-1, 1, 1)
    B0t = ((1 + (N - 1) * e0t) / N).view(-1, 1, 1)
    At1 = ((1 - et1) / N).view(-1, 1, 1)
    Bt1 = ((1 + (N - 1) * et1) / N).view(-1, 1, 1)
    A01 = ((1 - e01) / N).view(-1, 1, 1)
    B01 = ((1 + (N - 1) * e01) / N).view(-1, 1, 1)

    n = torch.arange(N, device=x0.device).view(1, 1, N)
    n_eq_x0 = x0.unsqueeze(-1) == n
    n_eq_x1 = x1.unsqueeze(-1) == n
    eq = (x0 == x1).unsqueeze(-1)

    probs = (~eq) * (
        A0t * At1 / A01 * (~n_eq_x0 & ~n_eq_x1)
        + B0t * At1 / A01 * n_eq_x0
        + A0t * Bt1 / A01 * n_eq_x1
    ) + eq * (A0t * At1 / B01 * (~n_eq_x0) + B0t * Bt1 / B01 * n_eq_x0)
    flat = probs.flatten(0, 1)
    idx = torch.multinomial(flat, 1, generator=generator)
    return idx.view(x0.shape)
