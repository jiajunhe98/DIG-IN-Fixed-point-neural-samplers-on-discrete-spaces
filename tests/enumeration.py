"""Brute-force enumeration helpers used by the exactness tests."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from dig_in.methods.dsi.interpolant import MixtureInterpolant
from dig_in.targets.base import DiscreteTarget


def all_states(target: DiscreteTarget) -> tuple[Tensor, Tensor]:
    """(xs (S, D), nu (S,)) for the full state space."""
    S = target.N**target.D
    xs = target._index_to_state(torch.arange(S))
    nu = torch.softmax(target.log_prob(xs), 0)
    return xs, nu


def pair_posterior(
    interp: MixtureInterpolant, target: DiscreteTarget, xs: Tensor, nu: Tensor,
    x: Tensor, t: Tensor,
):
    """Exact rho_t(x), full endpoint posterior (S, S), and per-site pair
    posterior (D, N, N) for the uniform-prior independent coupling."""
    S, D = xs.shape
    N = target.N
    kappa, _ = interp._kernel_tensors(t.view(1))
    lk = kappa[0].clamp_min(1e-300).log()  # (N, N, N) [c0, c1, m]
    term = torch.zeros(S, S)
    for d in range(D):
        term += lk[xs[:, d]][:, xs[:, d], x[d]]
    w = (term + nu.log().view(1, S)).exp() / S
    rho = w.sum()
    post = w / rho
    E = F.one_hot(xs, N).double()
    pd = torch.einsum("sdn,sz,zdm->dnm", E, post, E)
    return rho, post, pd


def exact_fields_fn(interp: MixtureInterpolant, target: DiscreteTarget,
                    xs: Tensor, nu: Tensor):
    """fields_fn(x, t) computing the exact posterior fields, batched."""
    S = xs.shape[0]
    N = target.N
    log_nu = target.log_prob(xs)
    E = F.one_hot(xs, N).double()

    def fn(xb: Tensor, tb: Tensor):
        B, D = xb.shape
        t = tb[0]
        kappa, _ = interp._kernel_tensors(t.view(1))
        lk = kappa[0].clamp_min(1e-300).log()
        term = torch.zeros(B, S, S)
        for d in range(D):
            lkd = lk[xs[:, d]][:, xs[:, d], :]
            term += lkd[:, :, xb[:, d]].permute(2, 0, 1)
        w = term + log_nu.view(1, 1, S)
        w = (w - w.flatten(1).logsumexp(1).view(B, 1, 1)).exp()
        pd = torch.einsum("sdn,bsz,zdm->bdnm", E, w, E)
        return interp.posterior_fields(pd.clamp_min(1e-300).log(), xb, t.expand(B))

    return fn


def exact_masked_logits_fn(target: DiscreteTarget, xs: Tensor):
    """logits_fn(x, a) for the exact masked denoiser q^d(n | visible)."""
    N = target.N
    log_nu = target.log_prob(xs)
    E = F.one_hot(xs, N).double()

    def fn(xb: Tensor, ab: Tensor):
        ok = (xs.unsqueeze(0) == xb.unsqueeze(1)) | (xb == N).unsqueeze(1)
        match = ok.all(dim=2)
        w = torch.where(match, log_nu.view(1, -1), torch.tensor(-1e30).double())
        w = (w - w.logsumexp(1, keepdim=True)).exp()
        q = torch.einsum("bs,sdn->bdn", w, E)
        return q.clamp_min(1e-300).log()

    return fn
