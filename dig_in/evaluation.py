"""Shared evaluation: log Z estimates, ESS, observables, reference
comparisons, and reference-sample management."""

from __future__ import annotations

import hashlib
import math
import os
import time

import torch
from torch import Tensor

from dig_in.methods.base import Method
from dig_in.targets import DiscreteTarget, get_mcmc
from dig_in.utils import metrics


def reference_cache_key(target: DiscreteTarget) -> str:
    parts = [
        type(target).__name__,
        str(tuple(target.lattice_shape)),
        f"N{target.N}",
        f"beta{target.beta:.6g}",
    ]
    for attr in ("J", "h", "q"):
        if hasattr(target, attr):
            parts.append(f"{attr}{getattr(target, attr):.6g}")
    # a target whose identity is not (J, h, q) declares its own key parts;
    # a cache hit on a differently-parameterised target would silently score
    # a sampler against the wrong reference
    if hasattr(target, "cache_key_parts"):
        parts.extend(target.cache_key_parts())
    return hashlib.md5("_".join(parts).encode()).hexdigest()[:16]


def get_reference_samples(
    target: DiscreteTarget,
    n: int = 4096,
    cache_dir: str | None = "data/reference",
    method: str = "mixed",
    burn_in: int = 500,
    thin: int = 5,
    device: str = "cpu",
) -> Tensor:
    """Reference MCMC samples, cached on disk."""
    path = None
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, f"{reference_cache_key(target)}_{n}.pt")
        if os.path.exists(path):
            return torch.load(path, map_location="cpu", weights_only=True)
    mcmc = get_mcmc(target, device=device)
    x = mcmc.sample(n, n_chains=min(256, n), method=method, burn_in=burn_in, thin=thin)
    x = x.cpu()
    if path is not None:
        # Concurrent jobs on a shared filesystem generate the same grid points at
        # the same time.  A plain torch.save races: one process reads a
        # half-written file and the run dies, or worse, trains against
        # garbage.  Writing to a process-unique temp and renaming makes the
        # worst case duplicated work instead of corruption.
        tmp = f"{path}.{os.getpid()}.tmp"
        torch.save(x, tmp)
        os.replace(tmp, path)
    return x


@torch.no_grad()
def evaluate(
    method: Method,
    n_samples: int = 2048,
    x_ref: Tensor | None = None,
    exact_log_Z: float | None = None,
    ref: dict | None = None,
    keep_samples: bool = False,
    smc: bool = True,
    smc_ess_threshold: float = 0.5,
) -> dict:
    """Standard evaluation block; returns a flat metrics dict.

    `keep_samples` additionally returns the scored draw itself under the
    non-metric key `_samples`, so a caller that wants to plot the distribution
    does not have to sample a second time -- and, more to the point, does not
    spend a second round of energy calls that would then not match the cost
    the history reports.

    `ref` is `metrics.reference_summary(target, x_ref)`; pass it when the
    same reference is scored repeatedly so it is not recomputed each time.
    """
    target = method.target
    t0 = time.time()
    x, log_w = method.sample_with_weights(n_samples)
    if x.is_cuda:
        torch.cuda.synchronize(x.device)
    out = {
        "eval/n_samples": float(n_samples),
        "eval/sample_time_s": time.time() - t0,
        "eval/ess": metrics.ess(log_w),
        "eval/log_w_std": log_w.std().item(),
    }
    if method.unbiased_log_Z:
        out["eval/log_Z"] = metrics.log_Z_is(log_w)
        out["eval/log_Z_lb"] = metrics.log_Z_lower_bound(log_w)
        # delta-method standard error of the IS estimate, sqrt(1/ESS - 1/n)
        # with the unnormalized ESS: how far the estimate can be trusted
        n = log_w.shape[0]
        ess_n = out["eval/ess"] * n
        out["eval/log_Z_se"] = math.sqrt(max(1.0 / ess_n - 1.0 / n, 0.0))
        if exact_log_Z is not None:
            out["eval/log_Z_err"] = abs(out["eval/log_Z"] - exact_log_Z)
            out["eval/log_Z_exact"] = exact_log_Z
            # log Z - E[log w] >= 0 is KL(q || nu) for a tractable-density
            # sampler and the path-space KL for a path-weight sampler; the
            # one criterion that is exact, scale-free and zero only at the
            # target, so it is the headline number where log Z is known
            out["eval/elbo_gap"] = exact_log_Z - out["eval/log_Z_lb"]
    else:
        # the method's weights are a path-measure RND, not a Z estimator
        # (e.g. ASBS in non-memoryless mode: -> 0 at convergence)
        out["eval/log_w_logmeanexp"] = metrics.log_Z_is(log_w)
        out["eval/log_w_mean"] = metrics.log_Z_lower_bound(log_w)
    out.update(metrics.summarize_observables(target, x, prefix="eval/", log_w=log_w))
    if x_ref is not None:
        x_ref = x_ref.to(x.device)
        out.update(
            metrics.compare_to_reference(
                target, x, x_ref, prefix="eval/", log_w=log_w, ref=ref)
        )
        # forward KL(nu || q) on the reference sample, for samplers with a
        # tractable marginal density (the autoregressive baseline).  It is
        # the criterion that blows up when a mode is dropped, which the
        # reverse direction is blind to.
        if exact_log_Z is not None and hasattr(method, "log_density"):
            log_q = method.log_density(x_ref)
            log_nu = target.log_prob(x_ref) - exact_log_Z
            out["eval/kl_forward"] = (log_nu - log_q).mean().item()
    # Twisted sequential Monte Carlo along the generation order, where the
    # method and the target both support it.  Plain importance sampling scores a
    # configuration only when it is finished, so its effective sample size
    # decays with the dimension; an autoregressive model can be resampled at
    # every block against the energy of what it has placed so far.  Reported
    # beside the plain numbers rather than instead of them, because the pair is
    # the interesting quantity: the gap between them is what the resampling
    # bought.
    # the causal sampler twists on a prefix energy, the any-order one on the
    # energy of a revealed subset, and the any-order one also needs its reveal
    # block to be whole units of the target.  The method answers all of that in
    # one call, so a run that cannot be resampled is scored with plain weights
    # and says nothing about it, rather than raising once per evaluation.
    if smc and method.supports_smc():
        try:
            t1 = time.time()
            x_s, lw_s, info = method.sample_with_weights_smc(
                n_samples, ess_threshold=smc_ess_threshold)
            w_s = torch.softmax(lw_s.double(), 0)
            out["eval/smc_log_Z"] = info["log_Z"]
            out["eval/smc_ess"] = info["ess_final"]
            out["eval/smc_ess_min"] = info["ess_min"]
            out["eval/smc_n_resample"] = float(info["n_resample"])
            out["eval/smc_time_s"] = time.time() - t1
            if exact_log_Z is not None:
                out["eval/smc_log_Z_err"] = abs(info["log_Z"] - exact_log_Z)
            obs = getattr(target, "observables", None)
            if obs is not None:
                for k, v in obs(x_s).items():
                    out[f"eval/smc_{k}_rw"] = (w_s * v.detach().double()).sum().item()
        except Exception as e:  # noqa: BLE001 - a diagnostic is never fatal
            print(f"[smc] evaluation failed, continuing: {e!r}")
    if keep_samples:
        out["_samples"] = (x.detach().cpu(), log_w.detach().float().cpu())
    return out
