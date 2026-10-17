"""Evaluation metrics shared by all methods.

Conventions: `log_w` are unnormalized log importance weights of a batch of
samples; log Z estimators assume E[exp(log_w)] = Z (weights already include
any prior/proposal normalization).
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor


# ----------------------------------------------------------------------
# Importance weights
# ----------------------------------------------------------------------
def ess(log_w: Tensor, normalize: bool = True) -> float:
    """Effective sample size of importance weights, in (0, 1] if normalized."""
    w = torch.softmax(log_w.detach().double(), dim=0)
    out = 1.0 / (w**2).sum()
    return (out / log_w.shape[0]).item() if normalize else out.item()


def log_Z_is(log_w: Tensor) -> float:
    """Unbiased IS estimator: log( mean exp(log_w) )."""
    n = log_w.shape[0]
    return (torch.logsumexp(log_w.detach().double(), dim=0) - math.log(n)).item()


def log_Z_lower_bound(log_w: Tensor) -> float:
    """Jensen lower bound: mean(log_w) <= log Z."""
    return log_w.detach().double().mean().item()


# ----------------------------------------------------------------------
# Distribution comparisons
# ----------------------------------------------------------------------
def _np(x) -> np.ndarray:
    return np.asarray(x.detach().cpu() if torch.is_tensor(x) else x).ravel()


def _weighted_quantile(v: np.ndarray, w: np.ndarray | None, qs: np.ndarray) -> np.ndarray:
    """Quantiles of a (possibly weighted) 1-D empirical distribution.

    Uses the midpoint plotting position (i + 1/2) / n, which stays monotone for
    any positive weights.  Uniform weights go through the same branch so that a
    weighted and an unweighted distance are computed by the same estimator and
    differ only because of the weights (np.quantile's (i)/(n-1) convention would
    otherwise introduce an O(1/n) offset between them).
    """
    idx = np.argsort(v)
    v = v[idx]
    w = np.full(len(v), 1.0 / len(v)) if w is None else w[idx]
    cw = np.cumsum(w) / w.sum()
    cw = cw - 0.5 * w / w.sum()
    return np.interp(qs, cw, v)


def wasserstein1d(
    a: Tensor | np.ndarray, b: Tensor | np.ndarray, p: int = 2,
    wa: np.ndarray | None = None, wb: np.ndarray | None = None,
) -> float:
    """W_p distance between two 1-D empirical distributions.

    `wa`/`wb` are optional per-sample probability weights (e.g. self-normalized
    importance weights); None means uniform.
    """
    a, b = _np(a), _np(b)
    n = max(len(a), len(b))
    qs = (np.arange(n) + 0.5) / n
    aq = _weighted_quantile(a, wa, qs)
    bq = _weighted_quantile(b, wb, qs)
    return float(np.mean(np.abs(aq - bq) ** p) ** (1.0 / p))


def snis_weights(log_w: Tensor) -> Tensor:
    """Self-normalized importance weights, summing to 1."""
    return torch.softmax(log_w.detach().double(), dim=0)


def histogram_tv(
    a: Tensor | np.ndarray, b: Tensor | np.ndarray, bins: int = 50,
    range_: tuple[float, float] | None = None,
    wa: np.ndarray | None = None, wb: np.ndarray | None = None,
) -> float:
    """Total variation between histograms of two scalar samples."""
    a, b = _np(a), _np(b)
    if range_ is None:
        lo = min(a.min(), b.min())
        hi = max(a.max(), b.max())
        range_ = (lo, hi)
    pa, _ = np.histogram(a, bins=bins, range=range_, weights=wa, density=False)
    pb, _ = np.histogram(b, bins=bins, range=range_, weights=wb, density=False)
    pa = pa / pa.sum()
    pb = pb / pb.sum()
    return float(0.5 * np.abs(pa - pb).sum())


# ----------------------------------------------------------------------
# Lattice observables
# ----------------------------------------------------------------------
def two_point_correlation(x: Tensor, lattice_shape: tuple[int, ...], mode: str = "spin") -> Tensor:
    """Connected two-point function C(r) along the axes, averaged over
    directions and sites, for r = 0..L//2.

    mode='spin': states {0,1} -> spins {-1,+1}, C(r) = <s_0 s_r> - <s>^2.
    mode='potts': C(r) = <delta(x_0, x_r)> - 1/q  (q inferred from values).
    """
    z = x.view(x.shape[0], *lattice_shape)
    L = min(lattice_shape)
    rs = range(L // 2 + 1)
    out = []
    if mode == "spin":
        s = 2.0 * z.float() - 1.0
        mean2 = s.mean() ** 2
        for r in rs:
            c = 0.0
            for axis in range(1, z.dim()):
                c = c + (s * torch.roll(s, r, dims=axis)).mean()
            out.append(c / (z.dim() - 1) - mean2)
    elif mode == "potts":
        q = int(x.max().item()) + 1
        for r in rs:
            c = 0.0
            for axis in range(1, z.dim()):
                c = c + (z == torch.roll(z, r, dims=axis)).float().mean()
            out.append(c / (z.dim() - 1) - 1.0 / q)
    else:
        raise ValueError(mode)
    return torch.stack([torch.as_tensor(v) for v in out])


def local_inconsistency(target, x: Tensor, chunk: int = 1024) -> Tensor:
    """Per-sample `-mean_d log nu(x_d | x^{-d})`, shape (B,).

    How surprising each site of a configuration is to the target's own
    single-site conditional.  It needs only the local energy differences, so
    it is exact and cheap, and unlike the energy it is sensitive to *which*
    configurations the sampler produces rather than only to their energy.
    Compare it against the same quantity on reference samples: the target's
    value is not zero, it is whatever the target's own typical set gives.
    """
    out = []
    for i in range(0, x.shape[0], chunk):
        xb = x[i : i + chunk]
        hb = target.heat_bath(xb)  # (b, D, N)
        p_cur = torch.gather(hb, 2, xb.unsqueeze(-1)).squeeze(-1)  # (b, D)
        out.append(-p_cur.clamp_min(1e-12).log().mean(dim=-1))
    return torch.cat(out)


# ----------------------------------------------------------------------
# Thermodynamic observables (finite-size definitions, all per sample batch)
# ----------------------------------------------------------------------
def _wmean(v: Tensor, w: Tensor | None = None) -> float:
    v = v.detach().double()
    if w is None:
        return v.mean().item()
    return (w.to(v.device) * v).sum().item()


def susceptibility(mag: Tensor, beta: float, dim: int, w: Tensor | None = None) -> float:
    """chi = beta D (<m^2> - <|m|>^2), the finite-size susceptibility built
    on |m| so that it stays finite in the ordered phase."""
    m2 = _wmean(mag**2, w)
    ma = _wmean(mag.abs(), w)
    return beta * dim * (m2 - ma**2)


def binder_cumulant(mag: Tensor, w: Tensor | None = None) -> float:
    """U4 = 1 - <m^4> / (3 <m^2>^2): 0 for a Gaussian (disordered), 2/3 for
    two sharp peaks (ordered), size-independent at T_c."""
    m2 = _wmean(mag**2, w)
    m4 = _wmean(mag**4, w)
    return 1.0 - m4 / max(3.0 * m2**2, 1e-300)


def specific_heat(e_site: Tensor, beta: float, dim: int, w: Tensor | None = None) -> float:
    """c = beta^2 D (<e^2> - <e>^2) per site, e the energy per site."""
    e1 = _wmean(e_site, w)
    e2 = _wmean(e_site**2, w)
    return beta**2 * dim * (e2 - e1**2)


def structure_factor(x: Tensor, lattice_shape: tuple[int, ...], w: Tensor | None = None,
                     q: int | None = None):
    """(S(0), S(k_min)) of the order-parameter field, k_min = 2 pi / L along
    each axis, averaged over axes; S(k) = <|sum_i s_i e^{i k r_i}|^2> / D.

    `q > 2` embeds each state in the regular simplex in R^(q-1) (the Potts
    analogue of the spin) and sums the components, because the Ising map
    s = 2 x - 1 is not a valid field for a q-state alphabet: it would order
    the colours on a line and read {0, 1, 2} as {-1, +1, +3}.  At q = 2 the
    simplex vertices are -1 and +1, so this reduces to the spin field exactly
    and the Ising numbers are unchanged.
    """
    dt = torch.get_default_dtype()
    z = x.view(x.shape[0], *lattice_shape)
    n_axes = len(lattice_shape)
    if q is not None and q > 2:
        from dds.targets.potts.diagnostics import simplex_vertices

        s = simplex_vertices(q, device=x.device, dtype=dt)[z]  # (B, *shape, q-1)
    else:
        s = (2.0 * z.to(dt) - 1.0).unsqueeze(-1)               # (B, *shape, 1)
    B, comp = s.shape[0], s.shape[-1]
    D = 1
    for L in lattice_shape:
        D *= L
    s0 = (s.reshape(B, D, comp).sum(1) ** 2).sum(-1) / D       # (B,)
    sk = torch.zeros_like(s0)
    for axis in range(n_axes):
        L = lattice_shape[axis]
        phase = 2.0 * math.pi * torch.arange(L, device=x.device, dtype=dt) / L
        shape = [1] * (n_axes + 2)
        shape[axis + 1] = L
        c = (s * torch.cos(phase).view(shape)).reshape(B, D, comp).sum(1)
        si = (s * torch.sin(phase).view(shape)).reshape(B, D, comp).sum(1)
        sk = sk + ((c**2 + si**2).sum(-1)) / D
    sk = sk / n_axes
    return _wmean(s0, w), _wmean(sk, w)


def correlation_length(x: Tensor, lattice_shape: tuple[int, ...], w: Tensor | None = None,
                       q: int | None = None) -> float:
    """Second-moment correlation length
        xi = sqrt(S(0) / S(k_min) - 1) / (2 sin(pi / L)),
    the standard finite-lattice estimator; it saturates near L in the
    ordered phase, which is what the reference does too.  `q` selects the
    field the structure factor is built on (see `structure_factor`)."""
    S0, Sk = structure_factor(x, lattice_shape, w, q=q)
    L = min(lattice_shape)
    ratio = max(S0 / max(Sk, 1e-300) - 1.0, 0.0)
    return math.sqrt(ratio) / (2.0 * math.sin(math.pi / L))


def two_point_correlation_weighted(
    x: Tensor, lattice_shape: tuple[int, ...], w: Tensor | None = None,
    q: int | None = None,
) -> Tensor:
    """Connected two-point function C(r), r = 0..L//2, averaged over axes;
    importance-weighted per sample when `w` is given.

    `q > 2` uses the Potts form C(r) = <delta(x_0, x_r)> - 1/q, which is the
    right connected correlator for an unordered alphabet; the spin form
    s = 2 x - 1 below is its q = 2 special case.
    """
    dt = torch.get_default_dtype()
    if q is not None and q > 2:
        zc = x.view(x.shape[0], *lattice_shape)
        B = zc.shape[0]
        ww = (torch.full((B,), 1.0 / B, device=x.device, dtype=dt) if w is None
              else w.to(x.device).to(dt))
        out = []
        for r in range(min(lattice_shape) // 2 + 1):
            c = 0.0
            for axis in range(1, zc.dim()):
                per = (zc == torch.roll(zc, r, dims=axis)).to(dt).flatten(1).mean(1)
                c = c + (ww * per).sum()
            out.append(c / (zc.dim() - 1) - 1.0 / q)
        return torch.stack(out)
    z = x.view(x.shape[0], *lattice_shape).to(dt)
    s = 2.0 * z - 1.0
    B = s.shape[0]
    ww = (torch.full((B,), 1.0 / B, device=x.device) if w is None
          else w.to(x.device).to(s.dtype))
    mean = (ww * s.flatten(1).mean(1)).sum()
    out = []
    for r in range(min(lattice_shape) // 2 + 1):
        c = 0.0
        for axis in range(1, s.dim()):
            per = (s * torch.roll(s, r, dims=axis)).flatten(1).mean(1)  # (B,)
            c = c + (ww * per).sum()
        out.append(c / (s.dim() - 1) - mean**2)
    return torch.stack(out)


# ----------------------------------------------------------------------
# Generic observables
# ----------------------------------------------------------------------
# Lattice-spin targets are summarized by magnetization and its moments.  A
# target whose natural observables are different -- a fluid has no
# magnetization and no two-point spin correlation -- declares them instead by
# implementing `observables(x) -> {name: per-sample scalar}`, and the three
# public entry points below route to the generic path.  Targets without the
# hook keep the original code path byte for byte.
def _has_generic(target) -> bool:
    return hasattr(target, "observables")


def _generic_reference_summary(target, x_ref: Tensor) -> dict:
    obs = target.observables(x_ref)
    ref = {"_obs": {k: v.detach().double().cpu() for k, v in obs.items()}}
    for k, v in ref["_obs"].items():
        ref[f"{k}_mean"] = v.mean().item()
        ref[f"{k}_std"] = v.std().item()
        ref[f"{k}_range"] = (v.min().item(), v.max().item())
    ref["local_inconsistency"] = local_inconsistency(target, x_ref).mean().item()
    return ref


def _generic_summarize(target, x: Tensor, prefix: str, log_w: Tensor | None) -> dict:
    obs = target.observables(x)
    li = local_inconsistency(target, x)
    out = {f"{prefix}local_inconsistency": li.mean().item()}
    for k, v in obs.items():
        v = v.detach().double()
        out[f"{prefix}{k}"] = v.mean().item()
        out[f"{prefix}{k}_std"] = v.std().item()
    if log_w is not None:
        w = snis_weights(log_w).double()
        for k, v in obs.items():
            out[f"{prefix}{k}_rw"] = (w * v.detach().double()).sum().item()
        out[f"{prefix}local_inconsistency_rw"] = (w * li.double()).sum().item()
    return out


def _generic_compare(
    target, x: Tensor, x_ref: Tensor, prefix: str,
    log_w: Tensor | None, ref: dict | None,
) -> dict:
    if ref is None or "_obs" not in ref:
        ref = _generic_reference_summary(target, x_ref)
    obs = target.observables(x)
    w = _np(snis_weights(log_w).double()) if log_w is not None else None
    out = {}
    for k, v in obs.items():
        v = v.detach().double().cpu()
        r = ref["_obs"][k]
        lo, hi = ref[f"{k}_range"]
        rng = (min(lo, v.min().item()) - 1e-9, max(hi, v.max().item()) + 1e-9)
        out[f"{prefix}{k}_w1"] = wasserstein1d(v, r, p=1)
        out[f"{prefix}{k}_w2"] = wasserstein1d(v, r, p=2)
        out[f"{prefix}{k}_tv"] = histogram_tv(v, r, bins=41, range_=rng)
        out[f"{prefix}{k}_err"] = abs(v.mean().item() - ref[f"{k}_mean"])
        out[f"{prefix}{k}_ref"] = ref[f"{k}_mean"]
        if w is not None:
            out[f"{prefix}{k}_w1_rw"] = wasserstein1d(v, r, p=1, wa=w)
            out[f"{prefix}{k}_w2_rw"] = wasserstein1d(v, r, p=2, wa=w)
            out[f"{prefix}{k}_tv_rw"] = histogram_tv(v, r, bins=41, range_=rng, wa=w)
            out[f"{prefix}{k}_err_rw"] = abs(
                float((w * _np(v)).sum()) - ref[f"{k}_mean"]
            )
    li = local_inconsistency(target, x)
    out[f"{prefix}local_inconsistency_err"] = abs(
        li.mean().item() - ref["local_inconsistency"]
    )
    out[f"{prefix}local_inconsistency_ref"] = ref["local_inconsistency"]
    return out


def reference_summary(target, x_ref: Tensor) -> dict:
    """Observables of the reference sample, computed once per target.

    They are the denominators of every comparison, so they are reported in
    the history alongside the sampler's own values.
    """
    if _has_generic(target):
        return _generic_reference_summary(target, x_ref)
    shape = tuple(target.lattice_shape)
    q = int(target.N)
    mag = target.magnetization(x_ref)
    e = target.energy_per_site(x_ref)
    out = {
        "mag": mag,
        "energy": e,
        "mag_abs": mag.abs().mean().item(),
        "energy_per_site": e.mean().item(),
        "chi": susceptibility(mag, target.beta, target.D),
        "binder": binder_cumulant(mag),
        "specific_heat": specific_heat(e, target.beta, target.D),
        "xi": correlation_length(x_ref, shape, q=q),
        "corr": two_point_correlation_weighted(x_ref, shape, q=q),
        "local_inconsistency": local_inconsistency(target, x_ref).mean().item(),
        "energy_range": (e.min().item(), e.max().item()),
    }
    if q > 2:
        from dds.targets.potts.diagnostics import mode_fractions

        out["mode_min"] = float(mode_fractions(x_ref, q).min())
    return out


def summarize_observables(
    target, x: Tensor, prefix: str = "", log_w: Tensor | None = None
) -> dict:
    """Standard scalar observables of a sample batch.

    With `log_w`, additionally report self-normalized importance-weighted
    ('_rw') estimates.  The plain values describe the raw sampler; the
    reweighted ones describe the estimator actually used for log Z, so a
    large gap between them means good weights on a poor proposal.
    """
    if _has_generic(target):
        return _generic_summarize(target, x, prefix, log_w)
    shape = tuple(target.lattice_shape)
    beta, D = target.beta, target.D
    q = int(target.N)
    mag = target.magnetization(x)
    e = target.energy_per_site(x)
    li = local_inconsistency(target, x)
    out = {
        f"{prefix}mag_mean": mag.mean().item(),
        f"{prefix}mag_abs_mean": mag.abs().mean().item(),
        f"{prefix}mag_std": mag.std().item(),
        # fraction of samples in the m > 0 mode: 1/2 at zero field, and the
        # complement of mag_mean as a mode-collapse diagnostic
        f"{prefix}mag_pos_frac": (mag > 0).double().mean().item(),
        f"{prefix}energy_per_site": e.mean().item(),
        f"{prefix}energy_per_site_std": e.std().item(),
        f"{prefix}chi": susceptibility(mag, beta, D),
        f"{prefix}binder": binder_cumulant(mag),
        f"{prefix}specific_heat": specific_heat(e, beta, D),
        f"{prefix}xi": correlation_length(x, shape, q=q),
        f"{prefix}local_inconsistency": li.mean().item(),
    }
    if q > 2:
        # Every observable above is S_q invariant: below beta_c the q ordered
        # modes are symmetry-related, so a sampler holding ONE of them reports
        # the same energy, the same |m| and the same chi as a correct one.
        # `mode_min` is the share of the batch held by the least-visited
        # majority state: 1/q when balanced, 0 when collapsed.  Diagnostic
        # only -- nothing here may enter a loss, or the symmetry would be
        # injected rather than discovered.
        from dds.targets.potts.diagnostics import (magnetisation_vector,
                                                   majority_state,
                                                   mode_fractions)

        mf = mode_fractions(x, q)
        out[f"{prefix}mode_min"] = float(mf.min())
        out[f"{prefix}mode_gap"] = float(mf.max() - mf.min())
        for a in range(q):
            out[f"{prefix}mode_frac_{a}"] = float(mf[a])
        out[f"{prefix}M_norm"] = float(
            magnetisation_vector(x, q).norm(dim=1).mean())
    if log_w is not None:
        w = snis_weights(log_w)
        md, ed = mag.double(), e.double()
        out[f"{prefix}mag_mean_rw"] = (w * md).sum().item()
        out[f"{prefix}mag_abs_mean_rw"] = (w * md.abs()).sum().item()
        out[f"{prefix}mag_pos_frac_rw"] = (w * (md > 0).double()).sum().item()
        out[f"{prefix}energy_per_site_rw"] = (w * ed).sum().item()
        out[f"{prefix}chi_rw"] = susceptibility(mag, beta, D, w)
        out[f"{prefix}binder_rw"] = binder_cumulant(mag, w)
        out[f"{prefix}specific_heat_rw"] = specific_heat(e, beta, D, w)
        out[f"{prefix}xi_rw"] = correlation_length(x, shape, w, q=q)
        out[f"{prefix}local_inconsistency_rw"] = (w * li.double()).sum().item()
        if q > 2:
            maj = majority_state(x, q)
            wm = torch.zeros(q, dtype=w.dtype, device=w.device).index_add_(
                0, maj.to(w.device), w)
            out[f"{prefix}mode_min_rw"] = float(wm.min())
    return out


def compare_to_reference(
    target, x: Tensor, x_ref: Tensor, prefix: str = "",
    log_w: Tensor | None = None, ref: dict | None = None,
) -> dict:
    """Discrepancy metrics between model samples and reference samples.

    `ref` is `reference_summary(target, x_ref)`, computed once by the caller
    when the same reference is scored many times.
    """
    if _has_generic(target):
        return _generic_compare(target, x, x_ref, prefix, log_w, ref)
    shape = tuple(target.lattice_shape)
    beta, D = target.beta, target.D
    q = int(target.N)
    if ref is None:
        ref = reference_summary(target, x_ref)
    mag, mag_ref = target.magnetization(x), ref["mag"]
    e, e_ref = target.energy_per_site(x), ref["energy"]
    lo, hi = ref["energy_range"]
    e_range = (min(lo, e.min().item()) - 1e-6, max(hi, e.max().item()) + 1e-6)
    corr = two_point_correlation_weighted(x, shape, q=q)
    out = {
        f"{prefix}mag_w1": wasserstein1d(mag, mag_ref, p=1),
        f"{prefix}energy_w2": wasserstein1d(e, e_ref, p=2),
        f"{prefix}mag_tv": histogram_tv(mag, mag_ref, bins=41, range_=(-1.02, 1.02)),
        f"{prefix}energy_tv": histogram_tv(e, e_ref, bins=41, range_=e_range),
        f"{prefix}mag_abs_err": abs(mag.abs().mean().item() - ref["mag_abs"]),
        f"{prefix}energy_err": abs(e.mean().item() - ref["energy_per_site"]),
        f"{prefix}chi_err": abs(susceptibility(mag, beta, D) - ref["chi"]),
        f"{prefix}binder_err": abs(binder_cumulant(mag) - ref["binder"]),
        f"{prefix}specific_heat_err": abs(specific_heat(e, beta, D) - ref["specific_heat"]),
        f"{prefix}xi_err": abs(correlation_length(x, shape, q=q) - ref["xi"]),
        f"{prefix}corr_err": (corr - ref["corr"].to(corr.device)).abs().mean().item(),
        f"{prefix}local_inconsistency_err": abs(
            local_inconsistency(target, x).mean().item() - ref["local_inconsistency"]
        ),
        # the reference's own values, so the history is self-contained
        f"{prefix}mag_abs_ref": ref["mag_abs"],
        f"{prefix}energy_per_site_ref": ref["energy_per_site"],
        f"{prefix}chi_ref": ref["chi"],
        f"{prefix}binder_ref": ref["binder"],
        f"{prefix}specific_heat_ref": ref["specific_heat"],
        f"{prefix}xi_ref": ref["xi"],
        f"{prefix}local_inconsistency_ref": ref["local_inconsistency"],
    }
    if q > 2 and "mode_min" in ref:
        # the reference chain's own mode balance, so the history says what
        # "balanced" is at this lattice and temperature rather than assuming
        # 1/q: a finite reference sample scatters around it too
        from dds.targets.potts.diagnostics import mode_fractions

        out[f"{prefix}mode_min_ref"] = ref["mode_min"]
        out[f"{prefix}mode_min_err"] = abs(
            float(mode_fractions(x, q).min()) - ref["mode_min"])
    if log_w is not None:
        wt = snis_weights(log_w)
        w = _np(wt)
        out[f"{prefix}mag_w1_rw"] = wasserstein1d(mag, mag_ref, p=1, wa=w)
        out[f"{prefix}energy_w2_rw"] = wasserstein1d(e, e_ref, p=2, wa=w)
        out[f"{prefix}mag_tv_rw"] = histogram_tv(
            mag, mag_ref, bins=41, range_=(-1.02, 1.02), wa=w
        )
        out[f"{prefix}energy_tv_rw"] = histogram_tv(e, e_ref, bins=41, range_=e_range, wa=w)
        mabs_rw = float((w * _np(mag.double().abs())).sum())
        e_rw = float((w * _np(e.double())).sum())
        out[f"{prefix}mag_abs_err_rw"] = abs(mabs_rw - ref["mag_abs"])
        out[f"{prefix}energy_err_rw"] = abs(e_rw - ref["energy_per_site"])
        out[f"{prefix}chi_err_rw"] = abs(susceptibility(mag, beta, D, wt) - ref["chi"])
        out[f"{prefix}binder_err_rw"] = abs(binder_cumulant(mag, wt) - ref["binder"])
        out[f"{prefix}specific_heat_err_rw"] = abs(
            specific_heat(e, beta, D, wt) - ref["specific_heat"])
        out[f"{prefix}xi_err_rw"] = abs(correlation_length(x, shape, wt) - ref["xi"])
        corr_rw = two_point_correlation_weighted(x, shape, wt)
        out[f"{prefix}corr_err_rw"] = (corr_rw - ref["corr"].to(corr.device)).abs().mean().item()
        li = local_inconsistency(target, x).double()
        out[f"{prefix}local_inconsistency_err_rw"] = abs(
            (wt.to(li.device) * li).sum().item() - ref["local_inconsistency"])
    return out
