"""Ramachandran plots, energy histograms and distribution metrics of bridge endpoints against MD."""
from __future__ import annotations

import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _norm_w(w, n):
    if w is None:
        return np.full(n, 1.0 / n)
    w = np.asarray(w, dtype=np.float64)
    return w / w.sum()


def rama_hist(phi, psi, w=None, bins=36):
    h, _, _ = np.histogram2d(phi, psi, bins=bins, range=[[-math.pi, math.pi]] * 2,
                             weights=_norm_w(w, len(phi)))
    return h / max(h.sum(), 1e-300)


def kl(p, q, eps=1e-6):
    """KL(p || q) of two histograms, q floored at eps (as is common for Ramachandran KL)."""
    p = p.ravel(); q = q.ravel()
    q = (q + eps) / (q + eps).sum()
    m = p > 0
    return float((p[m] * np.log(p[m] / q[m])).sum())


def js(p, q):
    p = p.ravel(); q = q.ravel(); m = 0.5 * (p + q)
    def _kl(a, b):
        k = a > 0
        return (a[k] * np.log(a[k] / b[k])).sum()
    return float(0.5 * _kl(p, m) + 0.5 * _kl(q, m))


def w1(a, b, wa=None, wb=None):
    """Wasserstein-1 between two weighted empirical 1D distributions."""
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    wa = _norm_w(wa, len(a)); wb = _norm_w(wb, len(b))
    x = np.concatenate([a, b]); order = np.argsort(x)
    w = np.concatenate([wa, -wb])[order]
    cdf = np.cumsum(w)[:-1]
    return float(np.sum(np.abs(cdf) * np.diff(x[order])))


def metrics(ref, phi, psi, E, w):
    """Model (unweighted and reweighted) against the reference MD."""
    out = {}
    hr = rama_hist(ref["phi"], ref["psi"])
    # a few clashed draws at 1e10 kJ/mol would make W1 meaningless: cap them
    # 200 kJ/mol above the highest reference energy
    Ec = np.minimum(E, np.max(ref["energy"]) + 200.0)
    for tag, ww in (("", None), ("_rw", w)):
        hm = rama_hist(phi, psi, ww)
        out[f"rama_kl{tag}"] = kl(hr, hm)
        out[f"rama_js{tag}"] = js(hr, hm)
        out[f"energy_w1{tag}"] = w1(Ec, ref["energy"], ww)
        ww_ = _norm_w(ww, len(phi))
        out[f"phi_pos_frac{tag}"] = float(ww_[phi > 0].sum())
        out[f"energy_mean{tag}"] = float((ww_ * E).sum())
    out["ref_phi_pos_frac"] = float((ref["phi"] > 0).mean())
    out["ref_energy_mean"] = float(np.mean(ref["energy"]))
    return out


def _fe(h, cap=8.0):
    with np.errstate(divide="ignore"):
        f = -np.log(h)
    f -= np.nanmin(f[np.isfinite(f)]) if np.isfinite(f).any() else 0.0
    f[~np.isfinite(f)] = np.nan
    return np.minimum(f, cap)


def rama_figure(phi, psi, w=None, ref=None, title=""):
    panels = []
    if ref is not None:
        panels.append(("MD reference", ref["phi"], ref["psi"], None))
    panels.append(("model", phi, psi, None))
    if w is not None:
        panels.append(("model, reweighted", phi, psi, w))
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.0), squeeze=False)
    for ax, (name, a, b, ww) in zip(axes[0], panels):
        h = rama_hist(a, b, ww, bins=72)
        im = ax.imshow(_fe(h).T, origin="lower", extent=[-180, 180, -180, 180], cmap="viridis",
                       vmin=0, vmax=8, aspect="equal")
        ax.set_xlabel(r"$\phi$ (deg)"); ax.set_ylabel(r"$\psi$ (deg)")
        ax.set_xticks([-180, -90, 0, 90, 180]); ax.set_yticks([-180, -90, 0, 90, 180])
        ess = ""
        if ww is not None:
            wn = _norm_w(ww, len(a)); ess = f"  ESS={1.0 / (wn ** 2).sum() / len(a):.3f}"
        ax.set_title(f"{name}{ess}", fontsize=10)
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, label=r"$-\log p$ ($k_BT$)")
    if title:
        fig.suptitle(title, fontsize=10)
    return fig


def energy_figure(E, w=None, ref=None, title=""):
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    lo_hi = []
    if ref is not None:
        lo_hi = [np.percentile(ref["energy"], 0.01) - 10, np.percentile(ref["energy"], 99.99) + 40]
    else:
        lo_hi = [np.percentile(E, 0.5) - 10, np.percentile(E, 99.0) + 20]
    bins = np.linspace(lo_hi[0], lo_hi[1], 120)
    if ref is not None:
        ax.hist(ref["energy"], bins=bins, density=True, histtype="stepfilled", alpha=0.35,
                color="0.3", label="MD reference")
    ax.hist(np.clip(E, bins[0], bins[-1]), bins=bins, density=True, histtype="step", lw=1.6,
            color="C0", label="model")
    if w is not None:
        ax.hist(np.clip(E, bins[0], bins[-1]), bins=bins, density=True, weights=_norm_w(w, len(E)),
                histtype="step", lw=1.6, color="C3", label="model, reweighted")
    ax.set_xlabel("potential energy (kJ/mol)"); ax.set_ylabel("density")
    ax.legend(fontsize=8)
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig


def phi_profile_figure(phi, psi, w=None, ref=None):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    edges = np.linspace(-math.pi, math.pi, 73)
    mid = np.degrees(0.5 * (edges[1:] + edges[:-1]))
    for ax, k, name in ((axes[0], 0, r"$\phi$"), (axes[1], 1, r"$\psi$")):
        series = []
        if ref is not None:
            series.append(("MD reference", ref["phi" if k == 0 else "psi"], None, "0.3"))
        series.append(("model", phi if k == 0 else psi, None, "C0"))
        if w is not None:
            series.append(("model, reweighted", phi if k == 0 else psi, w, "C3"))
        for lab, a, ww, c in series:
            h, _ = np.histogram(a, bins=edges, weights=_norm_w(ww, len(a)))
            with np.errstate(divide="ignore"):
                f = -np.log(h / h.sum())
            f -= np.nanmin(f[np.isfinite(f)])
            ax.plot(mid, f, color=c, label=lab)
        ax.set_xlabel(f"{name} (deg)"); ax.set_ylabel(r"free energy ($k_BT$)"); ax.set_ylim(0, 12)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    return fig
