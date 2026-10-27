"""Ramachandran and energy metrics against the MD reference, and their figures."""
from __future__ import annotations

import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RAMA_BINS = 36          # 10 degree cells for the KL / JS


def _norm_w(w, n):
    if w is None:
        return np.full(n, 1.0 / n)
    w = np.asarray(w, dtype=np.float64)
    return w / w.sum()


def rama_hist(phi, psi, w=None, bins=RAMA_BINS):
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


def basin_fractions(rama, w=None):
    """(R, 3) weighted fractions of alphaR (phi < 0, -120 < psi < 50 deg), beta/PPII (phi < 0, other psi)
    and alphaL (phi >= 0) per residue."""
    phi, psi = np.degrees(rama[..., 0]), np.degrees(rama[..., 1])
    wn = _norm_w(w, len(phi))[:, None]
    ar = (phi < 0) & (psi > -120) & (psi < 50)
    be = (phi < 0) & ~ar
    al = phi >= 0
    return np.stack([(wn * ar).sum(0), (wn * be).sum(0), (wn * al).sum(0)], -1)


def peptide_metrics(ref, obs, w, names):
    """Per-residue Ramachandran KL(ref || model) / JS (and their mean), basin fractions and the energy W1,
    for the model's samples as drawn and reweighted (_rw)."""
    out = {}
    R = obs["rama"].shape[1]
    Ec = np.minimum(obs["energy"], np.max(ref["energy"]) + 200.0)     # clashed draws would dominate W1
    hr = [rama_hist(ref["rama"][:, r, 0], ref["rama"][:, r, 1]) for r in range(R)]
    fr = basin_fractions(ref["rama"])
    for tag, ww in (("", None), ("_rw", w)):
        kls, jss = [], []
        fm = basin_fractions(obs["rama"], ww)
        for r in range(R):
            hm = rama_hist(obs["rama"][:, r, 0], obs["rama"][:, r, 1], ww)
            kls.append(kl(hr[r], hm)); jss.append(js(hr[r], hm))
            out[f"rama_kl{tag}/{names[r]}"] = kls[-1]
            for b, bn in enumerate(("alphaR", "beta", "alphaL")):
                out[f"basin{tag}/{names[r]}_{bn}"] = float(fm[r, b])
        out[f"rama_kl{tag}"] = float(np.mean(kls))
        out[f"rama_js{tag}"] = float(np.mean(jss))
        out[f"energy_w1{tag}"] = w1(Ec, ref["energy"], ww)
        out[f"energy_mean{tag}"] = float((_norm_w(ww, len(Ec)) * obs["energy"]).sum())
    for r in range(R):
        for b, bn in enumerate(("alphaR", "beta", "alphaL")):
            out[f"basin_ref/{names[r]}_{bn}"] = float(fr[r, b])
    out["ref_energy_mean"] = float(np.mean(ref["energy"]))
    return out


# --- figures ------------------------------------------------------------------------------
def _fe(h, cap=8.0):
    with np.errstate(divide="ignore"):
        f = -np.log(h)
    f -= np.nanmin(f[np.isfinite(f)]) if np.isfinite(f).any() else 0.0
    f[~np.isfinite(f)] = np.nan
    return np.minimum(f, cap)


def rama_grid_figure(rama, w, names, ref=None, title=""):
    """rama (N, R, 2) model; ref['rama'] (F, R, 2).  Rows: residues; columns: MD / model / reweighted."""
    cols = (["MD reference"] if ref is not None else []) + ["model", "model, reweighted"]
    R = len(names)
    fig, axes = plt.subplots(R, len(cols), figsize=(3.3 * len(cols), 3.1 * R), squeeze=False)
    for a in range(R):
        series = ([(ref["rama"][:, a, 0], ref["rama"][:, a, 1], None)] if ref is not None else []) + \
                 [(rama[:, a, 0], rama[:, a, 1], None), (rama[:, a, 0], rama[:, a, 1], w)]
        for b, (phi, psi, ww) in enumerate(series):
            ax = axes[a, b]
            im = ax.imshow(_fe(rama_hist(phi, psi, ww, bins=72)).T, origin="lower",
                           extent=[-180, 180, -180, 180], cmap="viridis", vmin=0, vmax=8, aspect="equal")
            ax.set_xticks([-180, 0, 180]); ax.set_yticks([-180, 0, 180])
            if a == 0:
                ax.set_title(cols[b], fontsize=9)
            if b == 0:
                ax.set_ylabel(f"{names[a]}\n" + r"$\psi$", fontsize=9)
            if a == R - 1:
                ax.set_xlabel(r"$\phi$", fontsize=9)
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6, label=r"$-\log p$ ($k_BT$)")
    if title:
        fig.suptitle(title, fontsize=10)
    return fig


def hist_figure(v, w, ref_v=None, xlabel="", title=""):
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    src = ref_v if ref_v is not None else v
    lo, hi = np.percentile(src, 0.05), np.percentile(src, 99.95)
    pad = 0.15 * (hi - lo)
    bins = np.linspace(lo - pad, hi + pad, 120)
    if ref_v is not None:
        ax.hist(ref_v, bins=bins, density=True, histtype="stepfilled", alpha=0.35, color="0.3", label="MD reference")
    vc = np.clip(v, bins[0], bins[-1])
    ax.hist(vc, bins=bins, density=True, histtype="step", lw=1.6, color="C0", label="model")
    if w is not None:
        ax.hist(vc, bins=bins, density=True, weights=_norm_w(w, len(v)), histtype="step", lw=1.6, color="C3",
                label="model, reweighted")
    ax.set_xlabel(xlabel); ax.set_ylabel("density"); ax.legend(fontsize=8)
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    return fig
