"""The common-tangent phase diagram, computed from a model's own draws.

    python scripts/model_common_tangent.py runs/alloy_n5000/agpd125/digin/s0

The semi-grand construction the report draws -- peak of dc/d(delta_mu), then
the composition a fixed width either side -- is not an estimator of the
coexisting compositions at N ~ 125.  The jump is smeared over a finite
delta_mu window, so a fixed-width readout measures the smearing.  Narrowing
the window does not fix it: against the packaged Ag-Pd tangent the mean error
is 0.13 at +-0.005 eV, 0.05 at +-0.01, 0.08 at +-0.02 and 0.12 at +-0.04.
There is no width that converges.

The construction that does converge is the one the reference uses: the double
tangent on the fixed-composition free energy.  A semi-grand sampler can reach
it, because each draw carries a definite composition n.  At every state point
the self-normalised weights estimate

    p(n | T, delta_mu) = Z_n(T) exp(beta delta_mu (n - N/2)) / Z(T, delta_mu),

so, taking logs and rearranging,

    ln Z_n(T) = ln p(n | T, delta_mu) + ln Z(T, delta_mu)
                - beta delta_mu (n - N/2).

Every delta_mu on the sweep gives an absolute estimate of ln Z_n over the
compositions it populates -- absolute, with no free constant, because the
sampler estimates ln Z(T, delta_mu) itself.  That is the part plain umbrella
sampling cannot do: WHAM has to solve for one offset per window, and here
they are measured.  Stitching the windows is then an average, and the spread
between windows that overlap is a free diagnostic, reported as `stitch_sd`.

The tangency points are invariant under adding any linear function of c, so
the mixing reference cancels and -(kT/N) ln Z_n can be used directly.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

KB_EV_PER_K = 8.617e-05


def snis(log_w: np.ndarray) -> np.ndarray:
    a = log_w - log_w.max()
    w = np.exp(a)
    return w / w.sum()


#: eV/site.  Below this the double well is shallower than the numerical
#: noise in ln Z_n and the hull picks up edges that are not gaps.  Calibrated
#: on the packaged Ag-Pd data, where every reported tie line has depth
#: >= 3.36e-5 and every rejected one <= 1.53e-5, so the cut is unambiguous.
MIN_DEPTH = 2e-5


def common_tangent(c: np.ndarray, f: np.ndarray, min_span: int = 4,
                   min_depth: float = 0.0, sigma: np.ndarray | None = None,
                   n_sigma: float = 3.0,
                   visited: np.ndarray | None = None
                   ) -> list[tuple[float, float, float]]:
    """Lower convex hull of f(c); a hull edge that skips compositions is a
    two-phase region and its ends are the coexisting compositions.

    The acceptance rules are FrOGS's (paper appendix A.4): an edge is a tie
    line only if (i) it skips at least four compositions, (ii) the measured
    f lies at least `n_sigma` standard errors above the chord somewhere in
    between, or no sample ever visited those compositions, and (iii) --
    applied by the caller, since it needs the neighbouring isotherm -- the
    same gap appears at the temperature below.

    EVERY ONE OF THOSE IS A STATEMENT ABOUT THE COMPOSITION AXIS, and the
    arrays here are indexed by composition n = 0..N with NaN wherever the
    sampler resolved nothing.  Judging them on the compacted list of resolved
    points instead is wrong whenever the curve is sparse, and it fails
    exactly where the physics is most interesting: on Cu-Au at 200 K the
    sampler resolves 11 of 129 compositions, the hull is the four correct
    edges 0 -> 0.25 -> 0.50 -> 0.75 -> 1.00, and three of the four have a
    list-index span of 1 or 2 while spanning 32 compositions each.  Counted
    on the list they are rejected as single steps; counted on the
    composition axis they are the ordered phases of Cu-Au.

    `sigma` is the per-composition standard error of f, and the three-sigma
    test is pointwise against each composition's own error.  `visited` marks
    compositions some draw actually landed on at any condition, so an
    interior nothing ever visited counts as evidence of a gap in its own
    right -- which is the low-temperature ordered-phase case, where the
    stable compositions are sharp and the space between them is empty.

    `min_depth` is an absolute floor in eV/site, off by default.  It is not
    part of the FrOGS construction; it exists for exact input, where sigma
    is identically zero and the three-sigma test cannot bind.
    """
    f = np.asarray(f, dtype=float)
    c = np.asarray(c, dtype=float)
    resolved = np.isfinite(f)
    if int(resolved.sum()) < 3:
        return []
    sig = (np.asarray(sigma, dtype=float) if sigma is not None
           else np.zeros_like(f))
    vis = np.asarray(visited, dtype=bool) if visited is not None else None

    # hull over the resolved points, but carrying their composition index
    hull: list[int] = []
    for j in np.flatnonzero(resolved):
        while len(hull) >= 2:
            j1, j2 = hull[-2], hull[-1]
            if ((f[j2] - f[j1]) * (c[j] - c[j1])
                    >= (f[j] - f[j1]) * (c[j2] - c[j1])):
                hull.pop()
            else:
                break
        hull.append(int(j))

    out = []
    for j1, j2 in zip(hull, hull[1:]):
        # (i) at least `min_span` compositions strictly inside the edge
        if (j2 - j1 - 1) < min_span:
            continue
        inner = np.arange(j1 + 1, j2)
        if inner.size == 0:
            continue
        chord = f[j1] + (c[inner] - c[j1]) * (f[j2] - f[j1]) / (c[j2] - c[j1])
        seen = resolved[inner]
        rise = np.where(seen, f[inner] - chord, -np.inf)
        depth = float(np.max(rise)) if seen.any() else 0.0
        if seen.any() and depth < min_depth:
            continue
        # (ii) three standard errors above the chord at some interior
        # composition, judged against that composition's own error -- or an
        # interior no draw ever visited
        ok_sigma = bool(np.any(rise[seen] >= n_sigma * sig[inner][seen])) \
            if seen.any() else False
        ok_empty = bool(vis is not None and not np.any(vis[inner]))
        if not (ok_sigma or ok_empty):
            continue
        out.append((float(c[j1]), float(c[j2]), depth))
    return out


def load_draws(path: str):
    """Composition per draw, log weights and conditions, read once.

    Indexing a compressed .npz re-inflates the whole member every time, so
    the arrays are pulled into memory here rather than inside the loops --
    and only the composition is kept, since the tie lines need n, not the
    configuration.
    """
    z = np.load(path)
    n_sites = int(z["n_sites"])
    x = np.unpackbits(z["x_packed"], axis=-1)[:, :, :n_sites]
    return (x.sum(axis=2).astype(np.int32),      # (points, draws)
            z["log_w"].astype(np.float64),
            z["cond"].astype(np.float64),
            n_sites)


def model_ln_Z_n(n_all, log_w, cond, T_val: float, n_sites: int):
    """Stitch ln Z_n(T) from every delta_mu window at this T.

    Returns (ln Z_n, standard error, visited) -- `visited` being every
    composition some draw landed on, which is a weaker condition than being
    resolved well enough to carry a free energy and is what the FrOGS
    never-visited clause is about.
    """
    sel = np.flatnonzero(np.abs(cond[:, 0] - T_val) < 1e-6)
    if sel.size == 0:
        return None, None, None
    beta = 1.0 / (KB_EV_PER_K * T_val)
    acc = [[] for _ in range(n_sites + 1)]
    wacc = [[] for _ in range(n_sites + 1)]
    seen = np.zeros(n_sites + 1, dtype=bool)
    for k in sel:
        lw = log_w[k]
        n = n_all[k]
        w = snis(lw)
        # the sampler's own normalising constant at this window
        ln_Z = float(np.logaddexp.reduce(lw) - np.log(lw.size))
        dmu = float(cond[k, 1])
        seen[np.unique(n)] = True
        p = np.bincount(n, weights=w, minlength=n_sites + 1)
        # a bin carried by one or two draws is noise, not a histogram
        ess_bin = p ** 2 / np.maximum(
            np.bincount(n, weights=w ** 2, minlength=n_sites + 1), 1e-300)
        for m in np.flatnonzero(p > 0):
            if ess_bin[m] < 2.0:
                continue
            acc[m].append(np.log(p[m]) + ln_Z
                          - beta * dmu * (m - 0.5 * n_sites))
            wacc[m].append(ess_bin[m])
    ln_Zn = np.full(n_sites + 1, np.nan)
    se = np.full(n_sites + 1, np.nan)
    for m in range(n_sites + 1):
        if not acc[m]:
            continue
        v = np.asarray(acc[m])
        u = np.asarray(wacc[m])              # w_i = N_eff,i = s_i^-2
        ln_Zn[m] = float(np.sum(u * v) / np.sum(u))
        # FrOGS eq. 30-32: each condition estimates the same ln Z_c with
        # standard error N_eff,i^-1/2, so the merged error is (sum w_i)^-1/2,
        # widened by the Birge ratio when the conditions disagree by more
        # than their own errors predict
        s0 = float(np.sum(u)) ** -0.5
        if v.size > 1:
            chi2 = float(np.sum(u * (v - ln_Zn[m]) ** 2) / (v.size - 1))
            s0 *= max(1.0, chi2 ** 0.5)
        se[m] = s0
    return ln_Zn, se, seen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--min-span", type=int, default=4,
                    help="FrOGS accepts an edge only if it skips >= 4 "
                         "compositions")
    ap.add_argument("--no-persist", action="store_true",
                    help="drop FrOGS criterion (iii), the requirement that "
                         "the same gap appear at the temperature below")
    ap.add_argument("--min-depth", type=float, default=0.0,
                    help="absolute floor in eV/site; not part of "
                         "the FrOGS rules, for exact input where "
                         "the standard errors are zero")
    ap.add_argument("--fmix-every", type=float, default=100.0,
                    help="write the free-energy curve at multiples of this "
                         "many kelvin")
    ap.add_argument("--n-sigma", type=float, default=3.0,
                    help="a well must be this many times deeper than "
                         "the uncertainty in the curve it sits on")
    a = ap.parse_args()

    n_all, log_w, cond, N = load_draws(
        os.path.join(a.run_dir, "grid_samples.npz"))
    Ts = np.unique(cond[:, 0])
    comp = np.arange(N + 1) / N

    per_T = {}
    for T_val in Ts:
        ln_Zn, se, seen = model_ln_Z_n(n_all, log_w, cond, float(T_val), N)
        if ln_Zn is None:
            continue
        scale = KB_EV_PER_K * float(T_val) / N
        F = -scale * ln_Zn
        per_T[float(T_val)] = (
            common_tangent(comp, F, min_span=a.min_span,
                           min_depth=a.min_depth, sigma=scale * se,
                           n_sigma=a.n_sigma, visited=seen),
            float(np.nanmean(se)), int(np.isfinite(ln_Zn).sum()))

    # FrOGS criterion (iii): a two-phase region has to persist downward in
    # temperature, so a tie line is kept only if an overlapping one exists at
    # the isotherm below -- and that one must itself have been ACCEPTED, or
    # the rule is not a chain at all.  Checking the raw edges instead lets a
    # rejected isotherm carry its neighbour: on Ag-Pd the masked sampler had
    # nothing accepted at 550, 575 or 600 K and still reported gaps at 625 and
    # 650, because 600 K had raw edges that were themselves thrown out.  The
    # paper is explicit that a two-phase region ends at the highest
    # temperature at which its edge is still accepted.  The coldest isotherm
    # has nothing below it and is taken as given.
    order = sorted(per_T)
    rows = []
    kept_below: list[tuple[float, float, float]] | None = None
    for k, T_val in enumerate(order):
        ties, sd, n_res = per_T[T_val]
        below = kept_below if k > 0 else None
        kept_here = []
        for cl, cr, depth in ties:
            if below is not None and not a.no_persist:
                if not any(bl < cr and cl < br for bl, br, _d in below):
                    continue
            kept_here.append((cl, cr, depth))
            rows.append({
                "temperature": float(T_val),
                "c_left": cl,
                "c_right": cr,
                "depth": depth,
                "stitch_se": sd,
                "n_resolved": n_res,
            })
        kept_below = kept_here
    # the free-energy curve itself, at round temperatures, so the report can
    # draw what the tangent construction is built on rather than only its
    # output.  f is -(kT/N) ln Z_n, absolute: the mixing reference is left to
    # the consumer, because the model does not always resolve the pure
    # endpoints its own chord would need.
    want = [T for T in Ts if abs(T / a.fmix_every - round(T / a.fmix_every)) < 1e-9]
    curves = {"n_sites": N, "composition": comp.tolist(), "T": [], "f": [],
              "se": [], "resolved": []}
    for T_val in want:
        ln_Zn, se, _seen = model_ln_Z_n(n_all, log_w, cond, float(T_val), N)
        if ln_Zn is None:
            continue
        scale = KB_EV_PER_K * float(T_val) / N
        f = -scale * ln_Zn
        curves["T"].append(float(T_val))
        curves["f"].append([None if not np.isfinite(v) else float(v) for v in f])
        curves["se"].append([None if not np.isfinite(v) else float(scale * v)
                             for v in se])
        curves["resolved"].append(int(np.isfinite(ln_Zn).sum()))
    with open(os.path.join(a.run_dir, "fmix.json"), "w", encoding="utf-8") as fh:
        json.dump(curves, fh)

    out = a.out or os.path.join(a.run_dir, "common_tangent.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)
    print(f"{a.run_dir}: {len(rows)} tie lines -> {out}")
    for r in rows[:60]:
        print("  T=%5.0f  c %.3f -> %.3f   depth %.2e  (%d/%d compositions"
              " resolved, stitch se %.3f)" % (
                  r["temperature"], r["c_left"], r["c_right"], r["depth"],
                  r["n_resolved"], N + 1, r["stitch_se"]))


if __name__ == "__main__":
    main()
