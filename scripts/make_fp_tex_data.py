"""The amortized (beta, h) experiment of the paper: maps and summary numbers.

    python scripts/make_fp_tex_data.py

Reads the scored grid (`runs/amortized_scores/<run>/`, from
`score_fixed_point.py`) and the reference (`runs/amortized_reference/`, from
`fp_reference.py` and `fp_mode_reference.py`), and writes into
`results/amortized/`:

  fp_numbers.tex                every number the section quotes, as macros
  fp_<sys>_mtv_{raw,rw}.dat     mode TVD per lattice cell, mean over seeds,
                                over the raw and over the reweighted draws
  fp_<sys>_ess.dat              ESS/n per lattice cell, mean over seeds
  fp_<sys>_comp_{ref,raw,rw}.dat   mean composition: reference, the model's
                                raw draws, and the model's draws reweighted

A separate macro file rather than numbers.tex, which the phase-sweep generator
owns and rewrites whole.

Definitions, all stated in the appendix
---------------------------------------
* Cells: the regular 9 x 9 lattice over each run's (beta, h) training box.  The
  runs were also scored at their 15 evaluation cells, which sit off that
  lattice; those are left out here so that an average over cells is a uniform
  average over the domain, not one weighted toward beta_c.
* m: the order parameter conjugate to h.  Ising: the mean spin.  Potts: the
  state-0 excess (q n_0 - 1)/(q - 1), n_0 = N_0/N.  Unlike the
  permutation-invariant (q f_max - 1)/(q - 1) of the phase sweep, this one is
  not blind to which state a configuration ordered into.
* composition c: the fraction of sites in the state the field favours
  (spin up; state 0), so that c = (1 + m)/2 for Ising and c = n_0 for Potts.
* mode TVD: each draw is assigned to one of the q ordered modes -- the sign
  of m for Ising (m = 0 shared between the two), the majority colour for Potts
  (a tie shared between the tied colours) -- and the draws' distribution over
  the q modes is compared in total variation with the reference's.  Under a
  field the reference distribution is not uniform: for Ising it follows from
  the stored counts of M, for Potts from runs/amortized_reference/
  potts_modes.json (`fp_mode_reference.py`), since N_0 alone does not
  determine the majority colour.  Raw: over the draws; reweighted: each draw
  carrying its self-normalised importance weight.
* |Delta ln Z|: |logmeanexp(log w) - ln Z_ref|.
* ESS: Kish, over the 5000 raw weights.
"""

from __future__ import annotations

import glob
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(HERE, "runs", "amortized")
SCORES = os.path.join(HERE, "runs", "amortized_scores")
REF = os.path.join(HERE, "runs", "amortized_reference")
GT = os.path.join(HERE, "runs", "gt_samples")
OUT = os.path.join(HERE, "results", "amortized")


def _runs(prefix: str) -> list[str]:
    """Every scored seed of a family (`experiments/amortized.py` names them)."""
    return sorted(os.path.basename(p)
                  for p in glob.glob(os.path.join(SCORES, f"{prefix}-s*")))


SYSTEMS = {
    "ising": dict(runs=_runs("ising50"), L=50, q=2, stem="Ising"),
    "potts": dict(runs=_runs("potts24"), L=24, q=3, stem="Potts"),
}
BINS, RANGE = 41, (-1.02, 1.02)


def lattice(sysname):
    """The 9 x 9 cells, read back from a run's own config."""
    if not SYSTEMS[sysname]["runs"]:
        raise SystemExit(f"no scored {sysname} runs under {SCORES}")
    run = SYSTEMS[sysname]["runs"][0]
    cfg = json.load(open(os.path.join(RUNS, run, "cfg.json")))
    betas = [round(float(b), 6) for b in np.linspace(cfg["beta_lo"], cfg["beta_hi"], 9)]
    hs = [round(float(h), 6) for h in np.linspace(-cfg["h_max"], cfg["h_max"], 9)]
    return betas, hs, cfg


def cell_file(run, b, h):
    return os.path.join(SCORES, run, f"beta{b:.5f}_h{h:+.4f}.npz")


def order_param(sysname, z):
    """m for each of the model's draws."""
    if sysname == "ising":
        return z["mean_spin"].astype(np.float64)
    q = SYSTEMS[sysname]["q"]
    return (q * z["state_fractions"][:, 0].astype(np.float64) - 1.0) / (q - 1.0)


def m_of_counts(sysname, n):
    """The m value of each index of a reference count vector."""
    k = np.arange(n + 1, dtype=np.float64)
    if sysname == "ising":
        return (2.0 * k - n) / n                        # index = (M + N)/2
    q = SYSTEMS[sysname]["q"]
    return (q * k / n - 1.0) / (q - 1.0)                # index = N_0


def comp_of_m(sysname, m):
    return (1.0 + m) / 2.0 if sysname == "ising" else (m * (SYSTEMS[sysname]["q"] - 1) + 1) / SYSTEMS[sysname]["q"]


# ---------------------------------------------------------------------------
# reference
# ---------------------------------------------------------------------------
def load_reference(sysname):
    """{(beta, h): dict(lnZ, se, counts)} over the lattice, or {} if absent."""
    out = {}
    if sysname == "ising":
        for f in glob.glob(os.path.join(REF, "ising_h_b*.json")):
            r = json.load(open(f))
            b = round(float(r["beta"]), 6)
            for key, c in r["cells"].items():
                h = round(float(key), 6)
                cnt = np.array(c["counts"], dtype=np.float64)
                out[(b, h)] = dict(lnZ=c["lnZ"], se=c["se"], counts=cnt)
                # spin flip: ln Z is even in h and P(M) mirrors
                out[(b, -h)] = dict(lnZ=c["lnZ"], se=c["se"], counts=cnt[::-1].copy())
        for r in out.values():
            # the modes are (m < 0, m > 0); M = 0 is shared between the two
            N = len(r["counts"]) - 1
            M = 2 * np.arange(N + 1) - N
            c = r["counts"]
            neg = c[M < 0].sum() + 0.5 * c[M == 0].sum()
            pos = c[M > 0].sum() + 0.5 * c[M == 0].sum()
            r["modes"] = np.array([neg, pos]) / c.sum()
        return out
    fb = os.path.join(REF, "potts_beta.json")
    if not os.path.exists(fb):
        return out
    base = json.load(open(fb))["cells"]
    for f in glob.glob(os.path.join(REF, "potts_h_b*.json")):
        r = json.load(open(f))
        b = round(float(r["beta"]), 6)
        anchor = base[f"{b:.6f}"]
        for key, c in r["cells"].items():
            h = round(float(key), 6)
            out[(b, h)] = dict(lnZ=anchor["lnZ"] + c["dlnZ"],
                               se=math.hypot(anchor["se"], c["se"]),
                               counts=np.array(c["counts"], dtype=np.float64))
        # the h-ladder file stores cells relative to h = 0; h = 0 itself is the anchor
        out[(b, 0.0)] = dict(lnZ=anchor["lnZ"], se=anchor["se"],
                             counts=np.array(anchor["counts"], dtype=np.float64))
    # the distribution over the majority colour, which N_0 does not determine:
    # sampled separately by fp_mode_reference.py with the same SW chains
    fm = os.path.join(REF, "potts_modes.json")
    if os.path.exists(fm):
        for c in json.load(open(fm))["cells"]:
            key = (round(float(c["beta"]), 6), round(float(c["h"]), 6))
            if key in out:
                v = np.asarray(c["mode_counts"], dtype=np.float64)
                out[key]["modes"] = v / v.sum()
    return out


# ---------------------------------------------------------------------------
# per cell
# ---------------------------------------------------------------------------
def model_modes(sysname, z):
    """[n, q]: each draw's share of each ordered mode (1 for its mode; a tie is
    shared equally).  Ising: (m < 0, m > 0).  Potts: the majority colour."""
    if sysname == "ising":
        m = z["mean_spin"].astype(np.float64)
        return np.stack([(m < 0) + 0.5 * (m == 0), (m > 0) + 0.5 * (m == 0)], 1)
    f = z["state_fractions"].astype(np.float64)
    top = f == f.max(1, keepdims=True)
    return top / top.sum(1, keepdims=True)


def mode_tv(p, pref):
    return float(0.5 * np.abs(np.asarray(p) - np.asarray(pref)).sum())


def mode_tv_floor(pref, n, b, h, reps=400, salt=0):
    """An exact sampler's mode TVD at n draws: multinomial draws from pref."""
    rng = np.random.default_rng((abs(hash((round(b, 6), round(h, 6)))) + salt) % (2 ** 32))
    k = rng.multinomial(n, pref, size=reps) / n
    return float(np.mean(0.5 * np.abs(k - pref).sum(1)))


def cell_metrics(sysname, run, b, h, ref):
    z = np.load(cell_file(run, b, h))
    lw = z["log_w"].astype(np.float64)
    n = len(lw)
    mx = lw.max()
    w = np.exp(lw - mx)
    ess = float(w.sum() ** 2 / (n * (w ** 2).sum()))
    lnz = float(mx + math.log(w.mean()))
    wn = w / w.sum()
    m = order_param(sysname, z)
    c = comp_of_m(sysname, m)
    # which mode a draw sits in: m > 0 for Ising, state 0 in the majority for Potts
    mode = m > (0.0 if sysname == "ising" else 0.25)
    out = dict(ess=ess, lnz=lnz, c_raw=float(c.mean()), c_rw=float((wn * c).sum()),
               mode_raw=float(mode.mean()), mode_rw=float(wn[mode].sum()))
    r = ref.get((round(b, 6), round(h, 6)))
    if r is not None:
        N = SYSTEMS[sysname]["L"] ** 2
        mref = m_of_counts(sysname, N)
        # mode TVD, raw and with each draw carrying its self-normalised weight
        md = model_modes(sysname, z)
        if "modes" in r:
            out["tv"] = mode_tv(md.mean(0), r["modes"])
            out["tv_rw"] = mode_tv((wn[:, None] * md).sum(0), r["modes"])
        out["dlnz"] = abs(lnz - r["lnZ"])
        out["ref_se"] = r["se"]
        out["c_ref"] = float((r["counts"] * comp_of_m(sysname, mref)).sum() / r["counts"].sum())
        # What an unbiased sampler would show at this draw count.  |Delta ln Z|:
        # the mean of |N(0, se_model^2 + se_ref^2)|, with the model's own error
        # from its ESS by the delta method.  Mode TVD: 5000 exact draws from the
        # reference mode distribution, fixed seed per cell.
        se_m = math.sqrt(max(1.0 / ess - 1.0, 0.0) / n)
        out["dlnz_floor"] = math.sqrt(2.0 / math.pi) * math.hypot(se_m, r["se"])
        if "modes" in r:
            out["tv_floor"] = mode_tv_floor(r["modes"], n, b, h)
            # a weighted estimate is as noisy as its Kish effective count of
            # exact draws, so that is the floor the reweighted TVD is held to
            n_eff = max(int(round(1.0 / float((wn ** 2).sum()))), 1)
            out["tv_rw_floor"] = mode_tv_floor(r["modes"], n_eff, b, h, salt=1)
    return out


def tv_floor(counts, mref, pr, n, b, h, reps=20, salt=0):
    rng = np.random.default_rng((abs(hash((round(b, 6), round(h, 6)))) + salt) % (2 ** 32))
    p = counts / counts.sum()
    out = []
    for _ in range(reps):
        k = rng.multinomial(n, p)
        ps, _ = np.histogram(mref, bins=BINS, range=RANGE, weights=k)
        out.append(0.5 * np.abs(ps / n - pr).sum())
    return float(np.mean(out))


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------
def write_map(path, betas, hs, grid):
    """Row-major in beta, h fastest: what `matrix plot*` with mesh/cols reads."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("h beta v\n")
        for i, b in enumerate(betas):
            for j, h in enumerate(hs):
                v = grid[i, j]
                f.write(f"{h:.6f} {b:.6f} {'nan' if not np.isfinite(v) else f'{v:.6f}'}\n")
            f.write("\n")


def pm(vals, d, scale=1.0, lead=True):
    """`mean_{\\pm sd}` over seeds, the phase-sweep tables' cell style."""
    v = np.asarray(vals, dtype=np.float64) * scale
    mu, sd = v.mean(), v.std()
    s_mu, s_sd = f"{mu:.{d}f}", f"{sd:.{d}f}"
    if not lead:
        s_mu = s_mu[1:] if s_mu.startswith("0.") else s_mu
        s_sd = s_sd[1:] if s_sd.startswith("0.") else s_sd
    return rf"${s_mu}_{{\pm{s_sd}}}$"


def main():
    os.makedirs(OUT, exist_ok=True)
    macros = []
    have_ref_all = True
    for sysname, S in SYSTEMS.items():
        betas, hs, cfg = lattice(sysname)
        ref = load_reference(sysname)
        n_ref = sum((round(b, 6), round(h, 6)) in ref for b in betas for h in hs)
        have_ref = n_ref == len(betas) * len(hs)
        have_ref_all &= have_ref
        print(f"{sysname}: {len(S['runs'])} runs x {len(betas)}x{len(hs)} cells, "
              f"reference at {n_ref}/{len(betas) * len(hs)}")

        per = {k: np.full((len(S["runs"]), len(betas), len(hs)), np.nan)
               for k in ("ess", "tv", "dlnz", "c_raw", "c_rw", "c_ref", "ref_se",
                         "mode_raw", "mode_rw", "dlnz_floor", "tv_floor",
                         "tv_rw", "tv_rw_floor")}
        for si, run in enumerate(S["runs"]):
            for i, b in enumerate(betas):
                for j, h in enumerate(hs):
                    mtr = cell_metrics(sysname, run, b, h, ref)
                    for k, v in mtr.items():
                        if k in per:
                            per[k][si, i, j] = v

        st = S["stem"]
        write_map(os.path.join(OUT, f"fp_{sysname}_ess.dat"), betas, hs, per["ess"].mean(0))
        write_map(os.path.join(OUT, f"fp_{sysname}_comp_raw.dat"), betas, hs, per["c_raw"].mean(0))
        write_map(os.path.join(OUT, f"fp_{sysname}_comp_rw.dat"), betas, hs, per["c_rw"].mean(0))
        macros += [
            rf"\newcommand{{\FP{st}ESS}}{{{pm(per['ess'].mean((1, 2)), 3, lead=False)}}}",
            # the lowest cell of the map as drawn, i.e. of the seed average
            rf"\newcommand{{\FP{st}ESSmin}}{{{np.nanmin(per['ess'].mean(0)):.2f}}}",
            rf"\newcommand{{\FP{st}Rounds}}{{{rounds(S['runs'])}}}",
            rf"\newcommand{{\FP{st}NSeeds}}{{{len(S['runs'])}}}",
            rf"\newcommand{{\FP{st}BetaLo}}{{{cfg['beta_lo']}}}",
            rf"\newcommand{{\FP{st}BetaHi}}{{{cfg['beta_hi']}}}",
            rf"\newcommand{{\FP{st}Hmax}}{{{cfg['h_max']}}}",
            rf"\newcommand{{\FP{st}Steps}}{{{cfg['eval_steps']}}}",
        ]
        if have_ref:
            # mode TVD both ways: over the raw draws (the sampler's own distribution)
            # and over the draws weighted by their self-normalised importance
            # weights (what an estimate from this sampler actually uses)
            tv_raw, tv_rw = per["tv"].mean(0), per["tv_rw"].mean(0)
            write_map(os.path.join(OUT, f"fp_{sysname}_mtv_raw.dat"), betas, hs, tv_raw)
            write_map(os.path.join(OUT, f"fp_{sysname}_mtv_rw.dat"), betas, hs, tv_rw)
            write_map(os.path.join(OUT, f"fp_{sysname}_comp_ref.dat"), betas, hs, per["c_ref"][0])
            dc_raw = np.abs(per["c_raw"] - per["c_ref"]).mean((1, 2))
            dc_rw = np.abs(per["c_rw"] - per["c_ref"]).mean((1, 2))
            iw, jw = np.unravel_index(np.nanargmax(tv_rw), tv_rw.shape)
            macros += [
                rf"\newcommand{{\FP{st}MTVRaw}}{{{pm(per['tv'].mean((1, 2)), 3, lead=False)}}}",
                # four decimals: the seed spread of the reweighted TV is ~2e-4,
                # which three would print as a misleading "+-.000"
                rf"\newcommand{{\FP{st}MTVRW}}{{{pm(per['tv_rw'].mean((1, 2)), 4, lead=False)}}}",
                rf"\newcommand{{\FP{st}Z}}{{{pm(per['dlnz'].mean((1, 2)), 1, scale=1e3)}}}",
                rf"\newcommand{{\FP{st}RefSE}}{{{np.nanmean(per['ref_se'][0]) * 1e3:.1f}}}",
                rf"\newcommand{{\FP{st}RefSEmax}}{{{np.nanmax(per['ref_se'][0]) * 1e3:.1f}}}",
                rf"\newcommand{{\FP{st}CErrRaw}}{{{pm(dc_raw, 1, scale=1e3)}}}",
                rf"\newcommand{{\FP{st}CErrRW}}{{{pm(dc_rw, 1, scale=1e3)}}}",
                rf"\newcommand{{\FP{st}MTVRawmax}}{{{np.nanmax(tv_raw):.2f}}}",
                rf"\newcommand{{\FP{st}MTVRWmax}}{{{np.nanmax(tv_rw):.3f}}}",
                rf"\newcommand{{\FP{st}MTVRWmaxFloor}}{{{np.nanmean(per['tv_rw_floor'][:, iw, jw]):.3f}}}",
                rf"\newcommand{{\FP{st}MTVRWmaxBeta}}{{{betas[iw]:g}}}",
                rf"\newcommand{{\FP{st}MTVRWmaxH}}{{{hs[jw]:g}}}",
                # the same metrics for an unbiased sampler at this draw count;
                # for the reweighted TV, at the effective count of the weights
                rf"\newcommand{{\FP{st}ZFloor}}{{{np.nanmean(per['dlnz_floor']) * 1e3:.1f}}}",
                rf"\newcommand{{\FP{st}MTVRawFloor}}{{{np.nanmean(per['tv_floor'][0]):.3f}}}",
                rf"\newcommand{{\FP{st}MTVRWFloor}}{{{np.nanmean(per['tv_rw_floor']):.4f}}}",
            ]
            # raw TV on the zero-field line against everywhere else: the line is
            # where the field's selection between modes jumps
            k0 = [j for j, h in enumerate(hs) if abs(h) < 1e-12][0]
            off = [j for j in range(len(hs)) if j != k0]
            macros += [
                rf"\newcommand{{\FP{st}MTVRawZero}}{{{np.nanmean(tv_raw[:, k0]):.2f}}}",
                rf"\newcommand{{\FP{st}MTVRawOff}}{{{np.nanmean(tv_raw[:, off]):.3f}}}",
                rf"\newcommand{{\FP{st}MTVRawFloorOff}}{{{np.nanmean(per['tv_floor'][0][:, off]):.3f}}}",
            ]
            if sysname == "potts":
                macros += potts_config_panels(betas, hs)
            if sysname == "ising":
                macros += worst_cell_macros(per["tv"].mean(0), betas, hs, ref)
                # at h = 0 the target puts exactly half its mass at m > 0
                raw, rw = per["mode_raw"][:, :, k0], per["mode_rw"][:, :, k0]
                macros += [
                    rf"\newcommand{{\FPIsingBalRawLo}}{{{np.nanmin(raw):.2f}}}",
                    rf"\newcommand{{\FPIsingBalRawHi}}{{{np.nanmax(raw):.2f}}}",
                    rf"\newcommand{{\FPIsingBalRWLo}}{{{np.nanmin(rw):.2f}}}",
                    rf"\newcommand{{\FPIsingBalRWHi}}{{{np.nanmax(rw):.2f}}}",
                ]
        else:
            for k in ("MTVRaw", "MTVRW", "Z", "RefSE", "RefSEmax", "CErrRaw", "CErrRW"):
                macros.append(rf"\newcommand{{\FP{st}{k}}}{{\todo{{ref}}}}")

    macros.append(rf"\newcommand{{\FPNDraws}}{{5000}}")
    macros += gt_macros()
    macros += reference_settings()
    # One colour scale per TV figure, shared by its Ising and Potts panels: the
    # smallest round ceiling over the data.  The raw and reweighted figures get
    # their own, since the reweighted maps sit an order of magnitude lower and
    # would read as uniformly dark on the raw scale.
    for kind, name in (("raw", "FPMTVRawScale"), ("rw", "FPMTVRWScale")):
        tvmax = 0.0
        for sysname in SYSTEMS:
            f = os.path.join(OUT, f"fp_{sysname}_mtv_{kind}.dat")
            if os.path.exists(f):
                v = np.loadtxt(f, skiprows=1, usecols=2)
                tvmax = max(tvmax, float(np.nanmax(v)))
        scale = next((c for c in (0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0) if tvmax <= c), 1.0)
        macros.append(rf"\newcommand{{\{name}}}{{{scale}}}")
    macros += validation_macros()
    with open(os.path.join(OUT, "fp_numbers.tex"), "w", encoding="utf-8") as f:
        f.write("% generated by scripts/make_fp_tex_data.py -- do not edit\n")
        f.write("\n".join(macros) + "\n")
    print(f"{OUT}/fp_numbers.tex: {len(macros)} macros"
          + ("" if have_ref_all else "  (reference incomplete: placeholders written)"))


SEED_LETTER = "ABC"          # LaTeX macro names cannot carry digits


def potts_config_panels(betas, hs):
    """Three Potts draws of the amortized model at h = 0, for the main figure.

    One per phase: the coldest, the hottest and the lattice beta nearest the
    zero-field critical point ln(1 + sqrt 3).  From seed 0, the draw whose
    permutation-invariant order parameter (q f_max - 1)/(q - 1) is closest to
    the median of the cell -- typical in the observable that tells the phases
    apart.  Written as `x y v` rows for `matrix plot*`, one lattice row per
    block.
    """
    S = SYSTEMS["potts"]
    L, q = S["L"], S["q"]
    j0 = [j for j, h in enumerate(hs) if abs(h) < 1e-12][0]
    bc = math.log(1.0 + math.sqrt(3.0))
    pick = {"Dis": min(betas), "Crit": min(betas, key=lambda b: abs(b - bc)),
            "Ord": max(betas)}
    out = []
    for tag, b in pick.items():
        z = np.load(cell_file(S["runs"][0], b, hs[j0]))
        x = z["x"].astype(np.int64)
        frac = np.stack([(x == k).mean(1) for k in range(q)], 1)
        m = (q * frac.max(1) - 1.0) / (q - 1.0)
        cfg = x[int(np.argmin(np.abs(m - np.median(m))))].reshape(L, L)
        with open(os.path.join(OUT, f"fp_potts_cfg_{tag}.dat"), "w", encoding="utf-8") as f:
            f.write("x y v\n")
            for i in range(L):
                for j in range(L):
                    f.write(f"{j} {i} {int(cfg[i, j])}\n")
                f.write("\n")
        out.append(rf"\newcommand{{\FPPottsCfgBeta{tag}}}{{{b:g}}}")
    return out


def worst_cell_macros(tvmap, betas, hs, ref):
    """The Ising cell with the largest seed-averaged raw mode TVD, looked at closely.

    Writes the distribution of m at that cell around each ordered mode, on
    fine bins aligned to the lattice of values m takes (steps of 2/N), for the
    reference and for every seed raw and reweighted; and the numbers the
    appendix quotes to decide whether the excess is a collapsed mode (no draws
    in one mode, which no weight can repair and which costs ln 2 in ln Z) or a
    misweighted one (draws in both, repaired by the weights).
    """
    S, N = SYSTEMS["ising"], SYSTEMS["ising"]["L"] ** 2
    i, j = np.unravel_index(np.nanargmax(tvmap), tvmap.shape)
    b, h = betas[i], hs[j]
    r = ref[(b, h)]
    mref = m_of_counts("ising", N)
    # 4 lattice values of m per bin, edges halfway between them so no value
    # sits on an edge; the modes sit near +-0.98 and nothing lies below |m|=0.95
    step = 2.0 / N
    hi = 1.0 + step / 2
    edges = hi - 4 * step * np.arange(16)[::-1]            # ascending, |m| side
    series = {"ref": r["counts"]}
    out = [rf"\newcommand{{\FPWorstBeta}}{{{b}}}",
           rf"\newcommand{{\FPWorstH}}{{{h:g}}}",
           rf"\newcommand{{\FPWorstMTV}}{{{tvmap[i, j]:.2f}}}",
           rf"\newcommand{{\FPWorstRefPos}}{{{r['counts'][mref > 0].sum() / r['counts'].sum():.3f}}}"]
    mins, ess_all, rw_all, tvrw_all, zs_all, err_all, mabs = [], [], [], [], [], [], []
    for s, run in enumerate(S["runs"]):
        z = np.load(cell_file(run, b, h))
        m = z["mean_spin"].astype(np.float64)
        lw = z["log_w"].astype(np.float64)
        n = len(lw)
        w = np.exp(lw - lw.max())
        ess = float(w.sum() ** 2 / (n * (w ** 2).sum()))
        wn = w / w.sum()
        lnz = float(lw.max() + math.log(w.mean()))
        se = math.hypot(math.sqrt(max(1 / ess - 1, 0) / n), r["se"])
        md = model_modes("ising", z)
        tv_raw_s = mode_tv(md.mean(0), r["modes"])
        tv_rw_s = mode_tv((wn[:, None] * md).sum(0), r["modes"])
        npos, nneg = int((m > 0).sum()), int((m < 0).sum())
        mins.append(min(npos, nneg))
        ess_all.append(ess)
        rw_all.append(float(wn[m > 0].sum()))
        tvrw_all.append(tv_rw_s)
        zs_all.append(abs(lnz - r["lnZ"]) / se)
        err_all.append(abs(lnz - r["lnZ"]))
        mabs += [np.abs(m[m > 0]).mean(), np.abs(m[m < 0]).mean()]
        L = SEED_LETTER[s]
        out += [rf"\newcommand{{\FPWorstNeg{L}}}{{{nneg}}}",
                rf"\newcommand{{\FPWorstPos{L}}}{{{npos}}}",
                rf"\newcommand{{\FPWorstRaw{L}}}{{{(m > 0).mean():.2f}}}",
                rf"\newcommand{{\FPWorstRW{L}}}{{{wn[m > 0].sum():.2f}}}",
                rf"\newcommand{{\FPWorstESS{L}}}{{{ess:.2f}}}",
                rf"\newcommand{{\FPWorstMTVRaw{L}}}{{{tv_raw_s:.3f}}}",
                rf"\newcommand{{\FPWorstMTVRW{L}}}{{{tv_rw_s:.3f}}}",
                rf"\newcommand{{\FPWorstZ{L}}}{{{(lnz - r['lnZ']) * 1e3:+.0f}}}",
                rf"\newcommand{{\FPWorstZs{L}}}{{{(lnz - r['lnZ']) / se:+.1f}}}"]
        series[f"raw{L}"] = m
        series[f"rw{L}"] = (m, wn)
    c = r["counts"]
    ref_mabs = float((c * np.abs(mref)).sum() / c.sum())
    out += [rf"\newcommand{{\FPWorstMinDraws}}{{{min(mins)}}}",
            rf"\newcommand{{\FPWorstESSmin}}{{{min(ess_all):.2f}}}",
            rf"\newcommand{{\FPWorstESSmax}}{{{max(ess_all):.2f}}}",
            rf"\newcommand{{\FPWorstRWlo}}{{{min(rw_all):.2f}}}",
            rf"\newcommand{{\FPWorstRWhi}}{{{max(rw_all):.2f}}}",
            rf"\newcommand{{\FPWorstMTVRWlo}}{{{min(tvrw_all):.3f}}}",
            rf"\newcommand{{\FPWorstMTVRWhi}}{{{max(tvrw_all):.3f}}}",
            rf"\newcommand{{\FPWorstZsMax}}{{{max(zs_all):.1f}}}",
            # how many times the largest ln Z error here a missing mode would cost
            rf"\newcommand{{\FPWorstLnTwoRatio}}{{{int(math.log(2) / max(err_all))}}}",
            rf"\newcommand{{\FPWorstMabsLo}}{{{min(mabs):.3f}}}",
            rf"\newcommand{{\FPWorstMabsHi}}{{{max(mabs):.3f}}}",
            rf"\newcommand{{\FPWorstMabsRef}}{{{ref_mabs:.3f}}}"]

    # one file per mode, so the step plots do not join across the empty middle
    ytop = 0.0
    for sign, tag in ((-1, "neg"), (+1, "pos")):
        e = np.sort(sign * edges)
        cols = {}
        pr, _ = np.histogram(mref, bins=e, weights=r["counts"])
        cols["ref"] = pr / r["counts"].sum()
        for s in range(len(S["runs"])):
            L = SEED_LETTER[s]
            m = series[f"raw{L}"]
            a, _ = np.histogram(m, bins=e)
            cols[f"raw{L}"] = a / len(m)
            mw, wn = series[f"rw{L}"]
            a, _ = np.histogram(mw, bins=e, weights=wn)
            cols[f"rw{L}"] = a
        ytop = max(ytop, max(float(np.max(v)) for v in cols.values()))
        with open(os.path.join(OUT, f"fp_ising_worst_{tag}.dat"), "w", encoding="utf-8") as f:
            names = list(cols)
            f.write("m " + " ".join(names) + "\n")
            # `const plot` draws from each x to the next: give left edges, and
            # repeat the last value at the right edge so the final bin closes
            for k in range(len(e) - 1):
                f.write(f"{e[k]:.5f} " + " ".join(f"{cols[c][k]:.6f}" for c in names) + "\n")
            f.write(f"{e[-1]:.5f} " + " ".join(f"{cols[c][-1]:.6f}" for c in names) + "\n")
    # one y scale for all six panels, so the split between modes reads directly
    out.append(rf"\newcommand{{\FPWorstYmax}}{{{math.ceil(ytop * 20) / 20:.2f}}}")
    return out


def rounds(runs):
    """Training rounds at the scored checkpoint, as a range over seeds; read
    from what each scored cell recorded, not from a config default."""
    r = sorted({json.loads(str(np.load(sorted(glob.glob(os.path.join(SCORES, run, "*.npz")))[0])["meta"]))["outer"]
                for run in runs})
    return f"{r[0]}" if len(r) == 1 else f"{r[0]}--{r[-1]}"


def reference_settings():
    """Ladder sizes and sampling settings, read from the reference outputs."""
    fi = sorted(glob.glob(os.path.join(REF, "ising_h_b*.json")))
    fb = os.path.join(REF, "potts_beta.json")
    fh = sorted(glob.glob(os.path.join(REF, "potts_h_b*.json")))
    if not (fi and os.path.exists(fb) and fh):
        return [rf"\newcommand{{\FPRef{k}}}{{\todo{{ref}}}}"
                for k in ("IsingRungs", "PottsBetaRungs", "PottsHRungs", "Chains",
                          "Burn", "RoundsH", "RoundsB")]
    ri, rb, rh = json.load(open(fi[0])), json.load(open(fb)), json.load(open(fh[0]))
    return [rf"\newcommand{{\FPRefIsingRungs}}{{{len(ri['ladder'])}}}",
            rf"\newcommand{{\FPRefPottsBetaRungs}}{{{len(rb['ladder'])}}}",
            rf"\newcommand{{\FPRefPottsHRungs}}{{{len(rh['ladder'])}}}",
            rf"\newcommand{{\FPRefChains}}{{{ri['args']['chains']}}}",
            rf"\newcommand{{\FPRefBurn}}{{{ri['args']['burn']}}}",
            rf"\newcommand{{\FPRefRoundsH}}{{{ri['args']['rounds']}}}",
            rf"\newcommand{{\FPRefRoundsB}}{{{rb['args']['rounds']}}}"]


def gt_macros():
    """The model against the archive's Swendsen-Wang ground truth.

    The runs were also scored at their 15 evaluation cells, which lie off the
    9 x 9 lattice but have 16384 exact samples each.  The reweighted energy
    per site there is a check on the scored data that does not depend on the
    reference built here.
    """
    gt = GT
    sub = {"ising": "ising_L50", "potts": "potts_q3_L24"}
    zs = []
    for sysname, S in SYSTEMS.items():
        for run in S["runs"]:
            for g in glob.glob(os.path.join(gt, sub[sysname], "*.npz")):
                f = os.path.join(SCORES, run, os.path.basename(g))
                if not os.path.exists(f):
                    continue
                z, t = np.load(f), np.load(g)
                lw = z["log_w"].astype(np.float64)
                w = np.exp(lw - lw.max())
                w /= w.sum()
                u = z["u"].astype(np.float64)
                mu = float((w * u).sum())
                se = float(np.sqrt(((w ** 2) * (u - mu) ** 2).sum()))
                ug = t["u"].astype(np.float64)
                zs.append(abs(mu - ug.mean()) / math.hypot(se, ug.std() / math.sqrt(len(ug))))
    if not zs:
        return [r"\newcommand{\FPGTN}{\todo{gt}}", r"\newcommand{\FPGTMaxZ}{\todo{gt}}",
                r"\newcommand{\FPGTMeanZ}{\todo{gt}}"]
    return [rf"\newcommand{{\FPGTN}}{{{len(zs)}}}",
            rf"\newcommand{{\FPGTMaxZ}}{{{max(zs):.1f}}}",
            rf"\newcommand{{\FPGTMeanZ}}{{{np.mean(zs):.1f}}}"]


def validation_macros():
    """How well the reference machinery reproduces what is exactly known."""
    out = []
    f = os.path.join(REF, "ising24_beta.json")
    if os.path.exists(f):
        r = json.load(open(f))
        z = [(c["lnZ"] - c["kaufman"]) / c["se"] for c in r["cells"].values() if c["se"] > 0]
        d = [abs(c["lnZ"] - c["kaufman"]) for c in r["cells"].values()]
        se = [c["se"] for c in r["cells"].values() if c["se"] > 0]
        out += [rf"\newcommand{{\FPValN}}{{{len(z)}}}",
                rf"\newcommand{{\FPValMaxZ}}{{{max(abs(v) for v in z):.1f}}}",
                rf"\newcommand{{\FPValMaxErr}}{{{max(d) * 1e3:.1f}}}",
                rf"\newcommand{{\FPValMaxSE}}{{{max(se) * 1e3:.1f}}}"]
    else:
        out += [rf"\newcommand{{\FPVal{k}}}{{\todo{{val}}}}" for k in ("N", "MaxZ", "MaxErr", "MaxSE")]
    fs = glob.glob(os.path.join(REF, "ising_h_b*.json"))
    if fs:
        de = [abs(json.load(open(f))["energy_h0_sampled"] - json.load(open(f))["energy_h0_exact"])
              for f in fs]
        out.append(rf"\newcommand{{\FPBurnCheck}}{{{max(de):.1e}}}")
    else:
        out.append(r"\newcommand{\FPBurnCheck}{\todo{val}}")
    return out


if __name__ == "__main__":
    main()
