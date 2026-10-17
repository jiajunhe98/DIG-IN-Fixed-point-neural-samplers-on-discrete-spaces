"""The Ising temperature bridge of the paper's appendix.

    python scripts/make_bridge_tex_data.py

Reads the scored bridge runs (`runs/bridge_scores/`, from `score_bridge.py`),
their training configs (`runs/bridge/`) and the Swendsen-Wang ground truth
(`runs/gt_samples/ising_L25/`, from `fixed_point/gt_samples/gt_sw.py`), and
writes into `results/bridge/`.

Written per run:

  numbers.tex                every table cell and every inline number
  pm_<pair>_t<k>.dat         P(m) of the transported draws at snapshot k
  pm_<pair>_ref<0|1>.dat     P(m) of the ground truth at each endpoint
  pmrw_<pair>_t8.dat         P(m) at t=1 after reweighting by the path weights
  flow_<pair>.dat            P(m) against transport time, as a matrix plot
  cfg_<pair>_t<k>.dat        one configuration at snapshot k, as a matrix plot

The free-energy difference between the two endpoints is known in closed form
(Kaufman), so `|dlnZ|` here is an error against ground truth and not against
another sampler.  `E_path[w] = Z_1/Z_0` exactly, so the estimator is
`logsumexp(logw) - log n`.

The scored distance is the MODE TVD, as everywhere in the paper: the TV
between the draws' distribution over the two ordered modes (the sign of m)
and the target's, which the spin-flip symmetry fixes at exactly (1/2, 1/2),
so it is |P(m > 0) - 1/2|.  It is quoted beside the floor an exact sampler
shows at the same draw count (raw) and at the Kish effective count of the
weights (reweighted).  The plotted curves bin the magnetization into 81 bins
on [-1.0125, 1.0125]: the ordered endpoints sit at |m| = 0.97 with a width of
about 0.01, and coarser bins would put both peaks in a single bin each.
"""

from __future__ import annotations

import glob
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "fixed_point", "bridge_ising"))
import ising_theory_stable as th   # noqa: E402

#: where the scored runs are read from, in order of preference
SRCS = (os.path.join(HERE, "runs", "bridge_scores"),)
GT = os.path.join(HERE, "runs", "gt_samples", "ising_L25")
OUT = os.path.join(HERE, "results", "bridge")


def runs_for(prefix: str):
    """Every seed of one transport, from the first source that has any."""
    for src in SRCS:
        hits = sorted(glob.glob(os.path.join(src, f"{prefix}-s*.npz")))
        if hits:
            return hits, src
    return [], None


def load(path: str):
    """One scored run, whichever of the two layouts it is in.

    The export stores the magnetization as an integer spin sum; the scorer's
    own output stores it as a float.  Everything downstream wants `m_snap`.
    """
    d = dict(np.load(path))
    if "m_snap" not in d:
        L = int(d["L"])
        d["m_snap"] = d["msum"].astype(np.float64) / (L * L)
    return d

#: macro stem -> (run prefix, beta_0, beta_1, printed name)
PAIRS = {
    "Cross": ("bw-b020-b060", 0.20, 0.60, r"$0.20\to0.60$"),
    "Ord": ("bw-b05-b06", 0.50, 0.60, r"$0.50\to0.60$"),
}
NBINS, MLO, MHI = 81, -1.0125, 1.0125
#: rows of the transport map that reach the .dat; the archive keeps every
#: recorded time, and this only bounds what one tikzpicture has to draw
FLOW_ROWS = 129
#: sample paths drawn over the transport map, picked at random with this seed
N_TRAJ, TRAJ_SEED = 8, 0
#: a path has lost its order at mid-time when |m(0.5)| is below this; the
#: split is clean either way -- paths sit near 0 or still near 0.9
DISSOLVE = 0.5
EDGES = np.linspace(MLO, MHI, NBINS + 1)
L = 25


def hist(m, w=None):
    """Normalized histogram of the magnetization on the shared grid."""
    h, _ = np.histogram(m, bins=EDGES, weights=w)
    return h / h.sum()


def write_hist(path, p):
    """A step plot: the left edge of each bin, then a closing point."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("m p\n")
        for e, v in zip(EDGES[:-1], p):
            f.write(f"{e:.5f} {v:.6g}\n")
        f.write(f"{EDGES[-1]:.5f} {p[-1]:.6g}\n")


def write_stairs(path, p, closed=False):
    """A histogram turned on its side: density along x, magnetization along y.

    `const plot` only steps along x, so the outline is written out point by
    point -- each bin a vertical run at its density between its two edges.
    With `closed`, the outline starts and ends on the axis (density 0), so a
    filled plot of it closes along that axis rather than across the bins.
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write("p m\n")
        if closed:
            f.write(f"0 {EDGES[0]:.5f}\n")
        for i, v in enumerate(p):
            f.write(f"{v:.6g} {EDGES[i]:.5f}\n{v:.6g} {EDGES[i + 1]:.5f}\n")
        if closed:
            f.write(f"0 {EDGES[-1]:.5f}\n")


def write_cfg(path, x):
    """One configuration as (col, row, spin) for `matrix plot*`."""
    g = np.asarray(x).reshape(L, L)
    with open(path, "w", encoding="utf-8") as f:
        f.write("i j v\n")
        for i in range(L):
            for j in range(L):
                f.write(f"{j} {i} {int(g[i, j])}\n")


def pm(vals, d, scale=1.0, lead=True):
    """`mean_{\\pm sd}` over seeds, the style of the other sections."""
    v = np.asarray([x for x in vals if x is not None], dtype=np.float64) * scale
    if not len(v):
        return "--"
    mu, sd = v.mean(), v.std()
    s_mu, s_sd = f"{mu:.{d}f}", f"{sd:.{d}f}"
    if not lead:
        s_mu = s_mu[1:] if s_mu.startswith("0.") else s_mu
        s_sd = s_sd[1:] if s_sd.startswith("0.") else s_sd
    return rf"${s_mu}_{{\pm{s_sd}}}$"


def gt_m(beta):
    """Ground-truth magnetizations at one temperature."""
    d = np.load(os.path.join(GT, f"beta{beta:.5f}_h+0.0000.npz"))
    return d["mean_spin"].astype(np.float64)


#: appendix hyperparameter table: macro suffix -> (cfg key, how to print)
HPARAMS = [
    ("Hidden", "hidden", str), ("Blocks", "n_blocks", str),
    ("Lr", "lr", lambda v: f"{v:g}"), ("Ema", "ema", lambda v: f"{v:g}"),
    ("Eta", "eta", lambda v: f"{v:g}"),
    ("EssTarget", "ess_target", lambda v: f"{v:g}"),
    ("Batch", "batch", str), ("BufCap", "buffer_cap", str),
    ("Explore", "n_explore", str), ("Inner", "n_inner", str),
    ("CtmcSteps", "ctmc_steps", str), ("PriorN", "prior_n", str),
    ("PriorSweeps", "prior_sweeps", str),
]
RUN_DIR = os.path.join(HERE, "runs", "bridge")


def hparam_macros():
    """The settings the runs trained under, read off their own configs.

    Every run shares these but the two that define the task (`beta0`) and the
    prior sampler that follows from it, so a disagreement is a bug and is
    raised rather than averaged away.
    """
    out, seen = [], {}
    for d in sorted(glob.glob(os.path.join(RUN_DIR, "bw-*"))):
        p = os.path.join(d, "cfg.json")
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            cfg = json.load(f)
        for suffix, key, _fmt in HPARAMS:
            seen.setdefault(suffix, set()).add(cfg[key])
    if not seen:
        print("  no run configs for the hyperparameter table")
        return out
    for suffix, key, fmt in HPARAMS:
        vals = seen[suffix]
        if len(vals) != 1:
            raise SystemExit(f"runs disagree on {key}: {sorted(vals)}")
        out.append(rf"\newcommand{{\BrHp{suffix}}}{{{fmt(vals.pop())}}}")
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    macros, n_seeds, n_draws, steps = [], set(), set(), set()

    for stem, (prefix, b0, b1, _name) in PAIRS.items():
        runs, src = runs_for(prefix)
        if not runs:
            print(f"  no scores: {prefix}")
            continue
        print(f"  {stem}: {len(runs)} seeds from {os.path.relpath(src, HERE)}")
        dlnz_exact = (float(th.calculate_ising_partition_function(b1, L)[0])
                      - float(th.calculate_ising_partition_function(b0, L)[0]))

        ref0, ref1 = hist(gt_m(b0)), hist(gt_m(b1))
        write_hist(os.path.join(OUT, f"pm_{stem}_ref0.dat"), ref0)
        write_hist(os.path.join(OUT, f"pm_{stem}_ref1.dat"), ref1)

        err, ess, tv_raw, tv_rw, bal_raw, bal_rw, mabs = [], [], [], [], [], [], []
        tv_prior, mabs_rw, neff = [], [], []
        logw_sd, cross_frac, cross_share, mid_cross = [], [], [], []
        diss_frac, diss_switch, diss_share, diss_ambig = [], [], [], []
        ess_diss, ess_kept, sd_diss, sd_kept = [], [], [], []
        for k, p in enumerate(runs):
            d = load(p)
            meta = json.loads(str(d["meta"]))
            n_draws.add(meta["draws"])
            steps.add(meta["steps"])
            lw = d["logw"].astype(np.float64)
            a = lw.max()
            w = np.exp(lw - a)
            est = a + math.log(w.mean())
            err.append(abs(est - dlnz_exact))
            ess.append(w.sum() ** 2 / (len(w) * (w ** 2).sum()))

            m1 = d["m_snap"][-1].astype(np.float64)
            wn = w / w.sum()
            bal_raw.append(float((m1 > 0).mean()))
            bal_rw.append(float(wn[m1 > 0].sum()))
            # Mode TVD: TV between the draws' distribution over the two ordered
            # modes (the sign of m; L = 25 is odd, so m is never 0) and the
            # target's, which the spin-flip symmetry fixes at exactly (1/2, 1/2).
            # For two modes that is |P(m > 0) - 1/2|.
            tv_raw.append(abs(bal_raw[-1] - 0.5))
            tv_rw.append(abs(bal_rw[-1] - 0.5))
            mabs.append(float(np.abs(m1).mean()))
            mabs_rw.append(float(np.sum(wn * np.abs(m1))))
            neff.append(ess[-1] * len(w))
            logw_sd.append(float(lw.std()))
            # Draws that end in the other mode from the one they started in,
            # and -- the split that matters -- draws that lose their order at
            # mid-time.  The interpolant couples X_0 and X_1 independently, so
            # between two ordered ends it pairs opposite modes half the time,
            # and those pairs sit at m ~ 0 halfway.  Most such paths return to
            # the mode they left; the mode-switchers are a subset of them.
            # Splitting on switching alone files the rest under "stayed" and
            # hides where the weight variance is.
            m0 = d["m_snap"][0].astype(np.float64)
            crossed = np.sign(m0) != np.sign(m1)
            cross_frac.append(float(crossed.mean()))
            cross_share.append(float(wn[crossed].sum()))
            # t = 0.5 is a row of the fine grid, whose length depends on
            # --m-every, so it is located rather than indexed
            half = int(np.argmin(np.abs(d["t_snap"] - 0.5)))
            mmid = np.abs(d["m_snap"][half].astype(np.float64))
            mid_cross.append(float(mmid[crossed].mean()))
            diss = mmid < DISSOLVE
            diss_frac.append(float(diss.mean()))
            diss_switch.append(float(crossed[diss].mean()) if diss.any() else 0.0)
            diss_share.append(float(wn[diss].sum()))
            # how clean the split is: paths neither near zero nor still ordered
            diss_ambig.append(float(((mmid > 0.2) & (mmid < 0.8)).mean()))
            for sel, ess_l, sd_l in ((diss, ess_diss, sd_diss), (~diss, ess_kept, sd_kept)):
                if sel.sum() > 1:
                    v = np.exp(lw[sel] - lw[sel].max())
                    ess_l.append(float(v.sum() ** 2 / (sel.sum() * (v ** 2).sum())))
                    sd_l.append(float(lw[sel].std()))
            # the prior bank itself, against ground truth at beta_0: the 0.50
            # runs build it by parallel tempering, and a Gibbs bank there would
            # already be mode-collapsed before the transport starts
            tv_prior.append(abs(float((d["m_snap"][0] > 0).mean()) - 0.5))

            if k == 0:                      # seed 0 draws the figures
                if "t_cfg" not in d:
                    raise SystemExit(f"{p}: scored before the two time grids "
                                     f"existed -- re-run scripts/score_bridge.py")
                # the map uses every recorded time; the panels use the coarse
                # grid the configurations were kept on, located within it
                t_fine, t_cfg = d["t_snap"], d["t_cfg"]
                idx = [int(np.argmin(np.abs(t_fine - t))) for t in t_cfg]
                # Which of the kept draws to draw as the example path.  Index 0
                # is not a safe default: on the ordered pair it happened to be
                # one of the ~9% that change mode, which is the opposite of
                # typical.  Take the first that keeps the sign it started with
                # -- and, where such a draw exists, also keep one that does not,
                # since the crossing paths get their own discussion.
                cm = 2.0 * d["cfg_snap"].astype(np.float64).mean(-1) - 1.0  # [T, k]
                keep = np.sign(cm[-1]) == np.sign(cm[0])
                typ = int(np.argmax(keep)) if keep.any() else 0
                crs = int(np.argmax(~keep)) if (~keep).any() else None

                for j, t in enumerate(t_cfg):
                    write_hist(os.path.join(OUT, f"pm_{stem}_t{j}.dat"),
                               hist(d["m_snap"][idx[j]].astype(np.float64)))
                    write_cfg(os.path.join(OUT, f"cfg_{stem}_t{j}.dat"),
                              d["cfg_snap"][j][typ])
                    if crs is not None:
                        write_cfg(os.path.join(OUT, f"cfgx_{stem}_t{j}.dat"),
                                  d["cfg_snap"][j][crs])
                macros.append(rf"\newcommand{{\Br{stem}CfgM}}"
                              rf"{{{cm[0, typ]:+.2f}}}")
                if crs is not None:
                    macros.append(rf"\newcommand{{\Br{stem}CfgXM}}"
                                  rf"{{{cm[0, crs]:+.2f} \to {cm[-1, crs]:+.2f}}}")
                write_hist(os.path.join(OUT, f"pmrw_{stem}_t8.dat"), hist(m1, wn))
                # the whole transport in one map: P(m) against t.  Each row is
                # scaled to its own maximum, so the shape of the distribution
                # is legible at every time instead of being swamped by the
                # ordered endpoint, whose peaks are ten times taller.
                # Thin the rows for the plot only.  Every recorded time is in
                # the archive; what goes into the .dat is bounded because each
                # cell is a separate filled path and TeX runs out of main
                # memory somewhere above ten thousand of them per picture.  At
                # FLOW_ROWS x NBINS the cells come out roughly square in the
                # printed panel, so more rows would not resolve anything.
                keep_j = np.unique(np.linspace(0, len(t_fine) - 1,
                                               min(FLOW_ROWS, len(t_fine))
                                               ).astype(int))
                cent = 0.5 * (EDGES[:-1] + EDGES[1:])
                cols = [hist(d["m_snap"][j].astype(np.float64)) for j in keep_j]
                cols = [c / c.max() for c in cols]
                # Time runs along x, so the file is written m-major: for each
                # magnetization bin, every time.  That is x varying fastest,
                # pgfplots' default mesh ordering, with mesh/cols = the number
                # of times.
                with open(os.path.join(OUT, f"flow_{stem}.dat"), "w",
                          encoding="utf-8") as f:
                    f.write("t m p\n")
                    for b, c in enumerate(cent):
                        for jj, j in enumerate(keep_j):
                            f.write(f"{t_fine[j]:.5f} {c:.5f} {cols[jj][b]:.6g}\n")

                # The two marginals, turned on their side so they share the
                # magnetization axis with the map: density runs along x.
                write_stairs(os.path.join(OUT, f"stair_{stem}_ref0.dat"), ref0, closed=True)
                write_stairs(os.path.join(OUT, f"stair_{stem}_ref1.dat"), ref1, closed=True)
                write_stairs(os.path.join(OUT, f"stair_{stem}_t0.dat"),
                             hist(d["m_snap"][0].astype(np.float64)))
                write_stairs(os.path.join(OUT, f"stair_{stem}_t1.dat"), hist(m1))
                write_stairs(os.path.join(OUT, f"stair_{stem}_rw.dat"), hist(m1, wn))

                # Individual paths over the map: N_TRAJ draws picked at random
                # with a fixed seed -- not chosen to look good.  Every recorded
                # time, since a path is 129 points and costs nothing to draw.
                rng_t = np.random.default_rng(TRAJ_SEED)
                pick = np.sort(rng_t.choice(d["m_snap"].shape[1], N_TRAJ,
                                            replace=False))
                paths = d["m_snap"][:, pick].astype(np.float64)
                with open(os.path.join(OUT, f"traj_{stem}.dat"), "w",
                          encoding="utf-8") as f:
                    f.write("t " + " ".join(f"d{i}" for i in range(N_TRAJ)) + "\n")
                    for j, t in enumerate(t_fine):
                        f.write(f"{t:.5f} " + " ".join(f"{v:.4f}" for v in paths[j]) + "\n")
                n_switch = int((np.sign(paths[0]) != np.sign(paths[-1])).sum())
                h_t = int(np.argmin(np.abs(t_fine - 0.5)))
                n_diss = int((np.abs(paths[h_t]) < DISSOLVE).sum())
                macros += [
                    rf"\newcommand{{\Br{stem}NTraj}}{{{N_TRAJ}}}",
                    rf"\newcommand{{\Br{stem}NTrajSwitch}}{{{n_switch}}}",
                    rf"\newcommand{{\Br{stem}NTrajDiss}}{{{n_diss}}}",
                ]
                # The map is drawn as a shaded surface, which spans exactly its
                # data, so the axis needs no half-cell margin -- it runs from
                # the first bin centre to the last and from t=0 to t=1.
                macros += [
                    rf"\newcommand{{\Br{stem}NTimes}}{{{len(keep_j)}}}",
                    rf"\newcommand{{\Br{stem}NRec}}{{{len(t_fine)}}}",
                    rf"\newcommand{{\Br{stem}Ts}}"
                    rf"{{{', '.join(f'{t:g}' for t in t_cfg)}}}",
                ]
        n_seeds.add(len(runs))

        gt1 = gt_m(b1)
        macros += [
            rf"\newcommand{{\Br{stem}DlnZExact}}{{{dlnz_exact:.1f}}}",
            rf"\newcommand{{\Br{stem}Err}}{{{pm(err, 3)}}}",
            rf"\newcommand{{\Br{stem}ErrRel}}{{{pm([e / abs(dlnz_exact) for e in err], 1, 1e4)}}}",
            rf"\newcommand{{\Br{stem}MTVPrior}}{{{pm(tv_prior, 3, lead=False)}}}",
            rf"\newcommand{{\Br{stem}MabsRW}}{{{pm(mabs_rw, 3, lead=False)}}}",
            rf"\newcommand{{\Br{stem}Neff}}{{{int(round(np.mean(neff)))}}}",
            rf"\newcommand{{\Br{stem}LogwSd}}{{{np.mean(logw_sd):.0f}}}",
            rf"\newcommand{{\Br{stem}CrossFrac}}{{{pm(cross_frac, 3, lead=False)}}}",
            rf"\newcommand{{\Br{stem}CrossShare}}{{{pm(cross_share, 3, lead=False)}}}",
            rf"\newcommand{{\Br{stem}MidCross}}{{{pm(mid_cross, 3, lead=False)}}}",
            rf"\newcommand{{\Br{stem}DissFrac}}{{{pm(diss_frac, 3, lead=False)}}}",
            rf"\newcommand{{\Br{stem}DissSwitch}}{{{pm(diss_switch, 3, lead=False)}}}",
            rf"\newcommand{{\Br{stem}DissShare}}{{{pm(diss_share, 3, lead=False)}}}",
            rf"\newcommand{{\Br{stem}DissAmbig}}{{{pm(diss_ambig, 3, lead=False)}}}",
            rf"\newcommand{{\Br{stem}EssDiss}}{{{pm(ess_diss, 3, lead=False)}}}",
            rf"\newcommand{{\Br{stem}EssKept}}{{{pm(ess_kept, 3, lead=False)}}}",
            rf"\newcommand{{\Br{stem}LogwSdDiss}}{{{np.mean(sd_diss):.0f}}}",
            rf"\newcommand{{\Br{stem}LogwSdKept}}{{{np.mean(sd_kept):.0f}}}",
            rf"\newcommand{{\BrDissolve}}{{{DISSOLVE:g}}}" if stem == "Cross" else "",
            rf"\newcommand{{\Br{stem}ESS}}{{{pm(ess, 3, lead=False)}}}",
            rf"\newcommand{{\Br{stem}MTVRaw}}{{{pm(tv_raw, 3, lead=False)}}}",
            rf"\newcommand{{\Br{stem}MTVRW}}{{{pm(tv_rw, 3, lead=False)}}}",
            rf"\newcommand{{\Br{stem}BalRaw}}{{{pm(bal_raw, 3, lead=False)}}}",
            rf"\newcommand{{\Br{stem}BalRW}}{{{pm(bal_rw, 3, lead=False)}}}",
            rf"\newcommand{{\Br{stem}Mabs}}{{{pm(mabs, 3, lead=False)}}}",
            rf"\newcommand{{\Br{stem}MabsRef}}{{{np.abs(gt1).mean():.3f}}}",
            rf"\newcommand{{\Br{stem}BetaZero}}{{{b0:.2f}}}",
            rf"\newcommand{{\Br{stem}BetaOne}}{{{b1:.2f}}}",
        ]
        # What an exact sampler would show: ground-truth draws scored against
        # the whole ground truth.  Twice -- once at the full draw count, once
        # at the Kish effective count of the weights, which is the honest floor
        # for a reweighted histogram and is several times the raw one.
        rng = np.random.default_rng(0)

        def floor_at(size):
            # an exact sampler's mode TVD at this many draws: |Bin(n, 1/2)/n - 1/2|
            n = max(int(round(size)), 1)
            return float(np.mean(np.abs(rng.binomial(n, 0.5, 20000) / n - 0.5)))

        macros += [
            rf"\newcommand{{\Br{stem}MTVFloor}}{{{floor_at(max(n_draws)):.3f}}}",
            rf"\newcommand{{\Br{stem}MTVFloorRW}}{{{floor_at(np.mean(neff)):.3f}}}",
        ]
        print(f"  {stem}: {len(runs)} seeds, exact dlnZ {dlnz_exact:.2f}, "
              f"err {np.mean(err):.3f}, ESS {np.mean(ess):.3f}, "
              f"mode TVD raw {np.mean(tv_raw):.3f} rw {np.mean(tv_rw):.3f}")

    macros += hparam_macros()
    macros += [
        rf"\newcommand{{\BrNSeeds}}{{{max(n_seeds) if n_seeds else 0}}}",
        rf"\newcommand{{\BrNDraws}}{{{max(n_draws) if n_draws else 0}}}",
        rf"\newcommand{{\BrSteps}}{{{max(steps) if steps else 0}}}",
        rf"\newcommand{{\BrL}}{{{L}}}",
        rf"\newcommand{{\BrNBins}}{{{NBINS}}}",
        # the columns of traj_<pair>.dat, for \pgfplotsinvokeforeach
        rf"\newcommand{{\BrTrajList}}{{{','.join(str(i) for i in range(N_TRAJ))}}}",
        rf"\newcommand{{\BrBetaC}}{{0.4407}}",
    ]
    with open(os.path.join(OUT, "numbers.tex"), "w", encoding="utf-8") as f:
        f.write("% generated by scripts/make_bridge_tex_data.py -- do not edit\n")
        f.write("\n".join(macros) + "\n")
    print(f"{OUT}/numbers.tex: {len(macros)} macros, "
          f"{len(glob.glob(os.path.join(OUT, '*.dat')))} data files")


if __name__ == "__main__":
    main()
