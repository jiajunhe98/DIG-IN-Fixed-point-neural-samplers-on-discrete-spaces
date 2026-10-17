"""The alloy phase diagrams of the paper and their table.

    python scripts/make_alloy_tex_data.py

Reads the 5000-draw sweeps of `experiments/alloy.py`
(`runs/alloy_n5000/<system>/<method>/s<seed>/`: grid.json, common_tangent.json,
fmix.json) and the training histories (`runs/alloy/...`), prints the table, and
writes `results/alloy/`:

  <system>_ref_ties.dat         packaged common-tangent tie lines
  <system>_<method>_ties.dat    the method's tie lines, seed 0
  <system>_<method>_ess.dat     ESS on the (T, delta_mu) sweep, mean over seeds
  numbers.tex                   table entries and sweep counts as macros

A method with no finished sweep is left out.
"""

from __future__ import annotations

import glob
import json
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
#: the training runs (history.jsonl) and their 5000-draw re-evaluation
TRAIN = os.path.join(ROOT, "runs", "alloy")
RUNS = os.path.join(ROOT, "runs", "alloy_n5000")
#: the packaged exact reference of each system
REF = os.path.join(ROOT, "dds", "targets", "alloy", "data")
REF_DIR = {"agpd125": "agpd_N125", "cuau128": "cuau_N128"}
OUT = os.path.join(ROOT, "results", "alloy")

SYSTEMS = {"agpd125": "AgPd", "cuau128": "CuAu"}
DISPLAY = {"agpd125": "Ag-Pd", "cuau128": "Cu-Au"}
# key -> (run directory, macro name)
METHODS = {"dsi": ("digin", "DSI"),
           "frogs": ("frogs", "FrOGS")}
# macro suffix, grid.json field, better, format.  "sci" prints (m +- s) x 10^e
# per cell: across methods these errors span four orders of magnitude, so one
# shared scale factor for a column would either lose the small values or print
# the large ones with four digits.
METRICS = (("CErr", "c_err_rw", min, "sci"),
           ("ZErr", "log_Z_err_per_site", min, "sci"),
           ("ESS", "ess", max, "plain"))
N_DRAWS = 5000
#: the free-energy panels of the appendix.  Two isotherms of one system: the
#: coldest, where the ordered phases make the curve cusped and the sampler
#: reaches few compositions, and the last one before the well closes.
FMIX_SYSTEM = "cuau128"
FMIX_TEMPS = (200.0, 700.0)
KB = 8.617e-05  # eV/K


def load(system: str, method: str) -> list[dict]:
    """Every seed whose 5000-draw sweep has been scored: the per-state-point
    rows and the tie lines from the sweep, the cost from the training run."""
    runs = []
    for d in sorted(glob.glob(os.path.join(RUNS, system, method, "s*"))):
        grid, ct = os.path.join(d, "grid.json"), os.path.join(d, "common_tangent.json")
        if not (os.path.exists(grid) and os.path.exists(ct)):
            continue
        r = {"seed": os.path.basename(d),
             "grid": json.load(open(grid, encoding="utf-8")),
             "ct": json.load(open(ct, encoding="utf-8"))}
        fm = os.path.join(d, "fmix.json")
        if os.path.exists(fm):
            r["fmix"] = json.load(open(fm, encoding="utf-8"))
        h = os.path.join(TRAIN, system, method, r["seed"], "history.jsonl")
        if os.path.exists(h):
            rows = [json.loads(l) for l in open(h, encoding="utf-8") if l.strip()]
            r["final"] = next((x for x in reversed(rows) if x.get("final")), None)
        runs.append(r)
    return runs


def write_ties(path: str, ties: list[dict]) -> None:
    """Each tie line as two points and a nan row, so `unbounded coords=jump`
    draws it as its own segment."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("c T\n")
        for t in sorted(ties, key=lambda t: (t["temperature"], t["c_left"])):
            T = t["temperature"]
            f.write(f"{t['c_left']:.4f} {T:.0f}\n{t['c_right']:.4f} {T:.0f}\nnan nan\n")
        if not ties:
            f.write("nan nan\n")


def boundary_chains(ties: list[dict]) -> list[list[tuple[float, float, float]]]:
    """Group tie lines into two-phase regions that persist in temperature.

    A phase boundary is the locus of coexisting compositions, so the curve to
    draw is the chain of tie lines that overlap from one isotherm to the next --
    the same overlap rule the constructor's persistence criterion uses.  Cu-Au
    carries up to four two-phase regions on one isotherm, so they have to be
    chained rather than reduced to one region per temperature.
    """
    by_T: dict[float, list[tuple[float, float]]] = {}
    for t in ties:
        by_T.setdefault(round(float(t["temperature"]), 3), []).append(
            (float(t["c_left"]), float(t["c_right"])))
    chains: list[list[tuple[float, float, float]]] = []
    active: list[int] = []
    for T in sorted(by_T):
        nxt, used = [], set()
        for cl, cr in sorted(by_T[T]):
            best, bi = 0.0, None
            for ci in active:
                if ci in used:
                    continue
                _pT, pcl, pcr = chains[ci][-1]
                ov = min(cr, pcr) - max(cl, pcl)
                if ov > best:
                    best, bi = ov, ci
            if bi is None:
                chains.append([(T, cl, cr)])
                nxt.append(len(chains) - 1)
            else:
                chains[bi].append((T, cl, cr))
                used.add(bi)
                nxt.append(bi)
        active = nxt
    return chains


def write_bounds(path: str, ties: list[dict]) -> None:
    """One closed outline per two-phase region: up the left branch, across the
    top, down the right branch, closed along the lowest tie line."""
    with open(path, "w", encoding="utf-8") as f:
        f.write("c T\n")
        for ch in boundary_chains(ties):
            pts = [(cl, T) for T, cl, _cr in ch]
            pts += [(cr, T) for T, _cl, cr in reversed(ch)]
            pts.append(pts[0])
            for c, T in pts:
                f.write(f"{c:.4f} {T:.0f}\n")
            f.write("nan nan\n")
        if not ties:
            f.write("nan nan\n")


def write_ess(path: str, runs: list[dict]) -> tuple[list[float], list[float]]:
    """Seed-mean ESS in scanline order (T outer, delta_mu inner), which is what
    `matrix plot*` with `mesh/cols` = number of delta_mu values expects."""
    acc: dict[tuple[float, float], list[float]] = {}
    for r in runs:
        for g in r["grid"]:
            acc.setdefault((round(g["T"], 3), round(g["dmu"], 4)), []).append(g["ess"])
    pts = sorted(acc)
    Ts = sorted({k[0] for k in pts})
    dmus = sorted({k[1] for k in pts})
    if len(pts) != len(Ts) * len(dmus):
        raise ValueError(f"{path}: the sweep is not a full (T, delta_mu) grid")
    if len({len(v) for v in acc.values()}) != 1:
        raise ValueError(f"{path}: the seeds do not cover the same state points")
    with open(path, "w", encoding="utf-8") as f:
        f.write("T dmu ess\n")
        for k in pts:
            f.write(f"{k[0]:.0f} {k[1]:.3f} {np.mean(acc[k]):.4f}\n")
    return Ts, dmus


def write_fmix_panels(system: str, model_runs: list[dict],
                      ref_ties: list[dict]) -> dict:
    """Data for the appendix panels that show the common-tangent construction.

    Both curves are measured from the same mixing reference, the chord through
    the two pure compositions of the exact free energy.  The tangency points are
    invariant to adding any linear function of c, so the choice cannot move a
    boundary, and it keeps the model's own free-energy offset visible instead of
    absorbing it.
    """
    z = np.load(os.path.join(REF, REF_DIR[system], "lnZ.npz"))
    Tx = np.asarray(z["T"], float)
    cx = np.asarray(z["composition"], float)
    fmix = np.asarray(z["f_mix"], float)
    valid = np.asarray(z["valid"], bool)
    lnZ = np.asarray(z["lnZ"], float)
    N = int(z["n_sites"])
    mrun = model_runs[0]
    mf = mrun.get("fmix") or {}
    mT = [round(float(t)) for t in mf.get("T", [])]
    info: dict = {}
    lo, hi = np.inf, -np.inf

    def curve(name, cs, ys):
        with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
            f.write("c f\n")
            for c, y in zip(cs, ys):
                f.write(f"{float(c):.5f} {float(y):.6f}\n")
            if len(cs) == 0:
                f.write("nan nan\n")

    def tangents(name, ties, cs, ys):
        """Each accepted tie line as the chord between its two tangency points,
        read off the curve it was constructed from."""
        table = {round(float(c), 5): float(y) for c, y in zip(cs, ys)}
        n, rises = 0, []
        with open(os.path.join(OUT, name), "w", encoding="utf-8") as f:
            f.write("c f\n")
            for t in ties:
                pts = []
                for c in (float(t["c_left"]), float(t["c_right"])):
                    if not table:
                        break
                    k = min(table, key=lambda s: abs(s - c))
                    if abs(k - c) < 2e-3:
                        pts.append((c, table[k]))
                if len(pts) == 2:
                    (cl, yl), (cr, yr) = pts
                    # how far the curve stands above its own chord: the depth
                    # criterion (ii) tests, and the thing the panel has to show
                    inner = [(c, y) for c, y in zip(cs, ys) if cl < c < cr]
                    if inner:
                        rises.append(max(
                            y - float(np.interp(c, [cl, cr], [yl, yr]))
                            for c, y in inner))
                    for c, y in pts:
                        f.write(f"{c:.5f} {y:.6f}\n")
                    f.write("nan nan\n")
                    n += 1
            if not n:
                f.write("nan nan\n")
        return n, rises

    for tag, T in zip("AB", FMIX_TEMPS):
        j = int(np.argmin(np.abs(Tx - T)))
        ok = valid[j]
        cs_ref, ys_ref = cx[ok], fmix[j][ok]
        curve(f"{system}_fmix_ref_{tag}.dat", cs_ref, ys_ref)
        n_ref, rises = tangents(
            f"{system}_tangent_ref_{tag}.dat",
            [t for t in ref_ties if abs(t["temperature"] - T) < 1e-6],
            cs_ref, ys_ref)
        F = -(KB * float(Tx[j]) / N) * lnZ[j]
        chord = (1 - cx) * F[0] + cx * F[-1]
        cs_m, ys_m = [], []
        if round(T) in mT:
            i = mT.index(round(T))
            for c, v, ch in zip(mf["composition"], mf["f"][i], chord):
                if v is not None:
                    cs_m.append(float(c))
                    ys_m.append(float(v) - float(ch))
        curve(f"{system}_fmix_dsi_{tag}.dat", cs_m, ys_m)
        n_mod, _ = tangents(
            f"{system}_tangent_dsi_{tag}.dat",
            [t for t in mrun["ct"] if abs(t["temperature"] - T) < 1e-6],
            cs_m, ys_m)
        vals = list(ys_ref) + ys_m
        lo, hi = min(lo, min(vals)), max(hi, max(vals))
        info[tag] = {"T": float(Tx[j]), "resolved": len(cs_m), "n_comp": len(cx),
                     "n_ref": n_ref, "n_mod": n_mod,
                     "rise": (f"{min(rises) * 1000:.1f}--{max(rises) * 1000:.1f}"
                              if rises else "?")}
    pad = 0.06 * (hi - lo)
    info["lo"], info["hi"] = lo - pad, hi + pad
    print(f"  fmix panels: {system} "
          + " ".join(f"{t['T']:.0f}K {t['resolved']}/{t['n_comp']} resolved, "
                     f"{t['n_ref']} ref / {t['n_mod']} model tie lines"
                     for k, t in info.items() if k in "AB")
          + f"; y in [{info['lo']:.4f}, {info['hi']:.4f}]")
    return info


def train_hours(method: str) -> float | None:
    """Mean wall-clock training time over a method's runs, from the training
    histories (a run reports its cost before its 5000-draw sweep exists)."""
    vals = []
    for system in SYSTEMS:
        for h in sorted(glob.glob(os.path.join(
                TRAIN, system, method, "s*", "history.jsonl"))):
            rows = [json.loads(l) for l in open(h, encoding="utf-8") if l.strip()]
            fin = next((r for r in reversed(rows) if r.get("final")), None)
            if fin and fin.get("wall_train_s"):
                vals.append(float(fin["wall_train_s"]) / 3600)
    return float(np.mean(vals)) if vals else None


def seed_means(runs: list[dict], field: str) -> np.ndarray:
    return np.array([np.mean([g[field] for g in r["grid"]]) for r in runs])


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    macros = ["% Generated by scripts/make_alloy_tex_data.py -- edit that, not this."]
    table: list[tuple] = []
    n_seeds = {}

    for system, sysname in SYSTEMS.items():
        pd = json.load(open(os.path.join(REF, REF_DIR[system], "phase_diagram.json"),
                            encoding="utf-8"))
        write_ties(os.path.join(OUT, f"{system}_ref_ties.dat"), pd["common_tangent"])
        write_bounds(os.path.join(OUT, f"{system}_ref_bounds.dat"), pd["common_tangent"])

        runs = {}
        for key, (mdir, _name) in METHODS.items():
            rs = load(system, mdir)
            if rs:
                runs[key] = rs
            else:
                print(f"  {system}: no scored {mdir} sweep under {RUNS}; left out")
        if "dsi" not in runs:
            raise SystemExit(f"no DIG-IN sweep for {system} under {RUNS}")
        n_seeds[system] = len(runs["dsi"])
        for key, rs in runs.items():
            write_ties(os.path.join(OUT, f"{system}_{key}_ties.dat"), rs[0]["ct"])
            write_bounds(os.path.join(OUT, f"{system}_{key}_bounds.dat"), rs[0]["ct"])
            Ts, dmus = write_ess(os.path.join(OUT, f"{system}_{key}_ess.dat"), rs)

        if system == FMIX_SYSTEM:
            fx = write_fmix_panels(system, runs["dsi"], pd["common_tangent"])
            macros += [
                f"\\newcommand{{\\FmixTA}}{{{fx['A']['T']:.0f}}}",
                f"\\newcommand{{\\FmixTB}}{{{fx['B']['T']:.0f}}}",
                f"\\newcommand{{\\FmixResolvedA}}{{{fx['A']['resolved']}}}",
                f"\\newcommand{{\\FmixResolvedB}}{{{fx['B']['resolved']}}}",
                f"\\newcommand{{\\FmixNcomp}}{{{fx['A']['n_comp']}}}",
                f"\\newcommand{{\\FmixRiseA}}{{{fx['A']['rise']}}}",
                f"\\newcommand{{\\FmixRiseB}}{{{fx['B']['rise']}}}",
                f"\\newcommand{{\\FmixTiesRefA}}{{{fx['A']['n_ref']}}}",
                f"\\newcommand{{\\FmixTiesModA}}{{{fx['A']['n_mod']}}}",
                f"\\newcommand{{\\FmixTiesRefB}}{{{fx['B']['n_ref']}}}",
                f"\\newcommand{{\\FmixTiesModB}}{{{fx['B']['n_mod']}}}",
                f"\\newcommand{{\\FmixLo}}{{{fx['lo']:.4f}}}",
                f"\\newcommand{{\\FmixHi}}{{{fx['hi']:.4f}}}",
            ]

        g0 = runs["dsi"][0]["grid"]
        dT, dd = Ts[1] - Ts[0], dmus[1] - dmus[0]
        macros += [
            f"\\newcommand{{\\{sysname}NT}}{{{len(Ts)}}}",
            f"\\newcommand{{\\{sysname}Ndmu}}{{{len(dmus)}}}",
            f"\\newcommand{{\\{sysname}Npoints}}{{{len(g0)}}}",
            f"\\newcommand{{\\{sysname}Ntrapped}}{{{sum(bool(g['trapped']) for g in g0)}}}",
            f"\\newcommand{{\\{sysname}DmuMin}}{{{dmus[0]:.2f}}}",
            f"\\newcommand{{\\{sysname}DmuMax}}{{{dmus[-1]:.2f}}}",
            f"\\newcommand{{\\{sysname}DmuLo}}{{{dmus[0] - dd / 2:.3f}}}",
            f"\\newcommand{{\\{sysname}DmuHi}}{{{dmus[-1] + dd / 2:.3f}}}",
            f"\\newcommand{{\\{sysname}TLo}}{{{Ts[0] - dT / 2:.1f}}}",
            f"\\newcommand{{\\{sysname}THi}}{{{Ts[-1] + dT / 2:.1f}}}",
            f"\\newcommand{{\\{sysname}Nties}}{{{len(pd['common_tangent'])}}}",
            f"\\newcommand{{\\{sysname}TClose}}"
            f"{{{max(t['temperature'] for t in pd['common_tangent']):.0f}}}",
        ]

        # table cells: mean +- sd over seeds; the best is marked only when
        # there are at least two methods to compare
        measured = list(runs)
        for suffix, field, better, style in METRICS:
            means = {k: seed_means(rs, field) for k, rs in runs.items()}
            best = (better(measured, key=lambda k: means[k].mean())
                    if len(measured) > 1 else None)
            # One exponent per (system, metric), printed once in the row label
            # rather than in every cell: the table is narrow because it sits
            # beside the figure.  Taken from the geometric mean, since the
            # methods can sit two orders of magnitude apart and an exponent
            # from either end would waste digits on the other.
            scale = 1.0
            if style == "sci":
                mus = [float(m.mean()) for m in means.values() if m.mean() > 0]
                g = math.exp(sum(math.log(v) for v in mus) / len(mus))
                e = int(math.floor(math.log10(g)))
                scale = 10.0 ** e
                macros.append(
                    f"\\newcommand{{\\{sysname}{suffix}Unit}}{{10^{{{e}}}}}")
            for key, m in means.items():
                mu = float(m.mean()) / scale
                sd = (float(m.std(ddof=1)) if len(m) > 1 else 0.0) / scale
                if style == "plain":
                    dec = 3
                else:
                    dec = 2 - int(math.floor(math.log10(mu))) if mu > 0 else 2
                    dec = max(0, min(2, dec))
                body = f"{mu:.{dec}f} \\pm {sd:.{dec}f}"
                cell = f"${body}$"
                if key == best:
                    # the leader is shaded, not bold: \best is defined in the
                    # preamble as \cellcolor{cBest}, one colour for every table
                    cell = f"\\best{{{cell}}}"
                macros.append(f"\\newcommand{{\\{sysname}{METHODS[key][1]}{suffix}}}{{{cell}}}")
                table.append((DISPLAY[system], METHODS[key][1], suffix,
                              float(m.mean()), float(m.std(ddof=1)) if len(m) > 1 else 0.0))

    for key, (mdir, name) in METHODS.items():
        h = train_hours(mdir)
        if h is not None:
            macros.append(f"\\newcommand{{\\{name}TrainHours}}{{{h:.1f}}}")
    macros += [f"\\newcommand{{\\AlloyNDraws}}{{{N_DRAWS}}}",
               f"\\newcommand{{\\AlloyNSeeds}}{{{max(n_seeds.values())}}}"]

    with open(os.path.join(OUT, "numbers.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(macros) + "\n")
    names = {"CErr": "|dc|", "ZErr": "|dlnZ|/D", "ESS": "ESS"}
    print(f"\n  {'system':7s} {'method':7s} {'metric':9s} mean +- sd over seeds")
    for system, method, suffix, mu, sd in table:
        print(f"  {system:7s} {method:7s} {names[suffix]:9s} {mu:.3g} +- {sd:.2g}")
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
