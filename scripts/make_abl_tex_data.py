"""The component ablation of the paper's appendix (25x25 Ising).

    python scripts/make_abl_tex_data.py

Reads the ablation runs (`runs/ablation/`, from `experiments/lattice.py
--ablation`) together with the phase sweep's own `digin`, `mdns` and `pdns`
runs (`runs/lattice/`), prints the table, and writes into `results/ablation/`:

  abl_numbers.tex            every table cell, as macros
  abl_<arm>_<phase>.dat      the training curve of each arm: step, ESS,
                             |Delta ln Z| and the mode TVD, averaged over seeds

The baseline is the sweep's own runs, not a re-run, so the ablation is
measured against exactly the row that \\cref{tab:ising} reports.  MDNS and
PDNS are read too: they are the two hard-label arms without damping, so each
pairs with an ablation arm that differs from it only by eta.

Curves are means over seeds at each evaluation step.  A collapsed seed and a
correct one average to something that is neither, so the table carries the
per-seed magnetization and the text reads collapse off that, not off a curve.
"""

from __future__ import annotations

import glob
import json
import math
import os
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "results", "ablation")
PHASES = ("ordered", "critical", "disordered")
PHASE_STEM = {"ordered": "Ord", "critical": "Crit", "disordered": "Dis"}

#: (run directory, run root, macro stem, printed name)
ARMS = [
    ("digin", "runs/lattice", "Base", "DIG-IN (baseline)"),
    ("no_damping", "runs/ablation", "NoDamp", "no damping"),
    ("no_gibbs_tempered", "runs/ablation", "NoGibbsT", "no Gibbs score"),
    ("no_gibbs_plain", "runs/ablation", "NoGibbsP", "no Gibbs score, no smoothing"),
    ("no_iw", "runs/ablation", "NoIW", "no importance weights"),
    ("pdns", "runs/lattice", "PDNS", "PDNS"),
    ("mdns", "runs/lattice", "MDNS", "MDNS"),
]
METRICS = [("ess", "ESS"), ("log_Z_err", "Z"), ("mode_tv", "MTV")]


def mode_tv(r: dict):
    """Mode TVD of one evaluation row: TV between the raw draws' distribution
    over the two ordered modes (sign of m) and the target's, which is exactly
    (1/2, 1/2) at zero field.  For q = 2 that is |P(m > 0) - 1/2|."""
    f = r.get("mag_pos_frac")
    return None if f is None else abs(float(f) - 0.5)


def rows_of(root: str, phase: str, arm: str) -> list[list[dict]]:
    """Every seed's history at one (phase, arm), oldest row first."""
    out = []
    for p in sorted(glob.glob(os.path.join(HERE, root, "ising25", phase, arm,
                                           "s*", "history.jsonl"))):
        rows = {}
        with open(p, encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                rows[int(r["step"])] = r          # a resumed run repeats steps
        if rows:
            out.append([rows[k] for k in sorted(rows)])
    return out


def pm(vals, d, scale=1.0, lead=True):
    """`mean_{\\pm sd}` over seeds, the style of the phase-sweep tables."""
    v = np.asarray([x for x in vals if x is not None], dtype=np.float64) * scale
    if not len(v):
        return "--"
    mu, sd = v.mean(), v.std()
    s_mu, s_sd = f"{mu:.{d}f}", f"{sd:.{d}f}"
    if not lead:
        s_mu = s_mu[1:] if s_mu.startswith("0.") else s_mu
        s_sd = s_sd[1:] if s_sd.startswith("0.") else s_sd
    return rf"${s_mu}_{{\pm{s_sd}}}$"


def main():
    os.makedirs(OUT, exist_ok=True)
    macros, seeds_seen, steps_seen, table = [], set(), set(), []
    # per-seed magnetization at the end, which is what says "collapsed"
    collapsed = defaultdict(list)

    for arm, root, stem, _name in ARMS:
        for phase in PHASES:
            hs = rows_of(root, phase, arm)
            if not hs:
                print(f"  no runs: {arm} {phase}")
                continue
            seeds_seen.add(len(hs))
            last = [h[-1] for h in hs]
            table.append((_name, phase, [r.get("log_Z_err") for r in last],
                          [r.get("ess") for r in last], [mode_tv(r) for r in last]))
            steps_seen.add(max(r.get("step", 0) for r in last))
            key = f"Abl{stem}{PHASE_STEM[phase]}"
            macros += [
                rf"\newcommand{{\{key}Z}}{{{pm([r.get('log_Z_err') for r in last], 1, 1e3)}}}",
                rf"\newcommand{{\{key}ESS}}{{{pm([r.get('ess') for r in last], 3, lead=False)}}}",
                rf"\newcommand{{\{key}MTV}}{{{pm([mode_tv(r) for r in last], 3, lead=False)}}}",
            ]
            # <|m|> against the reference's: <m> ~ 0 is ambiguous between two
            # modes covered equally and draws that never ordered at all
            macros.append(rf"\newcommand{{\{key}Mabs}}"
                          rf"{{{pm([r.get('mag_abs_mean') for r in last], 2, lead=False)}}}")
            ref = [r.get("mag_abs_ref") for r in last if r.get("mag_abs_ref") is not None]
            if ref:
                macros.append(rf"\newcommand{{\{key}MabsRef}}{{{np.mean(ref):.2f}}}")
            mags = [r.get("mag_mean") for r in last if r.get("mag_mean") is not None]
            macros.append(rf"\newcommand{{\{key}M}}{{{'/'.join(f'{m:+.2f}' for m in mags)}}}")
            if phase == "ordered":
                # A seed sitting at |<m>| ~ 0.97 has one mode and not the other;
                # the target is symmetric, so this is a lost mode and it costs
                # ln 2 in ln Z.  A milder imbalance is NOT this: the no-IW arm
                # reaches |<m>| = 0.6 with both modes present and |Delta ln Z|
                # of 0.016, which no threshold on <m> alone separates.
                collapsed[stem] = [m for m in mags if abs(m) > 0.9]

            # the curve: mean over seeds at every step all seeds reached
            common = sorted(set.intersection(*[{r["step"] for r in h} for h in hs]))
            by = [{r["step"]: r for r in h} for h in hs]
            with open(os.path.join(OUT, f"abl_{stem}_{phase}.dat"), "w",
                      encoding="utf-8") as f:
                f.write("step ess z tv\n")
                for s in common:
                    vals = {}
                    for mk, _ in METRICS:
                        get = mode_tv if mk == "mode_tv" else (lambda r, k=mk: r.get(k))
                        v = [get(b[s]) for b in by if get(b[s]) is not None]
                        vals[mk] = float(np.mean(v)) if v else float("nan")
                    z, tv = vals["log_Z_err"], vals["mode_tv"]
                    f.write(f"{s} {vals['ess']:.6f} "
                            f"{'nan' if not np.isfinite(z) or z <= 0 else f'{z:.6g}'} "
                            f"{'nan' if not np.isfinite(tv) or tv <= 0 else f'{tv:.6g}'}\n")

    for stem, mags in collapsed.items():
        macros.append(rf"\newcommand{{\Abl{stem}SingleMode}}{{{len(mags)}}}")
    macros.append(rf"\newcommand{{\AblNSeeds}}{{{max(seeds_seen)}}}")
    macros.append(rf"\newcommand{{\AblSteps}}{{{max(steps_seen) // 1000}k}}")
    with open(os.path.join(OUT, "abl_numbers.tex"), "w", encoding="utf-8") as f:
        f.write("% generated by scripts/make_abl_tex_data.py -- do not edit\n")
        f.write("\n".join(macros) + "\n")
    def cell(v, scale=1.0):
        v = [x for x in v if x is not None]
        return (f"{np.mean(v) * scale:.3g} +- {np.std(v) * scale:.2g}"
                if v else "--")

    print(f"  {'arm':30s} {'phase':11s} {'|dlnZ| x1e3':>18s} {'ESS':>18s} "
          f"{'mode TVD':>18s}")
    for name, phase, z, ess, tv in table:
        print(f"  {name:30s} {phase:11s} {cell(z, 1e3):>18s} {cell(ess):>18s} "
              f"{cell(tv):>18s}")
    print(f"{OUT}/abl_numbers.tex: {len(macros)} macros, "
          f"{len(glob.glob(os.path.join(OUT, 'abl_*.dat')))} curves")


if __name__ == "__main__":
    main()
