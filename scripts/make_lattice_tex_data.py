"""The Ising / Potts phase-sweep table of the paper and its computational-cost table.

    python scripts/make_lattice_tex_data.py

Reads the runs of `experiments/lattice.py` (`runs/lattice/<task>/<phase>/
<method>/s<seed>/`), prints the table, and writes `results/lattice/`:

  numbers.tex               every table cell, as macros
  <sys>_<phase>_config.dat  one configuration per (system, phase), matrix plot
  pm_target.dat             P(m) of the reference chain, Ising 25 ordered
  pm_<method>.dat           P(m) of a model at the same state point

Two things about the metrics are decided here rather than in the .tex, and
both are stated in the appendix:

* Ising has an exact log Z at every size (transfer matrix), so
  |Delta log Z| is measured against ground truth.  Potts 16x16 at q = 3 does
  NOT -- q^cols = 3^16 = 43046721 is far past the transfer-matrix limit and
  enumeration of 3^256 states is out -- so there is no exact value to
  subtract.  For Potts the reference is instead the tightest ELBO lower
  bound over every method and seed, which IS a rigorous lower bound on
  log Z; the errors it gives are therefore lower bounds on the true errors.
  Where the well-behaved methods agree to a millinat (critical, disordered)
  the bound is tight and the numbers mean what they look like.

* ASBS with memory has no importance weights at all.  Its sampler returns
  the path-space Radon-Nikodym derivative between the controlled and the
  reference process (`unbiased_log_Z = False` in its method class); that
  quantity never evaluates the target at the endpoint, so it estimates
  neither log Z nor the weight nu(x)/q(x).  BOTH its |Delta log Z| and its
  ESS cells are therefore empty, rather than filled with numbers that do not
  estimate the quantity the column heads.  Its mode TVD is unaffected: it
  is computed from the raw draws alone.

* The third column is the MODE TVD: each draw is assigned to one of the q
  symmetry-related ordered modes -- the sign of its magnetization for Ising,
  its majority colour for Potts, a tie shared equally between the tied
  colours -- and the resulting categorical distribution over the q modes is
  compared, in total variation, with the target's.  At zero field the target
  is exactly invariant under the spin flip / colour permutation, so its mode
  distribution is exactly uniform, 1/q each; that exact value is the
  reference (a cluster-MC reference would only add its own sampling noise).
  It replaces TVD(m), the histogram distance of the order parameter, which on
  Potts is permutation-invariant and cannot see a colour collapse.
"""

from __future__ import annotations

import glob
import json
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# running a file in scripts/ puts scripts/ on sys.path, not the repo root
sys.path.insert(0, HERE)
OUT = os.path.join(HERE, "results", "lattice")

#: system -> (run roots to search, pretty name, lattice shape)
SYSTEMS = {
    "ising25": (("runs/lattice",), "Ising", (25, 25)),
    "potts16": (("runs/lattice",), "Potts", (16, 16)),
}
PHASES = ("ordered", "critical", "disordered")
#: run directory -> (macro stem, printed name)
METHODS = [
    ("digin", "DSI", "DIG-IN (ours)"),
    ("mdns", "MDNS", "MDNS"),
    ("pdns", "PDNS", "PDNS"),
    ("dasbs", "ASBS", "DASBS"),
    ("dasbs_memoryless", "ASBSM", "DASBS, memoryless"),
    ("leaps", "LEAPS", "LEAPS"),
]
PHASE_STEM = {"ordered": "Ord", "critical": "Crit", "disordered": "Dis"}
#: the rows the paper's table shows, and so the only ones ranked for shading
RANKED = {"DSI", "MDNS", "PDNS", "ASBS", "ASBSM", "LEAPS"}


def finals(system: str, phase: str, method: str) -> list[dict]:
    roots, _name, _shape = SYSTEMS[system]
    out = []
    for root in roots:
        for p in glob.glob(os.path.join(HERE, root, system, phase, method,
                                        "s*", "history.jsonl")):
            rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
            fin = [r for r in rows if r.get("final")]
            if fin:
                out.append(fin[-1])
    return out


def run_dirs(system: str, phase: str, method: str) -> list[str]:
    roots, _n, _s = SYSTEMS[system]
    out = []
    for root in roots:
        out += sorted(glob.glob(os.path.join(HERE, root, system, phase,
                                             method, "s*")))
    return out


def _num(x: float, d: int) -> str:
    """`0.021` -> `.021`: the leading zero costs a column of width in a
    nine-column table and carries no information for a quantity in [0,1]."""
    s = f"{x:.{d}f}"
    return s[1:] if s.startswith("0.") else s


def fmt(vals, kind="plain"):
    r"""`mean{\pm}sd` over seeds, short enough for a nine-column table.

    Nine numeric columns have to fit \textwidth, so every cell is held to
    about nine characters.  `dlnz` is reported in units of 1e-3, stated in
    the column header, because |Delta ln Z| spans four decades here -- from
    under a millinat to ten nats -- and no fixed number of decimals covers
    that.  Everything else is a probability in [0,1] and gets three decimals
    with the leading zero dropped.
    """
    if not vals:
        return r"--"
    mu, sd = float(np.mean(vals)), float(np.std(vals))
    if kind == "hours":
        return rf"${mu:.2f}_{{\pm{sd:.2f}}}$"
    if kind == "secs":
        return rf"${mu:.0f}$"
    if kind == "dlnz":
        # this column is not confined to [0,1] -- it runs to five figures --
        # so it keeps its leading zero
        mu, sd = mu * 1e3, sd * 1e3
        d = 2 if abs(mu) < 10 else 0
        return rf"${mu:.{d}f}_{{\pm{sd:.{d}f}}}$"
    d = 3
    # the spread as a subscript rather than inline: nine numeric columns do
    # not fit 	extwidth with a full-size `mean +- sd` in each, and the
    # subscript keeps the number visible at about two thirds of the width
    return rf"${_num(mu, d)}_{{\pm{_num(sd, d)}}}$"


def fmt_calls(vals):
    """Target-evaluation counts, as one significant figure and a power.

    These are counts of *configurations*, not of scalar energies: a call of
    `energy_diff` returns the whole table of D single-site differences in one
    neighbour-count pass, which is the same order of work as one `energy`
    call on the same configuration, so the two columns are comparable.
    """
    if not vals:
        return r"--"
    mu = float(np.mean(vals))
    if mu <= 0:
        return "$0$"
    e = int(math.floor(math.log10(mu)))
    return rf"${mu / 10 ** e:.1f}{{\cdot}}10^{{{e}}}$"


def compute_macros() -> list[str]:
    """Wall clock and target evaluations, pooled over phases and seeds.

    Pooled rather than split by phase because the step budget is fixed and
    the call counts are a property of the method, not the temperature: they
    are identical across all nine runs of a method to within rounding.
    """
    out = []
    for system, (_roots, _name, shape) in SYSTEMS.items():
        sysstem = "Ising" if system == "ising25" else "Potts"
        # a black-box energy has no shortcut for the single-site difference
        # table: it costs the energy of every one-site change, D (q - 1) + 1
        q = 2 if system == "ising25" else 3
        per_table = int(np.prod(shape)) * (q - 1) + 1
        for meth, stem, _name in METHODS:
            fs = [f for phase in PHASES for f in finals(system, phase, meth)]
            key = f"Cmp{sysstem}{stem}"
            hrs = [f["wall_train_s"] / 3600.0 for f in fs if "wall_train_s" in f]
            drw = [f["sample_time_s"] for f in fs if "sample_time_s" in f]
            ce = [f["target_calls_train_energy"] for f in fs
                  if "target_calls_train_energy" in f]
            cd = [f["target_calls_train_energy_diff"] for f in fs
                  if "target_calls_train_energy_diff" in f]
            out.append(rf"\newcommand{{\{key}Train}}{{{fmt(hrs, 'hours')}}}")
            out.append(rf"\newcommand{{\{key}Draw}}{{{fmt(drw, 'secs')}}}")
            out.append(rf"\newcommand{{\{key}CE}}{{{fmt_calls(ce)}}}")
            out.append(rf"\newcommand{{\{key}CD}}{{{fmt_calls(cd)}}}")
            # total target calls, one per configuration and primitive: the
            # single-site difference table counts as one call (a local energy
            # returns it in one pass); the paper states the black-box cost
            tot = [a + b for a, b in zip(ce, cd)] if len(ce) == len(cd) else []
            out.append(rf"\newcommand{{\{key}Total}}{{{fmt_calls(tot)}}}")
            bb = ([a + b * per_table for a, b in zip(ce, cd)]
                  if len(ce) == len(cd) else [])
            out.append(rf"\newcommand{{\{key}BB}}{{{fmt_calls(bb)}}}")

    # how much more the baselines spend on the target than DSI does, so the
    # appendix can state a factor instead of hand-waving at orders of magnitude
    tot = {}
    for meth, stem, _name in METHODS:
        fs = [f for phase in PHASES for f in finals("ising25", phase, meth)]
        v = [f["target_calls_train_energy"] + f["target_calls_train_energy_diff"]
             for f in fs if "target_calls_train_energy" in f]
        if v:
            tot[stem] = float(np.mean(v))
    for stem in ("ASBS", "ASBSM", "LEAPS"):
        if stem in tot and tot.get("DSI"):
            out.append(rf"\newcommand{{\CmpRatio{stem}}}"
                       rf"{{{tot[stem] / tot['DSI']:.0f}}}")
    return out


def majority_shares(x: torch.Tensor, q: int) -> np.ndarray:
    """[B, q]: 1/k on each of the k colours tied for the majority, else 0.

    For q = 2 this is the sign of the magnetization, with m = 0 shared."""
    cnt = torch.stack([(x == k).sum(1) for k in range(q)], 1)
    top = (cnt == cnt.max(1, keepdim=True).values).double()
    return (top / top.sum(1, keepdim=True)).numpy()


def mode_dists(system: str, phase: str, method: str) -> list[np.ndarray]:
    """Per seed, the distribution of the raw draws over the q ordered modes."""
    out = []
    for d in run_dirs(system, phase, method):
        f = os.path.join(d, "samples.pt")
        if not os.path.exists(f):
            continue
        z = torch.load(f, map_location="cpu", weights_only=False)
        out.append(majority_shares(z["x"], int(z["N"])).mean(0))
    return out


def mode_tvd(p: np.ndarray) -> float:
    """TV between a mode distribution and the exactly uniform one of the target."""
    p = np.asarray(p, dtype=np.float64)
    return float(0.5 * np.abs(p - 1.0 / len(p)).sum())


def potts_reference(phase: str) -> float:
    """Tightest ELBO lower bound over every method and seed.

    A rigorous lower bound on log Z, which is the best available reference
    where no exact value exists.  Errors measured against it are therefore
    lower bounds on the true errors.
    """
    best = -math.inf
    for meth, _s, _n in METHODS:
        for f in finals("potts16", phase, meth):
            if "log_Z_lb" in f and np.isfinite(f["log_Z_lb"]):
                best = max(best, float(f["log_Z_lb"]))
    return best


def main():
    os.makedirs(OUT, exist_ok=True)
    macros: list[str] = []
    seen_seeds, seen_draws = set(), set()
    table: dict[tuple[str, str], dict] = {}

    # (macro suffix, key in the final record, sign of "better", fmt kind)
    COLS = [("Z", None, -1, "dlnz"), ("ESS", "ess", +1, "plain"),
            ("MTV", None, -1, "plain")]

    for system in SYSTEMS:
        sysstem = "Ising" if system == "ising25" else "Potts"
        for phase in PHASES:
            ref = potts_reference(phase) if system == "potts16" else None
            vals: dict[tuple[str, str], list[float]] = {}
            for meth, stem, _name in METHODS:
                fs = finals(system, phase, meth)
                seen_seeds.add(len(fs))
                for f in fs:
                    if "n_samples" in f:
                        seen_draws.add(int(f["n_samples"]))
                if system == "ising25":
                    dz = [abs(f["log_Z_err"]) for f in fs if "log_Z_err" in f]
                else:
                    dz = [abs(f["log_Z"] - ref) for f in fs
                          if "log_Z" in f and ref is not None and np.isfinite(ref)]
                vals[(stem, "Z")] = dz
                for suf, rec, _sgn, _kind in COLS[1:]:
                    if rec is not None:
                        vals[(stem, suf)] = [f[rec] for f in fs if rec in f]
                vals[(stem, "MTV")] = [mode_tvd(p) for p in
                                        mode_dists(system, phase, meth)]
                # An ESS is the effective size of an importance-weighted
                # sample, so it needs weights nu(x)/q(x).  ASBS with memory
                # has none: its sampler returns the path-space RND between
                # the controlled and the reference process, which never
                # evaluates the target at the endpoint and tends to a
                # constant as the control converges, so the ESS computed
                # from it tends to 1 whatever the samples look like.
                # `dds/evaluation.py` writes log_w_logmeanexp instead of
                # log_Z for exactly those runs; key off that rather than off
                # the method name, so any method that reports a path RND is
                # caught the same way.
                if any("log_Z" not in f and "log_w_logmeanexp" in f for f in fs):
                    vals[(stem, "ESS")] = []

            # Mark the best method in each column.  Ranked on the PESSIMISTIC
            # value over seeds -- mean + sd for an error, mean - sd for the
            # ESS -- so that a method is not shaded on the strength of a good
            # mean it only reaches in some seeds.  Every method whose
            # pessimistic value lies within one of the leader's standard
            # deviations of the leader's is shaded with it.  (The earlier rule,
            # "within sd_i + sd_lead of the leading mean", let a high-variance
            # method tie through its own spread.)  Only the rows the paper's
            # table shows are ranked.
            table[(system, phase)] = vals
            for suf, _rec, sgn, kind in COLS:
                have = {stem: vals[(stem, suf)] for _m, stem, _n in METHODS
                        if vals.get((stem, suf)) and stem in RANKED}
                best: set[str] = set()
                if have:
                    mu = {st: float(np.mean(v)) for st, v in have.items()}
                    sd = {st: float(np.std(v)) for st, v in have.items()}
                    pess = {st: mu[st] - sgn * sd[st] for st in mu}
                    lead = (max if sgn > 0 else min)(pess, key=pess.get)
                    best = {st for st in pess
                            if abs(pess[st] - pess[lead]) <= sd[lead]}
                for _m, stem, _n in METHODS:
                    cell = fmt(vals.get((stem, suf)), kind)
                    if stem in best and cell != "--":
                        # the paper marks the leader by shading the cell, not
                        # by bold: \best is defined in the preamble
                        # (\cellcolor{cBest}), so the colour is set in one place
                        cell = r"\best{" + cell + "}"
                    key = f"{sysstem}{PHASE_STEM[phase]}{stem}{suf}"
                    macros.append(rf"\newcommand{{\{key}}}{{{cell}}}")

    # what an exact sampler shows at the same draw count: the mode TVD of
    # NDraws multinomial draws from the uniform mode distribution
    for system, (_r, _n, _shape) in SYSTEMS.items():
        sysstem = "Ising" if system == "ising25" else "Potts"
        q = 2 if system == "ising25" else 3
        n = max(seen_draws) if seen_draws else 8192
        rng = np.random.default_rng(0)
        fl = np.mean([mode_tvd(rng.multinomial(n, np.full(q, 1.0 / q)) / n)
                      for _ in range(4000)])
        macros.append(rf"\newcommand{{\{sysstem}MTVFloor}}{{${_num(fl, 3)}$}}")

    macros.append(rf"\newcommand{{\NSeeds}}{{{max(seen_seeds) if seen_seeds else 3}}}")
    macros.append(rf"\newcommand{{\NDraws}}{{{max(seen_draws) if seen_draws else 8192}}}")
    macros += compute_macros()
    for phase in PHASES:
        macros.append(rf"\newcommand{{\PottsRef{PHASE_STEM[phase]}}}"
                      rf"{{{potts_reference(phase):.3f}}}")

    # ---- one configuration per (system, phase), for the appendix figure ----
    for system, (_roots, _name, shape) in SYSTEMS.items():
        for phase in PHASES:
            ds = run_dirs(system, phase, "digin")
            if not ds:
                continue
            f = os.path.join(ds[0], "samples.pt")
            if not os.path.exists(f):
                continue
            d = torch.load(f, map_location="cpu", weights_only=False)
            x = d["x"]
            # the draw whose order parameter is closest to the batch median:
            # typical in exactly the observable that defines the phases, which
            # is what the figure is about.  (Taking the median log weight
            # instead picks a nearly identical configuration here -- the two
            # criteria agree to within 0.02 in m at every state point.)
            q = int(d["N"])
            frac = torch.stack([(x == k).double().mean(1) for k in range(q)], 1)
            m = (q * frac.max(1).values - 1.0) / (q - 1.0)
            idx = int((m - m.median()).abs().argmin())
            cfg = x[idx].reshape(shape).numpy()
            p = os.path.join(OUT, f"{system}_{phase}_config.dat")
            with open(p, "w", encoding="utf-8") as fh:
                fh.write("x y v\n")
                for i in range(shape[0]):
                    for j in range(shape[1]):
                        fh.write(f"{j} {i} {int(cfg[i, j])}\n")
                    fh.write("\n")

    # ---- P(m): the target, and a model that collapsed ----
    _write_pm()

    with open(os.path.join(OUT, "numbers.tex"), "w", encoding="utf-8") as fh:
        fh.write("% generated by scripts/make_lattice_tex_data.py -- do not edit\n")
        fh.write("\n".join(macros) + "\n")
    print_table(table)
    print(f"{OUT}: numbers.tex with {len(macros)} macros, "
          f"{len(glob.glob(os.path.join(OUT, '*.dat')))} data files")


def print_table(table: dict) -> None:
    """The paper's lattice table as plain text: mean +- sd over seeds."""
    def cell(v, scale=1.0):
        if not v:
            return "--"
        return f"{np.mean(v) * scale:.3g} +- {np.std(v) * scale:.2g}"

    for (system, phase), vals in table.items():
        print(f"\n{SYSTEMS[system][1]} {phase}"
              + (" (|dlnZ| against the tightest ELBO)" if system == "potts16" else ""))
        print(f"  {'method':20s} {'|dlnZ| x1e3':>18s} {'ESS':>18s} {'mode TVD':>18s}")
        for _m, stem, name in METHODS:
            print(f"  {name:20s} {cell(vals.get((stem, 'Z')), 1e3):>18s} "
                  f"{cell(vals.get((stem, 'ESS'))):>18s} "
                  f"{cell(vals.get((stem, 'MTV'))):>18s}")


def _mag_hist(x: torch.Tensor, n_states: int, bins: np.ndarray) -> np.ndarray:
    """Normalised histogram of the per-configuration magnetisation."""
    if n_states == 2:
        m = (2.0 * x.double() - 1.0).mean(dim=1).numpy()     # signed mean spin
    else:
        # the same order parameter the target defines, (q f_max - 1)/(q - 1);
        # note it is permutation-invariant, so it cannot resolve which colour
        # a configuration ordered into (see app:lattice-metrics)
        f = np.stack([(x == k).double().mean(dim=1).numpy()
                      for k in range(n_states)], axis=1).max(axis=1)
        m = (n_states * f - 1.0) / (n_states - 1.0)
    h, _ = np.histogram(m, bins=bins, density=True)
    return h


def _write_pm():
    """Ising 25 ordered: the reference P(m) and two models' P(m).

    The ordered phase has two symmetry-related modes at +-m*.  A sampler that
    finds only one of them still reports a high ESS -- its weights are
    self-consistent within the mode it did find -- so ESS cannot detect this
    and the magnetisation histogram is what does.
    """
    bins = np.linspace(-1, 1, 81)
    centres = 0.5 * (bins[:-1] + bins[1:])
    rows = {}

    # the reference chain costs minutes and never changes; the tables around
    # it get regenerated often, so reuse it once it is on disk
    if not os.path.exists(os.path.join(OUT, "pm_target.dat")):
        from dds.targets import build_target, get_mcmc
        # the benchmark's ordered target: configs/target/ising25_ordered.yaml
        t = build_target({"name": "ising", "lattice_shape": [25, 25],
                          "beta": 0.6})
        mc = get_mcmc(t, device="cpu")
        ref = mc.sample(8192, n_chains=512, burn_in=2000, thin=10)
        rows["target"] = _mag_hist(ref.cpu(), 2, bins)

    for meth in ("digin", "mdns"):
        ds = run_dirs("ising25", "ordered", meth)
        if not ds:
            continue
        f = os.path.join(ds[0], "samples.pt")
        if not os.path.exists(f):
            continue
        d = torch.load(f, map_location="cpu", weights_only=False)
        rows[meth] = _mag_hist(d["x"], int(d["N"]), bins)

    for name, h in rows.items():
        with open(os.path.join(OUT, f"pm_{name}.dat"), "w", encoding="utf-8") as fh:
            fh.write("m p\n")
            for c, v in zip(centres, h):
                fh.write(f"{c:.4f} {v:.6f}\n")


if __name__ == "__main__":
    main()
