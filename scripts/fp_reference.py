"""Reference ln Z and order-parameter distributions over the amortisation grid.

    python scripts/fp_reference.py test                      # vs enumeration
    python scripts/fp_reference.py run TASK --out DIR [--workers N]

The fixed-point runs were scored on a 9 x 9 lattice over their (beta, h) box
(`score_fixed_point.py`).  Two of the three metrics need a reference at every
one of those cells, and none exists: the archive's Swendsen-Wang ground truth
covers only each run's 15 evaluation cells, and carries no ln Z at all.  This
builds one.

ln Z
----
* **Ising, L = 50.**  Kaufman's formula gives ln Z(beta, 0) exactly on the
  finite torus (`ising_theory_stable.py`, checked against enumeration).  The
  field is added by BAR along an h-ladder at each beta,
      ln Z(beta, h) = ln Z_Kaufman(beta) + sum_k  Delta_BAR(h_k -> h_{k+1}),
  sampled with Swendsen-Wang at each rung.  Spin-flip symmetry makes
  ln Z(beta, -h) = ln Z(beta, h) exactly, so only h >= 0 is sampled.
* **Potts, q = 3, L = 24.**  Nothing is exact at this size.  The one exact
  anchor is beta = 0, where Z = q^N for every h.  BAR along a beta-ladder at
  h = 0 gives ln Z(beta, 0) at every lattice beta, and BAR along an h-ladder
  at each of those gives the rest.  There is no symmetry in h here: the field
  couples to state 0 only.

Every rung stores only the sufficient statistics the reduced potentials need
-- (bond sum, spin sum) for Ising, (agreeing bonds, N_0) for Potts -- so the
BAR differences between any two rungs are exact functions of what is kept.

Order parameter
---------------
At every rung that is a lattice cell, the full distribution of the
field-conjugate order parameter is kept as exact integer counts: the spin sum
M for Ising, N_0 for Potts.  Any histogram binning, TV, or mean composition is
then computed later without re-running anything.

Errors
------
Samples within a chain are correlated, so the standard error of every ln Z is
a delete-one-group jackknife over 16 groups of independent chains, carried
through the whole ladder.  That accounts for autocorrelation without having to
estimate it.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "fixed_point", "gt_samples"))
sys.path.insert(0, os.path.join(HERE, "fixed_point", "ising_beta_h"))
import gt_sw  # noqa: E402  exact-tested Swendsen-Wang, vendored byte-identical

N_GROUPS = 16


# ---------------------------------------------------------------------------
# one rung
# ---------------------------------------------------------------------------
def stats(model: str, x: np.ndarray, L: int):
    """Sufficient statistics per configuration, as int64: (coupling term, field term).

    Ising: (sum_<ij> s_i s_j, sum_i s_i).   Potts: (A, N_0).
    log pi~ = beta * (first + h * second) in both, each bond counted once.
    """
    g = x.reshape(-1, L, L)
    if model == "ising":
        s = 2 * g.astype(np.int64) - 1
        bond = (s * np.roll(s, 1, 1)).sum((1, 2)) + (s * np.roll(s, 1, 2)).sum((1, 2))
        return bond, s.sum((1, 2))
    a = (g == np.roll(g, 1, 1)).sum((1, 2)) + (g == np.roll(g, 1, 2)).sum((1, 2))
    return a.astype(np.int64), (g == 0).sum((1, 2)).astype(np.int64)


def run_rung(args):
    """Sample one (beta, h) with Swendsen-Wang; keep per-sample statistics only.

    Returns arrays of shape [n_rounds, n_chains] so the chain index -- the
    unit of independence the jackknife needs -- survives.
    """
    model, L, q, beta, h, n_chains, burn, n_rounds, seed = args
    rng = np.random.default_rng(seed)
    t0 = time.time()
    if beta == 0.0:
        # exact: at beta = 0 the target is uniform for every h
        X = rng.integers(0, q, (n_rounds * n_chains, L * L)).astype(np.uint8)
        c, f = stats(model, X, L)
        c, f = c.reshape(n_rounds, n_chains), f.reshape(n_rounds, n_chains)
    else:
        E = gt_sw.edges(L)
        x = rng.integers(0, q, (n_chains, L * L)).astype(np.uint8)
        for _ in range(burn):
            x = gt_sw.sw_update(model, x, E, q, beta, h, rng)
        cs, fs = [], []
        for _ in range(n_rounds):
            x = gt_sw.sw_update(model, x, E, q, beta, h, rng)
            c1, f1 = stats(model, x, L)
            cs.append(c1); fs.append(f1)
        c, f = np.stack(cs), np.stack(fs)
    # int32 halves what a 105-rung ladder holds in memory; every value fits
    # (|bond| <= 2N = 5000 at L = 50, N_0 <= 576 at L = 24)
    return dict(model=model, L=L, q=q, beta=float(beta), h=float(h),
                coup=c.astype(np.int32), field=f.astype(np.int32),
                seconds=time.time() - t0)


# ---------------------------------------------------------------------------
# BAR along a ladder
# ---------------------------------------------------------------------------
def _bar(wf: np.ndarray, wr: np.ndarray) -> float:
    """Bennett's estimate of ln(Z_1/Z_0) from forward and reverse works.

    wf = log pi~_1(x) - log pi~_0(x) for x ~ 0;  wr = log pi~_0(x) - log pi~_1(x)
    for x ~ 1.  Solves the self-consistent equation by bisection on the
    monotone residual, which cannot diverge the way a Newton step can when
    the overlap is poor.
    """
    nf, nr = len(wf), len(wr)
    m = math.log(nf / nr)

    def resid(d):     # d = ln(Z_1/Z_0)
        a = -np.logaddexp(0.0, -(wf - d + m))          # log sigma(wf - d + m)
        b = -np.logaddexp(0.0, -(wr + d - m))          # log sigma(wr + d - m)
        return np.logaddexp.reduce(a) - np.logaddexp.reduce(b)

    # the one-sided (exponential-averaging) estimates bracket the root
    lo = min(np.logaddexp.reduce(wf) - math.log(nf),
             -(np.logaddexp.reduce(wr) - math.log(nr))) - 1.0
    hi = max(np.logaddexp.reduce(wf) - math.log(nf),
             -(np.logaddexp.reduce(wr) - math.log(nr))) + 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if resid(mid) > 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-10:
            break
    return 0.5 * (lo + hi)


def _logpi(r, beta, h):
    return beta * (r["coup"].astype(np.float64) + h * r["field"].astype(np.float64))


def _pair(args):
    """BAR for one consecutive pair: the full estimate and its 16 jackknife
    replicates.  One pool task per pair keeps what is pickled to a worker to
    the two work arrays, rather than the whole ladder."""
    la, lb = args
    grp = np.arange(la.shape[1]) % N_GROUPS
    full = _bar(la.ravel(), lb.ravel())
    jk = [_bar(la[:, grp != g].ravel(), lb[:, grp != g].ravel())
          for g in range(N_GROUPS)]
    return full, jk


def ladder_with_errors(rungs: list[dict], workers: int = 1):
    """Cumulative ln(Z_k / Z_0) along an ordered ladder, with jackknife errors.

    Replicate g deletes chain group g from *every* rung at once, so a
    replicate of the cumulative sum is the sum of that replicate's increments.
    """
    pairs = []
    for k in range(len(rungs) - 1):
        a, b = rungs[k], rungs[k + 1]
        la = _logpi(a, b["beta"], b["h"]) - _logpi(a, a["beta"], a["h"])
        lb = _logpi(b, a["beta"], a["h"]) - _logpi(b, b["beta"], b["h"])
        pairs.append((la, lb))
    if workers > 1 and len(pairs) > 1:
        with Pool(min(workers, len(pairs))) as pool:
            res = pool.map(_pair, pairs, chunksize=1)
    else:
        res = [_pair(p) for p in pairs]
    inc = np.array([r[0] for r in res])
    jinc = np.array([r[1] for r in res])                  # [pairs, groups]
    est = np.concatenate([[0.0], np.cumsum(inc)])
    jk = np.concatenate([np.zeros((1, N_GROUPS)), np.cumsum(jinc, 0)])
    se = np.sqrt((N_GROUPS - 1) / N_GROUPS
                 * ((jk - jk.mean(1, keepdims=True)) ** 2).sum(1))
    return est, se


# ---------------------------------------------------------------------------
# tasks
# ---------------------------------------------------------------------------
ISING = dict(model="ising", L=50, q=2,
             betas=[round(float(b), 6) for b in np.linspace(0.39, 0.65, 9)],
             hs=[0.0, 0.025, 0.05, 0.075, 0.1])
POTTS = dict(model="potts", L=24, q=3,
             betas=[round(float(b), 6) for b in np.linspace(0.8, 1.3, 9)],
             hs=[round(float(h), 6) for h in np.linspace(-0.01, 0.01, 9)])


def h_ladder_ising():
    """h = 0, a geometric run from 2e-4 to 0.1, and every lattice h exactly.

    Fine near h = 0, where the order parameter fluctuates most and the rung
    spacing is set by overlap; coarse further out, where the field has pinned
    it and consecutive rungs overlap well even far apart.
    """
    geo = 2e-4 * 1.3 ** np.arange(0, 40)
    geo = geo[geo < 0.1]
    return sorted({0.0, *[round(float(v), 7) for v in geo], *ISING["hs"]})


def h_ladder_potts():
    return [round(float(v), 7) for v in np.arange(-0.01, 0.0100001, 0.00125)]


def beta_ladder(beta_max, step=0.0125):
    k = int(round(beta_max / step))
    return [round(i * step, 7) for i in range(k + 1)]


def _cells_counts(r, N):
    """Exact integer counts of the field-conjugate statistic, over all samples."""
    f = r["field"].ravel()
    if r["model"] == "ising":
        return np.bincount((f + N) // 2, minlength=N + 1)     # M in -N..N, step 2
    return np.bincount(f, minlength=N + 1)                    # N_0 in 0..N


def _pool_map(jobs, workers):
    with Pool(min(workers, len(jobs))) as pool:
        res = pool.map(run_rung, jobs, chunksize=1)
    return res


def task_ising_h(beta, a):
    """One beta of the Ising h-ladder, anchored on Kaufman."""
    import ising_theory_stable as th
    L, N = ISING["L"], ISING["L"] ** 2
    hs = h_ladder_ising()
    jobs = [("ising", L, 2, beta, h, a.chains, a.burn, a.rounds,
             _seed("ising", beta, h, a.seed)) for h in hs]
    rungs = _pool_map(jobs, a.workers)
    rungs.sort(key=lambda r: r["h"])
    est, se = ladder_with_errors(rungs, a.workers)
    lnz0 = float(th.calculate_ising_partition_function(beta, L)[0])
    # free check on burn-in: the h = 0 rung's energy against Kaufman's exact one
    import energy_mdns as em
    b0 = rungs[0]
    e_samp = -b0["coup"].astype(np.float64).mean() / N
    e_exact = em.bare_onsager_energy_per_spin(beta, L)
    cells = {}
    for k, r in enumerate(rungs):
        if any(abs(r["h"] - h) < 1e-9 for h in ISING["hs"]):
            cells[f"{r['h']:+.4f}"] = dict(lnZ=lnz0 + float(est[k]), se=float(se[k]),
                                           counts=_cells_counts(r, N).tolist())
    return dict(kind="ising_h", beta=beta, L=L, lnZ0_kaufman=lnz0,
                energy_h0_sampled=e_samp, energy_h0_exact=e_exact,
                ladder=[r["h"] for r in rungs], cells=cells,
                seconds=sum(r["seconds"] for r in rungs))


def task_potts_beta(a, model="potts", L=24, q=3, beta_max=1.3, targets=None):
    """The h = 0 beta-ladder from beta = 0, where ln Z = N ln q exactly."""
    N = L * L
    bs = beta_ladder(beta_max)
    jobs = [(model, L, q, b, 0.0, a.chains, a.burn, a.rounds,
             _seed(model, b, 0.0, a.seed)) for b in bs]
    rungs = _pool_map(jobs, a.workers)
    rungs.sort(key=lambda r: r["beta"])
    est, se = ladder_with_errors(rungs, a.workers)
    lnz = N * math.log(q) + est
    targets = targets if targets is not None else POTTS["betas"]
    out = {}
    for k, r in enumerate(rungs):
        if any(abs(r["beta"] - b) < 1e-9 for b in targets):
            out[f"{r['beta']:.6f}"] = dict(lnZ=float(lnz[k]), se=float(se[k]),
                                           counts=_cells_counts(r, N).tolist())
    return dict(kind=f"{model}_beta", L=L, q=q, ladder=bs, cells=out,
                lnZ_all=[float(v) for v in lnz], se_all=[float(v) for v in se],
                seconds=sum(r["seconds"] for r in rungs))


def task_potts_h(beta, a):
    """One beta of the Potts h-ladder; anchored later on the beta-ladder's h = 0."""
    L, N = POTTS["L"], POTTS["L"] ** 2
    hs = h_ladder_potts()
    jobs = [("potts", L, 3, beta, h, a.chains, a.burn, a.rounds,
             _seed("potts", beta, h, a.seed)) for h in hs]
    rungs = _pool_map(jobs, a.workers)
    rungs.sort(key=lambda r: r["h"])
    k0 = [k for k, r in enumerate(rungs) if abs(r["h"]) < 1e-12][0]
    # the anchor (h = 0) sits mid-ladder, so walk outward from it in both
    # directions; each half then carries its own jackknife error from zero
    up, se_up = ladder_with_errors(rungs[k0:], a.workers)
    dn, se_dn = ladder_with_errors(rungs[k0::-1], a.workers)
    cells = {}
    for k, r in enumerate(rungs):
        if any(abs(r["h"] - h) < 1e-9 for h in POTTS["hs"]):
            j = k - k0
            d, s = (up[j], se_up[j]) if j >= 0 else (dn[-j], se_dn[-j])
            cells[f"{r['h']:+.4f}"] = dict(dlnZ=float(d), se=float(s),
                                           counts=_cells_counts(r, N).tolist())
    return dict(kind="potts_h", beta=beta, L=L, ladder=hs, cells=cells,
                seconds=sum(r["seconds"] for r in rungs))


def _cores() -> int:
    """Cores this process may run on.  os.cpu_count() reports the host's, which
    inside a container limited by affinity oversubscribes the node."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:                                  # not Linux
        return os.cpu_count() or 1


def _seed(model, beta, h, base):
    import zlib
    return (base * 1_000_003 + zlib.crc32(f"{model}{beta:.7f}{h:+.7f}".encode())) % (2**31 - 1)


# ---------------------------------------------------------------------------
# exactness test on lattices small enough to enumerate
# ---------------------------------------------------------------------------
def test():
    import itertools

    class A:
        chains, burn, rounds, seed = 1024, 50, 60, 3
        workers = max(1, (os.cpu_count() or 2) - 2)

    def exact(model, L, q, beta, h):
        allx = np.array(list(itertools.product(range(q), repeat=L * L)), dtype=np.uint8)
        c, f = stats(model, allx, L)
        lp = beta * (c + h * f)
        return float(np.logaddexp.reduce(lp))

    worst = 0.0
    # Ising h-ladder on a 4x4 torus, anchored on enumeration at h = 0
    L = 4
    for beta in (0.35, 0.6):
        hs = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3]
        rungs = [run_rung(("ising", L, 2, beta, h, A.chains, A.burn, A.rounds, 7 + i))
                 for i, h in enumerate(hs)]
        est, se = ladder_with_errors(rungs)
        base = exact("ising", L, 2, beta, 0.0)
        for k, h in enumerate(hs[1:], 1):
            ex = exact("ising", L, 2, beta, h)
            z = (base + est[k] - ex) / max(se[k], 1e-12)
            worst = max(worst, abs(z))
            print(f"ising 4x4 beta={beta} h={h:.2f}: {base+est[k]:.5f} vs exact {ex:.5f} "
                  f"({z:+.1f} se, se={se[k]:.5f})")
    # Potts beta-ladder from beta = 0 on 3x3, then an h-step
    L, q = 3, 3
    bs = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
    rungs = [run_rung(("potts", L, q, b, 0.0, A.chains, A.burn, A.rounds, 20 + i))
             for i, b in enumerate(bs)]
    est, se = ladder_with_errors(rungs)
    for k, b in enumerate(bs[1:], 1):
        ex = exact("potts", L, q, b, 0.0)
        got = L * L * math.log(q) + est[k]
        z = (got - ex) / max(se[k], 1e-12)
        worst = max(worst, abs(z))
        print(f"potts 3x3 beta={b:.2f} h=0: {got:.5f} vs exact {ex:.5f} ({z:+.1f} se, se={se[k]:.5f})")
    hs = [-0.3, -0.15, 0.0, 0.15, 0.3]
    rungs = [run_rung(("potts", L, q, 1.2, h, A.chains, A.burn, A.rounds, 40 + i))
             for i, h in enumerate(hs)]
    est, se = ladder_with_errors(rungs)
    for k, h in enumerate(hs):
        ex = exact("potts", L, q, 1.2, h) - exact("potts", L, q, 1.2, -0.3)
        z = (est[k] - ex) / max(se[k], 1e-12) if k else 0.0
        worst = max(worst, abs(z))
        print(f"potts 3x3 beta=1.2 h={h:+.2f} rel. to h=-0.3: {est[k]:.5f} vs exact {ex:.5f} ({z:+.1f} se)")
    print("LADDER TEST", "OK" if worst < 4.0 else "FAIL", f"(worst {worst:.1f} se)")
    return worst < 4.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=("test", "run"))
    p.add_argument("task", nargs="?", default="",
                   help="ising_h:BETA | potts_beta | potts_h:BETA | ising24_beta")
    p.add_argument("--out", default="")
    p.add_argument("--chains", type=int, default=1024)
    p.add_argument("--burn", type=int, default=200)
    p.add_argument("--rounds", type=int, default=150)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=_cores())
    a = p.parse_args()
    if a.cmd == "test":
        sys.exit(0 if test() else 1)

    os.makedirs(a.out, exist_ok=True)
    kind, _, arg = a.task.partition(":")
    name = a.task.replace(":", "_b")
    f = os.path.join(a.out, name + ".json")
    if os.path.exists(f):
        print(f"{f} exists, skip", flush=True)
        return
    t0 = time.time()
    print(f"{a.task}: chains={a.chains} burn={a.burn} rounds={a.rounds} "
          f"workers={a.workers} cpus={_cores()}/{os.cpu_count()}", flush=True)
    if kind == "ising_h":
        res = task_ising_h(float(arg), a)
    elif kind == "potts_beta":
        res = task_potts_beta(a)
    elif kind == "potts_h":
        res = task_potts_h(float(arg), a)
    elif kind == "ising24_beta":
        # validation of the beta-ladder at the Potts lattice's N, against Kaufman
        import ising_theory_stable as th
        res = task_potts_beta(a, model="ising", L=24, q=2, beta_max=0.65,
                              targets=beta_ladder(0.65))
        for k, v in res["cells"].items():
            # Kaufman's prefactor carries ln sinh(2 beta), singular at beta = 0,
            # where the answer is simply N ln 2
            b = float(k)
            v["kaufman"] = (24 * 24 * math.log(2.0) if b == 0.0 else
                            float(th.calculate_ising_partition_function(b, 24)[0]))
    else:
        raise SystemExit(f"unknown task {a.task}")
    res["args"] = dict(chains=a.chains, burn=a.burn, rounds=a.rounds, seed=a.seed)
    tmp = f + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(res, fh)
    os.replace(tmp, f)
    print(f"done in {time.time() - t0:.0f}s wall, {res['seconds']:.0f}s summed over rungs",
          flush=True)


if __name__ == "__main__":
    main()
