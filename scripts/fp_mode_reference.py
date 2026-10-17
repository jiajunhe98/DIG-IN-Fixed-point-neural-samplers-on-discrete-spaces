"""Reference distribution over the ordered modes, Potts q = 3, L = 24, per cell.

    python scripts/fp_mode_reference.py [--chains 1024] [--burn 100] [--rounds 300]

`fp_reference.py` keeps, at every cell of the 9 x 9 (beta, h) lattice, the
exact counts of N_0 -- the statistic conjugate to the field.  The mode TVD
needs something N_0 does not determine: the distribution of each
configuration's MAJORITY colour, a categorical over the q states.  This
re-samples every cell with the same vendored Swendsen-Wang update, the same
chain/burn/round settings and the same per-cell seeds as that script, and
keeps both

    mode_counts   the majority colour, a tie shared equally between the tied
                  colours (so the counts are multiples of 1/2 or 1/3)
    n0_counts     N_0, as fp_reference.py keeps it

The N_0 counts are the cross-check: they are compared against the existing
reference (runs/amortized_reference/potts_*.json), and the script prints
the largest shift of the mean N_0 per cell, in units of the reference's
standard deviation of N_0.  Ising needs none of
this -- its mode is the sign of M, which the stored counts of M determine.

Writes runs/amortized_reference/potts_modes.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "scripts"))
from fp_reference import POTTS, _seed  # noqa: E402
import gt_sw  # noqa: E402  (fp_reference put its directory on sys.path)

OUT = os.path.join(HERE, "runs", "amortized_reference", "potts_modes.json")


def majority_shares(x: np.ndarray, q: int) -> np.ndarray:
    """[B, q]: 1/k on each of the k colours tied for the majority, else 0."""
    cnt = np.stack([(x == k).sum(1) for k in range(q)], 1)
    top = cnt == cnt.max(1, keepdims=True)
    return top / top.sum(1, keepdims=True)


def run_cell(args):
    beta, h, chains, burn, rounds, base = args
    L, q = POTTS["L"], POTTS["q"]
    N = L * L
    # the seed fp_reference.py used at this (beta, h) rung
    rng = np.random.default_rng(_seed("potts", beta, h, base))
    E = gt_sw.edges(L)
    x = rng.integers(0, q, (chains, N)).astype(np.uint8)
    t0 = time.time()
    for _ in range(burn):
        x = gt_sw.sw_update("potts", x, E, q, beta, h, rng)
    modes = np.zeros(q)
    n0 = np.zeros(N + 1, dtype=np.int64)
    for _ in range(rounds):
        x = gt_sw.sw_update("potts", x, E, q, beta, h, rng)
        modes += majority_shares(x, q).sum(0)
        n0 += np.bincount((x == 0).sum(1), minlength=N + 1)
    return dict(beta=beta, h=h, mode_counts=modes.tolist(), n0_counts=n0.tolist(),
                seconds=time.time() - t0)


def existing_n0():
    """{(beta, h): N_0 counts} from fp_reference.py's output."""
    ref = os.path.join(HERE, "runs", "amortized_reference")
    out = {}
    base = json.load(open(os.path.join(ref, "potts_beta.json")))["cells"]
    for b in POTTS["betas"]:
        f = os.path.join(ref, f"potts_h_b{b:g}.json")
        if not os.path.exists(f):
            continue
        for key, c in json.load(open(f))["cells"].items():
            out[(b, round(float(key), 6))] = np.array(c["counts"], dtype=np.float64)
        out[(b, 0.0)] = np.array(base[f"{b:.6f}"]["counts"], dtype=np.float64)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--chains", type=int, default=1024)
    p.add_argument("--burn", type=int, default=100)
    p.add_argument("--rounds", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    a = p.parse_args()
    jobs = [(b, h, a.chains, a.burn, a.rounds, a.seed)
            for b in POTTS["betas"] for h in POTTS["hs"]]
    with Pool(min(a.workers, len(jobs))) as pool:
        res = pool.map(run_cell, jobs, chunksize=1)

    old = existing_n0()
    worst = 0.0
    for r in res:
        o = old.get((round(r["beta"], 6), round(r["h"], 6)))
        if o is not None:
            # the mean state-0 fraction, in units of the reference's spread of
            # N_0/N: a per-bin TV over 577 bins would mostly measure noise
            n = np.asarray(r["n0_counts"], dtype=np.float64)
            k = np.arange(len(n), dtype=np.float64)
            mu_n, mu_o = (k * n).sum() / n.sum(), (k * o).sum() / o.sum()
            sd_o = np.sqrt(((k - mu_o) ** 2 * o).sum() / o.sum())
            dev = abs(mu_n - mu_o) / sd_o
            r["n0_mean_shift_in_sd"] = float(dev)
            worst = max(worst, dev)
    json.dump(dict(kind="potts_modes", L=POTTS["L"], q=POTTS["q"], args=vars(a),
                   cells=res), open(OUT, "w"))
    print(f"{OUT}: {len(res)} cells, {sum(r['seconds'] for r in res):.0f} s of sampling; "
          f"largest shift of mean N_0 against the existing reference {worst:.3f} sd")


if __name__ == "__main__":
    main()
