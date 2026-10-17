"""Ground-truth samples for the fixed-point sampler benchmarks: Swendsen-Wang with the field carried
by the cluster colours.  CPU only (numpy + scipy connected components).

Targets, MDNS conventions, L x L torus, sites row-major:
  Ising  log pi~(s) = beta * ( sum_<ij> s_i s_j + h * sum_i s_i ),  tokens x in {0,1}, s = 2x - 1
  Potts  log pi~(x) = beta * ( A(x) + h * N_0(x) ),  A = agreeing bonds, N_0 = sites in state 0
Each bond is counted once.

Update (Edwards-Sokal).  Given x, occupy every agreeing nearest-neighbour bond independently with
probability p = 1 - exp(-J), J = 2 beta (Ising) or beta (Potts).  Given the bonds the colours of the
connected clusters are independent, and cluster c of size |c| takes colour a with probability
    Ising:  proportional to exp(beta h |c| (2a - 1))
    Potts:  proportional to exp(beta h |c| 1[a = 0])
This is an exact Gibbs sweep of the joint (x, bonds) whose x-marginal is pi, valid for either sign
of h, and it is ergodic: with no bond occupied every site recolours independently.

Sampling: n_chains independent chains from uniform random starts; burn_in updates; then one sample
per chain every `thin` updates for n_rounds rounds, n_chains * n_rounds samples in total.

usage:
  python gt_sw.py test                                   exactness against enumeration
  python gt_sw.py run OUT_DIR SPEC [n_chains burn_in n_rounds thin workers]
     SPEC = bridge | ising50 | potts24 | all
"""

import itertools
import json
import math
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from multiprocessing import Pool
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

CELLS = {
    # bridge_ising (L=25, h=0): both priors and the common target
    "bridge": [("ising", 25, 2, b, 0.0) for b in (0.20, 0.50, 0.60)],
    # fixed_point_ising_beta_h, i50b-mdns eval grid
    "ising50": [("ising", 50, 2, b, h) for b in (0.400, 0.4407, 0.600)
                for h in (-0.10, -0.01, 0.0, 0.01, 0.10)],
    # fixed_point_potts_beta_h, p24d-mdns eval grid (q = 3, beta_c = log(1 + sqrt 3) = 1.00505)
    "potts24": [("potts", 24, 3, b, h) for b in (0.900, 1.00505, 1.200)
                for h in (-0.01, -0.002, 0.0, 0.002, 0.01)],
}


def edges(L):
    idx = np.arange(L * L).reshape(L, L)
    return np.concatenate([np.stack([idx.ravel(), np.roll(idx, -1, 0).ravel()], 1),
                           np.stack([idx.ravel(), np.roll(idx, -1, 1).ravel()], 1)])


def log_prob(model, x, L, beta, h):
    """log pi~ per configuration, float64.  x: [B, D] uint8 tokens."""
    g = x.reshape(-1, L, L).astype(np.int8)
    if model == "ising":
        s = 2.0 * g - 1.0
        bond = (s * np.roll(s, 1, 1)).sum((1, 2)) + (s * np.roll(s, 1, 2)).sum((1, 2))
        return beta * (bond + h * s.sum((1, 2)))
    a = (g == np.roll(g, 1, 1)).sum((1, 2)) + (g == np.roll(g, 1, 2)).sum((1, 2))
    return beta * (a + h * (g == 0).sum((1, 2)))


def sw_update(model, x, E, q, beta, h, rng):
    B, D = x.shape
    J = 2.0 * beta if model == "ising" else beta
    same = x[:, E[:, 0]] == x[:, E[:, 1]]
    act = same & (rng.random(same.shape, dtype=np.float32) < (1.0 - math.exp(-J)))
    bi, ei = np.nonzero(act)
    g = coo_matrix((np.ones(len(bi), dtype=np.int8), (E[ei, 0] + bi * D, E[ei, 1] + bi * D)),
                   shape=(B * D, B * D))
    _, lab = connected_components(g, directed=False)
    size = np.bincount(lab).astype(np.float64)
    u = rng.random(size.shape[0])
    if model == "ising":
        col = (u < 1.0 / (1.0 + np.exp(-2.0 * beta * h * size))).astype(np.uint8)
    else:
        p0 = 1.0 / (1.0 + (q - 1) * np.exp(-beta * h * size))
        other = 1 + rng.integers(0, q - 1, size.shape[0])
        col = np.where(u < p0, 0, other).astype(np.uint8)
    return col[lab].reshape(B, D)


def run_cell(args):
    model, L, q, beta, h, n_chains, burn, n_rounds, thin, seed = args
    rng = np.random.default_rng(seed)
    E = edges(L)
    x = rng.integers(0, q, (n_chains, L * L)).astype(np.uint8)
    t0 = time.time()
    for _ in range(burn):
        x = sw_update(model, x, E, q, beta, h, rng)
    out = []
    for _ in range(n_rounds):
        for _ in range(thin):
            x = sw_update(model, x, E, q, beta, h, rng)
        out.append(x.copy())
    X = np.concatenate(out)
    return dict(model=model, L=L, q=q, beta=beta, h=h, x=X, seed=seed, n_chains=n_chains,
                burn_in=burn, n_rounds=n_rounds, thin=thin, seconds=time.time() - t0)


def observables(r):
    X, L, q = r["x"], r["L"], r["q"]
    u = -log_prob(r["model"], X, L, r["beta"], r["h"]) / (L * L)
    if r["model"] == "ising":
        return dict(u=u.astype(np.float32), mean_spin=(2.0 * X - 1.0).mean(1).astype(np.float32))
    counts = np.stack([(X == a).sum(1) for a in range(q)], 1)
    return dict(u=u.astype(np.float32), state_fractions=(counts / (L * L)).astype(np.float32),
                majority_state=counts.argmax(1).astype(np.uint8))


def exact_test():
    """Enumerate small lattices and compare E[u] and the field observable with SW sample means."""
    worst = 0.0
    for model, L, q, beta, h in [("ising", 4, 2, 0.44, 0.0), ("ising", 4, 2, 0.60, 0.08),
                                 ("ising", 4, 2, 0.35, -0.15), ("potts", 3, 3, 1.0, 0.0),
                                 ("potts", 3, 3, 1.2, 0.05), ("potts", 3, 3, 0.9, -0.08)]:
        allx = np.array(list(itertools.product(range(q), repeat=L * L)), dtype=np.uint8)
        lp = log_prob(model, allx, L, beta, h)
        p = np.exp(lp - lp.max()); p /= p.sum()
        u_ex = float((p * (-lp / (L * L))).sum())
        f = ((2.0 * allx - 1.0).mean(1) if model == "ising" else (allx == 0).mean(1))
        f_ex = float((p * f).sum())
        r = run_cell((model, L, q, beta, h, 4096, 50, 20, 3, 11))
        ob = observables(r)
        fs = ob["mean_spin"] if model == "ising" else ob["state_fractions"][:, 0]
        # standard errors from per-chain means (rounds of one chain are correlated)
        def se(v):
            per = v.reshape(r["n_rounds"], r["n_chains"]).mean(0)
            return per.std(ddof=1) / math.sqrt(r["n_chains"])
        zu = (ob["u"].mean() - u_ex) / se(ob["u"])
        zf = (fs.mean() - f_ex) / se(fs)
        worst = max(worst, abs(zu), abs(zf))
        print(f"{model} L={L} beta={beta} h={h:+.2f}: u {ob['u'].mean():+.5f} vs exact {u_ex:+.5f} "
              f"({zu:+.1f} se) | {'m' if model == 'ising' else 'n0/D'} {fs.mean():+.5f} vs {f_ex:+.5f} ({zf:+.1f} se)",
              flush=True)
    print("EXACT TEST", "OK" if worst < 4.0 else "FAIL", f"(worst {worst:.1f} se)")
    return worst < 4.0


def main():
    if sys.argv[1] == "test":
        sys.exit(0 if exact_test() else 1)
    out_dir, spec = sys.argv[2], sys.argv[3]
    n_chains = int(sys.argv[4]) if len(sys.argv) > 4 else 4096
    burn = int(sys.argv[5]) if len(sys.argv) > 5 else 500
    n_rounds = int(sys.argv[6]) if len(sys.argv) > 6 else 4
    thin = int(sys.argv[7]) if len(sys.argv) > 7 else 25
    workers = int(sys.argv[8]) if len(sys.argv) > 8 else 33
    specs = list(CELLS) if spec == "all" else [spec]
    jobs = []
    for sp in specs:
        for i, (model, L, q, beta, h) in enumerate(CELLS[sp]):
            jobs.append((model, L, q, beta, h, n_chains, burn, n_rounds, thin, 1000003 * (len(jobs) + 1)))
    os.makedirs(out_dir, exist_ok=True)
    with Pool(min(workers, len(jobs))) as pool:
        for r in pool.imap_unordered(run_cell, jobs):
            sub = f"{r['model']}_q{r['q']}_L{r['L']}" if r["model"] == "potts" else f"ising_L{r['L']}"
            os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
            name = f"beta{r['beta']:.5f}_h{r['h']:+.4f}"
            ob = observables(r)
            meta = {k: r[k] for k in ("model", "L", "q", "beta", "h", "seed", "n_chains", "burn_in",
                                      "n_rounds", "thin")}
            meta["n_samples"] = int(r["x"].shape[0])
            meta["seconds"] = round(r["seconds"], 1)
            np.savez_compressed(os.path.join(out_dir, sub, name + ".npz"), x=r["x"],
                                meta=np.array(json.dumps(meta)), **ob)
            extra = (f"<m>={ob['mean_spin'].mean():+.4f} <|m|>={np.abs(ob['mean_spin']).mean():.4f}"
                     if r["model"] == "ising" else
                     "modes=" + str(np.round(np.bincount(ob["majority_state"], minlength=r["q"]) / len(ob["u"]), 3).tolist()))
            print(f"{sub}/{name}: n={meta['n_samples']} u={ob['u'].mean():+.5f} {extra} [{r['seconds']:.0f}s]",
                  flush=True)


if __name__ == "__main__":
    main()
