"""Our chunked fixed-count unmasking vs MD4 / MDLM ancestral sampling, at matched NFE.

Both samplers draw from the same trained denoiser and both return an exact path log-density,
so they can be compared on three axes that matter for this project:

  * BIAS      -- energy per site and the mode fractions against the PT reference.  A sampler
                 that factorises too aggressively lands at the wrong energy.
  * PROPOSAL  -- ESS/n and the spread of log w = log pi~(x_1) - log q_path.  This is the axis
                 that decides how much the tempered importance weights can correct, and it is
                 the one the training loop actually consumes.
  * COST      -- wall time per sample at that NFE.

NFE is the number of network calls, which is the whole cost at these lattice sizes, so every
comparison is at matched NFE rather than matched wall time.

  python compare_samplers.py CKPT [--device cuda] [--n 512] [--steps 32,64,128,288,576]

With no checkpoint it builds a randomly-initialised network, which exercises the harness but
says nothing about sampler quality.
"""

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import potts_field as pf
from network_hq import PottsFieldDenoiser, md4_sample, unmask_sample


def ess_frac(logw):
    y = logw.double()
    y = y - y.max()
    w = torch.exp(y)
    return float((w.sum() ** 2) / (y.numel() * (w * w).sum()))


def run(tag, fn, n, beta, h, L, q, ref=None):
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    t0 = time.time()
    x, logq = fn()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    dt = time.time() - t0
    u = pf.energy_per_spin(x, beta, h, L)
    lw = pf.log_prob_unnormalised(x, beta, h, L).double() - logq
    mf = pf.mode_fractions(x, L, q)
    row = dict(tag=tag, u=float(u.mean()), mode_min=float(mf.min()),
               modes=[round(v, 3) for v in mf.tolist()],
               ess=ess_frac(lw), span=float(lw.max() - lw.min()),
               sd=float(lw.std()), sec=dt)
    if ref is not None:
        row["du"] = row["u"] - ref
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="?", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--steps", default="")
    ap.add_argument("--beta", type=float, default=None)
    ap.add_argument("--h", type=float, default=0.0)
    ap.add_argument("--n_single", type=int, default=64)
    ap.add_argument("--ref_sweeps", type=int, default=800)
    ap.add_argument("--ref_chains", type=int, default=256)
    a = ap.parse_args()
    dev = torch.device(a.device)
    torch.manual_seed(0)

    if a.ckpt:
        ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
        cfg = ck["cfg"]
        L, q = cfg["L"], cfg["q"]
        net = PottsFieldDenoiser(L, q, cfg["hidden"], cfg["n_blocks"], cfg["kernel"],
                                 beta_range=tuple(cfg["beta_range"]), h_norm=cfg["h_max"],
                                 dilations=cfg["dilations"]).to(dev)
        net.load_state_dict(ck["ema"])
        print(f"loaded {a.ckpt}  (outer {ck.get('outer','?')})  L={L} q={q}")
    else:
        L, q = 16, 3
        net = PottsFieldDenoiser(L, q, dilations=None).to(dev)
        print(f"NO CHECKPOINT -- randomly initialised L={L} q={q}; harness test only")
    net.eval()
    D = L * L
    beta = a.beta if a.beta is not None else pf.beta_critical(q) + 0.2
    steps = ([int(s) for s in a.steps.split(",")] if a.steps
             else [s for s in (D // 16, D // 8, D // 4, D // 2, D) if s >= 4])

    print(f"\nbeta={beta:.5f}  h={a.h:+.4f}  L={L}  D={D}  n={a.n}")
    t0 = time.time()
    xr = pf.pt_reference(beta, a.h, L, q, a.ref_chains, a.ref_sweeps, dev, seed=1234, n_temps=16)
    ref_u = float(pf.energy_per_spin(xr, beta, a.h, L).mean())
    ref_mf = pf.mode_fractions(xr, L, q)
    print(f"PT reference ({a.ref_chains}x{a.ref_sweeps}, {time.time()-t0:.0f}s): "
          f"u={ref_u:+.4f}  modes={[round(v,3) for v in ref_mf.tolist()]}\n")

    b = torch.full((a.n,), beta, device=dev)
    h = torch.full((a.n,), a.h, device=dev)
    hdr = (f"{'sampler':>10} {'NFE':>5} {'u':>9} {'du':>8} {'ESS/n':>7} "
           f"{'sd(logw)':>9} {'span':>8} {'mode_min':>9} {'sec':>6}")
    print(hdr); print("-" * len(hdr))
    for T in steps:
        rows = [
            run("ours", lambda T=T: unmask_sample(net, b, h, a.n, T, dev, D, q,
                                                  n_single=min(a.n_single, T - 1),
                                                  return_logq=True), a.n, b, h, L, q, ref_u),
            run("MD4", lambda T=T: md4_sample(net, b, h, a.n, T, dev, D, q,
                                              return_logq=True), a.n, b, h, L, q, ref_u),
        ]
        for r in rows:
            print(f"{r['tag']:>10} {T:>5} {r['u']:>+9.4f} {r['du']:>+8.4f} {r['ess']:>7.4f} "
                  f"{r['sd']:>9.2f} {r['span']:>8.1f} {r['mode_min']:>9.3f} {r['sec']:>6.1f}")
        print()


if __name__ == "__main__":
    main()
