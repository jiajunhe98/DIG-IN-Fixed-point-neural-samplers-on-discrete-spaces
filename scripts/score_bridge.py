"""Score a fixed-point bridge run: transport one Ising temperature to another.

    python scripts/score_bridge.py --run-dir <dir> --out <dir> --draws 8192
    python scripts/score_bridge.py --run-dir <dir> --selftest        # no scoring

The runs come from `experiments/bridge.py` (`runs/bridge/<run>/`).  Each
transports `pi_0 = Ising(beta_0)` to
`pi_1 = Ising(beta_1)` on a $25\times25$ torus at $h=0$: `bw-b020-b060-*` runs
0.20 -> 0.60, across the transition, and `bw-b05-b06-*` runs 0.50 -> 0.60,
between two ordered distributions.  Neither endpoint is a tractable reference
distribution, which is the point.

What comes out, per run, in one `.npz`:

  x0, x1        prior draws and transported endpoints, tokens in {0,1}
  logw          the path-space weight of each endpoint (float64)
  m_snap, t_snap  magnetization of every draw on a fine grid of times
  cfg_snap, t_cfg  whole configurations on a coarse grid, for the panels
  u1            -log pi_1~(x1) / L^2, the bare energy per site

The two time grids are separate on purpose.  The magnetization is one float
per draw, so it can be recorded every few steps and gives the transport map
its vertical resolution; a configuration is 625 bytes per draw, so those are
kept at nine times only and for a handful of draws.

`logw` is what `train_bridge.ctmc_sample(return_logw=True)` returns,

    log w = log pi_1~(x_1) - log pi_0~(x_0) + sum_k [log B_k - log F_k],

so `E_path[w] = Z_1 / Z_0` exactly: `logsumexp(logw) - log n` estimates the
free-energy difference between the two temperatures, and Kaufman gives that
difference in closed form at every finite L.  No target sample is used
anywhere -- the prior bank is MCMC at `beta_0`, and `pi_1` enters only through
its energy.

The snapshot loop is a copy of the vendored `ctmc_sample` with the state
recorded at a few times; `--selftest` checks it reproduces that function's
endpoint and weight bit for bit under a shared seed, so the copy cannot drift
silently.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE, "fixed_point", "bridge_ising"))

import ising_energy as ie          # noqa: E402
import network_bs as nb            # noqa: E402

# `train_bridge` is imported lazily, inside --selftest only: it is the training
# module and pulls matplotlib and wandb at import time, neither of which a
# scoring container has any reason to carry.


def load_run(path: str, device: str):
    """The EMA weights and the config they were trained under.

    Takes either a run directory (`cfg.json` + `ckpt.pt`, what the trainer
    writes) or a single stripped `{cfg, ema, outer}` checkpoint file
    (`<run>_ema.pt`).
    """
    if os.path.isdir(path):
        with open(os.path.join(path, "cfg.json"), encoding="utf-8") as f:
            cfg = json.load(f)
        ck = torch.load(os.path.join(path, "ckpt.pt"), map_location="cpu",
                        weights_only=False)
    else:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        cfg = ck["cfg"]
    model = nb.BlindSpotNet(L=cfg["L"], hidden=cfg["hidden"],
                            n_blocks=cfg["n_blocks"]).to(device).eval()
    model.load_state_dict(ck["ema"])
    return model, cfg, int(ck.get("outer", -1))


@torch.no_grad()
def prior_bank(cfg, n: int, device: str, seed: int, chunk: int = 2048):
    """Draws from pi_0, by the sampler the run itself used.

    Blocked Gibbs is adequate below beta_c and is what the 0.20 runs used; at
    beta_0 = 0.50 the prior is already ordered and Gibbs cannot move between
    its two modes, so those runs built the bank by parallel tempering.  Getting
    this wrong would hand the transport a prior that is itself mode-collapsed.
    """
    pt = str(cfg.get("prior_sampler", "gibbs")).lower() == "pt"
    out = []
    done = 0
    while done < n:
        b = min(chunk, n - done)
        if pt:
            x = ie.pt_reference(cfg["beta0"], 0.0, cfg["L"], b,
                                cfg["prior_sweeps"], device, seed=seed + done)
        else:
            x = ie.gibbs_reference(cfg["beta0"], 0.0, cfg["L"], b,
                                   cfg["prior_sweeps"], device, seed=seed + done)
        out.append(x.to(torch.uint8))
        done += b
    return torch.cat(out).long()


@torch.no_grad()
def ctmc_with_snapshots(model, x0, K, bet0, bet1, L, snap_t, autocast=True,
                        m_every=1):
    """`train_bridge.ctmc_sample(..., return_logw=True)` plus snapshots.

    Line for line the vendored sampler -- the same heads, the same tau-leap,
    the same float64 path terms -- with the state copied out at the requested
    times.  `--selftest` pins it to the original.

    Configurations are kept at the times in `snap_t`; the magnetization is
    kept every `m_every` steps, which is what sets the resolution of the
    transport map and costs one float per draw per row.
    """
    x = x0.clone()
    n = x.shape[0]
    dt = 1.0 / K
    logF = torch.zeros(n, device=x.device, dtype=torch.float64)
    logB = torch.zeros(n, device=x.device, dtype=torch.float64)
    prev_flip = None
    snaps: dict[float, torch.Tensor] = {}
    want = sorted(snap_t)
    m_t: list[float] = []
    m_v: list[torch.Tensor] = []

    def take_m(tv):
        m_t.append(tv)
        m_v.append((2.0 * x.float() - 1.0).mean(-1).cpu())

    def heads(xs, tv):
        t = torch.full((n,), tv, device=xs.device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            ql, phi = model(xs, t)
        qp = torch.softmax(ql.float(), -1)
        phi = phi.float()
        cur, oth = xs, 1 - xs
        q_oth = torch.gather(qp, 2, oth[..., None]).squeeze(-1)
        dphi = (torch.gather(phi, 2, oth[..., None])
                - torch.gather(phi, 2, cur[..., None])).squeeze(-1)
        return q_oth, dphi

    def maybe_snap(tv):
        for s in want:
            if abs(tv - s) < 0.5 / K and s not in snaps:
                snaps[s] = x.detach().to(torch.uint8).cpu()

    for k in range(K):
        maybe_snap(k / K)
        if k % m_every == 0:
            take_m(k / K)
        q_oth, dphi = heads(x, k / K)
        if prev_flip is not None:
            rb = (q_oth * (-dphi).clamp_min(0.0)).double()
            pb = 1.0 - torch.exp(-rb * dt)
            logB += torch.where(prev_flip, torch.log(pb.clamp_min(1e-300)),
                                -rb * dt).sum(-1)
        rate = q_oth * dphi.clamp_min(0.0)
        pf = 1.0 - torch.exp(-rate * dt)
        flip = torch.rand_like(rate) < pf
        r64 = rate.double()
        pf64 = 1.0 - torch.exp(-r64 * dt)
        logF += torch.where(flip, torch.log(pf64.clamp_min(1e-300)),
                            -r64 * dt).sum(-1)
        x = torch.where(flip, 1 - x, x)
        prev_flip = flip
    q_oth, dphi = heads(x, 1.0)
    rb = (q_oth * (-dphi).clamp_min(0.0)).double()
    pb = 1.0 - torch.exp(-rb * dt)
    logB += torch.where(prev_flip, torch.log(pb.clamp_min(1e-300)), -rb * dt).sum(-1)
    maybe_snap(1.0)
    take_m(1.0)
    logw = (bet1 * ie.bond_sum(x, L) - bet0 * ie.bond_sum(x0, L)).double() + logB - logF
    return x, logw, snaps, torch.tensor(m_t), torch.stack(m_v)


def selftest(run_dir: str, device: str, n: int = 8, K: int = 16) -> None:
    """The snapshot copy must be the vendored sampler, not a lookalike."""
    from train_bridge import ctmc_sample      # noqa: PLC0415 -- see the note above

    model, cfg, _ = load_run(run_dir, device)
    L, b0, b1 = cfg["L"], cfg["beta0"], cfg["beta1"]
    ac = device == "cuda"
    x0 = (torch.rand(n, L * L, device=device) < 0.5).long()

    torch.manual_seed(1234)
    xa, wa = ctmc_sample(model, x0, K, autocast=ac, return_logw=True,
                         bet0=b0, bet1=b1, L=L)
    torch.manual_seed(1234)
    xb, wb, snaps, mt, mv = ctmc_with_snapshots(model, x0, K, b0, b1, L,
                                                [0.0, 0.5, 1.0], autocast=ac)
    dx = int((xa != xb).sum())
    dw = float((wa - wb).abs().max())
    print(f"  endpoint mismatches {dx}   |dlogw|max {dw:.3e}   "
          f"cfg snaps {sorted(snaps)}   m rows {len(mt)}")
    assert dx == 0 and dw < 1e-9, "the snapshot loop has drifted from ctmc_sample"
    # the magnetization grid must end on the endpoint it reports
    dm = float((mv[-1] - (2.0 * xb.float() - 1.0).mean(-1).cpu()).abs().max())
    assert len(mt) == K + 1 and abs(float(mt[-1]) - 1.0) < 1e-9 and dm < 1e-6, \
        "the magnetization grid does not line up with the path"

    # an untrained model never flips and the path terms vanish, leaving the
    # exact prior -> target weight; the docstring's own sanity check
    fresh = nb.BlindSpotNet(L=L, hidden=cfg["hidden"], n_blocks=cfg["n_blocks"]).to(device)
    with torch.no_grad():
        for p in fresh.parameters():
            p.zero_()
    _, w0, _, _, _ = ctmc_with_snapshots(fresh, x0, K, b0, b1, L, [],
                                         autocast=False)
    want = ((b1 - b0) * ie.bond_sum(x0, L)).double()
    err = float((w0 - want).abs().max())
    # bond_sum accumulates 2 L^2 = 1250 float32 terms, so the weight itself is
    # O(500) with O(1e-6) of rounding in it; anything at that scale is the sum,
    # not the sampler.
    print(f"  zero-model weight vs (b1-b0)B(x0): {err:.3e} (float32 bond sum)")
    assert err < 1e-3
    print("  selftest OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", required=True,
                    help="archived run directory, or a stripped {cfg, ema, outer} .pt")
    ap.add_argument("--out", default=None)
    ap.add_argument("--draws", type=int, default=8192)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=None, help="default: cfg ctmc_steps")
    ap.add_argument("--n-cfg", type=int, default=8, help="configurations kept per snapshot")
    ap.add_argument("--m-every", type=int, default=2,
                    help="record the magnetization every N tau-leap steps; "
                         "this sets the resolution of the transport map")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    name = os.path.basename(a.run_dir.rstrip("/\\"))
    name = name[:-7] if name.endswith("_ema.pt") else name
    print(f"{name} on {device}")

    if a.selftest:
        selftest(a.run_dir, device)
        return

    model, cfg, outer = load_run(a.run_dir, device)
    L, b0, b1 = cfg["L"], cfg["beta0"], cfg["beta1"]
    K = a.steps or cfg["ctmc_steps"]
    snap_t = [round(i / 8, 3) for i in range(9)]
    print(f"  beta {b0} -> {b1}, L={L}, K={K}, round {outer}, "
          f"prior={cfg.get('prior_sampler')}")

    x0s, x1s, lws, ms, cs = [], [], [], [], []
    t_fine = None
    t_start = time.time()
    done = 0
    while done < a.draws:
        b = min(a.batch, a.draws - done)
        torch.manual_seed(a.seed * 100003 + done)
        x0 = prior_bank(cfg, b, device, seed=a.seed * 7919 + done)
        x1, logw, snaps, mt, mv = ctmc_with_snapshots(
            model, x0, K, b0, b1, L, snap_t,
            autocast=(device == "cuda"), m_every=a.m_every)
        t_fine = mt.numpy()
        x0s.append(x0.to(torch.uint8).cpu().numpy())
        x1s.append(x1.to(torch.uint8).cpu().numpy())
        lws.append(logw.cpu().numpy())
        ms.append(mv.numpy())
        cs.append(np.stack([snaps[t][:a.n_cfg].numpy() for t in snap_t]))
        done += b
        print(f"  {done}/{a.draws}  {time.time() - t_start:.0f} s", flush=True)

    x1 = np.concatenate(x1s)
    logw = np.concatenate(lws)
    u1 = -(b1 * ie.bond_sum(torch.from_numpy(x1).long(), L).numpy()) / (L * L)
    out = a.out or os.path.join(HERE, "runs", "bridge_scores")
    os.makedirs(out, exist_ok=True)
    p = os.path.join(out, f"{name}.npz")
    np.savez_compressed(
        p,
        x0=np.concatenate(x0s), x1=x1, logw=logw,
        m_snap=np.concatenate(ms, axis=1).astype(np.float32),
        t_snap=np.asarray(t_fine, dtype=np.float32),
        cfg_snap=cs[0], t_cfg=np.asarray(snap_t, dtype=np.float32),
        u1=u1.astype(np.float32),
        meta=json.dumps({"run": name, "beta0": b0, "beta1": b1, "L": L,
                         "steps": K, "draws": int(a.draws), "round": outer,
                         "prior_sampler": cfg.get("prior_sampler"),
                         "prior_sweeps": cfg.get("prior_sweeps"),
                         "m_every": int(a.m_every),
                         "m_rows": int(len(t_fine)),
                         "seed": a.seed}))
    lw = logw - logw.max()
    w = np.exp(lw)
    print(f"-> {p}  ({os.path.getsize(p) / 1e6:.1f} MB)")
    print(f"   dlnZ_hat {logw.max() + np.log(w.mean()):.4f}   "
          f"ESS/n {w.sum() ** 2 / (len(w) * (w ** 2).sum()):.3f}   "
          f"<m> {np.mean(np.abs(ie.magnetisation(torch.from_numpy(x1).long(), L).numpy())):.3f}")


if __name__ == "__main__":
    main()
