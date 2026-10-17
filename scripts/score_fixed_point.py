"""Score an amortised fixed-point checkpoint over its whole (beta, h) domain.

    python scripts/score_fixed_point.py --system ising --ckpt CKPT --out DIR

Draws `--n` independent configurations per grid cell with the run's own
evaluation sampler and writes, per cell, the configurations, the proposal
log-density, the unnormalised log target and the log importance weight.  It
computes no metric: |Delta ln Z|, ESS and TVD are all derived from those four
arrays, and keeping the arrays means the metrics can be recomputed or changed
without re-running an H100.

The grid
--------
`beta_range` in the checkpoint config is the network's *normalisation* range,
not where it was trained: `train_h.py` draws beta uniformly on
[`beta_lo`, `beta_hi`] and h uniformly on [-`h_max`, `h_max`], and the
acquisition map tiles exactly that box.  So that box is the amortisation
domain and is what gets swept here.

The sweep is a regular `--grid-b` x `--grid-h` lattice over the box, unioned
with the run's own `eval_betas` x `eval_hs` cells.  The union matters: the
regular lattice gives a map over the whole domain, but only the run's
evaluation cells have Swendsen-Wang ground truth beside them in
`gt_samples/`, and those cells sit at values no round lattice contains
(beta_c = 0.4407 for Ising, 1.00505 for Potts).  Dropping them would leave
every ground-truth comparison uncomputable.

Weights
-------
`unmask_sample(..., return_logq=True)` accumulates
log q_path(x) = sum over reveals of log P(token drawn).  For absorbing masking
the reveal-order law is model-independent and cancels between proposal and
target on the extended space, so

    log w = log pi~(x) - log q_path(x)

is a valid importance log-weight.  `log pi~` is the unnormalised log target in
each family's own convention (see `energy_mdns.py` / `potts_field.py`); it is
NOT a probability, and log w therefore estimates ln Z up to nothing -- that is
the point.  Both are accumulated in float64.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib

import numpy as np
import torch

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXT = os.path.join(HERE, "fixed_point")


def _load(system: str):
    """Import the vendored modules for one family.

    They use flat imports (`import ising_energy as ie`), exactly as on the
    trainers they came from, so their directory goes on sys.path rather than
    being rewritten into a package -- keeping them identical to the code
    the checkpoints were trained with.
    """
    if system == "ising":
        sys.path.insert(0, os.path.join(EXT, "ising_beta_h"))
        import energy_mdns as energy
        import network_h as net
        return net, energy
    sys.path.insert(0, os.path.join(EXT, "potts_beta_h"))
    import network_hq as net
    import potts_field as energy
    return net, energy


def build(system: str, ckpt: str, device):
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    net, energy = _load(system)
    if system == "ising":
        model = net.FieldDenoiser(
            cfg["L"], cfg["hidden"], cfg["n_blocks"], cfg["kernel"],
            beta_range=tuple(cfg["beta_range"]), h_norm=cfg["h_max"],
            dilations=cfg["dilations"])
    else:
        model = net.PottsFieldDenoiser(
            cfg["L"], cfg["q"], cfg["hidden"], cfg["n_blocks"], cfg["kernel"],
            beta_range=tuple(cfg["beta_range"]), h_norm=cfg["h_max"],
            dilations=cfg["dilations"])
    # the EMA weights are the ones the run sampled from, not `model`
    model.load_state_dict(ck["ema"])
    model.to(device).eval()
    return model, cfg, energy, net, int(ck.get("outer", -1))


def grid(cfg, n_b: int, n_h: int, with_eval: bool) -> list[tuple[float, float]]:
    b = np.linspace(cfg["beta_lo"], cfg["beta_hi"], n_b)
    h = np.linspace(-cfg["h_max"], cfg["h_max"], n_h)
    cells = [(float(x), float(y)) for x in b for y in h]
    if with_eval:
        cells += [(float(x), float(y))
                  for x in cfg["eval_betas"] for y in cfg["eval_hs"]]
    # round before de-duplicating: linspace and the config literals disagree in
    # the last ulp for cells that are genuinely the same point (h = 0.0, the
    # box corners), and scoring those twice would waste a cell and produce two
    # files that differ only by seed
    seen, out = set(), []
    for bb, hh in cells:
        k = (round(bb, 6), round(hh, 6))
        if k not in seen:
            seen.add(k)
            out.append(k)
    return sorted(out)


def cell_name(b: float, h: float) -> str:
    return f"beta{b:.5f}_h{h:+.4f}"


def score_cell(model, energy, net, system, cfg, b, h, n, steps, n_single,
               batch, device, seed, autocast):
    L, D = cfg["L"], cfg["L"] ** 2
    q = cfg.get("q", 2)
    torch.manual_seed(seed)

    xs, lqs, lps = [], [], []
    done = 0
    while done < n:
        m = min(batch, n - done)
        bt = torch.full((m,), b, device=device)
        ht = torch.full((m,), h, device=device)
        if system == "ising":
            x, logq = net.unmask_sample(model, bt, ht, m, steps, device, D,
                                        n_single, autocast, return_logq=True)
        else:
            x, logq = net.unmask_sample(model, bt, ht, m, steps, device, D, q,
                                        n_single, autocast, return_logq=True)
        logpi = energy.log_prob_unnormalised(x, b, h, L).double()
        xs.append(x.to(torch.uint8).cpu())
        lqs.append(logq.cpu())
        lps.append(logpi.cpu())
        done += m
    x = torch.cat(xs)[:n]
    log_q = torch.cat(lqs)[:n].double()
    log_pi = torch.cat(lps)[:n].double()
    log_w = log_pi - log_q

    out = {
        "x": x.numpy(),
        "log_q": log_q.numpy(),
        "log_pi": log_pi.numpy(),
        "log_w": log_w.numpy(),
        # -log pi~ / L^2, the per-site energy in each family's own convention
        "u": (-log_pi / (L * L)).float().numpy(),
    }
    xi = x.to(torch.long)
    if system == "ising":
        out["mean_spin"] = (2.0 * xi.double() - 1.0).mean(1).float().numpy()
    else:
        fr = torch.stack([(xi == k).double().mean(1) for k in range(q)], 1)
        out["state_fractions"] = fr.float().numpy()
        out["majority_state"] = fr.argmax(1).to(torch.uint8).numpy()
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--system", required=True, choices=("ising", "potts"))
    p.add_argument("--ckpt", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, default=5000)
    p.add_argument("--steps", type=int, default=0, help="0 = the run's eval_steps")
    p.add_argument("--n-single", type=int, default=-1, help="-1 = the run's n_single")
    p.add_argument("--batch", type=int, default=0, help="0 = 500 ising / 1000 potts")
    p.add_argument("--grid-b", type=int, default=9)
    p.add_argument("--grid-h", type=int, default=9)
    p.add_argument("--no-eval-cells", action="store_true")
    p.add_argument("--eval-only", action="store_true",
                   help="only the run's own eval_betas x eval_hs cells -- the "
                        "ones with Swendsen-Wang ground truth beside them")
    p.add_argument("--cells", default="",
                   help="explicit 'beta,h;beta,h' instead of the swept grid; "
                        "for re-running a single cell or checking one against "
                        "the ground truth")
    p.add_argument("--shard", type=int, default=0)
    p.add_argument("--n-shards", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--no-autocast", action="store_true")
    a = p.parse_args()

    dev = torch.device(a.device)
    # the samplers open a CUDA autocast region unconditionally; off-GPU that is
    # at best a no-op with a warning, so it follows the device by default
    autocast = (dev.type == "cuda") and not a.no_autocast
    model, cfg, energy, net, outer = build(a.system, a.ckpt, dev)
    steps = a.steps or cfg["eval_steps"] or cfg["explore_steps"]
    n_single = cfg["n_single"] if a.n_single < 0 else a.n_single
    batch = a.batch or (500 if a.system == "ising" else 1000)
    if a.eval_only:
        cells = sorted({(round(float(x), 6), round(float(y), 6))
                        for x in cfg["eval_betas"] for y in cfg["eval_hs"]})
    elif a.cells:
        cells = [(round(float(t.split(",")[0]), 6), round(float(t.split(",")[1]), 6))
                 for t in a.cells.split(";") if t.strip()]
    else:
        cells = grid(cfg, a.grid_b, a.grid_h, not a.no_eval_cells)
    mine = cells[a.shard::a.n_shards]

    os.makedirs(a.out, exist_ok=True)
    print(f"{a.system}: L={cfg['L']} q={cfg.get('q', 2)} outer={outer}  "
          f"steps={steps} n_single={n_single} batch={batch}", flush=True)
    print(f"domain beta [{cfg['beta_lo']},{cfg['beta_hi']}] "
          f"h [-{cfg['h_max']},{cfg['h_max']}]  "
          f"{len(cells)} cells, shard {a.shard}/{a.n_shards} -> {len(mine)}",
          flush=True)

    for i, (b, h) in enumerate(mine):
        f = os.path.join(a.out, cell_name(b, h) + ".npz")
        if os.path.exists(f):                       # preemption-safe: skip done
            print(f"[{i + 1}/{len(mine)}] {cell_name(b, h)} exists, skip", flush=True)
            continue
        t0 = time.time()
        # Seed per cell, not per shard, so a cell is reproducible wherever it
        # runs -- and crc32 rather than hash(), which is salted per process for
        # str and would silently give a different stream on every launch.
        sd = (a.seed * 1_000_003
              + zlib.crc32(cell_name(b, h).encode())) % (2 ** 31 - 1)
        d = score_cell(model, energy, net, a.system, cfg, b, h, a.n, steps,
                       n_single, batch, dev, sd, autocast)
        lw = d["log_w"].astype(np.float64)
        w = np.exp(lw - lw.max())
        ess = float(w.sum() ** 2 / (len(w) * (w ** 2).sum()))
        d["meta"] = json.dumps({
            "system": a.system, "L": cfg["L"], "q": cfg.get("q", 2),
            "beta": b, "h": h, "n": a.n, "steps": steps, "n_single": n_single,
            "ckpt": os.path.basename(a.ckpt), "outer": outer, "seed": sd,
            "run_name": cfg.get("run_name", ""),
        })
        tmp = f + ".tmp.npz"                        # never leave a half file
        np.savez_compressed(tmp, **d)
        os.replace(tmp, f)
        print(f"[{i + 1}/{len(mine)}] {cell_name(b, h)}  "
              f"ESS/n={ess:.4f}  u={d['u'].mean():+.5f}  "
              f"sd(logw)={lw.std():.2f}  {time.time() - t0:.0f}s", flush=True)

    print("shard done", flush=True)


if __name__ == "__main__":
    main()
