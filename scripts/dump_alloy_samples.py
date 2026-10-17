"""Re-run an alloy run's final grid evaluation and keep the raw draws.

    python scripts/dump_alloy_samples.py <run_dir> [--n 512] [--device cuda]

Loads the run's own config and checkpoint, re-samples the packaged (T, delta_mu)
sweep, and writes `grid_samples.npz` and `grid_dump.json` next to the
checkpoint.  Training is untouched -- evaluation is separable, so this scores
an existing model rather than needing a second training run.

The npz holds, per state point:

    x_packed   (n_points, n_samples, ceil(D/8)) uint8, bit-packed along sites
    log_w      (n_points, n_samples) float32, log nu_hat(x) - log q(x)
    energy     (n_points, n_samples) float32, the semi-grand H(s) in eV
    cond       (n_points, 2) float32, (T [K], delta_mu [eV])
    n_sites    the D to slice back to after unpacking

Recovering the configurations:

    z = np.load("grid_samples.npz")
    x = np.unpackbits(z["x_packed"], axis=-1)[..., :int(z["n_sites"])]

`x` is {0, 1} tokens in raster order over the cell; 1 is the spin-up species
(Ag for AgPd, Au for CuAu).  `AlloyFamily.spins` converts to the +-1 ordering
the cluster tables use.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# running a file in scripts/ puts scripts/ on sys.path, not the repo root, so
# `import dds` fails from a plain checkout without an editable install
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from omegaconf import OmegaConf

from dds.eval_alloy import (compare_phase_diagram, evaluate_alloy_grid,
                            phase_diagram_from_rows)
from dds.methods import build_method
from dds.targets import build_target
from dds.utils.checkpoint import load_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--n", type=int, default=512,
                    help="samples per state point (default 512, as in training)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--coarse", action="store_true",
                    help="use the training grid instead of the packaged sweep")
    ap.add_argument("--out", default=None, help="npz path (default in run_dir)")
    ap.add_argument("--shard", type=int, default=None,
                    help="evaluate only every --n-shards-th state point, "
                         "starting here; merge afterwards with "
                         "scripts/merge_alloy_shards.py")
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--out-dir", default=None,
                    help="where the shards land; defaults to run_dir.  Point "
                         "it elsewhere to re-evaluate a finished run without "
                         "overwriting the dump it already has.")
    a = ap.parse_args()

    cfg = OmegaConf.load(os.path.join(a.run_dir, ".hydra", "config.yaml"))
    target = build_target(OmegaConf.to_container(cfg.target, resolve=True))
    assert getattr(target, "grid_kind", None) == "alloy", (
        f"{a.run_dir} is not an alloy run")
    method_cfg = OmegaConf.to_container(cfg.method, resolve=True)
    name = method_cfg.pop("name")
    # ARM pins its own backbone; everything else takes the model group, the
    # same merge dds.train does before building the method
    if "model" not in method_cfg:
        method_cfg["model"] = OmegaConf.to_container(cfg.model, resolve=True)
    method = build_method(name, target, method_cfg, device=a.device)

    # load_checkpoint takes the two-slot PREFIX (<dir>/checkpoint ->
    # checkpoint_a.pt / checkpoint_b.pt), not the directory
    state, where = load_checkpoint(os.path.join(a.run_dir, "checkpoint"),
                                   map_location=a.device)
    if state is None:
        raise SystemExit(f"no checkpoint under {a.run_dir}")
    method.load_state_dict(state["method"] if "method" in state else state)
    print(f"loaded step {state.get('step', '?')} from {where}", flush=True)

    dest = a.out_dir or a.run_dir
    os.makedirs(dest, exist_ok=True)
    sharded = a.shard is not None and a.n_shards > 1
    if sharded:
        out = a.out or os.path.join(
            dest, f"grid_samples.shard{a.shard:03d}.npz")
    else:
        out = a.out or os.path.join(dest, "grid_samples.npz")
    summary, rows = evaluate_alloy_grid(
        method,
        n_per_point=a.n,
        n_T=cfg.grid.n_T,
        n_dmu=cfg.grid.n_dmu,
        use_reference_grid=not a.coarse,
        save_path=out,
        shard=(a.shard, a.n_shards) if sharded else None,
    )
    if sharded:
        # a shard holds a slice of the plane, so a phase diagram off it would
        # be built from isotherms with most of their delta_mu missing
        with open(os.path.join(dest,
                               f"grid_rows.shard{a.shard:03d}.json"),
                  "w", encoding="utf-8") as f:
            json.dump(rows, f)
        print(f"shard {a.shard}/{a.n_shards}: {len(rows)} state points -> {out}",
              flush=True)
        return
    pd = phase_diagram_from_rows(rows)
    summary.update(compare_phase_diagram(pd, target))
    with open(os.path.join(dest, "grid_dump.json"), "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": rows, "phase_diagram": pd}, f, indent=2)
    print(" ".join(f"{k}={v:.5g}" for k, v in summary.items()
                   if isinstance(v, float)))


if __name__ == "__main__":
    main()
