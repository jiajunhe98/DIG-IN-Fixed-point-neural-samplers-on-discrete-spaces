"""Alloy phase diagrams: DIG-IN and FrOGS on Ag-Pd and Cu-Au.

    python experiments/alloy.py train                  # print the training commands
    python experiments/alloy.py train --run --systems agpd125 --methods digin --seeds 0
    python experiments/alloy.py sweep --run            # 5000 draws per state point
    python experiments/alloy.py tangent --run          # phase diagram from the draws

Three stages, each a list of independent commands:

  train    one model per (system, method, seed), amortized over the whole
           (T, delta_mu) plane: 60k steps, one 80 GB H100 in the paper.
           -> runs/alloy/<system>/<method>/s<seed>/
  sweep    re-evaluate the trained model on the packaged (T, delta_mu) sweep
           (2349 state points for Ag-Pd, 1044 for Cu-Au) with 5000 draws per
           point, split into shards that run independently, then merged.
           -> runs/alloy_n5000/<system>/<method>/s<seed>/grid.json, grid_samples.npz
  tangent  the common-tangent construction on the sweep's free-energy curves.
           -> runs/alloy_n5000/<system>/<method>/s<seed>/common_tangent.json, fmix.json

The phase diagram is read off the 5000-draw sweep, not off the training
run's own final evaluation (256 draws per point): at 256 draws Cu-Au misses
whole two-phase regions.  `scripts/make_alloy_tex_data.py` then builds the
table and the figure data from runs/alloy_n5000/.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SYSTEMS = ("agpd125", "cuau128")
#: shards of the 5000-draw sweep in the paper's runs; any number gives the same
#: sweep, this is only how it was parallelized
SHARDS = {"agpd125": 31, "cuau128": 14}

#: evaluation during training: the coarse grid while it trains, the packaged
#: sweep at the end
GRID = ("log_every=500 eval_every=5000 checkpoint_every=2000 eval_at_start=false "
        "grid.n_T=8 grid.n_dmu=13 grid.final_reference_grid=true "
        "reference.n_samples=0")

#: run directory -> (draws per state point in the evaluations during training,
#: in the final one, hydra overrides)
METHODS = {
    "digin": (256, 256, "method=dig_in model=rescnn method.acq_frac_uniform=0.5 "
                   "method.rollout_every=200 method.batch_size=1024 "
                   "method.buffer_capacity=25000 method.eta=0.5 "
                   "method.n_rollout=2048 method.target_ess=0.9 "
                   "method.warm_start=12 method.weight_bins=[4,4] "
                   "method.weighting=tempered_local grid.save_samples=true"),
    "frogs": (256, 512, "method=frogs"),
}


def posix(path: str) -> str:
    """Forward slashes: hydra's override grammar reads a backslash as an escape."""
    return path.replace(os.sep, "/")


def train_jobs(a):
    for system in a.systems:
        for method in a.methods:
            n_eval, n_final, over = METHODS[method]
            for seed in a.seeds:
                out = os.path.join(a.root, "alloy", system, method, f"s{seed}")
                args = (f"target={system} seed={seed} n_steps={a.n_steps} {GRID} "
                        f"eval_samples={n_eval} final_eval_samples={n_final} "
                        f"grid.n_per_point={n_final} "
                        f"wandb.enabled={'true' if a.wandb else 'false'} {over} "
                        f"hydra.run.dir={posix(out)}")
                done = os.path.join(out, "result.json")
                yield done, [[sys.executable, "-m", "discrete_sampler.train",
                              *args.split()]]


def sweep_jobs(a):
    for system in a.systems:
        k = a.shards or SHARDS[system]
        for method in a.methods:
            for seed in a.seeds:
                run = os.path.join(a.root, "alloy", system, method, f"s{seed}")
                out = os.path.join(a.root, "alloy_n5000", system, method, f"s{seed}")
                shards = [[sys.executable, "scripts/dump_alloy_samples.py", posix(run),
                           "--n", str(a.n_draws), "--shard", str(i),
                           "--n-shards", str(k), "--out-dir", posix(out)]
                          for i in range(k)]
                merge = [sys.executable, "scripts/merge_alloy_shards.py", posix(out)]
                yield os.path.join(out, "grid.json"), shards + [merge]


def tangent_jobs(a):
    for system in a.systems:
        for method in a.methods:
            for seed in a.seeds:
                out = os.path.join(a.root, "alloy_n5000", system, method, f"s{seed}")
                yield (os.path.join(out, "common_tangent.json"),
                       [[sys.executable, "scripts/model_common_tangent.py", posix(out)]])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=["train", "sweep", "tangent"])
    ap.add_argument("--run", action="store_true",
                    help="run the commands here, one after another (default: print them)")
    ap.add_argument("--systems", nargs="+", default=list(SYSTEMS), choices=SYSTEMS)
    ap.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-steps", type=int, default=60000)
    ap.add_argument("--n-draws", type=int, default=5000,
                    help="sweep: draws per state point")
    ap.add_argument("--shards", type=int, default=None,
                    help="sweep: shards per run (default: as in the paper)")
    ap.add_argument("--root", default="runs")
    ap.add_argument("--wandb", action="store_true", help="log to Weights & Biases")
    a = ap.parse_args()

    stage = {"train": train_jobs, "sweep": sweep_jobs, "tangent": tangent_jobs}[a.stage]
    # paths stay relative to the repository root, where the commands run
    for done, cmds in stage(a):
        if not a.run:
            for cmd in cmds:
                print(shlex.join(["python", *cmd[1:]]))
            continue
        if os.path.exists(os.path.join(ROOT, done)):
            print(f"[skip] {done} exists")
            continue
        for cmd in cmds:
            print(f"[run] {shlex.join(cmd[1:])}", flush=True)
            subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
