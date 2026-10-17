"""Amortized DIG-IN over (beta, h): 50x50 Ising and 24x24 q=3 Potts.

    python experiments/amortized.py train              # print the training commands
    python experiments/amortized.py train --run --systems ising --seeds 0
    python experiments/amortized.py score --run        # 5000 draws per cell, 9x9 grid
    python experiments/amortized.py reference --run    # Swendsen-Wang + BAR ln Z
    python experiments/amortized.py groundtruth --run  # optional: SW samples at the eval cells

One network per seed, conditioned on (beta, h) and trained over the whole box
-- Ising beta in [0.39, 0.65], |h| <= 0.1; Potts beta in [0.8, 1.3],
|h| <= 0.01 -- with the standalone trainers in fixed_point/.  The settings
below are the ones the paper's runs were trained with, read off their saved
configs; only the number of rounds is fixed here, where the paper's runs were
stopped by hand (Ising after 4120-4200 rounds of 100 steps, Potts after
18713-19063) and evaluated at their last checkpoint.

  train        -> runs/amortized/<system>-s<seed>/ (cfg.json, ckpt.pt)
  score        -> runs/amortized_scores/<system>-s<seed>/  (one .npz per cell)
  reference    -> runs/amortized_reference/  (ln Z and order-parameter counts)
  groundtruth  -> runs/gt_samples/  (also the bridge's endpoint references)

The trainers log to Weights & Biases (`wandb_mode=disabled` is passed unless
--wandb is given).  They do not resume on their own: to continue a run, repeat
its command with `init_from=<run>/ckpt.pt step_offset=<round>`.  At batch 1536
the 50x50 Ising run needs a 141 GB GPU (fixed_point/ising_beta_h/README.md).
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: system -> (trainer, rounds, seeds in the paper, settings of the paper's runs)
SYSTEMS = {
    "ising50": ("fixed_point/ising_beta_h/train_h.py", 4200, (0, 1, 2), (
        "L=50 acq=fisher acq_bins_b=8 acq_bins_h=12 acq_ema=0.8 acq_fisher_norm=1 "
        "acq_floor=0.05 acq_min_n=8 acq_power=0.5 adaptive=1 batch=1536 beta_hi=0.65 "
        "beta_lo=0.39 beta_range=[0.25,0.65] buffer_cap=49152 ckpt_every=10 compile=1 "
        "dilations=[1,1,2,4,8,1,1] ema=0.995 ess_target=0.8 eta=0.5 "
        "eval_betas=[0.4,0.4407,0.6] eval_every=200 eval_hs=[-0.1,-0.01,0.0,0.01,0.1] "
        "eval_steps=512 explore_steps=512 gibbs_sweeps=4000 grad_clip=1.0 h_max=0.1 "
        "hidden=96 is_alpha_bins=1 is_center_bins=1 is_min_bin=32 is_weight=1 "
        "keep_every=4 kernel=3 lr=0.0003 n_blocks=7 n_eval=384 n_explore=1536 "
        "n_inner=100 n_single=64 ref_chains=1024 ref_temps=24 warmup_outer=10 "
        "weight_decay=0.0")),
    "potts24": ("fixed_point/potts_beta_h/train_hq.py", 19000, (0, 1), (
        "L=24 acq=fisher acq_bins_b=8 acq_bins_h=12 acq_ema=0.8 acq_fisher_norm=1 "
        "acq_floor=0.05 acq_min_n=8 acq_power=0.5 adaptive=1 batch=1536 beta_hi=1.3 "
        "beta_lo=0.8 beta_range=[0.7,1.4] buffer_cap=196608 ckpt_every=10 compile=1 "
        "dilations=None ema=0.995 ess_target=0.8 eta=0.5 eval_betas=[0.9,1.00505,1.2] "
        "eval_every=200 eval_hs=[-0.01,-0.002,0.0,0.002,0.01] eval_steps=288 "
        "explore_steps=288 gibbs_sweeps=800 grad_clip=1.0 h_max=0.01 hidden=96 "
        "is_alpha_bins=1 is_center_bins=1 is_min_bin=32 is_weight=1 keep_every=4 "
        "kernel=3 lr=0.0003 n_blocks=7 n_eval=384 n_explore=1024 n_inner=100 "
        "n_single=64 q=3 ref_chains=256 ref_temps=16 warmup_outer=10 weight_decay=0.0")),
}
ALIAS = {"ising": "ising50", "potts": "potts24"}


def train_jobs(a):
    for system in a.systems:
        script, rounds, seeds, cfg = SYSTEMS[system]
        for seed in a.seeds if a.seeds is not None else seeds:
            name = f"{system}-s{seed}"
            args = (f"{cfg} seed={seed} n_outer={a.rounds or rounds} "
                    f"out_dir={a.root}/amortized run_name={name} "
                    f"wandb_mode={'online' if a.wandb else 'disabled'}")
            yield [sys.executable, script, *args.split()]


def score_jobs(a):
    for system in a.systems:
        _s, _r, seeds, _c = SYSTEMS[system]
        kind = system.rstrip("0123456789")
        for seed in a.seeds if a.seeds is not None else seeds:
            name = f"{system}-s{seed}"
            yield [sys.executable, "scripts/score_fixed_point.py", "--system", kind,
                   "--ckpt", f"{a.root}/amortized/{name}/ckpt.pt",
                   "--out", f"{a.root}/amortized_scores/{name}", "--n", "5000"]


def reference_jobs(a):
    """Swendsen-Wang + BAR reference at every cell of the 9 x 9 grid.

    Ising: Kaufman's exact ln Z(beta, 0) plus an h-ladder at each beta.
    Potts: a beta-ladder from the exact beta = 0 point, then an h-ladder at
    each beta, and the distribution over the ordered modes at every cell.
    `ising24_beta` validates the Potts procedure on Ising, where it is exact.
    CPU work; each command uses every core it is given.
    """
    out = f"{a.root}/amortized_reference"

    def run(task, rounds):
        return [sys.executable, "scripts/fp_reference.py", "run", task, "--out", out,
                "--burn", "100", "--rounds", str(rounds)]

    if "ising50" in a.systems:
        for b in np.linspace(0.39, 0.65, 9):
            yield run(f"ising_h:{round(float(b), 6)}", 300)
        yield run("ising24_beta", 1100)
    if "potts24" in a.systems:
        yield run("potts_beta", 1100)
        for b in np.linspace(0.8, 1.3, 9):
            yield run(f"potts_h:{round(float(b), 6)}", 300)
        yield [sys.executable, "scripts/fp_mode_reference.py"]


def groundtruth_jobs(a):
    """16384 Swendsen-Wang samples at each run's evaluation cells, a check on
    the scored draws that does not go through the reference above (and the
    endpoint references of experiments/bridge.py)."""
    yield [sys.executable, "fixed_point/gt_samples/gt_sw.py", "run",
           f"{a.root}/gt_samples", "all"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=["train", "score", "reference", "groundtruth"])
    ap.add_argument("--run", action="store_true",
                    help="run the commands here, one after another (default: print them)")
    ap.add_argument("--systems", nargs="+", default=list(SYSTEMS),
                    choices=[*SYSTEMS, *ALIAS])
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="default: the paper's (three for Ising, two for Potts)")
    ap.add_argument("--rounds", type=int, default=None,
                    help="train: rounds of 100 steps (default: as in the paper)")
    ap.add_argument("--root", default="runs")
    ap.add_argument("--wandb", action="store_true", help="log to Weights & Biases")
    a = ap.parse_args()
    a.systems = [ALIAS.get(s, s) for s in a.systems]

    stage = {"train": train_jobs, "score": score_jobs, "reference": reference_jobs,
             "groundtruth": groundtruth_jobs}[a.stage]
    # paths stay relative to the repository root, where the commands run
    for cmd in stage(a):
        if not a.run:
            print(shlex.join(["python", *cmd[1:]]))
            continue
        print(f"[run] {shlex.join(cmd[1:])}", flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
