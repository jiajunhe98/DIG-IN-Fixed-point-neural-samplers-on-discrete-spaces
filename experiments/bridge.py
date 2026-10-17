"""Ising temperature bridge (appendix): transport one Boltzmann distribution to another.

    python experiments/bridge.py train                 # print the training commands
    python experiments/bridge.py train --run --pairs cross --seeds 0
    python experiments/bridge.py score --run           # 16384 draws with path weights

Two transports on the 25x25 torus at h = 0, three seeds each, neither endpoint
a tractable reference:

    cross   beta 0.20 -> 0.60, across the transition (prior bank: Gibbs)
    ord     beta 0.50 -> 0.60, between two ordered distributions (prior bank:
            parallel tempering)

The settings are the ones the paper's runs were trained with, read off their
saved configs; the runs were stopped by hand after 1781-1840 rounds of 100
steps and scored at their last checkpoint.

  train   -> runs/bridge/bw-<pair>-s<seed>/ (cfg.json, ckpt.pt)
  score   -> runs/bridge_scores/bw-<pair>-s<seed>.npz

The free-energy difference is exact (Kaufman), so `scripts/make_bridge_tex_data.py`
reports |dlnZ| against ground truth.  It also needs the Swendsen-Wang samples
at the endpoints: `python experiments/amortized.py groundtruth --run`.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

COMMON = ("L=25 batch=512 buffer_cap=16384 ckpt_every=5 ctmc_steps=256 ema=0.997 "
          "ess_target=0.5 eta=0.3 eval_every=25 gibbs_rate=0.0 grad_clip=1.0 "
          "hidden=96 is_weight=1 keep_every=8 lam_phi=1.0 load_buffer=1 lr=0.0003 "
          "n_blocks=5 n_eval=2048 n_explore=1024 n_inner=100 pc_final=0 pc_sweeps=0 "
          "pc_tmin=0.75 phi_loss=wq prior_n=32768 prior_sweeps=400 ref_chains=4096 "
          "ref_sweeps=6000 warmup_outer=10 weight_decay=0.0")

#: pair -> (run-name stem, the settings that define the task)
PAIRS = {
    "cross": ("bw-b020-b060", "beta0=0.2 beta1=0.6 prior_sampler=gibbs"),
    "ord": ("bw-b05-b06", "beta0=0.5 beta1=0.6 prior_sampler=pt"),
}


def train_jobs(a):
    for pair in a.pairs:
        stem, task = PAIRS[pair]
        for seed in a.seeds:
            args = (f"{COMMON} {task} seed={seed} n_outer={a.rounds} "
                    f"out_dir={a.root}/bridge run_name={stem}-s{seed} "
                    f"wandb_mode={'online' if a.wandb else 'disabled'}")
            yield [sys.executable, "fixed_point/bridge_ising/train_bridge.py", *args.split()]


def score_jobs(a):
    for pair in a.pairs:
        stem, _task = PAIRS[pair]
        for seed in a.seeds:
            yield [sys.executable, "scripts/score_bridge.py",
                   "--run-dir", f"{a.root}/bridge/{stem}-s{seed}",
                   "--out", f"{a.root}/bridge_scores", "--draws", "16384"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=["train", "score"])
    ap.add_argument("--run", action="store_true",
                    help="run the commands here, one after another (default: print them)")
    ap.add_argument("--pairs", nargs="+", default=list(PAIRS), choices=list(PAIRS))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--rounds", type=int, default=1840,
                    help="train: rounds of 100 steps")
    ap.add_argument("--root", default="runs")
    ap.add_argument("--wandb", action="store_true", help="log to Weights & Biases")
    a = ap.parse_args()

    stage = {"train": train_jobs, "score": score_jobs}[a.stage]
    # paths stay relative to the repository root, where the commands run
    for cmd in stage(a):
        if not a.run:
            print(shlex.join(["python", *cmd[1:]]))
            continue
        print(f"[run] {shlex.join(cmd[1:])}", flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
