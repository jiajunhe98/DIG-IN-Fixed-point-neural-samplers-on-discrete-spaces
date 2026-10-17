"""Ising and Potts phase sweep and the component ablation (appendix).

    python experiments/lattice.py                       # print every command
    python experiments/lattice.py --run --tasks ising25 --methods digin --seeds 0
    python experiments/lattice.py --ablation            # the four ablation arms
    python experiments/lattice.py --quick --run --methods digin --seeds 0  # smoke test

Six samplers -- DIG-IN, MDNS, PDNS, DASBS, DASBS memoryless and LEAPS -- on
25x25 Ising and 16x16 q=3 Potts, three phases each (disordered, critical,
ordered), three seeds, 150k gradient steps.  The paper trained every run on
one 80 GB H100.

Each run goes to runs/lattice/<task>/<phase>/<method>/s<seed>/ (ablation:
runs/ablation/ising25/<phase>/<arm>/s<seed>/).  `dds.train` resumes from the
checkpoint in its run directory, so an interrupted run continues when its
command is repeated, and a finished one (result.json present) is skipped.
Without --run the commands are printed, one per line, ready for a job array;
they expect to be run from the repository root.

MDNS and PDNS are run inside the DIG-IN codebase, with the same interpolant,
backbone, sampler, buffer and optimizer; they differ from DIG-IN only in the
three knobs below.  This is the comparison the paper reports, not the
published implementations:

    DIG-IN  soft Gibbs labels, weights tempered to ESS 0.8, damping eta = 0.1
    PDNS    hard labels,       weights tempered to ESS 0.8, no damping
    MDNS    hard labels,       full importance weights,     no damping
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: task -> (target config prefix, phases)
TASKS = {
    "ising25": ("ising25", ("ordered", "critical", "disordered")),
    "potts16": ("potts16", ("ordered", "critical", "disordered")),
}

#: run directory -> hydra overrides
METHODS = {
    "digin": "method=dsi_masked_ce model=rescnn method.eta=0.1 "
             "method.weighting=tempered method.target_ess=0.8",
    "mdns": "method=dsi_masked_ce model=rescnn method.labels=hard "
            "method.weighting=tempered method.target_ess=0.0 method.eta=1.0",
    "pdns": "method=dsi_masked_ce model=rescnn method.labels=hard "
            "method.weighting=tempered method.target_ess=0.8 method.eta=1.0",
    "dasbs": "method=asbs model=rescnn",
    "dasbs_memoryless": "method=asbs_memoryless model=rescnn",
    # a rollout at the default walker count dominates LEAPS' step cost
    "leaps": "method=leaps method.loss_batch=128 method.n_walkers=128 "
             "method.eval_chunk=256",
}

#: component ablation of DIG-IN on 25x25 Ising: each arm removes one
#: ingredient.  The baseline is the sweep's own `digin` runs.
ABLATIONS = {
    "no_damping": "method=dsi_masked_ce model=rescnn method.eta=1.0 "
                  "method.weighting=tempered method.target_ess=0.8",
    "no_gibbs_tempered": "method=dsi_masked_ce model=rescnn method.labels=hard "
                         "method.eta=0.1 method.weighting=tempered "
                         "method.target_ess=0.8",
    "no_gibbs_plain": "method=dsi_masked_ce model=rescnn method.labels=hard "
                      "method.eta=0.1 method.weighting=tempered "
                      "method.target_ess=0.0",
    "no_iw": "method=dsi_masked_ce model=rescnn method.eta=0.1 "
             "method.weighting=none",
}


def posix(path: str) -> str:
    """Forward slashes: hydra's override grammar reads a backslash as an escape."""
    return path.replace(os.sep, "/")


def common(a) -> str:
    return " ".join([
        f"n_steps={a.n_steps} log_every={a.log_every} eval_every={a.eval_every}",
        f"checkpoint_every={a.checkpoint_every} eval_at_start=true",
        f"eval_samples={a.eval_samples} final_eval_samples={a.final_eval_samples}",
        f"reference.n_samples={a.ref_samples} reference.method=mixed",
        "reference.burn_in=1000 reference.thin=5",
        f"reference.cache_dir={posix(a.ref_dir)}",
        "save_final_samples=true",
        f"wandb.enabled={'true' if a.wandb else 'false'}",
    ])


def method_overrides(over: str) -> str:
    # the masked sampler reveals one site per network call (D calls a draw);
    # the 64 single reveals only matter for a reduced step budget
    if "method=dsi_masked_ce" in over:
        over += " method.unmask_single=64"
    return over


def jobs(a):
    if a.ablation:
        table, tasks, root = ABLATIONS, ["ising25"], os.path.join(a.root, "ablation")
    else:
        table, tasks, root = METHODS, a.tasks, os.path.join(a.root, "lattice")
    names = a.methods or list(table)
    for name in names:
        if name not in table:
            sys.exit(f"unknown {'arm' if a.ablation else 'method'} {name!r}; "
                     f"have {list(table)}")
    for task in tasks:
        prefix, phases = TASKS[task]
        for phase in a.phases or phases:
            for name in names:
                for seed in a.seeds:
                    out = os.path.join(root, task, phase, name, f"s{seed}")
                    args = (f"target={prefix}_{phase} seed={seed} {common(a)} "
                            f"{method_overrides(table[name])} "
                            f"hydra.run.dir={posix(out)}")
                    yield out, [sys.executable, "-m", "dds.train", *args.split()]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", action="store_true",
                    help="run the jobs here, one after another (default: print them)")
    ap.add_argument("--ablation", action="store_true",
                    help="the component ablation on 25x25 Ising instead of the sweep")
    ap.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    ap.add_argument("--methods", nargs="+", default=None,
                    help="methods (or ablation arms) to run; default: all")
    ap.add_argument("--phases", nargs="+", default=None,
                    choices=["ordered", "critical", "disordered"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--n-steps", type=int, default=150000)
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--eval-every", type=int, default=5000)
    ap.add_argument("--checkpoint-every", type=int, default=500)
    ap.add_argument("--eval-samples", type=int, default=2048)
    ap.add_argument("--final-eval-samples", type=int, default=8192)
    ap.add_argument("--ref-samples", type=int, default=8192)
    ap.add_argument("--ref-dir", default=os.path.join("data", "reference"))
    ap.add_argument("--root", default="runs")
    ap.add_argument("--wandb", action="store_true", help="log to Weights & Biases")
    ap.add_argument("--quick", action="store_true",
                    help="300 steps and small evaluations, under runs/quick/: "
                         "checks the setup, says nothing about the results")
    a = ap.parse_args()
    if a.quick:
        a.n_steps, a.log_every, a.eval_every, a.checkpoint_every = 300, 50, 100, 100
        a.eval_samples, a.final_eval_samples, a.ref_samples = 256, 512, 512
        a.root = os.path.join(a.root, "quick")

    # paths stay relative to the repository root, where the commands run
    for out, cmd in jobs(a):
        if not a.run:
            print(shlex.join(["python", *cmd[1:]]))
            continue
        if os.path.exists(os.path.join(ROOT, out, "result.json")):
            print(f"[skip] {out}: finished")
            continue
        print(f"[run] {out}", flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
