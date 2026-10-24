"""Maximum entropy RL on the 20 DeepMind Control tasks (Sec. 5.4 of the paper).

    python experiments/dmc.py                              # print all 200 commands
    python experiments/dmc.py --run --tasks CheetahRun --seeds 0
    python experiments/dmc.py --quick --run --tasks CartpoleBalance --seeds 0   # smoke test

DIG-IN with the settings of App. E.4, ten seeds per task, 55M environment
steps each.  The paper trained every run on a single A100 or V100 GPU.

Each run goes to runs/dmc/<task>/s<seed>/; a run whose metrics.npz exists is
skipped.  Without --run the commands are printed, one per line, ready for a
job array; they expect to be run from rl_jax/.
"""

import argparse
import os
import shlex
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from dig_in_rl.envs import TASKS  # noqa: E402

# a few iterations of a small batch: checks the setup, says nothing about the results
QUICK = "total_steps=4096 num_envs=32 num_steps=32 minibatches=4 num_evals=2 env.episode_length=200"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", action="store_true",
                    help="run the jobs here, one after another (default: print them)")
    ap.add_argument("--tasks", nargs="+", default=list(TASKS), choices=list(TASKS))
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--root", default="runs")
    ap.add_argument("--quick", action="store_true",
                    help=f"under runs/quick/ with {QUICK}")
    a = ap.parse_args()
    root = os.path.join(a.root, "quick", "dmc") if a.quick else os.path.join(a.root, "dmc")

    for task in a.tasks:
        for seed in a.seeds:
            out = os.path.join(root, task, f"s{seed}").replace(os.sep, "/")
            args = f"env.name={task} seed={seed} hydra.run.dir={out}" + (f" {QUICK}" if a.quick else "")
            cmd = [sys.executable, "-m", "dig_in_rl.train", *args.split()]
            if not a.run:
                print(shlex.join(["python", *cmd[1:]]))
                continue
            if os.path.exists(os.path.join(ROOT, out, "metrics.npz")):
                print(f"[skip] {out}: finished")
                continue
            print(f"[run] {out}", flush=True)
            subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
