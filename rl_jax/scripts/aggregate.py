"""Final returns and the IQM over tasks and seeds, as reported in the paper.

    python scripts/aggregate.py                      # runs/dmc/<task>/s<seed>/metrics.npz
    python scripts/aggregate.py --paper              # the runs of the paper (paper/dmc_returns.csv)
    python scripts/aggregate.py --compare            # these runs next to the paper's on the same seeds
    python scripts/aggregate.py --curve curve.csv    # also write the IQM curve

The IQM (interquartile mean, the mean of the middle 50% of the returns) is
taken over all runs of all tasks, with returns divided by 1000, and its 95%
interval comes from a stratified bootstrap that resamples the seeds within
each task (Agarwal et al., 2021).  The per-task value is the IQM over seeds
of the final return.  The paper reports the last evaluation, at 52.4M steps:
IQM 0.917 [0.908, 0.923] over 20 tasks x 10 seeds.
"""

import argparse
import csv
import glob
import os
from collections import defaultdict

import numpy as np
from scipy.stats import trim_mean

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(ROOT, "paper", "dmc_returns.csv")


def iqm(x, axis=0):
    return trim_mean(x, 0.25, axis=axis)


def boot_iqm(per_task, reps=2000, seed=0):
    """IQM over all runs and its stratified 95% bootstrap interval; per_task:
    one (seeds, evaluations) array per task."""
    pooled = np.concatenate(per_task, 0)
    rng = np.random.default_rng(seed)
    bs = np.empty((reps, pooled.shape[1]))
    for b in range(reps):
        bs[b] = iqm(np.concatenate([c[rng.integers(0, len(c), len(c))] for c in per_task], 0))
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    return iqm(pooled), lo, hi


def load_runs(root):
    """{task: {seed: (steps, returns)}} from <root>/<task>/s<seed>/metrics.npz;
    a run with num_seeds > 1 gives the seeds "<seed>:<j>"."""
    runs = defaultdict(dict)
    for path in sorted(glob.glob(os.path.join(root, "*", "s*", "metrics.npz"))):
        parts = os.path.normpath(path).split(os.sep)
        task, seed = parts[-3], int(parts[-2][1:])
        z = np.load(path)
        ret, steps = np.asarray(z["eval/episode_return"]), np.asarray(z["time_step"])
        for j in range(ret.shape[1]):
            runs[task][seed if ret.shape[1] == 1 else f"{seed}:{j}"] = (steps[:, j], ret[:, j])
    return dict(runs)


def load_paper(path=PAPER):
    rows = defaultdict(lambda: defaultdict(list))
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows[r["task"]][int(r["seed"])].append((float(r["step"]), float(r["return"])))
    return {t: {s: tuple(np.array(v).T) for s, v in sorted(seeds.items())} for t, seeds in rows.items()}


def summary(runs):
    tasks = sorted(runs)
    n_eval = min(len(r[1]) for t in tasks for r in runs[t].values())
    per_task = [np.stack([r[1][:n_eval] for r in runs[t].values()]) / 1000.0 for t in tasks]
    steps = next(iter(runs[tasks[0]].values()))[0][:n_eval]
    y, lo, hi = boot_iqm(per_task)
    return dict(steps=steps, iqm=y, lo=lo, hi=hi,
                finals={t: 1000.0 * iqm(c[:, -1]) for t, c in zip(tasks, per_task)},
                seeds={t: len(c) for t, c in zip(tasks, per_task)})


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", default=os.path.join(ROOT, "runs", "dmc"))
    ap.add_argument("--paper", action="store_true", help="the runs of the paper instead")
    ap.add_argument("--compare", action="store_true",
                    help="also the paper's runs of the same tasks and seeds")
    ap.add_argument("--curve", default=None, help="write step, iqm, lo, hi to this csv")
    a = ap.parse_args()

    runs = load_paper() if a.paper else load_runs(a.runs)
    if not runs:
        raise SystemExit(f"no runs under {a.runs}")
    ours, ref = summary(runs), None
    if a.compare:
        paper = load_paper()
        same = {t: {s: paper[t][s] for s in seeds if s in paper.get(t, {})} for t, seeds in runs.items()}
        same = {t: v for t, v in same.items() if v}
        ref = summary(same) if same else None

    print(f"{'task':<24}{'seeds':>6}{'final':>8}" + (f"{'paper':>8}{'diff':>7}" if ref else ""))
    for t, v in ours["finals"].items():
        line = f"{t:<24}{ours['seeds'][t]:>6}{v:>8.0f}"
        if ref and t in ref["finals"]:
            line += f"{ref['finals'][t]:>8.0f}{v - ref['finals'][t]:>+7.0f}"
        print(line)
    print(f"\nIQM at {ours['steps'][-1] / 1e6:.1f}M steps, {len(ours['finals'])} tasks: "
          f"{ours['iqm'][-1]:.3f} [{ours['lo'][-1]:.3f}, {ours['hi'][-1]:.3f}]")
    if ref:
        print(f"the paper on the same tasks and seeds:  "
              f"{ref['iqm'][-1]:.3f} [{ref['lo'][-1]:.3f}, {ref['hi'][-1]:.3f}]")
        if ref["steps"][-1] != ours["steps"][-1]:
            print(f"note: the paper's values are at {ref['steps'][-1] / 1e6:.1f}M steps, these runs end earlier")
    if a.curve:
        with open(a.curve, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "iqm", "lo", "hi"])
            w.writerows(zip(ours["steps"], ours["iqm"], ours["lo"], ours["hi"]))


if __name__ == "__main__":
    main()
