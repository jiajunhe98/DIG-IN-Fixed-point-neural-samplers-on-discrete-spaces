# DIG-IN for maximum entropy reinforcement learning (JAX)

This folder holds the code of the reinforcement learning experiment of the
paper (Section 5.4 and Figure 4; Appendices E.4 and F.2): DIG-IN as the
policy of an actor-critic method on the 20 continuous control tasks of the
DeepMind Control suite in [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground).
It is written in JAX, with the environments simulated on the GPU by MJX, and
is independent of the PyTorch package in the rest of the repository.

## The method

The continuous actions `a ∈ [-1, 1]^m` are discretized into `L = 64` bins per
dimension, and the policy is a masked (any-order) model over the bins, the
sampler of DIG-IN. Its target at every state is the Boltzmann distribution of
the critic, `π(a | s) ∝ exp(Q(s, a) / τ)`.

- **Policy.** An MLP takes the state and the revealed dimensions and gives a
  categorical over the bins of every dimension. An action reveals the
  dimensions one per network call, in a uniformly random order, and is executed
  at the chosen bins plus uniform in-bin noise. The order is stored with the
  transition, so the likelihood, the entropy and the KL along it are exact.
- **Gibbs labels.** For a stored transition, the label of dimension `d` is
  `h^d(l | s, a^{-d}) = softmax_l(Q(s, a^{-d}, c_l) / τ)`, from `L` critic
  evaluations at the bin centres `c_l`, with the other dimensions at the
  stored action. In every epoch, each transition draws one step of its order
  and one dimension still masked at that step, and the policy minimizes the
  cross-entropy to its label at that context.
- **No importance weights, no damping.** Every state is visited once, so
  weights that are self-normalized per state are identically one (`ν = 0`).
  Instead of damping the labels, a trust region bounds each update,
  `KL(μ_old ‖ μ_θ) ≤ 0.1` nats per action, with a Lagrange multiplier `κ`.
- **Temperature.** `τ` is the Lagrange multiplier of the constraint that the
  entropy stays above `H* = -0.1 m` (in action-density units, as in FFM),
  capped at `τ = 1`.
- **Critic.** The categorical critic of FFM (151 atoms on [0, 150], HL-Gauss
  targets), trained on soft TD(λ) returns with the reward `r + γ τ H(s')`.
- **Loop.** Each iteration collects 1024 × 128 transitions with the current
  policy, fits the critic and then the policy for 4 epochs of 128 minibatches,
  for 420 iterations (55M environment steps). Every 20 iterations the policy
  is evaluated on 1024 episodes with greedy decoding: the dimensions are
  revealed in a random order, each at its most likely bin.

| paper | code |
|---|---|
| binned actions, masked policy, sampler | `dig_in_rl/policy.py`: `Binning`, `MaskedBinnedPolicy`, `sample_action` |
| Gibbs labels and their cross-entropy | `policy.py`: `gibbs_label_logits`, `actor_loss` |
| trust region (`κ`) and temperature (`τ`) | `policy.py`: `actor_loss`; config `trust_region`, `temperature` |
| critic | `dig_in_rl/critic.py` |
| environments | `dig_in_rl/envs.py` |
| actor-critic loop, evaluation | `dig_in_rl/train.py` |
| settings (App. E.4) | `configs/dmc.yaml` |

## Installation

Linux with an NVIDIA GPU (CUDA 12) and Python 3.10–3.12:

```bash
cd rl_jax
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest tests                 # about 5 minutes on a CPU
```

`requirements.txt` pins the JAX and MuJoCo versions of the runs in the paper
(JAX 0.5.2, Flax 0.10.6, MuJoCo and MJX 3.3.7, Brax 0.13.0) and the version of
MuJoCo Playground they used, a fork at a fixed commit. On a machine without a
GPU, install `jax==0.5.2` instead of `jax[cuda12]==0.5.2`; the tests and the
smoke test run on a CPU, the experiments need a GPU.

On its first import, MuJoCo Playground clones the MuJoCo Menagerie repository
into its package folder, which needs `git` and can take a while. The DMC tasks
use none of it; to skip the download, create the empty folder first:

```bash
mkdir -p "$(python -c 'import importlib.util, os; print(os.path.dirname(importlib.util.find_spec("mujoco_playground").origin))')/external_deps/mujoco_menagerie"
```

## Quick start

All commands run from `rl_jax/`.

```bash
# one run of the paper: CheetahRun, seed 0, 55M steps
python -m dig_in_rl.train env.name=CheetahRun seed=0

# a few iterations on a small batch, to check the setup (CPU or GPU)
python experiments/dmc.py --quick --run --tasks CartpoleBalance --seeds 0
```

A run logs one line per evaluation (the greedy return, the entropy, the
temperature, the KL of the last update) and writes `metrics.npz` to its
directory, by default `runs/dmc/<task>/s<seed>/`: every metric at each of the
21 evaluations, as arrays of shape `(21, num_seeds)`. The return the paper
reports is `eval/episode_return`. Any setting of `configs/dmc.yaml` can be
overridden on the command line, e.g. `eval_stochastic=true` to also evaluate
the stochastic policy.

## Reproducing the paper

Twenty tasks with ten seeds each, 200 runs. A run trains for 12–30 minutes on
an A100 and 35–70 minutes on a V100, longer for more action dimensions, after
about 3 minutes of compilation.

```bash
python experiments/dmc.py               # print the 200 commands, one per line
python experiments/dmc.py --run         # or run them here, one after another
python scripts/aggregate.py --compare   # final returns and IQM, next to the paper's
```

The launcher skips runs whose `metrics.npz` exists, so it can be repeated
after an interruption. `scripts/aggregate.py` computes what the paper reports
in Figure 4 and in the per-task table of App. F.2: the IQM (interquartile
mean) of the return divided by 1000 over all runs, with a 95% interval from a
stratified bootstrap over the seeds of each task, and the IQM over seeds of
the final return of every task. With `--compare` it prints the runs of the
paper on the same tasks and seeds next to yours, and `--paper` prints the
paper's results alone. These come from `paper/dmc_returns.csv`, the
evaluation returns of all 200 runs of the paper, from which the script
recomputes the numbers below exactly.

| task | m | final return | task | m | final return |
|---|---|---|---|---|---|
| AcrobotSwingup | 1 | 418 | FishSwim | 5 | 542 |
| AcrobotSwingupSparse | 1 | 113 | HopperHop | 4 | 277 |
| BallInCup | 2 | 975 | HopperStand | 4 | 943 |
| CartpoleBalance | 1 | 999 | PendulumSwingup | 1 | 838 |
| CartpoleBalanceSparse | 1 | 1000 | ReacherEasy | 2 | 981 |
| CartpoleSwingup | 1 | 867 | ReacherHard | 2 | 976 |
| CartpoleSwingupSparse | 1 | 789 | WalkerRun | 6 | 857 |
| CheetahRun | 6 | 893 | WalkerStand | 6 | 985 |
| FingerSpin | 2 | 938 | WalkerWalk | 6 | 974 |
| FingerTurnEasy | 2 | 963 | | | |
| FingerTurnHard | 2 | 956 | **IQM / 1000** | | **0.917 [0.908, 0.923]** |

The runs of the paper were made with the implementation this code was
extracted from. Given the same inputs and random keys, every part of the
two, the sampler, the Gibbs labels, the losses, their gradients and the
optimizer steps, gives the same result bit for bit on a CPU, and complete runs
from the same parameters and seed agree up to floating-point rounding. On a
GPU, XLA does not make runs bitwise reproducible, so single runs differ from
those of the paper, most on the sparse tasks, where a seed either solves the
task or does not. We checked the code as it is here, installed from
`requirements.txt` on fresh GPU nodes: seeds 0–2 of all 20 tasks give an IQM
of 0.909 [0.894, 0.924], against 0.897 [0.860, 0.921] for the paper's runs
with the same seeds, and ten seeds of PendulumSwingup give 837 (8 of 10 seeds
swing the pendulum up), against 838 (9 of 10) in the paper.

The other methods of Figure 4 are not part of this folder. The curves and
final values of REPPO, PPO, SPO, TRUDI, AMDP, DPPO, FPO and FFM are those
reported by the FFM paper, *Reinforcement Learning via Fixed-point Flow
Matching* (Blessing, Celik and Neumann, 2026).

## Layout

```
dig_in_rl/
  policy.py              binning, masked policy, sampler, Gibbs labels, and the
                         loss with its two dual variables
  critic.py              the HL-Gauss critic of FFM
  envs.py                the DMC tasks of MuJoCo Playground: batching, action
                         clipping, observation normalization, episode returns
  train.py               the actor-critic loop and its evaluation (hydra entry point)
configs/dmc.yaml         the settings of the paper
experiments/dmc.py       launcher of the 200 runs
scripts/aggregate.py     final returns and IQM with bootstrap intervals
paper/dmc_returns.csv    evaluation returns of the 200 runs of the paper
tests/                   policy, labels and fixed point (CPU); environment and trainer
```
