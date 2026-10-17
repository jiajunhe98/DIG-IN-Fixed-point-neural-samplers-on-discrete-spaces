# DIG-IN: Fixed-point neural samplers on discrete spaces

Code for **Fixed-point neural samplers on discrete spaces**
([arXiv:2610.01739](https://arxiv.org/abs/2610.01739)).

DIG-IN (Discrete Gibbs Iterative Neural sampler) learns to sample a discrete,
unnormalized distribution `pi(x) ∝ p~(x)` on `{0, …, q-1}^D` from the energy
alone: no samples of the target are ever used. It is a masked-diffusion
sampler trained by a fixed-point iteration:

1. the current model draws configurations into a replay buffer;
2. each buffered configuration is labelled with the target's exact
   single-site conditionals (its Gibbs labels), computed from local energy
   differences;
3. the denoiser is trained by cross-entropy onto these labels at freshly
   masked copies, with the labels damped towards the previous model and each
   configuration importance-weighted (weights tempered to a target ESS);
4. the updated model becomes the next sampler.

The target is an exact fixed point of this map. Every draw comes with an exact
proposal density, so the sampler also gives importance weights and an estimate
of `ln Z`.

This repository reproduces every experiment in the paper:

| experiment | launcher | tables and figure data |
|---|---|---|
| Ising 25×25 and Potts 16×16 across the phase transition, against MDNS, PDNS, DASBS and LEAPS | `experiments/lattice.py` | `scripts/make_lattice_tex_data.py` |
| Amortized sampling over (β, h): Ising 50×50, Potts 24×24 | `experiments/amortized.py` | `scripts/make_fp_tex_data.py` |
| Alloy phase diagrams (Ag–Pd, Cu–Au), against FrOGS | `experiments/alloy.py` | `scripts/make_alloy_tex_data.py` |
| Component ablation (appendix) | `experiments/lattice.py --ablation` | `scripts/make_abl_tex_data.py` |
| Bridge between two Ising temperatures (appendix) | `experiments/bridge.py` | `scripts/make_bridge_tex_data.py` |

## Installation

Python ≥ 3.10 and PyTorch ≥ 2.0 (install the CUDA build that matches your
driver first, see [pytorch.org](https://pytorch.org/get-started/locally/)), then

```bash
pip install -e ".[dev]"
pytest                     # 5-10 minutes on a CPU
```

The tests check, mostly by exhaustive enumeration on small systems, that the
identities the method rests on hold to machine precision (masked interpolant,
label construction, unbiasedness of the `ln Z` estimators, the reference
MCMC, exact `ln Z` of the targets), and that every method trains and evaluates
on Ising, Potts and the alloys.

## Repository layout

```
discrete_sampler/        the sampler package
  methods/dig_in/        DIG-IN: masked interpolant, fixed-point loop, sampler
                         (also runs MDNS and PDNS, see below)
  methods/asbs/          DASBS, the discrete adjoint Schrodinger bridge sampler
  methods/leaps/         LEAPS
  methods/frogs/         FrOGS (alloys)
  models/                periodic residual CNN backbone, PixelCNN prior (FrOGS)
  targets/               Ising, Potts (energies, exact ln Z, reference MCMC),
                         alloy cluster expansions with their packaged exact
                         reference (targets/alloy/data/)
  train.py               training entry point (hydra)
  evaluation.py          ESS, ln Z, observables, comparison with reference samples
  eval_alloy.py          per-state-point evaluation and phase diagrams
configs/                 hydra configs: method/ model/ target/
fixed_point/             standalone trainers of the amortized (beta, h) samplers
                         and of the temperature bridge, plus the Swendsen-Wang
                         ground-truth sampler
experiments/             one launcher per experiment, with the paper's settings
scripts/                 scoring, reference generation, and the scripts that turn
                         runs into the paper's tables (make_*_tex_data.py)
tests/
```

## Quick start

A single run is one call of the training entry point; the method, target and
model are hydra config groups:

```bash
# a 4x4 Ising sanity check, about 5 minutes on a CPU: exact ln Z by
# enumeration, so the final |log_Z_err| should be around 0.01
python -m discrete_sampler.train target=ising_small n_steps=1000 \
    eval_every=500 model.hidden_size=32 model.n_blocks=2

# DIG-IN's method settings on 25x25 Ising at the critical point (GPU); the
# paper's step budget and evaluation come from experiments/lattice.py
python -m discrete_sampler.train target=ising25_critical method=dig_in \
    method.eta=0.1 method.weighting=tempered method.target_ess=0.8
```

Each run directory holds `history.jsonl` (one row per evaluation: ESS, `ln Z`
error, observables against the reference, wall time and energy calls so far),
`result.json` (the final evaluation), `samples.pt` (the final draws and their
log-weights) and a checkpoint. A run resumes from its checkpoint when it is
relaunched with the same `hydra.run.dir`; without one, runs go to
`runs/single/<timestamp>_<method>_<target>/`.

To check on a GPU that a paper configuration runs end to end before launching
the full sweep (300 steps, small evaluations, under `runs/quick/`):

```bash
python experiments/lattice.py --quick --run --tasks ising25 --methods digin \
    --phases critical --seeds 0
```

## Reproducing the paper

Every launcher prints its commands, one per line, so they can be sent to a job
array on a cluster. With `--run` it executes them here, one after another,
skipping runs that have finished. All commands are run from the repository
root, and everything they write goes to `runs/`. In the paper, every lattice
and alloy model was trained on a single 80 GB H100 GPU.

### Ising and Potts across the phase transition

25×25 Ising (D = 625) and 16×16 three-state Potts (D = 256) at a disordered,
a critical and an ordered temperature; six samplers, three seeds, 150k
gradient steps each (108 runs). Training takes 6–12 h per run on an H100 for
Ising and 2–6 h for Potts, depending on the method; the whole sweep with its
evaluations took about 620 GPU-hours.

```bash
python experiments/lattice.py              # print all 108 commands
python experiments/lattice.py --run        # or run them here
python scripts/make_lattice_tex_data.py    # the results and cost tables -> results/lattice/
```

The reference MCMC samples each run is compared with (8192 configurations,
Swendsen-Wang and Metropolis) are generated on first use and cached in
`data/reference/`.

MDNS and PDNS are run inside this codebase, with the same interpolant,
backbone, sampler, buffer and optimizer as DIG-IN, and differ from it in
exactly the ingredients the paper compares:

| | labels | importance weights | damping |
|---|---|---|---|
| DIG-IN | Gibbs (soft) | tempered to ESS 0.8 | η = 0.1 |
| PDNS | endpoint token (hard) | tempered to ESS 0.8 | none |
| MDNS | endpoint token (hard) | full self-normalized | none |

### Component ablation (appendix)

Four arms on 25×25 Ising, each removing one ingredient of DIG-IN; the
baseline is the `digin` runs of the sweep above (36 runs, about 6.5 h each).

```bash
python experiments/lattice.py --ablation --run
python scripts/make_abl_tex_data.py        # -> results/ablation/
```

### Amortized sampling over (β, h)

One network per seed for the whole box, 50×50 Ising with β ∈ [0.39, 0.65],
|h| ≤ 0.1, and 24×24 Potts with β ∈ [0.8, 1.3], |h| ≤ 0.01, trained with the
standalone trainers in `fixed_point/`. At the paper's batch size of 1536 the
Ising runs need a GPU with more than 80 GB of memory.

```bash
python experiments/amortized.py train --run        # three Ising and two Potts seeds
python experiments/amortized.py score --run        # 5000 draws at each cell of a 9x9 grid
python experiments/amortized.py reference --run    # reference ln Z and mode distributions (CPU)
python experiments/amortized.py groundtruth --run  # optional check, Swendsen-Wang samples (CPU)
python scripts/make_fp_tex_data.py                 # -> results/amortized/
```

The reference combines Kaufman's exact finite-lattice `ln Z(β, 0)` with
Bennett acceptance ratios along Swendsen-Wang ladders in `h` (Ising), and a
β-ladder from the exact β = 0 point (Potts).

### Alloy phase diagrams

Ag–Pd (D = 125) and Cu–Au (D = 128) cluster expansions in the semi-grand
canonical ensemble, one model per seed amortized over the whole
(T, Δμ) plane, against FrOGS. The phase diagram is built by the common-tangent
construction on the free-energy curves the model's weighted draws give, and is
compared with the exact phase diagram of each system, which is packaged in
`discrete_sampler/targets/alloy/data/`.

Training takes about 6 h per DIG-IN run and 4 h per FrOGS run on an H100.

```bash
python experiments/alloy.py train --run     # 60k steps per run
python experiments/alloy.py sweep --run     # 5000 draws at every packaged state point
python experiments/alloy.py tangent --run   # the phase diagram of each run
python scripts/make_alloy_tex_data.py       # -> results/alloy/
```

The sweep stage is split into shards (31 for Ag–Pd, 14 for Cu–Au) that can
run in parallel; the merge command after them stitches the shards back
together.

### Bridge between two Ising temperatures (appendix)

Transport from β = 0.20 to 0.60 (across the transition) and from 0.50 to 0.60
(between two ordered distributions) on a 25×25 lattice; neither endpoint is a
tractable reference distribution. The free-energy difference is known exactly,
so the `ln Z` error is measured against ground truth.

```bash
python experiments/bridge.py train --run
python experiments/bridge.py score --run
python experiments/amortized.py groundtruth --run   # endpoint samples, if not done above
python scripts/make_bridge_tex_data.py             # -> results/bridge/
```

### Preemptible clusters

`discrete_sampler.train` checkpoints into its run directory (two alternating
slots with a checksum, so a kill during a write cannot corrupt the last good
checkpoint), handles `SIGTERM` by checkpointing at the next step, and resumes
when it is relaunched with the same `hydra.run.dir`.
`scripts/run_resumable.sh --out <run dir> <overrides>` wraps a run in a retry
loop that gives up only after repeated attempts without progress.

## Data

The alloy energies are the cluster expansions of Damewood et al.
(npj Comput. Mater. 8, 61, 2022), packaged with their exact fixed-composition
`ln Z_n(T)` and phase diagrams in `discrete_sampler/targets/alloy/data/`. The
exact finite-lattice Ising free energy in `fixed_point/*/ising_theory_stable.py` is
adapted from [DiffUCO](https://github.com/ml-jku/DiffUCO).

## Citation

```bibtex
@article{he2026fixedpoint,
  title   = {Fixed-point neural samplers on discrete spaces},
  author  = {He, Jiajun and Blessing, Denis and Cheng, Mouyang
             and Du, Yuanqi and Domingo-Enrich, Carles},
  journal = {arXiv preprint arXiv:2610.01739},
  year    = {2026}
}
```
