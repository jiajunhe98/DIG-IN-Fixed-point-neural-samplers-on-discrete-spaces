# DIG-IN: Fixed-point neural samplers on discrete spaces

Code for **Fixed-point neural samplers on discrete spaces**
([paper](https://arxiv.org/abs/2610.01739) ·
[project page](https://denisbless.github.io/DIG_IN_website/)).

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
| Discretized Boltzmann generators: alanine dipeptide and tetrapeptide, internal and Cartesian coordinates | `peptides/` (`digin.train`, `digin.sample`) | `peptides/digin/paper.py` |
| Free-energy differences of alanine dipeptide (ALDP-S, ALDP-T) with the bridge | `free_energy/` (`digin_bridge.train`) | `free_energy/digin_bridge/paper.py` |
| Maximum entropy RL on 20 DeepMind Control tasks (JAX) | `rl_jax/experiments/dmc.py` | `rl_jax/scripts/aggregate.py` |

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
peptides/                DIG-IN as a Boltzmann generator on binned molecular
                         coordinates (alanine peptides), with its own README
free_energy/             the DIG-IN bridge on FEAT's alanine dipeptide free-energy
                         tasks, with its own README
rl_jax/                  DIG-IN as a policy for maximum entropy RL, in JAX, with
                         its own requirements, launcher and README
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

### Discretized Boltzmann generators

Alanine dipeptide (D = 60) and tetrapeptide (D = 120) in implicit solvent,
trained from the energy alone. The coordinates are binned internal coordinates
(L = 64) or Cartesian bond vectors (L = 512); 2 systems × 2 coordinate systems
× 3 seeds make 12 runs. A long parallel-tempering MD reference enters only the
evaluation. The package `digin` lives in [`peptides/`](peptides/README.md),
which has its own installation, commands and results:

```bash
cd peptides
pip install -r requirements.txt
python -m digin.md run --system aldp --n_temps 6 --copies 64 --max_ns 34 --out runs/md_aldp --publish ref/aldp_ref.npz
python -m digin.train --system aldp --coords internal --seed 0 --out runs/aldp_internal_s0 --ref ref/aldp_ref.npz
python -m digin.sample --run_dir runs/aldp_internal_s0 --out samples/aldp_internal_s0 --n 2000000
python -m digin.paper --samples samples --ref_dir ref --out paper_data/peptides
```

### Free-energy estimation with the bridge

FEAT's ALDP-S task (vacuum to implicit solvent) and ALDP-T task (between two
conformational states), each with 3 seeds. The bridge carries MD samples of
one state to the other state, which is given only by its energy. The package
`digin_bridge` lives in [`free_energy/`](free_energy/README.md); FEAT's MD
data is downloaded on first use:

```bash
cd free_energy
pip install -r requirements.txt
python -m digin_bridge.train --task aldps --seed 0 --out runs/aldps_s0
python -m digin_bridge.paper --runs_root runs --out paper_data/freeenergy
```

### Maximum entropy reinforcement learning

DIG-IN as the policy of an actor-critic method on the 20 DeepMind Control
tasks of MuJoCo Playground, with the continuous actions discretized into bins
and Gibbs labels from a learned critic; ten seeds per task, 55M environment
steps each (200 runs). This experiment is written in JAX and lives in
[`rl_jax/`](rl_jax/README.md), with its own installation, launcher and
results script:

```bash
cd rl_jax
pip install -r requirements.txt
python experiments/dmc.py --run          # the 200 runs
python scripts/aggregate.py --compare    # final returns and IQM, next to the paper's
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
adapted from [DiffUCO](https://github.com/ml-jku/DiffUCO). The alanine
dipeptide MD data used by `free_energy/` is FEAT's (He et al., 2025), which is
downloaded from the Hugging Face dataset
[`JJHE/FEAT`](https://huggingface.co/datasets/JJHE/FEAT) on first use. Its
amber ff96 topology, `free_energy/digin_bridge/assets/alanine-dipeptide.prmtop`,
comes from openmmtools.

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
