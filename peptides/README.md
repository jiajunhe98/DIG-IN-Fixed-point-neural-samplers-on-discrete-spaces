# DIG-IN as a discretized Boltzmann generator: alanine dipeptide and tetrapeptide

This folder holds the code of the Boltzmann-generator experiment of the paper
(Section 5.4 and Figure 4; Appendices E.4 and F.2), and the absolute free
energies of Section 5.5. DIG-IN learns the equilibrium distribution of two
capped alanine peptides in implicit solvent **from the potential energy
alone**: the MD reference enters only the evaluation. It is a PyTorch package,
`digin`, independent of the `discrete_sampler` package of the repository root.

| system | | atoms | coordinates D = 3n − 6 | Ramachandran residues |
|---|---|---|---|---|
| `aldp` | Ace–Ala–Nme | 22 | 60 | 1 |
| `al4` | Ace–(Ala)₃–Nme | 42 | 120 | 3 |

Force field: amber99SB-ILDN with OBC implicit solvent at 300 K, no cutoff and
no constraints, reimplemented in batched PyTorch (it matches OpenMM per force
group to 1e-6 kJ/mol).

## The method

- **Discretization.** Every coordinate is confined to an interval with `L`
  uniform bins, so a configuration is a sequence of `D` bin indices. The
  discrete target is the cell mass of `exp(−U)`, where
  `U(z) = E(x(z))/kT − log|J(z)|`.
- **Sampler.** An any-order masked transformer: a draw fills one coordinate
  per network call, in a uniformly random order, and its probability
  `q(bins | order)` is exact. A uniform point inside each bin gives a
  continuous configuration, with importance weight
  `log w = −U(z) + log vol − log q(bins | order)`. That weight is exact for
  the continuous Boltzmann density on the box, so `log mean w` estimates its
  `log Z`.
- **Gibbs labels.** The label of coordinate `d` is the softmax of `−U` over
  its `L` bin centres, with every other coordinate held at the sample: one
  batched scan of `D × L` energies per configuration.
- **Fixed-point loop.**
  - Every 100 steps the EMA model draws 512 configurations into a FIFO
    buffer of 4096, with their labels and log-weights.
  - A gradient step masks a random subset of each buffered configuration and
    damps the label towards the EMA model, `η · label + (1 − η) · q_ema`.
  - It then minimizes the soft cross-entropy at the masked coordinates,
    weighted by `w^ν / Σ w^ν`, where `ν` is the largest exponent that keeps
    the batch ESS ≥ 0.8.

Two coordinate systems:

- **`internal`**: a Z-matrix (BFS tree of the bond graph, rooted at the
  central Cα), with `L = 64`.
  - Bonds and angles span ± 10 σ, with `σ = sqrt(kT/k)` from the force
    constant.
  - φ, ψ and the methyl rotors span the full circle.
  - ω spans trans ± 60°, and the branch dihedrals ± 45°. These windows fix
    L-chirality and trans peptide bonds.
- **`cartesian`**: the bond vector of every atom to its parent in the
  spanning tree, in a frame on the central residue (Cα at the origin, N on
  +x, C in the xy-plane), with `L = 512`.
  - The boxes come from a torsion sweep of the minimized structure, plus
    0.12 nm.
  - L-chirality is an energy penalty.
  - A single-coordinate label translates one subtree and cannot rotate half a
    molecule across a φ/ψ barrier. Every fresh draw therefore gets 1000
    Metropolis pivot moves about the φ/ψ bonds before it is labelled.
  - The pivot kernel is π-invariant, so a moved sample keeps the weight of
    its draw. Evaluated samples are always exact model draws.

| paper | code |
|---|---|
| peptides, force field, minimization | `digin/systems.py`, `digin/forcefield.py`, `data/` |
| Z-matrix, φ/ψ | `digin/molecule.py` |
| boxes, bins, Jacobians, Gibbs labels, pivot moves | `digin/targets.py`: `InternalTarget`, `CartesianTarget` |
| any-order transformer and its exact sampler | `digin/model.py` |
| fixed-point loop, evaluation, final evaluation | `digin/train.py` |
| 2M-sample sets with likelihoods and energies | `digin/sample.py` |
| reference MD (BAOAB + parallel tempering) | `digin/md.py` |
| metrics (ESS, energy W1, Ramachandran JS/KL, basins) | `digin/metrics.py` |
| tables and figure data of the paper | `digin/paper.py` |

## Installation

Python ≥ 3.10 and a CUDA build of PyTorch ≥ 2.1, then

```bash
cd peptides
pip install -r requirements.txt
pip install openmm pytest        # optional: rebuild data/, run the OpenMM comparison
python -m pytest tests -q
```

The tests check:
- the torch force field against OpenMM;
- the Z-matrix round trip and its Jacobian;
- the Gibbs labels against a brute-force scan;
- that the pivot moves leave the target invariant;
- a short training run.

`data/` holds the minimized structures and the exported force-field
parameters. `python -m digin.systems` rebuilds them and needs OpenMM; training
does not.

## Reproducing the paper

Every command below runs from this folder. A run needs one GPU; the paper used
one 80 GB H100, 141 GB H200 or 80 GB A100 per run.

**1. Reference MD** (evaluation only). Batched BAOAB Langevin dynamics with
parallel tempering between 300 and 600 K, on the same torch force field:
- 2 fs steps, with hydrogen mass repartitioning to 4 amu;
- one frame every 1 ps from the 300 K replicas.

```bash
python -m digin.md run --system aldp --n_temps 6 --copies 64 --max_ns 34 \
    --out runs/md_aldp --publish ref/aldp_ref.npz
python -m digin.md run --system al4  --n_temps 8 --copies 64 --burn_ps 500 --hours 12 \
    --out runs/md_al4 --publish ref/al4_ref.npz
```

The references of the paper have 2,159,296 frames for `aldp` and 2,275,584
for `al4`. Over 64 copies at 300 K that is about 34 and 36 ns per copy. The
`al4` command is the one used; for `aldp`, `--max_ns 34` reproduces the size
of the reference.

**2. Training.** 12 runs: 2 systems × 2 coordinate systems × seeds 0, 1, 2,
all with the defaults of `digin.train`. At the end of training each run draws
as many samples as the reference has frames, and logs `final/*`.

```bash
for s in aldp al4; do for c in internal cartesian; do for k in 0 1 2; do
  python -m digin.train --system $s --coords $c --seed $k \
      --out runs/${s}_${c}_s${k} --ref ref/${s}_ref.npz
done; done; done
```

- A run checkpoints every 1000 steps and resumes when relaunched with the
  same `--out`.
- W&B logging is on by default (project `digin-peptides`); `--no_wandb`
  turns it off.

**3. Samples for the evaluation.** 2M independent draws per run, with their
log-likelihoods and energies:

```bash
python -m digin.sample --run_dir runs/aldp_internal_s0 --out samples/aldp_internal_s0 --n 2000000
```

Each sample set holds `x`, `energy`, `log_q`, `log_q_x`, `log_w` and `rama` as
`.npy` files, plus `meta.json` and `topology.pdb`. `digin/sample.py`
documents the fields.

**4. Tables and figures.** These are the paper's `data/peptides/`: macros,
per-seed and basin tables, Ramachandran images and energy histograms.

```bash
python -m digin.paper --samples samples --ref_dir ref --out paper_data/peptides
```

**On an amlt (Azure ML Singularity) cluster**, `cluster/submit.py` submits the
same steps as jobs (`md`, `train`, `sample`, `paper`, `test`). Copy
`cluster/amlt.example.yaml` to `cluster/amlt.yaml` and fill in your cluster.

```bash
python cluster/submit.py train digin-v1 -d "ALDP/AL4 x internal/Cartesian x 3 seeds"
```

## Results

All four configurations share one setting, the defaults of `digin.train`:
- η = 0.5, target ESS 0.8, 150k steps, batch 256;
- AdamW at 3e-4, EMA 0.995;
- transformer with d_model 256, 6 layers and 8 heads.

The metrics below use 2M samples per run and the importance weights. They are
mean ± sd over seeds 0, 1, 2; JS is in 1e-3 and W1 in kJ/mol.

| system | coordinates | ESS | energy W1 | Ramachandran JS | log Z |
|---|---|---|---|---|---|
| aldp | internal | 0.503 ± 0.002 | 0.02 ± 0.01 | 0.18 ± 0.01 | −195.324 ± 0.001 |
| aldp | Cartesian | 0.103 ± 0.023 | 0.06 ± 0.02 | 0.69 ± 0.06 | −193.249 ± 0.000 |
| al4 | internal | 0.062 ± 0.004 | 0.06 ± 0.02 | 0.83 ± 0.05 | −446.141 ± 0.001 |
| al4 | Cartesian | < 1e-3 | 3.1 ± 1.4 | 91 ± 47 | −442.69 ± 0.15 |

- **log Z across coordinate systems.** A Cartesian box contains both cyclic
  orders of each methyl group's hydrogens, while an internal box keeps one.
  The two log Z must therefore differ by `m log 2`, for `m` methyl groups:
  - `aldp`: 2.075 against 3 log 2 = 2.079.
  - `al4`: 3.45 ± 0.15 against 5 log 2 = 3.47.
- **AL4 in Cartesian coordinates** is the open case. As drawn, every residue
  has about half the MD β/PPII population, and the weights are too
  heavy-tailed to correct it. Its reweighted numbers rest on a few samples.

Training time per run: about 4 h (`aldp` internal), 7 h (`aldp` Cartesian),
8 h (`al4` internal) and 15 h (`al4` Cartesian). The energy evaluations,
almost all of them Gibbs-label scans, were 3.0e9, 2.5e10, 6.0e9 and 5.0e10
per run, respectively.

### The pivot moves (Cartesian)

The Cartesian setting uses 1000 pivot moves per fresh draw. This number comes
from an earlier AL4 Cartesian run with 100 moves, which collapsed into αR
after about 30k steps.

Pivot moves were applied to 4,096 draws of that run's checkpoint. The table
gives the β/PPII fraction per residue; MD has 0.28 / 0.27 / 0.37.

| pivot moves | about φ/ψ and methyls | about φ/ψ only |
|---|---|---|
| 0 | 0.05 / 0.03 / 0.11 | 0.05 / 0.03 / 0.11 |
| 100 | 0.07 / 0.05 / 0.14 | 0.09 / 0.07 / 0.17 |
| 1,000 | 0.13 / 0.12 / 0.22 | 0.15 / 0.14 / 0.23 |
| 10,000 | 0.16 / 0.15 / 0.24 | 0.17 / 0.15 / 0.24 |

The moves change only torsions, so from a collapsed model they level off below
MD; the Gibbs labels are what correct the rest. Started from MD frames, the
same moves keep the MD basin fractions, which checks that the kernel is
invariant.
