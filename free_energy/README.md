# Free energies of alanine dipeptide with the DIG-IN bridge

This folder holds the code of the free-energy-difference experiment of the
paper (Section 5.5 and Figure 5; Appendices E.5 and F.2). It runs the bridge
of Section 3.3 on FEAT's two alanine dipeptide (ALDP) tasks, in internal
coordinates. It is a PyTorch package, `digin_bridge`, independent of the
`discrete_sampler` package of the repository root.

- **Endpoint 0** is given only by MD samples.
- **Endpoint 1** is given only by its energy.
- A learned CTMC carries the endpoint-0 samples to endpoint 1. Its path
  weights estimate `log Z_1/Z_0`.

| `--task` | endpoint 0 (MD samples) | endpoint 1 (energy) | FEAT reference (MBAR) |
|---|---|---|---|
| `aldps` (ALDP-S) | vacuum | OBC1 implicit solvent | log Z_solv/Z_vac = 29.43 ± 0.01 |
| `aldpt` (ALDP-T) | vacuum, S_b: φ ∉ (0, 2.15) rad (C7eq, C5) | vacuum, S_a: φ ∈ (0, 2.15) rad (C7ax) | log Z_a/Z_b = −4.25 ± 0.05 |

## The method

**System.** openmmtools' AlanineDipeptideImplicit, as FEAT builds it: amber
ff96 with OBC1, no cutoff and no constraints, at 300 K.
- `λ` scales the GB charges: λ = 0 is vacuum, λ = 1 is solvent.
- The force field is reimplemented in batched PyTorch. It matches OpenMM per
  force group, at three λ, to 1e-6 kJ/mol.
- ALDP-T restricts both endpoints to their φ state, i.e. `U + ∞` outside it.

**Coordinates.** A Z-matrix rooted at Cα, D = 60, with L = 64 uniform bins per
coordinate.
- Bonds and angles span ± 10 σ.
- φ, ψ and the methyl rotors span the full circle.
- ω spans trans ± 60°, and the branch dihedrals ± 45°.
- 99.98 % of either MD set lies inside the box.

**Interpolant and labels.** Per coordinate, the interpolant is the mixture
`κ_t(c | x0, x1) = (1 − t) 1[c = x0] + t 1[c = x1]`, with independent
endpoints.
- An endpoint `X_1` of the current sampler has its coordinate `d` replaced by
  endpoint 1's Gibbs label `ρ_d(· | X_1^{-d})`, a 60 × 64 energy scan.
- That gives `ℓ = (1 − t) 1[c = X_0^d] + t ρ_d(c)` and
  `dℓ = ρ_d(c) − 1[c = X_0^d]`.
- For ALDP-T, the restriction to S_a only masks the label of φ.

**Network.** A hollow transformer: a forward and a backward causal stream of 4
layers each, and 2 readout blocks, so that output `d` never sees `x^d`.
- It has two zero-initialised heads, `q^d(c | x^{-d}, t)` and
  `φ^d(c | x^{-d}, t)`, with 11.1M parameters in total.
- `q` is fitted by soft cross-entropy and `φ` by the weighted quadratic loss.

**Sampler and weights.** The minimum-action CTMC, τ-leaped in K = 256 steps
from an MD frame:

| kernel | move | rate |
|---|---|---|
| forward | x^d → c | q^d(c) ([φ^d(c) − φ^d(x^d)]_+ + g) |
| backward | y^d → c | q^d(c) ([φ^d(y^d) − φ^d(c)]_+ + g) |

The path weight is `log w = −U_1(z_1) + U_0(z_0) + Σ_k [log B_k − log F_k]`,
and `E[w] = Z_1/Z_0` for any network.
- **Gibbs rate `g = 0.1`.** This term is a reversible heat-bath move under
  `q`. It is needed for unbiased weights: with `g = 0`, a downhill jump has
  zero forward rate while the backward kernel can still propose its reverse.
  On an enumerable toy, `g = 0` comes out 16 standard errors low.
- **ALDP-T endpoints outside S_a** get `log w = −∞`, which is exact. For the
  tempering exponent alone they count as the lowest finite `log w` minus 20.

**Loop.**
- Every 100 steps, the EMA model transports 1,024 training vacuum frames.
  Their endpoints, labels and path log-weights go into a buffer of 16,384.
- `ν` is chosen on the whole buffer so that ESS(w^ν) ≥ 0.5.
- A gradient step fits both heads at `X_t ~ K_t(· | X_0, X_1)`, with weights
  `w^ν`, damped towards the EMA model with η = 0.3.

| paper | code |
|---|---|
| FEAT's MD data (Hugging Face `JJHE/FEAT`) | `digin_bridge/data.py` |
| system, λ, torch force field | `digin_bridge/system.py`, `digin_bridge/forcefield.py`, `data/`, `digin_bridge/assets/` |
| Z-matrix, boxes, bins, Gibbs labels, ALDP-T states | `digin_bridge/zmatrix.py`, `digin_bridge/target.py` |
| interpolant, labels, CTMC sampler, path weights | `digin_bridge/bridge.py` |
| hollow transformer | `digin_bridge/model.py` |
| fixed-point loop, final evaluation, standard errors | `digin_bridge/train.py` |
| endpoint metrics (Ramachandran JS/KL, energy W1) | `digin_bridge/metrics.py` |
| tables and figure data of the paper | `digin_bridge/paper.py` |

## Installation

Python ≥ 3.10 and a CUDA build of PyTorch ≥ 2.1, then

```bash
cd free_energy
pip install -r requirements.txt
pip install openmm pytest        # optional: the OpenMM comparison, rebuilding data/
python -m pytest tests -q        # downloads FEAT's ALDP data (FEAT_DATA=<dir> to reuse a copy)
```

The tests check:
- the torch energy against OpenMM;
- that the network is hollow and the untrained bridge is the identity;
- that the path weights are unbiased with `g > 0` and biased without it, on an
  enumerable toy;
- the grouped standard error;
- the ALDP-T states and labels;
- a 20-step training run of each task with its final evaluation, whose saved
  paths must reproduce their energies, positions and weights.

## Reproducing the paper

Every command runs from this folder. FEAT's MD data (ALDP.zip, about 250k
vacuum and 210k solvent frames) is downloaded from Hugging Face on first use
into `data/feat_aldp/`, or into `--data_root`.

**1. Training.** 6 runs, the two tasks × seeds 0, 1, 2, all with the defaults
of `digin_bridge.train`:

```bash
for t in aldps aldpt; do for k in 0 1 2; do
  python -m digin_bridge.train --task $t --seed $k --out runs/${t}_s${k}
done; done
```

- Each run trains for 150k steps on one GPU: 5 h on an H200, 7 h on an H100,
  11 h on an A100.
- It then runs the **final evaluation**, one path per endpoint-0 MD frame:
  - 249,943 paths for ALDP-S and 246,191 for ALDP-T.
  - The paths start from the 10 % held-out frames in turn, about 10 paths per
    frame.
  - The paths are saved to `runs/<run>/final/`: `z`, `x`, `energy`, `log_w`,
    `log_w_endpoint`, `rama`, `start_frame` and `jumps` as `.npy`, with
    `meta.json` and `topology.pdb`.
  - `final/log_ratio_se` is the standard error with each starting frame as
    one unit.
- A run resumes from its checkpoint when relaunched with the same `--out`,
  and the final evaluation resumes part by part.
- W&B logging is on by default (project `digin-bridge`).

**2. Tables and figures.** These are the paper's `data/freeenergy/`: macros,
the per-seed table, convergence curves, endpoint Ramachandran images and
energy histograms.

```bash
python -m digin_bridge.paper --runs_root runs --out paper_data/freeenergy
```

**On an amlt (Azure ML Singularity) cluster**, `cluster/submit.py` submits the
same steps as jobs (`train`, `paper`, `test`). Copy `cluster/amlt.example.yaml`
to `cluster/amlt.yaml` and fill in your cluster.

## Results

The six runs use one setting:
- η = 0.3, target ESS 0.5, g = 0.1, K = 256, 150k steps.
- It was chosen from a 2 × 2 sweep, η ∈ {0.3, 1} × target ESS ∈ {0.3, 0.5},
  on seed 0. All four settings agreed within their late-training spread; this
  one had the smallest spread on ALDP-T.

| task | DIG-IN, 3 seeds (mean ± sd) | standard error per run | ESS | FEAT MBAR | umbrella + MBAR |
|---|---|---|---|---|---|
| ALDP-S | 29.436 ± 0.004 | 0.010 | 0.10 ± 0.01 | 29.43 ± 0.01 | |
| ALDP-T | −4.458 ± 0.018 | 0.011–0.029 | 0.05 ± 0.04 | −4.25 ± 0.05 | −4.44 ± 0.02 |

- **ALDP-S** agrees with FEAT's MBAR. BAR on the two MD sets with this
  package's energy gives 29.430.
- **ALDP-T** agrees with an independent umbrella-sampling check, not with
  FEAT's −4.25. See below.
- **Seeds share their start frames.** The held-out split is fixed (seed 1234)
  for every seed, so the spread over seeds leaves out the error from the
  choice of MD frames.

### The ALDP-T reference

FEAT's −4.25 was computed from per-state datasets that are not public. The
frame counts of the vacuum MD in FEAT's archive give log(3752 / 246191) =
−4.18, but that MD switches state rarely.

The independent check is umbrella sampling along φ: 40 windows of 2 ns each,
combined by MBAR. It gives −4.44 ± 0.02, in line with this bridge.
- That check is not fully converged in ψ near the C7eq/C7ax barrier.
- The two time halves of one set of windows give −4.25 and −5.00.
- The umbrella code is not part of this folder.
