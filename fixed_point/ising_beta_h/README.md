# Fixed-point masked-diffusion sampler amortised over (beta, h), MDNS convention

One network for the whole (beta, h) box on a **50x50** torus, in the convention of the
package's Ising target. Trained **only from the energy**:

    log pi~(s) = beta * ( sum_<ij> s_i s_j + h sum_i s_i ),   beta in [0.25, 0.65],  |h| <= 0.10

No target samples, no MCMC seeding, no symmetry hints (flip augmentation off; h != 0 breaks it).
PT-in-beta references are diagnostics only.

## Design notes

* **Convention.** `energy_mdns.py` matches `dds.targets.ising.IsingModel` exactly: the field is
  multiplied by beta, and each bond is counted once.
* **50x50.** Dilated residual blocks (`dilations=[1,1,2,4,8,1,1]`) reach a lattice-spanning receptive
  field in 7 blocks, where plain 3x3 blocks would need 25.
* **Tempering exponent per cell** (`is_alpha_bins=1`). Each (beta, h) acquisition cell gets the
  largest `alpha` with ESS/n >= `ess_target`, falling back to the global one below `is_min_bin`
  entries.
* **PT reference** set by `ref_chains`, `ref_temps`, `gibbs_sweeps`, cached per cell in
  `ref_hpt_ising_L{L}_{sweeps}.pt` and shareable across runs via `ref_dir`.

## Run

    python energy_mdns.py cuda
    python train_h.py L=50 "dilations=[1,1,2,4,8,1,1]" beta_lo=0.390 beta_hi=0.650 \
        "beta_range=[0.250,0.650]" h_max=0.10 batch=1536 n_explore=1536 buffer_cap=49152 \
        eta=0.5 is_weight=1 ess_target=0.8 is_alpha_bins=1 acq=fisher run_name=ising-beta-h

The exact settings of the paper's runs are in `experiments/amortized.py` at the repository root.
Batch 1536 peaks at 61.5 GiB at L=50, so it needs a 141 GB card.

Files: `train_h.py` (trainer, buffer, acquisitions, per-cell IS, damping), `network_h.py`
(`FieldDenoiser` with dilations, `unmask_sample(..., return_logq=True)`), `energy_mdns.py` (energy in the
MDNS convention, PT-in-beta reference, exact energy), `ising_energy.py` / `ising_theory_stable.py`
(lattice energy and exact finite-L results).
