# Fixed-point masked-diffusion sampler for the q-state Potts model, amortised over (beta, h)

The amortised Ising trainer (`../ising_beta_h/`) ported from 2 states to q. Trained **only from
the energy**:

    log pi~(x) = beta * ( A(x) + h N_0(x) ),   A = agreeing bonds (each once),  N_0 = sites in state 0

with beta_c(q) = log(1 + sqrt q), so beta_c(3) = 1.00505. No target samples, no symmetry hints (no
S_q permutation augmentation). PT-in-beta references are diagnostics only.

## What is different from the amortised Ising sampler

* **q states.** `PottsFieldDenoiser` outputs q logits per site. The field couples to state 0 only, so
  h != 0 breaks S_q to S_{q-1} and the correct mode weights stop being 1/q.
* **Mode-resolving diagnostics.** The scalar order parameter is a radius and cannot show a missing
  mode, so each eval cell logs `mode_0 .. mode_{q-1}` separately, plus `mode_plane`: every sample in
  the simplex magnetisation plane with the q pure states marked, where a missing mode is a missing
  lobe. Raw lattices go through matplotlib with a discrete colour map, because `wandb.Image` of an
  integer token array renders solid black.
* **`compare_samplers.py`.** The chunked fixed-count unmasking against MD4 / MDLM ancestral sampling
  on the same trained denoiser at matched NFE: bias against the PT reference, ESS/n, and cost.
* **PT reference** cached per cell in `ref_hpt_potts_q{q}_L{L}_{sweeps}.pt`. The cache is keyed on
  (beta, h) only, so the file name carries the model and q; runs sharing a `ref_dir` cannot read
  each other's references.

## Run

    python train_hq.py L=24 q=3 dilations=None beta_lo=0.800 beta_hi=1.300 "beta_range=[0.700,1.400]" \
        h_max=0.01 batch=1536 n_explore=1024 buffer_cap=196608 eta=0.5 is_weight=1 ess_target=0.8 \
        is_alpha_bins=1 acq=fisher "eval_betas=[0.900,1.00505,1.200]" \
        "eval_hs=[-0.01,-0.002,0.0,0.002,0.01]" run_name=potts-beta-h

The exact settings of the paper's runs are in `experiments/amortized.py` at the repository root.
Batch 1536 at L=24 peaks near 14 GiB.

Files: `train_hq.py` (trainer, buffer, acquisitions, per-cell IS, damping), `network_hq.py`
(`PottsFieldDenoiser`, `unmask_sample`, `md4_sample`), `potts_field.py` (energy with field, PT-in-beta
reference, mode diagnostics, exact anchor), `compare_samplers.py` (sampler comparison at matched NFE).
