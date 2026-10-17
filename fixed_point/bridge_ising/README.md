# Fixed-point transport between two Ising temperatures, with a hollow two-head network

Transport `pi_0 = Ising(beta 0.20)` to `pi_1 = Ising(beta 0.60)` on a 25×25 torus.
Samples from the prior are given; the target is known **only through its energy**.

Self-contained: no source file outside this directory is imported. Third-party requirements are torch, numpy, matplotlib and wandb (`wandb_mode=disabled` to run without an account).

## Convention

The target is the MDNS / discrete-ASBS Ising model, matching `discrete_sampler/targets/ising`:

```
E(s) = -J sum_<ij> s_i s_j - h sum_i s_i,  J = 1,  pi(s) ~ exp(-beta E(s))
log pi~(s) = beta * ( sum_<ij> s_i s_j + h sum_i s_i ),   beta_c = log(1+sqrt 2)/2 = 0.440687
```

Earlier versions of this bridge used a `sigma` parametrisation with
`log p = 2 sigma sum ss + h_old sum s`. The coupling is a pure rename, `beta = 2 sigma`, so
the old `sigma 0.10 -> 0.30` run is this `beta 0.20 -> 0.60` run and every measured number
below carries over unchanged. The **field** does not rename: the old `h` absorbed beta, so
`h = h_old / beta`. Old checkpoints and cached references are not loadable against this module
without converting their per-sample parameters. The bridge runs at `h = 0`, where the two
agree. `python ising_energy.py cuda` checks the equivalence numerically (0.00e+00) along with
the local conditionals and the reference sampler.

The transport straddles the transition: `beta_0 = 0.20` is disordered, `beta_c = 0.4407`,
`beta_1 = 0.60` is ordered with `|m| = 0.974`.

## The interpolant

Per coordinate, a mixture kernel with `alpha(t) = t`:

```
kappa_t(c | x0, x1) = (1-t) 1[c=x0] + t 1[c=x1]
X_t^d ~ kappa_t^d(. | X_0^d, X_1^d)   independently per coordinate
```

**Relation to `discrete_sampler/methods/dig_in`.** The per-coordinate kernel is the same functional form as
that package's mixture interpolant at `omega = 0`, `N = 2`, but this is not one of its four
variants, for two reasons:

* **The prior is not uniform and not masked.** `discrete_sampler/methods/dig_in` states it explicitly --
  "the prior is uniform, the coupling independent: `x0 ~ U`, `x1 ~ nu`" -- and its other
  option is the absorbing mask. Here `X_0` is drawn from a *different physical temperature*,
  `Ising(beta 0.20)`. That is what makes this a bridge between two Boltzmann distributions
  rather than a generative path from noise, and it is why the prior bank is built by MCMC.
* **The network parameterises `(q, phi)`, not `(v, vb)`.** `discrete_sampler/methods/dig_in` learns forward
  and backward velocity rates simulated with a uniform mobility `Q = 1/N` and a `zeta_t`
  noise level. Here the two heads are a conditional and a time-derivative ratio, from which
  the generator below follows in closed form.

What is shared with that package is the fixed-point idea: the endpoint comes from the model
and the labels come from the target's exact local energy, so no target samples are used.

## Fixed point, not regression onto data

`X_1` is drawn from the **current model**, never from `pi_1`. Coordinate `d`'s label is then
corrected by the target's exact single-site conditional given the rest of the endpoint,
`rho_d(a | X_1^{-d}) = sigmoid(2 beta_1 m_d)`:

```
kappa_bar^d(c)     = sum_a rho_d(a | X_1^{-d}) kappa_t(c | X_0^d, a)
d_t kappa_bar^d(c) = sum_a rho_d(a | X_1^{-d}) d_t kappa_t(c | X_0^d, a)
```

`pi_1` is a fixed point of this iteration. The substitution at coordinate `d` is only
consistent because the head never sees `X_t^d` and `X_t^{-d}` does not depend on `X_1^d` --
which is what the architecture below buys. Labels cost one local energy pass.

## The network: blind spot, two heads

`network_bs.py`. Output at site `d` depends on `X_t^{-d}` **only**. Construction follows
Laine et al. 2019 (arXiv:1901.10277): a shared half-plane branch whose receptive field is
strictly above the current row, applied to the four 90° rotations of the input; rotating back
and concatenating gives rows above ∪ rows below ∪ cols left ∪ cols right = everything but the
centre.

On a torus the branch rolls the input down one row and makes each conv causal upward, so after
`n` causal convs it sees row offsets `1..(1+2n)` above. Strictly above needs `1+2n < L`, which
is a constructor assertion, and circular padding is then safe because offsets never reach
`0 mod L`.

Two things are absent **because they would break the blind spot with no other symptom**:

* any normalisation whose statistics mix positions (GroupNorm, BatchNorm) -- only a
  per-position channel norm is used;
* any full-plane convolution after the branch -- the head is 1×1 only.

Both heads are hollow, per site, over the two candidate values:

| head | meaning |
|---|---|
| `q^d(c \| x^{-d}, t)` | logits; soft-label CE against `kappa` → `P_t(X_t^d = c \| X_t^{-d})` |
| `phi^d(c \| x^{-d}, t)` | all-candidate weighted quadratic (no-division form) → `E[d_t kappa \| x^{-d}] / E[kappa \| x^{-d}]` |

`test_blind_spot.py` proves it numerically: flipping input site `d` leaves both heads at `d`
unchanged to 0.00e+00 while changing them elsewhere by ~0.7-1.0, at three lattice sizes, with
randomised (not zero-initialised) weights so the test has teeth.

## The sampler

The minimum-action CTMC implied by the two heads:

```
R_t(x -> x^{d<-c}) = q^d(c | x^{-d}) [ phi^d(c | x^{-d}) - phi^d(x^d | x^{-d}) ]_+
```

satisfies the continuity equation exactly, using `sum_c q^d(c) phi^d(c) = 0`, which holds
because `sum_c d_t kappa = 0`. Simulated by tau-leaping from `X_0 ~ pi_0` at `t = 0` to `t = 1`.

## Measured

**Predictor-corrector is Gibbs sweeps in disguise** (`pc_sample.py`, two plateaued
checkpoints). A corrector is heat-bath sweeps from the score head, colour-group by colour-group
(9 groups on the odd-L torus). At `t = 1` that head **is** the target Gibbs conditional:
`|dq| ~ 1e-3` against exact, and PT samples are invariant under 64 sweeps of it.

| sweeps | predictor + corrector | same budget at `t=1` only | plain Gibbs from the prior |
|---|---|---|---|
| 0 | +0.18 / +0.16 | — | — |
| 256 | +0.024 / +0.015 | same | +0.037 |
| 1024 | +0.002 / +0.004 | same | +0.001 |

So the transport endpoint is worth **~50-60 sweeps** of head start (100-300 mid-way), about
10% of what Gibbs needs from scratch at L=25. Exact sweeps cost no network evaluation; learned
ones cost 9 passes.

**Correctors at `t < 1` amplify mode bias.** They sample the *learned* bridge conditionals, so
they resolve ambiguous regions toward whatever the buffer already favours: one arm went
`<m> +0.15 → +0.32` with 4 sweeps/step, another `+0.02 → +0.18`. Exact sweeps at `t = 1`
preserve balance; plain Gibbs is balanced by symmetry. **The energy histogram cannot see any of
this** -- all of these arms have the same energy.

**Inside training it compounds.** One learned sweep after every tau-leap step with `t >= 0.75`,
in the buffer sampler only, took the bare predictor `+0.177 → +0.063` (25 rounds) `→ +0.035`
(50) `→ +0.013-0.016` (250+), i.e. to the importance-weighted level with no weights. Collection cost
36 s → 119 s per round. Same caveat: a parent at `<m> +0.18` drifted to `+0.45`; a balanced
parent stayed balanced.

## Files

| file | contents |
|---|---|
| `network_bs.py` | the hollow two-head blind-spot CNN, plus `hollow_check` |
| `train_bridge.py` | the fixed-point loop: replay buffer, EMA sampler, function-space damping, tempered path weights, PT references, resume |
| `pc_sample.py` | predictor-corrector sampling and the Gibbs-budget comparison above |
| `ising_energy.py` | the MDNS-convention target: energy, exact local conditionals, blocked Gibbs, PT-in-beta reference, Kaufman/Yang exact values, and a self-test |
| `ising_theory_stable.py` | exact finite-L Ising free energy and internal energy |
| `test_blind_spot.py` | numerical proof of the blind spot |

```bash
python test_blind_spot.py cuda        # architecture check
python ising_energy.py cuda           # target self-test
python train_bridge.py                # L=25, beta 0.20 -> 0.60
```
