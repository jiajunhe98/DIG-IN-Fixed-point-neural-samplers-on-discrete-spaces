"""Ising on an L x L torus with an external field, in the MDNS / discrete-ASBS convention,
amortised over (beta, h).

Convention
----------
Matches `dds.targets.ising.IsingModel` (Molei Tao's group's discrete-sampler repo) exactly:

    E(s) = -J * sum_<ij> s_i s_j  -  h * sum_i s_i ,   J = 1 ,   pi(s) ~ exp(-beta E(s))

so the unnormalised log-target is

    log pi~(s) = beta * ( sum_<ij> s_i s_j  +  h * sum_i s_i )

with each bond counted once, spins s = 2x - 1 for tokens x in {0,1}, and MASK = 2.
The critical point at h = 0 is beta_c = log(1 + sqrt(2)) / 2 = 0.440687.

Mapping from the old (sigma, h_old) parametrisation
---------------------------------------------------
The previous code used  log p = 2*sigma * sum ss + h_old * sum s.  Matching term by term:

    beta = 2 * sigma          sigma = beta / 2        sigma_c = 0.2204  <->  beta_c = 0.4407
    h    = h_old / beta       h_old = beta * h

**sigma -> beta is a pure rename, but h is NOT.** The old `h` had beta already absorbed into
it, so it is a field measured in units of temperature; this `h` is the physical field, and the
Boltzmann factor supplies the beta. Sweeping h_old over a fixed box is therefore a different
physical sweep from sweeping h over a fixed box, and the two only coincide at one beta. Old
checkpoints and old references are NOT loadable against this module without converting the
per-sample parameters.

Local conditional
-----------------
Flipping site i changes log pi~ by 2*beta*(m_i + h)*s_i with m_i the 4-neighbour spin sum, so

    rho_i(s_i = +1 | rest) = sigmoid( 2 * beta * (m_i + h) ) ,

which is one neighbour tally and needs no energy call and no normalising constant.
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

import ising_energy as ie
import ising_theory_stable as theory

MASK = 2
to_spin = ie.to_spin
neighbour_sum = ie.neighbour_sum
bond_sum = ie.bond_sum
magnetisation = ie.magnetisation


def beta_critical():
    """Exact zero-field critical point, log(1 + sqrt(2)) / 2."""
    return 0.5 * math.log(1.0 + math.sqrt(2.0))


def sigma_to_beta(sigma):
    """Old parametrisation -> this one. Pure rename of the coupling."""
    return 2.0 * sigma


def h_old_to_h(h_old, beta):
    """Old field -> physical field. NOT a rename: the old h absorbed beta."""
    return h_old / beta


def log_prob_unnormalised(x, beta, h, L):
    """log pi~(s) = beta * (sum_<ij> s_i s_j + h * sum_i s_i).  beta, h: float or [B].  -> [B]."""
    s_sum = to_spin(x).view(-1, L * L).sum(-1)
    b = beta if torch.is_tensor(beta) else torch.full_like(s_sum, float(beta))
    hh = h if torch.is_tensor(h) else torch.full_like(s_sum, float(h))
    return b * (bond_sum(x, L) + hh * s_sum)


def energy_per_spin(x, beta, h, L):
    """u = -log pi~(s) / L^2, the quantity the diagnostics compare.  [B].

    Note this carries beta, matching what the old code logged, so the numbers stay comparable
    once (sigma, h_old) are converted.  For the beta-free energy use `bare_energy_per_spin`.
    """
    return -log_prob_unnormalised(x, beta, h, L) / (L * L)


def bare_energy_per_spin(x, h, L):
    """E(s)/L^2 with no beta, i.e. exactly `IsingModel.energy` divided by the site count."""
    s_sum = to_spin(x).view(-1, L * L).sum(-1)
    hh = h if torch.is_tensor(h) else torch.full_like(s_sum, float(h))
    return -(bond_sum(x, L) + hh * s_sum) / (L * L)


def local_target_p1(x, beta, h, L):
    """rho_i(token = 1 | x^{-i}) = sigmoid(2 * beta * (m_i + h)).  [B, D]."""
    s = to_spin(x).view(-1, L, L)
    m = neighbour_sum(s)
    b = beta.view(-1, 1, 1) if torch.is_tensor(beta) else beta
    hh = h.view(-1, 1, 1) if torch.is_tensor(h) else h
    return torch.sigmoid(2.0 * b * (m + hh)).reshape(x.shape[0], L * L)


def bare_onsager_energy_per_spin(beta, L):
    """Exact finite-L zero-field <E>/L^2 with E = -sum_<ij> s_i s_j, no beta (Kaufman)."""
    return float(theory.calculate_ising_internal_energy(beta, L))


def local_logprob_of_current(x, beta, h, L):
    """Mean over sites of -log rho_i(x_i | x^{-i}): how far endpoints sit from the target's
    own local structure.  [B]."""
    p1 = local_target_p1(x, beta, h, L)
    p = torch.where(x > 0.5, p1, 1.0 - p1)
    return -torch.log(p.clamp_min(1e-12)).mean(-1)


def onsager_energy_per_spin(beta, L):
    """Exact finite-L zero-field energy per site in THIS convention, i.e. beta * <E>/L^2,
    directly comparable to `energy_per_spin` at h = 0."""
    return beta * bare_onsager_energy_per_spin(beta, L)


@torch.no_grad()
def pt_reference(beta_target, h, L, n, n_sweeps, device, seed=0, beta_min=0.16, n_temps=16):
    """Parallel tempering in beta at fixed h. The hot end is disordered and mixes the +/- modes,
    which blocked Gibbs cannot do below beta_c at small h. Returns tokens at beta_target."""
    g = torch.Generator(device=device).manual_seed(seed)
    betas = torch.linspace(beta_min, beta_target, n_temps, device=device)
    R = n_temps * n
    x = (torch.rand(R, L, L, device=device, generator=g) < 0.5).float()
    bcol = betas.repeat_interleave(n)[:, None, None]
    groups = ie._independent_groups(L, device).unsqueeze(0)
    for sweep in range(n_sweeps):
        for c in groups.unique():
            m = neighbour_sum(2.0 * x - 1.0)
            p1 = torch.sigmoid(2.0 * bcol * (m + h))
            new = (torch.rand(R, L, L, device=device, generator=g) < p1).float()
            x = torch.where(groups == c, new, x)
        sp = 2.0 * x - 1.0
        bond = (sp * torch.roll(sp, 1, 1) + sp * torch.roll(sp, 1, 2)).sum((-2, -1)).view(n_temps, n)
        field = sp.sum((-2, -1)).view(n_temps, n)
        xv = x.view(n_temps, n, L, L)
        for k in range(sweep % 2, n_temps - 1, 2):
            # log pi~ = beta*(bond + h*field); swapping k and k+1 changes it by
            # (beta_{k+1} - beta_k) * ((bond_k + h field_k) - (bond_{k+1} + h field_{k+1}))
            d = (betas[k + 1] - betas[k]) * ((bond[k] + h * field[k])
                                             - (bond[k + 1] + h * field[k + 1]))
            sw = torch.rand(n, device=device, generator=g) < torch.exp(d.clamp(max=0.0))
            swk = sw[:, None, None]
            a, b = xv[k].clone(), xv[k + 1].clone()
            xv[k] = torch.where(swk, b, a)
            xv[k + 1] = torch.where(swk, a, b)
        x = xv.reshape(R, L, L)
    return x.view(n_temps, n, L * L)[-1]


if __name__ == "__main__":
    dev = torch.device(sys.argv[1] if len(sys.argv) > 1 else "cpu")
    torch.manual_seed(0)
    L = 8
    print(f"device {dev}   beta_c = {beta_critical():.6f}  (sigma_c = {beta_critical()/2:.6f})")

    # 1. agreement with the OLD (sigma, h_old) module on identical physical targets
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ising_field"))
    x = (torch.rand(64, L * L, device=dev) < 0.5).float()
    worst_u = worst_p = 0.0
    for sigma in (0.10, 0.2204, 0.30):
        for h_old in (-0.05, 0.0, 0.03):
            beta = sigma_to_beta(sigma)
            h = h_old_to_h(h_old, beta)
            u_new = energy_per_spin(x, beta, h, L)
            u_old = -(2.0 * sigma * bond_sum(x, L) + h_old * to_spin(x).view(-1, L * L).sum(-1)) / (L * L)
            worst_u = max(worst_u, float((u_new - u_old).abs().max()))
            p_new = local_target_p1(x, beta, h, L)
            p_old = torch.sigmoid(2.0 * (2.0 * sigma * neighbour_sum(to_spin(x).view(-1, L, L))
                                         + h_old)).reshape(x.shape[0], L * L)
            worst_p = max(worst_p, float((p_new - p_old).abs().max()))
    print(f"[1] vs old (sigma,h_old) parametrisation: max |du| {worst_u:.2e}  "
          f"max |drho| {worst_p:.2e}  {'OK' if max(worst_u, worst_p) < 1e-5 else 'FAIL'}")

    # 2. local conditional equals brute-force single-site reweighting
    beta, h = 0.4407, 0.02
    p = local_target_p1(x, beta, h, L)
    worst = 0.0
    for i in (0, 7, 20, L * L - 1):
        lp = []
        for a in (0.0, 1.0):
            y = x.clone(); y[:, i] = a
            lp.append(log_prob_unnormalised(y, beta, h, L))
        brute = torch.softmax(torch.stack(lp, -1), -1)[:, 1]
        worst = max(worst, float((brute - p[:, i]).abs().max()))
    print(f"[2] local conditional vs brute force: max err {worst:.2e}  {'OK' if worst < 1e-5 else 'FAIL'}")

    # 3. zero-field PT reproduces the exact finite-L energy
    for beta in (0.30, beta_critical(), 0.60):
        xs = pt_reference(beta, 0.0, L, 256, 400, dev, seed=1, n_temps=12)
        u = float(energy_per_spin(xs, beta, 0.0, L).mean())
        ex = onsager_energy_per_spin(beta, L)
        print(f"[3] beta={beta:.4f}  PT u={u:+.4f}  exact={ex:+.4f}  diff {u - ex:+.4f}")

    # 4. the field selects the mode it should
    for h in (-0.05, 0.0, 0.05):
        xs = pt_reference(0.60, h, L, 256, 400, dev, seed=2, n_temps=12)
        print(f"[4] beta=0.60 h={h:+.3f}: mean spin {float(magnetisation(xs, L).mean()):+.4f}")
