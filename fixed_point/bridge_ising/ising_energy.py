"""2D Ising model on an L x L torus, in the MDNS / discrete-ASBS convention.

    E(s) = -J * sum_<ij> s_i s_j  -  h * sum_i s_i ,   J = 1 ,   pi(s) ~ exp(-beta E(s))

so the unnormalised log-target is

    log pi~(s) = beta * ( sum_<ij> s_i s_j  +  h * sum_i s_i )

with each bond counted once, tokens 0/1, spins s = 2x - 1 in {-1,+1}.  The zero-field
critical point is beta_c = log(1 + sqrt(2)) / 2 = 0.440687.

This replaces the older `sigma` parametrisation (log p = 2 sigma sum ss + h_old sum s) that
earlier versions of this bridge used.  The coupling is a pure rename,

    beta = 2 * sigma          sigma = beta / 2          sigma_c 0.2204 <-> beta_c 0.4407

but the FIELD is not: the old h absorbed beta, so h = h_old / beta.  Old checkpoints and old
cached references are therefore not loadable against this module without converting their
per-sample parameters.  The bridge itself runs at h = 0 throughout, where the two agree.

Everything the sampler needs is exact and local: the single-site conditional is one
4-neighbour tally,

    rho_i(s_i = +1 | rest) = sigmoid( 2 * beta * (m_i + h) ) ,

no full energy call and no normalising constant.
"""

import math

import numpy as np
import torch

import ising_theory_stable as theory

MASK = 2  # mask token id, for callers that use a masked alphabet {0, 1, MASK}


def beta_critical():
    """Exact zero-field critical point, log(1 + sqrt(2)) / 2."""
    return 0.5 * math.log(1.0 + math.sqrt(2.0))


def sigma_to_beta(sigma):
    """Old parametrisation -> this one.  A pure rename of the coupling."""
    return 2.0 * sigma


def h_old_to_h(h_old, beta):
    """Old field -> physical field.  NOT a rename: the old h absorbed beta."""
    return h_old / beta


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------

def to_spin(x):
    """tokens {0,1} [B,L,L] or [B,D] -> spins {-1,+1} float, same shape."""
    return 2.0 * x.float() - 1.0


def neighbour_sum(s):
    """s: [B,L,L] spins -> sum of the 4 torus neighbours at each site, [B,L,L]."""
    return (torch.roll(s, 1, 1) + torch.roll(s, -1, 1)
            + torch.roll(s, 1, 2) + torch.roll(s, -1, 2))


def bond_sum(x, L):
    """sum_<ij> s_i s_j over the torus, each bond once.  x: [B,D] tokens -> [B]."""
    s = to_spin(x).view(-1, L, L)
    return (s * torch.roll(s, 1, 1)).sum((1, 2)) + (s * torch.roll(s, 1, 2)).sum((1, 2))


def field_sum(x, L):
    """sum_i s_i.  x: [B,D] -> [B]."""
    return to_spin(x).view(-1, L * L).sum(-1)


def magnetisation(x, L):
    """mean spin per configuration, [B]."""
    return to_spin(x).view(-1, L * L).mean(-1)


# ---------------------------------------------------------------------------
# target
# ---------------------------------------------------------------------------

def log_prob_unnormalised(x, beta, h, L):
    """log pi~(s) = beta * (sum_<ij> s_i s_j + h sum_i s_i).  beta, h float or [B].  -> [B]."""
    b = bond_sum(x, L)
    bb = beta.view(-1) if torch.is_tensor(beta) else beta
    hh = h.view(-1) if torch.is_tensor(h) else h
    return bb * (b + hh * field_sum(x, L))


def energy(x, beta, h, L):
    """U(s) = -log pi~(s)."""
    return -log_prob_unnormalised(x, beta, h, L)


def energy_per_spin(x, L):
    """The BARE internal energy per spin, u = -sum_<ij> s_i s_j / L^2, with no beta and no field.

    This is the quantity the bridge reports and the one `onsager_energy_per_spin` returns, so
    the two are directly comparable.  Use `log_prob_unnormalised` for anything that needs the
    actual target weight.
    """
    return -bond_sum(x, L) / (L * L)


def local_target_p1(x, beta, h, L):
    """Exact single-site conditional rho_i(token = 1 | x^{-i}) = sigmoid(2 beta (m_i + h)).  [B,D].

    beta, h: float or [B] tensor.
    """
    s = to_spin(x).view(-1, L, L)
    m = neighbour_sum(s)
    b = beta.view(-1, 1, 1) if torch.is_tensor(beta) else beta
    hh = h.view(-1, 1, 1) if torch.is_tensor(h) else h
    return torch.sigmoid(2.0 * b * (m + hh)).reshape(x.shape[0], L * L)


def local_logprob_of_current(x, beta, h, L):
    """Mean over sites of -log rho_i(x^i | x^{-i}): the endpoint local inconsistency.  [B]."""
    p1 = local_target_p1(x, beta, h, L)
    p_cur = torch.where(x.view(p1.shape) > 0.5, p1, 1.0 - p1)
    return -torch.log(p_cur.clamp_min(1e-12)).mean(-1)


# ---------------------------------------------------------------------------
# exact theory and reference samplers (diagnostics only -- never used for training)
# ---------------------------------------------------------------------------

def onsager_energy_per_spin(beta, L):
    """Exact finite-L zero-field bare energy per spin, -<sum_<ij> s_i s_j>/L^2 (Kaufman)."""
    return float(theory.calculate_ising_internal_energy(beta, L))


def onsager_free_energy_per_spin(beta, L):
    return float(theory.calculate_ising_free_energy(beta, L))


def yang_spontaneous_magnetisation(beta):
    """Exact |m| in the thermodynamic limit (Yang 1952), 0 above the transition.

    Useful at large L where a Gibbs reference for |m| needs very many sweeps to equilibrate,
    since the chain has to coarsen too.  Finite-L corrections are small deep in the ordered
    phase and large near beta_c, so it complements the PT reference rather than replacing it.
    """
    k = np.sinh(2.0 * beta) ** -4
    return float((1.0 - k) ** 0.125) if k < 1.0 else 0.0


def _independent_groups(L, device):
    """Site colouring such that no two sites of one colour are torus neighbours.

    The (i+j)%2 checkerboard is only bipartite for even L; for odd L the wrap-around bond joins
    two sites of equal parity, so the last row and column are split off.
    """
    ii, jj = torch.meshgrid(torch.arange(L, device=device), torch.arange(L, device=device),
                            indexing="ij")
    if L % 2 == 0:
        return (ii + jj) % 2
    return (ii % 2) + 2 * (jj % 2) + 4 * (ii == L - 1) + 8 * (jj == L - 1)


@torch.no_grad()
def gibbs_reference(beta, h, L, n, n_sweeps, device, seed=0):
    """Blocked heat-bath Gibbs sampler for pi.  Reference observables only.

    Adequate for the disordered prior.  NOT adequate for the ordered target: from a random
    quench it leaves a tail of domain-wall states at m ~ 0 after thousands of sweeps, which
    biases exactly the mode statistics this study measures.  Use `pt_reference` there.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    x = (torch.rand(n, L, L, device=device, generator=g) < 0.5).float()
    groups = _independent_groups(L, device).unsqueeze(0)
    colours = groups.unique()
    for _ in range(n_sweeps):
        for c in colours:
            m = neighbour_sum(2.0 * x - 1.0)
            p1 = torch.sigmoid(2.0 * beta * (m + h))
            new = (torch.rand(n, L, L, device=device, generator=g) < p1).float()
            x = torch.where(groups == c, new, x)
    return x.reshape(n, L * L)


@torch.no_grad()
def pt_reference(beta_target, h, L, n, n_sweeps, device, seed=0, beta_min=0.16, n_temps=16):
    """Parallel tempering in beta at fixed h.  Returns tokens at beta_target.

    The hot end is disordered and mixes the +/- modes, which blocked Gibbs cannot do below
    beta_c, so this is the reference to use for anything that reads mode weights.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    betas = torch.linspace(beta_min, beta_target, n_temps, device=device)
    R = n_temps * n
    x = (torch.rand(R, L, L, device=device, generator=g) < 0.5).float()
    bcol = betas.repeat_interleave(n)[:, None, None]
    groups = _independent_groups(L, device).unsqueeze(0)
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
    import sys
    dev = torch.device(sys.argv[1] if len(sys.argv) > 1 else "cpu")
    torch.manual_seed(0)
    L = 8
    print(f"device {dev}   beta_c = {beta_critical():.6f}  (old sigma_c {beta_critical()/2:.6f})")

    # 1. agreement with the OLD sigma parametrisation on identical physical targets
    x = (torch.rand(64, L * L, device=dev) < 0.5).float()
    worst_lp = worst_p = 0.0
    for sigma in (0.10, 0.2204, 0.30):
        beta = sigma_to_beta(sigma)
        lp_new = log_prob_unnormalised(x, beta, 0.0, L)
        lp_old = 2.0 * sigma * bond_sum(x, L)
        worst_lp = max(worst_lp, float((lp_new - lp_old).abs().max()))
        p_new = local_target_p1(x, beta, 0.0, L)
        p_old = torch.sigmoid(4.0 * sigma * neighbour_sum(to_spin(x).view(-1, L, L))
                              ).reshape(x.shape[0], L * L)
        worst_p = max(worst_p, float((p_new - p_old).abs().max()))
    print(f"[1] vs the old sigma parametrisation: max |dlogpi| {worst_lp:.2e}  "
          f"max |drho| {worst_p:.2e}  {'OK' if max(worst_lp, worst_p) < 1e-4 else 'FAIL'}")

    # 2. local conditional equals brute-force single-site reweighting
    beta, h = beta_critical(), 0.05
    p = local_target_p1(x, beta, h, L)
    worst = 0.0
    for i in (0, 7, 20, L * L - 1):
        lp = []
        for a in (0.0, 1.0):
            y = x.clone(); y[:, i] = a
            lp.append(log_prob_unnormalised(y, beta, h, L))
        brute = torch.softmax(torch.stack(lp, -1), -1)[:, 1]
        worst = max(worst, float((brute - p[:, i]).abs().max()))
    print(f"[2] local conditional vs brute force: max err {worst:.2e}  "
          f"{'OK' if worst < 1e-5 else 'FAIL'}")

    # 3. the PT reference reproduces the exact finite-L energy, and keeps both modes
    for beta in (0.20, beta_critical(), 0.60):
        xs = pt_reference(beta, 0.0, L, 256, 400, dev, seed=1, n_temps=12)
        u = float(energy_per_spin(xs, L).mean())
        m = magnetisation(xs, L)
        print(f"[3] beta={beta:.4f}  PT u={u:+.4f}  exact={onsager_energy_per_spin(beta, L):+.4f}  "
              f"frac(+)={float((m > 0).float().mean()):.3f}  |m|={float(m.abs().mean()):.3f}  "
              f"(Yang |m| {yang_spontaneous_magnetisation(beta):.3f})")
