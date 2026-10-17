"""2D Ising model on an L x L torus (periodic boundary conditions).

Conventions are pinned to `LatticeIsingModel` in ising_models.py (verified numerically):

    log p_unnorm(s) = s^T J s + b^T s ,   J = sigma * A(torus),  b = 0
                    = 2 * sigma * sum_<ij> s_i s_j

so with  beta := 2 * sigma  the target energy of the sampler note is

    U(s) = -beta * sum_<ij> s_i s_j ,        pi(s) = e^{-U(s)} / Z .

Tokens are 0/1, spins are s = 2*x - 1 in {-1,+1}.  The mask token is 2.
Critical point: beta_c = log(1+sqrt(2))/2 = 0.44069  ->  sigma_c = 0.22034.

Everything here is exact and local: the single-site Gibbs conditional needed by the
fixed-point masked-diffusion sampler is one 4-neighbour sum, no full energy call.
"""

import numpy as np
import torch

import ising_theory_stable as theory

MASK = 2  # mask token id (vocab is {0, 1, MASK})


def to_spin(x):
    """tokens {0,1} [B,L,L] or [B,D] -> spins {-1,+1} float, same shape."""
    return 2.0 * x.float() - 1.0


def neighbour_sum(s):
    """s: [B,L,L] spins -> sum of the 4 torus neighbours at each site, [B,L,L]."""
    return (
        torch.roll(s, 1, 1) + torch.roll(s, -1, 1) + torch.roll(s, 1, 2) + torch.roll(s, -1, 2)
    )


def bond_sum(x, L):
    """sum_<ij> s_i s_j over the L*L torus (each bond counted once). x: [B,D] tokens -> [B]."""
    s = to_spin(x).view(-1, L, L)
    return (s * torch.roll(s, 1, 1)).sum((1, 2)) + (s * torch.roll(s, 1, 2)).sum((1, 2))


def energy(x, sigma, L):
    """U(x) = -beta * sum_<ij> s_i s_j with beta = 2*sigma.  x: [B,D] tokens, sigma: float or [B]."""
    beta = 2.0 * sigma
    if torch.is_tensor(beta):
        beta = beta.view(-1)
    return -beta * bond_sum(x, L)


def energy_per_spin(x, L):
    """Onsager-convention internal energy per spin, u = -sum_<ij> s_i s_j / L^2.  [B]"""
    return -bond_sum(x, L) / (L * L)


def local_target_p1(x, sigma, L):
    """Exact single-site Gibbs conditional rho_i(token=1 | x^{-i}), shape [B,D].

    rho_i(s=+1) = softmax_a(-U(x^{i<-a})) = sigmoid(2 * beta * m_i),  m_i = sum of neighbour spins.
    sigma: float or [B] tensor.
    """
    s = to_spin(x).view(-1, L, L)
    m = neighbour_sum(s)
    beta = 2.0 * sigma
    if torch.is_tensor(beta):
        beta = beta.view(-1, 1, 1)
    return torch.sigmoid(2.0 * beta * m).reshape(x.shape[0], L * L)


def local_logprob_of_current(x, sigma, L):
    """-log rho_i(x^i | x^{-i}) averaged over sites: 'endpoint local inconsistency'.  [B]"""
    p1 = local_target_p1(x, sigma, L)
    p_cur = torch.where(x.view(p1.shape) > 0.5, p1, 1.0 - p1)
    return -torch.log(p_cur.clamp_min(1e-12)).mean(-1)


def magnetisation(x, L):
    """mean spin per configuration, [B]."""
    return to_spin(x).view(-1, L * L).mean(-1)


# ---------------------------------------------------------------------------
# reference sampler / exact theory (diagnostics only -- never used for training)
# ---------------------------------------------------------------------------

def _independent_groups(L, device):
    """Site colouring such that no two sites of the same colour are torus neighbours.

    Plain (i+j)%2 checkerboard is only valid for even L; for odd L the wrap-around bond
    (i,L-1)--(i,0) joins two sites of equal parity.  Splitting off the last row/column
    fixes that and costs a few extra passes.
    """
    ii, jj = torch.meshgrid(torch.arange(L, device=device), torch.arange(L, device=device), indexing="ij")
    if L % 2 == 0:
        return (ii + jj) % 2          # even L: the torus is bipartite, 2 colours suffice
    return (ii % 2) + 2 * (jj % 2) + 4 * (ii == L - 1) + 8 * (jj == L - 1)


@torch.no_grad()
def gibbs_reference(sigma, L, n, n_sweeps, device, seed=0):
    """Blocked (checkerboard-style) Gibbs sampler for pi.  Reference observables only."""
    g = torch.Generator(device=device).manual_seed(seed)
    x = (torch.rand(n, L, L, device=device, generator=g) < 0.5).float()
    groups = _independent_groups(L, device).unsqueeze(0)
    colours = groups.unique()
    beta = 2.0 * sigma
    for _ in range(n_sweeps):
        for c in colours:
            m = neighbour_sum(2.0 * x - 1.0)
            p1 = torch.sigmoid(2.0 * beta * m)
            new = (torch.rand(n, L, L, device=device, generator=g) < p1).float()
            x = torch.where(groups == c, new, x)
    return x.reshape(n, L * L)


def onsager_energy_per_spin(sigma, L):
    """Exact finite-L internal energy per spin, u = -<sum_<ij> s_i s_j>/L^2 (Kaufman)."""
    return float(theory.calculate_ising_internal_energy(2.0 * sigma, L))


def yang_spontaneous_magnetisation(sigma):
    """Exact |m| in the thermodynamic limit (Yang 1952): (1 - sinh^-4(2 beta))^(1/8), 0 above T_c.

    Useful at L=50 where the Gibbs reference for |m| needs very many sweeps to equilibrate
    (the MCMC has to coarsen too).  Finite-L corrections are small deep in the ordered phase
    and large near sigma_c, so this complements the Gibbs reference rather than replacing it.
    """
    beta = 2.0 * sigma
    k = np.sinh(2.0 * beta) ** -4
    return float((1.0 - k) ** 0.125) if k < 1.0 else 0.0


def onsager_free_energy_per_spin(sigma, L):
    beta = 2.0 * sigma
    return float(theory.calculate_ising_free_energy(beta, L))
