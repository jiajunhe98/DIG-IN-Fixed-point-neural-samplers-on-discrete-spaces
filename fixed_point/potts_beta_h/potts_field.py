"""q-state Potts on an L x L torus with a symmetry-breaking field, amortised over (beta, h).

Convention
----------
Extends the MDNS / discrete-ASBS Potts target (see `fixed_point_potts/potts_energy.py`) with a
field that couples to ONE state:

    E(x) = -sum_<ij> 1[x_i == x_j]  -  h * sum_i 1[x_i == 0]
    pi(x) ~ exp(-beta E(x)),   so   log pi~(x) = beta * ( A(x) + h * N_0(x) )

with A(x) the agreeing nearest-neighbour bonds (each counted once) and N_0(x) the number of
sites in state 0.  h = 0 recovers the plain Potts model, and beta_c(q) = log(1 + sqrt(q)).

Why this field
--------------
It is the Potts analogue of the Ising external field, and it plays the same role in the study.
At h = 0 the q ordered states are exactly degenerate and the sampler has to discover the 1/q
balance for itself; at h != 0 state 0 is favoured (h > 0) or suppressed (h < 0) and the
correct mode weights are no longer equal, so the sampler must track a moving target rather
than a symmetric one.  Flip/permutation augmentation is a cheat for exactly this reason.

Scale of h
----------
Two pure states differ in log pi~ by beta * h * L^2, so at L = 32, q = 3, beta = 1.005 a field
of h = 0.01 is worth about 10 nats and already decides the mode weights, while h = 0.001 is
about 1 nat and sits in the crossover.  Choose the sampling box accordingly: |h| <= 0.05 is
saturated at the ends with the interesting structure near |h| ~ 0.002 to 0.02.

Local conditional
-----------------
Setting site i to state a changes log pi~ by beta * (n_a(i) + h * 1[a == 0]) with n_a the
count of the four torus neighbours in state a, so

    rho_i(a | x^{-i}) = softmax_a( beta * (n_a(i) + h * 1[a == 0]) ) ,

one neighbour tally, no energy call, no normalising constant.
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F


def mask_token(q):
    return q


def beta_critical(q):
    """Exact zero-field transition point of the ferromagnetic square-lattice Potts model."""
    return math.log(1.0 + math.sqrt(q))


def u_critical(q):
    """Exact zero-field internal energy per site at beta_c (thermodynamic limit)."""
    return -(1.0 + 1.0 / math.sqrt(q))


# ---------------------------------------------------------------------------
# energy and the exact local conditionals
# ---------------------------------------------------------------------------

def agree_bonds(x, L):
    """A(x), agreeing nearest-neighbour bonds on the torus.  [B,D] -> [B] float."""
    g = x.view(-1, L, L)
    right = (g == torch.roll(g, 1, 2)).sum((1, 2))
    down = (g == torch.roll(g, 1, 1)).sum((1, 2))
    return (right + down).float()


def field_count(x, L):
    """N_0(x), the number of sites in state 0.  [B,D] -> [B] float."""
    return (x.view(x.shape[0], -1) == 0).sum(-1).float()


def log_prob_unnormalised(x, beta, h, L):
    """log pi~(x) = beta * (A(x) + h * N_0(x)).  beta,h float or [B].  -> [B]."""
    a = agree_bonds(x, L)
    n0 = field_count(x, L)
    b = beta if torch.is_tensor(beta) else torch.full_like(a, float(beta))
    hh = h if torch.is_tensor(h) else torch.full_like(a, float(h))
    return b * (a + hh * n0)


def energy_per_spin(x, beta, h, L):
    """u = -log pi~(x) / L^2, carrying beta, directly comparable across the run.  [B]."""
    return -log_prob_unnormalised(x, beta, h, L) / (L * L)


def bare_energy_per_spin(x, h, L):
    """E(x)/L^2 with no beta: -(A + h N_0)/L^2."""
    return -(agree_bonds(x, L) + (h if torch.is_tensor(h) else float(h)) * field_count(x, L)) / (L * L)


def neighbour_counts(x, L, q):
    """n_a(i), how many of the four torus neighbours of site i are in state a.  [B,L,L,q]."""
    oh = F.one_hot(x.view(-1, L, L), q).to(torch.float32)
    return (torch.roll(oh, 1, 1) + torch.roll(oh, -1, 1)
            + torch.roll(oh, 1, 2) + torch.roll(oh, -1, 2))


def field_vector(h, q, device, dtype=torch.float32):
    """The per-state field offset, h on state 0 and 0 elsewhere.  h scalar or [B] -> [...,q]."""
    e0 = torch.zeros(q, device=device, dtype=dtype)
    e0[0] = 1.0
    if torch.is_tensor(h):
        return h.view(-1, 1, 1, 1).to(dtype) * e0
    return float(h) * e0


def local_target_p(x, beta, h, L, q):
    """rho_i(a | x^{-i}) = softmax_a(beta * (n_a(i) + h * 1[a==0])).  [B, D, q]."""
    n = neighbour_counts(x, L, q)                                  # [B,L,L,q]
    b = beta.view(-1, 1, 1, 1) if torch.is_tensor(beta) else beta
    n = n + field_vector(h, q, x.device, n.dtype)
    return torch.softmax(b * n, dim=-1).reshape(x.shape[0], L * L, q)


def local_energy_residual(x, p, L, q):
    """n_{x_i}(i) - sum_a rho_i(a) n_a(i), per site.  [B, D].

    The Potts analogue of the Ising `s_i * m_i` test channel: the local energy of the realised
    state minus its conditional mean.  Zero in expectation at the fixed point, and non-zero for
    a model that is anywhere off it, including a uniform one.
    """
    n = neighbour_counts(x, L, q).reshape(x.shape[0], L * L, q)
    n_cur = n.gather(-1, x.view(x.shape[0], -1, 1)).squeeze(-1)
    return n_cur - (p * n).sum(-1)


def local_logprob_of_current(x, beta, h, L, q):
    """Mean over sites of -log rho_i(x_i | x^{-i}).  [B]."""
    p = local_target_p(x, beta, h, L, q)
    p_cur = p.gather(-1, x.view(x.shape[0], -1, 1)).squeeze(-1)
    return -torch.log(p_cur.clamp_min(1e-12)).mean(-1)


# ---------------------------------------------------------------------------
# order parameters and mode diagnostics
# ---------------------------------------------------------------------------

def state_fractions(x, L, q):
    return F.one_hot(x.view(x.shape[0], -1), q).to(torch.float32).mean(1)


def order_parameter(x, L, q):
    """m = (q f_max - 1)/(q - 1).  A MAGNITUDE: blind to which state won, so blind to collapse."""
    f = state_fractions(x, L, q)
    return (q * f.max(-1).values - 1.0) / (q - 1.0)


def simplex_vertices(q, device=None, dtype=torch.float32):
    """Regular-simplex directions in R^(q-1), canonically oriented (state 0 on the first axis)."""
    c = torch.eye(q, dtype=torch.float64) - 1.0 / q
    _, evecs = torch.linalg.eigh(c)
    b = evecs[:, 1:]
    b = b / b.norm(dim=1, keepdim=True)
    v = b[0]
    a = torch.cat([v[:, None], torch.eye(q - 1, dtype=torch.float64)], dim=1)
    rot, _ = torch.linalg.qr(a)
    rot[:, 0] = rot[:, 0] * torch.sign(rot[:, 0] @ v)
    b = b @ rot
    if q >= 3 and b[1, 1] < 0:
        b[:, 1] = -b[:, 1]
    return b.to(dtype=dtype, device=device)


def magnetisation_vector(x, L, q):
    """M = (1/D) sum_i e_{x_i} in R^(q-1); the CV that resolves the q modes."""
    return simplex_vertices(q, x.device)[x.view(x.shape[0], -1)].mean(1)


def majority_state(x, L, q):
    return state_fractions(x, L, q).argmax(-1)


def mode_fractions(x, L, q):
    """Batch fraction whose majority state is a.  [q].  1/q balanced, 1 on one entry collapsed.

    With a field this is NOT expected to sit at 1/q: h > 0 should push mass onto state 0.  The
    reference gives the correct value at each (beta, h), and the diagnostic is agreement with
    it, not flatness.
    """
    maj = majority_state(x, L, q)
    return torch.bincount(maj, minlength=q).float() / maj.numel()


def mode_fraction_0(x, L, q):
    """Fraction of the batch whose majority state is 0, the state the field couples to.  scalar.

    The direct analogue of the Ising `frac_plus`, and the quantity whose crossover in h is the
    thing this amortised run is measuring.
    """
    return float((majority_state(x, L, q) == 0).float().mean())


def exact_anchor(beta, h, q):
    """The one closed-form check available: beta * u_c(q) at h = 0, beta = beta_c(q).

    Returns None elsewhere.  Unlike Ising there is no Kaufman formula for finite L and no
    exact energy away from the critical point, so every other cell is checked against the
    PT reference only.  Note u_c is a thermodynamic-limit value; a finite lattice sits below
    it by an O(1/L) amount, so treat it as an anchor rather than a tolerance.
    """
    if abs(h) > 1e-12 or abs(beta - beta_critical(q)) > 5e-3:
        return None
    return beta * u_critical(q)


# ---------------------------------------------------------------------------
# reference sampler
# ---------------------------------------------------------------------------

def _independent_groups(L, device):
    ii, jj = torch.meshgrid(torch.arange(L, device=device), torch.arange(L, device=device),
                            indexing="ij")
    if L % 2 == 0:
        return (ii + jj) % 2
    return (ii % 2) + 2 * (jj % 2) + 4 * (ii == L - 1) + 8 * (jj == L - 1)


@torch.no_grad()
def pt_reference(beta_target, h, L, q, n, n_sweeps, device, seed=0, beta_min=0.30, n_temps=16):
    """Parallel tempering in beta at fixed h, returning tokens at beta_target.

    Swendsen-Wang would be cheaper at h = 0, but the field breaks the cluster construction
    (it needs the ghost-spin trick), and PT in beta is the same machinery the Ising side uses.
    The hot end is disordered and mixes the q modes, which blocked Gibbs cannot do below
    beta_c: a random quench there leaves domain walls that bias exactly the mode statistics
    this run is measuring.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    betas = torch.linspace(beta_min, beta_target, n_temps, device=device)
    R = n_temps * n
    x = torch.randint(0, q, (R, L, L), device=device, generator=g)
    bcol = betas.repeat_interleave(n).view(-1, 1, 1, 1)
    groups = _independent_groups(L, device).unsqueeze(0)
    e0 = torch.zeros(q, device=device); e0[0] = 1.0
    for sweep in range(n_sweeps):
        for c in groups.unique():
            nb = neighbour_counts(x.reshape(R, -1), L, q) + float(h) * e0
            p = torch.softmax(bcol * nb, dim=-1)
            u = torch.rand(R, L, L, 1, device=device, generator=g)
            new = (p.cumsum(-1) < u).sum(-1).clamp_max(q - 1)
            x = torch.where(groups == c, new, x)
        flat = x.reshape(R, -1)
        lp = (agree_bonds(flat, L) + float(h) * field_count(flat, L)).view(n_temps, n)
        xv = x.view(n_temps, n, L, L)
        for k in range(sweep % 2, n_temps - 1, 2):
            d = (betas[k + 1] - betas[k]) * (lp[k] - lp[k + 1])
            sw = torch.rand(n, device=device, generator=g) < torch.exp(d.clamp(max=0.0))
            swk = sw[:, None, None]
            a_, b_ = xv[k].clone(), xv[k + 1].clone()
            xv[k] = torch.where(swk, b_, a_)
            xv[k + 1] = torch.where(swk, a_, b_)
        x = xv.reshape(R, L, L)
    return x.view(n_temps, n, L * L)[-1]


if __name__ == "__main__":
    dev = torch.device(sys.argv[1] if len(sys.argv) > 1 else "cpu")
    torch.manual_seed(0)
    q, L = 3, 8
    print(f"device {dev}   beta_c(q={q}) = {beta_critical(q):.5f}   u_c = {u_critical(q):+.5f}")

    # 1. h = 0 must reproduce the plain Potts target exactly
    x = torch.randint(0, q, (64, L * L), device=dev)
    lp0 = log_prob_unnormalised(x, 1.005, 0.0, L)
    ref = 1.005 * agree_bonds(x, L)
    e1 = float((lp0 - ref).abs().max())
    print(f"[1] h=0 reduces to plain Potts: max err {e1:.2e}  {'OK' if e1 < 1e-4 else 'FAIL'}")

    # 2. local conditional == brute-force single-site reweighting, with the field on
    beta, h = 1.005, 0.03
    p = local_target_p(x, beta, h, L, q)
    worst = 0.0
    for i in (0, 7, 20, L * L - 1):
        lp = []
        for a in range(q):
            y = x.clone(); y[:, i] = a
            lp.append(log_prob_unnormalised(y, beta, h, L))
        brute = torch.softmax(torch.stack(lp, -1), -1)
        worst = max(worst, float((brute - p[:, i]).abs().max()))
    print(f"[2] local conditional vs brute force (h={h}): max err {worst:.2e}  "
          f"{'OK' if worst < 1e-5 else 'FAIL'}")

    # 3. the local-energy residual must vanish in expectation ON target samples
    xs = pt_reference(1.2, 0.0, L, q, 256, 300, dev, seed=3, n_temps=12)
    ps = local_target_p(xs, 1.2, 0.0, L, q)
    r = local_energy_residual(xs, ps, L, q).mean()
    print(f"[3] local-energy residual on target samples: {float(r):+.4f} (want ~0)")

    # 4. zero-field PT reproduces the exact critical energy, and visits all q modes
    for qq in (2, 3):
        bc = beta_critical(qq)
        xs = pt_reference(bc, 0.0, L, qq, 512, 400, dev, seed=qq, n_temps=16)
        u = float(energy_per_spin(xs, bc, 0.0, L).mean())
        print(f"[4] q={qq} beta_c={bc:.4f}: u={u:+.4f}  exact(inf L)={bc*u_critical(qq):+.4f}  "
              f"modes={[round(v,3) for v in mode_fractions(xs, L, qq).tolist()]}")

    # 5. the field selects state 0, and the crossover sits where the nat count says it should
    print(f"[5] beta=1.2, L={L}: one pure state is worth beta*h*L^2 = {1.2*L*L:.0f}*h nats")
    for h in (-0.05, -0.01, 0.0, 0.01, 0.05):
        xs = pt_reference(1.2, h, L, q, 512, 300, dev, seed=11, n_temps=12)
        mf = mode_fractions(xs, L, q)
        print(f"    h={h:+.3f} ({1.2*L*L*abs(h):5.1f} nats): modes={[round(v,3) for v in mf.tolist()]}"
              f"  frac(state 0)={mode_fraction_0(xs, L, q):.3f}")
