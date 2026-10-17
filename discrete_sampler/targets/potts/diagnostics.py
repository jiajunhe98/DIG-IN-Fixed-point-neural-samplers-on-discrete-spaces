"""Mode-resolving observables for the q-state Potts model.

Why this module exists
----------------------
`PottsModel.magnetization` returns the standard order parameter

    m(x) = (q * f_max(x) - 1) / (q - 1),

where f_max is the fraction of sites in the most common state.  It is a MAGNITUDE: it
does not record *which* of the q states won.  Below beta_c the target has q symmetry-
related ordered modes of equal weight, and m is identical for

  * a correct sampler that visits all q modes with weight 1/q each, and
  * a collapsed sampler that only ever produces state 2.

Both report the same m and the same energy, because the energy is also S_q-invariant.  So
neither the energy nor m can certify a Potts sampler below the transition, and a sampler
can look converged on every scalar the repo currently logs while placing all its mass on
one mode.

The observables here close that gap.  `mode_fractions` is the direct diagnostic (a batch
histogram over which state won) and `magnetisation_vector` is the underlying collective
variable, the Potts analogue of the Ising average spin.

Nothing here is used for training.  In particular none of it may be fed back into a loss:
telling a sampler that the q modes are equally weighted is injecting the target's S_q
symmetry, which makes the diagnostic true by construction.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "u_critical",
    "state_fractions",
    "majority_state",
    "mode_fractions",
    "simplex_vertices",
    "magnetisation_vector",
    "magnetisation_angle",
    "mode_report",
]


def u_critical(q: int, J: float = 1.0) -> float:
    """Exact internal energy per site at beta_c, thermodynamic limit.

        u_c(q) = -J * (1 + 1/sqrt(q))       q=2: -1.70711  q=3: -1.57735  q=4: -1.50000

    with u = <E>/D in the convention of `PottsModel`.  Together with
    `exact.beta_critical` this is the sharpest cheap check available on a reference chain
    or a trained sampler: no fitting, no finite-size extrapolation, one number.

    Finite lattices sit below it by an O(1/L) amount, so compare trends across L rather
    than demanding exact agreement at a single size.
    """
    return -J * (1.0 + 1.0 / math.sqrt(q))


def state_fractions(x: Tensor, q: int) -> Tensor:
    """Fraction of sites in each state, per configuration.  (B, D) -> (B, q)."""
    return F.one_hot(x, q).to(torch.get_default_dtype()).mean(dim=1)


def majority_state(x: Tensor, q: int) -> Tensor:
    """Which state occupies the most sites of each configuration.  (B,) long.

    Ties break to the lowest state id.  Above beta_c, where no state dominates, this is
    close to uniform noise, which is the correct behaviour: there are no modes to resolve.
    """
    return state_fractions(x, q).argmax(dim=-1)


def mode_fractions(x: Tensor, q: int) -> Tensor:
    """Fraction of the BATCH whose majority state is a, for each a.  (q,).

    The collapse diagnostic.  A correct sampler sits at 1/q on every entry; a collapsed one
    puts 1 on a single entry and 0 on the rest.  Report the minimum entry as a scalar
    summary: 1/q is healthy, 0 is total collapse.

    Note this is a property of a BATCH, not of a configuration, so it needs enough samples
    to be meaningful.  The sampling standard deviation of one entry is
    sqrt(p(1-p)/n) with p = 1/q, which is 0.012 at q=3 and n=1536.
    """
    maj = majority_state(x, q)
    return torch.bincount(maj, minlength=q).to(torch.get_default_dtype()) / maj.numel()


def simplex_vertices(q: int, device=None, dtype=None) -> Tensor:
    """The q vertices of a regular simplex in R^(q-1), as unit vectors.  (q, q-1).

    Unit length, pairwise dot product -1/(q-1), summing to zero.  Obtained as the non-null
    eigenvectors of the centring matrix I - 1/q, so the construction is q-generic rather
    than a hand-written table, then rotated into a canonical orientation: state 0 lies on
    the first axis and state 1 has a positive second coordinate.  At q=3 this puts the
    three states at 0, +120 and -120 degrees.  For q=2 the vertices are +1 and -1.
    """
    dtype = dtype or torch.get_default_dtype()
    c = torch.eye(q, dtype=torch.float64) - 1.0 / q
    _, evecs = torch.linalg.eigh(c)          # one zero eigenvalue, q-1 equal to one
    b = evecs[:, 1:]
    b = b / b.norm(dim=1, keepdim=True)

    # eigh fixes the simplex only up to a global rotation, so pin it: rotate state 0 onto
    # the first axis and, for q >= 3, reflect so state 1 has a positive second coordinate.
    # Without this the lobe angles differ between torch versions and platforms, which makes
    # any plot or angular histogram unreproducible.
    v = b[0]
    a = torch.cat([v[:, None], torch.eye(q - 1, dtype=torch.float64)], dim=1)
    rot, _ = torch.linalg.qr(a)
    rot[:, 0] = rot[:, 0] * torch.sign(rot[:, 0] @ v)     # first column is exactly v
    b = b @ rot
    if q >= 3 and b[1, 1] < 0:
        b[:, 1] = -b[:, 1]
    return b.to(dtype=dtype, device=device)


def magnetisation_vector(x: Tensor, q: int) -> Tensor:
    """Vector order parameter M(x) = (1/D) sum_d e_{x_d} in R^(q-1).  (B, q-1).

    The collective variable that separates the ordered modes, and the exact analogue of the
    Ising average spin: at q=2 it is the scalar average spin, with the familiar two peaks.

    At q=3 the disordered phase is a single blob at the origin and the ordered phase is
    three lobes 120 degrees apart at radius ~m_spont.  Plotting a batch in this plane shows
    mode collapse immediately, where `PottsModel.magnetization` shows nothing, since it is
    the radius of this vector up to normalisation.
    """
    b = simplex_vertices(q, device=x.device)
    return b[x].mean(dim=1)


def magnetisation_angle(x: Tensor, q: int) -> Tensor:
    """Angle of M in its first coordinate plane, in (-pi, pi].  (B,).  Requires q >= 3.

    For q=3 the three ordered modes sit near -2pi/3, 0 and +2pi/3, so a 1-D histogram over
    this angle has three peaks for a correct sampler and one for a collapsed one.  For
    q > 3 the modes do not lie in a single plane; use `magnetisation_vector` directly.
    """
    if q < 3:
        raise ValueError(
            f"magnetisation_angle needs q >= 3 (got q={q}); at q=2 the magnetisation is "
            "already the scalar average spin"
        )
    m = magnetisation_vector(x, q)
    return torch.atan2(m[:, 1], m[:, 0])


def mode_report(x: Tensor, q: int) -> dict:
    """Everything needed to judge mode balance from one batch, as plain floats.

    Keys: `mode_fracs` (list of q), `mode_min`, `mode_gap`, `M_norm` (mean |M|).
    `mode_min` is the one number to log per evaluation and alarm on.
    """
    mf = mode_fractions(x, q)
    return {
        "mode_fracs": [float(v) for v in mf],
        "mode_min": float(mf.min()),
        "mode_gap": float(mf.max() - mf.min()),
        "M_norm": float(magnetisation_vector(x, q).norm(dim=1).mean()),
    }
