"""Mode-resolving Potts diagnostics, and the exact critical energy.

The point of these tests is that the existing scalar observables cannot distinguish a
correct sampler from a collapsed one below beta_c.  `test_magnitude_cannot_see_collapse`
makes that failure explicit, and the rest check that the new observables can.
"""

import math

import numpy as np
import pytest
import torch

from dds.targets import PottsModel
from dds.targets.potts import exact as potts_exact
from dds.targets.potts.diagnostics import (
    magnetisation_angle,
    magnetisation_vector,
    mode_fractions,
    mode_report,
    simplex_vertices,
    u_critical,
)
from dds.targets.potts.mcmc import PottsMCMC


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
    np.random.seed(0)          # the Swendsen-Wang / Wolff cluster updates run in numpy


def _ground_truth(L, q, beta, n=512, burn_in=80, thin=4, n_chains=64):
    """Swendsen-Wang samples; cluster updates move freely between the q ordered modes.

    Single-site heat bath is NOT usable here: from a random quench it needs O(L^2) sweeps to
    coarsen and leaves a tail of domain-wall configurations, which biases exactly the mode
    statistics these tests are about.
    """
    t = PottsModel((L, L), q=q, beta=beta)
    x = PottsMCMC(t).sample(n_samples=n, n_chains=n_chains, method="swendsen_wang",
                            burn_in=burn_in, thin=thin)
    return t, x


# ---------------------------------------------------------------------------
# the gap this module fills
# ---------------------------------------------------------------------------

def test_magnitude_cannot_see_collapse():
    """PottsModel.magnetization is identical for a balanced and a collapsed batch."""
    L, q, beta = 12, 3, 1.2
    t, x = _ground_truth(L, q, beta)

    # relabel every configuration so that its majority state becomes 0.  This maps the
    # batch onto a single mode without touching any single configuration's statistics.
    from dds.targets.potts.diagnostics import majority_state
    maj = majority_state(x, q)
    perm = torch.stack([torch.roll(torch.arange(q), int(m)) for m in maj])
    collapsed = torch.gather(perm.to(x.device), 1, x)

    assert torch.allclose(t.magnetization(x), t.magnetization(collapsed))
    assert torch.allclose(t.energy(x), t.energy(collapsed))

    # ... but the mode fractions tell them apart unambiguously
    assert mode_fractions(x, q).min() > 0.15
    assert float(mode_fractions(collapsed, q).max()) == pytest.approx(1.0)
    assert float(mode_fractions(collapsed, q).min()) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# the exact critical energy
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("q", [2, 3, 4])
def test_critical_energy_matches_cluster_sampler(q):
    """u_c(q) = -(1 + 1/sqrt(q)); a finite lattice sits slightly below it."""
    L = 16
    beta = potts_exact.beta_critical(q)
    assert beta == pytest.approx(math.log(1.0 + math.sqrt(q)))
    t, x = _ground_truth(L, q, beta)
    u = float(t.energy_per_site(x).mean())
    assert u == pytest.approx(u_critical(q), abs=0.12)
    assert u < u_critical(q) + 1e-6      # finite size lowers the energy


def test_critical_energy_gap_shrinks_with_L():
    """The finite-size gap to u_c must shrink with L, or the convention is wrong.

    Needs many independent chains: thinned draws from a few chains leave a standard error
    comparable to the gap differences themselves.
    """
    q = 3
    beta = potts_exact.beta_critical(q)
    gaps = []
    for L in (8, 16, 32):
        t, x = _ground_truth(L, q, beta, n=2048, n_chains=512, burn_in=120)
        gaps.append(abs(float(t.energy_per_site(x).mean()) - u_critical(q)))
    # measured with this budget: roughly 0.075, 0.040, 0.020
    assert gaps[0] > gaps[1] > gaps[2]
    assert gaps[0] - gaps[2] > 0.02


# ---------------------------------------------------------------------------
# the vector order parameter
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("q", [2, 3, 4, 5])
def test_simplex_vertices_are_a_regular_simplex(q):
    b = simplex_vertices(q, dtype=torch.float64)
    assert b.shape == (q, q - 1)
    assert torch.allclose(b.norm(dim=1), torch.ones(q, dtype=torch.float64), atol=1e-10)
    assert torch.allclose(b.sum(0), torch.zeros(q - 1, dtype=torch.float64), atol=1e-10)
    off = b @ b.T
    off.fill_diagonal_(0.0)
    expected = -1.0 / (q - 1)
    assert torch.allclose(off[off != 0], torch.full_like(off[off != 0], expected), atol=1e-10)


def test_q2_reduces_to_ising_average_spin():
    """At q=2 the vector CV is the Ising average spin, up to an overall sign."""
    x = torch.randint(0, 2, (64, 8 * 8))
    m_vec = magnetisation_vector(x, 2)[:, 0]
    m_spin = (2.0 * x.to(torch.get_default_dtype()) - 1.0).mean(-1)
    corr = torch.corrcoef(torch.stack([m_vec, m_spin]))[0, 1]
    assert abs(float(corr)) == pytest.approx(1.0, abs=1e-6)


def test_ordered_phase_has_q_lobes_and_disordered_has_one():
    """q=3: three angular peaks below beta_c, a single blob at the origin above it."""
    L, q = 12, 3
    _, x_ord = _ground_truth(L, q, 1.3, n=1024)
    _, x_dis = _ground_truth(L, q, 0.5, n=1024)

    assert float(magnetisation_vector(x_ord, q).norm(dim=1).mean()) > 0.5
    assert float(magnetisation_vector(x_dis, q).norm(dim=1).mean()) < 0.15

    # with the canonical simplex the three modes sit at 0 and +-120 degrees
    ang = magnetisation_angle(x_ord, q) * 180.0 / math.pi
    near = sum(float(((ang - c + 180.0) % 360.0 - 180.0).abs().lt(30.0).float().mean())
               for c in (-120.0, 0.0, 120.0))
    assert near > 0.8

    # and all three are populated, which is the property that distinguishes a correct
    # sampler from a collapsed one; a single lobe would put `near` at 1.0 as well
    assert mode_fractions(x_ord, q).min() > 0.15


def test_magnetisation_angle_rejects_q2():
    x = torch.randint(0, 2, (4, 16))
    with pytest.raises(ValueError):
        magnetisation_angle(x, 2)


def test_mode_report_keys_and_balance():
    L, q = 12, 3
    _, x = _ground_truth(L, q, 1.2, n=1024)
    rep = mode_report(x, q)
    assert set(rep) == {"mode_fracs", "mode_min", "mode_gap", "M_norm"}
    assert len(rep["mode_fracs"]) == q
    assert sum(rep["mode_fracs"]) == pytest.approx(1.0)
    assert rep["mode_min"] > 0.15          # cluster updates keep all three modes populated
