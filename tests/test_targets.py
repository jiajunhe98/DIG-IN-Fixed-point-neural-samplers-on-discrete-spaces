import pytest
import torch

from dig_in.targets import IsingModel, PottsModel
from dig_in.targets.ising import exact as ising_exact
from dig_in.targets.ising.mcmc import IsingMCMC
from dig_in.targets.potts import exact as potts_exact
from dig_in.targets.potts.mcmc import PottsMCMC


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def test_ising_energy_diff():
    IsingModel((4, 4), beta=0.4, h=0.2).check_energy_diff(n=8)
    IsingModel((6,), beta=0.7).check_energy_diff(n=8)


def test_potts_energy_diff():
    PottsModel((2, 4), q=3, beta=0.9).check_energy_diff(n=8)
    PottsModel((4,), q=4, beta=0.5).check_energy_diff(n=8)


def test_ising_log_Z_agreement():
    t = IsingModel((4, 4), beta=0.4407)
    lz = t.log_Z_enumeration()
    assert abs(lz - ising_exact.log_Z_transfer(4, 4, 0.4407)) < 1e-5
    assert abs(lz - ising_exact.log_Z_kaufman(4, 4, 0.4407)) < 1e-5
    # field + non-square via transfer matrix
    t2 = IsingModel((2, 4), beta=0.3, h=0.2)
    assert abs(t2.log_Z_enumeration() - ising_exact.log_Z_transfer(2, 4, 0.3, B=0.06)) < 1e-5
    # Kaufman vs transfer at sizes beyond enumeration
    for L, K in [(6, 0.3), (8, 0.4407), (8, 0.6)]:
        assert abs(
            ising_exact.log_Z_transfer(L, L, K) - ising_exact.log_Z_kaufman(L, L, K)
        ) < 1e-6


def test_potts_log_Z_agreement():
    t = PottsModel((2, 4), q=3, beta=0.8)
    assert abs(t.log_Z_enumeration() - potts_exact.log_Z_transfer(2, 4, 3, 0.8)) < 1e-5


@pytest.mark.parametrize("method", ["metropolis", "heat_bath", "wolff", "swendsen_wang"])
def test_ising_mcmc_matches_enumeration(method):
    target = IsingModel((4, 4), beta=0.35)
    xs = target._index_to_state(torch.arange(2**16))
    p = torch.softmax(target.log_prob(xs), 0)
    m_exact = (p * target.magnetization(xs).abs()).sum().item()
    e_exact = (p * target.energy_per_site(xs)).sum().item()
    mcmc = IsingMCMC(target)
    s = mcmc.sample(3000, n_chains=100, method=method, burn_in=150, thin=3)
    assert abs(target.magnetization(s).abs().mean().item() - m_exact) < 0.04
    assert abs(target.energy_per_site(s).mean().item() - e_exact) < 0.06


@pytest.mark.parametrize("method", ["metropolis", "heat_bath", "wolff", "swendsen_wang"])
def test_potts_mcmc_matches_enumeration(method):
    target = PottsModel((2, 4), q=3, beta=0.6)
    xs = target._index_to_state(torch.arange(3**8))
    p = torch.softmax(target.log_prob(xs), 0)
    m_exact = (p * target.magnetization(xs)).sum().item()
    e_exact = (p * target.energy_per_site(xs)).sum().item()
    mcmc = PottsMCMC(target)
    s = mcmc.sample(3000, n_chains=100, method=method, burn_in=150, thin=3)
    assert abs(target.magnetization(s).mean().item() - m_exact) < 0.04
    assert abs(target.energy_per_site(s).mean().item() - e_exact) < 0.06
