import numpy as np
import pytest
import torch

from discrete_sampler.targets import IsingModel
from discrete_sampler.utils import metrics as M


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
    np.random.seed(0)


@pytest.mark.parametrize("p", [1, 2])
def test_uniform_weights_match_unweighted(p):
    """A weighted distance with flat weights must equal the unweighted one, so
    that a reported gap between them is caused by the weights alone."""
    a, b = torch.randn(500), torch.randn(600)
    w = M._np(M.snis_weights(torch.zeros(500)))
    assert M.wasserstein1d(a, b, p=p) == pytest.approx(
        M.wasserstein1d(a, b, p=p, wa=w), abs=1e-9
    )


def test_uniform_weights_match_unweighted_tv():
    a, b = torch.randn(500), torch.randn(600)
    w = M._np(M.snis_weights(torch.zeros(500)))
    assert M.histogram_tv(a, b, bins=41, range_=(-4, 4)) == pytest.approx(
        M.histogram_tv(a, b, bins=41, range_=(-4, 4), wa=w), abs=1e-12
    )


def test_weighted_quantiles_monotone_under_extreme_weights():
    v = M._np(torch.randn(1000))
    w = M._np(M.snis_weights(torch.randn(1000) * 8))
    q = M._weighted_quantile(v, w, np.linspace(0.0, 1.0, 50))
    assert np.all(np.diff(q) >= -1e-9)


def test_reweighting_corrects_a_biased_proposal():
    """q = N(1,1) reweighted towards p = N(0,1) by log w = log p - log q."""
    q = torch.randn(100_000) + 1.0
    w = M._np(M.snis_weights(-0.5 * q**2 + 0.5 * (q - 1.0) ** 2))
    assert float((w * M._np(q.double())).sum()) == pytest.approx(0.0, abs=0.05)
    tgt = torch.randn(100_000)
    assert M.wasserstein1d(q, tgt, p=1, wa=w) < 0.2 * M.wasserstein1d(q, tgt, p=1)


def test_reweighted_keys_are_additive():
    t = IsingModel((3, 3), beta=0.4)
    x, x_ref, log_w = torch.randint(0, 2, (256, 9)), torch.randint(0, 2, (300, 9)), torch.randn(256)
    o = M.summarize_observables(t, x, log_w=log_w)
    c = M.compare_to_reference(t, x, x_ref, log_w=log_w)
    assert {"mag_mean_rw", "mag_abs_mean_rw", "energy_per_site_rw"} <= set(o)
    assert {"mag_w1_rw", "energy_w2_rw", "mag_tv_rw", "mag_abs_err_rw",
            "energy_err_rw"} <= set(c)
    assert all(np.isfinite(v) for v in {**o, **c}.values())
    # omitting log_w must leave the plain metrics bit-identical
    o0 = M.summarize_observables(t, x)
    assert not any(k.endswith("_rw") for k in o0)
    assert all(o0[k] == pytest.approx(o[k], abs=1e-12) for k in o0)


def test_local_inconsistency_is_lowest_on_target_samples():
    """-mean_d log nu(x_d | x^-d) is an observable of the target's own
    conditionals, so exact samples must score better than uniform noise."""
    import torch

    from discrete_sampler.targets import IsingModel
    from discrete_sampler.targets.ising.mcmc import IsingMCMC
    from discrete_sampler.utils.metrics import local_inconsistency

    torch.manual_seed(0)
    target = IsingModel((8, 8), beta=0.6)          # ordered: far from uniform
    x_ref = IsingMCMC(target).sample(256, n_chains=64, method="heat_bath",
                                     burn_in=200, thin=2)
    x_unif = target.sample_uniform(256)
    li_ref = local_inconsistency(target, x_ref)
    li_unif = local_inconsistency(target, x_unif)
    assert li_ref.shape == (256,)
    assert li_ref.mean() < li_unif.mean()
