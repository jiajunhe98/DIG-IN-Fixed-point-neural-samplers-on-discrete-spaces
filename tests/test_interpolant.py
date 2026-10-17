"""Exactness of the mixture-interpolant identities against enumeration."""

import pytest
import torch

from discrete_sampler.methods.dig_in.interpolant import MixtureInterpolant
from discrete_sampler.methods.dig_in.sampling import log_weights_uniform_prior, tau_leap
from discrete_sampler.targets import IsingModel, PottsModel
from discrete_sampler.utils.metrics import ess, log_Z_is
from tests.enumeration import all_states, exact_fields_fn, pair_posterior

TARGETS = {
    "ising": lambda: IsingModel((5,), beta=0.45, h=0.1),
    "potts": lambda: PottsModel((4,), q=3, beta=0.8),
}
SCHEDULES = ["quadratic", "onesided-quadratic"]


@pytest.fixture(autouse=True)
def _f64():
    torch.manual_seed(0)
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


def _setup(target_name, schedule):
    target = TARGETS[target_name]()
    xs, nu = all_states(target)
    interp = MixtureInterpolant(target.N, schedule=schedule, clamp_ratio=1e12)
    t = torch.tensor(0.37)
    x = xs[torch.randint(0, xs.shape[0], (1,)).item()]
    return target, xs, nu, interp, t, x


@pytest.mark.parametrize("schedule", SCHEDULES)
@pytest.mark.parametrize("target_name", list(TARGETS))
def test_score_and_gsi(target_name, schedule):
    target, xs, nu, interp, t, x = _setup(target_name, schedule)
    S, D, N = xs.shape[0], target.D, target.N
    rho, post, pd = pair_posterior(interp, target, xs, nu, x, t)

    exact_score = torch.zeros(D, N)
    for d in range(D):
        for n in range(N):
            y = x.clone()
            y[d] = n
            ry, _, _ = pair_posterior(interp, target, xs, nu, y, t)
            exact_score[d, n] = ry / rho

    # score read-out from the exact pair posterior
    _, _, s = interp.posterior_fields(
        pd.clamp_min(1e-300).log().unsqueeze(0), x.unsqueeze(0), t.view(1)
    )
    assert (s[0] - exact_score).abs().max() < 1e-8

    # generalized score identity (uniform prior anchor)
    score1_all = target.score(xs)
    post1 = post.sum(0)  # posterior over x1
    for d in range(D):
        for n in range(N):
            idx = (xs[:, d] + n - x[d]) % N
            T = (post1 * score1_all[torch.arange(S), d, idx]).sum()
            assert abs(T - exact_score[d, n]) < 1e-8


@pytest.mark.parametrize("schedule", SCHEDULES)
@pytest.mark.parametrize("target_name", list(TARGETS))
def test_time_reversal(target_name, schedule):
    target, xs, nu, interp, t, x = _setup(target_name, schedule)
    rho, _, pd = pair_posterior(interp, target, xs, nu, x, t)
    v, vb, s = interp.posterior_fields(
        pd.clamp_min(1e-300).log().unsqueeze(0), x.unsqueeze(0), t.view(1)
    )
    for d in range(target.D):
        for n in range(target.N):
            if n == x[d]:
                continue
            y = x.clone()
            y[d] = n
            ry, _, pd_y = pair_posterior(interp, target, xs, nu, y, t)
            vy, _, _ = interp.posterior_fields(
                pd_y.clamp_min(1e-300).log().unsqueeze(0), y.unsqueeze(0), t.view(1)
            )
            assert abs(vb[0, d, n] - vy[0, d, x[d]] * (ry / rho)) < 1e-8


@pytest.mark.parametrize("schedule", SCHEDULES)
@pytest.mark.parametrize("target_name", list(TARGETS))
def test_soft_labels_conditional_mean(target_name, schedule):
    target, xs, nu, interp, t, x = _setup(target_name, schedule)
    S, D, N = xs.shape[0], target.D, target.N
    _, post, pd = pair_posterior(interp, target, xs, nu, x, t)
    hb_all = target.heat_bath(xs)
    kappa, _ = interp._kernel_tensors(t.view(1))
    post1 = post.sum(0)
    lbl_mean = torch.zeros(D, N, N)
    for d in range(D):
        for j in range(S):
            l = hb_all[j, d].view(1, N) * kappa[0, :, :, x[d]]
            lbl_mean[d] += post1[j] * l / l.sum()
    assert (lbl_mean - pd).abs().max() < 1e-8


@pytest.mark.parametrize("schedule", SCHEDULES)
@pytest.mark.parametrize("target_name", list(TARGETS))
def test_product_form_head(target_name, schedule):
    target, xs, nu, interp, t, x = _setup(target_name, schedule)
    _, _, pd = pair_posterior(interp, target, xs, nu, x, t)
    kappa, _ = interp._kernel_tensors(t.view(1))
    g = torch.zeros(1, target.D, target.N)
    for d in range(target.D):
        h = pd[d, 0] / kappa[0, 0, :, x[d]].clamp_min(1e-300)
        g[0, d] = (h / h.sum()).clamp_min(1e-300).log()
    log_q = interp.pair_log_posterior(g, x.unsqueeze(0), t.view(1))
    assert (log_q[0].exp() - pd).abs().max() < 1e-8


@pytest.mark.parametrize("target_name", list(TARGETS))
def test_exact_field_sampler_unbiased(target_name):
    target = TARGETS[target_name]()
    xs, nu = all_states(target)
    interp = MixtureInterpolant(target.N, schedule="quadratic", clamp_ratio=1e6)
    fn = exact_fields_fn(interp, target, xs, nu)
    n, M = 2000, 50
    x0 = target.sample_uniform(n)
    x1, log_fb = tau_leap(fn, x0, M, zeta=1.0, num_states=target.N, with_rnd=True)
    logw = log_weights_uniform_prior(target.log_prob(x1), log_fb, target.D, target.N)
    lz = target.log_Z_enumeration()
    assert abs(log_Z_is(logw) - lz) < 0.1
    assert ess(logw) > 0.5
    # terminal law close to nu up to the discretization floor
    idx = (x1 * (target.N ** torch.arange(target.D)).view(1, -1)).sum(1)
    emp = torch.bincount(idx, minlength=xs.shape[0]).double() / n
    assert 0.5 * (emp - nu).abs().sum() < 0.1
