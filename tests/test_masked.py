"""Exactness of the absorbing-mask construction."""

import pytest
import torch

from discrete_sampler.methods.dig_in.masked import MaskedInterpolant, reveal_edges
from discrete_sampler.targets import IsingModel, PottsModel
from discrete_sampler.utils.metrics import ess, log_Z_is
from tests.enumeration import all_states, exact_masked_logits_fn

TARGETS = {
    "ising": lambda: IsingModel((5,), beta=0.45, h=0.1),
    "potts": lambda: PottsModel((4,), q=3, beta=0.8),
}


@pytest.fixture(autouse=True)
def _f64():
    torch.manual_seed(0)
    old = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(old)


@pytest.mark.parametrize("target_name", list(TARGETS))
def test_masked_tsi_identity(target_name):
    """E[s_nu(X1)_{d,n} | visible coords] = N * q^d(n | x) at masked d."""
    target = TARGETS[target_name]()
    xs, nu = all_states(target)
    N = target.N
    minterp = MaskedInterpolant(N)
    log_nu = target.log_prob(xs)
    x1 = xs[torch.multinomial(nu, 5, replacement=True)]
    xt, mask = minterp.sample(x1, torch.rand(5))
    s_all = target.score(xs)
    for b in range(5):
        vis = ~mask[b]
        match = (xs[:, vis] == x1[b, vis].view(1, -1)).all(dim=1)
        w = torch.where(match, log_nu, torch.tensor(-1e30).double())
        w = (w - w.logsumexp(0)).exp()
        for d in range(target.D):
            if not mask[b, d]:
                continue
            lhs = (w.view(-1, 1) * s_all[:, d, :]).sum(0)
            q = torch.zeros(N).index_put_((xs[:, d],), w, accumulate=True)
            assert (lhs - N * q).abs().max() < 1e-8


@pytest.mark.parametrize("target_name", list(TARGETS))
def test_heat_bath_label_site_invariance(target_name):
    """The heat-bath label must not depend on the site's own value."""
    target = TARGETS[target_name]()
    x = target.sample_uniform(16)
    hb = target.heat_bath(x)
    for v in range(target.N):
        y = x.clone()
        y[:, 0] = v
        hb2 = target.heat_bath(y)
        assert (hb2[:, 0] - hb[:, 0]).abs().max() < 1e-10


@pytest.mark.parametrize("target_name", list(TARGETS))
def test_exact_masked_sampler(target_name):
    """With the exact denoiser: exact samples and zero-variance weights."""
    target = TARGETS[target_name]()
    xs, nu = all_states(target)
    minterp = MaskedInterpolant(target.N)
    fn = exact_masked_logits_fn(target, xs)
    lz = target.log_Z_enumeration()

    x, log_q = minterp.unmask_sample(fn, 3000, target.D, n_steps=target.D)
    logw = target.log_prob(x) - log_q
    assert abs(log_Z_is(logw) - lz) < 1e-6
    assert logw.std() < 1e-6

    # block reveals: worse proposal, still unbiased
    x, log_q = minterp.unmask_sample(fn, 3000, target.D, n_steps=2)
    logw = target.log_prob(x) - log_q
    assert abs(log_Z_is(logw) - lz) < 0.1
    assert ess(logw) > 0.3


@pytest.mark.parametrize("n_steps,n_single", [
    (64, 0), (64, 16), (64, 63), (64, 200), (10, 0), (10, 5), (1, 4), (64, -3),
])
def test_reveal_edges_is_a_valid_schedule(n_steps, n_single):
    """Every schedule must use exactly n_steps calls and reveal all D sites.

    A schedule that stopped short of D would leave mask tokens in the
    supposedly clean endpoint and silently corrupt the training labels, so
    this is the property that has to hold for any (n_steps, n_single).
    """
    D = 64
    edges = reveal_edges(D, n_steps, n_single)
    assert len(edges) == n_steps + 1
    assert edges[0] == 0 and edges[-1] == D
    assert all(b >= a for a, b in zip(edges, edges[1:])), "reveal counts must not go back"


def test_reveal_edges_front_loads_single_reveals():
    """n_single sites go one at a time, the rest in equal chunks."""
    D, n_steps, n_single = 100, 20, 8
    edges = reveal_edges(D, n_steps, n_single)
    steps = [b - a for a, b in zip(edges, edges[1:])]
    assert steps[:n_single] == [1] * n_single
    assert sum(steps) == D
    assert max(steps[n_single:]) - min(steps[n_single:]) <= 1
    # and it really is front-loading: without it the first call reveals a block
    assert steps[0] < (reveal_edges(D, n_steps, 0)[1] - reveal_edges(D, n_steps, 0)[0])


@pytest.mark.parametrize("target_name", list(TARGETS))
def test_front_loaded_reveals_stay_unbiased(target_name):
    """n_single changes only how the calls are spent, never the estimator:
    with the exact denoiser the weights are still a valid log Z estimator."""
    target = TARGETS[target_name]()
    xs, _ = all_states(target)
    minterp = MaskedInterpolant(target.N)
    fn = exact_masked_logits_fn(target, xs)
    lz = target.log_Z_enumeration()
    x, log_q = minterp.unmask_sample(fn, 3000, target.D, n_steps=3, n_single=2)
    logw = target.log_prob(x) - log_q
    assert abs(log_Z_is(logw) - lz) < 0.1


# ----------------------------------------------------------------------
# block reveal orders
# ----------------------------------------------------------------------
def test_block_order_keeps_a_blocks_coordinates_adjacent():
    """A block order must be a permutation whose blocks are contiguous and
    internally ordered -- otherwise it is not 'reveal one particle at a time'."""
    import torch

    from discrete_sampler.methods.dig_in.masked import MaskedInterpolant

    mi = MaskedInterpolant(5, block=3)
    torch.manual_seed(0)
    o = mi._order(16, 12, "cpu")
    assert o.shape == (16, 12)
    for b in range(16):
        assert sorted(o[b].tolist()) == list(range(12)), "not a permutation"
        for k in range(4):
            blk = o[b, 3 * k: 3 * k + 3].tolist()
            assert blk == [blk[0], blk[0] + 1, blk[0] + 2] and blk[0] % 3 == 0


def test_block_one_reproduces_the_uniform_order():
    """The default must be byte-identical to the previous behaviour, so the
    lattice benchmarks are untouched."""
    import torch

    from discrete_sampler.methods.dig_in.masked import MaskedInterpolant

    mi = MaskedInterpolant(5, block=1)
    g1 = torch.Generator().manual_seed(7)
    got = mi._order(8, 10, "cpu", g1)
    g2 = torch.Generator().manual_seed(7)
    want = torch.argsort(torch.rand(8, 10, generator=g2), dim=1)
    assert torch.equal(got, want)


def test_block_masks_are_prefixes_of_a_block_order():
    """The training contexts must be exactly the ones the block sampler visits:
    a prefix of some block order, never a scattered pattern."""
    import torch

    from discrete_sampler.methods.dig_in.masked import MaskedInterpolant

    mi = MaskedInterpolant(5, block=2)
    torch.manual_seed(1)
    x1 = torch.randint(0, 5, (256, 8))
    _xt, mask = mi.sample(x1, torch.rand(256))
    assert (~mask).any(dim=1).any() and mask.any(dim=1).all()
    for b in range(256):
        revealed = (~mask[b]).nonzero().flatten().tolist()
        # at most one block may be split, and only as its own prefix
        blocks = {}
        for c in revealed:
            blocks.setdefault(c // 2, []).append(c)
        partial = [v for v in blocks.values() if len(v) < 2]
        assert len(partial) <= 1
        for v in partial:
            assert v == [min(v)] and min(v) % 2 == 0


def test_block_order_sampler_still_reveals_every_site_exactly_once():
    """Whatever the order, the sampler must end with no mask tokens left and a
    log-density that is the sum of D categorical draws."""
    import torch

    from discrete_sampler.methods.dig_in.masked import MaskedInterpolant

    mi = MaskedInterpolant(4, block=2)
    D, n = 8, 32
    calls = []

    def logits_fn(x, a):
        calls.append(int((x == mi.mask_token).sum()))
        return torch.zeros(x.shape[0], D, 4)

    torch.manual_seed(2)
    x, log_q = mi.unmask_sample(logits_fn, n, D, device="cpu")
    assert (x < 4).all(), "mask tokens survived the sampler"
    assert len(calls) == D
    # uniform logits: every draw has log prob -log 4, D of them
    import math

    assert torch.allclose(log_q, torch.full((n,), -D * math.log(4.0)), atol=1e-5)
