"""The binned policy, the Gibbs labels and the fixed point of the loss
(dig_in_rl/policy.py).  No environment needed; a CPU is enough.

1. Binning: decoded actions stay in their bins, the centres round trip.
2. Likelihood: the log-likelihood and the entropy the sampler accumulates
   equal their recomputation at the stored contexts of the order.
3. Gibbs labels: softmax(Q(s, a^{-d}, c_l) / tau) for a known critic.
4. Duals: the temperature falls while the entropy is above its target and
   rises while it is below, except at its cap; kappa rises while the KL
   exceeds its bound.
5. Fixed point: on a 2-D bandit with a quadratic Q, training with the loss
   alone makes the enumerated marginal of the masked policy (averaged over
   the two orders) the Boltzmann distribution exp(Q / tau) over the bin
   centres, with and without the trust region.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx

from dig_in_rl.policy import (
    Binning,
    MaskedBinnedPolicy,
    actor_loss,
    gibbs_label_logits,
    per_order_logprob,
    sample_action,
)


def make_policy(m, L, obs_dim=3, hidden=64, layers=2, seed=0, random_head=True, **kw):
    return MaskedBinnedPolicy(m, obs_dim, num_bins=L, hidden=hidden, layers=layers,
                              head_init_scale=1.0 if random_head else 1e-8, rngs=nnx.Rngs(seed), **kw)


def quad_q(P, mu):
    def q(a):  # (..., m) -> (...)
        d = a - mu
        return -0.5 * jnp.einsum("...i,ij,...j->...", d, P, d)
    return q


def test_binning():
    b = Binning(3, 16)
    key = jax.random.PRNGKey(0)
    z = jax.random.randint(key, (1000, 3), 0, 16)
    assert jnp.all(b.encode(b.decode(z, key)) == z)
    assert jnp.all(b.encode(b.decode(z)) == z)
    np.testing.assert_allclose(b.log_delta_sum, 3 * np.log(2 / 16))
    np.testing.assert_allclose([b.centres[0] - b.delta / 2, b.centres[-1] + b.delta / 2], [-1, 1], atol=1e-6)


def test_likelihood():
    for m, L in ((5, 7), (1, 9), (3, 4)):
        model = make_policy(m, L)
        key = jax.random.PRNGKey(1)
        obs = jax.random.normal(key, (64, 3))
        a, z, step_idx, entropy, logp = sample_action(model, key, obs)
        np.testing.assert_allclose(logp, per_order_logprob(model, obs, z, step_idx), atol=1e-4)
        _, feat_own, _ = model.contexts(obs, z, step_idx)
        lsm = jax.nn.log_softmax(model.logits_own(feat_own), axis=-1)
        h = -jnp.sum(jnp.exp(lsm) * lsm, axis=(-1, -2)) + model.binning.log_delta_sum
        np.testing.assert_allclose(entropy, h, atol=1e-4)
        assert jnp.all(model.binning.encode(a) == z)
        assert jnp.all(jnp.sort(step_idx, axis=-1) == jnp.arange(m))  # every dimension revealed once
        # greedy decoding: bin centres, the most likely bin at every step
        ag, zg, sg, _, _ = sample_action(model, key, obs, greedy=True)
        np.testing.assert_array_equal(ag, model.binning.centres[zg])
        _, feat_own, _ = model.contexts(obs, zg, sg)
        assert jnp.all(jnp.argmax(model.logits_own(feat_own), axis=-1) == zg)


def test_gibbs_label():
    m, L, B, temp = 3, 16, 32, 0.5
    q = quad_q(jnp.array([[4.0, 1.0, 0.0], [1.0, 3.0, 0.5], [0.0, 0.5, 2.0]]), jnp.array([0.2, -0.3, 0.1]))
    b = Binning(m, L)
    key = jax.random.PRNGKey(0)
    a = jax.random.uniform(key, (B, m), minval=-1, maxval=1)
    dim = jax.random.randint(key, (B,), 0, m)
    lab = gibbs_label_logits(lambda o, acts: q(acts), jnp.zeros((B, 1)), a, dim, temp, b.centres)
    for r in range(B):
        a_l = np.repeat(np.asarray(a[r])[None], L, 0)
        a_l[:, int(dim[r])] = b.centres
        np.testing.assert_allclose(jax.nn.softmax(lab[r]), jax.nn.softmax(q(a_l) / temp), atol=1e-5)


def _dual_grads(model, batch, h_star, kl_bound, temp_max):
    def loss(params):
        return actor_loss(nnx.merge(graphdef, params), old, batch, lambda o, acts: -jnp.sum(acts**2, -1),
                          jax.random.PRNGKey(0), h_star, kl_bound, temp_max)[0]
    graphdef, params = nnx.split(model)
    old = make_policy(model.m, model.L, seed=1)  # a different policy: KL > 0
    g = jax.grad(loss)(params)
    return float(g["log_temperature"].value[0]), float(g["log_kappa"].value[0])


def test_duals():
    m, L = 2, 8
    model = make_policy(m, L, temp_start=0.5)
    obs = jax.random.normal(jax.random.PRNGKey(2), (128, 3))
    a, z, step_idx, entropy, _ = sample_action(model, jax.random.PRNGKey(3), obs)
    batch = dict(obs=obs, action=a, tokens=z, step_idx=step_idx)
    h = float(jnp.mean(entropy))
    g_tau, g_kappa = _dual_grads(model, batch, h_star=h - 1.0, kl_bound=1e-3, temp_max=1.0)
    assert g_tau > 0 and g_kappa < 0  # entropy above target: tau falls; KL above bound: kappa rises
    g_tau, g_kappa = _dual_grads(model, batch, h_star=h + 1.0, kl_bound=1e3, temp_max=1.0)
    assert g_tau < 0 and g_kappa > 0
    g_tau, _ = _dual_grads(model, batch, h_star=h + 1.0, kl_bound=1e3, temp_max=0.5)  # at the cap
    assert g_tau == 0.0


class Bandit:
    """2-D bandit, quadratic Q, 8 x 8 bins; the Gibbs labels' fixed point at
    temperature tau is softmax(Q(c) / tau) over the 64 bin centres."""
    m, L, tau = 2, 8, 1.0

    def __init__(self):
        self.q = quad_q(jnp.array([[4.0, 3.0], [3.0, 4.0]]), jnp.array([0.3, -0.2]))
        c = Binning(self.m, self.L).centres
        self.grid = jnp.stack(jnp.meshgrid(c, c, indexing="ij"), -1).reshape(-1, 2)
        self.target = jax.nn.softmax(self.q(self.grid) / self.tau)

    def marginal(self, model):
        """pi(x) of the masked policy, averaged over the two orders."""
        L = self.L
        p_first = jax.nn.softmax(model.logits(model.trunk(jnp.zeros((1, 1)), jnp.zeros((1, 2), jnp.int32),
                                                          jnp.zeros((1, 2), bool)))[0], -1)  # (2, L)
        probs = jnp.zeros((L, L))
        for first in (0, 1):
            zs = jnp.zeros((L, 2), jnp.int32).at[:, first].set(jnp.arange(L))
            rev = jnp.zeros((L, 2), bool).at[:, first].set(True)
            p_second = jax.nn.softmax(model.logits(model.trunk(jnp.zeros((L, 1)), zs, rev))[:, 1 - first], -1)
            joint = p_first[first][:, None] * p_second
            probs = probs + 0.5 * (joint if first == 0 else joint.T)
        return probs.reshape(-1)


def train_bandit(bandit, kappa, iters=400, batch=1024, mbs=4, lr=3e-3, seed=0):
    """The policy update of the trainer on the bandit, with the duals frozen
    (tau = bandit.tau, kappa as given): every iteration samples a batch from
    the current policy, then `mbs` minibatch steps of actor_loss."""
    model = make_policy(bandit.m, bandit.L, obs_dim=1, hidden=128, layers=2, seed=seed, random_head=False,
                        temp_start=bandit.tau, temp_min=0.0, kappa_start=kappa)
    graphdef, params = nnx.split(model)
    frozen = lambda p: jax.tree_util.tree_map_with_path(
        lambda path, _: "frozen" if "log_" in str(path[0].key) else "adam", p)
    tx = optax.transforms.partition({"adam": optax.adam(lr), "frozen": optax.set_to_zero()}, frozen)
    opt_state = tx.init(params)
    q_slice = lambda o, acts: bandit.q(acts)

    @jax.jit
    def iteration(params, opt_state, key):
        old = nnx.merge(graphdef, params)
        k_s, k_u = jax.random.split(key)
        obs = jnp.zeros((batch, 1))
        a, z, step_idx, _, _ = sample_action(old, k_s, obs)
        data = dict(obs=obs, action=a, tokens=z, step_idx=step_idx)
        n = batch // mbs

        def mb(carry, k):
            params, opt_state = carry
            k1, k2 = jax.random.split(k)
            idx = jax.random.permutation(k1, batch)[:n]
            loss_fn = lambda p: actor_loss(nnx.merge(graphdef, p), old, jax.tree.map(lambda x: x[idx], data),
                                           q_slice, k2, -1.0, 0.1, 1.0)
            (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            updates, opt_state = tx.update(grads, opt_state, params)
            return (optax.apply_updates(params, updates), opt_state), metrics

        (params, opt_state), _ = jax.lax.scan(mb, (params, opt_state), jax.random.split(k_u, mbs))
        return params, opt_state

    key = jax.random.PRNGKey(seed)
    for _ in range(iters):
        key, k = jax.random.split(key)
        params, opt_state = iteration(params, opt_state, k)
    return nnx.merge(graphdef, params)


def test_fixed_point():
    bandit = Bandit()
    tv = lambda model: float(0.5 * jnp.sum(jnp.abs(bandit.marginal(model) - bandit.target)))
    uniform = float(0.5 * jnp.sum(jnp.abs(1 / 64 - bandit.target)))
    assert tv(train_bandit(bandit, kappa=1e-8)) < 0.04 < uniform
    # the trust region slows the iteration down but leaves its fixed point unchanged
    assert tv(train_bandit(bandit, kappa=1.0, iters=600)) < 0.04
