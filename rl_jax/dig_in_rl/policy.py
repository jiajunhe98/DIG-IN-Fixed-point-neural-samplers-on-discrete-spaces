"""The any-order binned policy, its sampler and the Gibbs-label loss of DIG-IN
for maximum entropy RL (Sec. 5.4 and App. E.4 of the paper).

Binned actions.  Every dimension of an action a in [-1, 1]^m is cut into L
bins of width Delta = 2 / L with centres c_l.  A configuration of bins
x in {0, ..., L-1}^m is executed as a^d = c(x^d) + u^d with in-bin noise
u^d ~ U(-Delta/2, Delta/2), so the policy is a piecewise constant density over
actions whose entropy is that of the configuration plus m log Delta.  All
entropies here are in these action-density units.

Policy.  One MLP mu_theta(s, x_R) takes the state and the revealed dimensions
R (mask bit, bin centre and one-hot of every dimension; masked dimensions are
zero) and outputs a categorical over the L bins for every dimension.  An
action is drawn by revealing the dimensions one per network call in a
uniformly random order.  The order is stored with the transition
(step_idx[d] = the step at which dimension d was revealed), so that the
likelihood, the entropy and the KL along it are exact sums over the m
categoricals.

Loss (actor_loss).  The Gibbs label of dimension d at a stored transition is

    h^d(l | s, a^{-d}) = softmax_l( Q(s, a^{-d}, c_l) / tau ),

computed from L critic evaluations at the bin centres, with the other
dimensions at the stored action.  For every transition one step k of its
order and one dimension d still masked at step k are drawn, and the
cross-entropy between h^d and mu_theta^d(. | s, x_R) is minimized, where R
holds the dimensions revealed before step k.  Two dual variables complete the
loss: the multiplier kappa of the trust region KL(mu_old || mu_theta) <=
kl_bound along the stored orders, and the temperature tau, the multiplier of
the constraint that the entropy stays above its target.

The random keys are split as in the implementation that produced the runs of
the paper (keys the Gibbs labels do not use included), so that every step
draws the same random numbers as there.
"""

import math

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx


class Binning:
    """L uniform bins per dimension on [low, high); centres c(l) = low + (l + 1/2) Delta."""

    def __init__(self, action_dim: int, num_bins: int, low: float = -1.0, high: float = 1.0):
        self.m = int(action_dim)
        self.L = int(num_bins)
        self.low, self.high = float(low), float(high)
        self.delta = (self.high - self.low) / self.L
        # a NumPy constant, so that the module holds no traced array
        self._centres = (self.low + (np.arange(self.L, dtype=np.float32) + 0.5) * self.delta).astype(np.float32)
        # m log Delta: the entropy of the bins plus this is that of the action density
        self.log_delta_sum = self.m * math.log(self.delta)

    @property
    def centres(self) -> jax.Array:
        return jnp.asarray(self._centres)

    def encode(self, a: jax.Array) -> jax.Array:
        idx = jnp.floor((a - self.low) / self.delta)
        return jnp.clip(idx, 0, self.L - 1).astype(jnp.int32)

    def decode(self, z: jax.Array, key: jax.Array | None = None) -> jax.Array:
        """Bin centres; with `key`, plus uniform in-bin noise."""
        a = self.centres[z]
        if key is not None:
            a = a + (jax.random.uniform(key, z.shape) - 0.5) * self.delta
        return a


def random_schedule(key, batch: int, m: int):
    """A uniformly random order per row: order[b, j] is the j-th revealed
    dimension and step_idx[b, d] the step at which dimension d is revealed."""
    order = jnp.argsort(jax.random.uniform(key, (batch, m)), axis=-1)
    step_idx = jnp.argsort(order, axis=-1)
    return order, step_idx


class MaskedBinnedPolicy(nnx.Module):
    """mu_theta(s, x_R): logits (m, L) of every dimension given the revealed ones,
    and the two dual variables, the temperature tau and the multiplier kappa."""

    def __init__(
        self,
        action_dim: int,
        obs_dim: int,
        num_bins: int = 64,
        hidden: int = 512,
        layers: int = 3,
        head_init_scale: float = 1e-8,
        temp_start: float = 0.01,
        temp_min: float = 1e-8,
        kappa_start: float = 1.0,
        *,
        rngs: nnx.Rngs,
    ):
        self.binning = Binning(action_dim, num_bins)
        self.m, self.L = int(action_dim), int(num_bins)
        self.hidden = int(hidden)
        self.temp_min = float(temp_min)

        # input [s, r, r * c(x), r * onehot(x)] with r the mask bit of every dimension
        d_in = int(obs_dim) + self.m * (2 + self.L)
        trunk = []
        for _ in range(int(layers)):
            trunk += [nnx.Linear(d_in, self.hidden, rngs=rngs), nnx.LayerNorm(self.hidden, rngs=rngs), nnx.gelu]
            d_in = self.hidden
        self.trunk_net = nnx.Sequential(*trunk)

        # m x L logits; the scaled kernel starts the policy at the uniform one
        self.policy_head = nnx.Linear(self.hidden, self.m * self.L, rngs=rngs)
        self.policy_head.kernel.value = self.policy_head.kernel.value * head_init_scale
        self.policy_head.bias.value = jnp.zeros_like(self.policy_head.bias.value)

        self.log_temperature = nnx.Param(jnp.full((1,), math.log(temp_start)))
        self.log_kappa = nnx.Param(jnp.full((1,), math.log(kappa_start)))

    def temperature(self) -> jax.Array:
        return self.temp_min + jnp.exp(self.log_temperature.value)

    def kappa(self) -> jax.Array:
        return jnp.exp(self.log_kappa.value)

    def trunk(self, obs: jax.Array, z: jax.Array, revealed: jax.Array) -> jax.Array:
        """obs (..., o), bins z (..., m), revealed (..., m) bool -> features (..., H)."""
        rev = revealed.astype(jnp.float32)
        onehot = jax.nn.one_hot(z, self.L, dtype=jnp.float32) * rev[..., None]
        x = jnp.concatenate(
            [obs, rev, rev * self.binning.centres[z], onehot.reshape(*z.shape[:-1], self.m * self.L)], axis=-1)
        return self.trunk_net(x)

    def logits(self, feat: jax.Array) -> jax.Array:
        """(..., H) -> (..., m, L)"""
        return self.policy_head(feat).reshape(*feat.shape[:-1], self.m, self.L)

    def logits_own(self, feat_own: jax.Array) -> jax.Array:
        """feat_own (B, m, H), row d the features of the context at which
        dimension d was revealed -> (B, m, L) logits of dimension d there."""
        W = self.policy_head.kernel.value.reshape(self.hidden, self.m, self.L)
        b = self.policy_head.bias.value.reshape(self.m, self.L)
        return jnp.einsum("bih,hil->bil", feat_own, W) + b

    def contexts(self, obs: jax.Array, z: jax.Array, step_idx: jax.Array):
        """Features at the m contexts of stored actions along their orders.
        Returns feat (B, m, H) at the context of step k, feat_own (B, m, H) at
        the context at which dimension d was revealed, and masked (B, m, m),
        masked[b, k, d] = dimension d is masked at step k."""
        B, T = z.shape[0], self.m
        revealed = step_idx[:, None, :] < jnp.arange(T)[None, :, None]  # (B, T, m)
        obs_rep = jnp.broadcast_to(obs[:, None], (B, T, obs.shape[-1])).reshape(B * T, -1)
        z_rep = jnp.broadcast_to(z[:, None], (B, T, self.m)).reshape(B * T, self.m)
        feat = self.trunk(obs_rep, z_rep, revealed.reshape(B * T, self.m)).reshape(B, T, self.hidden)
        feat_own = jnp.take_along_axis(feat, step_idx[..., None], axis=1)
        return feat, feat_own, ~revealed


def sample_action(model: MaskedBinnedPolicy, key, obs: jax.Array, greedy: bool = False, with_entropy: bool = True):
    """Reveal the dimensions one per network call in a uniformly random order.
    greedy: the most likely bin at every step, executed at its centre (the
    evaluation of the paper); otherwise a sample with in-bin noise.
    Returns action (B, m), bins (B, m), step_idx (B, m), the entropy (B,) of
    the categoricals along the sampled order in action-density units and the
    log-likelihood (B,) of the bins along it."""
    B, m = obs.shape[0], model.m
    k_ord, k_tok, k_deq = jax.random.split(key, 3)
    order, step_idx = random_schedule(k_ord, B, m)
    rows = jnp.arange(B)
    z = jnp.zeros((B, m), dtype=jnp.int32)
    entropy = jnp.zeros((B,), dtype=jnp.float32)
    logp = jnp.zeros((B,), dtype=jnp.float32)
    for k in range(m):
        d = order[:, k]
        feat = model.trunk(obs, z, step_idx < k)
        logits = model.logits(feat)[rows, d]  # (B, L)
        lp = jax.nn.log_softmax(logits, axis=-1)
        if greedy:
            zk = jnp.argmax(logits, axis=-1)
        else:
            zk = jax.random.categorical(jax.random.fold_in(k_tok, k), lp, axis=-1)
        z = z.at[rows, d].set(zk)
        if with_entropy:
            entropy = entropy - jnp.sum(jnp.exp(lp) * lp, axis=-1)
            logp = logp + lp[rows, zk]
    action = model.binning.decode(z, None if greedy else k_deq)
    return action, z, step_idx, entropy + model.binning.log_delta_sum, logp


def per_order_logprob(model: MaskedBinnedPolicy, obs, z, step_idx):
    """log mu_theta(x | s, order) of stored bins along the order step_idx."""
    _, feat_own, _ = model.contexts(obs, z, step_idx)
    logp = jax.nn.log_softmax(model.logits_own(feat_own), axis=-1)
    return jnp.sum(jnp.take_along_axis(logp, z[..., None], axis=-1)[..., 0], axis=-1)


def gibbs_label_logits(q_slice, obs, action, dim, temp, centres):
    """Logits Q(s, a^{-d}, c_l) / tau (B, L) of the Gibbs label of dimension
    dim (B,) at the stored actions (B, m), from L critic evaluations per row.
    q_slice(obs (B, o), actions (B, K, m)) -> (B, K)."""
    B, m = action.shape
    L = centres.shape[0]
    onehot = jax.nn.one_hot(dim, m, dtype=bool)
    a_rep = jnp.broadcast_to(action[:, None, :], (B, L, m))
    vals = jnp.broadcast_to(centres[None, :], (B, L))
    a_lab = jnp.where(onehot[:, None, :], vals[..., None], a_rep)  # (B, L, m)
    return q_slice(obs, a_lab) / temp


def actor_loss(model, old_model, batch, q_slice, key, h_star: float, kl_bound: float, temp_max: float):
    """Loss of one policy minibatch and its metrics.
    batch: obs (B, o), action (B, m) the executed action, tokens (B, m) its
    bins, step_idx (B, m) its order.  old_model is the policy that collected
    the data, the centre of the trust region; h_star is the entropy target."""
    obs, z, step_idx, action = batch["obs"], batch["tokens"], batch["step_idx"], batch["action"]
    B, m = z.shape
    _, key_site, _, _ = jax.random.split(key, 4)

    temperature = model.temperature()
    temp_sg = jax.lax.stop_gradient(temperature).squeeze()

    # the categoricals at the contexts of the stored orders, and those of the old policy
    feat, feat_own, masked = model.contexts(obs, z, step_idx)
    logp = jax.nn.log_softmax(model.logits_own(feat_own), axis=-1)  # (B, m, L)
    p = jnp.exp(logp)
    _, feat_own_old, _ = old_model.contexts(obs, z, step_idx)
    logp_old = jax.lax.stop_gradient(jax.nn.log_softmax(old_model.logits_own(feat_own_old), axis=-1))
    p_old = jnp.exp(logp_old)
    kl = jnp.mean(jnp.sum(jnp.sum(p_old * (logp_old - logp), axis=-1), axis=-1))
    entropy_tok = -jnp.sum(p * logp, axis=-1)  # (B, m)
    entropy = jax.lax.stop_gradient(jnp.mean(jnp.sum(entropy_tok, axis=-1) + model.binning.log_delta_sum))

    # one site per transition: a step k of its order, a dimension d masked at k
    k_step, k_dim = jax.random.split(key_site)
    kk = jax.random.randint(k_step, (B,), 0, m)
    masked_k = jnp.take_along_axis(masked, kk[:, None, None], axis=1)[:, 0]  # (B, m)
    feat_k = jnp.take_along_axis(feat, kk[:, None, None], axis=1)[:, 0]  # (B, H)
    gumbel = jax.random.gumbel(k_dim, (B, m))
    d = jnp.argmax(jnp.where(masked_k, gumbel, -jnp.inf), axis=-1)  # (B,)
    logits_d = jnp.take_along_axis(model.logits(feat_k), d[:, None, None], axis=1)[:, 0]  # (B, L)
    lab = jax.lax.stop_gradient(gibbs_label_logits(q_slice, obs, action, d, temp_sg, model.binning.centres))
    label = jax.nn.softmax(lab, axis=-1)
    label_ce = -jnp.mean(jnp.sum(label * jax.nn.log_softmax(logits_d, axis=-1), axis=-1))

    # trust region: the policy pays kappa KL, kappa rises while KL exceeds the bound
    kappa = model.kappa()
    kappa_loss = jnp.mean(-kappa * jax.lax.stop_gradient(kl - kl_bound))
    # temperature: falls while the entropy is above its target and rises while it
    # is below; at temp_max it may only fall (no wind-up)
    gap = entropy - h_star
    gap = jnp.where((temp_sg >= temp_max) & (gap < 0.0), 0.0, gap)
    temp_loss = jnp.mean(temperature * jax.lax.stop_gradient(gap))

    loss = label_ce + jax.lax.stop_gradient(kappa).squeeze() * kl + temp_loss + kappa_loss
    metrics = dict(
        label_ce=label_ce,
        label_entropy=jnp.mean(-jnp.sum(label * jnp.log(label + 1e-20), axis=-1)),
        kl=kl,
        kappa=kappa.mean(),
        temp=temperature.mean(),
        entropy=entropy,
        loss=loss,
    )
    return loss, metrics
