"""The critic of FFM (Blessing et al., 2026), which DIG-IN uses unchanged.

Q_phi(s, a) is a categorical distribution over `atoms` returns on
[vmin, vmax], trained by cross-entropy to HL-Gauss encodings of the TD
targets (Farebrother et al., 2024).  An encoder maps (s, a) to an embedding,
from which one head gives the logits over the atoms and an auxiliary head
predicts the embedding of the next state-action pair.
"""

import jax
import jax.numpy as jnp
from flax import nnx


def hl_gauss(x, num_atoms: int, vmin: float, vmax: float):
    """HL-Gauss encoding of a scalar return: the mass of N(x, sigma^2) on each
    atom's cell, sigma = 0.75 atom spacings, renormalized to [vmin, vmax]."""
    x = jnp.clip(x, vmin, vmax).squeeze()
    width = (vmax - vmin) / (num_atoms - 1)
    edges = jnp.linspace(vmin - width / 2, vmax + width / 2, num_atoms + 1, dtype=jnp.float32)
    sigma = 0.75 * width
    cdf = jax.scipy.special.erf((edges - x) / (jnp.sqrt(2) * sigma))
    return (cdf[1:] - cdf[:-1]) / (cdf[-1] - cdf[0])


class MLP(nnx.Module):
    """layers - 1 blocks Linear-LayerNorm-swish and a linear output layer,
    optionally after a swish of the input."""

    def __init__(self, d_in: int, d_out: int, hidden: int, layers: int, input_activation: bool, *, rngs: nnx.Rngs):
        dims = [d_in] + [hidden] * (layers - 1)
        self.input_activation = input_activation
        self.blocks = [
            nnx.Sequential(nnx.Linear(a, b, rngs=rngs), nnx.LayerNorm(b, rngs=rngs), nnx.swish)
            for a, b in zip(dims[:-1], dims[1:])
        ]
        self.out = nnx.Linear(dims[-1], d_out, rngs=rngs)

    def __call__(self, x):
        if self.input_activation:
            x = nnx.swish(x)
        for block in self.blocks:
            x = block(x)
        return self.out(x)


class Critic(nnx.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden: int = 512,
        encoder_layers: int = 2,
        head_layers: int = 2,
        pred_layers: int = 2,
        num_atoms: int = 151,
        vmin: float = 0.0,
        vmax: float = 150.0,
        *,
        rngs: nnx.Rngs,
    ):
        self.num_atoms, self.vmin, self.vmax = int(num_atoms), float(vmin), float(vmax)
        self.encoder = MLP(obs_dim + action_dim, hidden, hidden, encoder_layers, input_activation=False, rngs=rngs)
        self.head = MLP(hidden, num_atoms, hidden, head_layers, input_activation=True, rngs=rngs)
        # as in FFM, the auxiliary head has one more output, a reward prediction the loss does not use
        self.pred = MLP(hidden, hidden + 1, hidden, pred_layers, input_activation=True, rngs=rngs)
        # the logits start at 40 times the encoding of a zero return (trained, as in FFM)
        self.zero_dist = nnx.Param(hl_gauss(jnp.zeros((1,)), num_atoms, vmin, vmax))

    @property
    def atoms(self) -> jax.Array:
        return jnp.linspace(self.vmin, self.vmax, self.num_atoms, endpoint=True)

    def embed(self, obs, action):
        return self.encoder(jnp.concatenate([obs, action], axis=-1))

    def logits_from(self, emb):
        return self.head(emb) + self.zero_dist.value * 40.0

    def logits(self, obs, action):
        return self.logits_from(self.embed(obs, action))

    def q(self, obs, action):
        """(N, o), (N, m) -> (N,) the mean of the return distribution."""
        return jax.nn.softmax(self.logits(obs, action), axis=-1).dot(self.atoms)

    def q_slice(self, obs, actions):
        """K actions per state: (B, o), (B, K, m) -> (B, K)."""
        B, K, m = actions.shape
        obs_rep = jnp.broadcast_to(obs[:, None, :], (B, K, obs.shape[-1]))
        return self.q(obs_rep.reshape(B * K, -1), actions.reshape(B * K, m)).reshape(B, K)

    def forward(self, obs, action):
        """The embedding, the prediction of the next embedding and Q."""
        emb = self.embed(obs, action)
        q = jax.nn.softmax(self.logits_from(emb), axis=-1).dot(self.atoms)
        return emb, self.pred(emb)[..., 1:], q
