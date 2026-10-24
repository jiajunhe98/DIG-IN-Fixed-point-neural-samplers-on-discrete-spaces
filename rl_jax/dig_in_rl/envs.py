"""The DeepMind Control tasks of MuJoCo Playground as a batched JAX environment.

DMCEnv runs `num_envs` copies of a task through MuJoCo Playground's Brax
training wrapper (episodes of `episode_length` steps, automatic reset to the
initial state of the episode) and adds what the runs of the paper used:
actions clipped to [-0.999, 0.999], observations normalized by running
statistics over all environments, and the return of the last finished
episode of every environment.
"""

from typing import Any

import jax
import jax.numpy as jnp
from flax import struct

#: the 20 tasks of the paper, with their action dimension m
TASKS = {
    "AcrobotSwingup": 1, "AcrobotSwingupSparse": 1, "BallInCup": 2, "CartpoleBalance": 1,
    "CartpoleBalanceSparse": 1, "CartpoleSwingup": 1, "CartpoleSwingupSparse": 1, "CheetahRun": 6,
    "FingerSpin": 2, "FingerTurnEasy": 2, "FingerTurnHard": 2, "FishSwim": 5, "HopperHop": 4,
    "HopperStand": 4, "PendulumSwingup": 1, "ReacherEasy": 2, "ReacherHard": 2, "WalkerRun": 6,
    "WalkerStand": 6, "WalkerWalk": 6,
}


class EnvState(struct.PyTreeNode):
    inner: Any  # the Playground state of all environments
    obs_mean: jax.Array
    obs_var: jax.Array
    obs_count: jax.Array
    episode_return: jax.Array
    episode_length: jax.Array
    returned_episode_return: jax.Array  # of the last finished episode
    returned_episode_length: jax.Array
    truncated: jax.Array  # the last step ended an episode at the time limit


def _update_stats(mean, var, count, obs):
    """Running mean and variance over all observations seen (Chan et al.)."""
    batch_mean = jnp.mean(obs, axis=0)
    batch_var = jnp.var(obs, axis=0)
    batch_count = obs.shape[0]
    delta = batch_mean - mean
    tot_count = count + batch_count
    new_mean = mean + delta * batch_count / tot_count
    m2 = var * count + batch_var * batch_count + jnp.square(delta) * count * batch_count / tot_count
    return new_mean, m2 / tot_count


class DMCEnv:
    def __init__(self, name: str, num_envs: int, episode_length: int = 1000, normalize: bool = True):
        if name not in TASKS:
            raise ValueError(f"unknown task {name!r}; the paper's tasks are {sorted(TASKS)}")
        from mujoco_playground import registry  # here, so that TASKS imports without MuJoCo
        from mujoco_playground._src.wrapper import wrap_for_brax_training

        env = registry.load(name, config=registry.get_default_config(name))
        self.env = wrap_for_brax_training(env, episode_length=episode_length, action_repeat=1)
        self.num_envs = int(num_envs)
        self.episode_length = int(episode_length)
        self.normalize = bool(normalize)
        self.obs_dim = int(self.env.observation_size)
        self.action_dim = int(self.env.action_size)

    def _norm(self, obs, state: EnvState):
        if not self.normalize:
            return obs
        return (obs - state.obs_mean) / jnp.sqrt(state.obs_var + 1e-2)

    def reset(self, keys, stats: EnvState | None = None):
        """keys (num_envs, 2).  The observation statistics are those of the
        first observations, or copied from `stats` (evaluation)."""
        inner = self.env.reset(keys)
        obs = inner.obs
        if stats is None:
            mean, var, count = jnp.mean(obs, axis=0), jnp.var(obs, axis=0), obs.shape[0]
        else:
            mean, var, count = stats.obs_mean, stats.obs_var, stats.obs_count
        n = self.num_envs
        state = EnvState(
            inner=inner, obs_mean=mean, obs_var=var, obs_count=count,
            episode_return=jnp.zeros((n,)), episode_length=jnp.zeros((n,), dtype=jnp.int32),
            returned_episode_return=jnp.zeros((n,)), returned_episode_length=jnp.zeros((n,), dtype=jnp.int32),
            truncated=jnp.ones((n,), dtype=jnp.float32),
        )
        return self._norm(obs, state), state

    def step(self, state: EnvState, action, update_stats: bool = True):
        """-> obs, state, reward, done, info (the episode bookkeeping of this step)."""
        inner = self.env.step(state.inner, jnp.clip(action, -0.999, 0.999))
        obs, reward, done = inner.obs, inner.reward, inner.done > 0.5
        ep_return = state.episode_return + reward
        ep_length = state.episode_length + 1
        returned_return = state.returned_episode_return * (1 - done) + ep_return * done
        returned_length = state.returned_episode_length * (1 - done) + ep_length * done
        if update_stats:
            mean, var = _update_stats(state.obs_mean, state.obs_var, state.obs_count, obs)
            count = state.obs_count + obs.shape[0]
        else:
            mean, var, count = state.obs_mean, state.obs_var, state.obs_count
        state = EnvState(
            inner=inner, obs_mean=mean, obs_var=var, obs_count=count,
            episode_return=ep_return * (1 - done), episode_length=ep_length * (1 - done),
            returned_episode_return=returned_return, returned_episode_length=returned_length,
            truncated=inner.info["truncation"],
        )
        info = {"returned_episode": done, "returned_episode_return": returned_return,
                "returned_episode_length": returned_length}
        return self._norm(obs, state), state, reward, done, info

    def randomize_steps(self, state: EnvState, key) -> EnvState:
        """Start every environment at a random step of its first episode, so
        that the episodes of the environments end at different times."""
        steps = state.inner.info["steps"]
        steps = jax.random.randint(key, steps.shape, 0, self.episode_length).astype(jnp.float32)
        return state.replace(inner=state.inner.replace(info={**state.inner.info, "steps": steps}))
