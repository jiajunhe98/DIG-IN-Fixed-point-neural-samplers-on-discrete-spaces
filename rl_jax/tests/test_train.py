"""The environment wrapper and an end-to-end run of the trainer on a small
batch (needs MuJoCo Playground; a few minutes on a CPU, mostly compilation)."""

import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("mujoco_playground")

from hydra import compose, initialize_config_dir  # noqa: E402

from dig_in_rl.envs import DMCEnv  # noqa: E402
from dig_in_rl.train import make_train  # noqa: E402

CONFIGS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs")


def test_env():
    env = DMCEnv("CartpoleBalance", num_envs=4, episode_length=5)
    obs, state = env.reset(jax.random.split(jax.random.PRNGKey(0), 4))
    assert obs.shape == (4, env.obs_dim) and env.action_dim == 1
    np.testing.assert_allclose(state.obs_mean, jnp.mean(state.inner.obs, axis=0))
    total = jnp.zeros(4)
    for t in range(5):
        obs, state, reward, done, info = env.step(state, jnp.zeros((4, 1)))
        total = total + reward
        assert bool(jnp.all(done)) == (t == 4)  # the time limit ends every episode at step 5
    np.testing.assert_allclose(info["returned_episode_return"], total, rtol=1e-6)
    assert jnp.all(state.truncated == 1.0) and int(state.obs_count) == 4 * 6
    assert jnp.all(state.episode_return == 0.0)


def test_train_smoke():
    with initialize_config_dir(config_dir=CONFIGS, version_base=None):
        cfg = compose("dmc", overrides=[
            "env.name=CartpoleBalance", "total_steps=1024", "num_envs=16", "num_steps=16", "minibatches=4",
            "epochs=2", "num_evals=2", "env.episode_length=50", "eval_stochastic=true"])
    init, block, num_blocks = make_train(cfg, DMCEnv("CartpoleBalance", 16, 50), DMCEnv("CartpoleBalance", 16, 50))
    assert num_blocks == 2

    @jax.jit
    def run(key):
        ts, keys = init(key)
        return jax.lax.scan(block, ts, keys)

    ts, metrics = run(jax.random.PRNGKey(0))
    assert metrics["eval/episode_return"].shape == (2, 1)
    for k, v in metrics.items():
        assert np.all(np.isfinite(np.asarray(v))), k
    assert int(ts.iteration[0]) == 4 and int(ts.time_steps[0]) == 1024
    assert np.all(np.asarray(metrics["eval/num_episodes"]) == 16)
    assert np.all(np.asarray(metrics["train/kl"]) >= 0)
