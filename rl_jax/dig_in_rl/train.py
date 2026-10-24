"""DIG-IN for maximum entropy RL (Sec. 5.4 and App. E.4 of the paper).

    python -m dig_in_rl.train env.name=CheetahRun seed=0      # from rl_jax/

An on-policy actor-critic loop.  Every iteration
  1. collects num_envs x num_steps transitions with the current policy, which
     replace the buffer; every transition stores the order in which its action
     was generated;
  2. fits the critic for `epochs` epochs of `minibatches` minibatches to soft
     TD(lambda) returns with the reward r_t + gamma tau H_{t+1}, where H_{t+1}
     is the entropy of the policy along the order sampled at s_{t+1};
  3. fits the policy for `epochs` epochs of `minibatches` minibatches to the
     Gibbs labels of the updated critic (policy.actor_loss), within a trust
     region around the policy that collected the data, while the temperature
     holds the entropy at its target.
The target critic of FFM is a copy of the critic after every update, so the
rollouts read the critic itself.  Before every block of total_steps /
(num_envs num_steps num_evals) iterations the policy is evaluated with greedy
decoding on num_envs episodes.

The run directory (hydra.run.dir, by default runs/dmc/<task>/s<seed>) gets
metrics.npz, every metric at each evaluation as an array of shape
(num_evals, num_seeds); eval/episode_return is the return the paper reports.
num_seeds > 1 trains that many independent runs in one process (vmapped);
the runs of the paper are num_seeds=1, seed=0..9.
"""

import logging
import os
import time

import hydra
import jax
import numpy as np
import optax
from flax import nnx, struct
from jax import numpy as jnp
from omegaconf import DictConfig, OmegaConf

from dig_in_rl.critic import Critic, hl_gauss
from dig_in_rl.envs import DMCEnv, EnvState
from dig_in_rl.policy import MaskedBinnedPolicy, actor_loss, sample_action

log = logging.getLogger(__name__)


class Transition(struct.PyTreeNode):
    obs: jax.Array
    action: jax.Array  # the executed action in [-1, 1]^m
    tokens: jax.Array  # its bins
    step_idx: jax.Array  # the step at which each dimension was revealed
    reward: jax.Array
    soft_reward: jax.Array  # r + gamma tau H(s')
    next_emb: jax.Array  # the critic's embedding of (s', a'), target of its auxiliary head
    value: jax.Array  # Q(s', a')
    done: jax.Array
    truncated: jax.Array


class TrainState(struct.PyTreeNode):
    actor: nnx.TrainState
    critic: nnx.TrainState
    iteration: int
    time_steps: int
    env_state: EnvState
    obs: jax.Array


def make_train(cfg: DictConfig, env: DMCEnv, eval_env: DMCEnv, log_fn=None):
    """-> (init(key) -> (state, keys), block(state, key) -> (state, metrics),
    number of blocks); one block is an evaluation and the iterations after it."""
    pc, cc = cfg.policy, cfg.critic
    m = env.action_dim
    n_batch = cfg.num_steps * cfg.num_envs
    mb_size = n_batch // cfg.minibatches
    h_star = cfg.temperature.target_entropy * m
    iters_per_eval = int((cfg.total_steps / n_batch) // cfg.num_evals)
    num_iterations = cfg.total_steps // n_batch
    num_blocks = num_iterations // iters_per_eval + int(num_iterations % iters_per_eval != 0)

    def sample(actor, key, obs, greedy=False, with_entropy=True):
        return sample_action(actor, key, obs, greedy=greedy, with_entropy=with_entropy)[:4]

    def init_one(key) -> TrainState:
        key, model_key = jax.random.split(key)
        actor = MaskedBinnedPolicy(
            m, env.obs_dim, num_bins=pc.bins, hidden=pc.hidden, layers=pc.layers,
            head_init_scale=pc.head_init_scale, temp_start=cfg.temperature.start, temp_min=cfg.temperature.min,
            kappa_start=cfg.trust_region.kappa_start, rngs=nnx.Rngs(model_key))
        critic = Critic(
            env.obs_dim, m, hidden=cc.hidden, encoder_layers=cc.encoder_layers, head_layers=cc.head_layers,
            pred_layers=cc.pred_layers, num_atoms=cc.atoms, vmin=cc.vmin, vmax=cc.vmax, rngs=nnx.Rngs(model_key))
        # one Adam for the policy and its two dual variables
        actor_ts = nnx.TrainState.create(
            graphdef=nnx.graphdef(actor), params=nnx.state(actor), tx=optax.adam(cfg.lr))
        critic_ts = nnx.TrainState.create(
            graphdef=nnx.graphdef(critic), params=nnx.state(critic),
            tx=optax.chain(optax.clip_by_global_norm(cc.max_grad_norm), optax.adam(cfg.lr)))

        key, env_key = jax.random.split(key)
        obs, env_state = env.reset(jax.random.split(env_key, cfg.num_envs))
        key, steps_key = jax.random.split(key)
        env_state = env.randomize_steps(env_state, steps_key)
        return TrainState(actor=actor_ts, critic=critic_ts, iteration=0, time_steps=0, env_state=env_state, obs=obs)

    def evaluate(key, ts: TrainState, greedy: bool):
        """Mean return of the first episode of each of num_envs environments,
        with the observation statistics of training."""
        actor = nnx.merge(ts.actor.graphdef, ts.actor.params)

        def step(carry, _):
            key, env_state, obs = carry
            key, act_key, _ = jax.random.split(key, 3)
            action, *_ = sample(actor, act_key, obs, greedy=greedy, with_entropy=False)
            obs, env_state, _, _, info = eval_env.step(env_state, action, update_stats=False)
            return (key, env_state, obs), info

        key, init_key = jax.random.split(key)
        obs, env_state = eval_env.reset(jax.random.split(init_key, cfg.num_envs), stats=ts.env_state)
        _, infos = jax.lax.scan(step, (key, env_state, obs), None, length=cfg.env.episode_length)
        dones = infos["returned_episode"]
        first = jnp.argmax(dones, axis=0)
        has_done = jnp.any(dones, axis=0)
        cols = jnp.arange(dones.shape[1])
        ret = infos["returned_episode_return"][first, cols]
        length = infos["returned_episode_length"][first, cols]
        return {
            "episode_return": jnp.mean(ret, where=has_done),
            "episode_return_std": jnp.std(ret, where=has_done),
            "episode_length": jnp.mean(length, where=has_done),
            "num_episodes": jnp.sum(has_done),
        }

    def rollout(key, ts: TrainState):
        actor = nnx.merge(ts.actor.graphdef, ts.actor.params)
        critic = nnx.merge(ts.critic.graphdef, ts.critic.params)

        def step(carry, _):
            key, env_state, obs, act = carry
            key, next_key, _ = jax.random.split(key, 3)
            # the action sampled at the previous step, for its bootstrap value
            action, tokens, step_idx, _ = act
            next_obs, next_env_state, reward, done, _ = env.step(env_state, action)
            action = jnp.clip(action, -1.0, 1.0)
            next_act = sample(actor, next_key, next_obs)
            next_action, _, _, next_entropy = next_act
            next_emb, _, value = critic.forward(next_obs, next_action)
            soft_reward = reward + cfg.gamma * next_entropy * actor.temperature().squeeze()
            transition = Transition(
                obs=obs, action=action, tokens=tokens, step_idx=step_idx, reward=reward, soft_reward=soft_reward,
                next_emb=next_emb, value=value, done=done, truncated=next_env_state.truncated)
            return (key, next_env_state, next_obs, next_act), transition

        key, init_key = jax.random.split(key)
        act = sample(actor, init_key, ts.obs)
        (_, env_state, obs, _), batch = jax.lax.scan(
            step, (key, ts.env_state, ts.obs, act), None, length=cfg.num_steps)
        return batch, ts.replace(env_state=env_state, obs=obs, time_steps=ts.time_steps + n_batch)

    def td_lambda(batch: Transition):
        """Soft TD(lambda) returns, backwards in time; an episode that ends at
        the time limit is bootstrapped with Q(s', a')."""
        def backward(carry, t):
            ret, truncated = carry
            mixed = cfg.lmbda * ret + (1 - cfg.lmbda) * t.value
            ret = t.soft_reward + cfg.gamma * jnp.where(truncated, t.value, (1.0 - t.done) * mixed)
            return (ret, t.truncated), ret

        _, returns = jax.lax.scan(backward, (batch.value[-1], jnp.ones_like(batch.truncated[0])), batch, reverse=True)
        return returns

    def learn(key, ts: TrainState, batch: Transition):
        targets = td_lambda(batch)
        data = jax.tree.map(lambda x: x.reshape((n_batch, *x.shape[2:])), (batch, targets))
        old_actor = nnx.merge(ts.actor.graphdef, ts.actor.params)  # the centre of the trust region

        def critic_epoch(ts, key):
            def minibatch(ts, idx):
                mb, target = jax.tree.map(lambda x: jnp.take(x, idx, axis=0), data)

                def loss_fn(params):
                    critic = nnx.merge(ts.critic.graphdef, params)
                    target_cat = jax.vmap(hl_gauss, in_axes=(0, None, None, None))(target, cc.atoms, cc.vmin, cc.vmax)
                    ce = optax.softmax_cross_entropy(critic.logits(mb.obs, mb.action), target_cat)
                    _, pred, q = critic.forward(mb.obs, mb.action)
                    aux = jnp.mean((1 - mb.done.reshape(-1, 1)) * optax.squared_error(pred, mb.next_emb), axis=-1)
                    # transitions that end at the time limit see the next episode's start: no loss
                    loss = jnp.mean((1.0 - mb.truncated) * (ce + cc.aux_weight * aux))
                    return loss, dict(critic_loss=loss, critic_ce=jnp.mean(ce), aux_loss=jnp.mean(aux),
                                      q=jnp.mean(q), td_target=jnp.mean(target))

                (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(ts.critic.params)
                return ts.replace(critic=ts.critic.apply_gradients(grads)), metrics

            key, shuffle_key = jax.random.split(key)
            idx = jax.random.permutation(shuffle_key, n_batch).reshape((cfg.minibatches, mb_size))
            ts, metrics = jax.lax.scan(minibatch, ts, idx)
            return ts, jax.tree.map(lambda x: x.mean(0), metrics)

        def actor_epoch(ts, key):
            def minibatch(ts, idx_key):
                idx, key = idx_key
                mb, _ = jax.tree.map(lambda x: jnp.take(x, idx, axis=0), data)
                critic = nnx.merge(ts.critic.graphdef, ts.critic.params)
                arrays = dict(obs=mb.obs, action=mb.action, tokens=mb.tokens, step_idx=mb.step_idx)

                def loss_fn(params):
                    actor = nnx.merge(ts.actor.graphdef, params)
                    return actor_loss(actor, old_actor, arrays, critic.q_slice, key, h_star,
                                      cfg.trust_region.kl_bound, cfg.temperature.max)

                (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(ts.actor.params)
                return ts.replace(actor=ts.actor.apply_gradients(grads)), metrics

            key, shuffle_key = jax.random.split(key)
            idx = jax.random.permutation(shuffle_key, n_batch).reshape((cfg.minibatches, mb_size))
            ts, metrics = jax.lax.scan(minibatch, ts, (idx, jax.random.split(key, cfg.minibatches)))
            return ts, jax.tree.map(lambda x: x.mean(0), metrics)

        key, critic_key = jax.random.split(key)
        ts, critic_metrics = jax.lax.scan(critic_epoch, ts, jax.random.split(critic_key, cfg.epochs))
        key, actor_key = jax.random.split(key)
        ts, actor_metrics = jax.lax.scan(actor_epoch, ts, jax.random.split(actor_key, cfg.epochs))
        metrics = jax.tree.map(lambda x: x[-1], {**actor_metrics, **critic_metrics})
        metrics["reward"] = jnp.mean(batch.reward)
        return ts, metrics

    def block_one(key, ts: TrainState):
        def iteration(ts, key):
            _, rollout_key, learn_key = jax.random.split(key, 3)
            batch, ts = rollout(rollout_key, ts)
            ts, metrics = learn(learn_key, ts, batch)
            return ts.replace(iteration=ts.iteration + 1), metrics

        train_key, eval_key = jax.random.split(key)
        # eval/ greedy decoding (the paper), eval_stochastic/ the sampler
        eval_key, stochastic_key, _ = jax.random.split(eval_key, 3)
        metrics = {"time_step": ts.time_steps}
        metrics.update({f"eval/{k}": v for k, v in evaluate(eval_key, ts, greedy=True).items()})
        if cfg.eval_stochastic:
            metrics.update({f"eval_stochastic/{k}": v for k, v in evaluate(stochastic_key, ts, greedy=False).items()})
        ts, train_metrics = jax.lax.scan(iteration, ts, jax.random.split(train_key, iters_per_eval))
        metrics.update({f"train/{k}": v[-1] for k, v in train_metrics.items()})
        metrics["train/episode_return"] = ts.env_state.returned_episode_return.mean()
        return ts, metrics

    def init(key):
        key, init_key = jax.random.split(key)
        ts = jax.vmap(init_one)(jax.random.split(init_key, cfg.num_seeds))
        return ts, jax.random.split(key, num_blocks)

    def block(ts, key):
        key, subkey = jax.random.split(key)
        ts, metrics = jax.vmap(block_one)(jax.random.split(subkey, cfg.num_seeds), ts)
        if log_fn is not None:
            jax.debug.callback(log_fn, metrics)
        return ts, metrics

    return init, block, num_blocks


def train(cfg: DictConfig):
    """Train with the settings in cfg and return the metrics, each (num_evals, num_seeds)."""
    env = DMCEnv(cfg.env.name, cfg.num_envs, cfg.env.episode_length, cfg.env.normalize_obs)
    eval_env = DMCEnv(cfg.env.name, cfg.num_envs, cfg.env.episode_length, cfg.env.normalize_obs)
    clock = {"t": time.perf_counter(), "steps": 0}

    def log_fn(metrics):
        steps = int(np.max(metrics["time_step"]))
        now = time.perf_counter()
        sps = (steps - clock["steps"]) / (now - clock["t"]) if steps > clock["steps"] else 0.0
        clock.update(t=now, steps=steps)
        mean = lambda k: float(np.mean(metrics[k]))
        msg = f"step={steps} return={mean('eval/episode_return'):.1f}"
        if "eval_stochastic/episode_return" in metrics:
            msg += f" stochastic={mean('eval_stochastic/episode_return'):.1f}"
        msg += (f" | entropy={mean('train/entropy'):.3g} temp={mean('train/temp'):.3g}"
                f" kl={mean('train/kl'):.3g} kappa={mean('train/kappa'):.3g} q={mean('train/q'):.4g}"
                f" sps={sps:.0f}")
        log.info(msg)

    init, block, num_blocks = make_train(cfg, env, eval_env, log_fn)
    log.info(f"{cfg.env.name}: m={env.action_dim} obs_dim={env.obs_dim}, {num_blocks} evaluations, "
             f"{cfg.total_steps // (cfg.num_envs * cfg.num_steps)} iterations")

    @jax.jit
    def run(key):
        ts, keys = init(key)
        return jax.lax.scan(block, ts, keys)

    start = time.perf_counter()
    _, metrics = run(jax.random.PRNGKey(cfg.seed))
    metrics = jax.device_get(metrics)
    log.info(f"training took {time.perf_counter() - start:.0f} s")
    return metrics


@hydra.main(version_base=None, config_path="../configs", config_name="dmc")
def main(cfg: DictConfig):
    log.info(OmegaConf.to_yaml(cfg))
    metrics = train(cfg)
    np.savez("metrics.npz", **{k: np.asarray(v) for k, v in metrics.items()})
    log.info(f"final return {float(np.mean(metrics['eval/episode_return'][-1])):.1f}, "
             f"metrics in {os.path.join(os.getcwd(), 'metrics.npz')}")


if __name__ == "__main__":
    main()
