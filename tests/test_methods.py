"""Smoke tests: every method trains a few steps and evaluates."""

import pytest
import torch

from discrete_sampler.evaluation import evaluate
from discrete_sampler.methods import build_method
from discrete_sampler.targets import IsingModel, PottsModel

# the default backbone, so the smoke tests exercise what a user actually runs
MODEL = {"name": "rescnn", "hidden_size": 16, "n_blocks": 2}

CONFIGS = {
    "uniform_ce": ("dig_in", dict(model=MODEL, noise="uniform", loss="ce",
                                  n_steps_train=8, n_steps_eval=10)),
    "uniform_tsi": ("dig_in", dict(model=MODEL, noise="uniform", loss="tsi",
                                   eta=0.25, n_steps_train=8, n_steps_eval=10)),
    "masked_ce": ("dig_in", dict(model=MODEL, noise="masked", loss="ce")),
    "masked_tsi": ("dig_in", dict(model=MODEL, noise="masked", loss="tsi", eta=0.5)),
    # the three masked arms of the paper's lattice table (experiments/lattice.py)
    "dig_in": ("dig_in", dict(model=MODEL, noise="masked", loss="ce", eta=0.1,
                              weighting="tempered", target_ess=0.8)),
    "mdns": ("dig_in", dict(model=MODEL, noise="masked", loss="ce", labels="hard",
                            weighting="tempered", target_ess=0.0, eta=1.0)),
    "pdns": ("dig_in", dict(model=MODEL, noise="masked", loss="ce", labels="hard",
                            weighting="tempered", target_ess=0.8, eta=1.0)),
    "leaps": ("leaps", dict(model={"channels": 4, "kernel_sizes": [3, 5]},
                            delta_t=0.2, n_anneal=6, n_walkers=16,
                            loss_batch=32, buffer_cycle=4)),
    "asbs": ("asbs", dict(model=MODEL, steps_controller=4, steps_corrector=3,
                          sampling_steps=8, num_repl=2, resample_every=4)),
    "asbs_memoryless": ("asbs", dict(model=MODEL, steps_controller=4,
                                     steps_corrector=0, sampling_steps=8,
                                     num_repl=2, resample_every=4,
                                     memoryless=True)),
}


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


@pytest.mark.parametrize("name", list(CONFIGS))
@pytest.mark.parametrize("target_fn", [
    lambda: IsingModel((4, 4), beta=0.4407),
    lambda: PottsModel((4, 4), q=3, beta=0.9),
], ids=["ising", "potts"])
def test_method_steps_and_eval(name, target_fn):
    target = target_fn()
    key, cfg = CONFIGS[name]
    cfg = dict(cfg)
    if key != "leaps":
        cfg.setdefault("batch_size", 32)
    if key == "dig_in":
        cfg.setdefault("rollout_every", 4)
    if key in ("dig_in", "asbs"):
        cfg.setdefault("n_rollout", 32)
    m = build_method(key, target, cfg, device="cpu")
    for _ in range(9):
        out = m.step()
    assert torch.isfinite(torch.tensor(out["loss"]))
    x, logw = m.sample_with_weights(64)
    assert x.shape == (64, target.D)
    assert x.min() >= 0 and x.max() < target.N
    assert torch.isfinite(logw).all()
    ev = evaluate(m, n_samples=64)
    assert all(torch.isfinite(torch.tensor(v)).item() for v in ev.values())
    # checkpoint round-trip
    m.load_state_dict(m.state_dict())


def test_dig_in_refuses_non_finite_updates():
    """An infinite gradient must not reach the weights.

    bregman_kl carries an unbounded exp(logit) term, so one pathological batch
    can overflow fp32; applying that update leaves NaN weights and the next
    rollout dies inside the sampler with a device-side assert instead of
    anything diagnosable.
    """
    torch.manual_seed(0)
    target = IsingModel((3, 3), beta=0.4)
    m = build_method("dig_in", target, dict(model=MODEL, noise="uniform", loss="tsi",
                                         eta=0.25, n_steps_train=4, n_steps_eval=4,
                                         batch_size=8, n_rollout=8, rollout_every=100),
                     device="cpu")
    m.step()
    good = [p.detach().clone() for p in m.net.parameters()]

    # drive the head to produce a huge logit, so exp(logit) overflows
    with torch.no_grad():
        last = [p for p in m.net.parameters() if p.dim() >= 1][-1]
        last.fill_(1e30)
    before = m.n_skipped
    out = m.step()

    assert m.n_skipped == before + 1, "non-finite update was not refused"
    assert all(torch.isfinite(p).all() for p in m.net.parameters()), "weights poisoned"
    assert out["skipped_updates"] == m.n_skipped
    # and the sampler still runs rather than tripping an assert
    with torch.no_grad():
        for p, g in zip(m.net.parameters(), good):
            p.copy_(g)
    x, log_w = m.sample_with_weights(8)
    assert torch.isfinite(log_w).all()


def test_asbs_refuses_non_finite_updates():
    """Same guard as DIG-IN: the Bregman target-matching losses exponentiate
    network outputs, so an overflowing batch must not reach the weights."""
    torch.manual_seed(0)
    target = IsingModel((3, 3), beta=0.4)
    m = build_method("asbs", target, dict(model=MODEL, steps_controller=4,
                                          steps_corrector=3, batch_size=8,
                                          n_rollout=8, num_repl=2,
                                          sampling_steps=4, resample_every=100),
                     device="cpu")
    m.step()
    with torch.no_grad():
        last = [p for p in m.controller.parameters() if p.dim() >= 1][-1]
        last.fill_(1e30)
    before = m.n_skipped
    out = m.step()
    assert m.n_skipped == before + 1, "non-finite update was not refused"
    assert all(torch.isfinite(p).all() for p in m.controller.parameters())
    assert out["skipped_updates"] == m.n_skipped


# ----------------------------------------------------------------------
# Cached exploration labels
# ----------------------------------------------------------------------
def _dig_in(target, **over):
    cfg = dict(model=MODEL, noise="masked", loss="ce", batch_size=16,
               n_rollout=16, rollout_every=4, buffer_capacity=64,
               warm_start=1, unmask_steps=4)
    cfg.update(over)
    return build_method("dig_in", target, cfg, device="cpu")


@pytest.mark.parametrize("N,shape", [(2, (4, 4)), (3, (4, 4))])
def test_cached_labels_survive_the_symmetry_augmentation(N, shape):
    """`symmetrize` relabels the alphabet of the buffered endpoints; the
    cached label table has to be relabeled with them.  Permuting the states
    but not their labels would train on mismatched targets silently."""
    torch.manual_seed(0)
    target = (IsingModel(shape, beta=0.4407) if N == 2
              else PottsModel(shape, q=N, beta=0.9))
    m = _dig_in(target, symmetrize=True)
    x = target.sample_uniform(32)
    xs, ls = m._random_symmetry(x, target.log_score(x))
    assert not torch.equal(xs, x), "the augmentation did nothing"
    assert torch.allclose(ls, target.log_score(xs), atol=1e-6)


def test_cached_labels_do_not_change_training():
    """Caching the label table at push time is an identity, not an
    approximation: the same seed must give the same loss either way."""
    for noise, loss in [("masked", "ce"), ("uniform", "ce"), ("masked", "tsi")]:
        losses = []
        for cache in (True, False):
            torch.manual_seed(0)
            target = IsingModel((4, 4), beta=0.4407)
            m = _dig_in(target, noise=noise, loss=loss, cache_labels=cache,
                     symmetrize=True, n_steps_train=4, n_steps_eval=4,
                     eta=0.5 if loss == "tsi" else 1.0)
            losses.append([m.step()["loss"] for _ in range(6)])
        assert losses[0] == pytest.approx(losses[1], rel=1e-6), (noise, loss)


def test_cached_labels_track_the_annealing_ramp():
    """The cache holds the raw log-score table and the inverse-temperature
    ramp is applied at use time, so annealing still moves the targets."""
    torch.manual_seed(0)
    target = IsingModel((4, 4), beta=0.4407)
    m = _dig_in(target, anneal_steps=50, anneal_start=0.2)
    first = m.step()["anneal_scale"]
    for _ in range(20):
        out = m.step()
    assert out["anneal_scale"] > first


def test_uniform_warm_start_is_high_entropy():
    """The default pre-fill must not hand the sampler an ordered mode."""
    torch.manual_seed(0)
    target = IsingModel((4, 4), beta=0.4407)
    m = _dig_in(target, warm_start=2, n_rollout=64, rollout_every=1000,
             buffer_capacity=256)
    x = m.buf.states
    assert x.shape[0] == 128
    # uniform endpoints at D=16 give E|m| ~ 0.2; mode seeds give exactly 1
    assert target.magnetization(x).abs().mean() < 0.4, "pre-fill is mode-seeded"
    # the ablation setting still gives the old zero-temperature seeds
    torch.manual_seed(0)
    m2 = _dig_in(target, warm_start=2, n_rollout=64, rollout_every=1000,
              buffer_capacity=256, warm_start_kind="mode")
    xs = m2.buf.states
    assert (xs == xs[:, :1]).all(), "mode seeds must be constant configurations"
