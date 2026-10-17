"""What the lattice benchmark rests on: valid blocked updates on odd
lattices (25 is odd), metered target calls, the evaluation history that
survives preemption, and the extra Ising criteria."""

import json
import math
import os
import subprocess
import sys

import numpy as np
import pytest
import torch

from discrete_sampler.targets import IsingModel
from discrete_sampler.targets.ising.mcmc import IsingMCMC
from discrete_sampler.utils import metrics as M
from discrete_sampler.utils.gibbs import heat_bath_sweeps
from discrete_sampler.utils.lattice import check_independent, independent_groups


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)
    np.random.seed(0)


# ----------------------------------------------------------------------
# Odd lattices
# ----------------------------------------------------------------------
@pytest.mark.parametrize("shape", [(4, 4), (3, 3), (5, 5), (25, 25), (3, 4), (4, 6),
                                   (7,), (10, 25), (3, 3, 3)])
def test_colouring_is_proper_on_every_torus(shape):
    """No two torus neighbours may share a colour, or a blocked sweep updates
    interacting sites together and samples the wrong distribution."""
    c = independent_groups(shape)
    assert c.shape == shape
    assert check_independent(shape, c)
    n_colours = int(c.max()) + 1
    assert n_colours == (2 if all(s % 2 == 0 for s in shape) else 3)


def test_checkerboard_alone_is_wrong_on_an_odd_torus():
    """The failure the colouring exists for: (i + j) % 2 puts the two ends
    of the wrap-around bond in the same class."""
    ii, jj = np.meshgrid(np.arange(5), np.arange(5), indexing="ij")
    assert not check_independent((5, 5), (ii + jj) % 2)


def _exact_moments(target):
    xs = target._index_to_state(torch.arange(target.N ** target.D))
    p = torch.softmax(target.log_prob(xs), 0)
    return ((p * target.magnetization(xs).abs()).sum().item(),
            (p * target.energy_per_site(xs)).sum().item())


@pytest.mark.parametrize("shape", [(3, 3), (3, 4)])
@pytest.mark.parametrize("method", ["metropolis", "heat_bath"])
def test_local_mcmc_matches_enumeration_on_odd_lattices(shape, method):
    target = IsingModel(shape, beta=0.45)
    m_exact, e_exact = _exact_moments(target)
    s = IsingMCMC(target).sample(4000, n_chains=200, method=method, burn_in=200, thin=3)
    assert abs(target.magnetization(s).abs().mean().item() - m_exact) < 0.03
    assert abs(target.energy_per_site(s).mean().item() - e_exact) < 0.05


def test_gibbs_sweeps_match_enumeration_on_an_odd_lattice():
    """The generic heat-bath sweep is what LEAPS mixes with and what the
    uniform-interpolant variants relax the buffer with, so it must be exact too."""
    target = IsingModel((3, 3), beta=0.5)
    m_exact, e_exact = _exact_moments(target)
    x = target.sample_uniform(3000)
    x = heat_bath_sweeps(target, x, n_sweeps=60)
    assert abs(target.magnetization(x).abs().mean().item() - m_exact) < 0.03
    assert abs(target.energy_per_site(x).mean().item() - e_exact) < 0.05


def test_mixed_reference_runs_on_an_odd_lattice():
    target = IsingModel((5, 5), beta=0.44)
    s = IsingMCMC(target).sample(64, n_chains=32, method="mixed", burn_in=5, thin=1)
    assert s.shape == (64, 25)


# ----------------------------------------------------------------------
# Target-call accounting
# ----------------------------------------------------------------------
def test_target_calls_are_metered_per_configuration():
    t = IsingModel((4, 4), beta=0.4)
    x = t.sample_uniform(7)
    t.energy(x)
    t.log_prob(x)                # -> energy
    t.energy_per_site(x)         # -> energy
    t.energy_diff(x)
    t.heat_bath(x)               # -> log_score -> energy_diff
    assert t.n_calls == {"energy": 21, "energy_diff": 14}


# ----------------------------------------------------------------------
# Extra Ising criteria
# ----------------------------------------------------------------------
def test_binder_and_susceptibility_limits():
    t = IsingModel((4, 4), beta=0.5)
    ordered = torch.cat([torch.zeros(50, 16, dtype=torch.long),
                         torch.ones(50, 16, dtype=torch.long)])
    mag = t.magnetization(ordered)
    assert M.binder_cumulant(mag) == pytest.approx(2.0 / 3.0)
    assert M.susceptibility(mag, t.beta, t.D) == pytest.approx(0.0)
    # a symmetric Gaussian m has U4 -> 0
    g = torch.randn(200_000)
    assert abs(M.binder_cumulant(g)) < 0.02
    assert M.specific_heat(t.energy_per_site(ordered), t.beta, t.D) == pytest.approx(0.0)


def test_correlation_length_is_small_on_noise_and_large_on_order():
    t = IsingModel((8, 8), beta=0.5)
    noise = t.sample_uniform(500)
    ordered = torch.ones(500, 64, dtype=torch.long)
    assert M.correlation_length(noise, (8, 8)) < 1.5
    assert M.correlation_length(ordered, (8, 8)) > 8.0


def test_reference_summary_feeds_compare_to_reference():
    t = IsingModel((4, 4), beta=0.4)
    x, x_ref = t.sample_uniform(64), t.sample_uniform(100)
    ref = M.reference_summary(t, x_ref)
    a = M.compare_to_reference(t, x, x_ref, log_w=torch.randn(64))
    b = M.compare_to_reference(t, x, x_ref, log_w=torch.randn(64), ref=ref)
    for k in ("chi_err", "binder_err", "xi_err", "corr_err", "energy_tv", "chi_ref"):
        assert k in a and a[k] == pytest.approx(b[k], abs=1e-12)
    assert all(np.isfinite(v) for v in a.values())


def test_weighted_correlation_reduces_to_unweighted_with_flat_weights():
    t = IsingModel((6, 6), beta=0.4)
    x = t.sample_uniform(40)
    c0 = M.two_point_correlation_weighted(x, (6, 6))
    c1 = M.two_point_correlation_weighted(x, (6, 6), torch.full((40,), 1 / 40.0))
    assert torch.allclose(c0, c1, atol=1e-6)


def test_tempered_weights_hit_the_target_ess():
    """nu* is the largest exponent in [0, 1] whose self-normalized weights
    keep ESS >= xi B; ESS is non-increasing in nu, so the rule is well posed."""
    from discrete_sampler.utils.tempering import log_ess_fraction, tempered_weights

    torch.manual_seed(0)
    lw = torch.randn(256) * 3.0  # badly degenerate raw weights

    ess = [float(log_ess_fraction(lw, n).exp()) for n in
           (0.0, 0.1, 0.25, 0.5, 0.75, 1.0)]
    assert ess[0] == pytest.approx(1.0, abs=1e-5)
    assert all(a >= b - 1e-6 for a, b in zip(ess, ess[1:]))  # monotone

    w, nu, got = tempered_weights(lw, 0.3)
    assert 0.0 < nu < 1.0 and got == pytest.approx(0.3, abs=1e-3)
    assert w.sum() == pytest.approx(1.0, abs=1e-5) and (w > 0).all()
    # the weights are the tempered ones, not something else
    assert torch.allclose(w, torch.softmax(nu * lw, 0), atol=1e-6)

    # already-tame weights are used in full, and a flat batch is uniform
    w1, nu1, _ = tempered_weights(torch.randn(256) * 0.05, 0.3)
    assert nu1 == 1.0
    w2, _, e2 = tempered_weights(torch.zeros(64), 0.3)
    assert e2 == pytest.approx(1.0, abs=1e-5)
    assert torch.allclose(w2, torch.full((64,), 1 / 64), atol=1e-6)


def test_tempered_weighting_reduces_to_the_plain_mean_at_flat_weights():
    """With equal weights the weighted CE is the unweighted one, so the only
    thing the option changes is how the batch is combined."""
    from discrete_sampler.methods.dig_in.masked import MaskedInterpolant

    torch.manual_seed(0)
    mi = MaskedInterpolant(2)
    B, D, N = 8, 6, 2
    logits = torch.randn(B, D, N)
    hb = torch.softmax(torch.randn(B, D, N), -1)
    mask = torch.rand(B, D) < 0.5
    mask[:, 0] = True  # no empty rows, so the per-row mean is well defined

    plain = mi.ce_loss(logits, hb, mask)
    # equal per-endpoint weights reproduce the mean only when every endpoint
    # carries the same number of masked sites, so compare against the
    # per-endpoint mean the weighted branch actually forms
    w = torch.full((B,), 1 / B)
    ce = -(hb * logits.log_softmax(-1)).sum(-1)
    per = (ce * mask.float()).sum(1) / mask.float().sum(1)
    assert mi.ce_loss(logits, hb, mask, w) == pytest.approx(float(per.mean()), abs=1e-6)
    assert plain == pytest.approx(
        float((ce * mask.float()).sum() / mask.float().sum()), abs=1e-6)


def test_weighted_variants_report_the_exponent_and_reject_bad_pairings():
    from discrete_sampler.methods import build_method

    t = IsingModel((4, 4), beta=0.4407)
    d = build_method("dig_in", t, dict(model={"name": "rescnn", "hidden_size": 16,
                                           "n_blocks": 2},
                                    noise="masked", loss="ce",
                                    weighting="tempered", target_ess=0.3,
                                    n_rollout=32, batch_size=16, warm_start=1,
                                    rollout_every=5), device="cpu")
    out = d.step()
    assert 0.0 <= out["temper_nu"] <= 1.0 and out["temper_ess"] >= 0.3 - 1e-3

    # the weight needs a tractable q and an endpoint nothing touched afterwards
    with pytest.raises(ValueError):
        build_method("dig_in", t, dict(model={"name": "rescnn", "hidden_size": 16,
                                           "n_blocks": 2},
                                    noise="uniform", loss="ce",
                                    weighting="tempered"), device="cpu")
    with pytest.raises(ValueError):
        build_method("dig_in", t, dict(model={"name": "rescnn", "hidden_size": 16,
                                           "n_blocks": 2},
                                    noise="masked", loss="ce",
                                    weighting="tempered", buffer_relax=1),
                     device="cpu")


def test_plain_importance_weights_are_the_ess_rule_at_zero():
    """target_ess = 0 accepts nu = 1 on every batch, however degenerate."""
    from discrete_sampler.utils.tempering import tempered_weights

    torch.manual_seed(0)
    lw = torch.randn(256) * 10.0
    w, nu, _ = tempered_weights(lw, 0.0)
    assert nu == 1.0
    assert torch.allclose(w, torch.softmax(lw, 0), atol=1e-6)


def test_hard_labels_are_the_endpoint_token_and_skip_the_label_table():
    from discrete_sampler.methods import build_method

    t = IsingModel((4, 4), beta=0.4407)
    cfg = dict(model={"name": "rescnn", "hidden_size": 16, "n_blocks": 2},
               noise="masked", loss="ce", n_rollout=32, batch_size=16,
               warm_start=1, rollout_every=5, eta=0.5)
    for weighting, ess in (("none", 0.3), ("tempered", 0.8), ("tempered", 0.0)):
        d = build_method("dig_in", t, dict(cfg, labels="hard", weighting=weighting,
                                        target_ess=ess), device="cpu")
        assert "log_score" not in d.buf.extras
        assert ("log_p" in d.buf.extras) == (weighting != "none")
        out = d.step()
        assert math.isfinite(out["loss"])
        if ess == 0.0 and weighting != "none":
            assert out["temper_nu"] == 1.0

    # at eta = 1 the hard-label loss is masked CE against the one-hot endpoint
    torch.manual_seed(0)
    d = build_method("dig_in", t, dict(cfg, labels="hard", eta=1.0), device="cpu")
    x1, _ = d.buf.sample(16)
    torch.manual_seed(1)
    got = d._step_masked(x1, 1.0)
    torch.manual_seed(1)
    a = torch.rand(16)
    xt, mask = d.minterp.sample(x1, a)
    onehot = torch.nn.functional.one_hot(x1, t.N).float()
    want = d.minterp.ce_loss(d._masked_logits(xt, a), onehot, mask)
    assert got.item() == pytest.approx(want.item(), abs=1e-6)

    with pytest.raises(ValueError):
        build_method("dig_in", t, dict(cfg, noise="uniform", labels="hard"),
                     device="cpu")


# ----------------------------------------------------------------------
# Evaluation history
# ----------------------------------------------------------------------
def test_history_is_truncated_to_the_checkpoint(tmp_path):
    from discrete_sampler.train import load_history, write_history

    p = tmp_path / "history.jsonl"
    rows = [{"step": 0, "a": 1}, {"step": 5, "a": 2}, {"step": 10, "a": 3},
            {"step": 5, "a": 4}]  # a re-evaluation after a resume: last wins
    write_history(str(p), rows)
    with open(p, "a") as f:
        f.write('{"step": 15, "a"')  # cut short by a kill
    assert [r["step"] for r in load_history(str(p))] == [0, 5, 10]
    assert load_history(str(p))[1]["a"] == 4
    assert [r["step"] for r in load_history(str(p), max_step=5)] == [0, 5]


def _train(run_dir, *overrides):
    args = [sys.executable, "-m", "discrete_sampler.train", "method=dig_in",
            "target=ising_small", "log_every=2", "eval_every=3",
            "checkpoint_every=100", "eval_samples=32", "final_eval_samples=32",
            "reference.n_samples=64", "reference.burn_in=10",
            f"reference.cache_dir={run_dir / 'ref'}",
            "method.batch_size=8", "method.n_rollout=8", "method.rollout_every=3",
            "model.hidden_size=16", "model.n_blocks=2",
            f"hydra.run.dir={run_dir}", *overrides]
    out = subprocess.run(args, capture_output=True, text=True,
                         env=dict(os.environ, WANDB_MODE="offline"))
    assert out.returncode == 0, out.stdout[-3000:] + out.stderr[-3000:]
    return out.stdout


def test_training_writes_the_history_and_extends_it_on_resume(tmp_path):
    """The history is the benchmark's record: one row per evaluation with
    the cost so far, extended (not restarted, not duplicated) by a relaunch
    in the same directory."""
    run = tmp_path / "run"
    _train(run, "n_steps=6")
    rows = [json.loads(l) for l in open(run / "history.jsonl")]
    assert [r["step"] for r in rows] == [0, 3, 6]
    assert rows[-1]["final"] and not rows[0]["final"]
    for k in ("wall_train_s", "wall_eval_s", "target_calls_train_total",
              "target_calls_energy_diff", "log_Z_err", "elbo_gap", "chi_err",
              "binder_err", "xi_err", "corr_err", "energy_tv", "train/loss"):
        assert k in rows[-1], k
    assert rows[-1]["wall_train_s"] > rows[1]["wall_train_s"] > 0
    assert rows[-1]["target_calls_train_total"] > rows[1]["target_calls_train_total"]
    calls6 = rows[-1]["target_calls_train_total"]

    (run / "result.json").unlink()
    out = _train(run, "n_steps=9")
    assert "[resume]" in out
    rows = [json.loads(l) for r_ in [0] for l in open(run / "history.jsonl")]
    assert [r["step"] for r in rows] == [0, 3, 6, 9]
    # the cost keeps accumulating across the relaunch instead of resetting
    assert rows[-1]["target_calls_train_total"] > calls6
    assert rows[-1]["wall_train_s"] > rows[2]["wall_train_s"]
    res = json.load(open(run / "result.json"))
    assert res["n_steps"] == 9 and res["n_evals"] == 4
    assert res["target_calls_train_total"] == rows[-1]["target_calls_train_total"]
