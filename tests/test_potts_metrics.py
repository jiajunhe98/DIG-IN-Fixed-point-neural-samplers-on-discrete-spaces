"""The evaluation block on a q-state alphabet.

Two things are checked: the lattice observables that were written for Ising
spins now take the Potts form for q > 2 and are unchanged at q = 2, and every
evaluation carries the mode-resolving diagnostics, without which a collapsed
Potts sampler is indistinguishable from a correct one.
"""

import pytest
import torch

from dig_in.targets import IsingModel, PottsModel
from dig_in.utils import metrics as M


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def test_q2_path_reproduces_the_spin_field_exactly():
    """q = 2 goes through the simplex/delta code and must not move a number."""
    shape = (6, 6)
    x = torch.randint(0, 2, (64, 36))
    w = torch.softmax(torch.randn(64), 0)

    assert M.structure_factor(x, shape) == pytest.approx(
        M.structure_factor(x, shape, q=2), abs=1e-9)
    assert M.correlation_length(x, shape, w) == pytest.approx(
        M.correlation_length(x, shape, w, q=2), abs=1e-9)
    assert torch.allclose(M.two_point_correlation_weighted(x, shape, w),
                          M.two_point_correlation_weighted(x, shape, w, q=2),
                          atol=1e-9)


def test_potts_two_point_function_is_the_delta_correlator():
    shape = (8, 8)
    q = 3
    # every site in one state: delta(x_0, x_r) = 1 at every separation
    same = torch.zeros(16, 64, dtype=torch.long)
    c = M.two_point_correlation_weighted(same, shape, q=q)
    assert torch.allclose(c, torch.full_like(c, 1.0 - 1.0 / q), atol=1e-6)

    # independent uniform colours: C(0) = 1 - 1/q, C(r > 0) ~ 0
    rnd = torch.randint(0, q, (2048, 64))
    c = M.two_point_correlation_weighted(rnd, shape, q=q)
    assert c[0].item() == pytest.approx(1.0 - 1.0 / q, abs=1e-6)
    assert c[1:].abs().max().item() < 0.02

    # the spin map would read states {0,1,2} as {-1,+1,+3}: a different number
    assert not torch.allclose(c, M.two_point_correlation_weighted(rnd, shape),
                              atol=1e-3)


def test_potts_correlation_length_separates_ordered_from_random():
    shape = (8, 8)
    q = 3
    ordered = torch.zeros(64, 64, dtype=torch.long)
    rnd = torch.randint(0, q, (64, 64))
    assert (M.correlation_length(ordered, shape, q=q)
            > 10 * M.correlation_length(rnd, shape, q=q))


def test_mode_diagnostics_separate_collapsed_from_balanced():
    q = 3
    t = PottsModel((8, 8), q=q, beta=1.2)
    D = t.D
    # balanced: a third of the batch in each ordered mode
    per = 64
    balanced = torch.cat([torch.full((per, D), a, dtype=torch.long)
                          for a in range(q)])
    collapsed = torch.zeros(per * q, D, dtype=torch.long)

    ob = M.summarize_observables(t, balanced, prefix="eval/")
    oc = M.summarize_observables(t, collapsed, prefix="eval/")

    assert ob["eval/mode_min"] == pytest.approx(1.0 / q, abs=1e-6)
    assert ob["eval/mode_gap"] == pytest.approx(0.0, abs=1e-6)
    assert oc["eval/mode_min"] == pytest.approx(0.0, abs=1e-12)
    assert oc["eval/mode_gap"] == pytest.approx(1.0, abs=1e-12)
    # the S_q-invariant observables cannot tell the two apart -- the point
    assert ob["eval/energy_per_site"] == pytest.approx(oc["eval/energy_per_site"])
    assert ob["eval/mag_abs_mean"] == pytest.approx(oc["eval/mag_abs_mean"])
    # per-state shares are reported too, and |M| is the same for both here
    assert sum(ob[f"eval/mode_frac_{a}"] for a in range(q)) == pytest.approx(1.0)
    assert oc["eval/mode_frac_0"] == pytest.approx(1.0)


def test_weighted_mode_min_follows_the_weights():
    q = 3
    t = PottsModel((6, 6), q=q, beta=1.2)
    D = t.D
    x = torch.cat([torch.full((32, D), a, dtype=torch.long) for a in range(q)])
    # weight one mode out of the batch: the reweighted estimate sees it gone
    log_w = torch.cat([torch.zeros(32), torch.zeros(32), torch.full((32,), -50.0)])
    out = M.summarize_observables(t, x, prefix="eval/", log_w=log_w)
    assert out["eval/mode_min"] == pytest.approx(1.0 / q, abs=1e-6)
    assert out["eval/mode_min_rw"] < 1e-9


def test_potts_evaluation_block_is_complete_and_finite():
    """A Potts run must produce usable metrics with no exact log Z."""
    t = PottsModel((6, 6), q=3, beta=1.005)
    assert t.exact_log_Z() is None or isinstance(t.exact_log_Z(), float)
    x = torch.randint(0, 3, (128, t.D))
    x_ref = torch.randint(0, 3, (256, t.D))
    ref = M.reference_summary(t, x_ref)
    assert "mode_min" in ref
    out = M.summarize_observables(t, x, prefix="eval/", log_w=torch.randn(128) * 0.1)
    out.update(M.compare_to_reference(t, x, x_ref, prefix="eval/", ref=ref))
    for k in ("eval/mode_min", "eval/mode_min_ref", "eval/mode_min_err",
              "eval/energy_err", "eval/corr_err", "eval/xi_err", "eval/mag_tv"):
        assert k in out, k
        assert out[k] == out[k] and abs(out[k]) < float("inf")


def test_ising_metrics_are_unchanged_by_the_q_plumbing():
    t = IsingModel((6, 6), beta=0.4407)
    x = torch.randint(0, 2, (128, t.D))
    x_ref = torch.randint(0, 2, (256, t.D))
    out = M.summarize_observables(t, x, prefix="eval/")
    out.update(M.compare_to_reference(t, x, x_ref, prefix="eval/"))
    # no Potts-only keys leak into an Ising run
    assert not [k for k in out if "mode_" in k or "M_norm" in k]
    assert out["eval/xi"] == pytest.approx(M.correlation_length(x, (6, 6)), abs=1e-9)
