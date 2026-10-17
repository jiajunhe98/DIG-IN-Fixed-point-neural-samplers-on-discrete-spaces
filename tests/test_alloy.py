"""What the alloy phase-diagram runs rest on.

The cluster expansion arrives pre-reduced in a packaged table, the site order
of that table is not the raster order the lattice backbones assume, and the
exact log Z is reconstructed from fixed-composition data.  Each of those is a
place where a silent convention error produces plausible-looking curves, so
each is checked directly.
"""

import numpy as np
import pytest
import torch

from dds.methods import build_method
from dds.models import build_model
from dds.targets.alloy import SYSTEMS, AlloyFamily
from dds.utils.acquisition import FisherAcquisition, mixed_cond

SMALL = {"name": "rescnn", "hidden_size": 16, "n_blocks": 5}


@pytest.fixture(scope="module", params=sorted(SYSTEMS))
def family(request):
    return AlloyFamily(request.param)


# ----------------------------------------------------------------------
# Energy conventions
# ----------------------------------------------------------------------
def test_energy_diff_is_the_actual_change_in_the_hamiltonian(family):
    """The (B, D, 2) table must equal H(x with site d set to n) - H(x), with
    the current token's column exactly zero.  This is the one identity the
    heat-bath labels and every local move depend on."""
    torch.manual_seed(0)
    x = family.sample_uniform(4)
    cond = family.sample_cond(4)
    ed = family.energy_diff(x, cond)
    assert ed.shape == (4, family.D, 2)
    # the current value costs nothing to "change" to
    assert ed.gather(2, x.unsqueeze(-1)).abs().max() < 1e-6

    base = family.energy(x, cond)
    for d in (0, 7, family.D - 1):
        y = x.clone()
        y[:, d] = 1 - y[:, d]
        lhs = family.energy(y, cond) - base
        rhs = ed[torch.arange(4), d, y[:, d]]
        assert torch.allclose(lhs, rhs, atol=1e-4), f"site {d}"


def test_site_order_round_trips_and_is_not_the_identity(family):
    """Tokens are stored in raster order over the lattice shape, the cluster
    table indexes its own site order, and the two differ.  A round trip must
    be exact; if the permutation were the identity the test would not be
    checking anything, so that is asserted too."""
    x = family.sample_uniform(3)
    s = family.spins(x)
    assert torch.equal(family.tokens(s), x)
    assert set(torch.unique(s).tolist()) <= {-1.0, 1.0}
    perm = family._site_of_flat
    assert not torch.equal(perm, torch.arange(family.D, device=perm.device))


def test_chemical_potential_enters_with_the_packaged_factor_of_two(family):
    """H = E - (dmu/2) sum_i s_i.  Dropping the half halves the delta_mu axis
    and still produces plausible curves, so it is pinned here."""
    x = family.sample_uniform(4)
    T = torch.full((4,), 600.0)
    m = family.spins(x).sum(-1).to(torch.float32)
    e0 = family.energy(x, torch.stack([T, torch.zeros(4)], -1))
    e1 = family.energy(x, torch.stack([T, torch.full((4,), 0.2)], -1))
    assert torch.allclose(e0 - e1, 0.1 * m, atol=1e-4)


# ----------------------------------------------------------------------
# Exact reference
# ----------------------------------------------------------------------
def test_exact_log_Z_matches_the_fixed_composition_sum(family):
    """ln Z(T, dmu) = logsumexp_n [ ln Z_n(T) + beta (dmu/2)(2n - N) ], and at
    a packaged temperature no interpolation is involved."""
    from dds.targets.alloy.potential import KB_EV_PER_K

    T = float(family._ref["lnZ_T"][3])
    for dmu in (0.0, 0.13, -0.2):
        # float64 on purpose: the tilt multiplies delta_mu by beta (2n - N),
        # about 2400 here, so a float32 delta_mu shifts ln Z by ~1e-5.  That
        # is the conditioning's precision, not the reconstruction's, and the
        # float32 path is checked separately below.
        got = family.exact_log_Z(torch.tensor([[T, dmu]], dtype=torch.float64))
        assert got is not None
        j = int(np.argmin(np.abs(family._ref["lnZ_T"] - T)))
        row = family._ref["lnZ"][j]
        ok = family._ref["lnZ_valid"][j]
        n = family._ref["n"]
        a = np.where(ok, row + (dmu / (2 * KB_EV_PER_K * T)) * (2 * n - family.D), -np.inf)
        want = float(np.max(a) + np.log(np.sum(np.exp(a - np.max(a)))))
        assert abs(got - want) < 1e-6


def test_exact_log_Z_survives_float32_conditioning(family):
    """The grid is carried as float32, which perturbs ln Z by about 1e-5 out
    of several hundred through the beta (2n - N) delta_mu tilt.  Pinned so a
    real regression is not mistaken for that."""
    T = float(family._ref["lnZ_T"][3])
    a = family.exact_log_Z(torch.tensor([[T, 0.13]], dtype=torch.float64))
    b = family.exact_log_Z(torch.tensor([[T, 0.13]], dtype=torch.float32))
    assert abs(a - b) < 1e-3


def test_exact_log_Z_declines_to_extrapolate(family):
    """Outside the packaged sweep there is no closed form to fall back on, so
    it returns None rather than a confident wrong number."""
    assert family.exact_log_Z(torch.tensor([[50.0, 0.0]])) is None
    assert family.exact_log_Z(torch.tensor([[5000.0, 0.0]])) is None


def test_reference_state_point_indexes_dmu_then_T(family):
    """The packaged arrays are (dmu, T, walker); getting the axes the wrong
    way round still returns a number, so the lookup is pinned to the values
    it was asked for."""
    r = family.reference_state_point(600.0, 0.05)
    assert r is not None
    assert abs(r["T"] - 600.0) <= 25.0 and abs(r["dmu"] - 0.05) <= 0.01
    assert 0.0 <= r["c"] <= 1.0


def test_cluster_tables_follow_the_states_to_their_device(family):
    """The target is built before the run knows its device, so the packaged
    tables start on the CPU and the states arrive on the accelerator.  The
    migration walk must find every tensor the expansion holds, including the
    ones kept in lists, or the first einsum dies on a device mismatch."""
    ce = family.ce
    named = [k for k, v in ce.__dict__.items() if torch.is_tensor(v)]
    listed = [k for k, v in ce.__dict__.items()
              if isinstance(v, (list, tuple)) and v
              and all(torch.is_tensor(t) for t in v)]
    assert named and listed, "nothing to migrate: the walk would be vacuous"

    ce.device = "not-a-device"          # force the migration branch
    family._ensure_device(torch.device("cpu"))
    assert ce.device == torch.device("cpu")
    for k in named:
        assert getattr(ce, k).device == torch.device("cpu"), k
    for k in listed:
        assert all(t.device == torch.device("cpu") for t in getattr(ce, k)), k
    # the permutation the family caches must be refreshed, not left dangling
    assert family._site_of_flat is ce._flat_of_site
    # and the whole path still works afterwards
    x = family.sample_uniform(2)
    assert family.energy(x, family.sample_cond(2)).shape == (2,)


# ----------------------------------------------------------------------
# Conditioning
# ----------------------------------------------------------------------
def test_cond_features_span_the_unit_box(family):
    corners = torch.tensor([
        [family.T_range[0], family.dmu_range[0]],
        [family.T_range[1], family.dmu_range[1]],
    ])
    f = family.to_cond_features(corners)
    assert torch.allclose(f[0], torch.tensor([1.0, 0.0]), atol=1e-5)
    assert torch.allclose(f[1], torch.tensor([0.0, 1.0]), atol=1e-5)


def test_sample_cond_stays_in_the_box_and_is_continuous(family):
    c = family.sample_cond(512)
    assert (c[:, 0] >= family.T_range[0] - 1e-3).all()
    assert (c[:, 0] <= family.T_range[1] + 1e-3).all()
    assert (c[:, 1] >= family.dmu_range[0] - 1e-6).all()
    assert (c[:, 1] <= family.dmu_range[1] + 1e-6).all()
    # continuous, not a ladder: 512 draws should not collide
    assert len(torch.unique(c[:, 0])) > 400


# ----------------------------------------------------------------------
# Fisher acquisition
# ----------------------------------------------------------------------
def test_acquisition_concentrates_where_the_variance_is():
    """A cell fed high-variance gradients must end up with more mass than one
    fed constants, and every draw must stay inside the box."""
    torch.manual_seed(0)
    acq = FisherAcquisition([(0.0, 1.0), (0.0, 1.0)], bins=(4, 4), ema=0.0, power=1.0)
    hot = torch.tensor([[0.15, 0.15]]).repeat(256, 1)
    cold = torch.tensor([[0.85, 0.85]]).repeat(256, 1)
    acq.update(hot, torch.randn(256, 2) * 10.0)
    acq.update(cold, torch.zeros(256, 2))
    w = acq.weights()
    assert w[0, 0] > w[3, 3]
    s = acq.sample(256)
    assert (s >= 0.0).all() and (s <= 1.0).all()


def test_acquisition_is_uniform_before_it_has_seen_anything():
    acq = FisherAcquisition([(0.0, 1.0), (-1.0, 1.0)], bins=(4, 4))
    w = acq.weights()
    assert torch.allclose(w, torch.full_like(w, 1.0 / w.numel()), atol=1e-6)


def test_mixed_cond_falls_back_exactly_to_the_uniform_draw(family):
    """frac_uniform = 1.0 (and no acquisition) must reproduce the family's own
    draw bit for bit -- that is what every non-amortized run relies on."""
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    a = mixed_cond(family, 32, None, frac_uniform=1.0, generator=g1)
    b = family.sample_cond(32, generator=g2)
    assert torch.equal(a, b)


def test_mixed_cond_splits_the_batch(family):
    acq = FisherAcquisition(family.acq_ranges, bins=(4, 4))
    c = mixed_cond(family, 64, acq, frac_uniform=0.5)
    assert c.shape == (64, 2)
    assert (c[:, 0] >= family.T_range[0] - 1e-3).all()
    assert (c[:, 0] <= family.T_range[1] + 1e-3).all()


# ----------------------------------------------------------------------
# Backbone and methods
# ----------------------------------------------------------------------
def test_rescnn_covers_a_three_dimensional_torus(family):
    m = build_model("rescnn", family.lattice_shape, vocab_size=2, out_channels=2,
                    require_time=True, hidden_size=16, n_blocks=5,
                    cond_ranges=family.cond_ranges)
    x = family.sample_uniform(2)
    y = m(x, torch.rand(2), family.to_cond_features(family.sample_cond(2)))
    assert y.shape == (2, family.D, 2)
    for r, s in zip(m.radius, family.lattice_shape):
        assert r >= s // 2


@pytest.mark.parametrize("name,cfg", [
    ("dsi", dict(model=SMALL, noise="masked", loss="ce", batch_size=4,
                 n_rollout=8, rollout_every=4, warm_start=1, acq_frac_uniform=0.5)),
    # the paper's alloy recipe: per-cell tempered weights and damping
    ("dsi", dict(model=SMALL, noise="masked", loss="ce", batch_size=4,
                 n_rollout=8, rollout_every=4, warm_start=1, acq_frac_uniform=0.5,
                 weighting="tempered_local", weight_bins=[4, 4], target_ess=0.9,
                 eta=0.5)),
    ("frogs", dict(model={"channels": 4, "kernel_sizes": [3]},
                   prior={"channels": 8, "n_blocks": 1, "kernel_size": 3},
                   delta_t=0.25, n_anneal=4, n_walkers=4, loss_batch=4,
                   buffer_cycle=2, buffer_size=2)),
    ("asbs", dict(model=SMALL, batch_size=4, n_rollout=8, resample_every=4,
                  steps_controller=2, steps_corrector=1, sampling_steps=4,
                  num_repl=2)),
])
def test_amortized_methods_train_and_sample_at_a_state_point(name, cfg):
    """Every amortized method must take a single (T, delta_mu) at evaluation
    and return finite weights for it -- that is what the grid evaluation and
    the phase diagram are built on."""
    torch.manual_seed(0)
    f = AlloyFamily("agpd125")
    m = build_method(name, f, dict(cfg), device="cpu")
    out = m.step()
    assert np.isfinite(out["loss"])
    x, logw = m.sample_with_weights(4, cond=torch.tensor([600.0, 0.05]))
    assert x.shape == (4, f.D)
    assert torch.isfinite(logw).all()


def test_grid_evaluation_and_phase_diagram_run_end_to_end():
    from dds.eval_alloy import (compare_phase_diagram, evaluate_alloy_grid,
                                phase_diagram_from_rows)

    torch.manual_seed(0)
    f = AlloyFamily("agpd125")
    m = build_method("dsi", f, dict(model=SMALL, noise="masked", loss="ce",
                                    batch_size=4, n_rollout=8, rollout_every=4,
                                    warm_start=1), device="cpu")
    summary, rows = evaluate_alloy_grid(m, n_per_point=8, n_T=2, n_dmu=5)
    assert len(rows) == 10
    assert "grid/log_Z_err_mean" in summary and "grid/ess_mean" in summary
    # the exact log Z is available on the whole plane here, not just one line
    assert all("log_Z_exact" in r for r in rows)
    pd = phase_diagram_from_rows(rows, min_jump=-1.0)
    assert pd and {"temperature", "dmu", "c_left", "c_right"} <= set(pd[0])
    cmp = compare_phase_diagram(pd, f)
    assert cmp == {} or "pd/c_left_err_mean" in cmp
