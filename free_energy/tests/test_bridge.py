"""The identities the bridge rests on, and a short training run of both tasks.

    python -m pytest tests -q        # OpenMM for the force-field tests, huggingface_hub for FEAT's data
"""
import json
import math
import os

import numpy as np
import pytest
import torch

from digin_bridge import bridge as br
from digin_bridge import system as sy
from digin_bridge import zmatrix as zm
from digin_bridge.forcefield import AmberOBC1
from digin_bridge.model import HollowTransformer, hollow_check
from digin_bridge.target import BridgeTarget


def _thermal_configs(n=16, seed=0, lamb=1.0):
    """Short hot MD of the lambda System, so every force group sees thermal and strained geometries."""
    import openmm
    from openmm import unit
    _, system = sy.make_system(lamb)
    integ = openmm.LangevinMiddleIntegrator(600 * unit.kelvin, 5 / unit.picosecond, 1 * unit.femtosecond)
    integ.setRandomNumberSeed(seed)
    ctx = openmm.Context(system, integ, openmm.Platform.getPlatformByName("Reference"))
    ctx.setPositions(sy.read_pdb() * unit.nanometer)
    xs = []
    for _ in range(n):
        integ.step(200)
        xs.append(ctx.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer))
    return np.array(xs)


@pytest.mark.parametrize("lamb", [0.0, 0.5, 1.0])
def test_energy_matches_openmm_per_force(lamb):
    pytest.importorskip("openmm")
    import openmm
    from openmm import unit
    xs = _thermal_configs(lamb=lamb)
    _, system = sy.make_system(lamb)
    ctx = openmm.Context(system, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName("Reference"))
    group = {type(f).__name__: n for n, f in enumerate(system.getForces())}
    ff = AmberOBC1(sy.FF_NPZ)
    for x in xs:
        ctx.setPositions(x * unit.nanometer)
        e = lambda g: ctx.getState(getEnergy=True, groups={g}).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        t = ff.terms(torch.as_tensor(x))
        for name, mine in (("HarmonicBondForce", t["bond"]), ("HarmonicAngleForce", t["angle"]),
                           ("PeriodicTorsionForce", t["torsion"]), ("NonbondedForce", t["nonbonded"]),
                           ("CustomGBForce", lamb ** 2 * t["gb_polar"] + t["gb_sa"])):
            assert abs(float(mine) - e(group[name])) < 1e-6, (lamb, name)
        total = ctx.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
        assert abs(float(ff.energy(torch.as_tensor(x), lamb)) - total) < 1e-6


def test_energy_on_internal_roundtrip():
    """E(x) = E(x(z(x))) for both endpoints: the energy is invariant to the rigid motion the Z-matrix drops."""
    pytest.importorskip("openmm")
    xs = torch.as_tensor(_thermal_configs(n=6, seed=3))
    ff = AmberOBC1(sy.FF_NPZ)
    a, b = ff(xs), ff(zm.to_cartesian(zm.to_internal(xs)))
    for u, v in zip(a, b):
        assert torch.allclose(u, v, atol=1e-8)


def test_hollow():
    torch.manual_seed(0)
    net = HollowTransformer(12, 6, d_model=32, n_layers=2, n_readout=2, n_heads=4).double()
    for p in net.parameters():
        torch.nn.init.normal_(p, std=0.3)
    worst, elsewhere = hollow_check(net, n=3)
    assert worst == 0.0 and elsewhere > 1e-4


def test_untrained_bridge_is_the_identity():
    """Zero-initialised heads: phi = 0, so no jump and no path term (the Zwanzig warm start)."""
    net = HollowTransformer(5, 4, d_model=16, n_layers=1, n_readout=1, n_heads=2)
    x0 = torch.randint(0, 4, (64, 5))
    x1, lr, jumps = br.ctmc_sample(net, x0, 16, path_weight=True)
    assert torch.equal(x1, x0) and float(lr.abs().max()) == 0.0 and float(jumps.sum()) == 0.0


def _toy_log_ratio(gibbs_rate, n=400000, seed=1):
    """log mean w on a toy space small enough to enumerate (D = 3, L = 4) with a
    random, untrained network: (estimate, exact log Z_1 / Z_0, standard error)."""
    torch.manual_seed(seed)
    D, L, K = 3, 4, 6
    net = HollowTransformer(D, L, d_model=16, n_layers=1, n_readout=1, n_heads=2)
    for p in net.parameters():
        torch.nn.init.normal_(p, std=0.3)
    states = torch.cartesian_prod(*[torch.arange(L)] * D)                 # (64, D)
    lp0 = torch.randn(len(states)).log_softmax(0)                         # normalized: Z_0 = 1
    lp1 = 0.5 * torch.randn(len(states)) + 1.0                            # unnormalized
    i = torch.multinomial(lp0.exp(), n, replacement=True)
    x1, lr, jumps = br.ctmc_sample(net, states[i], K, path_weight=True, gibbs_rate=gibbs_rate)
    assert float(jumps.mean()) > 0.5, "the test needs a network that moves"
    j = (x1 * (L ** torch.arange(D - 1, -1, -1))).sum(-1)                # index into `states`
    lw = lp1[j].double() - lp0[i].double() + lr
    lse = torch.logsumexp(lw, 0)
    ess = float((2 * lse - torch.logsumexp(2 * lw, 0)).exp() / n)
    return float(lse - math.log(n)), float(torch.logsumexp(lp1, 0)), math.sqrt((1.0 / ess - 1.0) / n)


def test_path_weights_are_unbiased():
    """E_{p0}[w] = Z_1 / Z_0 for an arbitrary network once both kernels have full support."""
    est, exact, se = _toy_log_ratio(gibbs_rate=0.3)
    assert abs(est - exact) < 4 * se, (est, exact, se)


def test_path_weights_need_the_gibbs_rate():
    """Without it, downhill jumps have zero forward rate and E[w] misses their mass (biased low)."""
    est, exact, se = _toy_log_ratio(gibbs_rate=0.0)
    assert est < exact - 8 * se, (est, exact, se)


def test_aldpt_states_and_labels():
    """aldpt: U_0 / U_1 are finite exactly on S_b / S_a, the phi label has no mass
    outside S_a, and the labels stay defined at an endpoint outside S_a."""
    tt, ts = BridgeTarget(n_bins=16, task="aldpt"), BridgeTarget(n_bins=16, task="aldps")
    z = tt.z_min.clone().repeat(2, 1)
    z[0, zm.PHI_INDEX], z[1, zm.PHI_INDEX] = -1.4, 1.2                # C7eq-like (S_b), C7ax-like (S_a)
    U0, U1 = tt.reduced(z)
    assert math.isfinite(U0[0]) and math.isinf(U0[1]) and math.isinf(U1[0]) and math.isfinite(U1[1])
    assert torch.allclose(U0[0], ts.reduced(z)[0][0]) and torch.allclose(U1[1], ts.reduced(z)[0][1])
    lab = tt.gibbs_labels(z).exp()                                    # (2, D, L)
    assert torch.isfinite(lab).all() and torch.allclose(lab.sum(-1), torch.ones(2, tt.D), atol=1e-5)
    phi_c = tt.centres[zm.PHI_INDEX]
    outside = ~((phi_c > 0.0) & (phi_c < 2.15))
    assert float(lab[:, zm.PHI_INDEX, outside].max()) == 0.0
    # every other coordinate: the unrestricted vacuum heat-bath scan
    vac = ts.gibbs_labels(z).exp()                                    # aldps endpoint 1 = solvent: compare to lambda 0
    tt0 = BridgeTarget(n_bins=16, task="aldpt"); tt0.label_mask.zero_()
    keep = torch.ones(tt.D, dtype=torch.bool); keep[zm.PHI_INDEX] = False
    assert torch.allclose(lab[:, keep], tt0.gibbs_labels(z).exp()[:, keep], atol=1e-6)
    assert not torch.allclose(lab[:, keep], vac[:, keep], atol=1e-3)   # and it is the vacuum, not the solvent


def test_bridge_labels():
    """kbar is a distribution at every t, dkbar sums to zero, and the endpoints are recovered."""
    B, D, L = 4, 3, 5
    x0 = torch.randint(0, L, (B, D))
    rho = torch.rand(B, D, L).softmax(-1)
    for tv in (0.0, 0.3, 1.0):
        kbar, dk = br.bridge_labels(x0, rho, torch.full((B,), tv))
        assert torch.allclose(kbar.sum(-1), torch.ones(B, D)) and torch.allclose(dk.sum(-1), torch.zeros(B, D), atol=1e-6)
    kbar0, _ = br.bridge_labels(x0, rho, torch.zeros(B))
    assert torch.equal(kbar0.argmax(-1), x0)
    kbar1, _ = br.bridge_labels(x0, rho, torch.ones(B))
    assert torch.allclose(kbar1, rho)


def test_grouped_se():
    """One path per frame: the grouped standard error is the delta-method one up to sqrt(n / (n - 1)).
    Ten identical copies of every path leave it unchanged, while the iid formula shrinks by sqrt(10)."""
    from digin_bridge.train import grouped_se, log_ratio_stats
    lw = np.random.default_rng(0).normal(size=20000)
    n = len(lw)
    se1 = grouped_se(lw, np.arange(n))
    iid = log_ratio_stats(torch.as_tensor(lw))["log_ratio_se"]
    assert abs(se1 / iid - math.sqrt(n / (n - 1))) < 1e-8
    rep, grp = np.repeat(lw, 10), np.repeat(np.arange(n), 10)
    assert abs(grouped_se(rep, grp) / se1 - 1.0) < 1e-8
    assert abs(log_ratio_stats(torch.as_tensor(rep))["log_ratio_se"] * math.sqrt(10) / iid - 1.0) < 1e-8
    assert math.isnan(grouped_se(np.full(5, -np.inf), np.arange(5)))


@pytest.fixture(scope="module")
def feat_root(tmp_path_factory):
    pytest.importorskip("huggingface_hub")
    from digin_bridge import data
    return data.ensure(os.environ.get("FEAT_DATA") or str(tmp_path_factory.mktemp("feat_aldp")))


@pytest.mark.parametrize("task", ["aldps", "aldpt"])
def test_training_smoke(task, feat_root, tmp_path):
    """A few steps on FEAT's data, then the final evaluation: its saved paths reproduce their energies,
    positions, endpoint weights and the logged estimate; a rerun finds the run finished."""
    from digin_bridge import data, train
    out = tmp_path / task
    argv = ["--task", task, "--out", str(out), "--data_root", feat_root, "--no_wandb", "--steps", "20",
            "--d_model", "32", "--layers", "1", "--readout", "1", "--heads", "2", "--ctmc_steps", "8",
            "--batch", "16", "--buffer", "256", "--n_rollout", "32", "--rollout_every", "10", "--warm_start", "1",
            "--sample_chunk", "32", "--eval_every", "10", "--eval_n", "64", "--ref_n", "500",
            "--final_eval_n", "100", "--final_part", "48", "--final_chunk", "16", "--log_every", "5",
            "--ckpt_every", "10"]
    train.main(argv)
    d = out / "final"
    meta = json.load(open(d / "meta.json"))
    assert meta["n_paths"] == 100 and meta["step"] == 20 and meta["n_start_frames"] == 100
    assert not (d / "parts").exists() and (d / "topology.pdb").exists()
    f = {k: np.load(d / f"{k}.npy") for k in train.FINAL_FIELDS}
    assert all(len(v) == 100 for v in f.values())
    assert f["z"].shape == (100, 60) and f["x"].shape == (100, 22, 3) and f["rama"].shape == (100, 2)
    tg = BridgeTarget(task=task)
    z = torch.as_tensor(f["z"])
    assert np.allclose(tg.energy1(z).numpy(), f["energy"], rtol=1e-10, atol=1e-8)
    assert np.allclose(zm.to_cartesian(z).numpy(), f["x"], atol=1e-6)
    assert np.allclose(f["rama"], torch.stack(tg.phi_psi(z), -1).numpy(), atol=1e-6)
    # the endpoint part of the weight is -U_1(z_1) + U_0(z_0) of the saved starting frame
    z0 = torch.as_tensor(data.internal(0, feat_root)[f["start_frame"]])
    ends = (-tg.reduced(z)[1] + tg.reduced(z0)[0]).numpy()
    fin = np.isfinite(ends)
    assert np.array_equal(fin, np.isfinite(f["log_w_endpoint"])) and np.isfinite(tg.reduced(z0)[0].numpy()).all()
    assert np.allclose(ends[fin], f["log_w_endpoint"][fin], rtol=1e-9, atol=1e-6)
    if np.isfinite(f["log_w"]).any():
        lme = float(torch.logsumexp(torch.as_tensor(f["log_w"]), 0)) - math.log(100)
        assert abs(lme - meta["log_ratio"]) < 1e-9
    assert json.load(open(out / "result.json"))["final/n_paths"] == 100
    train.main(argv)                              # finished: returns without doing anything
