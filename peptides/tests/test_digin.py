"""The identities the sampler rests on, for both systems and both coordinate systems.

    python -m pytest tests -q          # the OpenMM comparison is skipped without openmm
"""
import math

import numpy as np
import pytest
import torch

from digin import systems
from digin.forcefield import AmberOBC
from digin.geometry import wrap
from digin.molecule import Molecule
from digin.targets import CartesianTarget, InternalTarget

torch.set_default_dtype(torch.float64)
SYSTEMS = ["aldp", "al4"]


def _thermal_configs(name, n=12, seed=0):
    """Configurations from short 600 K OpenMM MD, so every force group is exercised away from the minimum."""
    import openmm
    from openmm import unit
    _, system, pos = systems.openmm_system(systems.paths(name)[0])
    integ = openmm.LangevinMiddleIntegrator(600 * unit.kelvin, 5 / unit.picosecond, 1 * unit.femtosecond)
    integ.setRandomNumberSeed(seed)
    ctx = openmm.Context(system, integ, openmm.Platform.getPlatformByName("Reference"))
    ctx.setPositions(pos)
    xs = []
    for _ in range(n):
        integ.step(200)
        xs.append(ctx.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(unit.nanometer))
    return np.array(xs), system


@pytest.mark.parametrize("name", SYSTEMS)
def test_energy_matches_openmm_per_force(name):
    pytest.importorskip("openmm")
    import openmm
    from openmm import unit
    xs, system = _thermal_configs(name)
    ctx = openmm.Context(system, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName("Reference"))
    groups = {type(f).__name__: g for g, f in enumerate(system.getForces())}
    ff = AmberOBC(systems.paths(name)[1])
    t = ff.terms(torch.as_tensor(xs))
    pairs = {"bond": "HarmonicBondForce", "angle": "HarmonicAngleForce", "torsion": "PeriodicTorsionForce",
             "nonbonded": "NonbondedForce", "gb": "GBSAOBCForce"}
    for k, x in enumerate(xs):
        ctx.setPositions(x * unit.nanometer)
        for mine, theirs in pairs.items():
            e = ctx.getState(getEnergy=True, groups={groups[theirs]}).getPotentialEnergy()
            err = abs(float(t[mine][k]) - e.value_in_unit(unit.kilojoule_per_mole))
            assert err < 1e-6, f"{name} {mine}: |dE| = {err} kJ/mol"


@pytest.mark.parametrize("name", SYSTEMS)
def test_zmatrix_roundtrip_and_jacobian(name):
    mol = Molecule(name)
    x = torch.as_tensor(mol.x_min)
    z = mol.to_internal(x)
    x2 = mol.to_cartesian(z)
    assert wrap(mol.to_internal(x2) - z).abs().max() < 1e-10
    # a rigid motion (explicit differences: cdist goes through a matmul above 25 points, ~1e-8 off)
    dist = lambda y: (y[:, None, :] - y[None, :, :]).norm(dim=-1)
    assert torch.allclose(dist(x), dist(x2), atol=1e-10)
    # log|det dx/dz| on the 3n - 6 frame coordinates (root at 0, second on +x, third in the xy plane)
    r, s, t = mol.root, mol.second, mol.third

    def f(v):
        c = mol.to_cartesian(v)
        keep = [c[s, 0], c[t, 0], c[t, 1]] + [c[a, k] for a in range(mol.n) if a not in (r, s, t) for k in range(3)]
        return torch.stack(keep)
    logdet = torch.linalg.slogdet(torch.autograd.functional.jacobian(f, z))[1]
    # integrating the orientation back in: the second atom carries r^2, the third r^2 sin(theta)
    orient = 2 * torch.log(z[0]) + torch.log(z[1]) + torch.log(torch.sin(z[2]))
    assert abs(float(logdet + orient - mol.log_jacobian(z))) < 1e-8
    kinds = mol.kinds()
    n_res = len(mol.rama_res)
    # phi/psi per residue plus the methyl rotors (ACE, NME, one CB per residue) are the free torsions
    assert kinds.count("torsion") == 2 * n_res + n_res + 2
    assert kinds.count("omega") == n_res + 1


def _brute_labels(tg, z, rows, cols, lab):
    for i in rows:
        for d in cols:
            var = z[i].repeat(tg.L, 1)
            var[:, d] = tg.centres[d]
            assert torch.allclose(lab[i, d].double(), torch.log_softmax(-tg.reduced(var), 0), atol=1e-6)


@pytest.mark.parametrize("name", SYSTEMS)
def test_internal_target(name):
    tg = InternalTarget(name, n_bins=8, label_dtype=torch.float64, label_budget=2 ** 22)
    assert tg.to_bins(tg.z_min.unsqueeze(0))[1].all()
    g = torch.Generator().manual_seed(1)
    b = torch.randint(0, tg.L, (3, tg.D), generator=g)
    z = tg.dequantize(b, generator=g)
    b2, inside = tg.to_bins(z)
    assert inside.all() and (b2 == b).all()
    lab = tg.gibbs_labels(z)
    assert lab.shape == (3, tg.D, tg.L)
    _brute_labels(tg, z, range(3), [0, 4, tg.D // 2, tg.D - 1], lab)


@pytest.mark.parametrize("name", SYSTEMS)
def test_cartesian_target(name):
    tg = CartesianTarget(name, n_bins=16, n_sweep=20000, label_dtype=torch.float64, label_budget=2 ** 22)
    mol = tg.mol
    x = torch.as_tensor(mol.x_min)
    z = tg.from_cartesian(x)
    assert tg.to_bins(z.unsqueeze(0))[1].all()
    assert torch.allclose(tg.from_cartesian(tg.to_cartesian(z)), z, atol=1e-10)
    assert abs(float(tg.ff(tg.to_cartesian(z)) - tg.ff(x))) < 1e-8
    # the mirror image (D-alanine) is excluded
    assert tg.reduced(tg.from_cartesian(x * torch.tensor([1.0, 1.0, -1.0]))) > tg.reduced(z) + 1e5
    # labels against brute force
    g = torch.Generator().manual_seed(2)
    zr = tg.dequantize(torch.randint(0, tg.L, (2, tg.D), generator=g), generator=g)
    lab = tg.gibbs_labels(zr)
    _brute_labels(tg, zr, range(2), [0, 2, 5, tg.D // 2, tg.D - 1], lab)
    # density bookkeeping: J_cart dz_cart = J_int dz_int
    zi = mol.to_internal(x)
    J = torch.autograd.functional.jacobian(lambda v: mol.to_internal(tg.to_cartesian(v)), z)
    assert abs(float(tg.log_jacobian(z) - (mol.log_jacobian(zi) + torch.linalg.slogdet(J)[1]))) < 1e-7
    # internal-coordinate reference frames map into this target's coordinates
    assert torch.allclose(tg.from_internal(zi.unsqueeze(0))[0], z, atol=1e-10)


@pytest.mark.parametrize("name", SYSTEMS)
def test_pivot_moves(name):
    """Pivot moves are rigid rotations: they keep every bond, angle and non-torsional dihedral,
    the frame coordinates of the root triple and the Jacobian, and stay in the box."""
    tg = CartesianTarget(name, n_bins=64, margin=0.12, n_sweep=20000)
    mol = tg.mol
    n_tor = sum(k == "torsion" for k in mol.kinds())
    assert len(tg.pivot_axes) == 2 * len(mol.rama_res)                     # phi and psi of each residue
    assert len(CartesianTarget(name, n_bins=64, n_sweep=1000, pivot_axes="all").pivot_axes) == n_tor
    torch.manual_seed(0)
    z0 = tg.z_min.expand(64, -1).clone()
    z1, st_ = tg.pivot_mh(z0, 40)
    assert 0.05 < st_["pivot_accept"] < 1.0
    assert tg.to_bins(z1)[1].all()
    zi0, zi1 = mol.to_internal(tg.to_cartesian(z0)), mol.to_internal(tg.to_cartesian(z1))
    rigid = [c for c, k in enumerate(mol.kinds()) if k != "torsion"]
    assert torch.allclose(zi0[:, rigid], zi1[:, rigid], atol=1e-10)
    assert (zi0 - zi1).abs().max() > 0.5
    assert torch.allclose(z0[:, :3], z1[:, :3], atol=1e-12)
    assert torch.allclose(tg.log_jacobian(z0), tg.log_jacobian(z1))


def test_tempered_weights_and_ess():
    from digin.train import ess_stats, tempered_weights
    lw = torch.randn(1000) * 5
    w, nu, ess = tempered_weights(lw, 0.8)
    assert 0 < nu < 1 and abs(ess - 0.8) < 1e-3 and abs(float(w.sum()) - 1) < 1e-5
    assert abs(ess_stats(torch.zeros(10))["ess"] - 1) < 1e-12


@pytest.mark.parametrize("coords", ["internal", "cartesian"])
def test_training_smoke(tmp_path, coords):
    """A few steps of each configuration, ending in a resumable final evaluation."""
    from digin.train import Trainer, parse
    torch.set_default_dtype(torch.float32)
    try:
        out = str(tmp_path / coords)
        args = ["--system", "aldp", "--coords", coords, "--out", out, "--no_wandb", "--steps", "6",
                "--d_model", "32", "--layers", "1", "--heads", "2", "--batch", "8", "--n_rollout", "8",
                "--rollout_every", "3", "--warm_start", "1", "--eval_every", "1000", "--eval_n", "16",
                "--sample_chunk", "8", "--final_eval_n", "40", "--final_part", "16", "--final_chunk", "8",
                "--label_budget", str(2 ** 22), "--device", "cpu"]
        if coords == "cartesian":
            args += ["--bins", "32", "--pivot_mh", "5"]
        t = Trainer(parse(args))
        t.run()
        import json, os
        res = json.load(open(os.path.join(out, "result.json")))
        assert res["final/n_samples"] == 40 and math.isfinite(res["final/log_Z"])
        assert sorted(os.listdir(os.path.join(out, "final_eval"))) == [f"part_{k:05d}.npz" for k in range(3)]
        # samples with likelihoods and energies from the trained run
        from digin.sample import main as sample
        so = os.path.join(out, "samples")
        sample(["--run_dir", out, "--out", so, "--n", "40", "--part", "16", "--chunk", "8"])
        meta = json.load(open(os.path.join(so, "meta.json")))
        x, E, lqx, lw = (np.load(os.path.join(so, f"{f}.npy")) for f in ("x", "energy", "log_q_x", "log_w"))
        assert x.shape == (40, 22, 3) and E.shape == lw.shape == (40,) and not os.path.exists(os.path.join(so, "parts"))
        u = E / meta["kT_kJ_per_mol"]
        rel = np.abs(lw + u + lqx) / (1 + np.abs(u))           # log_w = -E/kT - log_q_x
        if coords == "internal":
            assert rel.max() < 1e-9 and meta["frac_chirality_penalized"] == 0
        else:                                                  # up to the chirality penalty
            assert meta["max_rel_dev_log_w_identity"] < 1e-9 and meta["frac_chirality_penalized"] < 1
    finally:
        torch.set_default_dtype(torch.float64)
