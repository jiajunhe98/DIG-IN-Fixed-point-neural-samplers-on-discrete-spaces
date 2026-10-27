"""The ALDP-S system of FEAT (energy/aldp.py), exported for the torch force field.

openmmtools' AlanineDipeptideImplicit: amber ff96 prmtop, OBC1 implicit solvent
(a CustomGBForce), no cutoff, no constraints, 300 K.  The solvation strength
lambda scales the charges of the CustomGBForce, so the polar GB energy scales
as lambda^2 while the charge-free ACE surface term stays at full strength:

    lambda = 0   "vacuum"   bonded + nonbonded + ACE surface term
    lambda = 1   "solvent"  the full OBC1 system

The atom order is that of zmatrix.py, so its Z-matrix applies as is.

    python -m digin_bridge.system      # writes data/aldps_min.pdb, data/aldps_ff.npz

OpenMM is needed only here and in the tests.
"""
from __future__ import annotations

import math
import os

import numpy as np
import torch

from digin_bridge import zmatrix as zm

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data")
PRMTOP = os.path.join(HERE, "assets", "alanine-dipeptide.prmtop")
PDB_MIN = os.path.join(DATA, "aldps_min.pdb")
FF_NPZ = os.path.join(DATA, "aldps_ff.npz")
TEMPERATURE = 300.0
KB = 8.314e-3                  # kJ / mol / K, FEAT's value (fe/aldp.py R)
KT = KB * TEMPERATURE


def make_system(lamb: float = 1.0):
    """(topology, system) exactly as FEAT builds them."""
    import openmm.app as app
    prmtop = app.AmberPrmtopFile(PRMTOP)
    system = prmtop.createSystem(implicitSolvent=app.OBC1, constraints=None, nonbondedCutoff=None)
    if lamb != 1.0:
        n = 0
        for force in system.getForces():
            if force.__class__.__name__ == "CustomGBForce":
                for idx in range(force.getNumParticles()):
                    p = list(force.getParticleParameters(idx))
                    p[0] = p[0] * lamb
                    force.setParticleParameters(idx, p)
                n += 1
        assert n == 1, "expected one CustomGBForce"
    for n, f in enumerate(system.getForces()):
        f.setForceGroup(n)
    return prmtop.topology, system


def export_parameters(system, out: str) -> dict:
    """Every parameter of the lambda = 1 System, as arrays."""
    import openmm
    from openmm import unit
    p = {}
    for f in system.getForces():
        if isinstance(f, openmm.HarmonicBondForce):
            rows = [f.getBondParameters(i) for i in range(f.getNumBonds())]
            p["bond_idx"] = np.array([[r[0], r[1]] for r in rows])
            p["bond_r0"] = np.array([r[2].value_in_unit(unit.nanometer) for r in rows])
            p["bond_k"] = np.array([r[3].value_in_unit(unit.kilojoule_per_mole / unit.nanometer ** 2) for r in rows])
        elif isinstance(f, openmm.HarmonicAngleForce):
            rows = [f.getAngleParameters(i) for i in range(f.getNumAngles())]
            p["angle_idx"] = np.array([[r[0], r[1], r[2]] for r in rows])
            p["angle_t0"] = np.array([r[3].value_in_unit(unit.radian) for r in rows])
            p["angle_k"] = np.array([r[4].value_in_unit(unit.kilojoule_per_mole / unit.radian ** 2) for r in rows])
        elif isinstance(f, openmm.PeriodicTorsionForce):
            rows = [f.getTorsionParameters(i) for i in range(f.getNumTorsions())]
            p["tors_idx"] = np.array([[r[0], r[1], r[2], r[3]] for r in rows])
            p["tors_n"] = np.array([r[4] for r in rows], dtype=np.float64)
            p["tors_phase"] = np.array([r[5].value_in_unit(unit.radian) for r in rows])
            p["tors_k"] = np.array([r[6].value_in_unit(unit.kilojoule_per_mole) for r in rows])
        elif isinstance(f, openmm.NonbondedForce):
            n = f.getNumParticles()
            rows = [f.getParticleParameters(i) for i in range(n)]
            p["charge"] = np.array([r[0].value_in_unit(unit.elementary_charge) for r in rows])
            p["sigma"] = np.array([r[1].value_in_unit(unit.nanometer) for r in rows])
            p["epsilon"] = np.array([r[2].value_in_unit(unit.kilojoule_per_mole) for r in rows])
            ex = [f.getExceptionParameters(i) for i in range(f.getNumExceptions())]
            p["exc_idx"] = np.array([[r[0], r[1]] for r in ex])
            p["exc_qq"] = np.array([r[2].value_in_unit(unit.elementary_charge ** 2) for r in ex])
            p["exc_sigma"] = np.array([r[3].value_in_unit(unit.nanometer) for r in ex])
            p["exc_eps"] = np.array([r[4].value_in_unit(unit.kilojoule_per_mole) for r in ex])
            assert f.getNonbondedMethod() == openmm.NonbondedForce.NoCutoff
        elif isinstance(f, openmm.CustomGBForce):
            names = [f.getPerParticleParameterName(i) for i in range(f.getNumPerParticleParameters())]
            assert names == ["charge", "or", "sr"], names
            rows = np.array([f.getParticleParameters(i) for i in range(f.getNumParticles())])
            p["gb_charge"], p["gb_or"], p["gb_sr"] = rows[:, 0], rows[:, 1], rows[:, 2]
            assert f.getNonbondedMethod() == openmm.CustomGBForce.NoCutoff and f.getNumExclusions() == 0
            assert "2.909125" in f.getComputedValueParameters(1)[1], "not OBC1"
        elif isinstance(f, openmm.CMMotionRemover):
            pass
        else:
            raise TypeError(f"unhandled force {type(f).__name__}")
    p["temperature"] = np.array(TEMPERATURE)
    np.savez(out, **p)
    return p


def minimize(x_nm: np.ndarray, lamb: float = 1.0) -> np.ndarray:
    import openmm
    from openmm import unit
    _, system = make_system(lamb)
    ctx = openmm.Context(system, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName("Reference"))
    ctx.setPositions(x_nm * unit.nanometer)
    openmm.LocalEnergyMinimizer.minimize(ctx, 1e-3, 0)
    st = ctx.getState(getPositions=True, getEnergy=True)
    print(f"minimized energy (lambda={lamb}) {st.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole):.4f} kJ/mol")
    return st.getPositions(asNumpy=True).value_in_unit(unit.nanometer)


def write_pdb(path: str, x_nm: np.ndarray) -> None:
    import openmm.app as app
    from openmm import unit
    top = app.AmberPrmtopFile(PRMTOP).topology
    with open(path, "w") as f:
        app.PDBFile.writeFile(top, x_nm * unit.nanometer, f)


def read_pdb(path: str = PDB_MIN) -> np.ndarray:
    """(22, 3) positions in nm from the fixed PDB columns; no OpenMM, so training runs without it."""
    with open(path) as f:
        rows = [[float(l[30:38]), float(l[38:46]), float(l[46:54])] for l in f if l.startswith(("ATOM", "HETATM"))]
    return np.array(rows) / 10.0


def ideal_internal(phi: float, psi: float, chirality_sign: float = 1.0) -> torch.Tensor:
    """Rough ideal internal coordinates at (phi, psi) in degrees; the minimizer does the rest."""
    heavy = {0: "H", 1: "C", 2: "H", 3: "H", 4: "C", 5: "O", 6: "N", 7: "H", 8: "C", 9: "H", 10: "C",
             11: "H", 12: "H", 13: "H", 14: "C", 15: "O", 16: "N", 17: "H", 18: "C", 19: "H", 20: "H", 21: "H"}
    length = {"CH": 0.109, "HN": 0.101, "CC": 0.152, "CN": 0.140, "CO": 0.123}
    angle = {(14, 8, 6): 111.0, (4, 6, 8): 122.0, (16, 14, 8): 116.0, (1, 4, 6): 116.0, (18, 16, 14): 122.0,
             (5, 4, 6): 123.0, (7, 6, 8): 118.0, (15, 14, 16): 123.0, (17, 16, 14): 119.0}
    z = []
    for s in zm.SPEC:
        a = s["atoms"]
        if s["kind"] == "bond":
            z.append(length.get("".join(sorted(heavy[a[0]] + heavy[a[1]])), 0.15))
        elif s["kind"] == "angle":
            z.append(math.radians(angle.get(tuple(a), 109.5)))
        else:
            i = a[0]
            v = {4: phi, 16: psi, 9: -120.0 * chirality_sign, 10: 120.0 * chirality_sign,
                 2: 120.0, 12: 120.0, 20: 120.0, 3: -120.0, 13: -120.0, 21: -120.0}.get(i, 180.0)
            z.append(math.radians(v))          # omega, sp2 out-of-plane and methyl rotors: 180
    return torch.tensor(z, dtype=torch.float64)


def build_l_alanine(phi: float = -80.0, psi: float = 75.0) -> np.ndarray:
    """(22, 3) positions in nm of L-alanine dipeptide at (phi, psi) from ideal geometry."""
    for sign in (1.0, -1.0):
        x = zm.to_cartesian(ideal_internal(phi, psi, sign))
        if zm.chiral_volume(x) > 0:
            return x.numpy()
    raise RuntimeError("could not build L chirality")


def main():
    os.makedirs(DATA, exist_ok=True)
    # ideal L-alanine at the C7eq minimum, minimized in solvent
    x = minimize(build_l_alanine(-80.0, 75.0), lamb=1.0)
    assert zm.chiral_volume(torch.as_tensor(x)) > 0, "lost L chirality"
    write_pdb(PDB_MIN, x)
    _, system = make_system(1.0)
    export_parameters(system, FF_NPZ)
    z = zm.to_internal(torch.as_tensor(read_pdb()))
    print("phi/psi of the minimum (deg):",
          round(math.degrees(z[zm.PHI_INDEX].item()), 1), round(math.degrees(z[zm.PSI_INDEX].item()), 1))
    print("wrote", PDB_MIN, FF_NPZ)


if __name__ == "__main__":
    main()
