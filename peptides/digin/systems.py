"""The two systems: build, minimize (amber99sbildn + OBC implicit solvent, 300 K, no
constraints) and export every force-field parameter, so that training and the
reference MD need no OpenMM.

    python -m digin.systems aldp al4        # data/<name>_min.pdb, data/<name>_ff.npz

  aldp   alanine dipeptide ACE-ALA-NME          22 atoms
  al4    alanine tetrapeptide ACE-(ALA)3-NME    42 atoms

The files in data/ were made this way; OpenMM is needed only here and in the tests.
"""
from __future__ import annotations

import math
import os
import sys

import numpy as np
import torch

from digin.geometry import nerf

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
SYSTEMS = ("aldp", "al4")
TEMPERATURE = 300.0
KB = 8.314462618e-3            # kJ / mol / K


def paths(name: str) -> tuple[str, str]:
    return os.path.join(DATA, f"{name}_min.pdb"), os.path.join(DATA, f"{name}_ff.npz")


# --- capped poly-alanine from ideal internal coordinates ------------------------
_ACE = ["HH31", "CH3", "HH32", "HH33", "C", "O"]
_ALA = ["N", "H", "CA", "HA", "CB", "HB1", "HB2", "HB3", "C", "O"]
_NME = ["N", "H", "CH3", "HH31", "HH32", "HH33"]


def _capped_ala_zmatrix(n_ala: int, phi: float, psi: float, s: float):
    """Placement list ((res, name), parent, angle-ref, dihedral-ref, bond nm, angle deg, dihedral deg)."""
    Z = [((0, "CH3"), None, None, None, 0, 0, 0),
         ((0, "C"), (0, "CH3"), None, None, 0.152, 0, 0),
         ((0, "O"), (0, "C"), (0, "CH3"), None, 0.123, 121.0, 0),
         ((1, "N"), (0, "C"), (0, "CH3"), (0, "O"), 0.134, 116.0, 180.0),
         ((0, "HH31"), (0, "CH3"), (0, "C"), (1, "N"), 0.109, 109.5, 180.0),
         ((0, "HH32"), (0, "CH3"), (0, "C"), (0, "HH31"), 0.109, 109.5, 120.0),
         ((0, "HH33"), (0, "CH3"), (0, "C"), (0, "HH31"), 0.109, 109.5, -120.0)]
    for r in range(1, n_ala + 1):
        prev_ca = (r - 1, "CA") if r > 1 else (0, "CH3")
        Z += [((r, "H"), (r, "N"), (r - 1, "C"), (r - 1, "O"), 0.101, 119.0, 180.0),
              ((r, "CA"), (r, "N"), (r - 1, "C"), prev_ca, 0.146, 122.0, 180.0),
              ((r, "C"), (r, "CA"), (r, "N"), (r - 1, "C"), 0.152, 111.0, phi),
              ((r, "HA"), (r, "CA"), (r, "N"), (r - 1, "C"), 0.109, 109.5, phi + s * 120.0),
              ((r, "CB"), (r, "CA"), (r, "N"), (r - 1, "C"), 0.153, 110.0, phi - s * 120.0),
              ((r, "HB1"), (r, "CB"), (r, "CA"), (r, "N"), 0.109, 109.5, 60.0),
              ((r, "HB2"), (r, "CB"), (r, "CA"), (r, "N"), 0.109, 109.5, 180.0),
              ((r, "HB3"), (r, "CB"), (r, "CA"), (r, "N"), 0.109, 109.5, -60.0),
              ((r, "O"), (r, "C"), (r, "CA"), (r, "N"), 0.123, 121.0, psi + 180.0),
              ((r + 1, "N"), (r, "C"), (r, "CA"), (r, "N"), 0.134, 116.0, psi)]
    c = n_ala + 1
    Z += [((c, "H"), (c, "N"), (n_ala, "C"), (n_ala, "O"), 0.101, 119.0, 180.0),
          ((c, "CH3"), (c, "N"), (n_ala, "C"), (n_ala, "CA"), 0.146, 122.0, 180.0),
          ((c, "HH31"), (c, "CH3"), (c, "N"), (n_ala, "C"), 0.109, 109.5, 60.0),
          ((c, "HH32"), (c, "CH3"), (c, "N"), (n_ala, "C"), 0.109, 109.5, 180.0),
          ((c, "HH33"), (c, "CH3"), (c, "N"), (n_ala, "C"), 0.109, 109.5, -60.0)]
    return Z


def _place(Z):
    pos = {}
    t = lambda v: torch.tensor(v, dtype=torch.float64)
    for key, j, k, l, b, a, d in Z:
        if j is None:
            pos[key] = torch.zeros(3, dtype=torch.float64)
        elif k is None:
            pos[key] = pos[j] + t([b, 0.0, 0.0])
        elif l is None:
            u = pos[j] - pos[k]
            u = u / u.norm()
            rot = math.pi - math.radians(a)          # rotate the (k -> j) direction in the xy plane
            pos[key] = pos[j] + b * t([u[0] * math.cos(rot) - u[1] * math.sin(rot),
                                       u[0] * math.sin(rot) + u[1] * math.cos(rot), 0.0])
        else:
            pos[key] = nerf(pos[j], pos[k], pos[l], t(b), t(math.radians(a)), t(math.radians(d)))
    return pos


def build_capped_ala(n_ala: int, phi: float = -80.0, psi: float = 150.0):
    """(atom records, coordinates nm) of ACE-(ALA)n-NME with L chirality everywhere."""
    for s in (1.0, -1.0):
        pos = _place(_capped_ala_zmatrix(n_ala, phi, psi, s))
        ok = True
        for r in range(1, n_ala + 1):
            ca = pos[(r, "CA")]
            v = torch.dot(pos[(r, "N")] - ca, torch.cross(pos[(r, "C")] - ca, pos[(r, "CB")] - ca, dim=0))
            ok &= bool(v > 0)
        if ok:
            break
    else:
        raise RuntimeError("could not build L chirality")
    recs, xyz = [], []
    for r in range(n_ala + 2):
        names, res = (_ACE, "ACE") if r == 0 else ((_NME, "NME") if r == n_ala + 1 else (_ALA, "ALA"))
        for nm in names:
            recs.append((nm, res, r + 1, "H" if nm.startswith("H") else nm[0]))
            xyz.append(pos[(r, nm)].numpy())
    return recs, np.array(xyz)


def write_pdb(path: str, recs, x_nm: np.ndarray) -> None:
    lines = []
    for n, ((name, res, ri, el), p) in enumerate(zip(recs, x_nm * 10.0)):
        nm = f" {name:<3s}" if len(name) < 4 else name
        lines.append(f"ATOM  {n + 1:5d} {nm:4s} {res:3s} A{ri:4d}    "
                     f"{p[0]:8.3f}{p[1]:8.3f}{p[2]:8.3f}  1.00  0.00          {el:>2s}")
    lines.append("END")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# --- OpenMM ---------------------------------------------------------------------
def _forcefield():
    from openmm import app
    return app.ForceField("amber99sbildn.xml", "amber99_obc.xml")


def openmm_system(pdb_path: str):
    """(topology, system, positions): amber99sbildn + OBC, no cutoff, no constraints, one force group per force."""
    from openmm import app
    pdb = app.PDBFile(pdb_path)
    system = _forcefield().createSystem(pdb.topology, nonbondedMethod=app.NoCutoff, constraints=None,
                                        rigidWater=False, removeCMMotion=True)
    for n, f in enumerate(system.getForces()):
        f.setForceGroup(n)
    return pdb.topology, system, pdb.positions


def export_parameters(system, out: str) -> dict:
    """Every parameter of the assembled OpenMM System, as arrays."""
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
        elif isinstance(f, openmm.GBSAOBCForce):
            rows = [f.getParticleParameters(i) for i in range(f.getNumParticles())]
            p["gb_charge"] = np.array([r[0].value_in_unit(unit.elementary_charge) for r in rows])
            p["gb_radius"] = np.array([r[1].value_in_unit(unit.nanometer) for r in rows])
            p["gb_scale"] = np.array([r[2] for r in rows], dtype=np.float64)
            p["gb_solute_eps"] = np.array(f.getSoluteDielectric())
            p["gb_solvent_eps"] = np.array(f.getSolventDielectric())
            p["gb_sa_energy"] = np.array(f.getSurfaceAreaEnergy().value_in_unit(
                unit.kilojoule_per_mole / unit.nanometer ** 2))
            assert f.getNonbondedMethod() == openmm.GBSAOBCForce.NoCutoff
        elif isinstance(f, openmm.CMMotionRemover):
            pass
        else:
            raise TypeError(f"unhandled force {type(f).__name__}")
    p["temperature"] = np.array(TEMPERATURE)
    np.savez(out, **p)
    return p


def build(name: str) -> None:
    import openmm
    from openmm import app, unit
    assert name in SYSTEMS, name
    os.makedirs(DATA, exist_ok=True)
    pdb_out, npz_out = paths(name)
    tmp = os.path.join(DATA, f"_{name}_raw.pdb")
    recs, x = build_capped_ala(1 if name == "aldp" else 3)
    write_pdb(tmp, recs, x)
    pdb = app.PDBFile(tmp)
    system = _forcefield().createSystem(pdb.topology, nonbondedMethod=app.NoCutoff, constraints=None,
                                        rigidWater=False, removeCMMotion=True)
    ctx = openmm.Context(system, openmm.VerletIntegrator(0.001), openmm.Platform.getPlatformByName("Reference"))
    ctx.setPositions(pdb.positions)
    openmm.LocalEnergyMinimizer.minimize(ctx, 1e-3, 0)
    s = ctx.getState(getPositions=True, getEnergy=True)
    print(f"{name}: {pdb.topology.getNumAtoms()} atoms, minimized energy "
          f"{s.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole):.3f} kJ/mol")
    with open(pdb_out, "w") as f:
        app.PDBFile.writeFile(pdb.topology, s.getPositions(), f, keepIds=True)
    _, system2, _ = openmm_system(pdb_out)
    export_parameters(system2, npz_out)
    os.remove(tmp)


if __name__ == "__main__":
    for nm in sys.argv[1:] or list(SYSTEMS):
        build(nm)
