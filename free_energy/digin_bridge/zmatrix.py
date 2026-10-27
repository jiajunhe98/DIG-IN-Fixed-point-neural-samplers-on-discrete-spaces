"""Internal coordinates of alanine dipeptide (ACE-ALA-NME, 22 atoms, 60 DOF).

Atom order is the standard one of the amber ALDP topology:

     0 HH31  1 CH3   2 HH32  3 HH33  4 C    5 O      (ACE)
     6 N     7 H     8 CA    9 HA   10 CB  11 HB1  12 HB2  13 HB3  14 C  15 O   (ALA)
    16 N    17 H    18 CH3  19 HH31 20 HH32 21 HH33  (NME)

The Z-matrix is rooted at CA.  Every bond and every angle in it is a bond or
angle term of the force field.  The dihedrals come in three kinds:

  * torsion  phi, psi and the three methyl rotors: a full circle, cyclic bins
  * omega    the two peptide bonds: a window around trans
  * branch   an atom placed relative to a sibling on the same centre (carbonyl
             O, amide H, HA, CB, the second and third methyl H): a window
             around the minimized value.  These are the stiff out-of-plane and
             chirality coordinates, independent of phi/psi by construction, so
             a window around the L-form fixes the chirality of CA.

Placement is NeRF (Parsons et al. 2005).  The Cartesian volume element of an
atom placed by (r, theta, phi) is r^2 sin(theta) dr dtheta dphi; the second
atom has r^2 dr and the root nothing (rigid motions integrated out), so
log|J| = sum 2 log r + sum log sin theta.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor

N_ATOMS = 22
ROOT = 8
SECOND = (6, 8)              # atom, bonded to
THIRD = (14, 8, 6)           # atom, bonded to, angle with
# atom, bond partner j, angle partner k, dihedral partner l, dihedral kind
REST = [
    (4, 6, 8, 14, "torsion"),    # phi   C-N-CA-C
    (16, 14, 8, 6, "torsion"),   # psi   N-C-CA-N
    (1, 4, 6, 8, "omega"),       # omega0 CH3-C-N-CA
    (18, 16, 14, 8, "omega"),    # omega1 CH3-N-C-CA
    (5, 4, 6, 1, "branch"),      # O(ACE), out of plane of C
    (0, 1, 4, 6, "torsion"),     # ACE methyl rotor
    (2, 1, 4, 0, "branch"),
    (3, 1, 4, 0, "branch"),
    (7, 6, 8, 4, "branch"),      # H(ALA), out of plane of N
    (9, 8, 6, 14, "branch"),     # HA   } chirality of CA
    (10, 8, 6, 14, "branch"),    # CB   }
    (11, 10, 8, 6, "torsion"),   # CB methyl rotor
    (12, 10, 8, 11, "branch"),
    (13, 10, 8, 11, "branch"),
    (15, 14, 16, 8, "branch"),   # O(ALA), out of plane of C
    (17, 16, 14, 18, "branch"),  # H(NME), out of plane of N
    (19, 18, 16, 14, "torsion"), # NME methyl rotor
    (20, 18, 16, 19, "branch"),
    (21, 18, 16, 19, "branch"),
]
PHI_INDEX = None   # filled below: index of phi / psi in the coordinate vector
PSI_INDEX = None


def coordinate_spec() -> list[dict]:
    """One entry per internal coordinate, in the order of the vector z (D=60)."""
    spec = [{"kind": "bond", "atoms": (SECOND[0], SECOND[1]), "dih": None},
            {"kind": "bond", "atoms": (THIRD[0], THIRD[1]), "dih": None},
            {"kind": "angle", "atoms": THIRD, "dih": None}]
    for i, j, k, l, kind in REST:
        spec.append({"kind": "bond", "atoms": (i, j), "dih": None})
        spec.append({"kind": "angle", "atoms": (i, j, k), "dih": None})
        spec.append({"kind": "dihedral", "atoms": (i, j, k, l), "dih": kind})
    return spec


SPEC = coordinate_spec()
D = len(SPEC)
assert D == 3 * N_ATOMS - 6
PHI_INDEX = next(n for n, s in enumerate(SPEC) if s["kind"] == "dihedral" and s["atoms"][0] == 4)
PSI_INDEX = next(n for n, s in enumerate(SPEC) if s["kind"] == "dihedral" and s["atoms"][0] == 16)
BOND_IDX = [n for n, s in enumerate(SPEC) if s["kind"] == "bond"]
ANGLE_IDX = [n for n, s in enumerate(SPEC) if s["kind"] == "angle"]


def _normalize(v: Tensor) -> Tensor:
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def nerf(pj: Tensor, pk: Tensor, pl: Tensor, r: Tensor, theta: Tensor, phi: Tensor) -> Tensor:
    """Place atom i bonded to j at distance r, angle i-j-k = theta, dihedral i-j-k-l = phi."""
    bc = _normalize(pj - pk)
    n = _normalize(torch.cross(pk - pl, bc, dim=-1))
    m = torch.cross(n, bc, dim=-1)
    d0 = -r * torch.cos(theta)
    d1 = r * torch.sin(theta) * torch.cos(phi)
    d2 = r * torch.sin(theta) * torch.sin(phi)
    return pj + d0.unsqueeze(-1) * bc + d1.unsqueeze(-1) * m + d2.unsqueeze(-1) * n


def to_cartesian(z: Tensor) -> Tensor:
    """(..., 60) internal coordinates (nm, rad) -> (..., 22, 3) Cartesian (nm)."""
    shape = z.shape[:-1]
    zero = torch.zeros(shape, dtype=z.dtype, device=z.device)
    pos: list[Tensor | None] = [None] * N_ATOMS
    pos[ROOT] = torch.zeros(*shape, 3, dtype=z.dtype, device=z.device)
    pos[SECOND[0]] = torch.stack([z[..., 0], zero, zero], dim=-1)
    r, th = z[..., 1], z[..., 2]
    pos[THIRD[0]] = torch.stack([r * torch.cos(th), r * torch.sin(th), zero], dim=-1)
    c = 3
    for i, j, k, l, _ in REST:
        pos[i] = nerf(pos[j], pos[k], pos[l], z[..., c], z[..., c + 1], z[..., c + 2])
        c += 3
    return torch.stack(pos, dim=-2)


def bond(x: Tensor, i: int, j: int) -> Tensor:
    return (x[..., i, :] - x[..., j, :]).norm(dim=-1)


def angle(x: Tensor, i: int, j: int, k: int) -> Tensor:
    a = _normalize(x[..., i, :] - x[..., j, :])
    b = _normalize(x[..., k, :] - x[..., j, :])
    return torch.acos((a * b).sum(-1).clamp(-1.0, 1.0))


def dihedral_from(p0: Tensor, p1: Tensor, p2: Tensor, p3: Tensor) -> Tensor:
    """IUPAC dihedral p0-p1-p2-p3 in (-pi, pi]."""
    b0, b1, b2 = p1 - p0, p2 - p1, p3 - p2
    n1 = torch.cross(b0, b1, dim=-1)
    n2 = torch.cross(b1, b2, dim=-1)
    y = (b1.norm(dim=-1, keepdim=True) * b0 * n2).sum(-1)
    x = (n1 * n2).sum(-1)
    return torch.atan2(y, x)


def dihedral(x: Tensor, i: int, j: int, k: int, l: int) -> Tensor:
    return dihedral_from(x[..., i, :], x[..., j, :], x[..., k, :], x[..., l, :])


def to_internal(x: Tensor) -> Tensor:
    """(..., 22, 3) Cartesian -> (..., 60) internal coordinates."""
    out = []
    for s in SPEC:
        a = s["atoms"]
        if s["kind"] == "bond":
            out.append(bond(x, *a))
        elif s["kind"] == "angle":
            out.append(angle(x, *a))
        else:
            out.append(dihedral(x, *a))
    return torch.stack(out, dim=-1)


def log_jacobian(z: Tensor) -> Tensor:
    """log |d x / d z| up to a constant (rigid motions integrated out)."""
    return 2.0 * torch.log(z[..., BOND_IDX]).sum(-1) + torch.log(torch.sin(z[..., ANGLE_IDX])).sum(-1)


def chiral_volume(x: Tensor) -> Tensor:
    """(N - CA) . [(C - CA) x (CB - CA)]: positive for L-alanine (CORN rule)."""
    ca = x[..., 8, :]
    return ((x[..., 6, :] - ca) * torch.cross(x[..., 14, :] - ca, x[..., 10, :] - ca, dim=-1)).sum(-1)


def wrap(a: Tensor) -> Tensor:
    return torch.remainder(a + math.pi, 2 * math.pi) - math.pi
