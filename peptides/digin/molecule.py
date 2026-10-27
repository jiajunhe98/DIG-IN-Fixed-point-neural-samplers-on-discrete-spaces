"""Internal coordinates of a capped peptide, from its bond graph.

The Z-matrix is a breadth-first spanning tree rooted at the CA of the central
residue (second atom its N, third its C); heavy atoms are placed before
hydrogens.  The capped alanines have no rings, so the tree is the bond graph
and every bond and angle of the tree is a force-field bond or angle.  Each
dihedral i-j-k-l is classified:

  torsion   first child of j about a rotatable bond j-k: full circle, cyclic
            (phi, psi, methyl rotors)
  omega     first child about a peptide bond C(=O)-N: window around trans
  branch    every further child of j, placed relative to the first child
            (out-of-plane O and H, HA/CB chirality, methyl H's): window around
            the minimized value

so the windows fix L-chirality and trans peptide bonds, and the backbone
torsions and methyl rotors are free.  log|J| = sum 2 log r + sum log sin theta
(rigid motions integrated out).
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import torch
from torch import Tensor

from digin import systems
from digin.geometry import dihedral, nerf


def parse_pdb(path: str):
    atoms, xyz = [], []
    for ln in open(path):
        if ln.startswith(("ATOM", "HETATM")):
            name = ln[12:16].strip()
            el = ln[76:78].strip() or name[0]
            atoms.append({"name": name, "resname": ln[17:20].strip(), "resid": int(ln[22:26]), "element": el})
            xyz.append([float(ln[30:38]), float(ln[38:46]), float(ln[46:54])])
    return atoms, np.array(xyz) / 10.0


class Molecule:
    def __init__(self, name: str):
        self.name = name
        pdb, npz = systems.paths(name)
        self.atoms, self.x_min = parse_pdb(pdb)
        self.n = n = len(self.atoms)
        p = np.load(npz)
        self.bonds = [tuple(map(int, b)) for b in p["bond_idx"]]
        assert len(self.bonds) == n - 1, "the Z-matrix assumes an acyclic molecule"
        nb = defaultdict(set)
        for a, b in self.bonds:
            nb[a].add(b); nb[b].add(a)
        self.nbrs = nb
        self.is_h = [a["element"] == "H" for a in self.atoms]

        # residues in order, and the root
        resids = sorted({a["resid"] for a in self.atoms})
        self.residues = [(r, next(a["resname"] for a in self.atoms if a["resid"] == r)) for r in resids]
        idx = {(a["resid"], a["name"]): i for i, a in enumerate(self.atoms)}
        self.idx = idx
        with_ca = [r for r in resids if (r, "CA") in idx]
        mid = with_ca[(len(with_ca) - 1) // 2]
        root, second, third = idx[(mid, "CA")], idx[(mid, "N")], idx[(mid, "C")]

        # breadth-first tree, heavy atoms first
        parent = {root: None, second: root, third: root}
        order = [root, second, third]
        children = defaultdict(list)
        children[root] += [second, third]
        q = deque(order)
        while q:
            j = q.popleft()
            new = sorted((c for c in nb[j] if c not in parent), key=lambda c: (self.is_h[c], c))
            for c in new:
                parent[c] = j
                children[j].append(c)
                order.append(c)
                q.append(c)
        assert len(order) == n, "molecule is not connected"
        self.order, self.parent, self.children = order, parent, children

        def peptide(a, b):
            for c, nn in ((a, b), (b, a)):
                if self.atoms[c]["element"] == "C" and self.atoms[nn]["element"] == "N" and len(nb[c]) == 3 \
                        and any(self.atoms[o]["element"] == "O" and len(nb[o]) == 1 for o in nb[c]):
                    return True
            return False

        # internal coordinates
        spec = [{"kind": "bond", "atoms": (second, root)},
                {"kind": "bond", "atoms": (third, root)},
                {"kind": "angle", "atoms": (third, root, second)}]
        place = []          # (i, j, k, l, kind)
        for i in order[3:]:
            j = parent[i]
            if j == root:
                k, l, kind = second, third, "branch"
            else:
                k = parent[j]
                first = children[j][0]
                if i == first:
                    l = (third if j == second else second) if k == root else parent[k]
                    kind = "omega" if peptide(j, k) else "torsion"
                else:
                    l, kind = first, "branch"
            place.append((i, j, k, l, kind))
            spec += [{"kind": "bond", "atoms": (i, j)}, {"kind": "angle", "atoms": (i, j, k)},
                     {"kind": "dihedral", "atoms": (i, j, k, l), "dih": kind}]
        self.spec = spec
        self.D = len(spec)
        assert self.D == 3 * n - 6
        self.root, self.second, self.third = root, second, third

        # levels for parallel placement
        level = {root: 0, second: 0, third: 0}
        rows = []
        for c, (i, j, k, l, _) in enumerate(place):
            level[i] = 1 + max(level[j], level[k], level[l])
            rows.append((level[i], i, j, k, l, 3 + 3 * c))
        self.levels = []
        for lv in sorted({r[0] for r in rows}):
            sel = [r for r in rows if r[0] == lv]
            t = lambda col: torch.tensor([r[col] for r in sel], dtype=torch.long)
            self.levels.append((t(1), t(2), t(3), t(4), t(5)))
        self.bond_cols = torch.tensor([c for c, s in enumerate(spec) if s["kind"] == "bond"])
        self.angle_cols = torch.tensor([c for c, s in enumerate(spec) if s["kind"] == "angle"])
        self.dih_cols = torch.tensor([c for c, s in enumerate(spec) if s["kind"] == "dihedral"])
        ab = lambda kind: torch.tensor([s["atoms"] for s in spec if s["kind"] == kind], dtype=torch.long)
        self._bond_atoms, self._angle_atoms, self._dih_atoms = ab("bond"), ab("angle"), ab("dihedral")

        # backbone dihedrals (phi, psi) of every residue that has both
        self.rama_res, quads = [], []
        for n_, (r, rn) in enumerate(self.residues):
            need = [(r, "N"), (r, "CA"), (r, "C")]
            if not all(k in idx for k in need) or n_ == 0 or n_ == len(self.residues) - 1:
                continue
            prev_c = (self.residues[n_ - 1][0], "C")
            next_n = (self.residues[n_ + 1][0], "N")
            if prev_c in idx and next_n in idx:
                self.rama_res.append(f"{rn}{r}")
                quads.append([[idx[prev_c], idx[(r, "N")], idx[(r, "CA")], idx[(r, "C")]],
                              [idx[(r, "N")], idx[(r, "CA")], idx[(r, "C")], idx[next_n]]])
        self.rama_quads = torch.tensor(quads, dtype=torch.long)          # (R, 2, 4)

    # -- maps -------------------------------------------------------------------
    def to_cartesian(self, z: Tensor) -> Tensor:
        shape = z.shape[:-1]
        pos = torch.zeros(*shape, self.n, 3, dtype=z.dtype, device=z.device)
        zero = torch.zeros(shape, dtype=z.dtype, device=z.device)
        pos[..., self.second, :] = torch.stack([z[..., 0], zero, zero], -1)
        r, th = z[..., 1], z[..., 2]
        pos[..., self.third, :] = torch.stack([r * torch.cos(th), r * torch.sin(th), zero], -1)
        for I, J, K, L, C in self.levels:
            dev = z.device
            I, J, K, L, C = I.to(dev), J.to(dev), K.to(dev), L.to(dev), C.to(dev)
            pos[..., I, :] = nerf(pos[..., J, :], pos[..., K, :], pos[..., L, :],
                                  z[..., C], z[..., C + 1], z[..., C + 2])
        return pos

    def to_internal(self, x: Tensor) -> Tensor:
        dev = x.device
        out = torch.empty(*x.shape[:-2], self.D, dtype=x.dtype, device=dev)
        a = self._bond_atoms.to(dev)
        out[..., self.bond_cols.to(dev)] = (x[..., a[:, 0], :] - x[..., a[:, 1], :]).norm(dim=-1)
        a = self._angle_atoms.to(dev)
        u = x[..., a[:, 0], :] - x[..., a[:, 1], :]
        v = x[..., a[:, 2], :] - x[..., a[:, 1], :]
        c = (u * v).sum(-1) / (u.norm(dim=-1) * v.norm(dim=-1))
        out[..., self.angle_cols.to(dev)] = torch.acos(c.clamp(-1, 1))
        a = self._dih_atoms.to(dev)
        out[..., self.dih_cols.to(dev)] = dihedral(x[..., a[:, 0], :], x[..., a[:, 1], :],
                                                   x[..., a[:, 2], :], x[..., a[:, 3], :])
        return out

    def log_jacobian(self, z: Tensor) -> Tensor:
        dev = z.device
        return (2.0 * torch.log(z[..., self.bond_cols.to(dev)]).sum(-1)
                + torch.log(torch.sin(z[..., self.angle_cols.to(dev)])).sum(-1))

    # -- observables -------------------------------------------------------------
    def rama(self, x: Tensor) -> Tensor:
        """(..., R, 2) backbone (phi, psi) of every interior residue."""
        q = self.rama_quads.to(x.device)
        return dihedral(x[..., q[..., 0], :], x[..., q[..., 1], :], x[..., q[..., 2], :], x[..., q[..., 3], :])

    def kinds(self) -> list[str]:
        return [s["kind"] if s["kind"] != "dihedral" else s["dih"] for s in self.spec]
