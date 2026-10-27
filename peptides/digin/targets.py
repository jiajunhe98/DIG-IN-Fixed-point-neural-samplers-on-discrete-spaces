"""A peptide as a binned Boltzmann target, in internal or in Cartesian coordinates.

Both targets share the construction: a box [center - half, center + half] per
coordinate, L uniform bins on it, and the discrete target is the cell mass of
exp(-U) with U(z) = E(x(z))/kT - log|J(z)|.  The importance weight of a
dequantized draw z of the model is log w = -U(z) + log vol - log q(bins | order),
exact for the continuous target on the box.  The Gibbs label of coordinate d is
the softmax over bin centres c of -U(z with z_d = c): one batched energy scan of
D x L configurations per sample.

internal   (InternalTarget) the Z-matrix of molecule.py, D = 3n - 6.  Boxes:
           bond / angle   minimized value +- n_sigma sqrt(kT / k)
           torsion        full circle, cyclic (phi, psi, methyl rotors)
           omega          minimized (trans) value +- omega_half_deg
           branch         minimized value +- branch_half_deg
           The windows fix L-chirality and trans peptide bonds.

cartesian  (CartesianTarget) a body frame on the root triple of molecule.py (CA
           at the origin, N on +x, C in the xy-plane: N_x, C_x, C_y) and every
           other atom as its Cartesian bond vector to its parent in the spanning
           tree, x_i - x_parent(i), D = 3n - 6, log|J| = 2 log N_x + log C_y
           (rotations and translations integrated out).  The change from frame
           coordinates is linear with unit Jacobian; a single-coordinate (Gibbs)
           move translates a whole subtree instead of tearing one atom away from
           its bonded neighbours.  Boxes come from a torsion sweep of the
           minimized structure (phi, psi and methyl rotors uniform) plus a margin;
           the three hydrogens of a methyl share one box per component.
           L-chirality of every CA is enforced by the energy (a large penalty).
           Single-coordinate labels cannot carry mass across phi/psi barriers in
           Cartesian space, so fresh samples get Metropolis pivot moves
           (`pivot_mh`) before they are labelled.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor

from digin import systems
from digin.forcefield import AmberOBC
from digin.geometry import wrap
from digin.molecule import Molecule

CHIRAL_PENALTY = 1e6            # kT


class BinnedTarget:
    """Shared bin <-> continuous maps, energies, weights and labels.  Subclasses set
    mol, ff, kT, D, L, center, half, is_dihedral, kind and call _finish_box()."""

    def _setup(self, name, n_bins, device, label_budget, label_dtype):
        self.name = name
        self.mol = Molecule(name)
        self.L = int(n_bins)
        self.device = torch.device(device)
        self.ff = AmberOBC(systems.paths(name)[1]).to(self.device)
        self.kT = systems.KB * self.ff.temperature
        self.label_dtype = label_dtype
        self.D = self.mol.D
        n = self.mol.n
        # configurations per energy call: z, a copy, and ~14 (n x n) temporaries in the GB term
        self.label_chunk = max(64, int(label_budget // (2 * self.D + 14 * n * n)))

    def _finish_box(self):
        f64 = dict(dtype=torch.float64, device=self.device)
        self.width = 2.0 * self.half / self.L
        self.log_vol = float(torch.log(self.width).sum())
        c = (torch.arange(self.L, **f64) + 0.5) / self.L
        self.centres = self._from_unit(c.view(-1, 1).expand(self.L, self.D)).T.contiguous()   # (D, L)

    # -- bins <-> continuous ------------------------------------------------------
    def _from_unit(self, v: Tensor) -> Tensor:
        z = self.center + self.half * (2.0 * v - 1.0)
        return torch.where(self.is_dihedral, wrap(z), z)

    def dequantize(self, b: Tensor, generator=None) -> Tensor:
        u = torch.rand(b.shape, dtype=torch.float64, device=b.device, generator=generator)
        return self._from_unit((b.double() + u) / self.L)

    def to_bins(self, z: Tensor) -> tuple[Tensor, Tensor]:
        z = z.to(self.center)
        off = torch.where(self.is_dihedral, wrap(z - self.center), z - self.center)
        v = (off + self.half) / (2.0 * self.half)
        inside = ((v >= 0) & (v < 1)).all(-1)
        return (v * self.L).floor().long().clamp(0, self.L - 1), inside

    # -- energies -------------------------------------------------------------------
    def reduced(self, z: Tensor) -> Tensor:
        return self.ff(self.to_cartesian(z)) / self.kT - self.log_jacobian(z)

    def log_p(self, z: Tensor) -> Tensor:
        out = torch.empty(z.shape[0], dtype=torch.float64, device=z.device)
        for s in range(0, z.shape[0], self.label_chunk):
            out[s:s + self.label_chunk] = -self.reduced(z[s:s + self.label_chunk]) + self.log_vol
        return out

    @torch.no_grad()
    def gibbs_labels(self, z: Tensor) -> Tensor:
        """(B, D) configurations -> (B, D, L) log heat-bath conditionals of every coordinate."""
        B, D, L = z.shape[0], self.D, self.L
        dt = self.label_dtype
        zc, cent = z.to(dt), self.centres.to(dt)
        out = torch.empty(B, D, L, dtype=torch.float32, device=z.device)
        step = max(1, self.label_chunk // (D * L))
        ar = torch.arange(D, device=z.device)
        for s in range(0, B, step):
            zb = zc[s:s + step]
            m = zb.shape[0]
            var = zb.view(m, 1, 1, D).expand(m, D, L, D).clone()
            # row (d, c) of configuration i: coordinate d set to its bin centre c
            var.scatter_(3, ar.view(1, D, 1, 1).expand(m, D, L, 1), cent.unsqueeze(0).expand(m, D, L).unsqueeze(-1))
            flat = var.view(-1, D)
            if flat.shape[0] <= self.label_chunk:
                U = self.reduced(flat).view(m, D, L)
            else:          # a single configuration needs more than one energy call
                U = torch.cat([self.reduced(flat[q:q + self.label_chunk])
                               for q in range(0, flat.shape[0], self.label_chunk)]).view(m, D, L)
            out[s:s + step] = torch.log_softmax(-U, dim=-1).float()
        return out

    # -- observables -------------------------------------------------------------------
    @torch.no_grad()
    def observables(self, z: Tensor) -> dict:
        x = self.to_cartesian(z)
        return {"energy": self.ff(x), "rama": self.mol.rama(x)}


class InternalTarget(BinnedTarget):
    coords = "internal"

    def __init__(self, name: str, n_bins: int = 64, n_sigma: float = 10.0, branch_half_deg: float = 45.0,
                 omega_half_deg: float = 60.0, device="cpu", label_budget: float = 2 ** 28,
                 label_dtype=torch.float32):
        self._setup(name, n_bins, device, label_budget, label_dtype)
        mol = self.mol
        z_min = mol.to_internal(torch.as_tensor(mol.x_min))
        p = np.load(systems.paths(name)[1])
        kb = {tuple(sorted(map(int, ij))): k for ij, k in zip(p["bond_idx"], p["bond_k"])}
        ka = {}
        for (i, j, k), kk in zip(p["angle_idx"], p["angle_k"]):
            ka[(int(i), int(j), int(k))] = kk
            ka[(int(k), int(j), int(i))] = kk
        half_deg = {"omega": omega_half_deg, "branch": branch_half_deg}
        center, half, kind = [], [], []
        for s, z0 in zip(mol.spec, z_min.tolist()):
            a = s["atoms"]
            if s["kind"] == "bond":
                center.append(z0); half.append(n_sigma * math.sqrt(self.kT / kb[tuple(sorted(a))]))
                kind.append("bond")
            elif s["kind"] == "angle":
                h = min(n_sigma * math.sqrt(self.kT / ka[tuple(a)]), z0 - 0.05, math.pi - 0.05 - z0)
                center.append(z0); half.append(h); kind.append("angle")
            elif s["dih"] == "torsion":
                center.append(0.0); half.append(math.pi); kind.append("torsion")
            else:
                center.append(z0); half.append(math.radians(half_deg[s["dih"]])); kind.append(s["dih"])
        f64 = dict(dtype=torch.float64, device=self.device)
        self.center = torch.tensor(center, **f64)
        self.half = torch.tensor(half, **f64)
        self.is_dihedral = torch.tensor([s["kind"] == "dihedral" for s in mol.spec], device=self.device)
        self.kind = kind
        self._finish_box()
        self.z_min = z_min.to(self.device)

    def to_cartesian(self, z: Tensor) -> Tensor:
        return self.mol.to_cartesian(z)

    def log_jacobian(self, z: Tensor) -> Tensor:
        return self.mol.log_jacobian(z)

    def from_internal(self, z_int: Tensor) -> Tensor:
        """Internal coordinates of molecule.py (the reference's format) -> this target's coordinates."""
        return z_int.to(self.center)


def canonical_frame(x: Tensor, root: int, second: int, third: int) -> Tensor:
    """Rotate/translate (..., n, 3) so root is at 0, second on +x, third in the xy-plane with y > 0."""
    x = x - x[..., root:root + 1, :]
    e1 = x[..., second, :]
    e1 = e1 / e1.norm(dim=-1, keepdim=True)
    u = x[..., third, :]
    e2 = u - (u * e1).sum(-1, keepdim=True) * e1
    e2 = e2 / e2.norm(dim=-1, keepdim=True)
    e3 = torch.cross(e1, e2, dim=-1)
    R = torch.stack([e1, e2, e3], dim=-2)                    # rows: new axes
    return x @ R.transpose(-1, -2)


class CartesianTarget(BinnedTarget):
    coords = "cartesian"

    def __init__(self, name: str, n_bins: int = 512, margin: float = 0.12, n_sweep: int = 200000,
                 pivot_axes: str = "backbone", device="cpu", label_budget: float = 2 ** 28,
                 label_dtype=torch.float32, seed: int = 0):
        self._setup(name, n_bins, device, label_budget, label_dtype)
        mol = self.mol
        n = mol.n
        self.root, self.second, self.third = mol.root, mol.second, mol.third
        self.tree = [(i, mol.parent[i]) for i in mol.order[3:]]          # placement order, parent first
        self._tree_i = torch.tensor([i for i, _ in self.tree])
        self._tree_p = torch.tensor([p for _, p in self.tree])
        # chirality: (N, CA, C, CB) of every residue that has them
        chi = []
        for r, _ in mol.residues:
            keys = [(r, "N"), (r, "CA"), (r, "C"), (r, "CB")]
            if all(k in mol.idx for k in keys):
                chi.append([mol.idx[k] for k in keys])
        self.chiral = torch.tensor(chi, dtype=torch.long, device=self.device)

        # box from a torsion sweep of the minimized structure
        g = torch.Generator().manual_seed(seed)
        zi = mol.to_internal(torch.as_tensor(mol.x_min)).expand(n_sweep, -1).clone()
        tor = [c for c, k in enumerate(mol.kinds()) if k == "torsion"]
        zi[:, tor] = (torch.rand(n_sweep, len(tor), generator=g, dtype=torch.float64) * 2 - 1) * math.pi
        Zc = self.from_cartesian(mol.to_cartesian(zi))
        lo, hi = Zc.min(0).values - margin, Zc.max(0).values + margin
        # methyl hydrogens share a box per component
        self.methyls = []
        for c in range(n):
            hs = sorted(h for h in mol.nbrs[c] if mol.atoms[h]["element"] == "H")
            if mol.atoms[c]["element"] == "C" and len(hs) == 3:
                self.methyls.append(hs)
        for hs in self.methyls:
            for ax in range(3):
                cols = [self._col(h, ax) for h in hs]
                lo[cols] = lo[cols].min(); hi[cols] = hi[cols].max()
        f64 = dict(dtype=torch.float64, device=self.device)
        self.center = ((lo + hi) / 2).to(**f64)
        self.half = ((hi - lo) / 2).to(**f64)
        self.is_dihedral = torch.zeros(self.D, dtype=torch.bool, device=self.device)
        self.kind = ["frame"] * 3 + [f"{mol.atoms[i]['element']}{'xyz'[a]}" for i, _ in self.tree for a in range(3)]
        self._finish_box()
        self.z_min = self.from_cartesian(torch.as_tensor(mol.x_min)).to(self.device)

        # rotatable bonds for pivot moves: the axis (k -> j) of every full-circle torsion of the
        # Z-matrix, rotating everything below j in the spanning tree (never the root triple).
        # "backbone" keeps only the phi (N-CA) and psi (CA-C) bonds: methyl rotors mix by
        # themselves, and on AL4 the same number of backbone-only moves recovers more of the
        # beta basin (pivot diagnostic in the README).
        assert pivot_axes in ("backbone", "all"), pivot_axes
        bb = {frozenset(q[1:3]) for qq in mol.rama_quads.tolist() for q in qq}
        axes, masks = [], []
        for s in mol.spec:
            if s["kind"] == "dihedral" and s["dih"] == "torsion":
                _, j, k, _ = s["atoms"]
                if pivot_axes == "backbone" and frozenset((j, k)) not in bb:
                    continue
                below, st_ = [], list(mol.children[j])
                while st_:
                    u = st_.pop(); below.append(u); st_ += mol.children[u]
                m = torch.zeros(n, dtype=torch.bool); m[below] = True
                axes.append((k, j)); masks.append(m)
        self.pivot_axes = torch.tensor(axes, dtype=torch.long, device=self.device)        # (A, 2)
        self.pivot_masks = torch.stack(masks).to(self.device)                              # (A, n)

    # -- coordinates --------------------------------------------------------------
    def _col(self, atom: int, ax: int) -> int:
        return 3 + 3 * [i for i, _ in self.tree].index(atom) + ax

    def from_cartesian(self, X: Tensor) -> Tensor:
        X = canonical_frame(X, self.root, self.second, self.third)
        head = torch.stack([X[..., self.second, 0], X[..., self.third, 0], X[..., self.third, 1]], -1)
        rest = X[..., self._tree_i.to(X.device), :] - X[..., self._tree_p.to(X.device), :]
        return torch.cat([head, rest.reshape(*rest.shape[:-2], -1)], -1)

    def from_internal(self, z_int: Tensor) -> Tensor:
        return self.from_cartesian(self.mol.to_cartesian(z_int.double().cpu())).to(self.device)

    def to_cartesian(self, z: Tensor) -> Tensor:
        n = self.mol.n
        pos = torch.zeros(*z.shape[:-1], n, 3, dtype=z.dtype, device=z.device)
        pos[..., self.second, 0] = z[..., 0]
        pos[..., self.third, 0] = z[..., 1]
        pos[..., self.third, 1] = z[..., 2]
        rest = z[..., 3:].reshape(*z.shape[:-1], n - 3, 3)
        for k, (i, p) in enumerate(self.tree):
            pos[..., i, :] = pos[..., p, :] + rest[..., k, :]
        return pos

    def log_jacobian(self, z: Tensor) -> Tensor:
        return 2.0 * torch.log(z[..., 0]) + torch.log(z[..., 2])

    def reduced(self, z: Tensor) -> Tensor:
        x = self.to_cartesian(z)
        U = self.ff(x) / self.kT - self.log_jacobian(z)
        q = self.chiral.to(x.device)
        ca = x[..., q[:, 1], :]
        vol = ((x[..., q[:, 0], :] - ca) * torch.cross(x[..., q[:, 2], :] - ca, x[..., q[:, 3], :] - ca, dim=-1)).sum(-1)
        return U + CHIRAL_PENALTY * (vol <= 0).any(-1).to(U.dtype)

    # -- pivot moves --------------------------------------------------------------------
    @torch.no_grad()
    def pivot_mh(self, z: Tensor, n_steps: int, generator=None) -> tuple[Tensor, dict]:
        """n_steps Metropolis pivot moves on (B, D) configurations: a uniform random rotation about a
        random rotatable bond of the subtree below it.  The move is a rigid rotation of a subset of
        atoms about an axis through atoms that stay put, so it preserves volume (and the frame
        coordinates of the root triple), the proposal is symmetric, and the acceptance is
        min(1, exp(U - U')) inside the box.  p-invariant, so importance weights carry over unchanged."""
        B = z.shape[0]
        dev = z.device
        U = torch.cat([self.reduced(z[s:s + self.label_chunk]) for s in range(0, B, self.label_chunk)])
        acc = torch.zeros(B, device=dev)
        ar = torch.arange(B, device=dev)
        for _ in range(n_steps):
            X = self.to_cartesian(z)
            a = torch.randint(0, len(self.pivot_axes), (B,), device=dev, generator=generator)
            th = (torch.rand(B, dtype=z.dtype, device=dev, generator=generator) * 2 - 1) * math.pi
            k, j = self.pivot_axes[a, 0], self.pivot_axes[a, 1]
            o = X[ar, j]                                                       # (B, 3) on the axis
            u = o - X[ar, k]
            u = u / u.norm(dim=-1, keepdim=True)
            v = X - o.unsqueeze(1)
            c, s_ = torch.cos(th).view(B, 1, 1), torch.sin(th).view(B, 1, 1)
            uu = u.unsqueeze(1)
            rot = v * c + torch.cross(uu.expand_as(v), v, dim=-1) * s_ + uu * (v * uu).sum(-1, keepdim=True) * (1 - c)
            Xn = torch.where(self.pivot_masks[a].unsqueeze(-1), rot + o.unsqueeze(1), X)
            zn = self.from_cartesian(Xn)
            _, inside = self.to_bins(zn)
            Un = torch.cat([self.reduced(zn[s:s + self.label_chunk]) for s in range(0, B, self.label_chunk)])
            ok = inside & (torch.rand(B, dtype=U.dtype, device=dev, generator=generator).log() < U - Un)
            z = torch.where(ok.unsqueeze(-1), zn, z)
            U = torch.where(ok, Un, U)
            acc += ok.float()
        return z, {"pivot_accept": float(acc.mean()) / max(n_steps, 1)}


def make_target(a, device, label_budget: float = 2 ** 28) -> BinnedTarget:
    """The target of a train.py argument namespace."""
    ldt = torch.float64 if a.label_dtype == "float64" else torch.float32
    kw = dict(device=device, label_budget=label_budget, label_dtype=ldt)
    if a.coords == "internal":
        return InternalTarget(a.system, a.bins, a.n_sigma, a.branch_half, a.omega_half, **kw)
    return CartesianTarget(a.system, a.bins, a.cart_margin, pivot_axes=a.pivot_axes, **kw)
