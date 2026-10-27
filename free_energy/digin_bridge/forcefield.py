"""amber ff96 + OBC1 (FEAT's ALDP-S) in torch, batched over configurations.

Every parameter is read out of the OpenMM System (system.export_parameters)
and every term follows the expressions of OpenMM's CustomGBForce for OBC1, so
this is a copy of the potential (tests/test_bridge.py compares each force
group and both lambda states with OpenMM).  Unlike OpenMM's GBSAOBCForce
(OBC2) it has OBC1's tanh(0.8 psi + 2.909125 psi^3), no extra term for an atom
engulfed by a neighbour's descreening sphere, and the CustomGBForce's Coulomb
constant 138.935485.

The energy is returned in two parts, base + lambda^2 polar: `base` holds the
bonded, nonbonded and ACE surface terms, `polar` the generalized-Born
electrostatics at full charge.  One evaluation gives both endpoints.

Distances are floored at R_FLOOR = 0.005 nm, so coincident
atoms cost an enormous energy instead of NaN.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import Tensor, nn

ONE_4PI_EPS0 = 138.93545764438198  # kJ/mol nm / e^2, OpenMM 8's NonbondedForce
GB_COULOMB = 138.935485            # the constant written into OpenMM's OBC1 CustomGBForce
SOLUTE_EPS, SOLVENT_EPS = 1.0, 78.5
OBC1_ALPHA, OBC1_GAMMA = 0.8, 2.909125
DIELECTRIC_OFFSET = 0.009          # nm
PROBE_RADIUS = 0.14                # nm
SA_FACTOR = 28.3919551             # kJ/mol/nm^2, 4 pi * 2.25936
R_FLOOR = 0.005                    # nm
E_CAP = 1e12                       # kJ/mol, replaces inf/nan


def _dihedral(p0, p1, p2, p3):
    b0, b1, b2 = p1 - p0, p2 - p1, p3 - p2
    n1 = torch.cross(b0, b1, dim=-1)
    n2 = torch.cross(b1, b2, dim=-1)
    y = (b1.norm(dim=-1, keepdim=True) * b0 * n2).sum(-1)
    x = (n1 * n2).sum(-1)
    return torch.atan2(y, x)


class AmberOBC1(nn.Module):
    """(base, polar) energies in kJ/mol for x of shape (..., 22, 3) in nm."""

    def __init__(self, npz_path: str):
        super().__init__()
        p = np.load(npz_path)
        n = len(p["charge"])
        self.n = n
        t = lambda a, dt=torch.float64: torch.as_tensor(np.asarray(a), dtype=dt)
        self.register_buffer("bond_idx", t(p["bond_idx"], torch.long))
        self.register_buffer("bond_r0", t(p["bond_r0"]))
        self.register_buffer("bond_k", t(p["bond_k"]))
        self.register_buffer("angle_idx", t(p["angle_idx"], torch.long))
        self.register_buffer("angle_t0", t(p["angle_t0"]))
        self.register_buffer("angle_k", t(p["angle_k"]))
        self.register_buffer("tors_idx", t(p["tors_idx"], torch.long))
        self.register_buffer("tors_n", t(p["tors_n"]))
        self.register_buffer("tors_phase", t(p["tors_phase"]))
        self.register_buffer("tors_k", t(p["tors_k"]))
        # nonbonded pair table over i < j with the exceptions substituted
        iu = np.triu_indices(n, 1)
        q, sig, eps = p["charge"], p["sigma"], p["epsilon"]
        qq = q[iu[0]] * q[iu[1]]
        ss = 0.5 * (sig[iu[0]] + sig[iu[1]])
        ee = np.sqrt(eps[iu[0]] * eps[iu[1]])
        pos = {(int(a), int(b)): k for k, (a, b) in enumerate(zip(*iu))}
        for (a, b), cq, cs, ce in zip(p["exc_idx"], p["exc_qq"], p["exc_sigma"], p["exc_eps"]):
            k = pos[(min(a, b), max(a, b))]
            qq[k], ss[k], ee[k] = cq, cs, ce
        self.register_buffer("pair_i", t(iu[0], torch.long))
        self.register_buffer("pair_j", t(iu[1], torch.long))
        self.register_buffer("pair_qq", t(qq))
        self.register_buffer("pair_sig", t(ss))
        self.register_buffer("pair_eps", t(ee))
        # OBC1: charge, offset radius or, scaled offset radius sr
        self.register_buffer("gb_q", t(p["gb_charge"]))
        self.register_buffer("gb_or", t(p["gb_or"]))
        self.register_buffer("gb_sr", t(p["gb_sr"]))
        self.gb_pref = -GB_COULOMB * (1.0 / SOLUTE_EPS - 1.0 / SOLVENT_EPS)
        self.temperature = float(p["temperature"])

    # -- terms -------------------------------------------------------------
    def bonds(self, x: Tensor) -> Tensor:
        i, j = self.bond_idx.T
        r = (x[..., i, :] - x[..., j, :]).norm(dim=-1)
        return (0.5 * self.bond_k.to(x) * (r - self.bond_r0.to(x)) ** 2).sum(-1)

    def angles(self, x: Tensor) -> Tensor:
        i, j, k = self.angle_idx.T
        a = x[..., i, :] - x[..., j, :]
        b = x[..., k, :] - x[..., j, :]
        c = (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1)).clamp_min(1e-12)
        th = torch.acos(c.clamp(-1.0, 1.0))
        return (0.5 * self.angle_k.to(x) * (th - self.angle_t0.to(x)) ** 2).sum(-1)

    def torsions(self, x: Tensor) -> Tensor:
        i, j, k, l = self.tors_idx.T
        phi = _dihedral(x[..., i, :], x[..., j, :], x[..., k, :], x[..., l, :])
        return (self.tors_k.to(x) * (1.0 + torch.cos(self.tors_n.to(x) * phi - self.tors_phase.to(x)))).sum(-1)

    def nonbonded(self, x: Tensor) -> Tensor:
        r = (x[..., self.pair_i, :] - x[..., self.pair_j, :]).norm(dim=-1).clamp_min(R_FLOOR)
        s6 = (self.pair_sig.to(x) / r) ** 6
        lj = 4.0 * self.pair_eps.to(x) * (s6 * s6 - s6)
        coul = ONE_4PI_EPS0 * self.pair_qq.to(x) / r
        return (lj + coul).sum(-1)

    def born_radii(self, r: Tensor) -> Tensor:
        """r: (..., n, n) full distance matrix, diagonal ignored.  OpenMM's computed values I and B."""
        or1 = self.gb_or.to(r).view(-1, 1)                # offset radius of i
        sr2 = self.gb_sr.to(r).view(1, -1)                # scaled offset radius of j
        eye = torch.eye(self.n, dtype=torch.bool, device=r.device)
        rr = torch.where(eye, torch.ones_like(r), r)
        U = rr + sr2
        L = torch.maximum(or1.expand_as(rr), (rr - sr2).abs())
        term = 0.5 * (1.0 / L - 1.0 / U + 0.25 * (rr - sr2 * sr2 / rr) * (1.0 / U ** 2 - 1.0 / L ** 2)
                      + 0.5 * torch.log(L / U) / rr)
        active = (rr + sr2 - or1 >= 0) & ~eye             # step(r + sr2 - or1)
        I = torch.where(active, term, torch.zeros_like(term)).sum(-1)
        orad = self.gb_or.to(r)
        psi = I * orad
        radius = orad + DIELECTRIC_OFFSET
        return 1.0 / (1.0 / orad - torch.tanh(OBC1_ALPHA * psi + OBC1_GAMMA * psi ** 3) / radius)

    def gb(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """(polar at full charge, ACE surface) energies."""
        r = (x.unsqueeze(-2) - x.unsqueeze(-3)).norm(dim=-1).clamp_min(R_FLOOR)
        B = self.born_radii(r)
        q = self.gb_q.to(x)
        BB = B.unsqueeze(-1) * B.unsqueeze(-2)
        f = torch.sqrt(r * r + BB * torch.exp(-r * r / (4.0 * BB)))
        iu = torch.triu_indices(self.n, self.n, 1, device=x.device)
        pair = (q[iu[0]] * q[iu[1]] / f[..., iu[0], iu[1]]).sum(-1)
        self_ = 0.5 * (q * q / B).sum(-1)
        polar = self.gb_pref * (pair + self_)
        radius = self.gb_or.to(x) + DIELECTRIC_OFFSET
        sa = SA_FACTOR * ((radius + PROBE_RADIUS) ** 2 * (radius / B) ** 6).sum(-1)
        return polar, sa

    def terms(self, x: Tensor) -> dict:
        polar, sa = self.gb(x)
        return {"bond": self.bonds(x), "angle": self.angles(x), "torsion": self.torsions(x),
                "nonbonded": self.nonbonded(x), "gb_polar": polar, "gb_sa": sa}

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        polar, sa = self.gb(x)
        base = self.bonds(x) + self.angles(x) + self.torsions(x) + self.nonbonded(x) + sa
        fix = lambda e: torch.nan_to_num(e, nan=E_CAP, posinf=E_CAP, neginf=-E_CAP).clamp(-E_CAP, E_CAP)
        return fix(base), fix(polar)

    def energy(self, x: Tensor, lamb: float) -> Tensor:
        base, polar = self(x)
        return base + lamb ** 2 * polar
