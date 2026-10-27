"""amber99sbildn + OBC implicit solvent in torch, batched over configurations.

Every parameter is read out of the assembled OpenMM System
(systems.export_parameters), and every term follows the OpenMM reference
platform formula, so this is a copy of the potential rather than a model of
it (tests/test_digin.py compares each force group with OpenMM).

A Gibbs label needs the energy of L variants of one configuration per
coordinate; OpenMM holds one configuration per Context, so the scans run here
as one batched call.

At coincident atoms Lennard-Jones goes to +inf and Coulomb to -inf, and their
sum is NaN rather than the enormous positive energy the physics calls for.
Distances in the nonbonded and GB terms are therefore floored at
R_FLOOR = 0.005 nm, a twentieth of the closest physical contact, so every
physical configuration is untouched; inf/NaN totals become E_CAP.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor, nn

ONE_4PI_EPS0 = 138.93545764438198  # kJ/mol nm / e^2, OpenMM 8's constant
R_FLOOR = 0.005                    # nm
OBC_ALPHA, OBC_BETA, OBC_GAMMA = 1.0, 0.8, 4.85
DIELECTRIC_OFFSET = 0.009          # nm
PROBE_RADIUS = 0.14                # nm
E_CAP = 1e12                       # kJ/mol, replaces inf/nan


def _dihedral(p0, p1, p2, p3):
    b0, b1, b2 = p1 - p0, p2 - p1, p3 - p2
    n1 = torch.cross(b0, b1, dim=-1)
    n2 = torch.cross(b1, b2, dim=-1)
    y = (b1.norm(dim=-1, keepdim=True) * b0 * n2).sum(-1)
    x = (n1 * n2).sum(-1)
    return torch.atan2(y, x)


class AmberOBC(nn.Module):
    """E(x) in kJ/mol for x of shape (..., n, 3) in nm."""

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
        # OBC2
        self.register_buffer("gb_q", t(p["gb_charge"]))
        self.register_buffer("gb_rho", t(p["gb_radius"]))
        self.register_buffer("gb_scale", t(p["gb_scale"]))
        self.gb_pref = -ONE_4PI_EPS0 * (1.0 / float(p["gb_solute_eps"]) - 1.0 / float(p["gb_solvent_eps"]))
        self.sa_factor = 4.0 * math.pi * float(p["gb_sa_energy"])
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
        """r: (..., n, n) full distance matrix, diagonal ignored."""
        rho = self.gb_rho.to(r)
        orad = rho - DIELECTRIC_OFFSET                  # offset radius of i
        sr = orad * self.gb_scale.to(r)                 # scaled offset radius of j
        ori = orad.view(-1, 1)
        srj = sr.view(1, -1)
        eye = torch.eye(self.n, dtype=torch.bool, device=r.device)
        rr = torch.where(eye, torch.ones_like(r), r)
        active = (ori < rr + srj) & ~eye
        L = torch.maximum(ori.expand_as(rr), (rr - srj).abs())
        l = 1.0 / L
        u = 1.0 / (rr + srj)
        term = (l - u + 0.25 * rr * (u * u - l * l) + 0.5 / rr * torch.log(u / l)
                + 0.25 * srj * srj / rr * (l * l - u * u))
        term = term + torch.where(ori < srj - rr, 2.0 * (1.0 / ori - l), torch.zeros_like(term))
        psi = 0.5 * ori.view(-1) * torch.where(active, term, torch.zeros_like(term)).sum(-1)
        th = torch.tanh(OBC_ALPHA * psi - OBC_BETA * psi ** 2 + OBC_GAMMA * psi ** 3)
        return 1.0 / (1.0 / orad - th / rho)

    def gb(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """(polar, nonpolar ACE) energies."""
        r = (x.unsqueeze(-2) - x.unsqueeze(-3)).norm(dim=-1).clamp_min(R_FLOOR)
        B = self.born_radii(r)
        q = self.gb_q.to(x)
        BB = B.unsqueeze(-1) * B.unsqueeze(-2)
        f = torch.sqrt(r * r + BB * torch.exp(-r * r / (4.0 * BB)))
        iu = torch.triu_indices(self.n, self.n, 1, device=x.device)
        qq = (q[iu[0]] * q[iu[1]])
        pair = (qq / f[..., iu[0], iu[1]]).sum(-1)
        self_ = 0.5 * (q * q / B).sum(-1)
        polar = self.gb_pref * (pair + self_)
        rho = self.gb_rho.to(x)
        sa = self.sa_factor * ((rho + PROBE_RADIUS) ** 2 * (rho / B) ** 6).sum(-1)
        return polar, sa

    def terms(self, x: Tensor) -> dict:
        polar, sa = self.gb(x)
        return {"bond": self.bonds(x), "angle": self.angles(x), "torsion": self.torsions(x),
                "nonbonded": self.nonbonded(x), "gb": polar + sa}

    def forward(self, x: Tensor) -> Tensor:
        polar, sa = self.gb(x)
        e = self.bonds(x) + self.angles(x) + self.torsions(x) + self.nonbonded(x) + polar + sa
        return torch.nan_to_num(e, nan=E_CAP, posinf=E_CAP, neginf=E_CAP).clamp(max=E_CAP)
