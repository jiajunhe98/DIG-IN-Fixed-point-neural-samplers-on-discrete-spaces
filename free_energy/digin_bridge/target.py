"""The two binned endpoints of FEAT's ALDP tasks in internal coordinates.

The box: bonds and angles +- n_sigma sqrt(kT/k) around the minimum,
phi/psi/methyl rotors the full circle, omega trans +- 60 deg, branch dihedrals
+- 45 deg, with the ff96 force constants and the ff96 minimum of system.py.
Both endpoints share the box, with

    U_lambda(z) = (base(x(z)) + lambda^2 polar(x(z))) / kT - log|J(z)|

  aldps   endpoint 0 = vacuum (lambda = 0), endpoint 1 = solvent (lambda = 1)
  aldpt   both vacuum; endpoint 0 = S_b (phi outside (0, 2.15)), endpoint 1 =
          S_a (phi inside, C7ax), i.e. U + inf outside the state (FEAT F.1.4)

Endpoint 0 is given by samples, endpoint 1 by its energy.  Bins are uniform
per coordinate, so every cell has the same volume and the cell volumes cancel
in the bridge's path weight.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor

from digin_bridge import system as sy
from digin_bridge import zmatrix as zm
from digin_bridge.forcefield import AmberOBC1

# lambda of each endpoint, and for aldpt the phi interval of S_a (FEAT's PHI_A)
TASKS = {"aldps": {"lamb": (0.0, 1.0), "phi_a": None},
         "aldpt": {"lamb": (0.0, 0.0), "phi_a": (0.0, 2.15)}}


class BridgeTarget:
    def __init__(self, n_bins: int = 64, n_sigma: float = 10.0, branch_half_deg: float = 45.0,
                 omega_half_deg: float = 60.0, device="cpu", label_chunk: int = 32768,
                 label_dtype=torch.float32, task: str = "aldps"):
        self.task = task
        self.lamb = TASKS[task]["lamb"]
        self.phi_a = TASKS[task]["phi_a"]
        self.L = int(n_bins)
        self.D = zm.D
        self.device = torch.device(device)
        self.ff = AmberOBC1(sy.FF_NPZ).to(self.device)
        self.kT = sy.KT
        self.label_chunk = label_chunk
        self.label_dtype = label_dtype
        z_min = zm.to_internal(torch.as_tensor(sy.read_pdb()))
        p = np.load(sy.FF_NPZ)
        kb = {tuple(sorted(map(int, ij))): k for ij, k in zip(p["bond_idx"], p["bond_k"])}
        ka = {}
        for (i, j, k), kk in zip(p["angle_idx"], p["angle_k"]):
            ka[(int(i), int(j), int(k))] = kk
            ka[(int(k), int(j), int(i))] = kk
        center, half, kind = [], [], []
        for s, z0 in zip(zm.SPEC, z_min.tolist()):
            a = s["atoms"]
            if s["kind"] == "bond":
                center.append(z0); half.append(n_sigma * math.sqrt(self.kT / kb[tuple(sorted(a))])); kind.append("bond")
            elif s["kind"] == "angle":
                h = min(n_sigma * math.sqrt(self.kT / ka[tuple(a)]), z0 - 0.05, math.pi - 0.05 - z0)
                center.append(z0); half.append(h); kind.append("angle")
            elif s["dih"] == "torsion":
                center.append(0.0); half.append(math.pi); kind.append("torsion")
            elif s["dih"] == "omega":
                center.append(math.pi); half.append(math.radians(omega_half_deg)); kind.append("omega")
            else:
                center.append(z0); half.append(math.radians(branch_half_deg)); kind.append("branch")
        f64 = dict(dtype=torch.float64, device=self.device)
        self.center = torch.tensor(center, **f64)
        self.half = torch.tensor(half, **f64)
        self.is_dihedral = torch.tensor([s["kind"] == "dihedral" for s in zm.SPEC], device=self.device)
        self.kind = kind
        self.width = 2.0 * self.half / self.L
        self.log_vol = float(torch.log(self.width).sum())
        c = (torch.arange(self.L, **f64) + 0.5) / self.L
        self.centres = self._from_unit(c.view(1, -1).expand(self.D, -1).T).T.contiguous()   # (D, L)
        self.z_min = z_min.to(self.device)
        # endpoint 1's state restriction enters the label of phi only: bins whose centre is outside get -inf
        self.label_mask = torch.zeros(self.D, self.L, **f64)
        if self.phi_a is not None:
            self.label_mask[zm.PHI_INDEX] = torch.where(self.in_state_a(self.centres[zm.PHI_INDEX]), 0.0, -math.inf)

    # -- states ---------------------------------------------------------------
    def in_state_a(self, phi: Tensor) -> Tensor:
        return (phi > self.phi_a[0]) & (phi < self.phi_a[1])

    def in_endpoint(self, z: Tensor, which: int) -> Tensor:
        """Whether z lies in the support of endpoint `which` (always, for aldps)."""
        if self.phi_a is None:
            return torch.ones(z.shape[:-1], dtype=torch.bool, device=z.device)
        a = self.in_state_a(z[..., zm.PHI_INDEX])
        return a if which == 1 else ~a

    # -- bins <-> continuous ------------------------------------------------
    def _from_unit(self, v: Tensor) -> Tensor:
        z = self.center + self.half * (2.0 * v - 1.0)
        return torch.where(self.is_dihedral, zm.wrap(z), z)

    def dequantize(self, b: Tensor, generator=None) -> Tensor:
        u = torch.rand(b.shape, dtype=torch.float64, device=b.device, generator=generator)
        return self._from_unit((b.double() + u) / self.L)

    def to_bins(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """(bins, inside-the-box mask) of continuous internal coordinates."""
        z = z.to(self.center)
        off = torch.where(self.is_dihedral, zm.wrap(z - self.center), z - self.center)
        v = (off + self.half) / (2.0 * self.half)
        inside = ((v >= 0) & (v < 1)).all(-1)
        b = (v * self.L).floor().long().clamp(0, self.L - 1)
        return b, inside

    # -- energies -----------------------------------------------------------
    def parts(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """(base, polar) energies in kJ/mol of internal coordinates (..., D)."""
        return self.ff(zm.to_cartesian(z))

    def reduced(self, z: Tensor) -> tuple[Tensor, Tensor]:
        """(U_0, U_1) of the two endpoints: E_lambda / kT - log|J|, + inf outside the state."""
        base, polar = self.parts(z)
        lj = zm.log_jacobian(z)
        U = [(base + l * l * polar) / self.kT - lj for l in self.lamb]
        return tuple(torch.where(self.in_endpoint(z, k), U[k], math.inf) for k in (0, 1))

    def energy1(self, z: Tensor) -> Tensor:
        """Potential energy (kJ/mol) of endpoint 1's force field, for the plots."""
        base, polar = self.parts(z)
        return base + self.lamb[1] ** 2 * polar

    @torch.no_grad()
    def gibbs_labels(self, z: Tensor) -> Tensor:
        """(B, D) dequantized configurations -> (B, D, L) log heat-bath conditionals of endpoint 1.

        For aldpt the restriction to S_a depends on phi alone: it masks the
        phi label and is a constant factor of every other scan, which is
        therefore left out; that keeps the labels of an endpoint outside S_a
        (weight 0) defined."""
        lamb = self.lamb[1]
        B, D, L = z.shape[0], self.D, self.L
        zc = z.to(self.label_dtype)
        cent = self.centres.to(self.label_dtype)
        out = torch.empty(B, D, L, dtype=torch.float32, device=z.device)
        eye = torch.eye(D, dtype=torch.bool, device=z.device).view(1, D, D, 1)
        step = max(1, self.label_chunk // (D * L))
        for s in range(0, B, step):
            zb = zc[s:s + step]
            n = zb.shape[0]
            # var[b, d, k, c] = centre c of coordinate d if k == d else zb[b, k]
            base = zb.view(n, 1, D, 1).expand(n, D, D, L)
            repl = cent.view(1, D, 1, L).expand(n, D, D, L)
            var = torch.where(eye, repl, base).permute(0, 1, 3, 2).reshape(-1, D)
            e_base, e_polar = self.parts(var)
            U = (e_base + lamb ** 2 * e_polar) / self.kT - zm.log_jacobian(var)
            out[s:s + step] = torch.log_softmax(-U.view(n, D, L) + self.label_mask.to(U), dim=-1).float()
        return out

    def phi_psi(self, z: Tensor) -> tuple[Tensor, Tensor]:
        return z[..., zm.PHI_INDEX], z[..., zm.PSI_INDEX]
