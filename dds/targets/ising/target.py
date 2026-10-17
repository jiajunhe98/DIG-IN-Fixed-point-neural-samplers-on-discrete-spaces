"""Ferromagnetic Ising model on a periodic square lattice.

States are encoded as x in {0, 1}^D with spins s = 2x - 1 in {-1, +1};
D = prod(lattice_shape).  The energy is

    E(s) = -J * sum_<ij> s_i s_j  -  h * sum_i s_i,

with nearest-neighbour bonds on the periodic lattice.  The inverse
temperature beta multiplies the energy in log_prob (see DiscreteTarget),
so (beta, J) enter only through the product beta*J (and beta*h).
"""

from __future__ import annotations

import torch
from torch import Tensor

from dds.targets.base import DiscreteTarget


class IsingModel(DiscreteTarget):
    def __init__(
        self,
        lattice_shape: tuple[int, ...] = (8, 8),
        beta: float = 0.4407,
        J: float = 1.0,
        h: float = 0.0,
    ):
        lattice_shape = tuple(lattice_shape)
        D = 1
        for s in lattice_shape:
            D *= s
        super().__init__(dim=D, num_states=2, beta=beta)
        self.lattice_shape = lattice_shape
        self.J = J
        self.h = h

    # ------------------------------------------------------------------
    def spins(self, x: Tensor) -> Tensor:
        """(B, D) in {0,1} -> (B, *lattice_shape) in {-1,+1}, float."""
        s = (2.0 * x - 1.0).view(x.shape[0], *self.lattice_shape)
        return s.to(torch.get_default_dtype())

    def neighbor_sum(self, s: Tensor) -> Tensor:
        """Sum of nearest-neighbour spins, same shape as s."""
        m = torch.zeros_like(s)
        for axis in range(1, s.dim()):
            m = m + torch.roll(s, 1, dims=axis) + torch.roll(s, -1, dims=axis)
        return m

    # ------------------------------------------------------------------
    def energy(self, x: Tensor, cond: Tensor | None = None) -> Tensor:
        s = self.spins(x)
        bond = torch.zeros(s.shape[0], device=s.device)
        for axis in range(1, s.dim()):
            bond = bond + (s * torch.roll(s, 1, dims=axis)).flatten(1).sum(-1)
        field = s.flatten(1).sum(-1)
        return -self.J * bond - self.h * field

    def energy_diff(self, x: Tensor, cond: Tensor | None = None) -> Tensor:
        s = self.spins(x)
        m = self.neighbor_sum(s)
        # Energy change of flipping site i: 2 s_i (J m_i + h).
        flip = (2.0 * s * (self.J * m + self.h)).flatten(1)  # (B, D)
        de = torch.zeros(x.shape[0], self.D, 2, device=x.device, dtype=flip.dtype)
        # setting coordinate to its current value costs 0; to the other value costs `flip`
        other = 1 - x
        de.scatter_(2, other.unsqueeze(-1), flip.unsqueeze(-1))
        return de

    # ------------------------------------------------------------------
    # Observables
    # ------------------------------------------------------------------
    def magnetization(self, x: Tensor) -> Tensor:
        """Per-site magnetization m(x) in [-1, 1], shape (B,)."""
        return (2.0 * x - 1.0).to(torch.get_default_dtype()).mean(dim=1)

    def energy_per_site(self, x: Tensor) -> Tensor:
        return self.energy(x) / self.D

    def exact_log_Z(self) -> float | None:
        from dds.targets.ising import exact

        if self.N**self.D <= 2**24:
            return self.log_Z_enumeration()
        if len(self.lattice_shape) == 2 and self.h == 0.0:
            return exact.log_Z_kaufman(
                self.lattice_shape[0], self.lattice_shape[1], self.beta * self.J
            )
        return None
