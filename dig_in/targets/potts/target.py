"""q-state Potts model on a periodic square lattice.

States x in {0, ..., q-1}^D.  The energy is

    E(x) = -J * sum_<ij> delta(x_i, x_j),

with nearest-neighbour bonds on the periodic lattice (each bond counted
once).  The critical point of the 2-D model is
beta_c J = log(1 + sqrt(q)).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from dig_in.targets.base import DiscreteTarget


class PottsModel(DiscreteTarget):
    def __init__(
        self,
        lattice_shape: tuple[int, ...] = (8, 8),
        q: int = 3,
        beta: float = 1.005,
        J: float = 1.0,
    ):
        lattice_shape = tuple(lattice_shape)
        D = 1
        for s in lattice_shape:
            D *= s
        super().__init__(dim=D, num_states=q, beta=beta)
        self.lattice_shape = lattice_shape
        self.q = q
        self.J = J

    # ------------------------------------------------------------------
    def lattice(self, x: Tensor) -> Tensor:
        """(B, D) -> (B, *lattice_shape), long."""
        return x.view(x.shape[0], *self.lattice_shape)

    def neighbor_counts(self, x: Tensor) -> Tensor:
        """One-hot counts of neighbour values, shape (B, *lattice_shape, q)."""
        z = self.lattice(x)
        onehot = F.one_hot(z, self.q).to(torch.get_default_dtype())  # (B, *shape, q)
        cnt = torch.zeros_like(onehot)
        for axis in range(1, z.dim()):
            cnt = cnt + torch.roll(onehot, 1, dims=axis)
            cnt = cnt + torch.roll(onehot, -1, dims=axis)
        return cnt

    # ------------------------------------------------------------------
    def energy(self, x: Tensor, cond: Tensor | None = None) -> Tensor:
        z = self.lattice(x)
        bond = torch.zeros(z.shape[0], device=x.device)
        for axis in range(1, z.dim()):
            eq = (z == torch.roll(z, 1, dims=axis)).to(bond.dtype)
            bond = bond + eq.flatten(1).sum(-1)
        return -self.J * bond

    def energy_diff(self, x: Tensor, cond: Tensor | None = None) -> Tensor:
        cnt = self.neighbor_counts(x).flatten(1, -2)  # (B, D, q)
        cur = torch.gather(cnt, 2, x.unsqueeze(-1))  # (B, D, 1) matches of current value
        # E(x^{d<-n}) - E(x) = -J [cnt(n) - cnt(x_d)]
        return -self.J * (cnt - cur)

    # ------------------------------------------------------------------
    # Observables
    # ------------------------------------------------------------------
    def magnetization(self, x: Tensor) -> Tensor:
        """Normalized order parameter (q * f_max - 1) / (q - 1), shape (B,)."""
        counts = F.one_hot(x, self.q).to(torch.get_default_dtype()).mean(dim=1)  # (B, q)
        f_max = counts.max(dim=-1).values
        return (self.q * f_max - 1.0) / (self.q - 1.0)

    def energy_per_site(self, x: Tensor) -> Tensor:
        return self.energy(x) / self.D

    def exact_log_Z(self) -> float | None:
        if self.N**self.D <= 2**24:
            return self.log_Z_enumeration()
        if len(self.lattice_shape) == 2:
            from dig_in.targets.potts import exact

            rows, cols = self.lattice_shape
            if cols > rows:
                rows, cols = cols, rows
            if self.q**cols <= 6561:
                return exact.log_Z_transfer(rows, cols, self.q, self.beta * self.J)
        return None
