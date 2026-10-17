"""Reference MCMC samplers for the Ising model.

Batched checkerboard Metropolis / heat-bath sweeps (torch, GPU-friendly)
and cluster updates (Wolff, Swendsen-Wang; numpy/scipy, per chain).
Cluster updates require h = 0.  `sample()` produces decorrelated
reference samples with burn-in and thinning.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from dig_in.targets.ising.target import IsingModel
from dig_in.utils.lattice import independent_groups


def lattice_neighbors(lattice_shape: tuple[int, ...]) -> np.ndarray:
    """(D, 2*ndim) array of flat indices of nearest neighbours (periodic)."""
    D = int(np.prod(lattice_shape))
    idx = np.arange(D).reshape(lattice_shape)
    nbrs = []
    for axis in range(len(lattice_shape)):
        nbrs.append(np.roll(idx, 1, axis=axis).reshape(-1))
        nbrs.append(np.roll(idx, -1, axis=axis).reshape(-1))
    return np.stack(nbrs, axis=1)


def lattice_edges(lattice_shape: tuple[int, ...]) -> np.ndarray:
    """(n_edges, 2) array of bonds, each edge listed once (periodic)."""
    D = int(np.prod(lattice_shape))
    idx = np.arange(D).reshape(lattice_shape)
    edges = []
    for axis in range(len(lattice_shape)):
        edges.append(
            np.stack([idx.reshape(-1), np.roll(idx, -1, axis=axis).reshape(-1)], 1)
        )
    return np.concatenate(edges, axis=0)


class IsingMCMC:
    def __init__(self, target: IsingModel, device: str | torch.device = "cpu"):
        self.target = target
        self.device = torch.device(device)
        self.shape = target.lattice_shape
        # proper colouring of the torus, shape lattice_shape: the
        # checkerboard on all-even sides, three colours when a side is odd
        # (25 is), where the checkerboard would update interacting sites
        # together -- see dig_in/utils/lattice.py
        self.color = torch.from_numpy(independent_groups(self.shape)).to(self.device)
        self.colors = list(range(int(self.color.max().item()) + 1))
        self.neighbors = lattice_neighbors(self.shape)
        self.edges = lattice_edges(self.shape)

    # ------------------------------------------------------------------
    def init_state(self, n_chains: int, mode: str = "random") -> Tensor:
        if mode == "random":
            return self.target.sample_uniform(n_chains, device=self.device)
        if mode == "ordered":
            return torch.ones(
                n_chains, self.target.D, dtype=torch.long, device=self.device
            )
        raise ValueError(mode)

    # ------------------------------------------------------------------
    # Local updates (batched)
    # ------------------------------------------------------------------
    def _local_sweep(self, x: Tensor, kind: str) -> Tensor:
        t = self.target
        s = t.spins(x)  # (B, *shape)
        for c in self.colors:
            m = t.neighbor_sum(s)
            field = t.beta * 2.0 * (t.J * m + t.h)  # 2*beta*(J m + h)
            mask = self.color == c
            if kind == "metropolis":
                # flip prob min(1, exp(-beta dE)), dE = 2 s (J m + h)
                log_acc = -s * field
                flip = torch.rand_like(s).log() < log_acc
                s = torch.where(mask & flip, -s, s)
            elif kind == "heat_bath":
                p_up = torch.sigmoid(field)
                new_s = torch.where(torch.rand_like(s) < p_up, 1.0, -1.0)
                s = torch.where(mask, new_s, s)
            else:
                raise ValueError(kind)
        return ((s.flatten(1) + 1) / 2).long()

    def metropolis_sweep(self, x: Tensor, n_sweeps: int = 1) -> Tensor:
        for _ in range(n_sweeps):
            x = self._local_sweep(x, "metropolis")
        return x

    def heat_bath_sweep(self, x: Tensor, n_sweeps: int = 1) -> Tensor:
        for _ in range(n_sweeps):
            x = self._local_sweep(x, "heat_bath")
        return x

    # ------------------------------------------------------------------
    # Cluster updates (per chain, numpy)
    # ------------------------------------------------------------------
    def wolff_update(self, x: Tensor, n_updates: int = 1) -> Tensor:
        if self.target.h != 0.0:
            raise ValueError("Wolff updates require h = 0")
        p_add = 1.0 - np.exp(-2.0 * self.target.beta * self.target.J)
        s = (2 * x.cpu().numpy() - 1).astype(np.int8)  # (B, D)
        nbr = self.neighbors
        rng = np.random.default_rng(
            torch.randint(0, 2**31 - 1, (1,)).item()
        )
        for b in range(s.shape[0]):
            for _ in range(n_updates):
                seed = rng.integers(self.target.D)
                val = s[b, seed]
                in_cluster = np.zeros(self.target.D, dtype=bool)
                in_cluster[seed] = True
                stack = [seed]
                while stack:
                    i = stack.pop()
                    for j in nbr[i]:
                        if (
                            not in_cluster[j]
                            and s[b, j] == val
                            and rng.random() < p_add
                        ):
                            in_cluster[j] = True
                            stack.append(j)
                s[b, in_cluster] *= -1
        return torch.from_numpy(((s + 1) // 2).astype(np.int64)).to(x.device)

    def swendsen_wang_update(self, x: Tensor, n_updates: int = 1) -> Tensor:
        if self.target.h != 0.0:
            raise ValueError("Swendsen-Wang updates require h = 0")
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        p_add = 1.0 - np.exp(-2.0 * self.target.beta * self.target.J)
        s = (2 * x.cpu().numpy() - 1).astype(np.int8)
        E = self.edges
        B, D = s.shape[0], self.target.D
        rng = np.random.default_rng(torch.randint(0, 2**31 - 1, (1,)).item())
        # one block-diagonal graph for the whole batch; see ghost_sw_update
        for _ in range(n_updates):
            same = s[:, E[:, 0]] == s[:, E[:, 1]]            # (B, n_edges)
            act = same & (rng.random(same.shape) < p_add)
            bi, ei = np.nonzero(act)
            rows = E[ei, 0] + bi * D
            cols = E[ei, 1] + bi * D
            graph = coo_matrix(
                (np.ones(len(rows)), (rows, cols)), shape=(B * D, B * D)
            )
            n_comp, labels = connected_components(graph, directed=False)
            flip = rng.random(n_comp) < 0.5
            s = np.where(flip[labels.reshape(B, D)], -s, s).astype(np.int8)
        return torch.from_numpy(((s + 1) // 2).astype(np.int64)).to(x.device)

    def ghost_sw_update(self, x: Tensor, n_updates: int = 1) -> Tensor:
        """Swendsen-Wang with a ghost spin: cluster updates that stay valid
        at h != 0.

        Plain Wolff/SW require zero field, so without this the reference at
        every nonzero field falls back to purely local moves -- exactly the
        regime where local dynamics are slowest, and the regime the field
        axis of the comparison is made of.  A reference that is itself
        under-equilibrated does not measure the sampler, it measures the
        reference.

        The construction is standard: an extra spin s_0 coupled to every
        site with strength h reproduces the field term, `-h sum_i s_i =
        -sum_i h s_i s_0` at s_0 = +1, and the extended system has no field,
        so SW applies to it.  After flipping clusters the gauge is restored
        by reading off sigma_i = s_i * s_0.
        """
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        beta, J, h = self.target.beta, self.target.J, self.target.h
        # h < 0 maps onto h > 0 by the global spin flip, which is an exact
        # symmetry of the energy at zero field plus a sign change of the field
        flip_in = h < 0.0
        if flip_in:
            x = 1 - x
        p_lat = 1.0 - np.exp(-2.0 * beta * J)
        p_ghost = 1.0 - np.exp(-2.0 * beta * abs(h))

        s = (2 * x.cpu().numpy() - 1).astype(np.int8)
        E = self.edges
        B, D = s.shape[0], self.target.D
        W = D + 1  # nodes per configuration: the lattice plus its ghost
        rng = np.random.default_rng(torch.randint(0, 2**31 - 1, (1,)).item())
        # One block-diagonal graph over the whole batch instead of one scipy
        # call per configuration: node b*W + i belongs to configuration b, no
        # edge ever crosses a block, so the components are per-configuration
        # by construction.  The per-configuration loop cost 15x a plain
        # Swendsen-Wang sweep, which put the 121-point reference generation
        # at ~21 hours.
        off = (np.arange(B) * W)[:, None]
        for _ in range(n_updates):
            same = s[:, E[:, 0]] == s[:, E[:, 1]]            # (B, n_edges)
            act = same & (rng.random(same.shape) < p_lat)
            bi, ei = np.nonzero(act)
            rows = E[ei, 0] + bi * W
            cols = E[ei, 1] + bi * W
            # the ghost sits at +1, so it bonds only to up sites
            gact = (s == 1) & (rng.random(s.shape) < p_ghost)
            gb, gi = np.nonzero(gact)
            rows = np.concatenate([rows, np.full(len(gb), D) + gb * W])
            cols = np.concatenate([cols, gi + gb * W])
            graph = coo_matrix(
                (np.ones(len(rows)), (rows, cols)), shape=(B * W, B * W)
            )
            n_comp, labels = connected_components(graph, directed=False)
            flip = rng.random(n_comp) < 0.5
            lab = labels.reshape(B, W)
            s = np.where(flip[lab[:, :D]], -s, s).astype(np.int8)
            # regauge each configuration so its ghost is +1 again
            regauge = flip[lab[:, D]][:, None]
            s = np.where(regauge, -s, s).astype(np.int8)
        out = torch.from_numpy(((s + 1) // 2).astype(np.int64)).to(x.device)
        return 1 - out if flip_in else out

    # ------------------------------------------------------------------
    def sample(
        self,
        n_samples: int,
        n_chains: int = 64,
        method: str = "mixed",
        burn_in: int = 500,
        thin: int = 10,
        init: str = "random",
        progress: bool = False,
    ) -> Tensor:
        """Decorrelated samples of shape (n_samples, D).

        method: 'metropolis' | 'heat_bath' | 'wolff' | 'swendsen_wang' |
        'ghost_sw' | 'mixed'.

        mixed = cluster update + one Metropolis sweep per step.  At h != 0 the
        cluster update becomes the ghost-spin variant rather than degrading to
        purely local moves: local dynamics are slowest exactly where the field
        axis of a (T, field) comparison lives, and an under-equilibrated
        reference measures itself rather than the sampler under test.
        """
        h_free = self.target.h == 0.0
        if method == "wolff" and not h_free:
            method = "ghost_sw"  # Wolff proper needs h = 0

        def cluster(x: Tensor) -> Tensor:
            return (
                self.swendsen_wang_update(x) if h_free else self.ghost_sw_update(x)
            )

        def step(x: Tensor) -> Tensor:
            if method == "metropolis":
                return self.metropolis_sweep(x)
            if method == "heat_bath":
                return self.heat_bath_sweep(x)
            if method == "wolff":
                return self.wolff_update(x)
            if method == "ghost_sw":
                return self.ghost_sw_update(x)
            if method == "swendsen_wang":
                return cluster(x)
            if method == "mixed":
                return self.metropolis_sweep(cluster(x))
            raise ValueError(method)

        x = self.init_state(n_chains, init)
        iterator = range(burn_in)
        if progress:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc="burn-in")
        for _ in iterator:
            x = step(x)

        out = []
        n_rounds = (n_samples + n_chains - 1) // n_chains
        iterator = range(n_rounds)
        if progress:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc="sampling")
        for _ in iterator:
            for _ in range(thin):
                x = step(x)
            out.append(x.clone())
        return torch.cat(out, dim=0)[:n_samples]
