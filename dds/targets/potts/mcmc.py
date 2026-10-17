"""Reference MCMC samplers for the Potts model.

Batched checkerboard heat-bath / Metropolis sweeps (torch) and cluster
updates (Wolff, Swendsen-Wang; numpy/scipy, per chain).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from dds.targets.ising.mcmc import lattice_edges, lattice_neighbors
from dds.targets.potts.target import PottsModel
from dds.utils.lattice import independent_groups


class PottsMCMC:
    def __init__(self, target: PottsModel, device: str | torch.device = "cpu"):
        self.target = target
        self.device = torch.device(device)
        self.shape = target.lattice_shape
        # proper colouring of the torus (three colours when a side is odd);
        # see dds/utils/lattice.py
        self.color = torch.from_numpy(independent_groups(self.shape)).to(self.device)
        self.colors = list(range(int(self.color.max().item()) + 1))
        self.neighbors = lattice_neighbors(self.shape)
        self.edges = lattice_edges(self.shape)

    # ------------------------------------------------------------------
    def init_state(self, n_chains: int, mode: str = "random") -> Tensor:
        if mode == "random":
            return self.target.sample_uniform(n_chains, device=self.device)
        if mode == "ordered":
            return torch.zeros(
                n_chains, self.target.D, dtype=torch.long, device=self.device
            )
        raise ValueError(mode)

    # ------------------------------------------------------------------
    # Local updates (batched)
    # ------------------------------------------------------------------
    def _local_sweep(self, x: Tensor, kind: str) -> Tensor:
        t = self.target
        q = t.q
        for c in self.colors:
            z = t.lattice(x)  # (B, *shape)
            cnt = t.neighbor_counts(x)  # (B, *shape, q)
            mask = self.color == c  # (*shape,)
            if kind == "heat_bath":
                logits = t.beta * t.J * cnt  # (B, *shape, q)
                new = torch.distributions.Categorical(logits=logits.flatten(0, -2)).sample()
                new = new.view(z.shape)
            elif kind == "metropolis":
                # propose a uniformly random *different* color
                prop = (z + torch.randint_like(z, 1, q)) % q
                cur_match = torch.gather(cnt, -1, z.unsqueeze(-1)).squeeze(-1)
                prop_match = torch.gather(cnt, -1, prop.unsqueeze(-1)).squeeze(-1)
                d_e = -t.J * (prop_match - cur_match)
                acc = torch.rand_like(d_e).log() < -t.beta * d_e
                new = torch.where(acc, prop, z)
            else:
                raise ValueError(kind)
            z = torch.where(mask, new, z)
            x = z.flatten(1)
        return x

    def heat_bath_sweep(self, x: Tensor, n_sweeps: int = 1) -> Tensor:
        for _ in range(n_sweeps):
            x = self._local_sweep(x, "heat_bath")
        return x

    def metropolis_sweep(self, x: Tensor, n_sweeps: int = 1) -> Tensor:
        for _ in range(n_sweeps):
            x = self._local_sweep(x, "metropolis")
        return x

    # ------------------------------------------------------------------
    # Cluster updates (per chain, numpy)
    # ------------------------------------------------------------------
    def wolff_update(self, x: Tensor, n_updates: int = 1) -> Tensor:
        p_add = 1.0 - np.exp(-self.target.beta * self.target.J)
        z = x.cpu().numpy().astype(np.int64)
        nbr = self.neighbors
        q = self.target.q
        rng = np.random.default_rng(torch.randint(0, 2**31 - 1, (1,)).item())
        for b in range(z.shape[0]):
            for _ in range(n_updates):
                seed = rng.integers(self.target.D)
                val = z[b, seed]
                new_val = (val + rng.integers(1, q)) % q
                in_cluster = np.zeros(self.target.D, dtype=bool)
                in_cluster[seed] = True
                stack = [seed]
                while stack:
                    i = stack.pop()
                    for j in nbr[i]:
                        if (
                            not in_cluster[j]
                            and z[b, j] == val
                            and rng.random() < p_add
                        ):
                            in_cluster[j] = True
                            stack.append(j)
                z[b, in_cluster] = new_val
        return torch.from_numpy(z).to(x.device)

    def swendsen_wang_update(self, x: Tensor, n_updates: int = 1) -> Tensor:
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components

        p_add = 1.0 - np.exp(-self.target.beta * self.target.J)
        z = x.cpu().numpy().astype(np.int64)
        E = self.edges
        D, q = self.target.D, self.target.q
        rng = np.random.default_rng(torch.randint(0, 2**31 - 1, (1,)).item())
        for b in range(z.shape[0]):
            for _ in range(n_updates):
                same = z[b, E[:, 0]] == z[b, E[:, 1]]
                active = same & (rng.random(E.shape[0]) < p_add)
                a = E[active]
                graph = coo_matrix(
                    (np.ones(len(a)), (a[:, 0], a[:, 1])), shape=(D, D)
                )
                n_comp, labels = connected_components(graph, directed=False)
                new_vals = rng.integers(0, q, size=n_comp)
                z[b] = new_vals[labels]
        return torch.from_numpy(z).to(x.device)

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

        method: 'metropolis' | 'heat_bath' | 'wolff' | 'swendsen_wang' | 'mixed'
        (mixed = Swendsen-Wang cluster update + one heat-bath sweep per step).
        """

        def step(x: Tensor) -> Tensor:
            if method == "metropolis":
                return self.metropolis_sweep(x)
            if method == "heat_bath":
                return self.heat_bath_sweep(x)
            if method == "wolff":
                return self.wolff_update(x)
            if method == "swendsen_wang":
                return self.swendsen_wang_update(x)
            if method == "mixed":
                return self.heat_bath_sweep(self.swendsen_wang_update(x))
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
