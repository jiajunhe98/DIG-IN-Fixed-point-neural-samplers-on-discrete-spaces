"""Cluster-expansion energy for a binary fcc alloy, exactly as sampled.

Self-contained: it needs only ``numpy`` and ``torch``. The cluster expansion
travels with this file as ``cluster_tables.npz``, already reduced from CLEASE's
orbit list to a flat table, so neither ``clease`` nor ``ase`` is required and
there are no imports of the repository that generated the data.

Quick start
-----------
    from potential import load_target
    t = load_target(".")                       # reads cluster_tables.npz

    import torch
    s = t.random_spins(4)                      # (B, N) in {-1, +1}
    print(t.energy(s) / t.n_sites)             # eV per atom
    print(t.flip_log_score(s))                 # (B, N) log nu(flip_i)/nu

What the model is
-----------------
One binary spin per fcc site, no displacements and no volume. The
configurational energy is a **multilinear polynomial** in the spins,

    E(s) = const + sum_f w_f prod_{i in f} s_i ,      s_i in {-1, +1},

where ``f`` runs over a few thousand site tuples (the cluster figures) and the
weights come from DFT-fitted effective cluster interactions. Two consequences
are used everywhere below: a full energy is a gather-and-reduce, and the energy
change from flipping a single site is **exact** rather than incremental,

    E(s with s_i -> -s_i) - E(s) = -2 s_i h_i(s) ,
    h_i(s) = sum_{f containing i} w_f prod_{j in f, j != i} s_j .

Conventions, all of which the data shares
-----------------------------------------
* ``s_i = +1`` is the spin-up element named in ``potential.json`` (Ag or Au),
  ``s_i = -1`` the other. ``c`` is the fraction of spin-up sites.
* The semi-grand Hamiltonian is ``H(s; h) = E(s) - h * sum_i s_i`` and the
  **physical chemical-potential difference is** ``dmu = 2h``. The reference
  notebooks of Damewood et al. scan ``h`` and label their axes ``2*chem_p``;
  both papers quote ``dmu``. A factor of two here halves the entire
  chemical-potential axis while producing perfectly plausible curves, so
  everything in this file takes ``dmu`` and converts internally.
* ``E`` is the **raw** cluster-expansion energy for AgPd and the **mixing**
  energy for CuAu -- the pure-element energies subtracted -- exactly as in the
  source each expansion came from. ``potential.json`` says which. The two
  differ by a constant plus a term linear in composition, so they describe the
  same physics but *not* the same ``dmu`` axis; do not mix them.
* Sites are indexed by their primitive-cell integer coordinates
  ``(n1, n2, n3)``, stored in ``ijk``. That re-indexing makes a periodic 3-D
  convolution a valid, translation-equivariant operation on a configuration.

**The cluster expansion is the target.** These samples are drawn from this
polynomial, not from DFT. A comparison against a different cluster expansion --
including the same ECIs read against a different orbit enumeration -- is a
comparison against a different system. See the methods document: current CLEASE
enumerates AgPd's orbits differently from the version that fitted these ECIs,
and loading them naively drops 21 of 45 terms without any error.

Useful entry points
-------------------
    t.energy(s)                 configurational energy               [eV], (B,)
    t.energy_sgc(s, dmu)        E(s) - (dmu/2) sum_i s_i             [eV], (B,)
    t.local_field(s, i)         h_i(s) at one site                   [eV], (B,)
    t.local_field_all(s)        h_i(s) at every site                 [eV], (B, N)
    t.delta_energy(s, i)        exact single-flip dE                 [eV], (B,)
    t.flip_log_score(s, dmu, T) per-site log nu(flip)/nu             (B, N)
    t.log_unnorm(s, dmu, T)     log of the unnormalised SGC density  (B,)
    t.concentration(s)          fraction of spin-up sites            (B,)
    t.order_parameters(s)       the three fcc X-point eta            (B, 3)
    t.warren_cowley(s)          first-shell short-range order        (B,)
    t.to_grid(s) / t.to_sites(x)   (B, n1, n2, n3) <-> (B, N)
    t.unpack(packed)            bit-packed stored configs -> spins

Self-test: ``python potential.py`` re-evaluates the configurations shipped in
``03_sgc_raw`` and checks them against the energies recorded with them, then
checks the closed-form limits and, where they exist, the published ordered-phase
energies.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

KB_EV_PER_K = 0.00008617  # as used throughout the generating runs


class ClusterExpansionTarget:
    """The sampled Hamiltonian, rebuilt from the packaged table."""

    def __init__(self, table_path: str, meta: Optional[dict] = None,
                 device: str = "cpu", dtype: torch.dtype = torch.float64) -> None:
        z = np.load(table_path, allow_pickle=False)
        self.device = torch.device(device)
        self.dtype = dtype
        self.system = str(z["system"])
        self.size: Tuple[int, int, int] = tuple(int(v) for v in z["size"])
        self.n_sites = int(z["n_sites"])
        self.const = float(z["const"])
        self.mixing_energy = bool(z["mixing_energy"])
        self.spin_up_element = str(z["spin_up_element"])
        self.spin_down_element = str(z["spin_down_element"])
        self.eci: Dict[str, float] = json.loads(str(z["eci_json"]))
        self.eci_sha256 = str(z["eci_sha256"])
        self.validation = json.loads(str(z["validation"]))
        self.meta = meta or {}

        fig = np.where(z["fig_sites"] < 0, self.n_sites, z["fig_sites"])
        fig = torch.as_tensor(fig, dtype=torch.long, device=self.device)
        # Gathering MAX_BODY planes of (B, M) and multiplying costs a fraction
        # of the memory of materialising (B, M, MAX_BODY) and reducing it.
        self._fig_cols = [fig[:, k].contiguous() for k in range(fig.shape[1])]
        self.fig_w = torch.as_tensor(z["fig_w"], dtype=dtype, device=self.device)
        self.n_figures = int(self.fig_w.numel())

        nb = torch.as_tensor(z["nb_idx"], dtype=torch.long, device=self.device)
        self._nb_cols = [nb[:, :, k].contiguous() for k in range(nb.shape[2])]
        self._nb_idx = nb
        self.nb_w = torch.as_tensor(z["nb_w"], dtype=dtype, device=self.device)
        # Order of the cluster behind each (site, slot) of the neighbour table.
        # The table lists, for each site, the OTHER members of every cluster
        # containing it, padded with the index `n_sites`; so the order is one
        # plus the number of real entries.  A cluster of order p appears p
        # times in the table, once per site it contains, which is what lets
        # the energy be recovered from the same products as the local field.
        self.nb_order = (nb < self.n_sites).sum(dim=2).to(dtype) + 1.0
        self._nb_w_over_order = self.nb_w / self.nb_order
        self.n_figures_per_site = int(z["nb_count"][0])
        # Keep the (B, n_figures) transient bounded: at 512 sites AgPd has 436k
        # figures, where a batch of 4096 would ask for 14 GB per plane.
        self.default_chunk = max(1, int(6e7 / max(self.n_figures, 1)))

        self.ijk = torch.as_tensor(z["ijk"], dtype=torch.long, device=self.device)
        self.cell = np.asarray(z["cell"])
        self.positions = np.asarray(z["positions"])

        n1, n2, n3 = self.size
        self._flat_of_site = (self.ijk[:, 0] * n2 + self.ijk[:, 1]) * n3 + self.ijk[:, 2]

        self._x_signs = self._build_x_signs()
        self._nn_pairs, self.nn_distance = self._build_nn_pairs()

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (f"ClusterExpansionTarget({self.system}, {self.n_sites} sites "
                f"{self.size}, {self.n_figures} figures, "
                f"{self.n_figures_per_site}/site, "
                f"{'mixing' if self.mixing_energy else 'raw'} energy, "
                f"+1={self.spin_up_element})")

    def _extend(self, s: Tensor) -> Tensor:
        ones = torch.ones((*s.shape[:-1], 1), dtype=s.dtype, device=s.device)
        return torch.cat([s.to(self.dtype), ones], dim=-1)

    # ------------------------------------------------------------------
    # Energies
    # ------------------------------------------------------------------
    def energy(self, s: Tensor, chunk: int = 0) -> Tensor:
        """Configurational energy in eV.  `s` is (..., n_sites) in +-1."""
        flat = s.reshape(-1, self.n_sites)
        chunk = chunk or self.default_chunk
        if chunk and flat.shape[0] > chunk:
            parts = [self._energy(flat[i:i + chunk])
                     for i in range(0, flat.shape[0], chunk)]
            out = torch.cat(parts)
        else:
            out = self._energy(flat)
        return out.reshape(s.shape[:-1])

    def _energy(self, flat: Tensor) -> Tensor:
        ext = self._extend(flat)
        prod = ext[:, self._fig_cols[0]]
        for col in self._fig_cols[1:]:
            prod = prod * ext[:, col]
        return self.const + prod @ self.fig_w

    def energy_sgc(self, s: Tensor, dmu: float = 0.0, chunk: int = 0) -> Tensor:
        """`H(s) = E(s) - (dmu/2) sum_i s_i` in eV."""
        h = 0.5 * torch.as_tensor(dmu, dtype=self.dtype, device=self.device)
        return self.energy(s, chunk=chunk) - h * s.to(self.dtype).sum(dim=-1)

    def local_field(self, s: Tensor, site: int) -> Tensor:
        ext = self._extend(s)
        prod = ext[:, self._nb_cols[0][site]]
        for col in self._nb_cols[1:]:
            prod = prod * ext[:, col[site]]
        return prod @ self.nb_w[site]

    def local_field_all(self, s: Tensor, chunk: int = 0) -> Tensor:
        flat = s.reshape(-1, self.n_sites)
        outs = []
        step = chunk or flat.shape[0]
        for b0 in range(0, flat.shape[0], step):
            ext = self._extend(flat[b0:b0 + step])
            prod = ext[:, self._nb_idx].prod(dim=3)
            outs.append(torch.einsum("bnk,nk->bn", prod, self.nb_w))
        return torch.cat(outs).reshape(*s.shape[:-1], self.n_sites)

    def local_field_and_energy(self, s: Tensor, chunk: int = 0):
        """Both the local fields and the energy, from one pass.

        The expensive part of either quantity is building the cluster
        products; the contractions that follow are cheap.  For a multilinear
        energy E(s) = const + sum_f w_f prod_{i in f} s_i, the neighbour table
        already holds prod_{j in f, j != i} s_j for every cluster f and every
        site i it contains.  Multiplying by s_i completes the cluster product,
        and since a cluster of order p appears p times in the table, dividing
        its weight by p counts it exactly once:

            E - const = sum_i s_i sum_{f ∋ i} (w_f / |f|) prod_{j in f, j!=i} s_j

        Note the energy is NOT recoverable from the local fields themselves:
        sum_i s_i f_i is the order-weighted energy, and on the Ag-Pd expansion
        (four cluster orders) that ratio ranges over -4.7 to 15.6 across
        configurations.  It is the shared *products*, not the shared output,
        that make this free.
        """
        flat = s.reshape(-1, self.n_sites)
        fs, es = [], []
        step = chunk or flat.shape[0]
        for b0 in range(0, flat.shape[0], step):
            sb = flat[b0:b0 + step]
            ext = self._extend(sb)
            prod = ext[:, self._nb_idx].prod(dim=3)
            fs.append(torch.einsum("bnk,nk->bn", prod, self.nb_w))
            per_site = torch.einsum("bnk,nk->bn", prod, self._nb_w_over_order)
            es.append((sb.to(self.dtype) * per_site).sum(dim=-1))
        f = torch.cat(fs).reshape(*s.shape[:-1], self.n_sites)
        e = torch.cat(es).reshape(s.shape[:-1]) + self.const
        return f, e

    def delta_energy(self, s: Tensor, site: int) -> Tensor:
        """Exact change in E if `site` flips."""
        return -2.0 * s[:, site].to(self.dtype) * self.local_field(s, site)

    def flip_log_score(self, s: Tensor, dmu: float = 0.0,
                       temperature: float = 600.0, chunk: int = 0) -> Tensor:
        """`log nu(flip_i s) / nu(s)` for every site -> (..., n_sites)."""
        beta = 1.0 / (KB_EV_PER_K * torch.as_tensor(
            temperature, dtype=self.dtype, device=self.device))
        h = 0.5 * torch.as_tensor(dmu, dtype=self.dtype, device=self.device)
        f = self.local_field_all(s, chunk=chunk)
        b = beta.reshape(-1, 1) if beta.ndim else beta
        hh = h.reshape(-1, 1) if h.ndim else h
        return 2.0 * b * s.to(self.dtype) * (f - hh)

    def log_unnorm(self, s: Tensor, dmu: float = 0.0,
                   temperature: float = 600.0, chunk: int = 0) -> Tensor:
        beta = 1.0 / (KB_EV_PER_K * torch.as_tensor(
            temperature, dtype=self.dtype, device=self.device))
        return -beta * self.energy_sgc(s, dmu=dmu, chunk=chunk)

    # ------------------------------------------------------------------
    # Configurations and observables
    # ------------------------------------------------------------------
    def random_spins(self, batch: int, c_up: float = 0.5,
                     generator: Optional[torch.Generator] = None) -> Tensor:
        u = torch.rand((batch, self.n_sites), generator=generator,
                       device=self.device, dtype=self.dtype)
        return torch.where(u < c_up, 1.0, -1.0).to(self.dtype)

    def concentration(self, s: Tensor) -> Tensor:
        return (s > 0).to(self.dtype).mean(dim=-1)

    def to_grid(self, s: Tensor) -> Tensor:
        """(..., n_sites) -> (..., n1, n2, n3), the periodic-convolution layout."""
        out = torch.empty_like(s.reshape(-1, self.n_sites))
        out[:, self._flat_of_site] = s.reshape(-1, self.n_sites)
        return out.reshape(*s.shape[:-1], *self.size)

    def to_sites(self, x: Tensor) -> Tensor:
        """(..., n1, n2, n3) -> (..., n_sites), the cluster-table layout."""
        flat = x.reshape(-1, self.n_sites)
        return flat[:, self._flat_of_site].reshape(*x.shape[:-3], self.n_sites)

    def _build_x_signs(self) -> Optional[Tensor]:
        """The three fcc X-point sign patterns, if the cell can host them.

        `k_X = (2 pi / a)(1,0,0)` has `k . a1 = 0`, `k . a2 = k . a3 = pi` for
        the primitive vectors, so the phase factors collapse to signs. L1_0 is
        single-k (one large eta), L1_2 is triple-k (three equal). Undefined for
        an odd cell, where the wavevector is incommensurate -- AgPd's 5x5x5.
        """
        if any(v % 2 for v in self.size):
            return None
        p = torch.stack([(self.ijk[:, 1] + self.ijk[:, 2]) % 2,
                         (self.ijk[:, 0] + self.ijk[:, 2]) % 2,
                         (self.ijk[:, 0] + self.ijk[:, 1]) % 2])
        return (1 - 2 * p).to(self.dtype)

    def order_parameters(self, s: Tensor) -> Optional[Tensor]:
        """`eta` at the three fcc X points: (..., 3), or None for an odd cell."""
        if self._x_signs is None:
            return None
        return torch.einsum("...i,ki->...k", s.to(self.dtype),
                            self._x_signs) / self.n_sites

    def _build_nn_pairs(self):
        pos, cell = self.positions, self.cell
        inv = np.linalg.inv(cell)
        d = pos[:, None, :] - pos[None, :, :]
        frac = d @ inv
        frac -= np.rint(frac)
        dist = np.linalg.norm(frac @ cell, axis=-1)
        iu = np.triu_indices(len(pos), k=1)
        dd = dist[iu]
        target = np.unique(np.round(dd, 6))[0]
        m = np.abs(dd - target) < 1e-6
        pairs = np.stack([iu[0][m], iu[1][m]], axis=1)
        return torch.as_tensor(pairs, dtype=torch.long, device=self.device), float(target)

    def warren_cowley(self, s: Tensor) -> Tensor:
        """First-shell Warren-Cowley parameter; >0 clustering, <0 ordering."""
        f = s.to(self.dtype)
        m = f.mean(dim=-1)
        pc = (f[..., self._nn_pairs[:, 0]] * f[..., self._nn_pairs[:, 1]]).mean(dim=-1)
        den = 1.0 - m ** 2
        return torch.where(den > 1e-12, (pc - m ** 2) / den.clamp(min=1e-12),
                           torch.zeros_like(pc))

    # ------------------------------------------------------------------
    def unpack(self, packed: np.ndarray) -> Tensor:
        """Bit-packed stored configurations -> `(..., n_sites)` spins in +-1.

        Bit 1 means spin +1 (`spin_up_element`). The inverse of
        ``np.packbits((s > 0).astype(np.uint8), axis=-1)``.
        """
        bits = np.unpackbits(np.asarray(packed, dtype=np.uint8),
                             axis=-1)[..., :self.n_sites]
        arr = 2 * bits.astype(np.int8) - 1
        return torch.as_tensor(arr, dtype=self.dtype, device=self.device)


def load_target(directory: str = ".", device: str = "cpu",
                dtype: torch.dtype = torch.float64) -> ClusterExpansionTarget:
    """Rebuild the sampling target from the packaged metadata.

    Everything that defines the Hamiltonian is read from the package rather
    than passed in, so the object cannot be built with settings the data was
    not generated under.
    """
    meta_path = os.path.join(directory, "potential.json")
    meta = json.load(open(meta_path, encoding="utf-8"))
    return ClusterExpansionTarget(os.path.join(directory, meta["npz"]),
                                  meta=meta, device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
def _self_test(directory: str = ".") -> int:
    """Check this file reproduces the Hamiltonian the shipped data was drawn from.

    Four checks, in increasing order of how much they would have to conspire:

    1. the closed-form limits -- pure cells and single substitutions, where the
       energy is fixed by symmetry rather than by any sampling;
    2. the exact single-flip identity, against a brute-force recomputation;
    3. **every stored configuration's recorded energy**, re-evaluated here.
       This is the one that matters: it runs on the consumer's machine, torch
       and hardware, so agreement means this file is the same Hamiltonian, not
       merely a plausible one;
    4. for CuAu, the ordered-phase energies published by Damewood et al.
    """
    import glob
    t = load_target(directory)
    print(t)
    print(f"  ECI sha256 {t.eci_sha256[:16]}, agreement with CLEASE at "
          f"packaging time {t.validation['max_abs_energy_error_eV']:.1e} eV "
          f"over {t.validation['n_configs']} configurations")
    ok = True

    # 1. closed-form limits -------------------------------------------------
    up = torch.ones((1, t.n_sites), dtype=t.dtype)
    down = -torch.ones((1, t.n_sites), dtype=t.dtype)
    e_up, e_down = float(t.energy(up)[0]), float(t.energy(down)[0])
    print(f"  pure {t.spin_up_element}: {e_up:+.9f} eV   "
          f"pure {t.spin_down_element}: {e_down:+.9f} eV")
    if t.mixing_energy and (abs(e_up) > 1e-9 or abs(e_down) > 1e-9):
        print("  FAIL: a mixing energy must vanish in both pure limits")
        ok = False
    singles = -torch.ones((t.n_sites, t.n_sites), dtype=t.dtype)
    singles[torch.arange(t.n_sites), torch.arange(t.n_sites)] = 1.0
    es = t.energy(singles, chunk=256)
    spread = float(es.max() - es.min())
    print(f"  {t.n_sites} single substitutions degenerate to {spread:.2e} eV")
    if spread > 1e-9:
        print("  FAIL: single substitutions are not degenerate")
        ok = False

    # 2. the exact single-flip identity ------------------------------------
    g = torch.Generator(); g.manual_seed(0)
    s = t.random_spins(8, c_up=0.4, generator=g)
    e0 = t.energy(s)
    worst = 0.0
    for i in (0, 1, t.n_sites // 3, t.n_sites - 1):
        s2 = s.clone(); s2[:, i] *= -1
        worst = max(worst, float((t.delta_energy(s, i) - (t.energy(s2) - e0)).abs().max()))
    print(f"  single-flip dE vs brute force: max |err| {worst:.2e} eV")
    if worst > 1e-9:
        print("  FAIL: the single-flip identity does not hold")
        ok = False

    # 3. the stored configurations -----------------------------------------
    pats = [os.path.join(directory, "..", "02_sgc_merged", "*snapshot_energies.npz"),
            os.path.join(directory, "*snapshot_energies.npz")]
    ener = next((f for pat in pats for f in sorted(glob.glob(pat))), None)
    snap = next((f for f in sorted(glob.glob(os.path.join(
        directory, "..", "02_sgc_merged", "*direct_mc.npz")))), None)
    if ener and snap:
        ez = np.load(ener, allow_pickle=False)
        sz = np.load(snap, allow_pickle=False)
        packed, rec = sz["snapshots"], ez["energy"]
        flat = packed.reshape(-1, packed.shape[-1])
        rflat = rec.reshape(-1)
        rng = np.random.default_rng(0)
        pick = rng.choice(len(rflat), size=min(4096, len(rflat)), replace=False)
        got = t.energy(t.unpack(flat[pick]), chunk=256).cpu().numpy()
        err = float(np.abs(got - rflat[pick]).max())
        cerr = float(np.abs(t.concentration(t.unpack(flat[pick])).cpu().numpy()
                            - ez["concentration"].reshape(-1)[pick]).max())
        print(f"  {len(pick)} of {len(rflat):,} stored configurations "
              f"re-evaluated: max |dE| {err:.2e} eV, max |dc| {cerr:.2e}")
        if err > 1e-9 or cerr > 1e-12:
            print("  FAIL: recorded energies do not reproduce")
            ok = False
    else:
        print("  (no stored configurations found next to this package)")

    # 4. the published ordered-phase energies -------------------------------
    hull = t.meta.get("published_ground_states")
    if hull:
        ijk = t.ijk.cpu().numpy()
        r = np.stack([ijk[:, 1] + ijk[:, 2], ijk[:, 0] + ijk[:, 2],
                      ijk[:, 0] + ijk[:, 1]], axis=1)
        corner = (r[:, 0] % 2 == 0) & (r[:, 1] % 2 == 0) & (r[:, 2] % 2 == 0)
        decor = {"L1_2 Cu3Au": np.where(corner, 1, -1),
                 "L1_0 CuAu": np.where(r[:, 0] % 2 == 0, 1, -1),
                 "L1_2 CuAu3": np.where(corner, -1, 1)}
        worst = 0.0
        for name, want in hull.items():
            sp = torch.as_tensor(decor[name], dtype=t.dtype)[None, :]
            got = float(t.energy(sp)[0]) / t.n_sites
            worst = max(worst, abs(got - want))
            print(f"  {name:12s} c={float(t.concentration(sp)[0]):.2f}  "
                  f"{got:+.12f} eV/atom  (published {want:+.12f})")
        if worst > 1e-10:
            print("  FAIL: published ordered-phase energies do not reproduce")
            ok = False

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(_self_test(os.path.dirname(os.path.abspath(__file__))))
