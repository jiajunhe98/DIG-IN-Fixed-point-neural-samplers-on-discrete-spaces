"""Binary alloy cluster expansions as an amortized family over (T, delta_mu).

The target is the DFT-fitted cluster expansion of Damewood et al. (npj Comput.
Mater. 8, 61, 2022) on an fcc supercell, in the semi-grand-canonical ensemble

    H(s) = E(s) - (delta_mu / 2) * sum_i s_i,      s_i in {-1, +1},

with `E` the raw cluster-expansion energy and `delta_mu = mu_up - mu_down` in
eV.  The factor of two is the packaged convention and is not cosmetic: the
source notebooks scan an internal field and label their axes `2 * chem_p`,
so dropping it halves the chemical-potential axis and still produces
plausible curves.

The family conditions on `(T, delta_mu)` in physical units (kelvin, eV).
Two things about it are deliberate:

* the network sees *normalised* conditioning, both variables mapped to
  [0, 1] over their own box.  beta = 1 / (k_B T) runs from about 12 to 58
  per eV across the packaged temperature sweep while delta_mu spans 0.8 eV,
  and feeding raw values of such different magnitude to one conditioning MLP
  wastes its resolution on the larger.  `cond_ranges` therefore reports the
  normalised box and `to_cond_features` does the mapping.
* the exact log Z is available on the whole plane.  The packaged reference
  gives ln Z_n(T) at fixed composition, pinned by the closed form at n = 0,
  so the semi-grand partition function follows analytically,

      ln Z(T, dmu) = logsumexp_n [ ln Z_n(T) + beta (dmu/2) (2n - N) ],

  which is what `exact_log_Z` returns.  There is no finite-lattice closed
  form to fall back on here, so outside the packaged temperature sweep it
  returns None rather than extrapolating.
"""

from __future__ import annotations

import json
import os

import numpy as np
import torch
from torch import Tensor

from dds.targets.alloy.potential import KB_EV_PER_K, load_target
from dds.targets.base import DiscreteTarget

DATA_ROOT = os.path.join(os.path.dirname(__file__), "data")

#: packaged systems -> directory under `data/`
SYSTEMS = {
    "agpd125": "agpd_N125",
    "cuau128": "cuau_N128",
}


class AlloyFamily(DiscreteTarget):
    """Cluster-expansion alloy, amortized over the (T, delta_mu) plane.

    `cond` is always (B, 2) holding (T [K], delta_mu [eV]) in physical units.
    `cond=None` falls back to the reference state point, so any code path
    with no conditioning to offer still works.
    """

    amortized = True
    cond_dim = 2
    #: selects the alloy grid evaluator in dds.train
    grid_kind = "alloy"

    def __init__(
        self,
        system: str = "agpd125",
        T_range: tuple[float, float] = (200.0, 900.0),
        dmu_range: tuple[float, float] = (-0.4, 0.4),
        T_ref: float | None = None,
        dmu_ref: float = 0.0,
        T_sampling: str = "inverse",
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        if system not in SYSTEMS:
            raise ValueError(f"unknown alloy system {system!r}; have {list(SYSTEMS)}")
        self.system = system
        self.data_dir = os.path.join(DATA_ROOT, SYSTEMS[system])
        self.ce = load_target(self.data_dir, device=device, dtype=torch.float64)
        self.lattice_shape = tuple(self.ce.size)
        self._dtype = dtype
        # site order of the cluster tables, indexed by the row-major grid order
        # the lattice-shaped backbones assume
        self._site_of_flat = self.ce._flat_of_site
        super().__init__(dim=self.ce.n_sites, num_states=2, beta=1.0)

        self.T_range = tuple(float(v) for v in T_range)
        self.dmu_range = tuple(float(v) for v in dmu_range)
        self.T_sampling = T_sampling
        self.T_ref = float(T_ref) if T_ref is not None else float(np.mean(self.T_range))
        self.dmu_ref = float(dmu_ref)
        self.device = torch.device(device)
        self._ref = self._load_reference()

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (f"AlloyFamily({self.system}, {self.D} sites {self.lattice_shape}, "
                f"T {self.T_range} K, dmu {self.dmu_range} eV)")

    def _load_reference(self) -> dict:
        out = {}
        p = os.path.join(self.data_dir, "lnZ.npz")
        if os.path.exists(p):
            z = np.load(p)
            out["lnZ_T"] = np.asarray(z["T"], dtype=np.float64)
            out["lnZ"] = np.asarray(z["lnZ"], dtype=np.float64)
            out["lnZ_valid"] = np.asarray(z["valid"], dtype=bool)
            out["n"] = np.asarray(z["n"], dtype=np.float64)
        p = os.path.join(self.data_dir, "state_points.npz")
        if os.path.exists(p):
            sp = np.load(p)
            out["sp"] = {k: np.asarray(sp[k]) for k in sp.files}
        p = os.path.join(self.data_dir, "phase_diagram.json")
        if os.path.exists(p):
            with open(p) as fh:
                out["phase_diagram"] = json.load(fh)
        return out

    # ------------------------------------------------------------------
    # Conditioning
    # ------------------------------------------------------------------
    @property
    def beta_range(self) -> tuple[float, float]:
        lo = 1.0 / (KB_EV_PER_K * self.T_range[1])
        hi = 1.0 / (KB_EV_PER_K * self.T_range[0])
        return (lo, hi)

    @property
    def cond_ranges(self) -> list[tuple[float, float]]:
        """What the NETWORK sees: both variables normalised to [0, 1]."""
        return [(0.0, 1.0), (0.0, 1.0)]

    def to_cond_features(self, cond: Tensor | None) -> Tensor | None:
        """(T, delta_mu) -> (beta_hat, dmu_hat), both in [0, 1] over the box."""
        if cond is None:
            return None
        cond = cond.view(-1, 2)
        b_lo, b_hi = self.beta_range
        beta = 1.0 / (KB_EV_PER_K * cond[:, 0])
        d_lo, d_hi = self.dmu_range
        return torch.stack([
            (beta - b_lo) / (b_hi - b_lo),
            (cond[:, 1] - d_lo) / (d_hi - d_lo),
        ], dim=-1)

    def _split(self, cond: Tensor | None):
        if cond is None:
            return None, None
        cond = cond.view(-1, 2)
        return cond[:, 0], cond[:, 1]

    def _beta_of(self, cond: Tensor | None):
        T, _ = self._split(cond)
        if T is None:
            return 1.0 / (KB_EV_PER_K * self.T_ref)
        return 1.0 / (KB_EV_PER_K * T)

    def _dmu_of(self, cond: Tensor | None):
        _, dmu = self._split(cond)
        return self.dmu_ref if dmu is None else dmu

    def sample_cond(self, n: int, device=None, generator=None,
                    T_lo: float | None = None) -> Tensor:
        """n conditioning draws, continuous over the box.

        Temperature is drawn uniformly in beta by default: the log-density is
        linear in beta, so that spreads score magnitudes evenly where a
        uniform draw in T crowds them at the hot end.  `T_lo` raises the cold
        end, which is how the curriculum restricts training to the
        high-temperature part of the plane early on.
        """
        u = torch.rand(n, 2, device=device, generator=generator)
        t0 = self.T_range[0] if T_lo is None else max(T_lo, self.T_range[0])
        t1 = self.T_range[1]
        if self.T_sampling == "inverse":
            b0, b1 = 1.0 / t1, 1.0 / t0
            T = 1.0 / (b0 + u[:, 0] * (b1 - b0))
        else:
            T = t0 + u[:, 0] * (t1 - t0)
        d0, d1 = self.dmu_range
        dmu = d0 + u[:, 1] * (d1 - d0)
        return torch.stack([T, dmu], dim=-1)

    # -- acquisition coordinates -------------------------------------
    # The Fisher map lives in the coordinates the conditioning is *sampled*
    # in, which is (beta, delta_mu) rather than (T, delta_mu): binning a
    # uniform-in-beta draw on a uniform-in-T grid would leave the hot cells
    # almost empty and the acquisition would read that as difficulty.
    @property
    def acq_ranges(self) -> list[tuple[float, float]]:
        return [self.beta_range, self.dmu_range]

    def cond_to_acq(self, cond: Tensor) -> Tensor:
        T, dmu = self._split(cond)
        return torch.stack([1.0 / (KB_EV_PER_K * T), dmu], dim=-1)

    def acq_to_cond(self, a: Tensor) -> Tensor:
        a = a.view(-1, 2)
        return torch.stack([1.0 / (KB_EV_PER_K * a[:, 0]), a[:, 1]], dim=-1)

    def cond_score(self, x: Tensor, cond: Tensor) -> Tensor:
        """d U / d theta in the normalised coordinates of the sampling box.

        U(s; beta, dmu) = beta (E(s) - (dmu/2) M(s)), so the two channels are
        the semi-grand energy and the total spin -- the sufficient statistics
        of the family, which is the point of the observable-free acquisition.
        Each is scaled by its own box width so the trace does not depend on
        whether temperature is measured in kelvin or its inverse.
        """
        T, dmu = self._split(cond)
        beta = 1.0 / (KB_EV_PER_K * T)
        s = self.spins(x)
        M = s.sum(dim=-1).to(torch.float32)
        H = self.energy(x, cond).to(torch.float32)
        b_lo, b_hi = self.beta_range
        d_lo, d_hi = self.dmu_range
        return torch.stack([
            (b_hi - b_lo) * H,
            (d_hi - d_lo) * (-0.5 * beta.to(torch.float32) * M),
        ], dim=-1)

    def cond_grid(self, n_T: int = 11, n_dmu: int = 11, device=None,
                  T_range: tuple[float, float] | None = None,
                  dmu_range: tuple[float, float] | None = None) -> Tensor:
        Ts = torch.linspace(*(T_range or self.T_range), n_T, device=device)
        mus = torch.linspace(*(dmu_range or self.dmu_range), n_dmu, device=device)
        gT, gmu = torch.meshgrid(Ts, mus, indexing="ij")
        return torch.stack([gT.reshape(-1), gmu.reshape(-1)], dim=-1)

    def reference_grid(self, device=None) -> Tensor | None:
        """The packaged (T, delta_mu) sweep, so the final evaluation lands on
        the exact points the reference data was generated at."""
        sp = self._ref.get("sp")
        if sp is None:
            return None
        T = np.unique(sp["T"])
        dmu = np.unique(sp["dmu"])
        T = T[(T >= self.T_range[0] - 1e-9) & (T <= self.T_range[1] + 1e-9)]
        dmu = dmu[(dmu >= self.dmu_range[0] - 1e-9) & (dmu <= self.dmu_range[1] + 1e-9)]
        gT, gmu = torch.meshgrid(
            torch.as_tensor(T, dtype=torch.float32, device=device),
            torch.as_tensor(dmu, dtype=torch.float32, device=device),
            indexing="ij",
        )
        return torch.stack([gT.reshape(-1), gmu.reshape(-1)], dim=-1)

    def at(self, T: float, dmu: float = 0.0) -> "AlloyPoint":
        """The single-state member of the family at (T, delta_mu)."""
        return AlloyPoint(self, float(T), float(dmu))

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    def _ensure_device(self, device) -> None:
        """Migrate the cluster tables to wherever the states live.

        The target is built from the config before the run knows its device,
        and the packaged tables are large enough that they are loaded once and
        indexed rather than rebuilt.  The states then arrive on the GPU while
        the tables sit on the CPU, which fails inside the first einsum with a
        device mismatch rather than anywhere informative.  Everything tensor-
        shaped on the expansion object is moved together, once, the first time
        a state from a new device shows up.
        """
        device = torch.device(device)
        if device == self.ce.device:
            return
        ce = self.ce
        for k, v in list(ce.__dict__.items()):
            if torch.is_tensor(v):
                setattr(ce, k, v.to(device))
            elif isinstance(v, list) and v and all(torch.is_tensor(t) for t in v):
                setattr(ce, k, [t.to(device) for t in v])
            elif isinstance(v, tuple) and v and all(torch.is_tensor(t) for t in v):
                setattr(ce, k, tuple(t.to(device) for t in v))
        ce.device = device
        self._site_of_flat = ce._flat_of_site
        self.device = device

    # ------------------------------------------------------------------
    # Spins <-> tokens
    # ------------------------------------------------------------------
    def spins(self, x: Tensor) -> Tensor:
        """Tokens {0, 1} in grid order -> spins {-1, +1} in cluster-table order."""
        self._ensure_device(x.device)
        s = (2 * x.to(torch.float64) - 1.0)
        return s[..., self._site_of_flat]

    def tokens(self, s: Tensor) -> Tensor:
        """Spins in cluster-table order -> tokens {0, 1} in grid order."""
        out = torch.empty_like(s)
        out[..., self._site_of_flat] = s
        return (out > 0).long()

    # ------------------------------------------------------------------
    # Energy
    # ------------------------------------------------------------------
    def energy(self, x: Tensor, cond: Tensor | None = None) -> Tensor:
        """H(s) = E(s) - (delta_mu / 2) sum_i s_i, in eV."""
        s = self.spins(x)
        dmu = self._dmu_of(cond)
        e = self.ce.energy(s, chunk=self.ce.default_chunk)
        if not torch.is_tensor(dmu):
            dmu = torch.as_tensor(dmu, dtype=e.dtype, device=e.device)
        h = 0.5 * dmu.to(e.dtype).reshape(-1) if dmu.ndim else 0.5 * dmu
        return (e - h * s.sum(dim=-1)).to(self._dtype)

    def energy_diff(self, x: Tensor, cond: Tensor | None = None) -> Tensor:
        """H(x^{d<-n}) - H(x), shape (B, D, 2).

        Only two states, so one column is zero and the other is the flip:
        Delta H = -2 s_d f_d + delta_mu s_d, with f the local field the
        cluster tables give for every site at once.
        """
        s = self.spins(x)
        dmu = self._dmu_of(cond)
        f = self.ce.local_field_all(s, chunk=self.ce.default_chunk)
        if not torch.is_tensor(dmu):
            dmu = torch.as_tensor(dmu, dtype=f.dtype, device=f.device)
        d = dmu.to(f.dtype).reshape(-1, 1) if dmu.ndim else dmu
        flip = (-2.0 * s * f + d * s)              # (B, D) in site order
        out = torch.empty_like(flip)
        out[..., self._site_of_flat] = flip        # back to grid order
        cur = x.to(out.dtype)                      # 1 where the token is 1
        stacked = torch.stack([out, out], dim=-1)  # (B, D, 2)
        # column n holds 0 where n is the current token and the flip otherwise
        keep = torch.stack([1.0 - cur, cur], dim=-1)
        return (stacked * (1.0 - keep)).to(self._dtype)

    def energy_and_diff(self, x: Tensor, cond: Tensor | None = None):
        """`(H(x), H(x^{d<-n}) - H(x))` from one pass over the cluster tables.

        A weighted training step needs both: the flip table for the heat-bath
        label and the configuration's own energy for the importance weight.
        Computed separately that is two passes, and the pass -- building the
        cluster products -- is the whole cost; the contractions after it are
        cheap.  `local_field_and_energy` returns both from one set of
        products, so the weight becomes free in target passes rather than
        costing fifty percent more.

        Metered as one `energy_diff` and no `energy`, because `n_calls` is a
        proxy for passes over the target and this is one pass.  A run using
        this path is therefore not comparable call-for-call with one that
        called the two primitives separately -- it does the same arithmetic
        for two thirds of the count.
        """
        counts = getattr(self, "n_calls", None)
        if counts is not None:
            counts["energy_diff"] += int(x.shape[0])
        s = self.spins(x)
        dmu = self._dmu_of(cond)
        f, e = self.ce.local_field_and_energy(s, chunk=self.ce.default_chunk)
        if not torch.is_tensor(dmu):
            dmu = torch.as_tensor(dmu, dtype=f.dtype, device=f.device)
        d = dmu.to(f.dtype).reshape(-1, 1) if dmu.ndim else dmu
        h = 0.5 * dmu.to(e.dtype).reshape(-1) if dmu.ndim else 0.5 * dmu
        energy = (e - h * s.sum(dim=-1)).to(self._dtype)

        flip = (-2.0 * s * f + d * s)
        out = torch.empty_like(flip)
        out[..., self._site_of_flat] = flip
        cur = x.to(out.dtype)
        stacked = torch.stack([out, out], dim=-1)
        keep = torch.stack([1.0 - cur, cur], dim=-1)
        return energy, (stacked * (1.0 - keep)).to(self._dtype)

    # ------------------------------------------------------------------
    # Exact reference
    # ------------------------------------------------------------------
    def exact_log_Z(self, cond: Tensor | None = None) -> float | None:
        """ln Z(T, delta_mu) from the packaged fixed-composition ln Z_n(T).

        Returns None outside the packaged temperature sweep rather than
        extrapolating: there is no closed form to fall back on here.
        """
        if "lnZ" not in self._ref:
            return None
        T, dmu = self._split(cond)
        if T is None:
            T_val, dmu_val = self.T_ref, self.dmu_ref
        else:
            T_val = float(T.reshape(-1)[0])
            dmu_val = float(dmu.reshape(-1)[0])
        grid_T = self._ref["lnZ_T"]
        if T_val < grid_T.min() - 1e-6 or T_val > grid_T.max() + 1e-6:
            return None
        lnZ_n = self._interp_lnZ(T_val)
        if lnZ_n is None:
            return None
        beta = 1.0 / (KB_EV_PER_K * T_val)
        n = self._ref["n"]
        tilt = beta * 0.5 * dmu_val * (2.0 * n - self.D)
        a = lnZ_n + tilt
        m = np.max(a)
        return float(m + np.log(np.sum(np.exp(a - m))))

    def exact_grid_rows(self) -> list[dict] | None:
        """Exact <c> and dc/d(delta_mu) on the packaged sweep, from ln Z_n(T).

        The walker averages in `state_points.npz` are the wrong reference for
        a phase boundary.  They agree with the exact value to 1e-4 over the
        AgPd sweep, but they are wrong by up to 0.34 at the handful of points
        flagged `trapped` -- and those points lie on the transition line,
        which is precisely what the boundary is built from.  The package's own
        methods note says as much: across a first-order line the walkers sit in
        two phases and their average is a composition no configuration visits.

        Everything here is closed form.  Tilting the fixed-composition ln Z_n
        by beta (delta_mu / 2) (2n - N) gives the semi-grand distribution over
        n, so <c> = <n>/N and, differentiating the tilt,

            dc/d(delta_mu) = beta Var(n) / N,

        which is the susceptibility the packaged boundary peaks on (up to a
        constant factor of 2 in their reporting convention, which does not
        move the maximum).
        """
        sp = self._ref.get("sp")
        if sp is None or "lnZ" not in self._ref:
            return None
        n = self._ref["n"]
        # the same intersection `reference_grid` takes: a state point outside
        # the training box is one the model was never asked about, so it does
        # not belong in a comparison against the model
        T_all = np.unique(sp["T"])
        dmu_all = np.unique(sp["dmu"])
        T_all = T_all[(T_all >= self.T_range[0] - 1e-9)
                      & (T_all <= self.T_range[1] + 1e-9)]
        dmu_all = dmu_all[(dmu_all >= self.dmu_range[0] - 1e-9)
                          & (dmu_all <= self.dmu_range[1] + 1e-9)]
        rows: list[dict] = []
        for T_val in T_all:
            lnZ_n = self._interp_lnZ(float(T_val))
            if lnZ_n is None:
                continue
            beta = 1.0 / (KB_EV_PER_K * float(T_val))
            for dmu_val in dmu_all:
                a = lnZ_n + beta * 0.5 * float(dmu_val) * (2.0 * n - self.D)
                a = a - np.max(a)
                p = np.exp(a)
                p /= p.sum()
                m1 = float((p * n).sum())
                var = float((p * n * n).sum() - m1 * m1)
                rows.append({
                    "T": float(T_val),
                    "dmu": float(dmu_val),
                    "c": m1 / self.D,
                    "c_rw": m1 / self.D,
                    "dc_ddmu": beta * var / self.D,
                })
        return rows

    def _interp_lnZ(self, T_val: float) -> np.ndarray | None:
        """ln Z_n at T, linearly interpolated between packaged isotherms."""
        grid_T, lnZ, valid = self._ref["lnZ_T"], self._ref["lnZ"], self._ref["lnZ_valid"]
        j = int(np.searchsorted(grid_T, T_val))
        if j <= 0:
            row, ok = lnZ[0], valid[0]
        elif j >= len(grid_T):
            row, ok = lnZ[-1], valid[-1]
        else:
            w = (T_val - grid_T[j - 1]) / (grid_T[j] - grid_T[j - 1])
            row = (1.0 - w) * lnZ[j - 1] + w * lnZ[j]
            ok = valid[j - 1] & valid[j]
        if not ok.any():
            return None
        out = np.where(ok, row, -np.inf)
        return out

    def reference_state_point(self, T: float, dmu: float) -> dict | None:
        """Packaged plain-MC averages at the nearest sampled (T, delta_mu).

        The arrays are indexed (dmu, T, walker) and are kept per walker on
        purpose: `trapped` marks the points where the walkers never agreed,
        which is exactly where a learned sampler is supposed to win, so the
        flag travels with the value rather than being averaged away.
        """
        sp = self._ref.get("sp")
        if sp is None:
            return None
        i = int(np.argmin(np.abs(np.asarray(sp["dmu"]) - dmu)))
        j = int(np.argmin(np.abs(np.asarray(sp["T"]) - T)))
        return {
            "c": float(np.mean(np.asarray(sp["c"])[i, j])),
            "c_spread": float(np.asarray(sp["c_spread"])[i, j]),
            "E": float(np.mean(np.asarray(sp["E"])[i, j])),
            "warren_cowley": float(np.mean(np.asarray(sp["warren_cowley"])[i, j])),
            "trapped": bool(np.asarray(sp["trapped"])[i, j]),
            "T": float(sp["T"][j]),
            "dmu": float(sp["dmu"][i]),
        }

    # ------------------------------------------------------------------
    # Observables
    # ------------------------------------------------------------------
    def concentration(self, x: Tensor) -> Tensor:
        """Fraction of spin-up sites, the composition c."""
        return x.to(torch.float32).mean(dim=-1)

    def warren_cowley(self, x: Tensor) -> Tensor:
        return self.ce.warren_cowley(self.spins(x)).to(self._dtype)

    def order_parameters(self, x: Tensor) -> Tensor | None:
        eta = self.ce.order_parameters(self.spins(x))
        return None if eta is None else eta.to(self._dtype)

    def sample_uniform(self, n: int, device=None, generator=None) -> Tensor:
        return torch.randint(0, 2, (n, self.D),
                             device=device or self.device, generator=generator)


class AlloyPoint(DiscreteTarget):
    """One member of an `AlloyFamily`, at fixed (T, delta_mu).

    The single-target stack (reference MCMC, evaluation, log Z) operates on
    these, so an amortized run is scored with the same machinery as a
    non-amortized one.
    """

    def __init__(self, family: AlloyFamily, T: float, dmu: float):
        self.family = family
        self.T = float(T)
        self.dmu = float(dmu)
        self.lattice_shape = family.lattice_shape
        super().__init__(dim=family.D, num_states=2,
                         beta=1.0 / (KB_EV_PER_K * self.T))
        self.device = family.device
        self._cond = torch.tensor([[self.T, self.dmu]], dtype=torch.float32)

    def __repr__(self) -> str:
        return f"AlloyPoint({self.family.system}, T={self.T} K, dmu={self.dmu} eV)"

    def energy(self, x: Tensor, cond: Tensor | None = None) -> Tensor:
        c = self._cond.to(x.device).expand(x.shape[0], 2)
        return self.family.energy(x, c) * 0 + self._raw_energy(x)

    def _raw_energy(self, x: Tensor) -> Tensor:
        s = self.family.spins(x)
        e = self.family.ce.energy(s, chunk=self.family.ce.default_chunk)
        return (e - 0.5 * self.dmu * s.sum(dim=-1)).to(self.family._dtype)

    def energy_diff(self, x: Tensor, cond: Tensor | None = None) -> Tensor:
        c = self._cond.to(x.device).expand(x.shape[0], 2)
        return self.family.energy_diff(x, c)

    def exact_log_Z(self) -> float | None:
        return self.family.exact_log_Z(self._cond)

    def sample_uniform(self, n: int, device=None, generator=None) -> Tensor:
        return self.family.sample_uniform(n, device=device, generator=generator)

    def concentration(self, x: Tensor) -> Tensor:
        return self.family.concentration(x)
