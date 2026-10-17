"""Grid evaluation and phase diagrams for the alloy families.

The alloy packages carry more than a lattice target does, in two ways that
change what the right evaluation is:

* the exact ln Z is available on the *whole* plane, not just one line.  The
  packaged reference gives ln Z_n(T) at fixed composition, so the semi-grand
  value follows analytically and every grid point carries an absolute
  free-energy error with no MCMC in the loop.
* the reference observables are packaged too, per state point and per walker,
  so nothing has to be re-run.  Their `trapped` flag marks where the plain
  Metropolis walkers never agreed, which is exactly where a learned sampler
  is supposed to win, and is reported rather than averaged away.

The phase diagram follows the semi-grand construction, which is the one a
semi-grand sampler can draw directly: along each isotherm the composition
jumps across the first-order line, so the transition sits at the peak of
d<c>/d(delta_mu) and the coexisting compositions are the values bracketing
it.  The packaged `phase_diagram.json` reports the same construction under
`susceptibility`, alongside two others built from the fixed-composition free
energy, so the comparison is like for like.
"""

from __future__ import annotations

import os

import numpy as np
import torch

from discrete_sampler.targets.alloy.family import KB_EV_PER_K
from discrete_sampler.utils import metrics


def _wmean(w: torch.Tensor, v: torch.Tensor) -> float:
    return float((w * v.double()).sum())


@torch.no_grad()
def evaluate_alloy_grid(
    method,
    n_per_point: int = 1024,
    n_T: int = 11,
    n_dmu: int = 11,
    T_range: tuple[float, float] | None = None,
    dmu_range: tuple[float, float] | None = None,
    use_reference_grid: bool = False,
    with_order_parameters: bool = True,
    save_path: str | None = None,
    shard: tuple[int, int] | None = None,
) -> tuple[dict, list[dict]]:
    """Score an amortized alloy sampler over the (T, delta_mu) plane.

    `use_reference_grid` evaluates on the exact sweep the packaged data was
    generated at (2349 points for AgPd), which is what the final evaluation
    wants; the training-time curve uses a coarse `n_T x n_dmu` grid instead.

    `save_path` writes the raw draws to an .npz: the configurations (bit-packed
    along the site axis), their log importance weights and their semi-grand
    energies, with the state point each batch was drawn at.  Sampling the full
    sweep costs hours, so anything computed from these -- a different
    observable, a different boundary construction, a reweighting -- should be
    recomputed from the file rather than by drawing again.  At 1044 points and
    512 samples the file is about 6 MB; the 2349-point sweep is about 14 MB.
    """
    target = method.target
    assert getattr(target, "amortized", False), "needs a family target"
    if use_reference_grid and hasattr(target, "reference_grid"):
        grid = target.reference_grid(device=method.device)
    else:
        grid = target.cond_grid(n_T, n_dmu, device=method.device,
                                T_range=T_range, dmu_range=dmu_range)

    # `shard=(i, k)` evaluates every k-th state point starting at i.  The
    # sweep is embarrassingly parallel across state points -- each draws from
    # the model at its own conditioning and nothing is shared -- so the only
    # thing a shard has to carry is which points it did, for the merge to put
    # them back in order.  Interleaving rather than blocking keeps the cost
    # even: the ordered corner of the plane is slower per point than the hot
    # end, and a contiguous block would hand one job all of it.
    point_index = np.arange(len(grid))
    if shard is not None:
        i, k = shard
        grid = grid[i::k]
        point_index = point_index[i::k]

    rows: list[dict] = []
    keep_x, keep_w, keep_e, keep_c = [], [], [], []
    for point in grid:
        T, dmu = float(point[0]), float(point[1])
        x, log_w = method.sample_with_weights(n_per_point, cond=point)
        w = metrics.snis_weights(log_w)

        c = target.concentration(x).double()
        e = target.energy(x, point.view(1, 2).expand(x.shape[0], 2)).double() / target.D
        if save_path is not None:
            keep_x.append(np.packbits(
                x.detach().to(torch.uint8).cpu().numpy(), axis=-1))
            keep_w.append(log_w.detach().float().cpu().numpy())
            keep_e.append((e * target.D).float().cpu().numpy())
            keep_c.append(point.detach().float().cpu().numpy())
        # the semi-grand susceptibility as a fluctuation rather than a
        # difference of neighbours: dc/d(delta_mu) = beta Var(n) / N.  This is
        # the estimator the packaged boundary is built from, so carrying it
        # here lets the model be scored with the same construction instead of
        # a finite difference that lands a node away on the sharpest isotherms.
        c_rw = _wmean(w, c)
        var_rw = max(_wmean(w, c * c) - c_rw * c_rw, 0.0)
        row = {
            "T": T,
            "dmu": dmu,
            "c": float(c.mean()),
            "c_rw": c_rw,
            "c_std": float(c.std()),
            "c_var_rw": var_rw,
            "dc_ddmu": var_rw * target.D / (KB_EV_PER_K * T),
            "e_site": float(e.mean()),
            "e_site_rw": _wmean(w, e),
            "ess": metrics.ess(log_w),
            "log_w_std": float(log_w.double().std()),
        }
        try:
            wc = target.warren_cowley(x).double()
            row["warren_cowley"] = float(wc.mean())
            row["warren_cowley_rw"] = _wmean(w, wc)
        except Exception:  # noqa: BLE001 - optional observable
            pass
        if with_order_parameters:
            eta = target.order_parameters(x)
            if eta is not None:
                eta = eta.double().abs()
                row["eta_max"] = float(eta.max(dim=-1).values.mean())
                row["eta_max_rw"] = _wmean(w, eta.max(dim=-1).values)

        if method.unbiased_log_Z:
            lz = target.exact_log_Z(point.view(1, 2))
            if lz is not None:
                row["log_Z"] = metrics.log_Z_is(log_w)
                row["log_Z_exact"] = lz
                row["log_Z_err"] = abs(row["log_Z"] - lz)
                row["log_Z_err_per_site"] = row["log_Z_err"] / target.D

        ref = target.reference_state_point(T, dmu)
        if ref is not None:
            row["c_ref"] = ref["c"]
            row["e_site_ref"] = ref["E"] / target.D
            row["trapped"] = ref["trapped"]
            row["c_err"] = abs(row["c"] - ref["c"])
            row["c_err_rw"] = abs(row["c_rw"] - ref["c"])
            row["e_site_err"] = abs(row["e_site"] - row["e_site_ref"])
            if "warren_cowley" in row:
                row["warren_cowley_ref"] = ref["warren_cowley"]
                row["warren_cowley_err"] = abs(
                    row["warren_cowley"] - ref["warren_cowley"])
        rows.append(row)

    if save_path is not None and keep_x:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        np.savez_compressed(
            save_path,
            # (n_points, n_samples, ceil(D/8)) uint8; np.unpackbits(..., axis=-1)
            # then [..., :D] recovers the {0,1} tokens in raster order
            x_packed=np.stack(keep_x),
            point_index=point_index.astype(np.int64),
            n_sites=np.int64(target.D),
            log_w=np.stack(keep_w).astype(np.float32),
            energy=np.stack(keep_e).astype(np.float32),
            cond=np.stack(keep_c).astype(np.float32),
            system=str(getattr(target, "system", "")),
            note=("x_packed is bit-packed along the site axis; cond is "
                  "(T [K], delta_mu [eV]) per point; energy is the semi-grand "
                  "H(s) in eV; log_w = log nu_hat(x) - log q(x)."),
        )
        print(f"[eval] wrote {save_path} "
              f"({os.path.getsize(save_path) / 1e6:.1f} MB)", flush=True)

    summary = _summarize(rows)
    return summary, rows


def _summarize(rows: list[dict]) -> dict:
    def agg(key, how="mean", where=None):
        vals = [r[key] for r in rows if key in r and np.isfinite(r[key])
                and (where is None or where(r))]
        if not vals:
            return None
        return float(np.mean(vals)) if how == "mean" else float(np.max(vals))

    out: dict = {}
    for key in ("log_Z_err", "log_Z_err_per_site", "c_err", "c_err_rw",
                "e_site_err", "warren_cowley_err"):
        for how in ("mean", "max"):
            v = agg(key, how)
            if v is not None:
                out[f"grid/{key}_{how}"] = v
    out["grid/ess_mean"] = agg("ess")
    out["grid/ess_min"] = None if not rows else float(
        np.min([r["ess"] for r in rows]))
    out["grid/n_points"] = float(len(rows))
    # the points the reference itself could not resolve: a learned sampler
    # that is right there is doing something plain Metropolis could not
    trapped = [r for r in rows if r.get("trapped")]
    out["grid/n_trapped"] = float(len(trapped))
    if trapped:
        v = agg("log_Z_err", "mean", where=lambda r: r.get("trapped"))
        if v is not None:
            out["grid/log_Z_err_trapped_mean"] = v
    return {k: v for k, v in out.items() if v is not None}


# ----------------------------------------------------------------------
# Phase diagram
# ----------------------------------------------------------------------
def phase_diagram_from_rows(
    rows: list[dict], key: str = "c_rw", min_jump: float = 0.05,
    refine: bool = False, half_width: float = 0.01,
    deriv: str = "gradient",
) -> list[dict]:
    """Semi-grand phase boundaries: the composition jump along each isotherm.

    For every temperature the composition is scanned in delta_mu and the
    transition is placed at the peak of the finite-difference derivative.
    The coexisting compositions are the values on either side of that peak.
    `min_jump` suppresses isotherms above the critical point, where the
    composition varies smoothly and no boundary exists to report.

    `refine=False` reproduces the packaged construction exactly, peak on a
    grid node and compositions at the neighbouring nodes, which is what the
    reported comparison uses.  It has a visible artefact: the transition
    delta_mu can only take grid values, so it stays pinned for several
    isotherms while the true transition drifts and then jumps a whole step,
    and because the composition varies steeply there, every jump puts a
    sawtooth into c_left and c_right.  In the packaged Ag-Pd data all nine
    drops in c_right coincide with a step in delta_mu.

    `refine=True` removes it: the peak is located to sub-grid precision by a
    parabola through its three points, and the compositions are read at
    delta_mu* +- `half_width` eV by linear interpolation, a width fixed in eV
    rather than in grid steps so the answer no longer depends on the spacing.
    Use it for drawing; the two are not interchangeable in a comparison.

    `deriv` selects how dc/d(delta_mu) is obtained.  "gradient" finite-differences
    the composition column, which is all a set of sampled means supports.
    "stored" reads a `dc_ddmu` field off the rows, which is how the packaged
    construction is built: in the semi-grand ensemble the derivative is the
    exact fluctuation beta Var(n) / N rather than a difference of neighbours.
    The two disagree on 4 of the 29 Ag-Pd isotherms, each time by one 0.01 eV
    grid node, so a model scored with one against a reference built with the
    other carries that mismatch inside its reported error.  Score like with
    like.
    """
    by_T: dict[float, list[dict]] = {}
    for r in rows:
        by_T.setdefault(round(r["T"], 6), []).append(r)

    out = []
    for T in sorted(by_T):
        pts = sorted(by_T[T], key=lambda r: r["dmu"])
        if len(pts) < 3:
            continue
        dmu = np.array([p["dmu"] for p in pts], dtype=float)
        c = np.array([p.get(key, p["c"]) for p in pts], dtype=float)
        if deriv == "stored" and all("dc_ddmu" in p for p in pts):
            d = np.array([p["dc_ddmu"] for p in pts], dtype=float)
        else:
            d = np.gradient(c, dmu)
        i = int(np.argmax(d))
        if i == 0 or i == len(pts) - 1:
            continue
        jump = float(c[i + 1] - c[i - 1])
        if jump < min_jump:
            continue
        if not refine:
            out.append({
                "temperature": T,
                "dmu": float(dmu[i]),
                "c_left": float(c[i - 1]),
                "c_mid": float(c[i]),
                "c_right": float(c[i + 1]),
                "dc_ddmu": float(d[i]),
            })
            continue
        # sub-grid peak: a parabola through the three points around it.  The
        # denominator vanishes on a flat top, where the grid node is already
        # the best estimate.
        denom = d[i - 1] - 2.0 * d[i] + d[i + 1]
        delta = 0.0 if abs(denom) < 1e-30 else 0.5 * (d[i - 1] - d[i + 1]) / denom
        delta = float(np.clip(delta, -0.5, 0.5))
        step = float(dmu[i + 1] - dmu[i - 1]) / 2.0
        dmu_star = float(dmu[i] + delta * step)
        left = float(np.interp(dmu_star - half_width, dmu, c))
        right = float(np.interp(dmu_star + half_width, dmu, c))
        out.append({
            "temperature": T,
            "dmu": dmu_star,
            "c_left": left,
            "c_mid": float(np.interp(dmu_star, dmu, c)),
            "c_right": right,
            "dc_ddmu": float(d[i]),
        })
    return out


def compare_phase_diagram(model_pd: list[dict], target, deriv: str = "gradient",
                          **pd_kwargs) -> dict:
    """Boundary error against the same construction, run on the exact answer.

    The packaged `phase_diagram.json` is not the right thing to subtract.  It
    is built on the exact semi-grand concentration with the exact fluctuation
    derivative; a model boundary built by finite-differencing sampled means is
    a different construction, and the difference is not small.  Rebuilding the
    packaged list our way reproduces it to 3e-5 with `deriv="stored"`, but with
    a finite difference the peak lands one 0.01 eV node away on 4 of the 29
    Ag-Pd isotherms, moving the reported compositions by up to 0.065 at the
    coexistence branches and 0.41 at the midpoint -- larger than the boundary
    error of every method worth reporting.

    So the reference is rebuilt here with whatever construction produced
    `model_pd`, and only the distribution differs.  `pd/packaged_*` keeps the
    comparison against the packaged numbers so the two are visible side by side.
    """
    rows = target.exact_grid_rows() if hasattr(target, "exact_grid_rows") else None
    ref = (phase_diagram_from_rows(rows, deriv=deriv, **pd_kwargs)
           if rows else None)
    packaged = (target._ref.get("phase_diagram") or {}).get("susceptibility")
    if not ref:
        ref = packaged
    if not ref:
        return {}
    out = _boundary_err(model_pd, ref)
    if packaged and rows:
        out.update({f"pd/packaged_{k.split('/')[-1]}": v
                    for k, v in _boundary_err(model_pd, packaged).items()})
    return out


def _boundary_err(model_pd: list[dict], ref: list[dict]) -> dict:
    ref_by_T = {round(float(e["temperature"]), 6): e for e in ref}
    dl, dr, dmu_err, n = [], [], [], 0
    for e in model_pd:
        r = ref_by_T.get(round(float(e["temperature"]), 6))
        if r is None:
            continue
        n += 1
        dl.append(abs(e["c_left"] - float(r["c_left"])))
        dr.append(abs(e["c_right"] - float(r["c_right"])))
        dmu_err.append(abs(e["dmu"] - float(r["dmu"])))
    if not n:
        return {}
    return {
        "pd/n_matched": float(n),
        "pd/c_left_err_mean": float(np.mean(dl)),
        "pd/c_right_err_mean": float(np.mean(dr)),
        "pd/c_boundary_err_max": float(max(np.max(dl), np.max(dr))),
        "pd/dmu_err_mean": float(np.mean(dmu_err)),
    }
