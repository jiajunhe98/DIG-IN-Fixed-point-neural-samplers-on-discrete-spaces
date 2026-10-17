"""Tempered self-normalized importance weights with an ESS rule.

A buffer sample carries the configuration weight w(x) = nu_hat(x) / q(x),
the ratio of the unnormalized target to the law that produced it.  Weighting
a per-endpoint loss by the self-normalized w turns the buffer objective into
the data objective for *any* proposal, but the weights degenerate whenever
the model is far from the target.  The remedy is to raise them to a power
nu in [0, 1] before normalizing,

    w_bar(nu)_i = w_i^nu / sum_j w_j^nu,

which is the unweighted loss at nu = 0 and the fully weighted one at nu = 1.
In population this replaces the target by the geometric interpolation
q^(1-nu) nu_hat^nu / Z_nu, the annealing path run between the model and the
target rather than between a prior and the target.

The exponent is chosen per batch by the effective-sample-size rule

    nu* = max{ nu in [0, 1] : ESS(nu) >= xi B },
    ESS(nu) = (sum_i w_i^nu)^2 / sum_i w_i^(2 nu),

which is well posed because ESS is continuous and non-increasing in nu with
ESS(0) = B: either the raw weights already clear the bar, and nu* = 1, or
nu* is the unique root, found by bisection.
"""

from __future__ import annotations

import torch
from torch import Tensor


def log_ess_fraction(log_w: Tensor, nu: float | Tensor) -> Tensor:
    """log of ESS(nu) / B, computed in log space (weights may overflow)."""
    n = log_w.shape[0]
    lw = nu * log_w
    return (2.0 * torch.logsumexp(lw, 0) - torch.logsumexp(2.0 * lw, 0)
            - torch.log(torch.tensor(float(n), device=log_w.device)))


def tempered_weights(
    log_w: Tensor, target_ess: float, n_bisect: int = 40
) -> tuple[Tensor, float, float]:
    """Self-normalized weights w_bar(nu*) for the ESS rule.

    Returns (w_bar, nu, ess_fraction_at_nu).  Non-finite log-weights are
    treated as -inf (a sample the target gives zero mass); an all-degenerate
    batch falls back to uniform weights with nu = 0.
    """
    log_w = torch.nan_to_num(log_w.detach().float(), nan=float("-inf"),
                             posinf=float("inf"), neginf=float("-inf"))
    if not torch.isfinite(log_w).any():
        n = log_w.shape[0]
        return torch.full_like(log_w, 1.0 / n), 0.0, 1.0
    log_w = torch.where(torch.isfinite(log_w), log_w, torch.finfo(log_w.dtype).min)
    log_xi = torch.log(torch.tensor(float(target_ess), device=log_w.device))

    if log_ess_fraction(log_w, 1.0) >= log_xi:
        nu = 1.0
    else:
        lo, hi = 0.0, 1.0  # ESS(0)/B = 1 >= xi, ESS(1)/B < xi
        for _ in range(n_bisect):
            mid = 0.5 * (lo + hi)
            if log_ess_fraction(log_w, mid) >= log_xi:
                lo = mid
            else:
                hi = mid
        nu = lo  # the largest exponent that still clears the bar
    return (torch.softmax(nu * log_w, dim=0), float(nu),
            float(log_ess_fraction(log_w, nu).exp()))

def cell_index(cond: Tensor, ranges, bins) -> tuple[Tensor, int]:
    """Flat cell index of each conditioning value on a rectangular grid."""
    idx = torch.zeros(cond.shape[0], dtype=torch.long, device=cond.device)
    stride = 1
    for d, (lo, hi) in enumerate(ranges):
        n = int(bins[d]) if d < len(bins) else int(bins[-1])
        u = (cond[:, d] - float(lo)) / max(float(hi) - float(lo), 1e-12)
        b = (u.clamp(0.0, 1.0 - 1e-9) * n).long().clamp(0, n - 1)
        idx = idx + b * stride
        stride *= n
    return idx, stride


def centre_log_w_by_buffer(
    log_w: Tensor, cond: Tensor, ref_log_w: Tensor, ref_cond: Tensor,
    ranges, bins=(16, 16), min_count: int = 16
) -> tuple[Tensor, float]:
    """Remove the per-state-point offset, estimated from the whole buffer.

    `centre_log_w_by_cell` estimates each cell's mean from the minibatch that
    is being weighted, which couples two things that should be independent:
    how finely the box can be cut, and how many samples the loss happens to
    take.  64 cells cannot be filled from a batch of 128, and that is not a
    hypothetical -- it silently turned a weighted arm into an unweighted one.

    The offset is a property of the conditioning, not of the minibatch, and
    the buffer already holds `log_p`, `log_q` and `cond` for every endpoint in
    it.  Estimating the cell means from all of them decouples the two: 25000
    endpoints over the acquisition's own 16 x 16 grid is about a hundred per
    cell, whatever the batch size, on four times as fine a grid as the batch
    could support.  The endpoints were drawn from older parameters than the
    ones being trained, but the quantity being estimated -- how much log nu
    varies between state points -- is a property of the target, and the
    exploration mixture keeps the buffer spread over the box by construction.

    Returns the centred batch log-weights and the fraction of the batch whose
    cell was thick enough to use.
    """
    if (cond is None or ref_cond is None or cond.ndim != 2
            or cond.shape[0] != log_w.shape[0]
            or ref_cond.shape[0] != ref_log_w.shape[0]):
        return log_w, 0.0
    idx, n_cells = cell_index(cond, ranges, bins)
    ridx, _ = cell_index(ref_cond, ranges, bins)

    counts = torch.bincount(ridx, minlength=n_cells).to(log_w.dtype)
    sums = torch.zeros(n_cells, dtype=log_w.dtype,
                       device=log_w.device).index_add_(0, ridx, ref_log_w)
    thick = counts >= min_count
    means = torch.where(thick, sums / counts.clamp(min=1.0),
                        torch.zeros_like(sums))
    keep = thick[idx]
    out = torch.where(keep, log_w - means[idx], torch.zeros_like(log_w))
    return out, float(keep.float().mean())


def tempered_weights_local(
    log_w: Tensor, cond: Tensor, ref_log_w: Tensor, ref_cond: Tensor,
    ranges, target_ess: float, bins=(8, 8), min_count: int = 24,
    n_bisect: int = 40,
) -> tuple[Tensor, float, float, float]:
    """The ESS rule applied inside each conditioning cell, end to end.

    Degeneracy is a property of a state point, not of a batch.  If the model
    fits well at high temperature and badly at low, one exponent for the
    whole box is a compromise everywhere: too sharp where the fit is bad, so
    the weights collapse, and too flat where it is good, so the signal is
    left on the table.  Running the rule per cell gives each region the
    exponent its own weights can support.

    This also removes any need to centre.  The ESS ratio

        (sum_i e^{nu w_i})^2 / sum_i e^{2 nu w_i}

    is invariant to adding a constant to every w_i, so inside a cell the
    between-state-point offset -- the 260 nats that made a global rule
    useless -- cancels exactly and never has to be estimated.  The centring
    apparatus, its cell means and occupancy thresholds, existed only to
    rescue a global rule.  Verified: shifting log_w by 0, 50 and 260 nats
    leaves the ESS and nu* bit-identical.

    Each cell's exponent comes from the BUFFER, which holds about a hundred
    endpoints a cell, not from the handful in the batch.  Cells with fewer
    than `min_count` buffered endpoints fall back to the pooled exponent.

    Normalisation is per cell, each carrying the share of the batch it holds,
    so the mixture over the box is left to the conditioning draw.  A cell
    holding one batch sample gives it weight 1/B exactly -- the unweighted
    value -- so thin cells degrade to no weighting rather than to anything
    pathological.

    Returns (w_bar, mean nu over the batch, pooled ESS fraction, spread of nu
    across the batch's cells).

    This is the scheme the paper's alloy runs used.  The amortized lattice
    trainer (`alpha_and_weights_per_bin` in fixed_point/ising_beta_h/
    train_h.py) differs in the normalization: there the weights are
    normalized over ALL buffered endpoints of a cell, once per exploration
    round, and stored with the endpoint; here the normalization runs over the
    endpoints of each cell that land in the current training batch.
    """
    if (cond is None or ref_cond is None or cond.ndim != 2
            or cond.shape[0] != log_w.shape[0]):
        w, nu, ess = tempered_weights(log_w, target_ess, n_bisect)
        return w, nu, ess, 0.0

    idx, n_cells = cell_index(cond, ranges, bins)
    ridx, _ = cell_index(ref_cond, ranges, bins)
    rcount = torch.bincount(ridx, minlength=n_cells)

    _, nu_global, _ = tempered_weights(ref_log_w, target_ess, n_bisect)
    nu_of_cell = torch.full((n_cells,), float(nu_global), device=log_w.device,
                            dtype=log_w.dtype)
    for c in torch.unique(idx).tolist():
        if int(rcount[c]) >= min_count:
            _, nu_c, _ = tempered_weights(ref_log_w[ridx == c], target_ess,
                                          n_bisect)
            nu_of_cell[c] = nu_c

    lw = nu_of_cell[idx] * log_w
    m = torch.full((n_cells,), float("-inf"), device=lw.device, dtype=lw.dtype)
    m = m.scatter_reduce(0, idx, lw, reduce="amax", include_self=True)
    ex = torch.exp(lw - m[idx])
    den = torch.zeros(n_cells, device=lw.device, dtype=lw.dtype).index_add_(0, idx, ex)
    share = torch.bincount(idx, minlength=n_cells).to(lw.dtype) / lw.shape[0]
    w = ex / den[idx].clamp(min=1e-30) * share[idx]
    w = w / w.sum().clamp(min=1e-30)

    ess_pool = float(1.0 / (w.double() ** 2).sum() / w.shape[0])
    nb = nu_of_cell[idx]
    return (w, float(nb.mean()), ess_pool,
            float(nb.std()) if nb.numel() > 1 else 0.0)


def tempered_weights_per_cell(
    log_w: Tensor, cond: Tensor, ranges, target_ess: float,
    bins=(8, 8), min_count: int = 8, n_bisect: int = 40
) -> tuple[Tensor, float, float, float]:
    """Tempered weights normalised inside each conditioning cell.

    `tempered_weights` divides by one constant over the whole batch.  After
    centring, each cell's residuals are mean-zero, so the total mass a cell
    receives under a global softmax is about n_c exp(nu^2 sigma_c^2 / 2):
    cells where the model fits WORSE collect more weight, purely because
    their residuals are wider.  On four equal cells with spreads of 5, 15, 30
    and 50 nats that is a 5% tilt at nu = 0.009, a 45% tilt at nu = 0.023,
    and by nu = 0.2 one cell owns the entire batch.  Harmless while the
    weighting barely engages, which is the only reason it has not mattered
    yet, and wrong exactly when it starts working.

    Normalising inside each cell and giving each cell mass proportional to
    its own count removes the tilt by construction: the weights then rank
    samples against others at the same state point, which is what they were
    always meant to do, and the *relative* mass of the cells stays what the
    conditioning draw made it.

    The exponent is still chosen globally, on the pooled residuals.  A
    per-cell ESS rule would be the natural companion but a cell holds only a
    handful of the batch -- 4 at batch 1024 over a 16 x 16 grid -- and an ESS
    estimated from four samples is noise.  This is also why `bins` here
    defaults coarser than the centring grid: centring is estimated from the
    buffer and wants resolution, normalising happens inside the batch and
    wants occupancy.

    Cells below `min_count` in this batch are pooled and normalised together
    rather than dropped, so no endpoint loses its weight to a thin cell.

    Returns (w_bar, nu, pooled ess fraction, fraction of the batch in a cell
    thick enough to normalise on its own).
    """
    w_global, nu, ess = tempered_weights(log_w, target_ess, n_bisect)
    if cond is None or cond.ndim != 2 or cond.shape[0] != log_w.shape[0]:
        return w_global, nu, ess, 0.0

    idx, n_cells = cell_index(cond, ranges, bins)
    counts = torch.bincount(idx, minlength=n_cells)
    thick = counts >= min_count
    # everything in a thin cell shares one pooled group, index n_cells
    grp = torch.where(thick[idx], idx, torch.full_like(idx, n_cells))
    n_grp = n_cells + 1

    lw = nu * log_w
    # per-group logsumexp, in log space so an empty exponent cannot overflow
    m = torch.full((n_grp,), float("-inf"), device=lw.device, dtype=lw.dtype)
    m = m.scatter_reduce(0, grp, lw, reduce="amax", include_self=True)
    ex = torch.exp(lw - m[grp])
    den = torch.zeros(n_grp, device=lw.device, dtype=lw.dtype).index_add_(0, grp, ex)
    # inside a group the weights sum to 1; the group then carries the share of
    # the batch it actually holds, so the conditioning draw sets the mixture
    share = torch.bincount(grp, minlength=n_grp).to(lw.dtype) / lw.shape[0]
    w = ex / den[grp].clamp(min=1e-30) * share[grp]
    w = w / w.sum().clamp(min=1e-30)
    ess_pool = float(1.0 / (w.double() ** 2).sum() / w.shape[0])
    return w, nu, ess_pool, float(thick[idx].float().mean())


def centre_log_w_by_cell(
    log_w: Tensor, cond: Tensor, ranges, bins=(8, 8), min_count: int = 8
) -> tuple[Tensor, float]:
    """Remove the between-state-point offset from a batch's log weights.

    An amortized batch mixes conditioning: every endpoint was drawn at its own
    theta, and log nu_hat differs across the box by far more than it differs
    between good and bad samples at one theta.  On the alloy families the
    batch spread is 260 to 320 nats, which is beta times an eV-scale energy
    over a hundred sites, while the part the weighting is meant to exploit --
    how well the model fits at a given theta -- is a few nats.  Self-
    normalising across the whole batch therefore ranks state points by free
    energy, and the ESS rule answers by driving the exponent to nu ~ 1/spread,
    which flattens the weights to nothing.  Measured on Ag-Pd: target ESS 0.3,
    0.9 and 0.99 gave nu = 0.0036, 0.0011 and 0.00037, all with nu * spread
    near one nat, so every setting was the unweighted loss in disguise.

    Binning theta and subtracting the mean log weight inside each cell leaves
    exactly the within-state-point variation, which is what the importance
    weights are supposed to be about.  Cells too thin to estimate a mean from
    are zeroed rather than guessed, which makes their endpoints equally
    weighted instead of arbitrarily weighted.

    Returns the centred log weights and the fraction of the batch that landed
    in a cell thick enough to use.
    """
    if cond is None or cond.ndim != 2 or cond.shape[0] != log_w.shape[0]:
        return log_w, 0.0
    idx = torch.zeros(log_w.shape[0], dtype=torch.long, device=log_w.device)
    stride = 1
    for d, (lo, hi) in enumerate(ranges):
        n = int(bins[d]) if d < len(bins) else int(bins[-1])
        u = (cond[:, d] - float(lo)) / max(float(hi) - float(lo), 1e-12)
        b = (u.clamp(0.0, 1.0 - 1e-9) * n).long().clamp(0, n - 1)
        idx = idx + b * stride
        stride *= n

    out = log_w.clone()
    used = torch.zeros_like(log_w, dtype=torch.bool)
    n_cells = stride
    counts = torch.bincount(idx, minlength=n_cells).to(log_w.dtype)
    sums = torch.zeros(n_cells, dtype=log_w.dtype,
                       device=log_w.device).index_add_(0, idx, log_w)
    thick = counts >= min_count
    means = torch.where(thick, sums / counts.clamp(min=1.0),
                        torch.zeros_like(sums))
    keep = thick[idx]
    out = torch.where(keep, log_w - means[idx], torch.zeros_like(log_w))
    used = keep
    return out, float(used.float().mean())

def regress_out_cond(log_w: Tensor, feats: Tensor | None,
                     ridge: float = 1e-6) -> tuple[Tensor, float]:
    """Remove the smooth theta-dependence of the log weights by fitting it.

    The cell-centred version of this (`centre_log_w_by_cell`) works but is
    limited by its own resolution: on Ag-Pd it takes the batch spread from
    265 nats to 57 at 8x8 bins and to 42 at 16x16.  Halving the cell width
    in beta halves the drift inside a cell, and solving
    r^2 = drift^2 + floor^2 against those two numbers puts the drift at 44
    nats and the floor at 36 -- so about 60% of what the binning leaves is
    just unresolved drift, and finer bins are not the way to reach the floor
    because the batch has to fill them.

    Fitting instead of binning has no resolution limit.  log nu_hat is
    -beta (E - dmu M / 2), so its dependence on the conditioning is smooth
    and close to bilinear in (beta_hat, dmu_hat); a least-squares fit on
    [1, f_i, f_i f_j] removes it continuously and leaves the part that
    varies at fixed theta.

    The floor is not removable by any centring: it is how far the model's
    log density sits from the target's at a single state point.  Expect
    about 36 nats on Ag-Pd, i.e. nu near 0.03 -- twenty-five times the
    plain rule's 0.0011, and still nowhere near fully weighted.

    Returns the residual log weights and the fraction of variance removed.
    """
    if feats is None or feats.ndim != 2 or feats.shape[0] != log_w.shape[0]:
        return log_w, 0.0
    y = log_w.double()
    f = feats.double()
    cols = [torch.ones_like(y).unsqueeze(1), f]
    d = f.shape[1]
    for i in range(d):
        for j in range(i, d):
            cols.append((f[:, i] * f[:, j]).unsqueeze(1))
    X = torch.cat(cols, dim=1)
    if X.shape[0] <= X.shape[1]:
        return log_w, 0.0
    # normal equations with a ridge: the design is tiny and the columns are
    # correlated once the box is sampled unevenly
    XtX = X.T @ X
    XtX = XtX + ridge * torch.eye(XtX.shape[0], dtype=XtX.dtype,
                                  device=XtX.device) * XtX.diagonal().mean()
    beta = torch.linalg.solve(XtX, X.T @ y)
    resid = y - X @ beta
    v0 = float(y.var()) if y.numel() > 1 else 0.0
    removed = 0.0 if v0 <= 0 else max(0.0, 1.0 - float(resid.var()) / v0)
    return resid.to(log_w.dtype), removed
