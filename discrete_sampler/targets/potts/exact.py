"""Exact partition function for the 2-D q-state Potts model on a torus.

Transfer-matrix computation over rows; cost O(M * q^{2L}), so use
n_cols such that q^n_cols <= ~4096 (e.g. L <= 7 for q = 3).
Convention matches `PottsModel`: E = -J sum_<ij> delta(x_i, x_j), K = beta*J.
"""

from __future__ import annotations

import numpy as np


def log_Z_transfer(n_rows: int, n_cols: int, q: int, K: float) -> float:
    """log Z for an (n_rows x n_cols) torus by transfer matrix over rows."""
    L = n_cols
    S = q**L
    # row configurations: (S, L) in {0..q-1}
    idx = np.arange(S)
    conf = np.empty((S, L), dtype=np.int64)
    for i in range(L):
        conf[:, i] = idx % q
        idx = idx // q
    # intra-row bonds (periodic)
    intra = K * (conf == np.roll(conf, -1, axis=1)).sum(axis=1).astype(np.float64)
    # inter-row bonds between row configs a and b
    inter = K * (conf[:, None, :] == conf[None, :, :]).sum(axis=2).astype(np.float64)
    logV = 0.5 * intra[:, None] + inter + 0.5 * intra[None, :]
    c = logV.max()
    V = np.exp(logV - c)
    eig = np.linalg.eigvalsh(V)
    eig = np.clip(eig, 1e-300, None)
    log_eig = n_rows * (np.log(eig) + c)
    m = log_eig.max()
    return float(m + np.log(np.exp(log_eig - m).sum()))


def beta_critical(q: int, J: float = 1.0) -> float:
    """Critical inverse temperature of the 2-D Potts model: log(1+sqrt(q))/J."""
    return float(np.log(1.0 + np.sqrt(q)) / J)
