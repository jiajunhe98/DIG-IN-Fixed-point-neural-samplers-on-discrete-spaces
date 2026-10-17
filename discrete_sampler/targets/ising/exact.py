"""Exact partition functions for the 2-D Ising model on a torus.

Two independent routes, used to validate samplers (and each other):

* `log_Z_transfer`: exact transfer-matrix computation for an M x L torus
  with arbitrary field h.  Cost O(M * 2^{2L}); use L <= ~12.
* `log_Z_kaufman`: Kaufman's closed-form solution for the finite periodic
  lattice at h = 0.  Cost O(M L); any size.

Conventions match `IsingModel`: E = -J sum_<ij> s_i s_j - h sum_i s_i,
K = beta * J, B = beta * h; both functions return log Z (natural log).
"""

from __future__ import annotations

import numpy as np


def log_Z_transfer(n_rows: int, n_cols: int, K: float, B: float = 0.0) -> float:
    """log Z for an (n_rows x n_cols) torus by transfer matrix over rows.

    K = beta*J, B = beta*h.  Exact for any K, B; feasible for n_cols <= ~12.
    """
    L = n_cols
    S = 1 << L
    # spin configurations of one row: (S, L) in {-1, +1}
    bits = ((np.arange(S)[:, None] >> np.arange(L)[None, :]) & 1).astype(np.float64)
    s = 2.0 * bits - 1.0
    # intra-row bonds (periodic) and field, per configuration
    intra = K * (s * np.roll(s, -1, axis=1)).sum(axis=1) + B * s.sum(axis=1)
    # inter-row bonds between configurations a and b: K * s_a . s_b
    inter = K * (s @ s.T)
    # symmetric transfer matrix: V = D^{1/2} exp(inter) D^{1/2}, Z = tr(V^M)
    logV = 0.5 * intra[:, None] + inter + 0.5 * intra[None, :]
    # stabilize: factor out max before eigendecomposition
    c = logV.max()
    V = np.exp(logV - c)
    eig = np.linalg.eigvalsh(V)
    eig = np.clip(eig, 1e-300, None)
    M = n_rows
    log_eig = M * (np.log(eig) + c)
    m = log_eig.max()
    return float(m + np.log(np.exp(log_eig - m).sum()))


def log_Z_kaufman(n_rows: int, n_cols: int, K: float) -> float:
    """Kaufman's exact log Z for the (n_rows x n_cols) periodic lattice, h = 0.

    Z = (1/2) (2 sinh 2K)^{MN/2} * (P1 + P2 + P3 + P4), where the P_i are
    products of 2cosh(M gamma_l / 2) / 2sinh(M gamma_l / 2) over odd/even l,
    cosh gamma_l = cosh 2K coth 2K - cos(pi l / N), and
    gamma_0 = 2K + ln tanh K (with its sign).
    """
    M, N = n_rows, n_cols
    K = float(K)
    c2k = np.cosh(2 * K)
    coth2k = 1.0 / np.tanh(2 * K)

    def gamma(l: int) -> float:
        if l == 0:
            return 2 * K + np.log(np.tanh(K))  # signed
        cg = c2k * coth2k - np.cos(np.pi * l / N)
        return float(np.arccosh(cg))

    # log(2 cosh(M g / 2)) and log|2 sinh(M g / 2)| computed stably
    def log2cosh(a: float) -> float:
        a = abs(a)
        return a + np.log1p(np.exp(-2 * a))

    def log2sinh_abs(a: float) -> float:
        a = abs(a)
        if a < 1e-12:
            return -np.inf
        return a + np.log1p(-np.exp(-2 * a))

    odd = [gamma(2 * r + 1) for r in range(N)]
    even = [gamma(2 * r) for r in range(N)]

    logP1 = sum(log2cosh(M * g / 2) for g in odd)
    logP2 = sum(log2sinh_abs(M * g / 2) for g in odd)
    logP3 = sum(log2cosh(M * g / 2) for g in even)
    logP4 = sum(log2sinh_abs(M * g / 2) for g in even)
    # sign of P4: product of signs of sinh(M gamma_l / 2); only gamma_0 can be < 0
    sign4 = -1.0 if gamma(0) < 0 else 1.0

    logs = np.array([logP1, logP2, logP3, logP4])
    signs = np.array([1.0, 1.0, 1.0, sign4])
    m = logs.max()
    total = (signs * np.exp(logs - m)).sum()
    log_sum = m + np.log(total)

    return float(
        -np.log(2.0) + 0.5 * M * N * (np.log(2.0) + np.log(np.sinh(2 * K))) + log_sum
    )
