"""Target interface for discrete Boltzmann distributions.

A target is a distribution nu(x) = exp(-beta * E(x)) / Z on the discrete
product space X = {0, ..., N-1}^D.  All methods operate on batches of
integer-encoded states of shape (B, D) with values in [0, N).

The central quantity for energy-based training is the table of
single-coordinate energy differences

    energy_diff(x)[b, d, n] = E(x_b^{d <- n}) - E(x_b),

from which the discrete (concrete) target score

    s_nu(x)_{d,n} = nu(x^{d <- n}) / nu(x) = exp(-beta * energy_diff)

and the heat-bath (single-site Gibbs) conditional

    nu(n | x^{-d}) = softmax_n(-beta * E(x^{d <- n}))

follow without knowledge of Z.  Subclasses implement `energy` and
`energy_diff`; everything else is derived.
"""

from __future__ import annotations

import functools
import math
from abc import ABC, abstractmethod

import torch
from torch import Tensor

#: the two primitives every method's cost is measured in
CALL_KINDS = ("energy", "energy_diff")


def _counted(fn, kind: str):
    """Wrap a target primitive so every call adds its batch size to
    `self.n_calls[kind]`.  Applied to each subclass's own override, so a
    subclass that reimplements `energy` without calling super() is counted
    exactly once."""
    @functools.wraps(fn)
    def wrapper(self, x, *args, **kwargs):
        counts = getattr(self, "n_calls", None)
        if counts is not None:
            counts[kind] += int(x.shape[0])
        return fn(self, x, *args, **kwargs)
    wrapper._dds_counted = True
    return wrapper


class DiscreteTarget(ABC):
    """Boltzmann distribution nu(x) ∝ exp(-beta E(x)) on {0..N-1}^D.

    Every call of `energy` / `energy_diff` is metered in `n_calls`, keyed by
    primitive and counting *configurations* (batch rows), so the training
    cost of a method can be reported in target evaluations rather than in
    wall-clock alone.  `energy_diff` returns the full table of D single-site
    differences of one configuration in one pass; it is counted as one call
    of its own kind, not as D energy calls.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for kind in CALL_KINDS:
            fn = cls.__dict__.get(kind)
            if fn is not None and not getattr(fn, "_dds_counted", False):
                setattr(cls, kind, _counted(fn, kind))

    #: May a colouring of `lattice_shape` be used to update many sites at
    #: once?  True for a lattice model whose interactions reach one step, so
    #: same-coloured sites are conditionally independent.  False whenever the
    #: interaction is longer-ranged than the colouring assumes -- a fluid,
    #: where every particle interacts with every other, or an embedded-atom
    #: alloy, where independence would need twice the potential cutoff.  A
    #: target that says False is swept one site at a time by its own sampler.
    colour_independent = True

    def __init__(self, dim: int, num_states: int, beta: float = 1.0):
        self.D = dim
        self.N = num_states
        self.beta = beta
        self.n_calls = {k: 0 for k in CALL_KINDS}

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------
    @abstractmethod
    def energy(self, x: Tensor, cond: Tensor | None = None) -> Tensor:
        """E(x) for integer states x of shape (B, D).  Returns (B,)."""

    @abstractmethod
    def energy_diff(self, x: Tensor, cond: Tensor | None = None) -> Tensor:
        """E(x^{d<-n}) - E(x), shape (B, D, N); zero at n = x^d."""

    # ------------------------------------------------------------------
    # Derived quantities
    # ------------------------------------------------------------------
    # `cond` carries per-sample thermodynamic parameters for targets that are
    # a *family* rather than a single distribution (see AlloyFamily).  Plain
    # single-parameter targets ignore it, so every existing call site keeps
    # working and an amortized target can be dropped in without a second API.
    def _beta_of(self, cond: Tensor | None):
        """Inverse temperature: a float, or (B,) when amortized."""
        return self.beta

    def log_prob(self, x: Tensor, cond: Tensor | None = None) -> Tensor:
        """Unnormalized log nu(x) = -beta E(x), shape (B,)."""
        beta = self._beta_of(cond)
        return -beta * self.energy(x, cond)

    def log_prob_and_score(self, x: Tensor, cond: Tensor | None = None):
        """`(log nu(x), log s_nu(x))`, from one pass where the target offers it.

        Weighted training needs both at every buffered endpoint.  A target
        that can produce them together says so by implementing
        `energy_and_diff`; everything else falls back to the two primitives
        and pays for two passes.
        """
        fused = getattr(self, "energy_and_diff", None)
        if fused is None:
            return self.log_prob(x, cond), self.log_score(x, cond)
        beta = self._beta_of(cond)
        e, d = fused(x, cond)
        b = beta.view(-1, 1, 1) if torch.is_tensor(beta) else beta
        return -self._beta_of(cond) * e, -b * d

    def log_score(self, x: Tensor, cond: Tensor | None = None) -> Tensor:
        """log s_nu(x)_{d,n} = -beta [E(x^{d<-n}) - E(x)], shape (B, D, N)."""
        beta = self._beta_of(cond)
        if torch.is_tensor(beta):
            beta = beta.view(-1, 1, 1)
        return -beta * self.energy_diff(x, cond)

    def score(self, x: Tensor, cond: Tensor | None = None) -> Tensor:
        """s_nu(x)_{d,n} = nu(x^{d<-n}) / nu(x), shape (B, D, N)."""
        return torch.exp(self.log_score(x, cond))

    def heat_bath(self, x: Tensor, cond: Tensor | None = None) -> Tensor:
        """Single-site conditionals nu(n | x^{-d}), shape (B, D, N)."""
        return torch.softmax(self.log_score(x, cond), dim=-1)

    # ------------------------------------------------------------------
    # Exact reference (small systems)
    # ------------------------------------------------------------------
    def exact_log_Z(self) -> float | None:
        """Exact log partition function, or None if unavailable."""
        if self.N**self.D <= 2**24:
            return self.log_Z_enumeration()
        return None

    def log_Z_enumeration(self, batch_size: int = 2**16, device: str = "cpu") -> float:
        """log Z by exhaustive enumeration (feasible for N^D <~ 2^24)."""
        total = self.N**self.D
        log_terms = []
        for start in range(0, total, batch_size):
            idx = torch.arange(start, min(start + batch_size, total), device=device)
            x = self._index_to_state(idx)
            log_terms.append(torch.logsumexp(self.log_prob(x), dim=0))
        return torch.logsumexp(torch.stack(log_terms), dim=0).item()

    def _index_to_state(self, idx: Tensor) -> Tensor:
        """Decode integers in [0, N^D) to states of shape (len(idx), D)."""
        digits = []
        for _ in range(self.D):
            digits.append(idx % self.N)
            idx = idx // self.N
        return torch.stack(digits, dim=1).long()

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def sample_uniform(self, n: int, device=None, generator=None) -> Tensor:
        """n i.i.d. uniform states, shape (n, D)."""
        return torch.randint(
            0, self.N, (n, self.D), device=device, generator=generator
        )

    def check_energy_diff(self, n: int = 8, device: str = "cpu", atol: float = 1e-4):
        """Sanity check: energy_diff agrees with brute-force substitution."""
        x = self.sample_uniform(n, device=device)
        de = self.energy_diff(x)
        e0 = self.energy(x)
        for d in range(self.D):
            for v in range(self.N):
                y = x.clone()
                y[:, d] = v
                brute = self.energy(y) - e0
                if not torch.allclose(de[:, d, v], brute, atol=atol):
                    raise AssertionError(
                        f"energy_diff mismatch at d={d}, n={v}: "
                        f"{de[:, d, v]} vs {brute}"
                    )
        return True


def log_Z_uniform(target: DiscreteTarget) -> float:
    """log Z of the uniform distribution on the target's state space."""
    return target.D * math.log(target.N)
