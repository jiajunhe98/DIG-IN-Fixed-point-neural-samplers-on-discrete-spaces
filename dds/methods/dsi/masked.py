"""Absorbing-mask interpolant (Section `masked` of the method doc).

The alphabet is extended by a mask token M = N.  A coordinate is either
revealed (showing its endpoint value) or masked; the reveal probability is
a_t.  The endpoint posterior at masked coordinates is t-free, so the
denoiser network may be time-free (MDNS-style); the reveal fraction can
optionally be passed as a time input.

Training targets:
* soft CE:   heat-bath label  h_nu(n | x1^{-d})              (bounded)
* Bregman/TSI: raw target score s_nu(x1)_{d,n} = e^{-beta dE} (unbounded),
  whose conditional mean given the masked state is N * q^d(n | x)
  (uniform-shift form of the target score identity at the mask).

Sampling is any-order autoregressive unmasking with a tractable proposal
density, so log Z estimation is ordinary importance sampling.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor


def reveal_edges(dim: int, n_steps: int, n_single: int = 0) -> list[int]:
    """Cumulative reveal counts for `n_steps` network calls.

    `n_steps == dim` reveals one coordinate per call, i.e. exact sequential
    unmasking with no factorization error.  Below that, `n_single > 0`
    reveals the first `n_single` coordinates one at a time and splits the
    rest into equal chunks: the factorization error of revealing several
    coordinates at once is worst at high mask fraction, where the joint
    conditional is still globally correlated and nothing has been decided,
    so that is where the network calls should be spent.  Measured on the
    25x25 Ising sampler at identical cost (78 calls): 64 singles gives
    d|m| = +0.003 at sigma = 0.30 where 0 singles gives -0.441.

    A schedule that did not reach `dim` would leave mask tokens in the
    supposedly clean endpoint and silently corrupt the training labels, so
    that is asserted rather than clamped.
    """
    n_single = min(int(n_single), n_steps - 1, dim - 1)
    if n_single <= 0:
        edges = torch.linspace(0, dim, n_steps + 1).round().long().tolist()
    else:
        rest = torch.linspace(n_single, dim, n_steps - n_single + 1).round().long().tolist()
        edges = list(range(n_single)) + rest
    assert len(edges) == n_steps + 1 and edges[0] == 0 and edges[-1] == dim, (
        n_steps, n_single, dim
    )
    return edges


class MaskedInterpolant:
    """Absorbing-mask interpolant, optionally with a *block* reveal order.

    ``block > 1`` groups consecutive coordinates and makes the reveal order a
    random permutation of blocks, with the coordinates inside a block revealed
    consecutively.  One coordinate is still revealed per network call, so the
    sampler stays exact; what changes is which contexts it passes through.

    Why it matters.  For a fluid a coordinate is one Cartesian component of one
    particle, and a block is one whole particle.  Under a uniform coordinate
    order the model is repeatedly asked for the y of a particle whose x is known
    while its neighbours are half placed -- a different conditional at every mask
    pattern, none of them simple.  Under a block order it is asked to insert one
    particle into a configuration of complete particles, which for an
    exchangeable system is the sequential-insertion conditional of a fluid: the
    same object at every step, and the one a fixed-order model gets to learn
    exclusively.  ``block = 1`` reproduces the uniform order exactly, so lattice
    targets, whose sites carry no such grouping, are unaffected.

    The training mask has to follow the same order or the heads are optimised on
    contexts the sampler never presents; :meth:`sample` does that.
    """

    def __init__(self, num_states: int, block: int = 1, stage=None):
        self.N = num_states
        self.mask_token = num_states
        self.block = max(1, int(block))
        #: optional stage per block.  Blocks are then ordered by stage and
        #: shuffled freely within one, which is what turns a random any-order
        #: reveal into a coarse-to-fine one without giving up any-order inside a
        #: stage.  `None` keeps the plain random permutation, so every existing
        #: target is byte-identical.
        self.stage = None if stage is None else stage.long().clone()

    # ------------------------------------------------------------------
    def _order(self, n: int, dim: int, device, generator=None) -> Tensor:
        """``(n, dim)`` reveal order; block-grouped when ``block > 1``."""
        if self.block <= 1:
            return torch.argsort(
                torch.rand(n, dim, device=device, generator=generator), dim=1)
        if dim % self.block:
            raise ValueError(
                f"block={self.block} does not divide dim={dim}; the reveal "
                f"order would not be a permutation")
        nb = dim // self.block
        r = torch.rand(n, nb, device=device, generator=generator)
        if self.stage is not None:
            if self.stage.numel() != nb:
                raise ValueError(
                    f"block_stage has {self.stage.numel()} entries for "
                    f"{nb} blocks")
            # the random part is in [0, 1), so adding the integer stage sorts
            # by stage first and breaks ties uniformly at random within it
            r = r + self.stage.to(device=device, dtype=r.dtype).view(1, nb)
        bo = torch.argsort(r, dim=1)
        off = torch.arange(self.block, device=device).view(1, 1, -1)
        return ((bo * self.block).unsqueeze(-1) + off).reshape(n, dim)

    # ------------------------------------------------------------------
    def sample(self, x1: Tensor, a: Tensor, generator=None) -> tuple[Tensor, Tensor]:
        """Mask coordinates, each revealed with probability a (B,).

        With ``block == 1`` every coordinate is masked independently, which is
        the context distribution of a uniform reveal order.  With ``block > 1``
        the context is a *prefix* of a freshly drawn block order instead, which
        is what the block sampler actually visits; masking independently there
        would train the heads on contexts that never occur.

        Guarantees at least one masked coordinate per sample.
        Returns (xt, mask) with mask = True at masked coordinates.
        """
        B, D = x1.shape
        a = a.clamp(0.0, 1.0 - 1.0 / D)
        if self.block <= 1:
            u = torch.rand(B, D, device=x1.device, generator=generator)
            mask = u >= a.unsqueeze(-1)
            # force >= 1 masked site: mask the argmax of u where none is masked
            none_masked = ~mask.any(dim=1)
            if none_masked.any():
                top = u.argmax(dim=1)
                mask[none_masked, top[none_masked]] = True
        else:
            order = self._order(B, D, x1.device, generator)
            # how many are revealed is binomial(D, a), the same marginal the
            # independent mask has, so the two agree on how much is shown and
            # differ only in which patterns occur
            k = torch.binomial(
                torch.full_like(a, float(D)), a.clamp(0.0, 1.0)
            ).long().clamp(0, D - 1)
            rank = torch.argsort(order, dim=1)  # position of each coord in the order
            mask = rank >= k.unsqueeze(-1)
        xt = torch.where(mask, torch.full_like(x1, self.mask_token), x1)
        return xt, mask

    # ------------------------------------------------------------------
    @torch.no_grad()
    def unmask_sample(
        self,
        logits_fn,
        n: int,
        dim: int,
        n_steps: int | None = None,
        device: str | torch.device = "cpu",
        generator=None,
        n_single: int = 0,
    ) -> tuple[Tensor, Tensor]:
        """Any-order autoregressive generation.

        logits_fn(x, a) -> (B, D, N) logits over real tokens; `a` is the
        current reveal fraction (B,).  Reveals coordinates in a uniformly
        random order over `n_steps` network calls, front-loading `n_single`
        one-at-a-time reveals (see `reveal_edges`).  Returns (x, log_q) with
        log_q the exact proposal log-density given the order.
        """
        D = dim
        n_steps = n_steps or D
        x = torch.full((n, D), self.mask_token, dtype=torch.long, device=device)
        order = self._order(n, D, device, generator)
        log_q = torch.zeros(n, device=device)
        bidx = torch.arange(n, device=device)
        bounds = reveal_edges(D, n_steps, n_single)
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            if hi == lo:
                continue
            a = torch.full((n,), lo / D, device=device)
            logits = logits_fn(x, a).log_softmax(dim=-1)  # (B, D, N)
            idx = order[:, lo:hi]  # (B, k)
            lg = logits[bidx.unsqueeze(-1), idx]  # (B, k, N)
            tok = torch.distributions.Categorical(logits=lg).sample()  # (B, k)
            log_q += torch.gather(lg, 2, tok.unsqueeze(-1)).squeeze(-1).sum(dim=1)
            x = x.scatter(1, idx, tok)
        return x, log_q

    # ------------------------------------------------------------------
    @torch.no_grad()
    def unmask_sample_constrained(
        self,
        logits_fn,
        n_B: Tensor,
        dim: int,
        device: str | torch.device = "cpu",
        generator=None,
    ) -> tuple[Tensor, Tensor]:
        """Any-order generation with the number of `1` tokens pinned to `n_B`.

        This is JANUS's fixed-composition sampling: the reveal steps are
        restricted so that exactly `n_B` sites end up carrying species B, and
        the rung composition is what the network is conditioned on.  Inside a
        miscibility gap the semi-grand ensemble puts exponentially little
        mass on the interesting compositions, so a `delta_mu`-conditioned
        model cannot be *asked* about them; a rung-conditioned one can.

        The restriction is one scalar on the logit.  Revealing one site at a
        time, with `r` sites still masked and `b` ones still to place, a `1`
        leaves `C(r-1, b-1)` ways to finish and a `0` leaves `C(r-1, b)`, so
        weighting the two by their completion counts gives

            log q(1) - log q(0) = logit(1) - logit(0) + log b - log(r - b)

        after renormalising.  Feasibility is automatic: `b = 0` sends the
        `1` branch to `-inf` and `b = r` the `0` branch, so the walk can
        never paint itself into a corner and no rejection is needed.

        Two properties are worth stating because they are what make the
        estimator downstream honest.  The density is exact -- `log_q` is the
        realised log-probability under this proposal, not an approximation of
        one -- so `log nu(x) - log q(x)` is an ordinary importance weight
        inside the rung.  And on a uniform network the product telescopes to
        `1 / C(D, n_B)`, the uniform distribution over the rung, which is the
        right answer for a model that knows nothing and is the unit test in
        `tests/test_constrained.py`.

        One site per network call, always: a parallel reveal would have to
        weight the block by a Poisson-binomial convolution to stay exact, and
        the alloy cells here are small enough that `D` calls is cheap.
        """
        if self.N != 2:
            raise ValueError(
                f"the composition constraint is binary; alphabet is {self.N}")
        D = int(dim)
        n_B = torch.as_tensor(n_B, device=device).reshape(-1).long()
        n = n_B.numel()
        if int(n_B.min()) < 0 or int(n_B.max()) > D:
            raise ValueError(f"n_B outside [0, {D}]")
        x = torch.full((n, D), self.mask_token, dtype=torch.long, device=device)
        order = self._order(n, D, device, generator)
        log_q = torch.zeros(n, device=device)
        bidx = torch.arange(n, device=device)
        b = n_B.clone()
        for t in range(D):
            r = D - t                       # masked sites left, this one included
            site = order[:, t]
            lg = logits_fn(x, torch.full((n,), t / D, device=device))
            lg = lg[bidx, site].log_softmax(dim=-1)          # (n, 2)
            counts = torch.stack([(r - b), b], dim=-1).to(lg.dtype)
            lg = (lg + torch.log(counts)).log_softmax(dim=-1)
            tok = torch.distributions.Categorical(logits=lg).sample()
            log_q = log_q + lg.gather(1, tok.unsqueeze(-1)).squeeze(-1)
            x[bidx, site] = tok
            b = b - tok
        bad = int((x == 1).sum(dim=1).sub(n_B).abs().max())
        assert bad == 0, f"constrained decode missed its composition by {bad}"
        return x, log_q

    # ------------------------------------------------------------------
    def ce_loss(self, logits: Tensor, heat_bath1: Tensor, mask: Tensor,
                w: Tensor | None = None) -> Tensor:
        """Soft cross-entropy against heat-bath labels at masked sites.

        `w`, when given, are self-normalized per-endpoint importance weights
        (they sum to one): the per-endpoint loss is then averaged over that
        endpoint's masked sites and the batch is combined with `w` instead of
        uniformly."""
        logp = logits.log_softmax(dim=-1)
        ce = -(heat_bath1 * logp).sum(dim=-1)  # (B, D)
        m = mask.float()
        if w is None:
            return (ce * m).sum() / m.sum().clamp_min(1.0)
        per = (ce * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)  # (B,)
        return (w * per).sum()

    def tsi_loss(self, logits: Tensor, target_score1: Tensor, mask: Tensor) -> Tensor:
        """Generalized-KL Bregman regression of exp(logits) onto the raw
        target score ratios at masked sites.  Minimizer:
        exp(logits) = E[s_nu(x1)_{d,n} | xt] = N * q^d(n | xt)."""
        breg = torch.exp(logits) - target_score1 * logits  # (B, D, N)
        m = mask.float().unsqueeze(-1)
        return (breg * m).sum() / (m.sum() * logits.shape[-1]).clamp_min(1.0)
