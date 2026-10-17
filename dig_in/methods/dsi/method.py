"""DIG-IN and the discrete stochastic interpolant samplers it belongs to.

DIG-IN is `noise='masked', loss='ce'` (configs/method/dsi_masked_ce.yaml),
with Gibbs (heat-bath) labels, proximal damping and tempered importance
weights.  MDNS and PDNS, as compared in the paper, are the same sampler with
`labels='hard'` and no damping.  The class covers four variants via two
switches:

    noise = 'uniform' | 'masked'      (mixture interpolant vs absorbing mask)
    loss  = 'ce'      | 'tsi'         (soft cross-entropy on heat-bath-tilted
                                       labels vs generalized-KL Bregman
                                       regression on target-score-identity
                                       targets)

All variants share the data-free fixed-point loop, one round of which is

  A. exploration    self-sample terminal states with the EMA model
  B. target labels  the exact local energy-difference table at those states
  C. replay push    store (x1, log s_nu(x1)); neither depends on t or on the
                    noising pattern, so both are reused
  D. optimisation   `rollout_every` SGD steps on freshly noised pairs
  E. refresh        the updated denoiser becomes the next sampler

The target is an exact fixed point of this map.  Proximal damping (eta < 1)
blends the targets with the EMA model's own prediction.

Labels are cached at push time (`cache_labels`) rather than recomputed per
gradient step -- they are a function of the endpoint alone, so this is an
identity, and it is what makes the exploration cost amortise over many
cheap inner steps.  The inverse-temperature ramp is applied at use time, so
caching stays compatible with annealing.

Unbiased log Z: uniform variants via forward-backward RND tau-leaping
weights; masked variants via the tractable any-order AR proposal density.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from dig_in.methods.base import Method
from dig_in.methods.dsi.interpolant import MixtureInterpolant
from dig_in.methods.dsi.masked import MaskedInterpolant
from dig_in.methods.dsi.sampling import log_weights_uniform_prior, tau_leap
from dig_in.models import build_model
from dig_in.utils.acquisition import FisherAcquisition, mixed_cond
from dig_in.utils.buffer import ReplayBuffer
from dig_in.utils.tempering import (centre_log_w_by_buffer,
                                 centre_log_w_by_cell, regress_out_cond,
                                 tempered_weights,
                                 tempered_weights_local,
                                 tempered_weights_per_cell)
from dig_in.utils.ema import EMA
from dig_in.utils.gibbs import heat_bath_sweeps

_EPS = 1e-12


def bregman_kl(logit: Tensor, target: Tensor, mask: Tensor | None = None) -> Tensor:
    """Generalized-KL Bregman divergence D(target || exp(logit)) up to a
    target-only constant; minimizer of the free logit is log E[target]."""
    val = torch.exp(logit) - target * logit
    if mask is not None:
        return (val * mask).sum() / mask.sum().clamp_min(1.0)
    return val.mean()


class DSI(Method):
    def __init__(
        self,
        target,
        model: dict,
        noise: str = "uniform",
        loss: str = "ce",
        schedule: str = "quadratic",
        # sampling
        zeta_train: float = 4.0,
        zeta_eval: float = 0.0,
        n_steps_train: int = 40,
        n_steps_eval: int = 80,
        unmask_steps: int | None = None,
        unmask_single: int = 0,
        unmask_block: int | None = None,
        corrector_steps: int = 0,
        corrector_delta: float = 0.5,
        corrector_g: str = "mh",
        time_input_masked: bool = False,
        # optimization
        batch_size: int = 128,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        grad_clip: float = 0.0,
        ema_decay: float = 0.999,
        eta: float = 1.0,
        clamp_lo: float = 1e-3,
        clamp_hi: float = 1e3,
        # fixed-point loop
        buffer_capacity: int = 4096,
        n_rollout: int = 512,
        rollout_every: int = 20,
        mode_seed_frac: float = 0.0,
        weighting: str = "none",
        target_ess: float = 0.3,
        weight_bins=(8, 8),
        weight_min_count: int = 8,
        weight_per_cell: bool = False,
        acq_frac_uniform: float = 1.0,
        acq_bins: tuple[int, int] = (16, 16),
        acq_ema: float = 0.9,
        acq_floor: float = 0.05,
        acq_power: float = 0.25,
        warm_start: int = 0,
        warm_start_kind: str = "uniform",
        warm_start_burn_in: int = 200,
        warm_start_thin: int = 4,
        cache_labels: bool = True,
        labels: str = "soft",
        curriculum_steps: int = 0,
        curriculum_start: float = 1.0 / 3.0,
        buffer_relax: int = 0,
        symmetrize: bool = False,
        anneal_steps: int = 0,
        anneal_start: float = 0.5,
        eval_chunk: int = 512,
        device: str = "cpu",
    ):
        super().__init__(target, device)
        assert noise in ("uniform", "masked") and loss in ("ce", "tsi")
        self.noise, self.loss_kind = noise, loss
        N, D = target.N, target.D

        if noise == "uniform":
            self.interp = MixtureInterpolant(N, schedule=schedule)
            vocab, out_ch = N, (N if loss == "ce" else 3 * N)
            require_time = True
        else:
            # A fluid's coordinates come in groups of `dim` per particle, and
            # revealing a whole particle at a time turns every context into a
            # sequential-insertion problem rather than a scattered one (see
            # MaskedInterpolant).  Targets that declare no grouping keep 1, so
            # the lattice runs are byte-identical.
            blk = int(unmask_block or getattr(target, "reveal_block", 1) or 1)
            self.unmask_block = blk
            self.unmask_stage = getattr(target, "block_stage", None)
            # a target may also declare the *stage* of each block, which makes
            # the reveal coarse-to-fine across the whole configuration rather
            # than within one coordinate; see CoarseToFineLJBox
            self.minterp = MaskedInterpolant(
                N, block=blk, stage=getattr(target, "block_stage", None))
            vocab, out_ch = N + 1, N
            require_time = time_input_masked

        # An amortized target is a *family*: every endpoint carries its own
        # thermodynamic parameters, which condition both the network and the
        # energy the labels are computed from.
        self.amortized = bool(getattr(target, "amortized", False))
        model = dict(model)
        name = model.pop("name")
        if self.amortized:
            model.setdefault("cond_ranges", list(target.cond_ranges))
        # A binned continuous coordinate on a periodic box has a CIRCULAR
        # alphabet: bin values carry a metric that a free embedding table would
        # have to rediscover symbol by symbol.  Targets that declare
        # `cyclic = True` hand the period to the transformer, which adds the
        # low harmonics of the bin angle to its token embedding.  Lattice-spin
        # targets leave it at 0 and are untouched.
        if getattr(target, "cyclic", False) and name in (
            "transformer", "vit", "sit", "ropesit"
        ):
            model.setdefault("token_cyclic", target.N)
        elif getattr(target, "token_ordered", 0) and name in (
            "transformer", "vit", "sit", "ropesit"
        ):
            # ordered but not periodic: a binned displacement inside a cage
            model.setdefault("token_ordered", int(target.token_ordered))
        if (getattr(target, "site_coords_frac", None) is not None
                and name in ("transformer", "vit", "sit", "ropesit")):
            model.setdefault("site_coords", target.site_coords_frac)
            model.setdefault("pos_embed", "learned")
        try:
            self.net = build_model(
                name, target.lattice_shape, vocab, out_ch,
                require_time=require_time, **model
            ).to(self.device)
        except TypeError as e:
            if self.amortized and "cond_ranges" in str(e):
                raise TypeError(
                    f"model '{name}' does not accept amortized conditioning; "
                    f"an amortized target needs a model that takes "
                    f"`cond_ranges` (e.g. model=rescnn)"
                ) from e
            raise
        self.require_time = require_time

        self.opt = torch.optim.AdamW(
            self.net.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.ema = EMA(self.net.parameters(), decay=ema_decay)
        self.buf = ReplayBuffer(buffer_capacity)

        self.zeta_train, self.zeta_eval = zeta_train, zeta_eval
        self.n_steps_train, self.n_steps_eval = n_steps_train, n_steps_eval
        self.unmask_steps = unmask_steps or D
        self.unmask_single = unmask_single
        self.corrector_steps = corrector_steps
        self.corrector_delta = corrector_delta
        self.corrector_g = corrector_g
        self.batch_size = batch_size
        self.grad_clip = grad_clip
        self.eta = eta
        self.clamp_lo, self.clamp_hi = clamp_lo, clamp_hi
        self.n_rollout = n_rollout
        self.rollout_every = rollout_every
        self.mode_seed_frac = mode_seed_frac
        if weighting not in ("none", "tempered", "tempered_grouped",
                             "tempered_regressed", "tempered_buffer",
                             "tempered_local"):
            raise ValueError(f"unknown weighting '{weighting}'")
        if weighting in ("tempered", "tempered_grouped",
                         "tempered_regressed", "tempered_buffer",
                         "tempered_local"):
            # the weight of a buffered endpoint is nu_hat(x) / q(x) with q the
            # density of the draw that produced it, so anything that alters an
            # endpoint after it was drawn invalidates its stored log q
            if noise != "masked":
                raise ValueError("weighting='tempered' needs the masked "
                                 "sampler's tractable density")
            if buffer_relax > 0 or mode_seed_frac > 0:
                raise ValueError("weighting='tempered' is incompatible with "
                                 "buffer_relax > 0 / mode_seed_frac > 0")
        self.weighting = weighting
        self.target_ess = target_ess
        self.weight_bins = tuple(weight_bins)
        self.weight_min_count = weight_min_count
        # tempered_buffer bins on the acquisition's own grid, which exists
        # whether or not the acquisition itself is switched on
        self.acq_bins = tuple(acq_bins)
        self.weight_per_cell = bool(weight_per_cell)
        # Observable-free Fisher acquisition over the conditioning box.
        # frac_uniform = 1.0 leaves the family's uniform draw untouched, which
        # is what every non-amortized run and the whole Ising benchmark use.
        self.acq_frac_uniform = float(acq_frac_uniform)
        self.acq = None
        if (self.acq_frac_uniform < 1.0 and getattr(target, "amortized", False)
                and hasattr(target, "acq_ranges")):
            self.acq = FisherAcquisition(
                target.acq_ranges, bins=tuple(acq_bins), ema=acq_ema,
                floor=acq_floor, power=acq_power, device=self.device,
            )
        if warm_start_kind not in ("uniform", "mode", "mcmc"):
            raise ValueError(
                f"unknown warm_start_kind '{warm_start_kind}'; "
                f"expected uniform, mode or mcmc")
        if warm_start_kind == "mcmc" and self.weighting != "none":
            # a configuration the model did not draw has no log q to weight it
            # by, which is the same reason buffer_relax is refused above
            raise ValueError(
                "warm_start_kind='mcmc' needs weighting='none': a buffer entry "
                "the model did not draw has no proposal density, so it can "
                "carry a label but not a weight")
        self.warm_start_kind = warm_start_kind
        self.warm_start_burn_in = int(warm_start_burn_in)
        self.warm_start_thin = int(warm_start_thin)
        self.cache_labels = cache_labels
        # hard = the endpoint's own token: the soft label with the heat-bath
        # conditional replaced by the one sample it is an expectation over
        if labels not in ("soft", "hard"):
            raise ValueError(f"unknown labels '{labels}'; expected soft or hard")
        if labels == "hard" and (noise != "masked" or loss != "ce"):
            raise ValueError("labels='hard' is implemented for the masked "
                             "sampler with loss='ce' only")
        self.labels = labels
        self.curriculum_steps = curriculum_steps
        self.curriculum_start = curriculum_start
        self.buffer_relax = buffer_relax
        if symmetrize and getattr(target, "h", 0.0) != 0.0:
            print("[DSI] symmetrize disabled: target has a nonzero field")
            symmetrize = False
        # The global flip changes the number of `1` tokens, so on a
        # composition-constrained family it maps every endpoint off its own
        # rung -- wrong in a way that no metric on the rung would report,
        # because the augmented endpoint is a valid configuration of a
        # different one.
        if symmetrize and getattr(target, "composition_constrained", False):
            print("[DSI] symmetrize disabled: the flip changes the composition")
            symmetrize = False
        # An amortized family whose field range is not the single point 0
        # contains members with h != 0, where the spin flip is not a symmetry
        # of the energy and the augmentation would be wrong, not merely
        # uninformative.
        if symmetrize and self.amortized and tuple(target.dmu_range) != (0.0, 0.0):
            print("[DSI] symmetrize disabled: the family spans a nonzero field")
            symmetrize = False
        self.symmetrize = symmetrize
        self.anneal_steps = anneal_steps
        self.anneal_start = anneal_start
        self.eval_chunk = eval_chunk
        self.n_skipped = 0  # updates refused for a non-finite gradient

        # Pre-fill.  The default is high-entropy (uniformly random) endpoints:
        # the fixed-point loop must not start out already committed to a mode,
        # and constant "mode" seeds are zero-temperature configurations, i.e.
        # exactly such a commitment.  Keep `mode` only as an ablation.
        if warm_start > 0:
            for _ in range(warm_start):
                c = self._draw_cond(n_rollout)
                self._push(self._warm_states(n_rollout, c), c)

    # ------------------------------------------------------------------
    # Field / posterior evaluation
    # ------------------------------------------------------------------
    def _time(self, t: Tensor | None) -> Tensor | None:
        return t if self.require_time else None

    def _net(self, x: Tensor, t: Tensor | None, cond: Tensor | None = None) -> Tensor:
        """Network call, with the amortized conditioning when there is any.

        Non-amortized runs must not pass a third argument: the other
        backbones in the zoo take (x, t) only.
        """
        if not self.amortized:
            return self.net(x, t)
        # the family decides what the network is conditioned on; it need not
        # be the physical coordinates the buffer and the evaluation use
        return self.net(x, t, self.target.to_cond_features(cond))

    def _curriculum_T_lo(self) -> float | None:
        """Cold end of the conditioning box at the current step.

        The ordered phase is switched on gradually from the disordered end,
        so a newly reached temperature starts from the conditioned network's
        extrapolation of the warmer ones rather than from the uniform
        denoiser.  Order below T_c has to grow a correlation length across
        the lattice, and demanding that at every temperature from step zero
        is what the curriculum avoids.
        """
        n = self.curriculum_steps
        if not n or not self.amortized:
            return None
        lo, hi = self.target.T_range
        frac = min(1.0, self.step_count / n)
        active = self.curriculum_start + (1.0 - self.curriculum_start) * frac
        return hi - active * (hi - lo)

    def _draw_cond(self, n: int) -> Tensor | None:
        """Per-sample thermodynamic parameters, continuous over the family."""
        if not self.amortized:
            return None
        return mixed_cond(
            self.target, n, self.acq, frac_uniform=self.acq_frac_uniform,
            device=self.device, T_lo=self._curriculum_T_lo(),
        )

    def _fields(self, x: Tensor, t: Tensor, cond: Tensor | None = None):
        """(v, vb, s) fields for the uniform-noise sampler."""
        out = self._net(x, t, cond)
        if self.loss_kind == "ce":
            log_q = self.interp.pair_log_posterior(out, x, t)
            return self.interp.posterior_fields(log_q, x, t)
        N = self.target.N
        logv, logvb, logs = out.split(N, dim=-1)
        clamp = math.log(self.clamp_hi)
        return (
            torch.exp(logv.clamp(max=clamp)),
            torch.exp(logvb.clamp(max=clamp)),
            torch.exp(logs.clamp(max=clamp)),
        )

    def _masked_logits(
        self, x: Tensor, a: Tensor, cond: Tensor | None = None
    ) -> Tensor:
        return self._net(x, self._time(a), cond)

    # ------------------------------------------------------------------
    # Rollouts / buffer
    # ------------------------------------------------------------------
    def _seed_states(self, n: int) -> Tensor:
        """Constant configurations with a uniformly random value (mode seeds
        double as a zero-temperature init)."""
        vals = torch.randint(0, self.target.N, (n, 1), device=self.device)
        return vals.expand(n, self.target.D).contiguous()

    def _warm_states(self, n: int, cond: Tensor | None = None) -> Tensor:
        """Pre-fill endpoints: high-entropy by default (see __init__).

        `mcmc` draws them from the target's own sampler instead, so the loop
        starts on configurations the target actually has mass on.  The chains
        are short by design: the point is not an equilibrium set but a region,
        and the label channel is exact at whatever configuration it is handed.

        A composition-constrained family gets the highest-entropy state *on
        its own rung*: a free uniform draw sits at composition one half and
        would teach the network that every rung looks like that one.
        """
        n_B_of = getattr(self.target, "n_B_of", None)
        if (self.warm_start_kind == "uniform" and n_B_of is not None
                and cond is not None):
            D = self.target.D
            perm = torch.rand(n, D, device=self.device).argsort(dim=1)
            k = torch.arange(D, device=self.device).view(1, -1)
            first = (k < n_B_of(cond).view(-1, 1)).long()
            return torch.zeros(n, D, dtype=torch.long,
                               device=self.device).scatter(1, perm, first)
        if self.warm_start_kind == "uniform":
            return self.target.sample_uniform(n, device=self.device)
        if self.warm_start_kind == "mcmc":
            from dig_in.targets import get_mcmc

            mc = get_mcmc(self.target, device=self.device)
            return mc.sample(n, n_chains=min(n, 512),
                             burn_in=self.warm_start_burn_in,
                             thin=self.warm_start_thin).to(self.device)
        return self._seed_states(n)

    @torch.no_grad()
    def _push(self, x: Tensor, cond: Tensor | None = None,
              log_q: Tensor | None = None):
        """Replay push: store the endpoint together with its exact local
        energy-difference table, which is a function of the endpoint (and,
        when amortized, of its own state point) alone and so is reusable
        across every t and noising pattern drawn from it."""
        extras = {}
        weighted = self.weighting in ("tempered", "tempered_grouped",
                                      "tempered_regressed",
                                      "tempered_buffer", "tempered_local")
        # a hard label is the endpoint itself, so there is no table to store
        cache = self.cache_labels and self.labels == "soft"
        if cache and weighted:
            # both quantities come from the same cluster products, so ask for
            # them together: separately this endpoint costs two passes over
            # the target, fused it costs one
            extras["log_p"], extras["log_score"] =                 self.target.log_prob_and_score(x, cond)
        elif cache:
            extras["log_score"] = self.target.log_score(x, cond)
        if self.amortized:
            # the state point travels with the endpoint: a buffered sample is
            # only a valid training pair together with the parameters it was
            # generated at
            extras["cond"] = cond
        if weighted:
            # log nu_hat(x) and log q(x) of the draw that produced it; the
            # warm-start prefill is drawn uniformly, whose density is exact
            if log_q is None:
                log_q = torch.full((x.shape[0],),
                                   -self.target.D * math.log(self.target.N),
                                   device=x.device)
            extras["log_q"] = log_q
            if "log_p" not in extras:
                # no cached labels, so nothing to fuse with
                extras["log_p"] = self.target.log_prob(x, cond)
        self.buf.add(x, **extras)

    @torch.no_grad()
    def _rollout(self, n: int, cond: Tensor | None = None):
        if cond is None:
            cond = self._draw_cond(n)
        if self.noise == "uniform":
            x0 = self.target.sample_uniform(n, device=self.device)
            x, _ = tau_leap(
                lambda xx, tt: self._fields(xx, tt, cond),
                x0, self.n_steps_train, self.zeta_train, self.target.N,
            )
            log_q = None
        else:
            x, log_q = self._unmask(cond, n)
        return x, cond, log_q

    def _unmask(self, cond: Tensor | None, n: int):
        """One batch of the masked sampler, constrained if the target asks.

        A composition-constrained family (`AlloyCanonical`) pins the number of
        `1` tokens to the rung it conditioned on, which is what lets a model
        be asked about the middle of a miscibility gap at all.  Everything
        else -- the network, the labels, the loss -- is unchanged, so this is
        the only place the constraint enters training.
        """
        n_B_of = getattr(self.target, "n_B_of", None)
        if n_B_of is not None and cond is not None:
            return self.minterp.unmask_sample_constrained(
                lambda xx, aa: self._masked_logits(xx, aa, cond),
                n_B_of(cond), self.target.D, device=self.device,
            )
        return self.minterp.unmask_sample(
            lambda xx, aa: self._masked_logits(xx, aa, cond),
            n, self.target.D,
            n_steps=self.unmask_steps, device=self.device,
            n_single=self.unmask_single,
        )

    @torch.no_grad()
    def _refresh_buffer(self):
        params = list(self.net.parameters())
        self.ema.store(params)
        self.ema.copy_to(params)
        try:
            x, cond, log_q = self._rollout(self.n_rollout)
        finally:
            self.ema.restore(params)
        if self.mode_seed_frac > 0:
            k = int(self.mode_seed_frac * self.n_rollout)
            if k:
                x = torch.cat([x[: self.n_rollout - k], self._seed_states(k)], 0)
        if self.acq is not None and cond is not None:
            # the endpoints just drawn are samples from q(. | theta) at their
            # own theta, which is exactly what the Fisher trace averages over
            self.acq.update(self.target.cond_to_acq(cond),
                            self.target.cond_score(x, cond))
        if self.buffer_relax > 0:
            x = heat_bath_sweeps(
                self.target, x, self.buffer_relax, scale=self.anneal_scale(),
                cond=cond,
            )
        self._push(x, cond, log_q)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def _random_symmetry(
        self, x: Tensor, log_score: Tensor | None = None
    ) -> tuple[Tensor, Tensor | None]:
        """Per-sample uniformly random relabeling of the alphabet (the Ising
        spin flip and Potts color permutations leave the energy invariant
        at zero field).

        A cached label table has to be relabeled with it: energy invariance
        gives log s(perm(x))[d, perm(n)] = log s(x)[d, n], so the table is
        scattered along the alphabet axis by the same permutation.  Relabeling
        the states but not their labels would silently train on mismatched
        targets.
        """
        B, N = x.shape[0], self.target.N
        perms = torch.argsort(torch.rand(B, N, device=x.device), dim=1)
        xs = torch.gather(perms, 1, x)
        if log_score is None:
            return xs, None
        idx = perms[:, None, :].expand(B, self.target.D, N)
        return xs, torch.zeros_like(log_score).scatter_(2, idx, log_score)

    def step(self) -> dict:
        if self.step_count % self.rollout_every == 0 or len(self.buf) == 0:
            self._refresh_buffer()
        scale = self.anneal_scale()
        x1, extras = self.buf.sample(self.batch_size)
        log_s1 = extras.get("log_score")
        cond = extras.get("cond")
        if self.symmetrize:
            x1, log_s1 = self._random_symmetry(x1, log_s1)
        wbar = None
        temper = {}
        if self.weighting in ("tempered", "tempered_grouped",
                              "tempered_regressed", "tempered_buffer",
                             "tempered_local"):
            # w = nu_hat^scale(x) / q(x), tempered to the largest exponent
            # whose self-normalized weights still keep the target ESS
            log_w = scale * extras["log_p"] - extras["log_q"]
            raw_std = log_w.std().item()
            frac = 1.0
            wbar = None
            nu_sd = None
            if self.weighting == "tempered_local":
                # the rule runs inside each cell, so nothing is centred: the
                # ESS ratio is shift invariant and the offset cancels
                be = self.buf.extras
                to_acq = getattr(self.target, "cond_to_acq", None)
                if to_acq is not None and "log_p" in be and "cond" in be:
                    ref_lw = scale * be["log_p"] - be["log_q"]
                    wbar, nu, ess, nu_sd = tempered_weights_local(
                        log_w, to_acq(cond), ref_lw, to_acq(be["cond"]),
                        self.target.acq_ranges, self.target_ess,
                        bins=self.weight_bins,
                        min_count=max(self.weight_min_count, 24))
                    frac = 1.0
            elif self.weighting == "tempered_buffer":
                # the offset is a property of the conditioning, so estimate it
                # from every endpoint in the buffer rather than from the few
                # in this batch, on the acquisition's own grid
                be = self.buf.extras
                to_acq = getattr(self.target, "cond_to_acq", None)
                if to_acq is not None and "log_p" in be and "cond" in be:
                    ref_lw = scale * be["log_p"] - be["log_q"]
                    log_w, frac = centre_log_w_by_buffer(
                        log_w, to_acq(cond), ref_lw, to_acq(be["cond"]),
                        self.target.acq_ranges, bins=self.acq_bins,
                        min_count=self.weight_min_count)
                else:
                    frac = 0.0
            elif self.weighting == "tempered_regressed":
                feats = (self.target.to_cond_features(cond)
                         if cond is not None else None)
                log_w, removed = regress_out_cond(log_w, feats)
                frac = removed
            elif self.weighting == "tempered_grouped":
                # strip the between-state-point offset first: across the
                # conditioning box it is hundreds of nats and swamps the
                # within-point signal the weights are for
                # bin in the NORMALISED coordinates the network sees, not
                # in raw (T, delta_mu): cond_ranges describes the former, and
                # the offset is smooth in beta rather than in T
                feats = (self.target.to_cond_features(cond)
                         if cond is not None else None)
                log_w, frac = centre_log_w_by_cell(
                    log_w, feats, self.target.cond_ranges,
                    bins=self.weight_bins,
                    min_count=self.weight_min_count)
            cell_frac = None
            if wbar is not None:
                pass
            elif self.weight_per_cell and cond is not None:
                to_acq = getattr(self.target, "cond_to_acq", None)
                if to_acq is not None:
                    wbar, nu, ess, cell_frac = tempered_weights_per_cell(
                        log_w, to_acq(cond), self.target.acq_ranges,
                        self.target_ess, bins=self.weight_bins,
                        min_count=self.weight_min_count)
            if wbar is None and cell_frac is None:
                wbar, nu, ess = tempered_weights(log_w, self.target_ess)
            temper = {"temper_nu": nu, "temper_ess": ess,
                      "log_w_std": log_w.std().item(),
                      "log_w_std_raw": raw_std, "temper_cell_frac": frac}
            if cell_frac is not None:
                temper["temper_norm_frac"] = cell_frac
            if nu_sd is not None:
                temper["temper_nu_sd"] = nu_sd
        loss = (
            self._step_uniform(x1, scale, log_s1, cond)
            if self.noise == "uniform"
            else self._step_masked(x1, scale, log_s1, cond, wbar)
        )
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        # The Bregman objective carries an unbounded exp(logit) term, so a
        # single pathological batch can produce an infinite gradient; applying
        # it would leave NaN weights and the next rollout would then die inside
        # the sampler with an unhelpful device-side assert.  Refuse the update
        # instead -- a no-op whenever the gradient is finite.
        gnorm = torch.nn.utils.clip_grad_norm_(
            self.net.parameters(), self.grad_clip if self.grad_clip > 0 else math.inf
        )
        if not torch.isfinite(gnorm):
            self.opt.zero_grad(set_to_none=True)
            self.n_skipped += 1
        else:
            self.opt.step()
            self.ema.update(self.net.parameters())
        self.step_count += 1
        return {
            "loss": loss.item(),
            "grad_norm": gnorm.item(),
            "skipped_updates": self.n_skipped,
            "anneal_scale": scale,
            **temper,
            **({"curriculum_T_lo": self._curriculum_T_lo()}
               if self.curriculum_steps and self.amortized else {}),
        }

    def _anchor_out(
        self, x: Tensor, t: Tensor | None, cond: Tensor | None = None
    ) -> Tensor:
        """EMA-model forward pass (the proximal anchor)."""
        params = list(self.net.parameters())
        with torch.no_grad():
            self.ema.store(params)
            self.ema.copy_to(params)
            try:
                return self._net(x, t, cond)
            finally:
                self.ema.restore(params)

    def _log_score(
        self, x1: Tensor, cached: Tensor | None, cond: Tensor | None = None
    ) -> Tensor:
        """Cached label table for the buffered endpoints, or a fresh one."""
        return self.target.log_score(x1, cond) if cached is None else cached

    def _step_uniform(
        self, x1: Tensor, scale: float, log_s1: Tensor | None = None,
        cond: Tensor | None = None,
    ) -> Tensor:
        B = x1.shape[0]
        log_s1 = self._log_score(x1, log_s1, cond)
        x0 = self.target.sample_uniform(B, device=self.device)
        t = torch.rand(B, device=self.device)
        xt = self.interp.sample(x0, x1, t)
        # NOTE: anchor forward passes (EMA copy_to/restore mutate parameters
        # in place) must happen before the live forward pass builds its graph.

        if self.loss_kind == "ce":
            with torch.no_grad():
                hb1 = torch.softmax(scale * log_s1, dim=-1)
                lbl = self.interp.soft_labels(xt, x1, t, hb1)
                if self.eta < 1.0:
                    q_anchor = self.interp.pair_log_posterior(
                        self._anchor_out(xt, t, cond), xt, t
                    ).exp()
                    lbl = self.eta * lbl + (1.0 - self.eta) * q_anchor
            log_q = self.interp.pair_log_posterior(
                self._net(xt, t, cond), xt, t
            )
            return -(lbl * log_q).sum(dim=(2, 3)).mean()

        # Bregman + target score identity (rate route)
        N = self.target.N
        with torch.no_grad():
            T_v, T_vb = self.interp.conditional_targets(xt, x0, x1, t)
            score1 = torch.exp(scale * log_s1)
            T_s = self.interp.score_target_gsi(xt, x1, score1)
            T_v = T_v.clamp(self.clamp_lo, self.clamp_hi)
            T_vb = T_vb.clamp(self.clamp_lo, self.clamp_hi)
            T_s = T_s.clamp(self.clamp_lo, self.clamp_hi)
            if self.eta < 1.0:
                a_v, a_vb, a_s = self._anchor_out(xt, t, cond).split(N, dim=-1)
                clamp = math.log(self.clamp_hi)
                T_v = self.eta * T_v + (1 - self.eta) * a_v.clamp(max=clamp).exp()
                T_vb = self.eta * T_vb + (1 - self.eta) * a_vb.clamp(max=clamp).exp()
                T_s = self.eta * T_s + (1 - self.eta) * a_s.clamp(max=clamp).exp()
        logv, logvb, logs = self._net(xt, t, cond).split(N, dim=-1)
        offdiag = torch.ones_like(logv).scatter(
            2, xt.unsqueeze(-1), torch.zeros_like(xt, dtype=logv.dtype).unsqueeze(-1)
        )
        return (
            bregman_kl(logv, T_v, offdiag)
            + bregman_kl(logvb, T_vb, offdiag)
            + bregman_kl(logs, T_s, offdiag)
        )

    def _step_masked(
        self, x1: Tensor, scale: float, log_s1: Tensor | None = None,
        cond: Tensor | None = None, w: Tensor | None = None,
    ) -> Tensor:
        B = x1.shape[0]
        if self.labels == "soft":
            log_s1 = self._log_score(x1, log_s1, cond)
        a = torch.rand(B, device=self.device)
        xt, mask = self.minterp.sample(x1, a)
        # anchor passes must precede the live forward (see _step_uniform)

        if self.loss_kind == "ce":
            with torch.no_grad():
                if self.labels == "hard":
                    hb1 = F.one_hot(x1, self.target.N).float()
                else:
                    hb1 = torch.softmax(scale * log_s1, dim=-1)
                if self.eta < 1.0:
                    q_anchor = self._anchor_out(
                        xt, self._time(a), cond
                    ).softmax(-1)
                    hb1 = self.eta * hb1 + (1.0 - self.eta) * q_anchor
            return self.minterp.ce_loss(
                self._masked_logits(xt, a, cond), hb1, mask, w
            )

        with torch.no_grad():
            T = torch.exp(scale * log_s1).clamp(self.clamp_lo, self.clamp_hi)
            if self.eta < 1.0:
                a_out = self._anchor_out(xt, self._time(a), cond)
                T = self.eta * T + (1 - self.eta) * a_out.clamp(
                    max=math.log(self.clamp_hi)
                ).exp()
        return self.minterp.tsi_loss(
            self._masked_logits(xt, a, cond), T, mask
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def _eval_cond(self, k: int, cond: Tensor | None) -> Tensor | None:
        """Conditioning for one evaluation chunk.

        `cond` may be a single state point (broadcast to the chunk) or one
        row per sample; None draws from the family, which is the right
        default for a whole-family diagnostic but not for scoring a single
        state point -- the grid evaluation always passes one explicitly.
        """
        if not self.amortized:
            return None
        if cond is None:
            return self.target.sample_cond(k, device=self.device)
        # the width is the family's own: 2 for the (T, delta_mu) families,
        # 1 for a family amortized over temperature alone
        n_c = int(getattr(self.target, "cond_dim", 2))
        cond = torch.as_tensor(cond, device=self.device).float().view(-1, n_c)
        return cond.expand(k, -1) if cond.shape[0] == 1 else cond[:k]

    @torch.no_grad()
    def sample_with_weights(
        self, n: int, cond: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        params = list(self.net.parameters())
        self.ema.store(params)
        self.ema.copy_to(params)
        try:
            xs, ws = [], []
            for i in range(0, n, self.eval_chunk):
                k = min(self.eval_chunk, n - i)
                c = self._eval_cond(k, cond)
                if self.noise == "uniform":
                    x0 = self.target.sample_uniform(k, device=self.device)
                    x, log_fb = tau_leap(
                        lambda xx, tt: self._fields(xx, tt, c),
                        x0, self.n_steps_eval, self.zeta_eval,
                        self.target.N, with_rnd=True,
                        corrector_steps=self.corrector_steps,
                        corrector_delta=self.corrector_delta,
                        corrector_g=self.corrector_g,
                    )
                    logw = log_weights_uniform_prior(
                        self.target.log_prob(x, c), log_fb,
                        self.target.D, self.target.N,
                    )
                else:
                    x, log_q = self._unmask(c, k)
                    logw = self.target.log_prob(x, c) - log_q
                xs.append(x)
                ws.append(logw)
        finally:
            self.ema.restore(params)
        return torch.cat(xs), torch.cat(ws)

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Sequential Monte Carlo along the reveal order
    # ------------------------------------------------------------------
    def supports_smc(self) -> bool:
        """The any-order sampler adds one condition to the base one.

        The twist is the energy of the *revealed set*, which is only defined
        when the revealed set is a whole number of the target's units -- whole
        particles for a fluid.  A coordinate-at-a-time reveal leaves half a
        particle placed, so it is refused, and this reports that in advance
        instead of raising once per evaluation for the length of a run.
        """
        if self.noise != "masked" or not super().supports_smc():
            return False
        if not hasattr(self.target, "partial_energy_indices"):
            return False
        unit = int(getattr(self.target, "reveal_block", 1) or 1)
        if self.unmask_block % unit:
            return False
        # Divisibility is not enough.  A block has to *be* a unit the partial
        # energy can score, and a coarse-to-fine spelling breaks that while
        # dividing perfectly: its block is one particle at one resolution, so
        # after the first stage no particle is placed and the number of blocks
        # exceeds the number of particles.  Compare the two counts.
        units = int(getattr(self.target, "n_partial_units", 0) or 0)
        if units and self.target.D // unit != units:
            return False
        return True


    @torch.no_grad()
    def sample_with_weights_smc(
        self,
        n: int,
        ess_threshold: float = 0.5,
        cond: Tensor | None = None,
        generator=None,
    ) -> tuple[Tensor, Tensor, dict]:
        """Twisted sequential Monte Carlo over the block reveal order.

        A masked model with a block reveal order is an autoregressive model over
        blocks, so the correction the fixed-order sampler gets applies here
        unchanged: after each block the energy of what has been revealed is
        exact and cheap, the partial Boltzmann weights are the intermediate
        targets, and the incremental weights telescope to the plain weight.
        Resampling therefore changes the variance and nothing else, and at
        ``ess_threshold = 0`` this reduces to plain importance sampling.

        The order is drawn once for the whole batch, so the revealed set is
        shared and its energy is one call.  It is still independent of the
        configuration, so the weights are correct conditional on it.

        Returns ``(x, log_w, info)``; ``info["log_Z"]`` is the SMC estimate
        accumulated across resampling events.
        """
        if self.noise != "masked":
            raise TypeError("sequential resampling needs the masked sampler")
        target = self.target
        if not hasattr(target, "partial_energy_indices"):
            raise TypeError(
                f"{type(target).__name__} has no partial_energy_indices; "
                f"twisted SMC needs the energy of a revealed subset")
        blk = self.unmask_block
        D, N = target.D, target.N
        if D % blk:
            raise ValueError(f"block={blk} does not divide D={D}")
        # `partial_energy_indices` is indexed by the target's own unit -- a
        # particle for a fluid, a site for a lattice -- not by token.  A reveal
        # block that is not a whole number of those units leaves half a particle
        # placed, whose energy is not defined, so refuse rather than silently
        # index the wrong thing.  A coordinate-at-a-time model is exactly that
        # case, and it is the reason this is checked and not assumed.
        unit = int(getattr(target, "reveal_block", 1) or 1)
        if blk % unit:
            raise ValueError(
                f"sequential resampling needs the reveal block to be a whole "
                f"number of the target's units: block={blk} is not a multiple "
                f"of reveal_block={unit}.  Train or sample with "
                f"unmask_block={unit} (or a multiple) to use it.")
        ppb = blk // unit                       # units revealed per block
        nb = D // blk
        dev = self.device

        # one block order for the batch, and the token indices of each block
        stage = getattr(self, "unmask_stage", None)
        if stage is None:
            perm = torch.randperm(nb, device=dev, generator=generator)
        else:
            # same rule as the training mask: ordered by stage, random within
            r = torch.rand(nb, device=dev, generator=generator)
            perm = torch.argsort(r + stage.to(device=dev, dtype=r.dtype))
        off = torch.arange(blk, device=dev)
        tok_of = (perm.view(-1, 1) * blk + off.view(1, -1))       # (nb, blk)
        # the units each block reveals, in the same order
        unit_of = (perm.view(-1, 1) * ppb
                   + torch.arange(ppb, device=dev).view(1, -1))   # (nb, ppb)

        x = torch.full((n, D), self.minterp.mask_token, dtype=torch.long,
                       device=dev)
        prev_e = target.partial_energy_indices(
            x, unit_of[:0].reshape(-1)).double()
        log_w = -target.beta * prev_e
        log_Z, n_resample, ess_trace = 0.0, 0, []

        for k in range(nb):
            a = torch.full((n,), (k * blk) / D, device=dev)
            logits = self._masked_logits(x, a, cond).log_softmax(dim=-1)
            idx = tok_of[k]                                       # (blk,)
            lg = logits[:, idx, :]                                # (n, blk, N)
            tok = torch.distributions.Categorical(logits=lg).sample()
            log_q_block = torch.gather(
                lg, 2, tok.unsqueeze(-1)).squeeze(-1).sum(dim=1).double()
            x = x.index_copy(1, idx, tok)

            e = target.partial_energy_indices(
                x, unit_of[:k + 1].reshape(-1)).double()
            log_w = log_w - target.beta * (e - prev_e) - log_q_block
            prev_e = e

            w = torch.softmax(log_w, 0)
            ess = float(1.0 / (n * (w * w).sum()))
            ess_trace.append(ess)
            last = k == nb - 1
            if not last and ess < ess_threshold:
                log_Z += float(torch.logsumexp(log_w, 0) - math.log(n))
                sel = torch.multinomial(w, n, replacement=True,
                                        generator=generator)
                x = x[sel]
                prev_e = prev_e[sel]
                log_w = torch.zeros(n, device=dev, dtype=torch.float64)
                n_resample += 1

        log_Z += float(torch.logsumexp(log_w, 0) - math.log(n))
        info = {"log_Z": log_Z, "n_resample": n_resample,
                "ess_final": ess_trace[-1] if ess_trace else float("nan"),
                "ess_min": min(ess_trace) if ess_trace else float("nan"),
                "block": blk,
                # the order the weights are conditional on, so a caller can
                # recompute the proposal density and check the telescoping
                "order": perm}
        return x, log_w.float(), info

    def state_dict(self) -> dict:
        return {
            "step_count": self.step_count,
            "net": self.net.state_dict(),
            "opt": self.opt.state_dict(),
            "ema": self.ema.state_dict(),
            "n_skipped": self.n_skipped,
            "acq": self.acq.state_dict() if self.acq is not None else None,
            # The replay buffer is part of the fixed-point iterate, not a
            # cache: resuming without it restarts the sampler against an
            # empty buffer and throws away the exploration already paid for.
            # At 16x16 it is a few MB, so it always rides along.
            "buffer": {
                "states": self.buf.states,
                "extras": self.buf.extras,
            },
        }

    def load_state_dict(self, state: dict):
        if self.acq is not None and state.get("acq"):
            self.acq.load_state_dict(state["acq"])
        self.step_count = state["step_count"]
        self.net.load_state_dict(state["net"])
        self.opt.load_state_dict(state["opt"])
        self.ema.load_state_dict(state["ema"])
        self.n_skipped = state.get("n_skipped", 0)
        buf = state.get("buffer")
        if buf is not None and buf["states"] is not None:
            self.buf.states = buf["states"].to(self.device)
            self.buf.extras = {
                k: v.to(self.device) for k, v in buf["extras"].items()
            }
