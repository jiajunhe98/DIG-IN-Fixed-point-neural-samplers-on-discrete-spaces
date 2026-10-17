"""ASBS baseline: Discrete Adjoint Schrödinger Bridge Sampler (Guo et al.).

Two networks over a uniform reference CTMC:

    controller  log Phi_t(x)_{d,n}   (time-conditioned)  — sampler rate is
                u_t(x^{d<-n}, x) = gamma_t / N * Phi_t(x)_{d,n}
    corrector   log Phi_hat(x)_{d,n} (time-free, at t=1)

trained by alternating stages of Bregman matching losses:

    ctrl-TM: regress Phi_t(x_t) onto  s_nu(x1) / Phi_hat(x1)  aligned by the
             mod-N shift (x1 - x_t + n); this is where the energy enters.
    ctrl-DM / corr-DM: denoising regressions onto closed-form reference-
             bridge ratios.
    corr-TM: regress Phi_hat(x1) onto 1 / Phi_0(x0) (uniform init score),
             shift-aligned.

In `memoryless` mode (log-linear noise with alpha << 1) the reference
terminal law is uniform and the corrector is identically 1; then the path
weights give an exactly unbiased log Z estimator.  Otherwise log Z is only
consistent at convergence and we report it as biased (unbiased_log_Z=False).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor

from dds.methods.asbs import reference as ref
from dds.methods.base import Method
from dds.models import build_model
from dds.utils.buffer import ReplayBuffer
from dds.utils.ema import EMA
from dds.utils.schedules import NoiseSchedule

_EPS = 1e-10


def bregman_div(a: Tensor, b: Tensor, func: str = "tlogt") -> Tensor:
    if func == "tlogt":
        return a * (a / b.clamp_min(_EPS)).clamp_min(_EPS).log() - a + b
    if func == "-logt":
        r = a / b.clamp_min(_EPS)
        return r - r.clamp_min(_EPS).log() - 1
    if func == "t2":
        return (a - b) ** 2
    raise ValueError(func)


class ASBS(Method):
    def __init__(
        self,
        target,
        model: dict,
        noise: dict | None = None,
        loss_controller: str = "tm",
        loss_corrector: str = "dm",
        breg_func: str = "tlogt",
        dm_eps: float = 0.01,
        num_repl: int = 8,
        batch_size: int = 128,
        buffer_size: int = 2048,
        resample_every: int = 20,
        n_rollout: int = 512,
        steps_controller: int = 400,
        steps_corrector: int = 200,
        sampling_steps: int = 100,
        sampling_steps_eval: int | None = None,
        init: str = "uniform",  # 'uniform' | 'const'
        memoryless: bool = False,
        use_log_rnd: bool = False,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        grad_clip: float = 0.0,
        ema_decay: float = 0.9999,
        cap: float = 1.0,
        anneal_steps: int = 0,
        anneal_start: float = 0.5,
        eval_chunk: int = 512,
        device: str = "cpu",
    ):
        super().__init__(target, device)
        N = target.N
        self.amortized = bool(getattr(target, "amortized", False))
        model = dict(model)
        name = model.pop("name")
        if self.amortized:
            model.setdefault("cond_ranges", list(target.cond_ranges))
        self.controller = build_model(
            name, target.lattice_shape, N, N, require_time=True, **model
        ).to(self.device)
        # the corrector must start as the identity, so ASBS pins the zero-init
        # head rather than inheriting whatever the model group asked for
        corrector_model = {**model, "zero_init_head": True}
        self.corrector = build_model(
            name, target.lattice_shape, N, N, require_time=False,
            **corrector_model,
        ).to(self.device)
        self.opt_ctrl = torch.optim.AdamW(
            self.controller.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.opt_corr = torch.optim.AdamW(
            self.corrector.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.ema_ctrl = EMA(self.controller.parameters(), decay=ema_decay)
        self.ema_corr = EMA(self.corrector.parameters(), decay=ema_decay)

        noise = dict(noise or {"kind": "log_linear", "gamma": 1.0, "alpha": 0.5})
        if memoryless:
            noise = {"kind": "log_linear", "gamma": noise.get("gamma", 1.0), "alpha": 0.01}
        self.noise = NoiseSchedule(
            noise.get("kind", "log_linear"), noise.get("gamma", 1.0), noise.get("alpha", 0.5)
        )
        self.memoryless = memoryless
        self.unbiased_log_Z = bool(memoryless)
        if memoryless:
            steps_corrector = 0

        self.loss_controller = loss_controller
        self.loss_corrector = loss_corrector
        self.breg_func = breg_func
        self.dm_eps = dm_eps
        self.num_repl = num_repl
        self.batch_size = batch_size
        self.resample_every = resample_every
        self.n_rollout = n_rollout
        self.steps_controller = steps_controller
        self.steps_corrector = steps_corrector
        self.sampling_steps = sampling_steps
        self.sampling_steps_eval = sampling_steps_eval or sampling_steps
        self.init = init
        self.use_log_rnd = use_log_rnd
        self.grad_clip = grad_clip
        self.cap = cap
        self.anneal_steps = anneal_steps
        self.anneal_start = anneal_start
        self.eval_chunk = eval_chunk
        self.buf = ReplayBuffer(buffer_size)
        self._phase_prev = "controller"
        self.n_skipped = 0  # updates refused for a non-finite gradient

    # ------------------------------------------------------------------
    def _init_states(self, n: int) -> Tensor:
        if self.init == "uniform":
            return self.target.sample_uniform(n, device=self.device)
        vals = torch.randint(0, self.target.N, (n, 1), device=self.device)
        return vals.expand(n, self.target.D).contiguous()

    def _ctrl(self, x: Tensor, t: Tensor, cond: Tensor | None = None) -> Tensor:
        """Controller call, with the amortized conditioning when there is any."""
        if not self.amortized:
            return self.controller(x, t)
        return self.controller(x, t, self.target.to_cond_features(cond))

    def _draw_cond(self, n: int) -> Tensor | None:
        if not self.amortized:
            return None
        return self.target.sample_cond(n, device=self.device)

    def _corrector_out(self, x: Tensor, cond: Tensor | None = None) -> Tensor:
        if self.memoryless:
            return torch.zeros(
                x.shape[0], x.shape[1], self.target.N, device=x.device
            )
        if not self.amortized:
            return self.corrector(x)
        return self.corrector(x, None, self.target.to_cond_features(cond))

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _simulate(self, n: int, steps: int, scale: float = 1.0,
                  cond: Tensor | None = None):
        """Tau-leap the controlled CTMC; (x0, x1, log_rnd, logw, cond)."""
        N, D = self.target.N, self.target.D
        if self.amortized and cond is None:
            cond = self._draw_cond(n)
        t0 = 0.01 if self.memoryless else 0.0
        ts = torch.linspace(t0, 1.0, steps + 1, device=self.device)
        x0 = self._init_states(n)
        x = x0
        log_p_u = torch.zeros(n, device=self.device)
        log_p_r = torch.zeros(n, device=self.device)
        log_star_r = torch.zeros(n, device=self.device)
        for i in range(steps):
            t = ts[i].expand(n)
            tn = ts[i + 1].expand(n)
            phi = self._ctrl(x, t, cond).exp()
            phi = phi.scatter(
                2, x.unsqueeze(-1), torch.zeros_like(x, dtype=phi.dtype).unsqueeze(-1)
            )
            gbar = self.noise.integral(t, tn).view(-1, 1, 1)
            p = gbar / N * phi
            row = p.sum(-1, keepdim=True)
            p = p * (self.cap / row.clamp_min(_EPS)).clamp(max=1.0)
            stay = (1.0 - p.sum(-1, keepdim=True)).clamp_min(_EPS)
            probs = p.scatter(2, x.unsqueeze(-1), stay)
            x_new = torch.multinomial(probs.view(n * D, N), 1).view(n, D)

            log_p_u += (
                torch.gather(probs.clamp_min(_EPS).log(), 2, x_new.unsqueeze(-1))
                .squeeze(-1).sum(dim=1)
            )
            log_p_r += ref.ref_trans_log_p(self.noise, t, tn, x, x_new, N)
            jumped = (x_new != x).float()
            term1 = gbar[:, 0, 0] / N * (D * (N - 1) - phi.sum(dim=(1, 2)))
            term2 = (
                torch.gather(phi.clamp_min(_EPS).log(), 2, x_new.unsqueeze(-1))
                .squeeze(-1) * jumped
            ).sum(dim=1)
            log_star_r += term1 + term2
            x = x_new

        if self.memoryless:
            log_star_r = scale * self.target.log_prob(x, cond)
        log_rnd = log_star_r + log_p_r - log_p_u
        # unbiased weights (memoryless): E[exp] = Z
        logw = (
            scale * self.target.log_prob(x, cond) + log_p_r - log_p_u
            + D * math.log(N)
        )
        return x0, x, log_rnd, logw, cond

    @torch.no_grad()
    def _refresh_buffer(self, scale: float):
        params = list(self.controller.parameters())
        self.ema_ctrl.store(params)
        self.ema_ctrl.copy_to(params)
        try:
            x0, x1, log_rnd, _, cond = self._simulate(
                self.n_rollout, self.sampling_steps, scale)
        finally:
            self.ema_ctrl.restore(params)
        extras = {"x0": x0, "log_rnd": log_rnd}
        if cond is not None:
            # the state point travels with the endpoint; a buffered pair is
            # only valid together with the parameters it was generated at
            extras["cond"] = cond
        self.buf.add(x1, **extras)

    # ------------------------------------------------------------------
    def _phase(self) -> str:
        stage_len = self.steps_controller + self.steps_corrector
        return (
            "controller"
            if (self.step_count % stage_len) < self.steps_controller
            else "corrector"
        )

    def _path_weights(self, log_rnd: Tensor) -> Tensor:
        if not self.use_log_rnd:
            return torch.ones_like(log_rnd)
        return log_rnd.shape[0] * log_rnd.softmax(dim=0)

    def step(self) -> dict:
        scale = self.anneal_scale()
        phase = self._phase()
        if phase != self._phase_prev:
            # finished a phase: make its EMA weights permanent
            if self._phase_prev == "controller":
                self.ema_ctrl.copy_to(self.controller.parameters())
            else:
                self.ema_corr.copy_to(self.corrector.parameters())
            self._phase_prev = phase
        if self.step_count % self.resample_every == 0 or len(self.buf) == 0:
            self._refresh_buffer(scale)

        x1, extras = self.buf.sample(self.batch_size)
        x0, log_rnd = extras["x0"], extras["log_rnd"]
        cond = extras.get("cond")
        N = self.target.N

        if phase == "controller":
            loss = self._loss_controller(x0, x1, log_rnd, scale, cond)
            opt, net, ema = self.opt_ctrl, self.controller, self.ema_ctrl
        else:
            loss = self._loss_corrector(x0, x1, log_rnd, cond)
            opt, net, ema = self.opt_corr, self.corrector, self.ema_corr

        opt.zero_grad(set_to_none=True)
        loss.backward()
        # Bregman target matching exponentiates network outputs, so a single
        # bad batch can produce an infinite gradient; applying it leaves NaN
        # weights and the next rollout dies inside the sampler instead.
        gnorm = torch.nn.utils.clip_grad_norm_(
            net.parameters(), self.grad_clip if self.grad_clip > 0 else math.inf
        )
        if not torch.isfinite(gnorm):
            opt.zero_grad(set_to_none=True)
            self.n_skipped += 1
        else:
            opt.step()
            ema.update(net.parameters())
        self.step_count += 1
        return {
            "loss": loss.item(),
            "grad_norm": gnorm.item(),
            "skipped_updates": self.n_skipped,
            "phase": phase,
            "anneal_scale": scale,
        }

    # ------------------------------------------------------------------
    def _loss_controller(self, x0, x1, log_rnd, scale, cond=None) -> Tensor:
        N = self.target.N
        R = self.num_repl
        x0r = x0.repeat_interleave(R, 0)
        x1r = x1.repeat_interleave(R, 0)
        cr = None if cond is None else cond.repeat_interleave(R, 0)
        pw = self._path_weights(log_rnd).repeat_interleave(R, 0)
        B = x0r.shape[0]
        if self.loss_controller == "tm":
            t = torch.rand(B, device=self.device)
            xt = ref.sample_ref_bridge(self.noise, t, x0r, x1r, N)
            log_phi = self._ctrl(xt, t, cr)
            with torch.no_grad():
                tgt_log = (scale * self.target.log_score(x1r, cr)
                           - self._corrector_out(x1r, cr))
                idx = (
                    x1r.unsqueeze(-1) - xt.unsqueeze(-1)
                    + torch.arange(N, device=self.device).view(1, 1, N)
                ) % N
                tgt = torch.gather(tgt_log, 2, idx).exp()
        elif self.loss_controller == "dm":
            t = torch.rand(B, device=self.device) * (1 - self.dm_eps)
            xt = ref.sample_ref_bridge(self.noise, t, x0r, x1r, N)
            log_phi = self._ctrl(xt, t, cr)
            with torch.no_grad():
                tgt = ref.ref_trans_log_p_diff(
                    self.noise, t, torch.ones_like(t), xt, x1r, N, forward=False
                ).exp()
        else:
            raise ValueError(self.loss_controller)
        mask = (
            xt.unsqueeze(-1) != torch.arange(N, device=self.device).view(1, 1, N)
        ).float()
        divg = (bregman_div(tgt, log_phi.exp(), self.breg_func) * mask).sum(dim=(1, 2))
        return (divg * pw).mean() / (self.target.D * N)

    def _loss_corrector(self, x0, x1, log_rnd, cond=None) -> Tensor:
        N = self.target.N
        pw = self._path_weights(log_rnd)
        log_hat = self._corrector_out(x1, cond)
        if self.loss_corrector == "dm":
            R = self.num_repl
            x0r = x0.repeat_interleave(R, 0)
            x1r = x1.repeat_interleave(R, 0)
            cr = None if cond is None else cond.repeat_interleave(R, 0)
            pw = pw.repeat_interleave(R, 0)
            log_hat = self._corrector_out(x1r, cr)
            t = torch.rand(x0r.shape[0], device=self.device) * (1 - self.dm_eps)
            xt = ref.sample_ref_bridge(self.noise, t, x0r, x1r, N)
            with torch.no_grad():
                tgt = ref.ref_trans_log_p_diff(
                    self.noise, t, torch.ones_like(t), xt, x1r, N, forward=True
                ).exp()
            mask_x = x1r
        elif self.loss_corrector == "tm":
            with torch.no_grad():
                zeros = torch.zeros(x0.shape[0], device=self.device)
                # uniform initial distribution has unit discrete score
                tgt_log = -self._ctrl(x0, zeros, cond)
                idx = (
                    x0.unsqueeze(-1) - x1.unsqueeze(-1)
                    + torch.arange(N, device=self.device).view(1, 1, N)
                ) % N
                tgt = torch.gather(tgt_log, 2, idx).exp()
            mask_x = x1
        else:
            raise ValueError(self.loss_corrector)
        mask = (
            mask_x.unsqueeze(-1) != torch.arange(N, device=self.device).view(1, 1, N)
        ).float()
        divg = (bregman_div(tgt, log_hat.exp(), self.breg_func) * mask).sum(dim=(1, 2))
        return (divg * pw).mean() / (self.target.D * N)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample_with_weights(
        self, n: int, cond: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """Memoryless: exactly unbiased IS weights.  Otherwise the weights
        are the SB path RND (consistent at convergence only)."""
        params = list(self.controller.parameters())
        self.ema_ctrl.store(params)
        self.ema_ctrl.copy_to(params)
        try:
            xs, ws = [], []
            for i in range(0, n, self.eval_chunk):
                k = min(self.eval_chunk, n - i)
                c = None
                if self.amortized:
                    # one state point, k samples: the grid evaluation asks for
                    # a single (T, delta_mu) and wants every sample at it
                    c = (cond.view(1, -1).expand(k, -1) if cond is not None
                         else self._draw_cond(k))
                _, x1, log_rnd, logw, _ = self._simulate(
                    k, self.sampling_steps_eval, cond=c)
                xs.append(x1)
                ws.append(logw if self.memoryless else log_rnd)
        finally:
            self.ema_ctrl.restore(params)
        return torch.cat(xs), torch.cat(ws)

    # ------------------------------------------------------------------
    def state_dict(self) -> dict:
        return {
            "step_count": self.step_count,
            "controller": self.controller.state_dict(),
            "corrector": self.corrector.state_dict(),
            "opt_ctrl": self.opt_ctrl.state_dict(),
            "opt_corr": self.opt_corr.state_dict(),
            "ema_ctrl": self.ema_ctrl.state_dict(),
            "ema_corr": self.ema_corr.state_dict(),
        }

    def load_state_dict(self, state: dict):
        self.step_count = state["step_count"]
        self.controller.load_state_dict(state["controller"])
        self.corrector.load_state_dict(state["corrector"])
        self.opt_ctrl.load_state_dict(state["opt_ctrl"])
        self.opt_corr.load_state_dict(state["opt_corr"])
        self.ema_ctrl.load_state_dict(state["ema_ctrl"])
        self.ema_corr.load_state_dict(state["ema_corr"])
