"""LEAPS baseline (Holderrieth, Albergo, Jaakkola, 2025).

Anneals rho_t propto exp(-t * beta * E) from the uniform distribution (t=0)
to the target (t=1) and learns single-site CTMC rates Q_theta transporting
the path, by minimizing the squared residual of the Kolmogorov forward
(continuity) equation

    r(x, t) = -dU_t(x)/dt + dF_phi(t)/dt - [ sum_y Q(y,x) rho_t(y)/rho_t(x)
                                             - sum_y Q(x,y) ]

on states from replayed rollouts of the current sampler (CTMC transport +
exact Gibbs mixing at rho_t during training).  Jarzynski/Feynman-Kac work
accumulation A along a rollout gives importance weights with
E[exp(A)] = Z_1 / Z_0 (exact as dt -> 0; O(dt) biased at finite step).

Caveat on log Z: when the learned transport is imperfect, exp(A) is heavy-
tailed through rare off-manifold excursions that finite samples miss, so the
estimate can be biased while the *empirical* ESS looks perfect — the original
paper reports ESS, not log Z.  Diagnose by checking stability of log Z under
grid refinement AND larger sample sizes.

Deviations from the reference implementation, chosen deliberately:
* categorical locally-equivariant net for all alphabet sizes (their binary
  Ising variant used a multiplicative spin gate);
* MCMC mixing uses vectorized checkerboard Gibbs sweeps of rho_t instead of
  single-site updates at a shared random site;
* the work increment is always evaluated at the pre-step state.
"""

from __future__ import annotations

import math
import random

import torch
import torch.nn.functional as F
from torch import Tensor

from dds.methods.base import Method
from dds.methods.leaps.nets import FNet, LocEquivNet
from dds.utils.gibbs import heat_bath_sweeps


def horizon_schedule(k_max: int, starting_k: int, n_anneal: int, quadratic: bool = True):
    """List mapping training step -> integration horizon k."""
    ks = list(range(starting_k, k_max + 1))
    if not ks:
        return [k_max]
    if quadratic:
        weights = [i + 1 for i in range(len(ks))]
    else:
        weights = [1] * len(ks)
    total = sum(weights)
    out = []
    for k, w in zip(ks, weights):
        out.extend([k] * max(1, round(n_anneal * w / total)))
    return out


class LEAPS(Method):
    unbiased_log_Z = True  # exact only in the dt -> 0 limit (O(dt) bias)

    def __init__(
        self,
        target,
        model: dict | None = None,
        delta_t: float = 0.01,
        starting_k: int = 1,
        n_anneal: int = 2000,
        quadratic_anneal: bool = True,
        n_walkers: int = 128,
        loss_batch: int = 256,
        buffer_cycle: int = 20,
        buffer_size: int = 50,
        mcmc_sweeps: int = 1,
        rate_clamp: float = 0.0,
        lr: float = 3e-4,
        lr_gamma: float = 0.995,
        lr_step: int = 1000,
        grad_clip: float = 0.0,
        f_hidden: int = 200,
        n_steps_eval: int | None = None,
        anneal_steps: int = 0,
        anneal_start: float = 0.5,
        eval_chunk: int = 512,
        device: str = "cpu",
    ):
        super().__init__(target, device)
        model = dict(model or {})
        model.pop("name", None)
        self.net = LocEquivNet(target.lattice_shape, target.N, **model).to(self.device)
        self.f_net = FNet(hidden=f_hidden).to(self.device)
        self.opt = torch.optim.Adam(
            list(self.net.parameters()) + list(self.f_net.parameters()), lr=lr
        )
        self.sched = torch.optim.lr_scheduler.StepLR(
            self.opt, step_size=lr_step, gamma=lr_gamma
        )
        self.k_max = int(round(1.0 / delta_t))
        self.delta_t = delta_t
        self.ks = horizon_schedule(self.k_max, starting_k, n_anneal, quadratic_anneal)
        self.n_walkers = n_walkers
        self.loss_batch = loss_batch
        self.buffer_cycle = buffer_cycle
        self.buffer_size = buffer_size
        self.mcmc_sweeps = mcmc_sweeps
        self.rate_clamp = rate_clamp
        self.grad_clip = grad_clip
        self.n_steps_eval = n_steps_eval or self.k_max
        self.anneal_steps = anneal_steps
        self.anneal_start = anneal_start
        self.eval_chunk = eval_chunk
        self.buffer: list[tuple[Tensor, Tensor]] = []  # (states (k+1, B, D), ts)
        self._last_ess = float("nan")

    # ------------------------------------------------------------------
    def _net_out(self, x: Tensor, t: Tensor) -> Tensor:
        R = self.net(x, t)
        if self.rate_clamp > 0:
            R = R.clamp(-self.rate_clamp, self.rate_clamp)
        return R

    def _corrector(self, x: Tensor, t: Tensor, scale: float) -> Tensor:
        """sum_y Q(y,x) rho(y)/rho(x) - sum_y Q(x,y), shape (B,)."""
        R = self._net_out(x, t)  # (B, D, N), zero at current values
        out_rates = F.relu(R)
        in_rates = F.relu(-R)
        with torch.no_grad():
            log_ratio = t.view(-1, 1, 1) * scale * self.target.log_score(x)
        return (in_rates * log_ratio.exp()).sum(dim=(1, 2)) - out_rates.sum(dim=(1, 2))

    @torch.no_grad()
    def _ctmc_step(self, x: Tensor, t: Tensor, dt: Tensor) -> Tensor:
        R = self._net_out(x, t)
        rates = F.relu(R).scatter(
            2, x.unsqueeze(-1), torch.zeros_like(x, dtype=R.dtype).unsqueeze(-1)
        )
        total = rates.sum(-1)
        p_jump = 1.0 - torch.exp(-dt.view(-1, 1) * total)
        jump = torch.rand_like(p_jump) < p_jump
        probs = rates / total.clamp_min(1e-12).unsqueeze(-1)
        uniform = torch.full_like(probs, 1.0 / probs.shape[-1])
        probs = torch.where(total.unsqueeze(-1) > 1e-12, probs, uniform)
        new = torch.distributions.Categorical(probs=probs).sample()
        return torch.where(jump, new, x)

    @torch.no_grad()
    def _rollout(
        self, n: int, k: int, scale: float, mcmc: bool,
        uniform_grid: bool = False, total_time: float | None = None,
    ):
        """Returns (states (k+1, n, D), ts (k+1,), A (n,)).

        total_time defaults to k * delta_t (the annealed training horizon);
        evaluation passes total_time=1.0 so that more steps refine the grid
        instead of extending the horizon.
        """
        T = k * self.delta_t if total_time is None else total_time
        if uniform_grid:
            ts = torch.linspace(0.0, T, k + 1, device=self.device)
        else:
            mid = T * torch.rand(k - 1, device=self.device) if k > 1 else torch.empty(0, device=self.device)
            ts = torch.sort(
                torch.cat([torch.zeros(1, device=self.device), mid,
                           torch.full((1,), T, device=self.device)])
            ).values
        x = self.target.sample_uniform(n, device=self.device)
        A = torch.zeros(n, device=self.device)
        states = [x.clone()]
        for i in range(k):
            t = ts[i].expand(n)
            dt = (ts[i + 1] - ts[i]).expand(n)
            dtU = -scale * self.target.log_prob(x)  # d/dt of t*scale*beta*E
            corr = self._corrector(x, t, scale)
            A = A - dt * (dtU + corr)
            x = self._ctmc_step(x, t, dt)
            if mcmc and self.mcmc_sweeps > 0:
                x = heat_bath_sweeps(
                    self.target, x, self.mcmc_sweeps, scale=float(ts[i + 1]) * scale
                )
            states.append(x.clone())
        return torch.stack(states), ts, A

    # ------------------------------------------------------------------
    def step(self) -> dict:
        scale = self.anneal_scale()
        k = self.ks[min(self.step_count, len(self.ks) - 1)]
        if self.step_count % self.buffer_cycle == 0 or not self.buffer:
            states, ts, A = self._rollout(self.n_walkers, k, scale, mcmc=True)
            self.buffer.append((states.cpu(), ts.cpu()))
            if len(self.buffer) > self.buffer_size:
                self.buffer.pop(0)
            w = A - A.max()
            self._last_ess = (
                w.exp().sum() ** 2 / (len(A) * (2 * w).exp().sum())
            ).item()
        states, ts = random.choice(self.buffer)
        states, ts = states.to(self.device), ts.to(self.device)
        n_t, n_w = states.shape[0], states.shape[1]
        ti = torch.randint(0, n_t, (self.loss_batch,), device=self.device)
        wi = torch.randint(0, n_w, (self.loss_batch,), device=self.device)
        sigma = states[ti, wi]
        tvec = ts[ti]

        dtU = -scale * self.target.log_prob(sigma)
        corr = self._corrector(sigma, tvec, scale)
        dF = self.f_net.gradt(tvec)
        loss = ((-dtU - corr + dF) ** 2).mean()

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                list(self.net.parameters()) + list(self.f_net.parameters()),
                self.grad_clip,
            )
        self.opt.step()
        self.sched.step()
        self.step_count += 1
        return {
            "loss": loss.item(),
            "k": k,
            "train_ess": self._last_ess,
            "anneal_scale": scale,
        }

    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample_with_weights(self, n: int) -> tuple[Tensor, Tensor]:
        """Pure neural transport (no MCMC), Jarzynski weights:
        log w = D log N + A_1, E[exp(log w)] = Z up to O(dt) bias."""
        xs, ws = [], []
        log_Z0 = self.target.D * math.log(self.target.N)
        for i in range(0, n, self.eval_chunk):
            c = min(self.eval_chunk, n - i)
            states, _, A = self._rollout(
                c, self.n_steps_eval, scale=1.0, mcmc=False,
                uniform_grid=True, total_time=1.0,
            )
            xs.append(states[-1])
            ws.append(log_Z0 + A)
        return torch.cat(xs), torch.cat(ws)

    # ------------------------------------------------------------------
    def state_dict(self) -> dict:
        return {
            "step_count": self.step_count,
            "net": self.net.state_dict(),
            "f_net": self.f_net.state_dict(),
            "opt": self.opt.state_dict(),
            "sched": self.sched.state_dict(),
        }

    def load_state_dict(self, state: dict):
        self.step_count = state["step_count"]
        self.net.load_state_dict(state["net"])
        self.f_net.load_state_dict(state["f_net"])
        self.opt.load_state_dict(state["opt"])
        self.sched.load_state_dict(state["sched"])
