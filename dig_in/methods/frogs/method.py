"""FrOGS (Min, Hofgard, Smidt 2026): a learned prior in front of LEAPS transport.

LEAPS anneals from the uniform distribution to the target and learns a CTMC
that carries the mass.  Near a first-order boundary that fails: the path
crosses the transition at some t*, the mass has to move through energetically
unstable configurations between the two phases, and because the transport
only moves one site at a time the log-weights acquire the heavy tail that
collapses the ESS.

FrOGS replaces the fixed prior with a conditioned autoregressive network, so
the annealing path starts somewhere that already has mass in both basins and
the transport only has to refine it:

    U_t(sigma | c) = (1 - t) U_0(sigma | c) + t U_1(sigma),
    U_0 = -log rho_0^psi(sigma | c),        U_1 = beta H_c(sigma).

Prior, rate matrix and free-energy head are trained together under the one
PINN residual, so a single backward pass returns gradients for all three.
With the transport switched off the objective reduces to the log-variance
(VarGrad) loss on the prior, which is why the prior is not mode-seeking the
way a reverse-KL autoregressive model is -- that is the paper's Proposition 3
and it is what separates FrOGS from SEGAL.

THE COST.  The corrector (their Proposition 1) is

    K~ rho(sigma) = -d_t U_t(sigma) + sum_{i, tau != sigma_i} M_t(tau,i|sigma)
                                                              G_t(tau,i|sigma)

with the Metropolis factor M_t = exp(-[Delta_{i tau} U_t]_+).  The locally
equivariant rate net hands back the whole G table in one pass, and the energy
differences are the local field, also one pass.  The prior's contribution to
Delta U is neither: flipping site i changes every autoregressive conditional
downstream of it, so an exact table costs D * (N-1) forward passes of the
prior per state.  That is why the paper reports FrOGS at 125 and 128 sites
and no larger, and why this implementation is only run there.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from dig_in.methods.base import Method
from dig_in.methods.leaps.method import horizon_schedule
from dig_in.methods.leaps.nets import FNet, LocEquivNet
from dig_in.models import build_model


class FrOGS(Method):
    #: Z_0 = 1 exactly, because the prior is a normalised autoregressive
    #: model, so E[exp A_1] = Z_1 with no prior partition function to carry
    unbiased_log_Z = True

    def __init__(
        self,
        target,
        model: dict | None = None,
        prior: dict | None = None,
        delta_t: float = 0.02,
        starting_k: int = 1,
        n_anneal: int = 2000,
        quadratic_anneal: bool = True,
        n_walkers: int = 64,
        loss_batch: int = 64,
        buffer_cycle: int = 20,
        buffer_size: int = 20,
        rate_clamp: float = 10.0,
        lr: float = 3e-4,
        prior_lr: float = 1e-3,
        lr_gamma: float = 0.995,
        lr_step: int = 1000,
        grad_clip: float = 1.0,
        f_hidden: int = 200,
        n_steps_eval: int | None = None,
        anneal_steps: int = 0,
        anneal_start: float = 0.5,
        eval_chunk: int = 128,
        flip_chunk: int = 4096,
        device: str = "cpu",
    ):
        super().__init__(target, device)
        self.amortized = bool(getattr(target, "amortized", False))
        cond_dim = int(getattr(target, "cond_dim", 0)) if self.amortized else 0
        cond_ranges = list(getattr(target, "cond_ranges", [])) if self.amortized else []

        model = dict(model or {})
        model.pop("name", None)
        self.net = LocEquivNet(target.lattice_shape, target.N,
                               cond_dim=cond_dim, **model).to(self.device)
        self.f_net = FNet(hidden=f_hidden, cond_dim=cond_dim).to(self.device)
        prior = dict(prior or {})
        prior.pop("name", None)
        self.prior = build_model(
            "pixelcnn", target.lattice_shape, vocab_size=target.N,
            out_channels=target.N, require_time=False,
            cond_ranges=cond_ranges, **prior,
        ).to(self.device)

        self.opt = torch.optim.Adam([
            {"params": list(self.net.parameters()) + list(self.f_net.parameters()),
             "lr": lr},
            {"params": list(self.prior.parameters()), "lr": prior_lr},
        ])
        self.sched = torch.optim.lr_scheduler.StepLR(
            self.opt, step_size=lr_step, gamma=lr_gamma)

        self.k_max = int(round(1.0 / delta_t))
        self.delta_t = delta_t
        self.ks = horizon_schedule(self.k_max, starting_k, n_anneal, quadratic_anneal)
        self.n_walkers = n_walkers
        self.loss_batch = loss_batch
        self.buffer_cycle = buffer_cycle
        self.buffer_size = buffer_size
        self.rate_clamp = rate_clamp
        self.grad_clip = grad_clip
        self.n_steps_eval = n_steps_eval or self.k_max
        self.anneal_steps = anneal_steps
        self.anneal_start = anneal_start
        self.eval_chunk = eval_chunk
        self.flip_chunk = flip_chunk
        self.buffer: list[tuple[Tensor, Tensor, Tensor]] = []
        self._last_ess = float("nan")

    # ------------------------------------------------------------------
    def _feat(self, cond: Tensor | None) -> Tensor | None:
        if not self.amortized or cond is None:
            return None
        f = getattr(self.target, "to_cond_features", None)
        return f(cond) if f is not None else cond

    def log_rho0(self, x: Tensor, cond: Tensor | None = None) -> Tensor:
        """log rho_0^psi(x | c), shape (B,)."""
        lp = self.prior(x, None, cond).log_softmax(dim=-1)
        return lp.gather(2, x.unsqueeze(-1)).squeeze(-1).sum(dim=1)

    def delta_log_rho0(self, x: Tensor, cond: Tensor | None = None) -> Tensor:
        """log rho_0(x^{d<-a}) - log rho_0(x) for every (d, a), shape (B, D, N).

        Exact, and the expensive half of the method: an autoregressive model
        has no shortcut for a single-site mutation, because flipping site d
        moves every conditional after it.  Evaluated in chunks over the
        B * D * (N-1) mutated configurations.
        """
        B, D = x.shape
        N = self.target.N
        base = self.log_rho0(x, cond)
        out = torch.zeros(B, D, N, device=x.device, dtype=base.dtype)

        idx = [(d, a) for d in range(D) for a in range(N)]
        flat, conds = [], []
        for d, a in idx:
            y = x.clone()
            y[:, d] = a
            flat.append(y)
        flat = torch.cat(flat, dim=0)                       # (B*D*N, D)
        if cond is not None:
            conds = cond.repeat(len(idx), 1)
        vals = []
        for i in range(0, flat.shape[0], self.flip_chunk):
            c = conds[i:i + self.flip_chunk] if cond is not None else None
            vals.append(self.log_rho0(flat[i:i + self.flip_chunk], c))
        vals = torch.cat(vals).view(len(idx), B)
        for j, (d, a) in enumerate(idx):
            out[:, d, a] = vals[j] - base
        return out

    # ------------------------------------------------------------------
    def _dtU(self, x: Tensor, cond: Tensor | None, scale: float) -> Tensor:
        """d/dt of U_t = U_1 - U_0 = beta H(x) + log rho_0(x | c)."""
        return -scale * self.target.log_prob(x, cond) + self.log_rho0(x, cond)

    def _deltaU(self, x: Tensor, t: Tensor, cond: Tensor | None, scale: float,
                dlr: Tensor) -> Tensor:
        """Delta_{d a} U_t, shape (B, D, N)."""
        beta_dH = -scale * self.target.log_score(x, cond)   # = beta (H' - H)
        tt = t.view(-1, 1, 1)
        return tt * beta_dH - (1.0 - tt) * dlr

    def _rate_logits(self, x: Tensor, t: Tensor, feat: Tensor | None) -> Tensor:
        G = self.net(x, t, feat)
        if self.rate_clamp > 0:
            G = G.clamp(-self.rate_clamp, self.rate_clamp)
        return G

    def _corrector(self, x: Tensor, t: Tensor, cond: Tensor | None,
                   scale: float, dlr: Tensor) -> Tensor:
        """sum_{d, a != x_d} M_t G_t, the reparametrized corrector."""
        G = self._rate_logits(x, t, self._feat(cond))
        dU = self._deltaU(x, t, cond, scale, dlr)
        M = torch.exp(-F.relu(dU))
        term = M * G
        # a != x_d only; G is already zero at a = x_d by construction, but the
        # Metropolis factor there is 1, so drop the diagonal explicitly
        term = term.scatter(2, x.unsqueeze(-1), torch.zeros_like(x, dtype=G.dtype).unsqueeze(-1))
        return term.sum(dim=(1, 2))

    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample_prior(self, n: int, cond: Tensor | None = None):
        """Draw from rho_0 and return (x, log rho_0(x))."""
        caches = self.prior.new_kv_caches(n, self.device)
        prev = None
        logq = torch.zeros(n, device=self.device)
        for d in range(self.target.D):
            lg = self.prior.decode_step(prev, torch.tensor(d), caches, n, cond)
            p = torch.softmax(lg, dim=-1)
            tok = torch.multinomial(p, 1).squeeze(1)
            logq = logq + torch.log(p.gather(1, tok.unsqueeze(1)).squeeze(1).clamp_min(1e-30))
            caches[0]["x"][:, d] = tok
            prev = tok
        return caches[0]["x"].clone(), logq

    @torch.no_grad()
    def _ctmc_step(self, x: Tensor, t: Tensor, dt: Tensor, cond, scale, dlr):
        G = self._rate_logits(x, t, self._feat(cond))
        dU = self._deltaU(x, t, cond, scale, dlr)
        rates = (F.relu(G) * torch.exp(-F.relu(dU))).scatter(
            2, x.unsqueeze(-1), torch.zeros_like(x, dtype=G.dtype).unsqueeze(-1))
        total = rates.sum(-1)
        p_jump = 1.0 - torch.exp(-dt.view(-1, 1) * total)
        jump = torch.rand_like(p_jump) < p_jump
        probs = rates / total.clamp_min(1e-12).unsqueeze(-1)
        probs = torch.where(total.unsqueeze(-1) > 1e-12, probs,
                            torch.full_like(probs, 1.0 / probs.shape[-1]))
        new = torch.distributions.Categorical(probs=probs).sample()
        return torch.where(jump, new, x)

    @torch.no_grad()
    def _rollout(self, n: int, k: int, scale: float, cond: Tensor | None,
                 uniform_grid: bool = False, total_time: float | None = None):
        T = k * self.delta_t if total_time is None else total_time
        if uniform_grid or k == 1:
            ts = torch.linspace(0.0, T, k + 1, device=self.device)
        else:
            mid = T * torch.rand(k - 1, device=self.device)
            ts = torch.sort(torch.cat([
                torch.zeros(1, device=self.device), mid,
                torch.full((1,), T, device=self.device)])).values
        x, _ = self.sample_prior(n, cond)
        A = torch.zeros(n, device=self.device)
        states = [x.clone()]
        for i in range(k):
            t = ts[i].expand(n)
            dt = (ts[i + 1] - ts[i]).expand(n)
            dlr = self.delta_log_rho0(x, cond)
            A = A - dt * (self._dtU(x, cond, scale)
                          - self._corrector(x, t, cond, scale, dlr))
            x = self._ctmc_step(x, t, dt, cond, scale, dlr)
            states.append(x.clone())
        return torch.stack(states), ts, A

    # ------------------------------------------------------------------
    def _draw_cond(self, n: int) -> Tensor | None:
        if not self.amortized:
            return None
        return self.target.sample_cond(n, device=self.device)

    def step(self) -> dict:
        scale = self.anneal_scale()
        k = self.ks[min(self.step_count, len(self.ks) - 1)]
        if self.step_count % self.buffer_cycle == 0 or not self.buffer:
            cond = self._draw_cond(self.n_walkers)
            states, ts, A = self._rollout(self.n_walkers, k, scale, cond)
            self.buffer.append((states.cpu(), ts.cpu(),
                                None if cond is None else cond.cpu()))
            if len(self.buffer) > self.buffer_size:
                self.buffer.pop(0)
            w = A - A.max()
            self._last_ess = float(
                w.exp().sum() ** 2 / (len(A) * (2 * w).exp().sum()))

        import random
        states, ts, cond = random.choice(self.buffer)
        states, ts = states.to(self.device), ts.to(self.device)
        cond = None if cond is None else cond.to(self.device)
        n_t, n_w = states.shape[0], states.shape[1]
        ti = torch.randint(0, n_t, (self.loss_batch,), device=self.device)
        wi = torch.randint(0, n_w, (self.loss_batch,), device=self.device)
        sigma = states[ti, wi]
        tvec = ts[ti]
        cvec = None if cond is None else cond[wi]

        dlr = self.delta_log_rho0(sigma, cvec)
        dtU = self._dtU(sigma, cvec, scale)
        corr = self._corrector(sigma, tvec, cvec, scale, dlr)
        dF = self.f_net.gradt(tvec, self._feat(cvec))
        loss = ((-dtU + corr + dF) ** 2).mean()

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                list(self.net.parameters()) + list(self.f_net.parameters())
                + list(self.prior.parameters()), self.grad_clip)
        self.opt.step()
        self.sched.step()
        self.step_count += 1
        return {"loss": float(loss), "k": k, "train_ess": self._last_ess,
                "anneal_scale": scale}

    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample_with_weights(self, n: int, cond: Tensor | None = None):
        """(x, log w) with E[exp log w] = Z.

        Z_0 = 1 because the prior is normalised, so the Jarzynski weight is
        A_1 alone -- no D log N offset as in the uniform-prior LEAPS.
        """
        xs, ws = [], []
        for i in range(0, n, self.eval_chunk):
            c = min(self.eval_chunk, n - i)
            cc = None
            if cond is not None:
                cc = cond.view(1, -1).expand(c, -1).to(self.device)
            elif self.amortized:
                cc = self._draw_cond(c)
            states, _, A = self._rollout(c, self.n_steps_eval, 1.0, cc,
                                         uniform_grid=True, total_time=1.0)
            xs.append(states[-1])
            ws.append(A)
        return torch.cat(xs), torch.cat(ws)

    # ------------------------------------------------------------------
    def state_dict(self) -> dict:
        return {"step_count": self.step_count, "net": self.net.state_dict(),
                "f_net": self.f_net.state_dict(),
                "prior": self.prior.state_dict(),
                "opt": self.opt.state_dict(), "sched": self.sched.state_dict()}

    def load_state_dict(self, state: dict):
        self.step_count = state["step_count"]
        self.net.load_state_dict(state["net"])
        self.f_net.load_state_dict(state["f_net"])
        self.prior.load_state_dict(state["prior"])
        self.opt.load_state_dict(state["opt"])
        self.sched.load_state_dict(state["sched"])
