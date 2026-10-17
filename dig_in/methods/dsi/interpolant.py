"""Mixture (uniform-noise) discrete stochastic interpolant, general alphabet.

Notation: states x in {0..N-1}^D; per-coordinate
mixture kernel

    kappa_t(m | c0, c1) = alpha_t 1{m=c0} + beta_t 1{m=c1} + omega_t / N,

with simplex schedules alpha + beta + omega = 1, alpha_0 = 1, beta_1 = 1.
The prior is uniform, the coupling independent: x0 ~ U, x1 ~ nu.

Everything the four training variants and the sampler need is a closed-form
function of (kappa, kappa_dot) contracted against either the sampled
endpoints (per-sample regression targets) or the learned endpoint posterior
(field read-outs):

    conditional velocity   vhat(n, m | c0,c1) = (kd_n)_+ (kd_m)_- / (c+ k_m)
    backward velocity      vbhat(n, m | c0,c1) = (kd_n)_- (kd_m)_+ / (c+ k_m)
    kernel score ratio     r(n, m | c0,c1)    = k_n / k_m

All public functions take integer states of shape (B, D) and times (B,).
Fields/targets are returned as (B, D, N) tensors indexed by the destination
value n of a single-coordinate move d -> n.
"""

from __future__ import annotations

import torch
from torch import Tensor

_EPS = 1e-12


class MixtureInterpolant:
    def __init__(
        self,
        num_states: int,
        schedule: str = "quadratic",
        clamp_ratio: float = 1e3,
    ):
        self.N = num_states
        self.schedule = schedule
        self.clamp_ratio = clamp_ratio

    # ------------------------------------------------------------------
    def coeffs(self, t: Tensor):
        """(alpha, beta, omega, dalpha, dbeta, domega), each shaped like t."""
        if self.schedule == "quadratic":
            a, b, w = (1 - t) ** 2, t**2, 2 * t * (1 - t)
            da, db, dw = -2 * (1 - t), 2 * t, 2 - 4 * t
        elif self.schedule == "onesided-quadratic":
            # masked-type schedule: prior channel off, beta_t = t^2
            a, b, w = torch.zeros_like(t), t**2, 1 - t**2
            da, db, dw = torch.zeros_like(t), 2 * t, -2 * t
        elif self.schedule == "onesided-linear":
            a, b, w = torch.zeros_like(t), t, 1 - t
            da = torch.zeros_like(t)
            db, dw = torch.ones_like(t), -torch.ones_like(t)
        else:
            raise ValueError(self.schedule)
        return a, b, w, da, db, dw

    # ------------------------------------------------------------------
    def sample(self, x0: Tensor, x1: Tensor, t: Tensor, generator=None) -> Tensor:
        """Draw I_t given endpoints; x0, x1 (B, D), t (B,)."""
        a, b, _, _, _, _ = self.coeffs(t)
        r = torch.rand(x0.shape, device=x0.device, generator=generator)
        z = torch.randint(
            0, self.N, x0.shape, device=x0.device, generator=generator
        )
        a = a.unsqueeze(-1)
        b = b.unsqueeze(-1)
        return torch.where(r < a, x0, torch.where(r < a + b, x1, z))

    # ------------------------------------------------------------------
    # Closed-form kernel tensors (per batch element; no site dimension)
    # ------------------------------------------------------------------
    def _kernel_tensors(self, t: Tensor):
        """kappa, kappa_dot of shape (B, N, N, N) indexed [c0, c1, m]."""
        N = self.N
        a, b, w, da, db, dw = self.coeffs(t)
        eye = torch.eye(N, device=t.device)
        d0 = eye.view(N, 1, N)  # 1{m = c0}
        d1 = eye.view(1, N, N)  # 1{m = c1}
        sh = (-1, 1, 1, 1)
        kappa = a.view(sh) * d0 + b.view(sh) * d1 + w.view(sh) / N
        kdot = da.view(sh) * d0 + db.view(sh) * d1 + dw.view(sh) / N
        return kappa, kdot

    def _rate_tensors(self, t: Tensor):
        """(vhat, vbhat, ratio), each (B, N, N, N, N) indexed [c0, c1, n, m]."""
        kappa, kdot = self._kernel_tensors(t)
        kp = kdot.clamp_min(0.0)  # gains, [.., k]
        kn = (-kdot).clamp_min(0.0)  # losses
        cp = kp.sum(-1).clamp_min(_EPS)  # (B, N, N)
        km = kappa.clamp_min(_EPS)
        # [c0, c1, n, m]
        denom = cp[..., None, None] * km[:, :, :, None, :]
        vhat = (kp[..., :, None] * kn[..., None, :] / denom).clamp(max=self.clamp_ratio)
        vbhat = (kn[..., :, None] * kp[..., None, :] / denom).clamp(max=self.clamp_ratio)
        ratio = (kappa[..., :, None] / km[:, :, :, None, :]).clamp(max=self.clamp_ratio)
        return vhat, vbhat, ratio

    # ------------------------------------------------------------------
    # Per-sample regression targets (rates route / Bregman-TSI training)
    # ------------------------------------------------------------------
    def conditional_targets(self, xt: Tensor, x0: Tensor, x1: Tensor, t: Tensor):
        """(T_v, T_vb): conditional forward/backward velocities
        vhat(n, xt^d | x0^d, x1^d), each (B, D, N)."""
        vhat, vbhat, _ = self._rate_tensors(t)
        B, D = xt.shape
        bidx = torch.arange(B, device=xt.device).unsqueeze(-1).expand(B, D)
        # advanced indices at dims (0,1,2,4) with a slice at dim 3 -> (B, D, N)
        T_v = vhat[bidx, x0, x1, :, xt]
        T_vb = vbhat[bidx, x0, x1, :, xt]
        return T_v, T_vb

    def score_target_gsi(self, xt: Tensor, x1: Tensor, target_score1: Tensor) -> Tensor:
        """GSI score target for a uniform prior (Cor. gsi-indep):

            T_s[d, n] = s_nu(x1)_{d, x1^d + (n - xt^d)}   (mod N)

        target_score1 = s_nu(x1) of shape (B, D, N).  Returns (B, D, N);
        the diagonal entry n = xt^d equals 1.
        """
        N = self.N
        n = torch.arange(N, device=xt.device).view(1, 1, N)
        idx = (x1.unsqueeze(-1) + n - xt.unsqueeze(-1)) % N
        return torch.gather(target_score1, 2, idx)

    # ------------------------------------------------------------------
    # Endpoint-posterior (denoiser) route
    # ------------------------------------------------------------------
    def pair_log_posterior(self, g: Tensor, xt: Tensor, t: Tensor) -> Tensor:
        """Product-form pair posterior (Lemma productform, uniform prior):

            q(c0, c1 | x, t)  propto  kappa_t(x^d | c0, c1) * exp(g[d, c1])

        g: context logits (B, D, N).  Returns log q of shape (B, D, N, N).
        """
        kappa, _ = self._kernel_tensors(t)  # (B, N, N, N)
        B, D = xt.shape
        bidx = torch.arange(B, device=xt.device).unsqueeze(-1).expand(B, D)
        kappa_x = kappa[bidx, :, :, xt]  # (B, D, N, N) [c0, c1]
        logits = torch.log(kappa_x.clamp_min(_EPS)) + g.unsqueeze(2)  # g over c1
        return logits.flatten(2).log_softmax(-1).view(B, D, self.N, self.N)

    def posterior_fields(self, log_q: Tensor, xt: Tensor, t: Tensor):
        """Read-outs (v, vb, s), each (B, D, N), from the pair posterior.

        v[d, n]  = E_q[ vhat(n, xt^d | c0, c1) ]      (0 at n = xt^d)
        vb[d, n] = E_q[ vbhat(n, xt^d | c0, c1) ]     (0 at n = xt^d)
        s[d, n]  = E_q[ kappa(n|..) / kappa(xt^d|..) ]  (1 at n = xt^d)
        """
        vhat, vbhat, ratio = self._rate_tensors(t)
        B, D = xt.shape
        bidx = torch.arange(B, device=xt.device).unsqueeze(-1).expand(B, D)
        vhat_x = vhat[bidx, :, :, :, xt]  # (B, D, N, N, N) [c0, c1, n]
        vbhat_x = vbhat[bidx, :, :, :, xt]
        ratio_x = ratio[bidx, :, :, :, xt]
        q = log_q.exp()
        v = torch.einsum("bdij,bdijn->bdn", q, vhat_x)
        vb = torch.einsum("bdij,bdijn->bdn", q, vbhat_x)
        s = torch.einsum("bdij,bdijn->bdn", q, ratio_x)
        # exact diagonal values
        diag = xt.unsqueeze(-1)
        v = v.scatter(2, diag, torch.zeros_like(diag, dtype=v.dtype))
        vb = vb.scatter(2, diag, torch.zeros_like(diag, dtype=vb.dtype))
        s = s.scatter(2, diag, torch.ones_like(diag, dtype=s.dtype))
        return v, vb, s

    # ------------------------------------------------------------------
    # Soft cross-entropy labels (heat-bath tilt; Lemma tilted)
    # ------------------------------------------------------------------
    def soft_labels(self, xt: Tensor, x1: Tensor, t: Tensor, heat_bath1: Tensor) -> Tensor:
        """Tilted labels for the pair-posterior cross-entropy:

            l^d(c0, c1)  propto  nu(c1 | x1^{-d}) * kappa_t(xt^d | c0, c1)

        (uniform prior factor is constant).  heat_bath1 = nu(. | x1^{-d})
        of shape (B, D, N).  Returns normalized labels (B, D, N, N).
        """
        kappa, _ = self._kernel_tensors(t)
        B, D = xt.shape
        bidx = torch.arange(B, device=xt.device).unsqueeze(-1).expand(B, D)
        kappa_x = kappa[bidx, :, :, xt]  # (B, D, N, N)
        lbl = heat_bath1.unsqueeze(2) * kappa_x  # broadcast over c0
        return lbl / lbl.sum(dim=(2, 3), keepdim=True).clamp_min(_EPS)

    def onehot_labels(self, x0: Tensor, x1: Tensor) -> Tensor:
        """One-hot pair labels 1{(c0,c1) = (x0^d, x1^d)}, shape (B, D, N, N)."""
        B, D = x0.shape
        lbl = torch.zeros(B, D, self.N, self.N, device=x0.device)
        idx = (x0 * self.N + x1).view(B, D, 1)
        lbl.view(B, D, -1).scatter_(2, idx, 1.0)
        return lbl
