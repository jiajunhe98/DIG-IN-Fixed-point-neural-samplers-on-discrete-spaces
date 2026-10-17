"""Fixed-point masked-diffusion sampler for 25x25 PBC Ising, amortised over (beta, h).

v2 of this trainer:
  * EA-style buffer: ONE ring, every entry carries its own continuous (beta, h) -- the
    parameters are a property of the sample, no rungs anywhere.
  * Adaptive parameter sampling: 50% uniform over the (beta, h) box (coverage + unbiased
    acquisition statistics) + 50% weighted by an acquisition map over a coarse (beta, h)
    grid.  Acquisition = |finite-difference gradient| of the model's own binned mean spin
    along BOTH axes (the order parameter is known here, so using it is legitimate -- it
    shapes only where compute goes, never labels or loss), EMA-smoothed across iterations.
    Sampling concentrates wherever the model currently places sharp parameter dependence:
    the h-crossover and the beta transition, discovered rather than hard-coded.
  * Function-space damping eta (the beta=0.30 crossover oscillates at eta=1).
  * PT-in-beta references (blocked Gibbs cannot mix modes at small h below sigma_c),
    per-cell cached, reusable across experiments via ref_dir.

Correctness note: the fixed point pi(.|beta,h) is unchanged by ANY full-support parameter
sampling distribution -- adaptivity affects convergence speed only.
"""

import copy
import json
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

import potts_field as pf
from network_hq import PottsFieldDenoiser, unmask_sample

CFG = dict(
    L=32,
    q=3,
    # (beta, h) box in the MDNS convention, pi ~ exp(beta*(sum_<ij> s_i s_j + h*sum_i s_i)).
    # beta = 2*sigma_old and h = h_old/beta, so the old 25x25 box [0.195,0.325]x|h|<=0.05
    # maps to beta in [0.39,0.65] and |h| <= 0.128 at the cold end / 0.064 at the hot end.
    # beta_c = log(1+sqrt 2)/2 = 0.4407 sits inside.  See energy_mdns.py.
    # (beta, h) box for q=3 Potts with a field on state 0:
    #   log pi~ = beta * (A(x) + h * N_0(x)),  beta_c(3) = log(1+sqrt 3) = 1.00505
    # A pure state is worth beta*h*L^2 nats, i.e. about 1029*h at L=32 beta=1.005, so the
    # mode-selection crossover sits near |h| ~ 0.001 to 0.004 and |h| = 0.01 is saturated.
    beta_lo=0.800,
    beta_hi=1.300,
    beta_range=[0.700, 1.400],   # encoder normalisation, wider than the sampling box
    h_max=0.01,
    dilations=[1, 1, 2, 4, 8, 1, 1],   # radius 37 >= L//2 = 25 at L=50; None for plain blocks
    # acquisition grid
    adaptive=1,
    acq="fisher",         # "fp" = Rao-Blackwellised fixed-point residual (model error);
                          # "fisher" = Tr Cov[grad_theta U] (intrinsic difficulty); "grad_m" = |grad m|
    acq_power=0.5,        # w ~ a^power; Var(M) spans ~4 decades between disordered and coexisting
    acq_fisher_norm=1,    # standardise each Fisher channel by its across-cell median before summing
    acq_min_n=8,          # cells with fewer endpoints than this are not updated (fp residual)
    acq_bins_b=8,
    acq_bins_h=12,
    acq_ema=0.8,          # EMA of the binned mean spin across iterations
    acq_floor=0.05,       # uniform mass mixed into the adaptive half so no bin ever starves
    hidden=96,
    n_blocks=7,
    kernel=3,
    lr=3e-4,
    weight_decay=0.0,
    ema=0.995,
    grad_clip=1.0,
    n_outer=1000000,
    n_inner=100,
    batch=1024,
    n_explore=1024,
    explore_steps=625,
    n_single=64,
    buffer_cap=16384,     # single ring; ~16 iterations of history
    warmup_outer=10,
    eta=0.5,
    is_weight=1,          # tempered importance weighting of the fit term (method note):
                          # log w = log pi_tilde - log q_path, computed at generation time,
                          # stored per entry, never refreshed; alpha per round by ESS bisection
    ess_target=0.2,
    is_center_bins=1,     # center log w PER (beta,h) CELL before tempering.  E_q[w|theta]=Z(theta)
                          # and Z varies by ~100 nats across the box, vs ~16 nats WITHIN a cell:
                          # global self-normalisation therefore reweights between cells by Z and
                          # starves the low-beta half (measured: 28% of samples -> 0.1% of weight).
    is_alpha_bins=1,      # choose the tempering exponent PER (beta,h) cell, on the acquisition
                          # grid, instead of one global alpha set by the hardest cell in the box
    is_min_bin=32,        # cells with fewer entries than this fall back to global centering
    eval_every=200,
    eval_steps=0,         # 0 -> explore_steps.  An eval is
                          # len(eval_betas)*len(eval_hs) * n_eval * eval_steps forward passes;
                          # at L=50 with the 25x25 defaults that is 21*1024*1024 = 21504 full
                          # batches, about 50 min, which dwarfs the training it is measuring.
    eval_hs=[-0.01, -0.002, 0.0, 0.002, 0.01],
    eval_betas=[0.900, 1.00505, 1.200],     # below / at / above beta_c(3)
    ref_chains=1024,      # PT cost is 4x the 25x25 run per sweep; 21 cells x 24 temps adds up
    ref_temps=24,
    n_eval=1024,
    gibbs_sweeps=4000,
    ckpt_every=10,
    keep_every=4,
    compile=1,
    init_from="",
    ref_dir="",
    step_offset=-1,
    seed=0,
    device="cuda",
    wandb_project="ising-fixed-point-sampler",
    wandb_mode="online",
    run_name=None,
    out_dir="runs",
)


class Buffer:
    """One FIFO ring; every entry carries its own (beta, h)."""

    def __init__(self, cap, D, q, device):
        self.x = torch.zeros(cap, D, dtype=torch.uint8, device=device)
        self.p = torch.zeros(cap, D, q, dtype=torch.float16, device=device)
        self.bet = torch.zeros(cap, device=device)
        self.h = torch.zeros(cap, device=device)
        self.logw = torch.zeros(cap, device=device)
        self.logw_c = torch.zeros(cap, device=device)   # per-cell-centered, refreshed each round
        self.w = torch.ones(cap, device=device)         # per-cell tempered+normalised IS weight
        self.alpha = torch.ones(cap, device=device)     # the exponent its own cell was given
        self.cap = cap
        self.pos = 0
        self.n = 0

    def push(self, x, p, sg, hh, lw):
        b = x.shape[0]
        idx = (torch.arange(b, device=x.device) + self.pos) % self.cap
        self.x[idx] = x.to(torch.uint8)
        self.p[idx] = p.to(torch.float16)
        self.bet[idx] = sg
        self.h[idx] = hh
        self.logw[idx] = lw.float()
        self.pos = (self.pos + b) % self.cap
        self.n = min(self.n + b, self.cap)

    def draw(self, b):
        idx = torch.randint(0, self.n, (b,), device=self.x.device)
        return (self.x[idx].long(), self.p[idx].float(), self.bet[idx], self.h[idx],
                self.w[idx])


class Acquisition:
    """Binned mean-spin map over (beta, h); acquisition = |finite-difference gradient|."""

    def __init__(self, cfg, device):
        self.S, self.H = cfg["acq_bins_b"], cfg["acq_bins_h"]
        self.b_lo, self.b_hi = cfg["beta_lo"], cfg["beta_hi"]
        self.h_max = cfg["h_max"]
        self.ema = cfg["acq_ema"]
        self.floor = cfg["acq_floor"]
        self.m = torch.zeros(self.S, self.H, device=device)
        self.seen = torch.zeros(self.S, self.H, device=device)
        self.dev = device

    def _bin(self, bet, hh):
        si = ((bet - self.b_lo) / (self.b_hi - self.b_lo) * self.S).long().clamp(0, self.S - 1)
        hi = ((hh + self.h_max) / (2 * self.h_max) * self.H).long().clamp(0, self.H - 1)
        return si, hi

    def update(self, bet, hh, mag):
        si, hi = self._bin(bet, hh)
        flat = si * self.H + hi
        s_m = torch.zeros(self.S * self.H, device=self.dev).scatter_add_(0, flat, mag)
        s_c = torch.zeros(self.S * self.H, device=self.dev).scatter_add_(0, flat, torch.ones_like(mag))
        got = s_c > 0
        mean = torch.where(got, s_m / s_c.clamp_min(1), torch.zeros_like(s_m)).view(self.S, self.H)
        gotv = got.view(self.S, self.H)
        self.m = torch.where(gotv, self.ema * self.m + (1 - self.ema) * mean, self.m)
        self.seen = torch.where(gotv, self.seen + 1, self.seen)

    def map(self):
        return self.m

    def weights(self):
        """Per-bin sampling weights: |grad of binned mean spin| along both axes + floor."""
        gs = torch.zeros_like(self.m)
        gh = torch.zeros_like(self.m)
        gs[1:-1] = (self.m[2:] - self.m[:-2]).abs() / 2
        gh[:, 1:-1] = (self.m[:, 2:] - self.m[:, :-2]).abs() / 2
        acq = gs + gh
        acq = acq / acq.sum().clamp_min(1e-8)
        u = torch.full_like(acq, 1.0 / acq.numel())
        w = (1 - self.floor) * acq + self.floor * u
        return w / w.sum()

    def sample(self, n):
        """Draw n (beta, h): bin ~ weights, uniform jitter inside the bin."""
        w = self.weights().flatten()
        k = torch.multinomial(w, n, replacement=True)
        si, hi = (k // self.H).float(), (k % self.H).float()
        ds = (self.b_hi - self.b_lo) / self.S
        dh = 2 * self.h_max / self.H
        bet = self.b_lo + (si + torch.rand(n, device=self.dev)) * ds
        hh = -self.h_max + (hi + torch.rand(n, device=self.dev)) * dh
        return bet, hh

    def state(self):
        return {"m": self.m, "seen": self.seen}

    def load(self, st):
        if st["m"].shape == self.m.shape:
            self.m.copy_(st["m"])
            self.seen.copy_(st["seen"])


class FPResidualAcquisition:
    """Rao-Blackwellised fixed-point residual: where would an exact Gibbs update still move the
    model?  At stationarity  E[1(x_i=a)] = E[rho_i(a | x_{-i})], so per cell and per site

        d_i = mean_j ( 1[x_{j,i}=1] - rho_i(1 | x_{j,-i}; theta_j) )

    is zero in expectation exactly when the model is invariant under the target's own single-site
    update.  rho is the training label, so the score costs nothing.  Unlike the Fisher score this
    measures MODEL ERROR and decays to zero as q -> pi, at which point the allocation relaxes to
    uniform.

    DEBIASING (essential).  The plain estimator sum_a (p_hat - q_hat)^2 has expectation
    a_true + 2 E[rho(1-rho)]/M: a positive floor that is largest where rho ~ 1/2 (disordered,
    critical) and vanishes where rho is saturated (deep ordered).  Uncorrected, the acquisition
    would chase its own sampling noise into the disordered region.  The U-statistic form

        a_i = 2 * ( d_i^2 - v_i / M ),    v_i = unbiased Var_j(Z_{j,i})

    is unbiased for the squared discrepancy and vanishes in expectation at the fixed point for any
    M.  (Binary vocabulary: the a=0 and a=1 terms are equal, hence the factor 2.)
    The score is averaged over ALL sites before clamping, which cuts its own noise by sqrt(D).
    """

    def __init__(self, cfg, device):
        self.S, self.H = cfg["acq_bins_b"], cfg["acq_bins_h"]
        self.b_lo, self.b_hi = cfg["beta_lo"], cfg["beta_hi"]
        self.h_max = cfg["h_max"]
        self.ema, self.floor = cfg["acq_ema"], cfg["acq_floor"]
        self.power = cfg["acq_power"]
        self.min_n = cfg["acq_min_n"]
        nb = self.S * self.H
        self.a = torch.zeros(nb, device=device)
        self.seen = torch.zeros(nb, device=device)
        self.dev = device

    def _bin(self, bet, hh):
        si = ((bet - self.b_lo) / (self.b_hi - self.b_lo) * self.S).long().clamp(0, self.S - 1)
        hi = ((hh + self.h_max) / (2 * self.h_max) * self.H).long().clamp(0, self.H - 1)
        return si, hi

    def update(self, bet, hh, x, p, L, q):
        """x: [n, D] tokens; p: [n, D, q] exact conditionals rho_i(a | x^{-i}) -- the training labels.

        TEST FUNCTIONS.  Stationarity gives  E[f(s_i, x_{-i})] = E[ sum_a rho_i(a) f(a, x_{-i}) ]
        for ANY f; the note's f = 1[x_i = a] is the weakest member and is blind here -- at h = 0 an
        untrained model gives uniform spins, so both sides equal 1/2 by symmetry and the residual
        vanishes for a model that is nowhere near pi (measured: debiased residual ~1e-4 for both a
        converged and an untrained model).  A second channel f = s_i * m_i (site times neighbour
        sum, i.e. the local energy) fixes it: Z = 2 (x_i - rho_i) m_i, still free because m_i is
        already computed for rho, and manifestly non-zero for a uniform model since rho_i
        correlates with m_i.  Channels are standardised by their across-cell median before summing.
        """
        # q-state form of the two Ising channels.  Marginal: the one-hot residual
        # 1[x_i = a] - rho_i(a), q channels (the Ising z0 is its a=1 entry).  Energy:
        # n_{x_i}(i) - sum_a rho_i(a) n_a(i), the Potts analogue of s_i*m_i, and the channel
        # that carries signal for a model far from pi.
        oh = torch.nn.functional.one_hot(x, q).to(torch.float32)      # [n, D, q]
        z0 = oh - p                                                    # [n, D, q]
        ze = pf.local_energy_residual(x, p, L, q)[..., None]           # [n, D, 1]
        z = torch.cat([z0, ze], -1)                                    # [n, D, q+1]
        si, hi = self._bin(bet, hh)
        flat = (si * self.H + hi).long()
        nb = self.S * self.H
        one = torch.ones(x.shape[0], device=self.dev)
        cnt = torch.zeros(nb, device=self.dev).scatter_add_(0, flat, one)
        D2, C = z.shape[1], z.shape[2]
        zf = z.reshape(x.shape[0], D2 * C)
        idx = flat[:, None].expand(-1, D2 * C)
        s1 = torch.zeros(nb, D2 * C, device=self.dev).scatter_add_(0, idx, zf)
        s2 = torch.zeros(nb, D2 * C, device=self.dev).scatter_add_(0, idx, zf * zf)
        M = cnt.clamp_min(1.0)[:, None]
        d = s1 / M
        v = (s2 - M * d * d) / (M - 1).clamp_min(1.0)
        ac = (2.0 * (d * d - v / M)).view(nb, D2, C).mean(1)   # [nb, C] debiased, over sites
        got = cnt >= self.min_n
        # Sum the channels RAW.  Standardising them (as the Fisher score needs) would be wrong
        # here: the marginal channel is blind on this family, so rescaling it to unit median would
        # promote pure noise to the same weight as the energy channel, which carries the signal.
        # Raw, the energy channel dominates by ~3 orders of magnitude exactly when it should.
        a = ac.sum(-1)
        self.a = torch.where(got, self.ema * self.a + (1 - self.ema) * a, self.a)
        self.seen = torch.where(got, self.seen + 1, self.seen)

    def map(self):
        return self.a.clamp_min(0.0).view(self.S, self.H)

    def weights(self):
        a = self.map() ** self.power
        a = a / a.sum().clamp_min(1e-30)
        u = torch.full_like(a, 1.0 / a.numel())
        w = (1 - self.floor) * a + self.floor * u
        return w / w.sum()

    def sample(self, n):
        w = self.weights().flatten()
        k = torch.multinomial(w, n, replacement=True)
        si, hi = (k // self.H).float(), (k % self.H).float()
        ds = (self.b_hi - self.b_lo) / self.S
        dh = 2 * self.h_max / self.H
        return (self.b_lo + (si + torch.rand(n, device=self.dev)) * ds,
                -self.h_max + (hi + torch.rand(n, device=self.dev)) * dh)

    def state(self):
        return {"a": self.a, "seen": self.seen}

    def load(self, st):
        if "a" in st and st["a"].shape == self.a.shape:
            self.a.copy_(st["a"]); self.seen.copy_(st["seen"])


class FisherAcquisition:
    """Observable-free acquisition: a_k = Tr Cov_{x ~ q(.|theta), theta in C_k} [ grad_theta U ].

    For this family  U(x; beta, h) = -beta * (A(x) + h * N_0(x)),  A = agreeing bonds and
    N_0 = sites in state 0, so

        dU/dbeta = -(A + h N_0)        dU/dh = -beta N_0

    and the trace is Var(A + h N_0) + Var(beta N_0): the specific-heat and the field-response
    channels appear on their own, with no observable chosen by hand.

    Two refinements over the plain trace:
      * the derivatives are taken in the NORMALISED parameter coordinates of the sampling box
        (theta~ = (beta - b_lo)/Delta_sigma, h/Delta_h).  Tr Cov is not invariant under
        reparametrisation, so without this the acquisition would depend on the physical units of
        beta and h rather than on the family being amortised over.
      * each channel is standardised by its across-cell median before summing, and the result is
        raised to a power < 1.  Var(M) at coexistence exceeds its disordered value by ~4 decades;
        the raw trace would place essentially all adaptive mass in one or two cells.
    """

    def __init__(self, cfg, device):
        self.S, self.H = cfg["acq_bins_b"], cfg["acq_bins_h"]
        self.b_lo, self.b_hi = cfg["beta_lo"], cfg["beta_hi"]
        self.h_max = cfg["h_max"]
        self.ema, self.floor = cfg["acq_ema"], cfg["acq_floor"]
        self.power, self.norm = cfg["acq_power"], cfg["acq_fisher_norm"]
        nb = self.S * self.H
        self.m1 = torch.zeros(nb, 2, device=device)
        self.m2 = torch.zeros(nb, 2, device=device)
        self.seen = torch.zeros(nb, device=device)
        self.dev = device

    def _bin(self, bet, hh):
        si = ((bet - self.b_lo) / (self.b_hi - self.b_lo) * self.S).long().clamp(0, self.S - 1)
        hi = ((hh + self.h_max) / (2 * self.h_max) * self.H).long().clamp(0, self.H - 1)
        return si, hi

    def _flat(self, bet, hh):
        si, hi = self._bin(bet, hh)
        return (si * self.H + hi).long()

    def update(self, bet, hh, x, L, q):
        A = pf.agree_bonds(x, L)
        N0 = pf.field_count(x, L)
        # derivatives in the NORMALISED box coordinates, so the acquisition does not depend on
        # the physical units of beta and h
        g = torch.stack([(self.b_hi - self.b_lo) * (-(A + hh * N0)),
                         (2.0 * self.h_max) * (-bet * N0)], -1)
        flat = self._flat(bet, hh)
        nb = self.S * self.H
        cnt = torch.zeros(nb, device=self.dev).scatter_add_(0, flat, torch.ones_like(A))
        s1 = torch.zeros(nb, 2, device=self.dev).scatter_add_(0, flat[:, None].expand(-1, 2), g)
        s2 = torch.zeros(nb, 2, device=self.dev).scatter_add_(0, flat[:, None].expand(-1, 2), g * g)
        got = (cnt > 0)
        c = cnt.clamp_min(1.0)[:, None]
        self.m1 = torch.where(got[:, None], self.ema * self.m1 + (1 - self.ema) * (s1 / c), self.m1)
        self.m2 = torch.where(got[:, None], self.ema * self.m2 + (1 - self.ema) * (s2 / c), self.m2)
        self.seen = torch.where(got, self.seen + 1, self.seen)

    def map(self):
        var = (self.m2 - self.m1 ** 2).clamp_min(0.0)
        if self.norm:
            live = self.seen > 0
            for d in range(var.shape[1]):
                v = var[live, d]
                med = v.median() if v.numel() else torch.tensor(1.0, device=self.dev)
                var[:, d] = var[:, d] / med.clamp_min(1e-30)
        return var.sum(-1).view(self.S, self.H)

    def weights(self):
        a = self.map().clamp_min(0.0) ** self.power
        a = a / a.sum().clamp_min(1e-30)
        u = torch.full_like(a, 1.0 / a.numel())
        w = (1 - self.floor) * a + self.floor * u
        return w / w.sum()

    def sample(self, n):
        w = self.weights().flatten()
        k = torch.multinomial(w, n, replacement=True)
        si, hi = (k // self.H).float(), (k % self.H).float()
        ds = (self.b_hi - self.b_lo) / self.S
        dh = 2 * self.h_max / self.H
        return (self.b_lo + (si + torch.rand(n, device=self.dev)) * ds,
                -self.h_max + (hi + torch.rand(n, device=self.dev)) * dh)

    def state(self):
        return {"m1": self.m1, "m2": self.m2, "seen": self.seen}

    def load(self, st):
        if "m1" in st and st["m1"].shape == self.m1.shape:
            self.m1.copy_(st["m1"]); self.m2.copy_(st["m2"]); self.seen.copy_(st["seen"])


@torch.no_grad()
def center_logw(logw, bet, hh, acq, cfg):
    """Subtract each (beta,h) cell's mean log weight.  The importance weight corrects the
    measure WITHIN a target; across targets it carries Z(beta,h), which is not a property of
    any sample and must not reweight one parameter cell against another."""
    if not cfg["is_center_bins"]:
        return logw.double() - logw.double().mean()
    si, hi = acq._bin(bet, hh)
    flat = (si * acq.H + hi).long()
    nb = acq.S * acq.H
    y = logw.double()
    ssum = torch.zeros(nb, device=y.device, dtype=y.dtype).scatter_add_(0, flat, y)
    cnt = torch.zeros(nb, device=y.device, dtype=y.dtype).scatter_add_(0, flat, torch.ones_like(y))
    gmean = y.mean()
    bmean = torch.where(cnt >= cfg["is_min_bin"], ssum / cnt.clamp_min(1.0), gmean)
    return y - bmean[flat]


@torch.no_grad()
def alpha_and_weights_per_bin(logw_c, bet, hh, acq, cfg):
    """Per-(beta,h)-cell tempering exponent and per-cell-normalised importance weights.

    Why per cell.  A single global alpha is set by the WORST cell in the box.  The centred
    log-weight spread is not comparable across the box -- it is a few nats in the disordered
    corner and tens of nats where the two modes coexist -- and the bisection has to satisfy
    ESS on the pooled sample, so every cell is tempered down to what the hardest one needs.
    Cells that could safely take alpha = 1 end up reweighting at alpha ~ 0.02 and the
    importance correction does nothing where it would have been cheap and correct.

    The grid is exactly the acquisition grid (`acq._bin`), so "where compute goes" and "how
    hard we reweight" are indexed the same way and can be read against each other.

    Normalisation is also per cell.  E_q[w | theta] = Z(theta), which is a property of the
    target and not of any sample, so letting cells compete on total weight would reweight the
    box by Z -- the same failure `is_center_bins` fixes for the mean, one order up.  Each
    cell's weights are divided by their own mean, so a cell keeps exactly the mass the
    parameter-sampling distribution gave it and IS only redistributes WITHIN it.

    Cells with fewer than `is_min_bin` entries fall back to the global alpha and global
    normalisation.  Returns (w [n], alpha [n], stats).
    """
    y = logw_c.double()
    a_glob, ess_glob = select_alpha(y, cfg["ess_target"])
    if not cfg.get("is_alpha_bins", 1):
        z = a_glob * y
        w = torch.exp(z - z.max())
        return (w / w.mean().clamp_min(1e-300)), torch.full_like(y, a_glob), dict(
            alpha_global=a_glob, ess_global=ess_glob, alpha_min=a_glob, alpha_med=a_glob,
            alpha_max=a_glob, n_bins_own=0)

    si, hi = acq._bin(bet, hh)
    flat = (si * acq.H + hi).long()
    nb = acq.S * acq.H
    ones = torch.ones_like(y)
    cnt = torch.zeros(nb, device=y.device, dtype=y.dtype).scatter_add_(0, flat, ones)
    own = cnt >= cfg["is_min_bin"]                       # cells that get their own exponent

    def ess_per_bin(a_bin):
        """ESS/n within each cell, for a [nb] vector of exponents.  Returns [nb]."""
        z = a_bin[flat] * y
        zmax = torch.full((nb,), -float("inf"), device=y.device, dtype=y.dtype)
        zmax = zmax.scatter_reduce_(0, flat, z, reduce="amax", include_self=True)
        w = torch.exp(z - zmax[flat])
        s1 = torch.zeros(nb, device=y.device, dtype=y.dtype).scatter_add_(0, flat, w)
        s2 = torch.zeros(nb, device=y.device, dtype=y.dtype).scatter_add_(0, flat, w * w)
        return (s1 * s1) / (cnt * s2).clamp_min(1e-300)

    target = cfg["ess_target"]
    one = torch.ones(nb, device=y.device, dtype=y.dtype)
    ok_at_one = ess_per_bin(one) >= target
    # geometric bisection on log10 alpha, all cells at once: early spans are hundreds of nats
    lo = torch.full((nb,), -12.0, device=y.device, dtype=y.dtype)
    hi_ = torch.zeros(nb, device=y.device, dtype=y.dtype)
    for _ in range(50):
        mid = 0.5 * (lo + hi_)
        good = ess_per_bin(torch.pow(10.0, mid)) >= target
        lo = torch.where(good, mid, lo)
        hi_ = torch.where(good, hi_, mid)
    a_bin = torch.where(ok_at_one, one, torch.pow(10.0, lo))
    a_bin = torch.where(own, a_bin, torch.full_like(a_bin, a_glob))

    alpha = a_bin[flat]
    z = alpha * y
    zmax = torch.full((nb,), -float("inf"), device=y.device, dtype=y.dtype)
    zmax = zmax.scatter_reduce_(0, flat, z, reduce="amax", include_self=True)
    w = torch.exp(z - zmax[flat])
    bsum = torch.zeros(nb, device=y.device, dtype=y.dtype).scatter_add_(0, flat, w)
    bmean = (bsum / cnt.clamp_min(1.0)).clamp_min(1e-300)
    gmean = w.mean().clamp_min(1e-300)
    w = w / torch.where(own, bmean, torch.full_like(bmean, gmean))[flat]

    a_own = a_bin[own]
    stats = dict(
        alpha_global=a_glob, ess_global=ess_glob,
        alpha_min=float(a_own.min()) if a_own.numel() else a_glob,
        alpha_med=float(a_own.median()) if a_own.numel() else a_glob,
        alpha_max=float(a_own.max()) if a_own.numel() else a_glob,
        n_bins_own=int(own.sum()),
    )
    return w, alpha, stats


@torch.no_grad()
def select_alpha(logw, target):
    """Largest alpha in (0,1] with ESS(alpha)/n >= target.  Geometric bisection on log10 alpha:
    early log-weight spans are hundreds of nats, so linear bisection on alpha bottoms out."""
    lw = logw.double()
    n = lw.numel()

    def ess_frac(a):
        y = a * lw
        y = y - y.max()
        w = torch.exp(y)
        return float((w.sum() ** 2) / (n * (w * w).sum()))

    if ess_frac(1.0) >= target:
        return 1.0, ess_frac(1.0)
    lo, hi = -12.0, 0.0        # log10 alpha
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if ess_frac(10.0 ** mid) >= target:
            lo = mid
        else:
            hi = mid
    a = 10.0 ** lo
    return a, ess_frac(a)


@torch.no_grad()
def ema_update(ema, model, decay):
    for pe, pm in zip(ema.parameters(), model.parameters()):
        pe.lerp_(pm.detach(), 1.0 - decay)
    for be, bm in zip(ema.buffers(), model.buffers()):
        be.copy_(bm)


def sample_params(n, cfg, acq, device):
    """50% uniform over the (beta, h) box + 50% acquisition-weighted (once the map is warm)."""
    su = cfg["beta_lo"] + torch.rand(n, device=device) * (cfg["beta_hi"] - cfg["beta_lo"])
    hu = (torch.rand(n, device=device) * 2 - 1) * cfg["h_max"]
    if not cfg["adaptive"] or acq.seen.sum() < 2 * acq.S * acq.H:
        return su, hu
    sa, ha = acq.sample(n)
    pick = torch.rand(n, device=device) < 0.5
    return torch.where(pick, sa, su), torch.where(pick, ha, hu)


def soft_ce_loss(model, teacher, x1, p, bet, hh, cfg, w=None):
    """CE((1-eta)*d_old + eta*rho, d) decomposes linearly; the importance weight w corrects the
    measure of the FIT term (eta * CE(rho, d)) only -- the damping term is a proximal
    regulariser tying the iterate to its predecessor, not an expectation over the target."""
    B, D = x1.shape
    t = torch.rand(B, device=x1.device)
    revealed = torch.rand(B, D, device=x1.device) < t[:, None]
    xt = torch.where(revealed, x1, torch.full_like(x1, cfg["q"]))   # q == MASK
    tt = revealed.float().mean(-1)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(xt, tt, bet, hh)
    logp = torch.log_softmax(logits.float(), -1)
    m = (~revealed).float()

    def site_avg(ce):
        return (ce * m).sum(-1) / m.sum(-1).clamp_min(1.0)

    rho = p                                   # already [B, D, q]
    ce_fit = site_avg(-(rho * logp).sum(-1))
    if w is None:
        w = torch.ones_like(ce_fit)
    if teacher is not None and cfg["eta"] < 1.0:
        with torch.no_grad():
            old = torch.softmax(teacher(xt, tt, bet, hh).float(), -1)
        ce_damp = site_avg(-(old * logp).sum(-1))
        return (w * cfg["eta"] * ce_fit + (1.0 - cfg["eta"]) * ce_damp).mean()
    return (w * ce_fit).mean()


def main(cfg):
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    dev = cfg["device"]
    L, D, q = cfg["L"], cfg["L"] ** 2, cfg["q"]
    model = PottsFieldDenoiser(L, q, cfg["hidden"], cfg["n_blocks"], cfg["kernel"],
                               beta_range=tuple(cfg["beta_range"]), h_norm=cfg["h_max"],
                               dilations=cfg["dilations"]).to(dev)
    assert model.radius >= L // 2
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    buf = Buffer(cfg["buffer_cap"], D, q, dev)
    acq = ({"fp": FPResidualAcquisition, "fisher": FisherAcquisition}.get(cfg["acq"], Acquisition))(cfg, dev)
    print(f"PottsFieldDenoiser {sum(p.numel() for p in model.parameters())/1e6:.2f}M  radius={model.radius}  "
          f"beta [{cfg['beta_lo']},{cfg['beta_hi']}]  h [-{cfg['h_max']},{cfg['h_max']}]  "
          f"adaptive={cfg['adaptive']} ({cfg['acq_bins_b']}x{cfg['acq_bins_h']} bins)  "
          f"eta={cfg['eta']}  single-ring cap={cfg['buffer_cap']}", flush=True)

    step0 = 0
    if cfg["init_from"]:
        ck = torch.load(cfg["init_from"], map_location=dev, weights_only=False)
        step0 = int(ck.get("outer", 0)) + 1
        model.load_state_dict(ck["model"])
        ema.load_state_dict(ck["ema"])
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        b = ck.get("buf", {})
        if isinstance(b.get("x"), torch.Tensor) and b["x"].shape == buf.x.shape \
                and "logw" in b:
            for k in ("x", "p", "bet", "h", "logw"):
                getattr(buf, k).copy_(b[k])
            buf.pos, buf.n = b["pos"], b["n"]
        if "acq" in ck:
            acq.load(ck["acq"])
        print(f"initialised from {cfg['init_from']} (outer {ck.get('outer','?')})", flush=True)
    if cfg["step_offset"] >= 0:
        step0 = cfg["step_offset"]

    net = torch.compile(model) if cfg["compile"] else model
    samp = torch.compile(ema) if cfg["compile"] else ema

    # ---- PT-in-beta references, per-cell cached, reusable via ref_dir -------------------
    os.makedirs(cfg["out_dir"], exist_ok=True)
    # The cache is keyed on (beta, h) only, so the FILE name has to carry everything else that
    # changes the physics: the model and q.  ref_dir exists to share caches across runs, and a
    # q=4 run would otherwise read q=3 references silently -- same schema, no error.
    ref_path = os.path.join(cfg["out_dir"], f"ref_hpt_potts_q{q}_L{L}_{cfg['gibbs_sweeps']}.pt")
    cache = {}
    for d in (cfg["out_dir"], cfg["ref_dir"]):
        # NB: not `q` -- that is the number of states in this file
        cand = os.path.join(d, f"ref_hpt_potts_q{q}_L{L}_{cfg['gibbs_sweeps']}.pt") if d else ""
        if cand and os.path.exists(cand):
            cache = torch.load(cand, weights_only=False)
            print(f"loaded {len(cache)} cached references from {cand}", flush=True)
            break
    ref = {}
    for s in cfg["eval_betas"]:
        for hv in cfg["eval_hs"]:
            key = (round(s, 3), round(hv, 4))
            if key not in cache:
                xg = pf.pt_reference(s, hv, L, q, cfg["ref_chains"], cfg["gibbs_sweeps"], dev,
                                     seed=hash(key) % 100000, n_temps=cfg["ref_temps"])
                cache[key] = dict(u=pf.energy_per_spin(xg, s, hv, L).mean().item(),
                                  ms=pf.mode_fraction_0(xg, L, q),
                                  mags=pf.order_parameter(xg, L, q).cpu().numpy(),
                                  mvec=pf.magnetisation_vector(xg, L, q).cpu().numpy(),
                                  modes=pf.mode_fractions(xg, L, q).tolist())
                torch.save(cache, ref_path)   # per-cell save: hours of PT on a preemptible tier
            ref[key] = cache[key]
            _ex = pf.exact_anchor(s, hv, q)
            note = f"  (exact: {_ex:+.4f})" if _ex is not None else ""
            print(f"ref beta={s:.2f} h={hv:+.3f}: u={ref[key]['u']:+.4f} "
                  f"frac0={ref[key]['ms']:.4f} modes={[round(v,3) for v in ref[key]['modes']]}{note}", flush=True)
    torch.save(cache, ref_path)

    import wandb
    run = wandb.init(project=cfg["wandb_project"], name=cfg["run_name"], config=cfg,
                     mode=cfg["wandb_mode"])
    out_dir = os.path.join(cfg["out_dir"], cfg["run_name"] or (run.name or "run"))
    os.makedirs(out_dir, exist_ok=True)
    json.dump(cfg, open(os.path.join(out_dir, "cfg.json"), "w"), indent=2)
    t0 = time.time()

    for outer in range(cfg["n_outer"]):
        te = time.time()
        e_bet, e_h = sample_params(cfg["n_explore"], cfg, acq, dev)
        if outer < cfg["warmup_outer"] and not cfg["init_from"]:
            # uniform over ALL q states, not 2 -- a 2-state warmup never shows the model
            # state 2 and leaves 21% of the buffer biased until it cycles out
            x1 = torch.randint(0, q, (cfg["n_explore"], D), device=dev)
            logq = torch.full((cfg["n_explore"],), -D * float(np.log(q)), device=dev,
                              dtype=torch.float64)
        else:
            x1, logq = unmask_sample(samp, e_bet, e_h, cfg["n_explore"], cfg["explore_steps"],
                                     dev, D, q, cfg["n_single"], return_logq=True)
        p = pf.local_target_p(x1, e_bet, e_h, L, q)
        logpi = -(pf.energy_per_spin(x1, e_bet, e_h, L).double() * (L * L))
        buf.push(x1, p, e_bet, e_h, logpi - logq)
        if cfg["acq"] == "fp":
            acq.update(e_bet, e_h, x1, p, L, q)
        elif cfg["acq"] == "fisher":
            acq.update(e_bet, e_h, x1, L, q)
        else:
            # per-sample indicator that state 0 won: the Potts analogue of the signed
            # magnetisation the Ising version bins, and the quantity whose gradient in h
            # marks the crossover
            acq.update(e_bet, e_h, (pf.majority_state(x1, L, q) == 0).float())
        t_explore = time.time() - te

        tt = time.time()
        alpha, essf = 1.0, 1.0
        astats = {}
        if cfg["is_weight"] and buf.n > 0:
            lw_c = center_logw(buf.logw[: buf.n], buf.bet[: buf.n], buf.h[: buf.n], acq, cfg)
            buf.logw_c[: buf.n] = lw_c.float()
            wv, av, astats = alpha_and_weights_per_bin(
                lw_c, buf.bet[: buf.n], buf.h[: buf.n], acq, cfg)
            buf.w[: buf.n] = wv.float()
            buf.alpha[: buf.n] = av.float()
            alpha, essf = astats["alpha_global"], astats["ess_global"]
        model.train()
        losses = []
        for _ in range(cfg["n_inner"]):
            bx, bp, bs, bh, bw = buf.draw(cfg["batch"])
            w = None
            if cfg["is_weight"]:
                # already tempered and normalised per (beta,h) cell over the whole buffer;
                # do NOT renormalise on the minibatch, that would restore cross-cell competition
                w = bw.float()
            loss = soft_ce_loss(net, samp if cfg["eta"] < 1.0 else None, bx, bp, bs, bh, cfg, w)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            ema_update(ema, model, cfg["ema"])
            losses.append(loss.item())
        model.eval()
        t_train = time.time() - tt

        log = dict(outer=step0 + outer, loss=float(np.mean(losses)),
                   t_explore=t_explore, t_train=t_train, hours=(time.time() - t0) / 3600,
                   is_alpha=alpha, is_ess_frac=essf,
                   is_alpha_min=astats.get("alpha_min", alpha),
                   is_alpha_med=astats.get("alpha_med", alpha),
                   is_alpha_max=astats.get("alpha_max", alpha),
                   is_bins_own=astats.get("n_bins_own", 0),
                   logw_span=float((buf.logw[: buf.n].max() - buf.logw[: buf.n].min()).item()) if buf.n else 0.0,
                   logw_c_std=float(buf.logw_c[: buf.n].std().item()) if buf.n else 0.0,
                   gen_local_inconsistency=pf.local_logprob_of_current(x1, e_bet, e_h, L, q).mean().item())

        def _state():
            return {"model": model.state_dict(), "ema": ema.state_dict(), "cfg": cfg,
                    "opt": opt.state_dict(), "outer": step0 + outer, "acq": acq.state(),
                    "buf": {"x": buf.x, "p": buf.p, "bet": buf.bet, "h": buf.h,
                            "logw": buf.logw, "pos": buf.pos, "n": buf.n}}
        if cfg["ckpt_every"] and outer % cfg["ckpt_every"] == 0:
            torch.save(_state(), os.path.join(out_dir, "ckpt.pt"))

        if outer % cfg["eval_every"] == 0 or outer == cfg["n_outer"] - 1:
            nS, nH = len(cfg["eval_betas"]), len(cfg["eval_hs"])
            fig, ax = plt.subplots(1, nS, figsize=(4.2 * nS, 3.2), squeeze=False)
            figh, axh = plt.subplots(nS, nH, figsize=(2.3 * nH, 2.1 * nS), squeeze=False)
            # THE mode picture: every sample as a point in the simplex magnetisation plane.
            # The q ordered modes are q lobes at the simplex vertices; the disordered phase is
            # one blob at the origin.  Unlike the order parameter (a radius) this SHOWS which
            # modes are occupied, so a missing mode is a missing lobe.
            figm, axm = plt.subplots(nS, nH, figsize=(2.2 * nH, 2.2 * nS), squeeze=False)
            vtx = pf.simplex_vertices(q).cpu().numpy()
            shots = {}
            bins = np.linspace(-1, 1, 65)
            for si_, s in enumerate(cfg["eval_betas"]):
                ms_mod, ms_ref = [], []
                for hi_, hv in enumerate(cfg["eval_hs"]):
                    key = (round(s, 3), round(hv, 4))
                    xe = unmask_sample(samp, torch.full((cfg["n_eval"],), s, device=dev),
                                       torch.full((cfg["n_eval"],), hv, device=dev),
                                       cfg["n_eval"], cfg["eval_steps"] or cfg["explore_steps"],
                                       dev, D, q, cfg["n_single"])
                    o_u = pf.energy_per_spin(xe, s, hv, L).mean().item()
                    mags = pf.order_parameter(xe, L, q)
                    o_ms = pf.mode_fraction_0(xe, L, q)
                    o_modes = pf.mode_fractions(xe, L, q)
                    k = f"s{s:.2f}/h{hv:+.3f}"
                    log[f"{k}/u_err"] = o_u - ref[key]["u"]
                    log[f"{k}/frac0"] = o_ms
                    log[f"{k}/mode_min"] = float(o_modes.min())
                    log[f"{k}/frac0_err"] = o_ms - ref[key]["ms"]
                    log[f"{k}/mag_hist"] = wandb.Histogram(mags.cpu().numpy(), num_bins=64)
                    ah = axh[si_, hi_]
                    ah.hist(ref[key]["mags"], bins=bins, density=True, color="0.8", label="PT")
                    ah.hist(mags.cpu().numpy(), bins=bins, density=True, histtype="step",
                            color="C3", label="sampler")
                    ah.set_title(f"s={s:.2f} h={hv:+.3f}", fontsize=7)
                    ah.set_yticks([])

                    am = axm[si_, hi_]
                    rv = ref[key].get("mvec")
                    if rv is not None:
                        am.scatter(rv[:, 0], rv[:, 1], s=7, c="0.62", lw=0, label="PT")
                    shots[key] = xe[:3].reshape(-1, L, L).cpu().numpy()
                    mv = pf.magnetisation_vector(xe, L, q).cpu().numpy()
                    am.scatter(mv[:, 0], mv[:, 1], s=4, c="C3", lw=0, alpha=0.55, label="sampler")
                    for a_ in range(q):          # simplex vertices = the q pure states
                        am.plot([0, vtx[a_, 0]], [0, vtx[a_, 1]], c="0.5", lw=0.6, zorder=0)
                        am.annotate(str(a_), (vtx[a_, 0] * 1.12, vtx[a_, 1] * 1.12),
                                    fontsize=7, ha="center", va="center", color="0.3")
                    am.set_xlim(-1.25, 1.25); am.set_ylim(-1.25, 1.25)
                    am.set_aspect("equal"); am.set_xticks([]); am.set_yticks([])
                    # all q shares, not just the minimum: a zero entry is a lost mode
                    shares = "/".join(f"{v:.2f}" for v in o_modes.tolist())
                    rshares = ("  ref " + "/".join(f"{v:.2f}" for v in ref[key]["modes"])
                               if ref[key].get("modes") else "")
                    am.set_title(f"b={s:.2f} h={hv:+.3f}\n{shares}{rshares}", fontsize=6)
                    # log the per-mode shares so the picture has a number beside it
                    for a_ in range(q):
                        log[f"{k}/mode_{a_}"] = float(o_modes[a_])
                    ms_mod.append(o_ms)
                    ms_ref.append(ref[key]["ms"])
                a = ax[0, si_]
                a.plot(cfg["eval_hs"], ms_ref, "k.-", label="PT")
                a.plot(cfg["eval_hs"], ms_mod, "C3o--", label="sampler")
                a.axvline(0, ls=":", c="0.7"); a.axhline(0, ls=":", c="0.7")
                a.set_title(f"beta={s:.2f}", fontsize=9); a.set_xlabel("h"); a.set_ylabel("mean spin")
            ax[0, 0].legend(fontsize=8)
            fig.tight_layout()
            log["ms_vs_h"] = wandb.Image(fig)
            plt.close(fig)
            axh[0, 0].legend(fontsize=6)
            figh.tight_layout()
            log["mag_hists"] = wandb.Image(figh)
            plt.close(figh)
            axm[0, 0].legend(fontsize=6, markerscale=3, loc="upper left")
            figm.suptitle(f"simplex magnetisation, round {step0 + outer} "
                          f"(grey = PT reference, red = sampler; labels are the q pure states)",
                          fontsize=8)
            figm.tight_layout(rect=(0, 0, 1, 0.96))
            log["mode_plane"] = wandb.Image(figm)
            plt.close(figm)

            # raw lattice configurations.  Rendered through matplotlib with a DISCRETE colour
            # map rather than handed to wandb.Image as an array: integer tokens 0..q-1 are read
            # as 0..255 grey levels and come out solid black, and the float-rescale behaviour
            # differs between wandb versions.  A figure is unambiguous everywhere.
            nsh = 3
            figs, axs = plt.subplots(nS * nsh, nH, figsize=(1.5 * nH, 1.5 * nS * nsh),
                                     squeeze=False)
            cmap = plt.get_cmap("viridis", q)
            for si_, s_ in enumerate(cfg["eval_betas"]):
                for hi_, hv in enumerate(cfg["eval_hs"]):
                    xs_ = shots.get((round(s_, 3), round(hv, 4)))
                    for r_ in range(nsh):
                        a_ = axs[si_ * nsh + r_, hi_]
                        a_.set_xticks([]); a_.set_yticks([])
                        if xs_ is not None and r_ < xs_.shape[0]:
                            a_.imshow(xs_[r_], cmap=cmap, vmin=-0.5, vmax=q - 0.5,
                                      interpolation="nearest")
                        if r_ == 0:
                            a_.set_title(f"b={s_:.2f} h={hv:+.3f}", fontsize=6)
            figs.suptitle(f"lattice samples, round {step0 + outer} "
                          f"({q} states = {q} colours)", fontsize=8)
            figs.tight_layout(rect=(0, 0, 1, 0.97))
            log["samples"] = wandb.Image(figs)
            plt.close(figs)
            # acquisition map: where the adaptive half is spending its samples
            figa, axa = plt.subplots(1, 2, figsize=(9, 3.2))
            ext = [-cfg["h_max"], cfg["h_max"], cfg["beta_lo"], cfg["beta_hi"]]
            mp = acq.map().cpu().numpy()
            kw = (dict(cmap="coolwarm", vmin=-1, vmax=1) if cfg["acq"] == "grad_m"
                  else dict(cmap="viridis"))
            axa[0].imshow(mp, origin="lower", aspect="auto", extent=ext, **kw)
            axa[0].set_title({"fp": "fixed-point residual", "fisher": "Tr Cov[grad_theta U]"}
                             .get(cfg["acq"], "binned mean spin"))
            axa[0].set_xlabel("h"); axa[0].set_ylabel("beta")
            axa[1].imshow(acq.weights().cpu().numpy(), origin="lower", aspect="auto", extent=ext,
                          cmap="viridis")
            axa[1].set_title("sampling weights"); axa[1].set_xlabel("h")
            figa.tight_layout()
            log["acq_map"] = wandb.Image(figa)
            plt.close(figa)
            ck = _state()
            torch.save(ck, os.path.join(out_dir, "ckpt.pt"))
            if cfg["keep_every"] and (outer // cfg["eval_every"]) % cfg["keep_every"] == 0:
                torch.save(ck, os.path.join(out_dir, f"ckpt_{step0 + outer:06d}.pt"))
            hmid = sorted(cfg["eval_hs"], key=abs)[1:3]
            msg = " ".join(f"b{s:.2f}h{hv:+.3f}:{log[f's{s:.2f}/h{hv:+.3f}/frac0_err']:+.2f}"
                           for s in cfg["eval_betas"] for hv in hmid)
            print(f"[{step0+outer:5d}] loss={log['loss']:.4f} frac0_err@crossover {msg} "
                  f"({t_explore:.1f}s+{t_train:.1f}s)", flush=True)

        wandb.log(log, step=step0 + outer)

    wandb.finish()


if __name__ == "__main__":
    cfg = dict(CFG)
    for arg in sys.argv[1:]:
        k, v = arg.split("=", 1)
        d = CFG[k]
        if v.startswith("[") or v == "None":
            cfg[k] = eval(v)
        elif isinstance(d, (int, float)):
            cfg[k] = type(d)(v)
        else:
            cfg[k] = v
    main(cfg)
