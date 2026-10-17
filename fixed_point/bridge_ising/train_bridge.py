"""Fixed-point transport between two Ising distributions with a HOLLOW two-head network.

    prior   pi_0 = Ising(beta_0, h=0)  -- high temperature, disordered; samples are GIVEN
    target  pi_1 = Ising(beta_1, h=0)  -- low temperature, ordered; only its ENERGY is given

Bridge (per coordinate, mixture interpolant, alpha(t) = t):
    kappa_t(c | x0, x1) = (1-t) 1[c=x0] + t 1[c=x1],   d_t kappa = 1[c=x1] - 1[c=x0]
    X_t^d ~ kappa_t^d(. | X_0^d, X_1^d) independently per coordinate.

Two hollow heads (note: hollow_score_phi_losses.md), both seeing X_t^{-d} only:
    q_theta^d(c | X_t^{-d}, t)     soft-label CE against kappa            -> P_t(X_t^d = c | X_t^{-d})
    phi_theta^d(c | X_t^{-d}, t)   all-candidate weighted quadratic (no-division form)
                                   -> E[d_t kappa | X_t^{-d}] / E[kappa | X_t^{-d}]

FIXED POINT.  X_1 comes from the current model, not from pi_1.  Exactly as in the masked sampler,
coordinate d's label is corrected by the target's exact single-site conditional given the rest of
the endpoint, rho_d(a | X_1^{-d}) = sigmoid(2 beta_1 m_d):
    kappa_bar^d(c)     = sum_a rho_d(a | X_1^{-d}) kappa_t(c | X_0^d, a)
    d_t kappa_bar^d(c) = sum_a rho_d(a | X_1^{-d}) d_t kappa_t(c | X_0^d, a)
The hollow head never sees X_t^d, and X_t^{-d} does not depend on X_1^d, so replacing the
coordinate-d label is consistent; pi_1 is a fixed point of the iteration and the correction pulls
q_k toward it.  Labels cost one local energy pass.  No ground-truth target samples anywhere.

SAMPLER (the "minimum-action" CTMC implied by the two heads).  With q^d = P_t(.|x^{-d}) and
phi^d(x^d|x^{-d}) summing over d to d_t log P_t(x), the generator
    R_t(x -> x^{d<-c}) = q^d(c | x^{-d}) [ phi^d(c | x^{-d}) - phi^d(x^d | x^{-d}) ]_+
satisfies the continuity equation exactly (uses sum_c q^d(c) phi^d(c) = 0, which holds because
sum_c d_t kappa = 0).  Simulated by tau-leaping from X_0 ~ pi_0 at t=0 to t=1.

Everything else -- replay buffer, EMA sampler, function-space damping, PT references, checkpoint
and resume, W&B -- is the validated machinery, untouched.
"""

import copy, json, os, sys, time

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ising_energy as ie
from network_bs import BlindSpotNet, hollow_check

CFG = dict(
    L=25, beta0=0.20, beta1=0.60,
    hidden=96, n_blocks=5,
    lr=3e-4, weight_decay=0.0, ema=0.997, grad_clip=1.0,
    eta=0.3,                # function-space damping toward the EMA teacher (both heads)
    lam_phi=1.0,            # L = L_q + lam_phi * L_phi
    is_weight=0,            # tempered path-space importance weights (see ctmc_sample docstring)
    gibbs_rate=0.0,         # lambda: extra Gibbs-corrector rate lambda*q^d(c) in the sampler --
                            # a stationary kernel of p_t (q is its exact conditional), so it adds
                            # mixing without moving the marginals; the discrete analogue of the SDE
                            # noise level.  Reversible, hence identical in the backward kernel.
    pc_sweeps=0,            # predictor-corrector in the buffer sampler: heat-bath sweeps from the score
                            # head q^d(.|x^{-d},t) after every tau-leap step with t >= pc_tmin (colour
                            # groups, no two torus neighbours updated together); 0 = predictor only
    pc_tmin=0.75,
    pc_final=0,             # extra learned sweeps at t = 1 after the last predictor step
    load_buffer=1,          # warm start: reuse the parent checkpoint's buffer (0 = refill from scratch)
    ess_target=0.2,
    phi_loss="wq",          # "wq": all-candidate weighted quadratic, no-division form (note sec. 3,
                            #       recommended); "l2": regress only at the sampled X_t^d against
                            #       d_t log kappa (note sec. 2) -- needs the division, kept for ablation
    n_outer=1000000, n_inner=100, batch=512,
    n_explore=1024, buffer_cap=16384, warmup_outer=10,
    prior_n=32768, prior_sweeps=400,
    prior_sampler="gibbs",  # "gibbs": blocked heat-bath from a random quench -- fine for a
                            # DISORDERED prior.  "pt": parallel tempering in beta -- required
                            # when the prior is itself ORDERED, since Gibbs from a quench then
                            # cannot cross between the two modes and leaves domain-wall
                            # states, so the prior bank would be neither balanced nor
                            # equilibrated and the bridge would inherit both defects.
    ctmc_steps=256,
    eval_every=25, n_eval=2048, ref_sweeps=6000, ref_chains=4096,
    ckpt_every=5, keep_every=8,
    init_from="", step_offset=-1, seed=0, device="cuda",
    out_dir="runs_bridge", run_name=None,
    wandb_project="ising-fixed-point-sampler", wandb_mode="online",
)


class Buffer:
    """Endpoints from the model with their exact target conditionals (the fixed-point labels)."""

    def __init__(self, cap, D, device):
        self.x = torch.zeros(cap, D, dtype=torch.uint8, device=device)
        self.p = torch.zeros(cap, D, dtype=torch.float16, device=device)
        self.logw = torch.zeros(cap, device=device)
        self.cap, self.pos, self.n = cap, 0, 0

    def push(self, x, p, lw=None):
        b = x.shape[0]
        i = (torch.arange(b, device=x.device) + self.pos) % self.cap
        self.x[i] = x.to(torch.uint8); self.p[i] = p.to(torch.float16)
        self.logw[i] = 0.0 if lw is None else lw.float()
        self.pos = (self.pos + b) % self.cap; self.n = min(self.n + b, self.cap)

    def draw(self, b):
        i = torch.randint(0, self.n, (b,), device=self.x.device)
        return self.x[i].long(), self.p[i].float(), self.logw[i]


@torch.no_grad()
def select_alpha(logw, target):
    """largest alpha with ESS(alpha)/n >= target; geometric bisection on log10 alpha."""
    lw = logw.double(); n = lw.numel()
    def ess(a):
        y = a * lw; y = y - y.max(); w = torch.exp(y)
        return float((w.sum() ** 2) / (n * (w * w).sum()))
    if ess(1.0) >= target: return 1.0, ess(1.0)
    lo, hi = -12.0, 0.0
    for _ in range(50):
        m = 0.5 * (lo + hi)
        if ess(10.0 ** m) >= target: lo = m
        else: hi = m
    return 10.0 ** lo, ess(10.0 ** lo)


@torch.no_grad()
def ema_update(ema, model, d):
    for pe, pm in zip(ema.parameters(), model.parameters()): pe.lerp_(pm.detach(), 1.0 - d)
    for be, bm in zip(ema.buffers(), model.buffers()): be.copy_(bm)


_PC_GROUPS = {}


@torch.no_grad()
def corrector_sweeps(model, x, tv, n_sweeps, autocast=True):
    """Heat-bath sweeps whose site conditionals are the score head at time tv.  At t = 1 the head fits
    the target's exact Gibbs conditional (label construction), so this is target MCMC; at t < 1 it is
    heat-bath on the bridge marginal.  Colour groups are updated one at a time (9 on the odd-L torus);
    the learned conditional is nearest-neighbour up to ~1e-2 at distance 2 (measured), so grouped
    updates are a valid heat-bath for it.  Reversible w.r.t. p_t: adds no path weight."""
    L = int(round(x.shape[1] ** 0.5)); key = (L, str(x.device))
    if key not in _PC_GROUPS: _PC_GROUPS[key] = ie._independent_groups(L, x.device).reshape(-1)
    groups = _PC_GROUPS[key]; colours = groups.unique().tolist()
    t = torch.full((x.shape[0],), float(tv), device=x.device)
    for _ in range(n_sweeps):
        for col in colours:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
                ql, _ = model(x, t)
            p1 = torch.softmax(ql.float(), -1)[..., 1]
            new = (torch.rand_like(p1) < p1).long()
            x = torch.where(groups[None] == col, new, x)
    return x


@torch.no_grad()
def ctmc_sample(model, x0, K, autocast=True, return_logw=False, bet0=None, bet1=None, L=None,
                gibbs_rate=0.0, pc_sweeps=0, pc_tmin=0.75, pc_final=0):
    """tau-leap the minimum-action CTMC from X_0 (t=0) to t=1.  x0: [n, D] tokens.

    With return_logw, also returns the path-space importance weight of the endpoint,
        log w = log pi_1~(x_1) - log pi_0~(x_0) + sum_k sum_d [ log B_d(k) - log F_d(k) ],
    the discrete analogue of the tempered-IS note (p_ref = pi_0, whose samples we have).
    F is the forward tau-leap kernel actually sampled from; B is the time-reversed kernel.
    Reverse rates come from the SAME two heads: R~(y -> y^{d<-c}) = R(y^{d<-c} -> y) p(y^{d<-c})/p(y)
    and p(y^{d<-c})/p(y) = q^d(c)/q^d(y^d) (score head), which collapses to
        R~_d = q^d(c) [ phi^d(y^d) - phi^d(c) ]_+           (forward rate with Delta phi negated).
    The heads are hollow, so the network evaluated at (x_{k+1}, t_{k+1}) gives the site-d outputs
    needed for B of step k as well as F of step k+1: one call per step, plus one at t=1.
    Sanity: an untrained model (phi = 0) never flips, path terms vanish, and
    log w = (beta_1 - beta_0) B(x_0), the exact prior -> target importance weight.
    """
    assert not (return_logw and (pc_sweeps or pc_final)), \
        "path weights with corrector steps: B of the predictor step must be scored before the sweep (not implemented)"
    x = x0.clone(); n = x.shape[0]; dt = 1.0 / K
    logF = torch.zeros(n, device=x.device, dtype=torch.float64)
    logB = torch.zeros(n, device=x.device, dtype=torch.float64)
    prev_flip = None            # flips of the previous step, to score with B at the new state

    def heads(xs, tv):
        t = torch.full((n,), tv, device=xs.device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            ql, phi = model(xs, t)
        qp = torch.softmax(ql.float(), -1); phi = phi.float()
        cur, oth = xs, 1 - xs
        q_oth = torch.gather(qp, 2, oth[..., None]).squeeze(-1)
        dphi = (torch.gather(phi, 2, oth[..., None]) - torch.gather(phi, 2, cur[..., None])).squeeze(-1)
        return q_oth, dphi

    for k in range(K):
        q_oth, dphi = heads(x, k / K)
        if return_logw and prev_flip is not None:
            # B for step k-1: from the current state x_k, the reverse process must flip exactly
            # the sites that flipped forward (rate q(target)[ -Delta phi ]_+ toward the OLD value:
            # here the 'other' value relative to x_k is the old value, and Delta phi toward it is
            # -(phi(new) - phi(old)) = -dphi_old ... expressed at x_k: rate = q_oth * (-dphi)_+ )
            rb = (q_oth * ((-dphi).clamp_min(0.0) + gibbs_rate)).double()
            pb = 1.0 - torch.exp(-rb * dt)          # float64: 1e-300 underflows to 0 in float32
            logB += torch.where(prev_flip, torch.log(pb.clamp_min(1e-300)), -rb * dt).sum(-1)
        rate = q_oth * (dphi.clamp_min(0.0) + gibbs_rate)
        pf = 1.0 - torch.exp(-rate * dt)
        flip = torch.rand_like(rate) < pf
        if return_logw:
            r64 = rate.double(); pf64 = 1.0 - torch.exp(-r64 * dt)
            logF += torch.where(flip, torch.log(pf64.clamp_min(1e-300)), -r64 * dt).sum(-1)
        x = torch.where(flip, 1 - x, x)
        prev_flip = flip
        if pc_sweeps and (k + 1) / K >= pc_tmin - 1e-9:
            x = corrector_sweeps(model, x, (k + 1) / K, pc_sweeps, autocast)
    if pc_final:
        x = corrector_sweeps(model, x, 1.0, pc_final, autocast)
    if return_logw:
        q_oth, dphi = heads(x, 1.0)                       # B for the last step, at t = 1
        rb = (q_oth * ((-dphi).clamp_min(0.0) + gibbs_rate)).double(); pb = 1.0 - torch.exp(-rb * dt)
        logB += torch.where(prev_flip, torch.log(pb.clamp_min(1e-300)), -rb * dt).sum(-1)
        logw = (bet1 * ie.bond_sum(x, L) - bet0 * ie.bond_sum(x0, L)).double() + logB - logF
        return x, logw
    return x


def bridge_labels(x0, p, t):
    """Fixed-point soft labels for the mixture bridge, with X_1^d replaced by rho_d.
    x0: [B,D] tokens, p: [B,D] = rho_d(1 | X_1^{-d}), t: [B].  Returns kbar, dk: [B,D,2]."""
    tt = t[:, None]
    k1 = (1.0 - tt) * x0.float() + tt * p
    d1 = p - x0.float()
    return torch.stack([1.0 - k1, k1], -1), torch.stack([-d1, d1], -1)


def losses(model, teacher, xt, t, kbar, dk, cfg, w=None):
    """w: per-sample importance weights (mean 1) applied to the FIT terms only; the damping
    (proximal) terms tie the iterate to its predecessor and are not expectations over the target."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        ql, phi = model(xt, t)
    logq = torch.log_softmax(ql.float(), -1); phi = phi.float()
    eta = cfg["eta"]
    if w is None: w = torch.ones(xt.shape[0], device=xt.device)
    ce_fit = -(kbar * logq).sum(-1).mean(-1)                       # [B]
    if teacher is not None and eta < 1.0:
        with torch.no_grad():
            ql_t, phi_t = teacher(xt, t)
            q_t = torch.softmax(ql_t.float(), -1); phi_t = phi_t.float()
        ce_prox = -(q_t * logq).sum(-1).mean(-1)
        L_q = (w * eta * ce_fit + (1.0 - eta) * ce_prox).mean()
    else:
        L_q = (w * ce_fit).mean()
    if cfg["phi_loss"] == "wq_div":
        # literal sec. 3 form; identical gradient to "wq" up to a phi-independent constant, kept
        # only as a numerical check (needs the clamp: kbar can be exactly 0 at t -> 0)
        L_wq = 0.5 * (kbar * (phi - dk / kbar.clamp_min(1e-6)) ** 2).sum(-1).mean(-1)
    elif cfg["phi_loss"] == "l2":
        # sample X_t^d from the rho-corrected bridge so the regression point is consistent with
        # the corrected label; target = d_t log kbar at that value
        cs = torch.multinomial(kbar.reshape(-1, 2), 1).view(kbar.shape[0], kbar.shape[1], 1)
        tgt = torch.gather(dk, 2, cs) / torch.gather(kbar, 2, cs).clamp_min(1e-6)
        L_wq = 0.5 * ((torch.gather(phi, 2, cs) - tgt) ** 2).squeeze(-1).mean(-1)
    else:
        L_wq = (0.5 * kbar * phi * phi - dk * phi).sum(-1).mean(-1)   # no-division form, [B]
    if L_wq.dim() == 0: L_wq = L_wq.expand(xt.shape[0])
    if teacher is not None and eta < 1.0:
        L_prox = 0.5 * (kbar * (phi - phi_t) ** 2).sum(-1).mean(-1)
        L_phi = (w * eta * L_wq + (1.0 - eta) * L_prox).mean()
    else:
        L_phi = (w * L_wq).mean()
    return L_q + cfg["lam_phi"] * L_phi, L_q.item(), L_wq.mean().item()


def main(cfg):
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"])
    torch.backends.cuda.matmul.allow_tf32 = True
    dev = cfg["device"]; L = cfg["L"]; D = L * L
    b0, b1 = cfg["beta0"], cfg["beta1"]
    os.makedirs(cfg["out_dir"], exist_ok=True)

    model = BlindSpotNet(L, cfg["hidden"], cfg["n_blocks"]).to(dev)
    probe = copy.deepcopy(model).eval()           # randomised copy: zero-init heads would pass trivially
    for p_ in probe.parameters(): torch.nn.init.normal_(p_, std=0.3)
    worst, elsewhere = hollow_check(probe, L, dev, n_sites=32)
    assert worst == 0.0 and elsewhere > 0.0, f"blind spot violated: {worst}, {elsewhere}"
    del probe
    print(f"BlindSpotNet {sum(p.numel() for p in model.parameters())/1e6:.2f}M  "
          f"reach rows={model.reach_rows} cols={model.reach_cols}  hollow check: {worst:.1e} "
          f"(elsewhere {elsewhere:.2f})", flush=True)

    step0 = 0
    if cfg["init_from"]:
        ck = torch.load(cfg["init_from"], map_location=dev, weights_only=False)
        model.load_state_dict(ck["model"]); step0 = int(ck.get("outer", 0)) + 1
        print(f"resumed from {cfg['init_from']} (outer {ck['outer']})", flush=True)
    if cfg["step_offset"] >= 0: step0 = cfg["step_offset"]
    ema = copy.deepcopy(model).eval()
    for p_ in ema.parameters(): p_.requires_grad_(False)
    if cfg["init_from"]: ema.load_state_dict(ck["ema"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    if cfg["init_from"] and "opt" in ck: opt.load_state_dict(ck["opt"])

    # ---- prior samples: GIVEN (sampled once; the high-T Gibbs sampler mixes trivially) --------
    t0 = time.time()
    if cfg["prior_sampler"] == "pt":
        prior = ie.pt_reference(b0, 0.0, L, cfg["prior_n"], cfg["prior_sweeps"], dev, seed=1,
                                n_temps=24).long()
    else:
        prior = ie.gibbs_reference(b0, 0.0, L, cfg["prior_n"], cfg["prior_sweeps"], dev, seed=1).long()
    print(f"prior bank {tuple(prior.shape)} at beta0={b0}: u={float(-ie.bond_sum(prior,L).mean()/D):+.4f} "
          f"(exact {ie.onsager_energy_per_spin(b0,L):+.4f})  |m|={float(ie.magnetisation(prior,L).abs().mean()):.3f} "
          f"frac(+)={float((ie.magnetisation(prior,L) > 0).float().mean()):.3f} "
          f"({time.time()-t0:.0f}s)", flush=True)

    # ---- target reference: diagnostics only (exact Kaufman/Yang + PT for the histogram) ------
    u_exact = ie.onsager_energy_per_spin(b1, L)
    m_exact = ie.yang_spontaneous_magnetisation(b1)
    rp = os.path.join(cfg["out_dir"], f"ref_L{L}_s{b1:.3f}_{cfg['ref_sweeps']}.pt")
    if os.path.exists(rp):
        ref = torch.load(rp, weights_only=False)
    else:
        t0 = time.time()
        xr = ie.pt_reference(b1, 0.0, L, cfg["ref_chains"], cfg["ref_sweeps"], dev, seed=3, n_temps=24)
        ref = dict(m=ie.magnetisation(xr, L).cpu().numpy(),
                   u=float(-ie.bond_sum(xr, L).mean() / D))
        torch.save(ref, rp)
        print(f"target PT reference: u={ref['u']:+.4f} (Kaufman {u_exact:+.4f})  "
              f"|m|={np.abs(ref['m']).mean():.4f} (Yang {m_exact:.4f})  ({time.time()-t0:.0f}s)", flush=True)

    buf = Buffer(cfg["buffer_cap"], D, dev)
    if cfg["init_from"] and cfg["load_buffer"] and "buf" in ck and ck["buf"]["x"].shape == buf.x.shape \
            and not (cfg["is_weight"] and "logw" not in ck["buf"]):   # no weights stored: refill
        buf.x.copy_(ck["buf"]["x"]); buf.p.copy_(ck["buf"]["p"]); buf.pos, buf.n = ck["buf"]["pos"], ck["buf"]["n"]
        if "logw" in ck["buf"]: buf.logw.copy_(ck["buf"]["logw"])

    import wandb
    run = wandb.init(project=cfg["wandb_project"], name=cfg["run_name"], config=cfg, mode=cfg["wandb_mode"])
    out_dir = os.path.join(cfg["out_dir"], cfg["run_name"] or (run.name or "run"))
    os.makedirs(out_dir, exist_ok=True)
    json.dump(cfg, open(os.path.join(out_dir, "cfg.json"), "w"), indent=2)
    t_start = time.time()

    for outer in range(cfg["n_outer"]):
        # ---- A/B/C: endpoints from the model (identity transport during warm-up), exact labels
        te = time.time()
        i0 = torch.randint(0, prior.shape[0], (cfg["n_explore"],), device=dev)
        x0 = prior[i0]
        if outer < cfg["warmup_outer"] and not cfg["init_from"]:
            x1 = x0.clone()
            lw = ((b1 - b0) * ie.bond_sum(x0, L)).double()     # exact prior->target weight
        elif cfg["is_weight"]:
            x1, lw = ctmc_sample(ema, x0, cfg["ctmc_steps"], return_logw=True, bet0=b0, bet1=b1, L=L,
                                 gibbs_rate=cfg["gibbs_rate"])
        else:
            x1 = ctmc_sample(ema, x0, cfg["ctmc_steps"], gibbs_rate=cfg["gibbs_rate"], pc_sweeps=cfg["pc_sweeps"],
                             pc_tmin=cfg["pc_tmin"], pc_final=cfg["pc_final"]); lw = None
        p1 = ie.local_target_p1(x1, b1, 0.0, L)
        buf.push(x1, p1, lw)
        t_exp = time.time() - te

        # ---- D: optimise both hollow heads on bridge states ---------------------------------
        tt = time.time(); model.train(); Ls, Lqs, Lws = [], [], []
        alpha, essf = 1.0, 1.0
        if cfg["is_weight"] and buf.n > 0:
            alpha, essf = select_alpha(buf.logw[: buf.n], cfg["ess_target"])
        for _ in range(cfg["n_inner"]):
            bx1, bp, blw = buf.draw(cfg["batch"])
            w = None
            if cfg["is_weight"]:
                y = alpha * blw.double(); y = y - y.max()
                w = torch.exp(y); w = (w / w.mean().clamp_min(1e-300)).float()
            bx0 = prior[torch.randint(0, prior.shape[0], (cfg["batch"],), device=dev)]
            t = torch.rand(cfg["batch"], device=dev)
            keep = torch.rand(cfg["batch"], D, device=dev) < t[:, None]
            xt = torch.where(keep, bx1, bx0)
            kbar, dk = bridge_labels(bx0, bp, t)
            loss, lq, lw_ = losses(model, ema if cfg["eta"] < 1.0 else None, xt, t, kbar, dk, cfg, w)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"]); opt.step()
            ema_update(ema, model, cfg["ema"]); Ls.append(loss.item()); Lqs.append(lq); Lws.append(lw_)
        model.eval()

        mg = ie.magnetisation(x1, L)
        log = dict(outer=step0 + outer, loss=float(np.mean(Ls)), loss_q=float(np.mean(Lqs)),
                   loss_wq=float(np.mean(Lws)), is_alpha=alpha, is_ess=essf,
                   logw_span=float((buf.logw[: buf.n].max() - buf.logw[: buf.n].min()).item()) if buf.n else 0.0,
                   gen_u=float(-ie.bond_sum(x1, L).mean() / D), gen_absm=float(mg.abs().mean()),
                   gen_m=float(mg.mean()),
                   gen_local_inconsistency=float(-torch.log(torch.where(x1 > 0, p1, 1 - p1).clamp_min(1e-12)).mean()),
                   t_explore=t_exp, t_train=time.time() - tt, hours=(time.time() - t_start) / 3600)

        def _state():
            return {"model": model.state_dict(), "ema": ema.state_dict(), "cfg": cfg,
                    "opt": opt.state_dict(), "outer": step0 + outer,
                    "buf": {"x": buf.x, "p": buf.p, "logw": buf.logw, "pos": buf.pos, "n": buf.n}}
        if cfg["ckpt_every"] and outer % cfg["ckpt_every"] == 0:
            torch.save(_state(), os.path.join(out_dir, "ckpt.pt"))

        if outer % cfg["eval_every"] == 0:
            xe = ctmc_sample(ema, prior[torch.randint(0, prior.shape[0], (cfg["n_eval"],), device=dev)],
                             cfg["ctmc_steps"], gibbs_rate=cfg["gibbs_rate"])
            me = ie.magnetisation(xe, L); ue = float(-ie.bond_sum(xe, L).mean() / D)
            log.update({"eval/u": ue, "eval/u_err": ue - u_exact, "eval/absm": float(me.abs().mean()),
                        "eval/absm_err": float(me.abs().mean()) - m_exact, "eval/mean_m": float(me.mean()),
                        "eval/frac_plus": float((me > 0).float().mean())})
            if cfg["pc_sweeps"] or cfg["pc_final"]:          # the sampler actually used for the buffer
                xp = ctmc_sample(ema, prior[torch.randint(0, prior.shape[0], (cfg["n_eval"],), device=dev)],
                                 cfg["ctmc_steps"], gibbs_rate=cfg["gibbs_rate"], pc_sweeps=cfg["pc_sweeps"],
                                 pc_tmin=cfg["pc_tmin"], pc_final=cfg["pc_final"])
                mp = ie.magnetisation(xp, L); up = float(-ie.bond_sum(xp, L).mean() / D)
                log.update({"eval/u_err_pc": up - u_exact, "eval/absm_pc": float(mp.abs().mean()),
                            "eval/mean_m_pc": float(mp.mean())})
            fig, ax = plt.subplots(figsize=(6, 3))
            bins = np.linspace(-1, 1, 81)
            ax.hist(ie.magnetisation(prior[:4096], L).cpu().numpy(), bins=bins, density=True, color="#c7c7c7", label=f"prior β={b0}")
            ax.hist(ref["m"], bins=bins, density=True, color="#8a8a8a", alpha=.6, label=f"target PT β={b1}")
            ax.hist(me.cpu().numpy(), bins=bins, density=True, histtype="step", lw=1.8, color="C3", label="transport endpoints")
            ax.set_yscale("log"); ax.set_xlabel("m"); ax.legend(fontsize=7); fig.tight_layout()
            log["mag_hist"] = wandb.Image(fig); plt.close(fig)
            ck2 = _state(); torch.save(ck2, os.path.join(out_dir, "ckpt.pt"))
            if cfg["keep_every"] and (outer // cfg["eval_every"]) % cfg["keep_every"] == 0:
                torch.save(ck2, os.path.join(out_dir, f"ckpt_{step0+outer:06d}.pt"))
            print(f"[{step0+outer:5d}] loss={log['loss']:.4f} (q {log['loss_q']:.4f}, wq {log['loss_wq']:+.4f}) "
                  f"u={ue:+.4f} (err {ue-u_exact:+.4f}) |m|={float(me.abs().mean()):.3f} (exact {m_exact:.3f}) "
                  f"mean_m={float(me.mean()):+.3f} ({t_exp:.0f}s+{log['t_train']:.0f}s)", flush=True)
            if "eval/u_err_pc" in log:
                print(f"        with corrector: err {log['eval/u_err_pc']:+.4f} |m|={log['eval/absm_pc']:.3f} "
                      f"mean_m={log['eval/mean_m_pc']:+.3f}", flush=True)
        wandb.log(log, step=step0 + outer)
    wandb.finish()


if __name__ == "__main__":
    cfg = dict(CFG)
    for a in sys.argv[1:]:
        k, v = a.split("=", 1); d = CFG[k]
        cfg[k] = eval(v) if (v.startswith("[") or v == "None") else (type(d)(v) if isinstance(d, (int, float)) else v)
    main(cfg)
