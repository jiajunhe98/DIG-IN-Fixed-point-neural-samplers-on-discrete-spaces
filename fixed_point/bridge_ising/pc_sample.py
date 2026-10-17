"""Predictor-corrector sampling for the hollow-score bridge (beta 0.20 -> 0.60, L=25).

predictor  : one tau-leap step of the learned minimum-action CTMC, rates q^d(c)[phi^d(c)-phi^d(x^d)]_+
corrector  : heat-bath sweeps whose site conditionals are the learned score head q^d(. | x^{-d}, t)
             (at t=1 this IS the target's Gibbs conditional; at t<1 it is the bridge marginal's),
             applied colour-group by colour-group (no two torus neighbours updated together).
Also run: final-only corrector sweeps at t=1 (learned and exact conditionals), plain exact Gibbs
MCMC from the prior with the same sweep budgets, and validity diagnostics of the learned corrector.
usage: python pc_sample.py NAME CKPT OUT_JSON [N] [GPU]
"""
import os, sys, json, time, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ising_energy as ie
from network_bs import BlindSpotNet

name, ckpath, out_json = sys.argv[1:4]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 1024
gpu = sys.argv[5] if len(sys.argv) > 5 else "0"
dev = f"cuda:{gpu}"; torch.cuda.set_device(dev)
L = 25; D = L * L; B0, B1 = 0.20, 0.60; K = 256
SP = os.path.dirname(os.path.abspath(out_json)) or "."
torch.manual_seed(0)

ck = torch.load(ckpath, map_location=dev, weights_only=False); c = ck["cfg"]
net = BlindSpotNet(L, c["hidden"], c["n_blocks"]).to(dev).eval(); net.load_state_dict(ck["ema"])
print(f"== {name}: ckpt outer {ck['outer']}  N={N} K={K}  dev={dev}", flush=True)

groups = ie._independent_groups(L, dev).reshape(-1)          # [D] colour codes
colours = groups.unique().tolist()
u_exact = ie.onsager_energy_per_spin(B1, L); m_exact = ie.yang_spontaneous_magnetisation(B1)

def stats(x):
    m = ie.magnetisation(x, L).float(); u = (-ie.bond_sum(x, L).float() / D)
    return dict(m=m.cpu().tolist(), u=float(u.mean()), u_err=float(u.mean()) - u_exact,
                absm=float(m.abs().mean()), mean_m=float(m.mean()),
                frac_committed=float((m.abs() > 0.9).float().mean()),
                samples=x[:12].reshape(12, L, L).to(torch.uint8).cpu().tolist())

@torch.no_grad()
def heads(x, tv):
    t = torch.full((x.shape[0],), float(tv), device=dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        ql, phi = net(x, t)
    return torch.softmax(ql.float(), -1), phi.float()

def gat(a, idx): return torch.gather(a, 2, idx[..., None]).squeeze(-1)

@torch.no_grad()
def predictor_step(x, tv, dt):
    qp, phi = heads(x, tv)
    cur, oth = x, 1 - x
    rate = gat(qp, oth) * (gat(phi, oth) - gat(phi, cur)).clamp_min(0.0)
    flip = torch.rand_like(rate) < 1.0 - torch.exp(-rate * dt)
    return torch.where(flip, 1 - x, x)

@torch.no_grad()
def corrector_sweeps(x, tv, n_sweeps, exact=False):
    """heat-bath sweeps; colour groups updated one at a time (valid Gibbs for a nearest-neighbour
    conditional, approximate for the learned one at t<1 whose receptive field is wider)."""
    for _ in range(n_sweeps):
        for col in colours:
            p1 = ie.local_target_p1(x, B1, 0.0, L) if exact else heads(x, tv)[0][..., 1]
            new = (torch.rand_like(p1) < p1).long()
            x = torch.where(groups[None] == col, new, x)
    return x

@torch.no_grad()
def pc_sample(x0, n_corr=0, t_min=0.0):
    x = x0.clone(); dt = 1.0 / K
    for k in range(K):
        x = predictor_step(x, k / K, dt)
        tv = (k + 1) / K
        if n_corr > 0 and tv >= t_min - 1e-9:
            x = corrector_sweeps(x, tv, n_corr)
    return x

def sweep_curve(x, exact, snaps, tv=1.0):
    """run corrector sweeps at t=tv from x, recording stats at the given cumulative sweep counts."""
    out = {}; done = 0
    for s in snaps:
        x = corrector_sweeps(x, tv, s - done, exact=exact); done = s
        out[str(s)] = stats(x); print(f"      {done:5d} sweeps: u_err {out[str(s)]['u_err']:+.4f} |m| {out[str(s)]['absm']:.3f}", flush=True)
    return out, x

# ---------------------------------------------------------------- data
prior = ie.gibbs_reference(B0, 0.0, L, N, 300, dev, seed=11).long()
ref_path = SP + "pt_ref_s030_x.pt"
try:
    ref = torch.load(ref_path, map_location=dev, weights_only=False)
except FileNotFoundError:
    t0 = time.time(); ref = ie.pt_reference(B1, 0.0, L, 2048, 4000, dev, seed=3, n_temps=24).long()
    torch.save(ref.cpu(), ref_path); print(f"   PT reference built ({time.time()-t0:.0f}s)", flush=True)
ref = ref.to(dev).long()
res = dict(name=name, outer=int(ck["outer"]), N=N, K=K, u_exact=u_exact, m_exact=m_exact,
           prior=stats(prior), ref=stats(ref), runs={}, final={}, diag={})
print(f"   prior u={res['prior']['u']:+.4f} |m|={res['prior']['absm']:.3f};  PT ref u={res['ref']['u']:+.4f} "
      f"(exact {u_exact:+.4f}) |m|={res['ref']['absm']:.3f} (exact {m_exact:.3f})", flush=True)

# ---------------------------------------------------------------- validity diagnostics of the corrector
with torch.no_grad():
    xP = pc_sample(prior)                                          # predictor-only endpoints
    for lab, xx in (("endpoints", xP), ("pt_ref", ref[:N])):
        q1 = heads(xx, 1.0)[0][..., 1]; pe = ie.local_target_p1(xx, B1, L)
        d = (q1 - pe).abs()
        res["diag"][f"q1_vs_exact_{lab}"] = dict(mean=float(d.mean()), p99=float(d.flatten().kthvalue(int(0.99 * d.numel())).values), max=float(d.max()))
    # invariance: PT samples must be left where they are by the learned corrector at t=1
    xi = corrector_sweeps(ref[:N].clone(), 1.0, 64)
    res["diag"]["invariance_64sweeps"] = dict(before=dict(u=res["ref"]["u"], absm=res["ref"]["absm"]), after={k: v for k, v in stats(xi).items() if k in ("u", "u_err", "absm", "mean_m")})
    # locality of the learned conditional at intermediate t: |dq_d| when a site at distance r flips
    loc = {}
    for tv in (0.25, 0.5, 0.75, 0.9, 1.0):
        xt = torch.where(torch.rand(N, D, device=dev) < tv, xP, prior)      # bridge state at t
        base = heads(xt, tv)[0][..., 1]
        d0 = 12 * L + 12; row = {}
        for r in range(1, 7):
            y = xt.clone(); j = 12 * L + (12 + r); y[:, j] = 1 - y[:, j]
            row[str(r)] = float((heads(y, tv)[0][..., 1][:, d0] - base[:, d0]).abs().mean())
        loc[str(tv)] = row
    res["diag"]["locality"] = loc
print("   diag:", json.dumps({k: v for k, v in res["diag"].items() if k != "locality"}), flush=True)
print("   locality |dq| vs distance:", json.dumps(loc), flush=True)

# ---------------------------------------------------------------- predictor / predictor-corrector
configs = [("P", 0, 0.0), ("PC1_late", 1, 0.75), ("PC1", 1, 0.0), ("PC4", 4, 0.0)]
for lab, nc, tmin in configs:
    t0 = time.time(); x = xP if lab == "P" else pc_sample(prior, nc, tmin)
    res["runs"][lab] = dict(n_corr=nc, t_min=tmin, sweeps=int(nc * round(K * (1 - tmin))), **stats(x))
    r = res["runs"][lab]
    print(f"   {lab:9s} (corr sweeps {r['sweeps']:5d}): u_err {r['u_err']:+.4f}  |m| {r['absm']:.3f}  mean_m {r['mean_m']:+.3f}  committed {r['frac_committed']:.2f}  ({time.time()-t0:.0f}s)", flush=True)

# ---------------------------------------------------------------- final-only corrector at t=1, and plain MCMC
snaps = [16, 64, 256, 1024]
print("   final-only sweeps at t=1, learned conditional, from P endpoints:", flush=True)
res["final"]["learned_from_P"], _ = sweep_curve(xP.clone(), False, snaps)
print("   final-only sweeps at t=1, exact conditional, from P endpoints:", flush=True)
res["final"]["exact_from_P"], _ = sweep_curve(xP.clone(), True, snaps + [4096])
print("   plain exact Gibbs MCMC from the prior (no transport):", flush=True)
res["final"]["exact_from_prior"], _ = sweep_curve(prior.clone(), True, snaps + [4096])
json.dump(res, open(out_json, "w"))
print("PC_DONE", flush=True)
