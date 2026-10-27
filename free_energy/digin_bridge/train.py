"""DIG-IN bridge between two ALDP Boltzmann distributions, and log Z1/Z0.

FEAT's ALDP tasks in internal coordinates (target.TASKS): endpoint p_0 is
given by MD samples, endpoint p_1 only by its energy.

  --task aldps   vacuum -> implicit solvent             FEAT reference  29.43
  --task aldpt   vacuum S_b (C7eq/C5) -> S_a (C7ax)     FEAT reference  -4.25

The loop is that of the Ising bridge (DIG-IN App. F.1) on the binned target
of target.py:

  A. exploration  the EMA model tau-leaps n_rollout vacuum frames from t = 0
                  to t = 1, recording log B - log F of every path; endpoints
                  are dequantized uniformly in their cells
  B. labels       endpoint 1's heat-bath table rho_d(. | X_1^{-d}) of every
                  coordinate at the endpoint, one batched energy scan
  C. push         (bins, labels, path log-weight) into a replay buffer; the
                  tempering exponent nu is re-chosen on the whole buffer so
                  that ESS(w^nu) >= target_ess
  D. fit          rollout_every steps: buffer endpoint X_1, an independent
                  vacuum frame X_0, t ~ U(0, 1), X_t from the mixture
                  interpolant; both hollow heads fit the rho-corrected labels
                  (bridge.losses), damped towards the EMA model
  E. refresh      the EMA model is the next sampler

The path weight of a vacuum frame z_0 transported to the dequantized z_1 is

    log w = -U_1(z_1) + U_0(z_0) + sum_k [log B_k - log F_k]

(the uniform cells have one volume, which cancels), E[w] = Z_1 / Z_0 for any
network, and log mean w estimates log Z_1 / Z_0.  Evaluation transports
held-out endpoint-0 frames; the final evaluation transports one path per
endpoint-0 MD frame and saves every path's endpoint and weight to <out>/final.

    python -m digin_bridge.train --task aldps --out runs/aldps_s0 --seed 0
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import signal
import sys
import time
import uuid

import numpy as np
import torch

from digin_bridge import bridge as br
from digin_bridge import data, metrics
from digin_bridge import system as sy
from digin_bridge import zmatrix as zm
from digin_bridge.model import HollowTransformer
from digin_bridge.target import BridgeTarget

FINAL_FIELDS = ("z", "x", "energy", "log_w", "log_w_endpoint", "rama", "start_frame", "jumps")
FINAL_SEED = 1_000_003
FIELD_DOC = {
    "z": "(n, 60) float64 endpoint internal coordinates (nm, rad), in the order of zmatrix.SPEC",
    "x": "(n, 22, 3) float32 endpoint positions in nm, atom order of topology.pdb, CA at the origin",
    "energy": "(n,) float64 endpoint-1 potential energy in kJ/mol (aldps: OBC1 solvent, aldpt: vacuum)",
    "log_w": "(n,) float64 path log-weight -U_1(z_1) + U_0(z_0) + sum_k [log B_k - log F_k]; "
             "log mean exp(log_w) estimates log Z_1/Z_0; -inf for an aldpt endpoint outside S_a",
    "log_w_endpoint": "(n,) float64 its endpoint part -U_1(z_1) + U_0(z_0), without the path terms",
    "rama": "(n, 2) float32 (phi, psi) of the endpoint in radians",
    "start_frame": "(n,) int64 index of the path's starting frame in FEAT's vacuum MD (aldp_0.0_align.pt)",
    "jumps": "(n,) float32 coordinate jumps along the path",
}


def temper_exponent(log_w: torch.Tensor, target_ess: float, n_bisect: int = 50) -> tuple[float, float]:
    """The largest nu in [0, 1] with ESS(w^nu) / n >= target_ess."""
    lw = torch.nan_to_num(log_w.double(), nan=-1e300, posinf=-1e300, neginf=-1e300)
    n = lw.shape[0]

    def frac(nu):
        x = nu * lw
        return float((2 * torch.logsumexp(x, 0) - torch.logsumexp(2 * x, 0)).exp() / n)

    if frac(1.0) >= target_ess:
        return 1.0, frac(1.0)
    lo, hi = 0.0, 1.0
    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if frac(mid) >= target_ess else (lo, mid)
    return lo, frac(lo)


def floor_impossible(log_w: torch.Tensor, gap: float) -> tuple[torch.Tensor, float]:
    """For tempering only: an endpoint outside endpoint 1's support (aldpt, log w = -inf)
    gets the lowest finite log w minus `gap`.  With -inf, w^nu = 0 for every nu > 0, so a
    buffer with few endpoints inside S_a forces nu -> 0 onto those few and drops the labels
    that pull phi into S_a; with the floor, nu = 1 still gives them ~0 weight."""
    fin = torch.isfinite(log_w)
    floor = float(log_w[fin].min()) - gap if fin.any() else 0.0
    return torch.where(fin, log_w, torch.full_like(log_w, floor)), floor


def log_ratio_stats(log_w: torch.Tensor) -> dict:
    """log mean w (= log Z1/Z0), its delta-method standard error, ESS fraction."""
    lw = log_w.double()
    n = lw.shape[0]
    lse = torch.logsumexp(lw, 0)
    ess = float((2 * lse - torch.logsumexp(2 * lw, 0)).exp() / n)
    return {"log_ratio": float(lse - math.log(n)), "log_ratio_se": math.sqrt(max(1.0 / max(ess, 1e-300) - 1.0, 0.0) / n),
            "ess": ess, "log_w_mean": float(lw.mean())}


def grouped_se(log_w: np.ndarray, group: np.ndarray) -> float:
    """Standard error of log mean w with all paths from one starting frame as one unit (they share
    x_0, so they are not independent): the ratio estimator sum_i S_i / sum_i c_i over frames i."""
    lw = np.asarray(log_w, dtype=np.float64)
    fin = np.isfinite(lw)
    if not fin.any():
        return float("nan")
    w = np.where(fin, np.exp(np.where(fin, lw, 0.0) - lw[fin].max()), 0.0)
    _, inv = np.unique(group, return_inverse=True)
    S, c = np.bincount(inv, weights=w), np.bincount(inv).astype(np.float64)
    G, R = len(S), S.sum() / c.sum()
    if G < 2 or R <= 0:
        return float("nan")
    return float(math.sqrt(((S - R * c) ** 2).sum() / (G * (G - 1))) / (R * c.mean()))


def remove_parts(pdir: str) -> None:
    """Delete the merged parts, best effort: blobfuse can report the directory non-empty for a while."""
    if not os.path.isdir(pdir):
        return
    for f in os.listdir(pdir):
        try:
            os.remove(os.path.join(pdir, f))
        except OSError as e:
            print(f"could not remove {f}: {e}", flush=True)
    try:
        os.rmdir(pdir)
    except OSError as e:
        print(f"could not remove {pdir}: {e}", flush=True)


class Buffer:
    def __init__(self, cap, D, L, device):
        self.cap, self.n, self.ptr = cap, 0, 0
        self.x = torch.zeros(cap, D, dtype=torch.uint8, device=device)
        self.rho = torch.zeros(cap, D, L, dtype=torch.float16, device=device)
        self.log_w = torch.zeros(cap, dtype=torch.float64, device=device)

    def push(self, x, rho, log_w):
        k = x.shape[0]
        idx = (self.ptr + torch.arange(k, device=x.device)) % self.cap
        self.x[idx], self.rho[idx], self.log_w[idx] = x.to(torch.uint8), rho.to(torch.float16), log_w
        self.ptr = (self.ptr + k) % self.cap
        self.n = min(self.n + k, self.cap)

    def sample(self, b):
        i = torch.randint(0, self.n, (b,), device=self.x.device)
        rho = self.rho[i].float()
        return self.x[i].long(), rho / rho.sum(-1, keepdim=True), self.log_w[i]


class Trainer:
    def __init__(self, a):
        self.a = a
        self.device = torch.device(a.device if a.device != "auto" else
                                   ("cuda" if torch.cuda.is_available() else "cpu"))
        torch.manual_seed(a.seed); np.random.seed(a.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if a.label_chunk <= 0:      # configurations per energy call, ~2 GB of temporaries per 16 GB of GPU
            gb = (torch.cuda.get_device_properties(self.device).total_memory / 2 ** 30
                  if self.device.type == "cuda" else 8.0)
            a.label_chunk = int(32768 * max(gb / 16.0, 0.5))
        self.tg = BridgeTarget(a.bins, a.n_sigma, a.branch_half, a.omega_half, self.device,
                               label_chunk=a.label_chunk,
                               label_dtype=torch.float64 if a.label_dtype == "float64" else torch.float32,
                               task=a.task)
        self.ref_log_ratio = data.REF[a.task]
        D, L = self.tg.D, self.tg.L
        self.load_endpoints()
        self.net = HollowTransformer(D, L, a.d_model, a.layers, a.readout, a.heads, kinds=self.tg.kind).to(self.device)
        self.ema = copy.deepcopy(self.net).requires_grad_(False)
        self.opt = torch.optim.AdamW(self.net.parameters(), lr=a.lr, weight_decay=a.wd)
        self.buf = Buffer(a.buffer, D, L, self.device)
        self.nu, self.nu_ess = 1.0, 1.0
        self.lw_floor = 0.0
        self.step = 0
        self.energy_calls = 0
        self.stop = False
        self.wandb = None
        print(f"init: task={a.task} seed={a.seed} eta={a.eta} target_ess={a.target_ess} L={a.bins} "
              f"steps={a.steps} device={self.device} "
              f"params={sum(p.numel() for p in self.net.parameters()) / 1e6:.1f}M", flush=True)

    # -- data ------------------------------------------------------------------
    @torch.no_grad()
    def load_endpoints(self):
        """Endpoint-0 frames split into train / test banks with their -U_0, and a
        subsample of the endpoint-1 MD for the evaluation plots."""
        a, tg = self.a, self.tg
        z0, z1, self.frac_inside, frame0 = data.endpoint_frames(tg, data.ensure(a.data_root))
        self.n_frames0 = len(z0)
        g = torch.Generator().manual_seed(1234)                # one split for every seed
        perm = torch.randperm(len(z0), generator=g)
        z0, frame0 = torch.as_tensor(z0)[perm], torch.as_tensor(frame0)[perm]
        n_test = int(len(z0) * a.test_frac)
        b0, _ = tg.to_bins(z0)
        lp0 = torch.cat([-tg.reduced(z0[i:i + 16384].to(self.device))[0] for i in range(0, len(z0), 16384)])
        b0 = b0.to(torch.uint8).to(self.device)
        self.test_b0, self.test_lp0 = b0[:n_test], lp0[:n_test]
        self.test_frame = frame0[:n_test].to(self.device)
        self.train_b0, self.train_lp0 = b0[n_test:], lp0[n_test:]
        self.z1_all = torch.as_tensor(z1)
        zr = self.z1_all[torch.randperm(len(z1), generator=g)[:a.ref_n]].to(self.device)
        phi, psi = tg.phi_psi(zr)
        self.ref = {"phi": phi.cpu().numpy(), "psi": psi.cpu().numpy(), "energy": tg.energy1(zr).cpu().numpy()}
        phv, psv = tg.phi_psi(z0[:a.ref_n])
        self.start = {"phi": phv.numpy(), "psi": psv.numpy()}
        print(f"{a.task}: endpoint-0 frames {len(self.train_b0)} train, {len(self.test_b0)} test; "
              f"endpoint-1 frames {len(z1)}; MD inside the box {self.frac_inside[0]:.6f} / {self.frac_inside[1]:.6f}",
              flush=True)

    @torch.no_grad()
    def full_reference(self) -> dict:
        """(phi, psi, energy) of every endpoint-1 MD frame, for the final evaluation."""
        z1 = self.z1_all
        phi, psi = self.tg.phi_psi(z1)
        E = torch.cat([self.tg.energy1(z1[i:i + 16384].to(self.device)).cpu() for i in range(0, len(z1), 16384)])
        return {"phi": phi.numpy(), "psi": psi.numpy(), "energy": E.double().numpy()}

    # -- bookkeeping -----------------------------------------------------------
    def ckpt_paths(self):
        return [os.path.join(self.a.out, f"ckpt_{k}.pt") for k in (0, 1)]

    def save(self):
        path = self.ckpt_paths()[(self.step // max(self.a.ckpt_every, 1)) % 2]
        torch.save({"net": self.net.state_dict(), "ema": self.ema.state_dict(), "opt": self.opt.state_dict(),
                    "step": self.step, "energy_calls": self.energy_calls}, path + ".tmp")
        os.replace(path + ".tmp", path)

    def load(self) -> bool:
        best = None
        for p in self.ckpt_paths():
            if os.path.exists(p):
                try:
                    c = torch.load(p, map_location=self.device)
                except Exception as e:  # noqa: BLE001  (a kill during a write)
                    print(f"skipping unreadable checkpoint {p}: {e}", flush=True)
                    continue
                if best is None or c["step"] > best["step"]:
                    best = c
        if best is None:
            return False
        self.net.load_state_dict(best["net"]); self.ema.load_state_dict(best["ema"])
        self.opt.load_state_dict(best["opt"]); self.step = best["step"]
        self.energy_calls = best.get("energy_calls", 0)
        print(f"resumed from step {self.step}", flush=True)
        return True

    def init_wandb(self):
        a = self.a
        if a.no_wandb:
            return
        import wandb
        name = a.wandb_name or os.path.basename(os.path.normpath(a.out))
        idf = os.path.join(a.out, "wandb_id.txt")
        rid = open(idf).read().strip() if os.path.exists(idf) else ""
        if not rid and os.path.exists(os.path.join(a.out, "history.jsonl")):
            # resuming, but the id file was lost (a preemption between truncate and write):
            # find the run by name and group so the history continues in the same W&B run
            try:
                api = wandb.Api(timeout=60)
                found = api.runs(f"{a.wandb_entity or api.default_entity}/{a.wandb_project}",
                                 filters={"display_name": name, "group": a.wandb_group})
                rid = found[0].id if len(found) else ""
            except Exception as e:  # noqa: BLE001
                print(f"could not look up the W&B run {name}: {e}", flush=True)
        rid = rid or uuid.uuid4().hex[:12]
        with open(idf + ".tmp", "w") as f:
            f.write(rid)
        os.replace(idf + ".tmp", idf)
        self.wandb = wandb.init(project=a.wandb_project, entity=a.wandb_entity, group=a.wandb_group, name=name,
                                id=rid, resume="allow", config=vars(a), dir=a.out,
                                tags=["bridge", a.task, f"eta={a.eta}", f"ess={a.target_ess}", f"L={a.bins}",
                                      f"seed={a.seed}"])
        wandb.define_metric("step")
        wandb.define_metric("*", step_metric="step")

    def log(self, d: dict, images: dict | None = None):
        # an aldpt endpoint outside S_a has log w = -inf; keep the logs strict JSON
        d = {"step": self.step, **{k: v for k, v in d.items() if not (isinstance(v, float) and not math.isfinite(v))}}
        with open(os.path.join(self.a.out, "history.jsonl"), "a") as f:
            f.write(json.dumps({k: v for k, v in d.items() if isinstance(v, (int, float))}) + "\n")
        if self.wandb is not None:
            import wandb
            payload = dict(d)
            for k, fig in (images or {}).items():
                payload[k] = wandb.Image(fig)
            wandb.log(payload)
        if images:
            import matplotlib.pyplot as plt
            for k, fig in images.items():
                fig.savefig(os.path.join(self.a.out, "figs", f"{k.replace('/', '_')}_{self.step:07d}.png"), dpi=110)
                plt.close(fig)

    # -- the loop --------------------------------------------------------------
    def gibbs_rate(self):
        """Zero while the heads are still zero (step 0 of a fresh run): phi = 0 makes both
        kernels the identity, exact without it, and an untrained q would only scramble."""
        return self.a.gibbs_rate if self.step > 0 else 0.0

    @torch.no_grad()
    def transport(self, b0, lp0, net):
        """Endpoints of tau-leaped vacuum frames: (bins, dequantized z, log w, jumps, -U_1 + U_0 part)."""
        x1, lr, jumps = br.ctmc_sample(net, b0.long(), self.a.ctmc_steps, path_weight=True,
                                       gibbs_rate=self.gibbs_rate())
        z1 = self.tg.dequantize(x1)
        _, U1 = self.tg.reduced(z1)
        ends = -U1 - lp0
        return x1, z1, ends + lr, jumps, ends

    @torch.no_grad()
    def refresh(self, n):
        t0 = time.time()
        a, D, L = self.a, self.tg.D, self.tg.L
        lws, ends, jumps, lab_s = [], [], [], 0.0
        for i in range(0, n, a.sample_chunk):
            m = min(a.sample_chunk, n - i)
            idx = torch.randint(0, len(self.train_b0), (m,), device=self.device)
            x1, z1, lw, jp, en = self.transport(self.train_b0[idx], self.train_lp0[idx], self.ema)
            tl = time.time()
            rho = self.tg.gibbs_labels(z1).exp()
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            lab_s += time.time() - tl
            self.buf.push(x1, rho, lw)
            lws.append(lw); ends.append(en); jumps.append(jp)
        lw = torch.cat(lws); en = torch.cat(ends); jp = torch.cat(jumps)
        self.energy_calls += n * (1 + D * L)
        lw_buf, self.lw_floor = floor_impossible(self.buf.log_w[:self.buf.n], a.impossible_gap)
        self.nu, self.nu_ess = temper_exponent(lw_buf, a.target_ess)
        st_ = log_ratio_stats(lw)
        fin = torch.isfinite(lw)
        if fin.sum() > 1:
            st_.update({"log_w_std": float(lw[fin].std()), "endpoint_term_mean": float(en[fin].mean()),
                        "path_term_mean": float((lw - en)[fin].mean())})
        return {**{f"refresh/{k}": st_[k] for k in ("log_w_std", "endpoint_term_mean", "path_term_mean") if k in st_},
                "refresh/seconds": time.time() - t0, "refresh/label_s": lab_s,
                "refresh/log_ratio": st_["log_ratio"], "refresh/ess": st_["ess"],
                "refresh/frac_finite_w": float(fin.double().mean()), "refresh/jumps": float(jp.mean()),
                "temper_nu": self.nu, "temper_ess": self.nu_ess}

    def train_step(self):
        a = self.a
        x1, rho, lw = self.buf.sample(a.batch)
        if a.weighting == "tempered":
            lw = torch.where(torch.isfinite(lw), lw, torch.full_like(lw, self.lw_floor))
            y = self.nu * torch.nan_to_num(lw, nan=-1e300)
            w = torch.exp(y - y.max())
            w = (w / w.mean().clamp_min(1e-300)).float()
        else:
            w = torch.ones(a.batch, device=self.device)
        x0 = self.train_b0[torch.randint(0, len(self.train_b0), (a.batch,), device=self.device)].long()
        t = torch.rand(a.batch, device=self.device)
        xt = br.interpolate(x0, x1, t)
        kbar, dk = br.bridge_labels(x0, rho, t)
        loss, lq, lphi = br.losses(self.net, self.ema if a.eta < 1.0 else None, xt, t, kbar, dk, w, a.eta, a.lam_phi)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(self.net.parameters(), a.clip)
        if torch.isfinite(gn):
            self.opt.step()
            with torch.no_grad():
                for pe, p in zip(self.ema.parameters(), self.net.parameters()):
                    pe.lerp_(p, 1.0 - a.ema)
        return {"loss": float(loss.detach()), "loss_q": lq, "loss_phi": lphi, "grad_norm": float(gn),
                "batch_w_max": float(w.max())}

    def _figures(self, phi, psi, E, w, ref, title, tag=""):
        return {f"plots/{tag}ramachandran": metrics.rama_figure(phi, psi, w, ref, title),
                f"plots/{tag}energy_hist": metrics.energy_figure(E, w, ref, title),
                f"plots/{tag}phi_psi_profiles": metrics.phi_profile_figure(phi, psi, w, ref)}

    @torch.no_grad()
    def evaluate(self, n):
        t0 = time.time()
        a = self.a
        n = min(n, len(self.test_b0))
        idx = torch.randperm(len(self.test_b0), device=self.device)[:n]
        zs, lws, ends, jps, Es, moved, ins = [], [], [], [], [], [], []
        for i in range(0, n, a.sample_chunk):
            j = idx[i:i + a.sample_chunk]
            b0 = self.test_b0[j]
            x1, z1, lw, jp, en = self.transport(b0, self.test_lp0[j], self.ema)
            zs.append(z1.float()); lws.append(lw); ends.append(en); jps.append(jp); Es.append(self.tg.energy1(z1))
            moved.append((x1 != b0.long()).float().mean(-1)); ins.append(self.tg.in_endpoint(z1, 1).float())
        z = torch.cat(zs); lw = torch.cat(lws); en = torch.cat(ends); E = torch.cat(Es)
        st_ = log_ratio_stats(lw)
        res = {f"eval/{k}": v for k, v in st_.items()}
        res["eval/log_ratio_err"] = st_["log_ratio"] - self.ref_log_ratio
        res["eval/abs_log_ratio_err"] = abs(res["eval/log_ratio_err"])
        res["eval/endpoint_log_ratio"] = log_ratio_stats(en)["log_ratio"]   # the same endpoints without path terms
        res["eval/jumps"] = float(torch.cat(jps).mean())
        res["eval/frac_coords_moved"] = float(torch.cat(moved).mean())
        res["eval/frac_in_endpoint1"] = float(torch.cat(ins).mean())     # 1 for aldps; inside S_a for aldpt
        w = torch.softmax(torch.nan_to_num(lw, nan=-1e300), 0).cpu().numpy()
        phi, psi = (t.cpu().numpy() for t in self.tg.phi_psi(z))
        En = E.cpu().numpy()
        res["eval/energy_median"] = float(np.median(En))
        res.update({f"eval/{k}": v for k, v in metrics.metrics(self.ref, phi, psi, En, w).items()})
        res["eval/seconds"] = time.time() - t0
        res["energy_calls"] = self.energy_calls
        title = (f"{a.task} step {self.step}, eta={a.eta}: log Z1/Z0 = {st_['log_ratio']:.3f} +- "
                 f"{st_['log_ratio_se']:.3f} (FEAT {self.ref_log_ratio}), ESS={st_['ess']:.3f}")
        return res, self._figures(phi, psi, En, w, self.ref, title)

    @torch.no_grad()
    def _paths(self, j: torch.Tensor) -> dict:
        """Transport the held-out frames j (indices into the test bank): FINAL_FIELDS as CPU numpy."""
        tg = self.tg
        _, z1, lw, jp, en = self.transport(self.test_b0[j], self.test_lp0[j], self.ema)
        phi, psi = tg.phi_psi(z1)
        return {"z": z1.double().cpu().numpy(), "x": zm.to_cartesian(z1).float().cpu().numpy(),
                "energy": tg.energy1(z1).double().cpu().numpy(), "log_w": lw.double().cpu().numpy(),
                "log_w_endpoint": en.double().cpu().numpy(), "rama": torch.stack([phi, psi], -1).float().cpu().numpy(),
                "start_frame": self.test_frame[j].long().cpu().numpy(), "jumps": jp.float().cpu().numpy()}

    @torch.no_grad()
    def final_evaluate(self):
        """One path per endpoint-0 MD frame (or --final_eval_n paths), started from the held-out
        frames in turn, so each of them is transported about 1 / test_frac times with independent
        CTMC noise.  Parts of final_part paths, each seeded on its own and saved as it finishes: a
        restart skips the parts on disk.  The fields go to <out>/final/<field>.npy, meta.json last;
        the metrics compare with every endpoint-1 MD frame."""
        a, tg = self.a, self.tg
        n = a.final_eval_n if a.final_eval_n > 0 else self.n_frames0
        d = os.path.join(a.out, "final")
        pdir = os.path.join(d, "parts")
        os.makedirs(pdir, exist_ok=True)
        n_test = len(self.test_b0)
        n_parts = (n + a.final_part - 1) // a.final_part
        t0 = time.time()
        print(f"final evaluation: {n} paths from {min(n, n_test)} held-out frames, {n_parts} parts of {a.final_part}",
              flush=True)
        for k in range(n_parts):
            path = os.path.join(pdir, f"part_{k:05d}.npz")
            if os.path.exists(path):
                continue
            torch.manual_seed(FINAL_SEED * (a.seed + 1) + k)      # the CTMC noise and dequantization of this part
            j = torch.arange(k * a.final_part, min(n, (k + 1) * a.final_part), device=self.device) % n_test
            chunks = [self._paths(j[i:i + a.final_chunk]) for i in range(0, len(j), a.final_chunk)]
            tmp = os.path.join(pdir, f"tmp_{k:05d}.npz")
            np.savez(tmp, **{f: np.concatenate([c[f] for c in chunks]) for f in FINAL_FIELDS})
            os.replace(tmp, path)
            print(f"  part {k + 1}/{n_parts} ({time.time() - t0:.0f} s)", flush=True)
            if self.stop:
                print(f"SIGTERM during the final evaluation, {k + 1}/{n_parts} parts saved", flush=True)
                sys.exit(143)
        files = [np.load(os.path.join(pdir, f"part_{k:05d}.npz")) for k in range(n_parts)]
        obs = {f: np.concatenate([fl[f] for fl in files]) for f in FINAL_FIELDS}
        assert len(obs["log_w"]) == n, (len(obs["log_w"]), n)
        for f in FINAL_FIELDS:
            np.save(os.path.join(d, f"{f}.npy"), obs[f])
        shutil.copyfile(sy.PDB_MIN, os.path.join(d, "topology.pdb"))

        lw = torch.as_tensor(obs["log_w"])
        st_ = log_ratio_stats(lw)
        res = {f"final/{k}": v for k, v in st_.items() if k != "log_ratio_se"}
        res["final/log_ratio_se"] = grouped_se(obs["log_w"], obs["start_frame"])
        res["final/log_ratio_se_iid"] = st_["log_ratio_se"]
        res["final/log_ratio_err"] = st_["log_ratio"] - self.ref_log_ratio
        res["final/endpoint_log_ratio"] = log_ratio_stats(torch.as_tensor(obs["log_w_endpoint"]))["log_ratio"]
        res["final/jumps"] = float(obs["jumps"].mean())
        res["final/frac_in_endpoint1"] = float(tg.in_endpoint(torch.as_tensor(obs["z"]), 1).double().mean())
        ref = self.full_reference()
        w = torch.softmax(torch.nan_to_num(lw, nan=-1e300), 0).numpy() if torch.isfinite(lw).any() else None
        phi, psi = obs["rama"][:, 0], obs["rama"][:, 1]
        res["final/energy_median"] = float(np.median(obs["energy"]))
        res.update({f"final/{k}": v for k, v in metrics.metrics(ref, phi, psi, obs["energy"], w).items()})
        n_start = int(len(np.unique(obs["start_frame"])))
        res.update({"final/n_paths": n, "final/n_start_frames": n_start, "final/ref_n_frames": len(ref["phi"]),
                    "final/seconds": time.time() - t0})
        title = (f"{a.task} final, n={n}: log Z1/Z0 = {st_['log_ratio']:.3f} +- {res['final/log_ratio_se']:.3f} "
                 f"(FEAT {self.ref_log_ratio}), ESS={st_['ess']:.3f}")
        figs = self._figures(phi, psi, obs["energy"], w, ref, title, tag="final_")
        meta = {"run": os.path.basename(os.path.normpath(a.out)), "task": a.task, "seed": a.seed, "step": self.step,
                "n_paths": n, "n_start_frames": n_start, "n_endpoint0_frames": self.n_frames0,
                "endpoint1_md_frames": len(ref["phi"]), "reference_log_ratio": self.ref_log_ratio,
                **{k[6:]: res[k] for k in ("final/log_ratio", "final/log_ratio_se", "final/log_ratio_se_iid",
                                           "final/ess", "final/endpoint_log_ratio", "final/frac_in_endpoint1")},
                "kT_kJ_per_mol": tg.kT, "temperature_K": sy.TEMPERATURE,
                "seeds": f"part k: torch.manual_seed({FINAL_SEED} * (seed + 1) + k), parts of {a.final_part} paths",
                "fields": FIELD_DOC, "seconds": time.time() - t0}
        with open(os.path.join(d, "meta.json.tmp"), "w") as f:
            json.dump(meta, f, indent=1)
        os.replace(os.path.join(d, "meta.json.tmp"), os.path.join(d, "meta.json"))
        remove_parts(pdir)
        return res, figs

    def run(self):
        a = self.a
        os.makedirs(os.path.join(a.out, "figs"), exist_ok=True)
        with open(os.path.join(a.out, "args.json"), "w") as f:
            json.dump(vars(a), f, indent=1)
        resumed = self.load()
        if self.step >= a.steps and os.path.exists(os.path.join(a.out, "result.json")):
            print(f"{a.out} is finished (result.json); nothing to do", flush=True)
            return
        self.init_wandb()
        signal.signal(signal.SIGTERM, lambda *_: setattr(self, "stop", True))
        if self.step < a.steps:
            if not resumed:
                self.log({"data/frac_inside_box_md0": self.frac_inside[0], "data/frac_inside_box_md1": self.frac_inside[1],
                          "data/n_train": len(self.train_b0), "data/n_test": len(self.test_b0),
                          "data/n_endpoint1": len(self.z1_all)},
                         {"plots/endpoints": metrics.rama_figure(self.start["phi"], self.start["psi"], None, self.ref,
                                                                 f"{a.task}: endpoint 0 (right, given) vs endpoint 1 "
                                                                 "(left, reference only)")})
                if a.eval_at_start:
                    res, figs = self.evaluate(a.eval_n)
                    self.log(res, figs)
                    print(f"step 0 " + json.dumps({k: round(v, 4) for k, v in res.items() if isinstance(v, float)}),
                          flush=True)
            # the buffer is not checkpointed: start (or restart) with a full one
            self.log(self.refresh(a.n_rollout * a.warm_start))
        t_last, s_last = time.time(), self.step
        pending = {}
        while self.step < a.steps and not self.stop:
            stats = {}
            if self.step % a.rollout_every == 0 and self.step > 0:
                pending = self.refresh(a.n_rollout)
            stats.update(self.train_step())
            self.step += 1
            if self.step % a.log_every == 0:
                now = time.time()
                stats["steps_per_s"] = (self.step - s_last) / max(now - t_last, 1e-9)
                t_last, s_last = now, self.step
                stats["energy_calls"] = self.energy_calls
                stats.update(pending)
                pending = {}
                self.log(stats)
                if self.step % (a.log_every * 20) == 0:
                    print(json.dumps({k: round(v, 5) if isinstance(v, float) else v
                                      for k, v in stats.items()}), flush=True)
            if self.step % a.ckpt_every == 0:
                self.save()
            if self.step % a.eval_every == 0 and self.step < a.steps:
                res, figs = self.evaluate(a.eval_n)
                self.log(res, figs)
                print(f"step {self.step} " + json.dumps({k: round(v, 4) for k, v in res.items()
                                                         if isinstance(v, float)}), flush=True)
        self.save()
        if self.stop:
            print(f"SIGTERM: checkpointed at step {self.step}", flush=True)
            sys.exit(143)
        res, figs = self.final_evaluate()
        self.log(res, figs)
        with open(os.path.join(a.out, "result.json"), "w") as f:
            json.dump(res, f, indent=1)
        print("=== DONE", json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in res.items()}),
              flush=True)
        if self.wandb is not None:
            self.wandb.finish()


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    # target and data
    ap.add_argument("--task", default="aldps", choices=["aldps", "aldpt"])
    ap.add_argument("--bins", type=int, default=64)
    ap.add_argument("--n_sigma", type=float, default=10.0)
    ap.add_argument("--branch_half", type=float, default=45.0)
    ap.add_argument("--omega_half", type=float, default=60.0)
    ap.add_argument("--label_dtype", default="float32", choices=["float32", "float64"])
    ap.add_argument("--label_chunk", type=int, default=0, help="0 = from GPU memory")
    ap.add_argument("--data_root", default=None, help="FEAT's ALDP files (default data/feat_aldp)")
    ap.add_argument("--test_frac", type=float, default=0.1, help="held-out endpoint-0 frames for evaluation")
    ap.add_argument("--ref_n", type=int, default=50000, help="endpoint-1 MD frames for the in-run plots and metrics")
    # model
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4, help="layers of each causal stream")
    ap.add_argument("--readout", type=int, default=2, help="hollow readout blocks")
    ap.add_argument("--heads", type=int, default=8)
    # bridge
    ap.add_argument("--ctmc_steps", type=int, default=256)
    ap.add_argument("--gibbs_rate", type=float, default=0.1,
                    help="reversible heat-bath rate under q added to both kernels; > 0 for unbiased weights")
    ap.add_argument("--lam_phi", type=float, default=1.0, help="loss = L_q + lam_phi * L_phi")
    # DIG-IN (defaults of the Ising bridge, DIG-IN/experiments/bridge.py)
    ap.add_argument("--steps", type=int, default=150000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--ema", type=float, default=0.997)
    ap.add_argument("--eta", type=float, default=0.3, help="damping towards the EMA model")
    ap.add_argument("--weighting", default="tempered", choices=["tempered", "none"])
    ap.add_argument("--target_ess", type=float, default=0.5)
    ap.add_argument("--impossible_gap", type=float, default=20.0,
                    help="tempering only: log w of an endpoint outside S_a = lowest finite log w - gap")
    ap.add_argument("--buffer", type=int, default=16384)
    ap.add_argument("--n_rollout", type=int, default=1024)
    ap.add_argument("--rollout_every", type=int, default=100)
    ap.add_argument("--warm_start", type=int, default=4)
    ap.add_argument("--sample_chunk", type=int, default=1024)
    # evaluation / logging
    ap.add_argument("--eval_every", type=int, default=5000)
    ap.add_argument("--eval_n", type=int, default=8192)
    ap.add_argument("--final_eval_n", type=int, default=0, help="0 = one path per endpoint-0 MD frame")
    ap.add_argument("--final_part", type=int, default=16384, help="paths per saved part of the final evaluation")
    ap.add_argument("--final_chunk", type=int, default=4096, help="paths per transport call in the final evaluation")
    ap.add_argument("--eval_at_start", type=int, default=1)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--ckpt_every", type=int, default=1000)
    ap.add_argument("--no_wandb", action="store_true")
    ap.add_argument("--wandb_project", default="digin-bridge")
    ap.add_argument("--wandb_entity", default=None)
    ap.add_argument("--wandb_group", default=None)
    ap.add_argument("--wandb_name", default=None)
    return ap.parse_args(argv)


def main(argv=None):
    a = parse(argv)
    os.makedirs(a.out, exist_ok=True)
    Trainer(a).run()


if __name__ == "__main__":
    main()
