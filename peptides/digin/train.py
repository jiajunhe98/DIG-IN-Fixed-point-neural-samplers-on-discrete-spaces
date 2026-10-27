"""DIG-IN on a capped alanine peptide, from the energy alone.

Every `rollout_every` steps the EMA model draws `n_rollout` configurations, one
coordinate per network call in a uniformly random order.  In Cartesian
coordinates each draw then gets `pivot_mh` Metropolis pivot moves (exact for the
target, so it keeps the importance weight of the draw it came from).  The draws,
their Gibbs labels and log-weights go into a first-in-first-out buffer.  Each
gradient step then

  1. masks a fresh random subset of each buffered configuration,
  2. damps the label towards the EMA model: eta * label + (1 - eta) * q_ema(. | masked input),
  3. takes the soft cross-entropy at the masked positions, weighted by w^nu / sum w^nu
     with the largest nu in [0, 1] that keeps the batch ESS >= target_ess.

No target samples enter training; the MD reference is used only for evaluation.
The final evaluation draws as many samples as the reference has frames
(`--final_eval_n 0`), so histogram metrics such as the Ramachandran KL compare
equal sample sizes.  It runs in parts that are saved as they finish, so a
preempted job resumes it.

    python -m digin.train --system al4 --coords cartesian --seed 0 --out runs/al4_cart_s0 --ref ref/al4_ref.npz
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import signal
import sys
import time
import uuid

import numpy as np
import torch

from digin import metrics
from digin.model import AnyOrderTransformer, any_order_sample
from digin.targets import make_target

DEFAULTS = {"internal": {"bins": 64, "pivot_mh": 0},
            "cartesian": {"bins": 512, "pivot_mh": 1000}}     # 1000 phi/psi pivots: README, "The setting"


def tempered_weights(log_w: torch.Tensor, target_ess: float, n_bisect: int = 40):
    """DIG-IN's ESS rule: w^nu / sum w^nu with the largest nu in [0, 1] whose ESS >= target_ess * B."""
    lw = torch.nan_to_num(log_w.double(), nan=-1e300, posinf=-1e300, neginf=-1e300)
    n = lw.shape[0]

    def frac(nu):
        x = nu * lw
        return float((2 * torch.logsumexp(x, 0) - torch.logsumexp(2 * x, 0)).exp() / n)

    if frac(1.0) >= target_ess:
        nu = 1.0
    else:
        lo, hi = 0.0, 1.0
        for _ in range(n_bisect):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if frac(mid) >= target_ess else (lo, mid)
        nu = lo
    return torch.softmax(nu * lw, 0).float(), nu, frac(nu)


def ess_stats(log_w: torch.Tensor) -> dict:
    lw = torch.nan_to_num(log_w.double(), nan=-math.inf)
    n = lw.shape[0]
    lse = torch.logsumexp(lw, 0)
    return {"ess": float((2 * lse - torch.logsumexp(2 * lw, 0)).exp() / n),
            "log_Z": float(lse - math.log(n)), "elbo": float(log_w.double().mean())}


class Buffer:
    def __init__(self, cap, D, L, device):
        self.cap, self.n, self.ptr = cap, 0, 0
        self.x = torch.zeros(cap, D, dtype=torch.long, device=device)
        self.lab = torch.zeros(cap, D, L, dtype=torch.float32, device=device)
        self.log_p = torch.zeros(cap, dtype=torch.float64, device=device)
        self.log_q = torch.zeros(cap, dtype=torch.float64, device=device)

    def push(self, x, lab, log_p, log_q):
        k = x.shape[0]
        idx = (self.ptr + torch.arange(k, device=x.device)) % self.cap
        self.x[idx], self.lab[idx], self.log_p[idx], self.log_q[idx] = x, lab, log_p, log_q
        self.ptr = (self.ptr + k) % self.cap
        self.n = min(self.n + k, self.cap)

    def sample(self, b):
        i = torch.randint(0, self.n, (b,), device=self.x.device)
        return self.x[i], self.lab[i], self.log_p[i], self.log_q[i]


class Trainer:
    def __init__(self, a):
        self.a = a
        self.device = torch.device(a.device if a.device != "auto" else
                                   ("cuda" if torch.cuda.is_available() else "cpu"))
        torch.manual_seed(a.seed); np.random.seed(a.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        gb = (torch.cuda.get_device_properties(self.device).total_memory / 2 ** 30
              if self.device.type == "cuda" else 8.0)
        budget = a.label_budget if a.label_budget > 0 else 2 ** 28 * max(gb / 16.0, 0.5)
        self.tg = make_target(a, self.device, budget)
        D, L = self.tg.D, self.tg.L
        self.net = AnyOrderTransformer(D, L, a.d_model, a.layers, a.heads, kinds=self.tg.kind).to(self.device)
        self.ema = copy.deepcopy(self.net).requires_grad_(False)
        self.opt = torch.optim.AdamW(self.net.parameters(), lr=a.lr, weight_decay=a.wd)
        self.buf = Buffer(a.buffer, D, L, self.device)
        self.step = 0
        self.energy_calls = 0
        self.ref = None
        self.stop = False
        self.wandb = None
        print(f"{a.system} [{a.coords}]: n_atoms {self.tg.mol.n}, D {D}, L {L}, label chunk {self.tg.label_chunk}, "
              f"pivot moves {a.pivot_mh} ({a.pivot_axes}), params {sum(p.numel() for p in self.net.parameters()) / 1e6:.1f}M",
              flush=True)

    # -- bookkeeping -------------------------------------------------------------------
    def load_ref(self):
        if self.ref is not None or not self.a.ref or not os.path.exists(self.a.ref):
            return
        r = np.load(self.a.ref)
        self.ref = {"energy": r["energy"], "rama": r["rama"]}
        d = {"ref/n_frames": len(self.ref["energy"])}
        if "z" in r.files and len(r["z"]):
            _, inside = self.tg.to_bins(self.tg.from_internal(torch.as_tensor(r["z"])))
            d["ref/frac_inside_box"] = float(inside.float().mean())
        print(f"loaded reference {self.a.ref}: {d}", flush=True)
        self.log(d)

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
                except Exception as e:  # noqa: BLE001
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
                                tags=[a.system, a.coords, f"eta={a.eta}", f"L={a.bins}", f"seed={a.seed}"]
                                + ([f"pivot={a.pivot_mh}", f"pivot_axes={a.pivot_axes}"] if a.pivot_mh else []))
        wandb.define_metric("step")
        wandb.define_metric("*", step_metric="step")

    def log(self, d: dict, images: dict | None = None):
        d = {"step": self.step, **d}
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
                fig.savefig(os.path.join(self.a.out, "figs", f"{k.replace('/', '_')}_{self.step:07d}.png"), dpi=100)
                plt.close(fig)

    # -- the loop ----------------------------------------------------------------------
    def _sample(self, n, generator=None):
        D, L = self.tg.D, self.tg.L
        xs, lqs = [], []
        for i in range(0, n, self.a.sample_chunk):
            x, lq = any_order_sample(self.ema, min(self.a.sample_chunk, n - i), D, L, self.device, generator)
            xs.append(x); lqs.append(lq)
        return torch.cat(xs), torch.cat(lqs)

    @torch.no_grad()
    def refresh(self, n):
        t0 = time.time()
        D, L = self.tg.D, self.tg.L
        x, log_q = self._sample(n)
        t1 = time.time()
        z = self.tg.dequantize(x)
        log_p = self.tg.log_p(z)
        extra = {}
        if self.a.pivot_mh > 0:     # moved samples keep the weights of their draws
            z, st_ = self.tg.pivot_mh(z, self.a.pivot_mh)
            self.energy_calls += n * self.a.pivot_mh
            x = self.tg.to_bins(z)[0]
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            extra = {"refresh/pivot_accept": st_["pivot_accept"], "refresh/move_s": time.time() - t1}
        t2 = time.time()
        lab = self.tg.gibbs_labels(z)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t3 = time.time()
        self.energy_calls += n * (1 + D * L)
        self.buf.push(x, lab, log_p, log_q)
        lw = log_p - log_q
        return {"refresh/sample_s": t1 - t0, "refresh/label_s": t3 - t2,
                "refresh/label_evals_per_s": n * D * L / max(t3 - t2, 1e-9),
                "refresh/log_w_mean": float(lw.mean()), "refresh/log_w_std": float(lw.std()),
                "refresh/ess": ess_stats(lw)["ess"],
                "refresh/label_entropy": float(-(lab.exp() * lab).sum(-1).mean()), **extra}

    def train_step(self):
        a, D, L = self.a, self.tg.D, self.tg.L
        x1, lab, lp, lq = self.buf.sample(a.batch)
        B = x1.shape[0]
        w, nu, essf = tempered_weights(lp - lq, a.target_ess)
        alpha = torch.rand(B, 1, device=self.device).clamp(max=1.0 - 1.0 / D)
        u = torch.rand(B, D, device=self.device)
        mask = u >= alpha
        none = ~mask.any(1)
        if none.any():           # mask at least one coordinate per configuration
            mask[none, u[none].argmax(1)] = True
        xt = torch.where(mask, torch.full_like(x1, L), x1)
        with torch.no_grad():
            target = lab.softmax(-1)
            if a.eta < 1.0:
                target = a.eta * target + (1.0 - a.eta) * self.ema(xt).float().softmax(-1)
        logits = self.net(xt).float()
        ce = -(target * logits.log_softmax(-1)).sum(-1)
        m = mask.float()
        per = (ce * m).sum(1) / m.sum(1)
        loss = (w * per).sum()
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(self.net.parameters(), a.clip)
        if torch.isfinite(gn):
            self.opt.step()
            with torch.no_grad():
                for pe, p in zip(self.ema.parameters(), self.net.parameters()):
                    pe.lerp_(p, 1.0 - a.ema)
        return {"loss": float(loss.detach()), "grad_norm": float(gn), "temper_nu": nu, "temper_ess": essf,
                "masked_per_sample": float(m.sum(1).mean())}

    @torch.no_grad()
    def _draw(self, n, generator=None):
        """n samples of the EMA model with their observables and log-weights (CPU numpy)."""
        x, lq = any_order_sample(self.ema, n, self.tg.D, self.tg.L, self.device, generator)
        z = self.tg.dequantize(x, generator)
        o = self.tg.observables(z)
        return {"rama": o["rama"].float().cpu().numpy(), "energy": o["energy"].double().cpu().numpy(),
                "log_w": (self.tg.log_p(z) - lq).double().cpu().numpy()}

    def _summarize(self, obs, prefix):
        lw = torch.as_tensor(obs["log_w"])
        st_ = ess_stats(lw)
        w = torch.softmax(torch.nan_to_num(lw, nan=-1e300), 0).numpy()
        res = {f"{prefix}/{k}": v for k, v in st_.items()}
        res[f"{prefix}/energy_median"] = float(np.median(obs["energy"]))
        res[f"{prefix}/n_samples"] = len(lw)
        res["energy_calls"] = self.energy_calls
        names = self.tg.mol.rama_res
        if self.ref is not None:
            res.update({f"{prefix}/{k}": v for k, v in metrics.peptide_metrics(self.ref, obs, w, names).items()})
        title = (f"{self.a.system} [{self.a.coords}] step {self.step}, eta={self.a.eta}, n={len(lw)}, "
                 f"ESS={st_['ess']:.4f}")
        tag = "" if prefix == "eval" else f"{prefix}_"
        figs = {f"plots/{tag}ramachandran": metrics.rama_grid_figure(obs["rama"], w, names, self.ref, title),
                f"plots/{tag}energy_hist": metrics.hist_figure(obs["energy"], w, None if self.ref is None
                                                               else self.ref["energy"], "potential energy (kJ/mol)",
                                                               title)}
        return res, figs

    @torch.no_grad()
    def evaluate(self, n):
        t0 = time.time()
        self.load_ref()
        parts = [self._draw(min(self.a.sample_chunk, n - i)) for i in range(0, n, self.a.sample_chunk)]
        obs = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        res, figs = self._summarize(obs, "eval")
        res["eval/seconds"] = time.time() - t0
        return res, figs

    @torch.no_grad()
    def final_evaluate(self):
        """As many samples as the reference has frames (or --final_eval_n), in parts of final_part
        samples, each from its own seeded generator and saved as it finishes; a restart skips the
        parts on disk, so the result does not depend on preemptions."""
        a = self.a
        self.load_ref()
        if self.ref is None and a.final_eval_n <= 0:
            raise SystemExit(f"final evaluation needs the reference ({a.ref}) or --final_eval_n")
        n = a.final_eval_n if a.final_eval_n > 0 else len(self.ref["energy"])
        d = os.path.join(a.out, "final_eval")
        os.makedirs(d, exist_ok=True)
        n_parts = (n + a.final_part - 1) // a.final_part
        t0 = time.time()
        print(f"final evaluation: {n} samples in {n_parts} parts of {a.final_part}", flush=True)
        for k in range(n_parts):
            path = os.path.join(d, f"part_{k:05d}.npz")
            if os.path.exists(path):
                continue
            m = min(a.final_part, n - k * a.final_part)
            g = torch.Generator(device=self.device).manual_seed(1_000_003 * (a.seed + 1) + k)
            parts = [self._draw(min(a.final_chunk, m - i), g) for i in range(0, m, a.final_chunk)]
            tmp = os.path.join(d, f"tmp_{k:05d}.npz")
            np.savez(tmp, **{key: np.concatenate([p[key] for p in parts]) for key in parts[0]})
            os.replace(tmp, path)
            print(f"  part {k + 1}/{n_parts} ({time.time() - t0:.0f} s)", flush=True)
            if self.stop:
                print(f"SIGTERM during the final evaluation, {k + 1}/{n_parts} parts saved", flush=True)
                sys.exit(143)
        files = [np.load(os.path.join(d, f"part_{k:05d}.npz")) for k in range(n_parts)]
        obs = {key: np.concatenate([f[key] for f in files]) for key in ("rama", "energy", "log_w")}
        assert len(obs["energy"]) == n, (len(obs["energy"]), n)
        res, figs = self._summarize(obs, "final")
        res["final/ref_n_frames"] = 0 if self.ref is None else len(self.ref["energy"])
        res["final/seconds"] = time.time() - t0
        np.savez_compressed(os.path.join(a.out, "samples_final.npz"), **obs)
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
        self.load_ref()
        if self.step < a.steps:
            if not resumed and a.eval_at_start:
                self.log(*self.evaluate(a.eval_n))
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
                stats.update(pending); pending = {}
                self.log(stats)
                if self.step % (a.log_every * 20) == 0:
                    print(json.dumps({k: round(v, 5) if isinstance(v, float) else v for k, v in stats.items()}),
                          flush=True)
            if self.step % a.ckpt_every == 0:
                self.save()
            if self.step % a.eval_every == 0 and self.step < a.steps:
                res, figs = self.evaluate(a.eval_n)
                self.log(res, figs)
                print(f"step {self.step} " + json.dumps({k: round(v, 4) for k, v in res.items()
                                                         if isinstance(v, float) and "/" in k
                                                         and k.count("/") == 1}), flush=True)
        if self.stop:
            self.save()
            print(f"SIGTERM: checkpointed at step {self.step}", flush=True)
            sys.exit(143)
        self.save()
        res, figs = self.final_evaluate()
        self.log(res, figs)
        with open(os.path.join(a.out, "result.json"), "w") as f:
            json.dump(res, f, indent=1)
        print("=== DONE", json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in res.items()
                                     if k.count("/") == 1}), flush=True)
        if self.wandb is not None:
            self.wandb.finish()


def parse(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, choices=["aldp", "al4"])
    ap.add_argument("--coords", required=True, choices=["internal", "cartesian"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--bins", type=int, default=None, help="bins per coordinate (internal 64, cartesian 512)")
    ap.add_argument("--pivot_mh", type=int, default=None,
                    help="Metropolis pivot moves per fresh sample (internal 0, cartesian 1000)")
    ap.add_argument("--pivot_axes", default="backbone", choices=["backbone", "all"],
                    help="pivot about phi/psi only, or also the methyl rotors (cartesian)")
    ap.add_argument("--cart_margin", type=float, default=0.12, help="nm added to the torsion-sweep box (cartesian)")
    ap.add_argument("--n_sigma", type=float, default=10.0, help="bond/angle box half-width in sigma (internal)")
    ap.add_argument("--branch_half", type=float, default=45.0, help="degrees (internal)")
    ap.add_argument("--omega_half", type=float, default=60.0, help="degrees (internal)")
    ap.add_argument("--label_dtype", default="float32", choices=["float32", "float64"])
    ap.add_argument("--label_budget", type=float, default=0,
                    help="floats of temporaries per energy call; 0 = from GPU memory")
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--steps", type=int, default=150000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--ema", type=float, default=0.995)
    ap.add_argument("--eta", type=float, default=0.5)
    ap.add_argument("--target_ess", type=float, default=0.8)
    ap.add_argument("--buffer", type=int, default=4096)
    ap.add_argument("--n_rollout", type=int, default=512)
    ap.add_argument("--rollout_every", type=int, default=100)
    ap.add_argument("--warm_start", type=int, default=4)
    ap.add_argument("--sample_chunk", type=int, default=1024)
    ap.add_argument("--eval_every", type=int, default=2500)
    ap.add_argument("--eval_n", type=int, default=8192)
    ap.add_argument("--eval_at_start", type=int, default=1)
    ap.add_argument("--final_eval_n", type=int, default=0, help="0 = as many as the reference has frames")
    ap.add_argument("--final_part", type=int, default=65536, help="samples per saved part of the final evaluation")
    ap.add_argument("--final_chunk", type=int, default=4096, help="samples per network batch in the final evaluation")
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--ckpt_every", type=int, default=1000)
    ap.add_argument("--ref", default=None, help="reference MD (.npz with energy, rama, z) for evaluation only")
    ap.add_argument("--no_wandb", action="store_true")
    ap.add_argument("--wandb_project", default="digin-peptides")
    ap.add_argument("--wandb_entity", default=None)
    ap.add_argument("--wandb_group", default=None)
    ap.add_argument("--wandb_name", default=None)
    a = ap.parse_args(argv)
    for k, v in DEFAULTS[a.coords].items():
        if getattr(a, k) is None:
            setattr(a, k, v)
    return a


def main(argv=None):
    a = parse(argv)
    os.makedirs(a.out, exist_ok=True)
    Trainer(a).run()


if __name__ == "__main__":
    main()
