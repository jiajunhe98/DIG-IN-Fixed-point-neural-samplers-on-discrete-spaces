"""Tables and figure data of the paper's free-energy experiments, from the final evaluations of bridge-v1.

    python -m digin_bridge.paper --runs_root runs --out paper_data/freeenergy [--wandb_group bridge-v1]

Reads <runs_root>/*/<task>_s<seed>/final/{log_w,rama,energy,start_frame}.npy with meta.json (written by
digin_bridge.train's final evaluation) and FEAT's ALDP MD.  Writes to --out (the paper's data/freeenergy/):

  numbers.tex                       LaTeX macros (\FESEst, \FETEst, ...): estimates, standard errors, ESS,
                                    endpoint metrics, references
  seeds.tex                         rows of the per-seed table
  conv_<task>.dat                   running estimate log mean_{i <= M} w_i against M, per seed (paths in a
                                    random order), and for aldps the same for Zwanzig's estimator on the
                                    endpoint-0 MD frames (no transport)
  train_<task>.dat                  in-run estimate and its standard error against the training step (W&B)
  rama_<task>_<md0|md1|raw|rw>.png  ln p(phi, psi) - max: endpoint-0 MD, endpoint-1 MD, transported
                                    endpoints as drawn and reweighted (three seeds pooled)
  energy_<task>.dat                 endpoint-1 energy histograms: MD, transported, reweighted
  summary.json                      every number, unrounded
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os

import numpy as np
import torch

from digin_bridge import data, metrics
from digin_bridge.target import BridgeTarget
from digin_bridge.train import grouped_se

TASKS = {"aldps": "S", "aldpt": "T"}
SEEDS = (0, 1, 2)
# external references, log Z_1 / Z_0: FEAT's MBAR values (He et al. 2025, Table 1 and App. F.1.4) and
# our umbrella-sampling check of ALDP-T (40 windows along phi, 2 ns each, MBAR; README "The ALDP-T reference")
REF_MBAR = {"aldps": (29.43, 0.01), "aldpt": (-4.25, 0.05)}
REF_UMBRELLA = {"aldpt": (-4.44, 0.02)}
FE_BINS, FE_CAP = 60, 8.0


def find_final(root, name):
    cands = [os.path.dirname(p) for p in glob.glob(os.path.join(root, "*", name, "final", "meta.json"))
             + glob.glob(os.path.join(root, name, "final", "meta.json"))]
    if not cands:
        raise SystemExit(f"no final evaluation of {name} under {root}")
    return max(cands, key=lambda d: os.path.getmtime(os.path.join(d, "meta.json")))


def log_mean_exp(lw):
    lw = np.asarray(lw, np.float64)
    m = np.max(lw)
    return float(m + math.log(np.exp(lw - m).sum() / len(lw))) if np.isfinite(m) else -math.inf


def ess_frac(lw):
    lw = np.asarray(lw, np.float64)
    w = np.exp(lw - np.max(lw))
    return float(w.sum() ** 2 / (len(w) * (w ** 2).sum()))


def running(lw, grid, rng):
    """log mean exp of the first M paths of a random permutation, for every M in grid."""
    lw = np.asarray(lw, np.float64)[rng.permutation(len(lw))]
    m = np.max(lw[np.isfinite(lw)])
    c = np.cumsum(np.exp(lw - m))
    return np.array([m + math.log(c[M - 1] / M) if c[M - 1] > 0 else np.nan for M in grid])


def log_density(phi, psi, w=None):
    h, _, _ = np.histogram2d(phi, psi, bins=FE_BINS, range=[[-math.pi, math.pi]] * 2, weights=w)
    with np.errstate(divide="ignore"):
        f = np.log(h / h.sum())
    f -= f[np.isfinite(f)].max()
    return np.clip(np.where(np.isfinite(f), f, -FE_CAP), -FE_CAP, 0.0)


def save_surface(path, f):
    """f[i, j] over (phi_i, psi_j) as an image with phi along x and psi along y, viridis, no axes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(2, 2), dpi=180)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(f.T, origin="lower", extent=(-180, 180, -180, 180), cmap="viridis", vmin=-FE_CAP, vmax=0.0,
              interpolation="bilinear", aspect="auto")
    ax.set_axis_off()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def fmt(v, digits):
    s = f"{v:.{digits}f}"
    return s[1:] if s.startswith("0.") else ("-" + s[2:] if s.startswith("-0.") else s)


def cell(vals, digits, scale=1.0):
    v = np.asarray(vals, np.float64) * scale
    return f"${fmt(v.mean(), digits)}_{{\\pm{fmt(v.std(ddof=1) if len(v) > 1 else 0.0, digits)}}}$"


def pm(m, s, digits):
    return f"${m:.{digits}f}\\pm{s:.{digits}f}$"


@torch.no_grad()
def batched(fn, z, device, chunk=16384):
    return torch.cat([fn(torch.as_tensor(z[i:i + chunk]).to(device)).double().cpu() for i in range(0, len(z), chunk)]).numpy()


def wandb_history(group, entity=None, project="digin-bridge"):
    try:
        import wandb
        api = wandb.Api(timeout=120)
        out = {}
        for r in api.runs(f"{entity or api.default_entity}/{project}", filters={"group": group}):
            rows = r.history(keys=["step", "eval/log_ratio", "eval/log_ratio_se"], samples=100000, pandas=False)
            out[r.name] = {"rows": rows, "hours": r.summary.get("_runtime", 0) / 3600}
        return out
    except Exception as e:  # noqa: BLE001  (offline, or no key: the curves are optional)
        print(f"no W&B history: {e}", flush=True)
        return {}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", required=True,
                    help="directory holding <task>_s<seed>/final/ (or */<task>_s<seed>/final/, amlt's layout)")
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--wandb_group", default=None, help="also write the in-run curves of this W&B group")
    ap.add_argument("--wandb_entity", default=None)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = data.ensure(a.data_root)
    hist = wandb_history(a.wandb_group, a.wandb_entity) if a.wandb_group else {}
    macros, seed_rows, summary = [], [], {}

    def mac(name, value):
        macros.append(f"\\newcommand{{\\{name}}}{{{value}}}")

    for task, tt in TASKS.items():
        tg = BridgeTarget(task=task, device=device)
        z0, z1, frac, _ = data.endpoint_frames(tg, root)
        phi0, psi0 = (v.cpu().numpy() for v in tg.phi_psi(torch.as_tensor(z0)))
        phi1, psi1 = (v.cpu().numpy() for v in tg.phi_psi(torch.as_tensor(z1)))
        E1 = batched(tg.energy1, z1, device)
        ref = {"phi": phi1, "psi": psi1, "energy": E1}
        save_surface(os.path.join(a.out, f"rama_{task}_md0.png"), log_density(phi0, psi0))
        save_surface(os.path.join(a.out, f"rama_{task}_md1.png"), log_density(phi1, psi1))
        per, pool = [], {"phi": [], "psi": [], "E": [], "w": [], "wn": []}
        conv_rows, grid = {}, None
        for seed in SEEDS:
            name = f"{task}_s{seed}"
            d = find_final(a.runs_root, name)
            meta = json.load(open(os.path.join(d, "meta.json")))
            lw = np.load(os.path.join(d, "log_w.npy"))
            rama = np.load(os.path.join(d, "rama.npy"))
            E = np.load(os.path.join(d, "energy.npy"))
            start = np.load(os.path.join(d, "start_frame.npy"))
            n = len(lw)
            w = np.exp(lw - np.max(lw))
            m = {"n": n, "n_start": int(len(np.unique(start))), "log_ratio": log_mean_exp(lw),
                 "se": grouped_se(lw, start), "ess": ess_frac(lw), "frac_in1": meta["frac_in_endpoint1"],
                 "step": meta["step"], "seconds": meta["seconds"]}
            m.update(metrics.metrics(ref, rama[:, 0], rama[:, 1], E, w / w.sum()))
            per.append(m)
            summary[name] = m
            pool["phi"].append(rama[:, 0]); pool["psi"].append(rama[:, 1]); pool["E"].append(E)
            pool["w"].append(w / w.sum() / len(SEEDS)); pool["wn"].append(np.full(n, 1.0 / n / len(SEEDS)))
            if grid is None:
                grid = np.unique(np.geomspace(100, n, 60).astype(int))
            conv_rows[f"s{seed}"] = running(lw, grid, np.random.default_rng(seed))
            seed_rows.append(" & ".join([{"aldps": "ALDP-S", "aldpt": "ALDP-T"}[task] if seed == 0 else "", str(seed),
                                         pm(m["log_ratio"], m["se"], 3), f"${m['ess']:.3f}$",
                                         f"${1e3 * m['rama_js']:.2f}$", f"${1e3 * m['rama_js_rw']:.2f}$",
                                         f"${m['energy_w1']:.2f}$", f"${m['energy_w1_rw']:.2f}$",
                                         f"${m['frac_in1']:.4f}$"]) + " \\\\")
        # no transport: Zwanzig's estimator on the endpoint-0 frames (aldps only: the ALDP-T states are disjoint)
        if task == "aldps":
            polar = lambda z: tg.parts(z)[1]                              # noqa: E731  (U_1 - U_0 in kJ/mol)
            du0, du1 = batched(polar, z0, device) / tg.kT, batched(polar, z1, device) / tg.kT
            fep = log_mean_exp(-du0)
            conv_rows["fep"] = running(-du0, grid[grid <= len(du0)], np.random.default_rng(10))
            summary["aldps_fep"] = fep
            summary["aldps_bar"] = data.bar(du0, -du1)
            mac("FESFEP", f"${fep:.2f}$")
            mac("FESBAR", f"${summary['aldps_bar']:.3f}$")
            mac("FESFEPESS", f"${ess_frac(-du0):.4f}$")
        with open(os.path.join(a.out, f"conv_{task}.dat"), "w") as f:
            cols = list(conv_rows)
            f.write("M " + " ".join(cols) + "\n")
            for i, M in enumerate(grid):
                f.write(f"{M} " + " ".join(f"{conv_rows[c][i]:.6g}" if i < len(conv_rows[c]) and np.isfinite(conv_rows[c][i])
                                           else "nan" for c in cols) + "\n")
        phi, psi = np.concatenate(pool["phi"]), np.concatenate(pool["psi"])
        w, wn = np.concatenate(pool["w"]), np.concatenate(pool["wn"])
        save_surface(os.path.join(a.out, f"rama_{task}_raw.png"), log_density(phi, psi, wn))
        save_surface(os.path.join(a.out, f"rama_{task}_rw.png"), log_density(phi, psi, w))
        E = np.concatenate(pool["E"])
        lo, hi = np.quantile(E1, [5e-4, 0.9995])
        lo, hi = lo - 0.15 * (hi - lo), hi + 0.35 * (hi - lo)
        edges = np.linspace(lo, hi, 101)
        dens = []
        for e, ww in ((E1, None), (E, wn), (E, w)):
            h, _ = np.histogram(e, bins=edges, weights=ww)
            dens.append(h / (len(e) if ww is None else ww.sum()) / np.diff(edges))
        with open(os.path.join(a.out, f"energy_{task}.dat"), "w") as f:
            f.write("E md raw rw\n")
            for row in zip(0.5 * (edges[1:] + edges[:-1]), *dens):
                f.write(" ".join(f"{v:.6g}" for v in row) + "\n")
        if hist:
            with open(os.path.join(a.out, f"train_{task}.dat"), "w") as f:
                f.write("step " + " ".join(f"s{s} se{s}" for s in SEEDS) + "\n")
                series = {s: {r["step"]: (r.get("eval/log_ratio"), r.get("eval/log_ratio_se"))
                              for r in hist.get(f"{task}_s{s}", {"rows": []})["rows"] if r.get("eval/log_ratio") is not None}
                          for s in SEEDS}
                for step in sorted(set().union(*[set(v) for v in series.values()])):
                    vals = []
                    for s in SEEDS:
                        v, se = series[s].get(step, (None, None))
                        ok = v is not None and np.isfinite(v)
                        vals += [f"{v:.6g}" if ok else "nan", f"{se:.6g}" if ok and se is not None and np.isfinite(se) else "nan"]
                    f.write(f"{step} " + " ".join(vals) + "\n")
            hrs = [hist[f"{task}_s{s}"]["hours"] for s in SEEDS if f"{task}_s{s}" in hist]
            if hrs:
                mac(f"FE{tt}Hours", f"{min(hrs):.0f}--{max(hrs):.0f}")
        P = f"FE{tt}"
        mac(f"{P}Est", cell([m["log_ratio"] for m in per], 3))
        mac(f"{P}EstPlain", f"{np.mean([m['log_ratio'] for m in per]):.3f}")
        mac(f"{P}SE", f"{np.mean([m['se'] for m in per]):.3f}")
        mac(f"{P}ESS", cell([m["ess"] for m in per], 3))
        mac(f"{P}JS", cell([m["rama_js_rw"] for m in per], 2, 1e3))
        mac(f"{P}JSRaw", cell([m["rama_js"] for m in per], 2, 1e3))
        mac(f"{P}WOne", cell([m["energy_w1_rw"] for m in per], 2))
        mac(f"{P}WOneRaw", cell([m["energy_w1"] for m in per], 2))
        mac(f"{P}InState", cell([m["frac_in1"] for m in per], 4))
        mac(f"{P}NPaths", f"{per[0]['n']:,}".replace(",", "{,}"))
        mac(f"{P}NStart", f"{per[0]['n_start']:,}".replace(",", "{,}"))
        mac(f"{P}NRef", f"{len(z1):,}".replace(",", "{,}"))
        mac(f"{P}EvalMin", f"{min(m['seconds'] for m in per) / 60:.0f}--{max(m['seconds'] for m in per) / 60:.0f}")
        mac(f"{P}RefMBAR", pm(*REF_MBAR[task], 2))
        mac(f"{P}RefMBARPlain", f"{REF_MBAR[task][0]:.2f}")
        mac(f"{P}RefMBARErr", f"{REF_MBAR[task][1]:.2f}")
        if task in REF_UMBRELLA:
            mac(f"{P}RefUmb", pm(*REF_UMBRELLA[task], 2))
            mac(f"{P}RefUmbPlain", f"{REF_UMBRELLA[task][0]:.2f}")
            mac(f"{P}RefUmbErr", f"{REF_UMBRELLA[task][1]:.2f}")
        summary[f"{task}_frames"] = {"endpoint0": len(z0), "endpoint1": len(z1), "inside_box": frac}
    mac("FENSeeds", str(len(SEEDS)))
    mac("FEFECap", f"{FE_CAP:.0f}")
    with open(os.path.join(a.out, "numbers.tex"), "w") as f:
        f.write("% generated by digin_bridge/paper.py (digin-bridge) -- do not edit\n" + "\n".join(macros) + "\n")
    with open(os.path.join(a.out, "seeds.tex"), "w") as f:
        f.write("\n".join(seed_rows) + "\n")
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=float)
    print("\n".join(macros), flush=True)


if __name__ == "__main__":
    main()
