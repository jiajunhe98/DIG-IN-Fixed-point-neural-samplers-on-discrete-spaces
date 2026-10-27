"""Tables and figure data of the paper's peptide experiments, from the 2M-sample sets of digin.sample.

    python -m digin.paper --samples samples --ref_dir ref --out paper_data/peptides [--wandb_group digin-v1]

Reads <samples>/<system>_<coords>_s<seed>/{energy,log_w,rama}.npy with meta.json, and the MD references
<ref_dir>/<system>_ref.npz.  Writes to --out (the paper's data/peptides/):

  numbers.tex                          LaTeX macros, mean +- sd over seeds (\PepDiIntESS, ...)
  seeds.tex                            rows of the per-seed table
  basins.tex                           rows of the AL4 basin-fraction table
  rama_<system>_md_r<k>.png            MD free-energy surface of residue k
  rama_<system>_<coords>_<raw|rw>_r<k>.png   the same for the samples, as drawn and reweighted
  energy_<system>_<coords>.dat         energy histograms: MD, samples, reweighted samples
  summary.json                         every number, unrounded

Metrics (as in the final evaluation of digin.train, on these samples): ESS = (sum w)^2 / (n sum w^2);
log Z = log mean w; energy W1 against all MD frames, draws capped 200 kJ/mol above the highest MD energy;
Ramachandran JS divergence on 36 x 36 cells, mean over residues; basin fractions.  "rw" = with the
self-normalized importance weights.  The images pool the three seeds, each seed's weights normalized
to 1/3, and show ln p(phi, psi) - max ln p on 60 x 60 cells, clipped at -FE_CAP.
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np

from digin import metrics

SYSTEMS = {"aldp": "Di", "al4": "Tetra"}        # macro names cannot hold digits
COORDS = {"internal": "Int", "cartesian": "Cart"}
SEEDS = (0, 1, 2)
METHYLS = {"aldp": 3, "al4": 5}                 # Cartesian log Z = internal + methyls * log 2
FE_BINS = 60
FE_CAP = 8.0


def load_run(root, system, coords, seed):
    d = os.path.join(root, f"{system}_{coords}_s{seed}")
    meta = json.load(open(os.path.join(d, "meta.json")))
    out = {f: np.load(os.path.join(d, f"{f}.npy"), mmap_mode="r") for f in ("energy", "log_w", "rama")}
    return {"energy": np.asarray(out["energy"], np.float64), "log_w": np.asarray(out["log_w"], np.float64),
            "rama": np.asarray(out["rama"], np.float32), "meta": meta}


def weights(log_w):
    lw = np.nan_to_num(log_w, nan=-np.inf)
    m = lw.max()
    w = np.exp(lw - m)
    return w, m


def run_metrics(r, ref):
    w, m = weights(r["log_w"])
    n = len(w)
    out = {"n": n, "ess": float(w.sum() ** 2 / (n * (w ** 2).sum())),
           "log_Z": float(m + math.log(w.sum() / n))}
    out.update(metrics.peptide_metrics(ref, r, w / w.sum(), r["meta"]["rama_residues"]))
    return out


def residue_js(ref, r, w):
    """JS per residue (peptide_metrics logs the KL per residue and the JS only as a mean)."""
    out = {}
    for tag, ww in (("", None), ("_rw", w)):
        out[f"rama_js{tag}/res"] = [metrics.js(metrics.rama_hist(ref["rama"][:, k, 0], ref["rama"][:, k, 1]),
                                               metrics.rama_hist(r["rama"][:, k, 0], r["rama"][:, k, 1], ww))
                                    for k in range(r["rama"].shape[1])]
    return out


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
    """'$m_{\\pm s}$' over seeds, in the paper's style (no leading zero below 1)."""
    v = np.asarray(vals, np.float64) * scale
    return f"${fmt(v.mean(), digits)}_{{\\pm{fmt(v.std(ddof=1) if len(v) > 1 else 0.0, digits)}}}$"


def ess_cell(vals):
    v = np.asarray(vals, np.float64)
    if v.max() < 1e-3:
        return "$<\\!10^{-3}$"
    return cell(v, 3)


def sci(v):
    """One ESS: three decimals, or a power of ten below 1e-3."""
    if v >= 1e-3:
        return f"${fmt(v, 3)}$"
    e = int(math.floor(math.log10(v)))
    return f"${v / 10 ** e:.1f}{{\\cdot}}10^{{{e}}}$"


def energy_hist(ref_e, r, w, lo, hi, bins=100):
    edges = np.linspace(lo, hi, bins + 1)
    dens = []
    for e, ww in ((ref_e, None), (r["energy"], None), (r["energy"], w)):
        h, _ = np.histogram(e, bins=edges, weights=ww)
        total = len(e) if ww is None else ww.sum()       # mass outside [lo, hi] (clashes) stays missing
        dens.append(h / total / np.diff(edges))
    centres = 0.5 * (edges[1:] + edges[:-1])
    return centres, dens


def wandb_costs(group, entity=None, project="digin-peptides"):
    """Wall-clock hours, GPU type and energy evaluations of the training runs (W&B summaries)."""
    try:
        import wandb
        api = wandb.Api(timeout=120)
        runs = api.runs(f"{entity or api.default_entity}/{project}", filters={"group": group})
        return {r.name: {"hours": r.summary.get("_runtime", 0) / 3600, "energy_calls": r.summary.get("energy_calls"),
                         "gpu": (r.metadata or {}).get("gpu")} for r in runs}
    except Exception as e:  # noqa: BLE001  (offline, or no key: the costs are optional)
        print(f"no W&B costs: {e}", flush=True)
        return {}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True, help="directory of the sample sets <system>_<coords>_s<seed>/")
    ap.add_argument("--ref_dir", required=True, help="directory of the MD references <system>_ref.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--wandb_group", default=None, help="also report training costs from this W&B group")
    ap.add_argument("--wandb_entity", default=None)
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    summary, macros, seed_rows, basin_rows = {}, [], [], []

    def mac(name, value):
        macros.append(f"\\newcommand{{\\{name}}}{{{value}}}")

    for system, stag in SYSTEMS.items():
        r = np.load(os.path.join(a.ref_dir, f"{system}_ref.npz"))
        ref = {"energy": np.asarray(r["energy"], np.float64), "rama": np.asarray(r["rama"], np.float32)}
        mac(f"Pep{stag}NFrames", f"{len(ref['energy']):,}".replace(",", "{,}"))
        lo, hi = np.quantile(ref["energy"], [5e-4, 0.9995])
        lo, hi = lo - 0.15 * (hi - lo), hi + 0.35 * (hi - lo)
        R = ref["rama"].shape[1]
        for k in range(R):
            save_surface(os.path.join(a.out, f"rama_{system}_md_r{k}.png"),
                         log_density(ref["rama"][:, k, 0], ref["rama"][:, k, 1]))
        names = None
        basins = {"md": metrics.basin_fractions(ref["rama"])}
        for coords, ctag in COORDS.items():
            per, pooled = [], {"rama": [], "w": [], "energy": [], "wn": []}
            for seed in SEEDS:
                run = load_run(a.samples, system, coords, seed)
                names = run["meta"]["rama_residues"]
                w, _ = weights(run["log_w"])
                m = run_metrics(run, ref)
                m.update(residue_js(ref, run, w / w.sum()))
                m["step"] = run["meta"]["step"]
                per.append(m)
                summary[f"{system}_{coords}_s{seed}"] = m
                pooled["rama"].append(run["rama"]); pooled["energy"].append(run["energy"])
                pooled["w"].append(w / w.sum() / len(SEEDS)); pooled["wn"].append(np.full(len(w), 1.0 / len(w) / len(SEEDS)))
                seed_rows.append(" & ".join([
                    {"aldp": "ALDP", "al4": "AL4"}[system] if seed == 0 and coords == "internal" else "",
                    coords if seed == 0 else "", str(seed),
                    sci(m["ess"]),
                    f"${m['log_Z']:.3f}$", f"${m['energy_w1']:.2f}$", f"${m['energy_w1_rw']:.2f}$",
                    f"${1e3 * m['rama_js']:.2f}$", f"${1e3 * m['rama_js_rw']:.2f}$"]) + " \\\\")
            P = f"Pep{stag}{ctag}"
            mac(f"{P}ESS", ess_cell([m["ess"] for m in per]))
            mac(f"{P}LogZ", cell([m["log_Z"] for m in per], 3))
            for key, mname, dig, sc in (("energy_w1", "WOne", 2, 1.0), ("rama_js", "JS", 2, 1e3),
                                        ("rama_kl", "KL", 2, 1e3)):
                mac(f"{P}{mname}Raw", cell([m[key] for m in per], dig, sc))
                mac(f"{P}{mname}", cell([m[key + "_rw"] for m in per], dig, sc))
            if R > 1:
                for k in range(R):
                    for tag, tname in (("", "Raw"), ("_rw", "")):
                        mac(f"{P}JS{tname}Res{'ABC'[k]}", cell([m[f"rama_js{tag}/res"][k] for m in per], 2, 1e3))
            rama = np.concatenate(pooled["rama"]); w = np.concatenate(pooled["w"]); wn = np.concatenate(pooled["wn"])
            for k in range(R):
                for tag, ww in (("raw", wn), ("rw", w)):
                    save_surface(os.path.join(a.out, f"rama_{system}_{coords}_{tag}_r{k}.png"),
                                 log_density(rama[:, k, 0], rama[:, k, 1], ww))
            basins[f"{coords}_raw"] = metrics.basin_fractions(rama, wn)
            basins[f"{coords}_rw"] = metrics.basin_fractions(rama, w)
            e_pool = {"energy": np.concatenate(pooled["energy"])}
            c, (dm, dr, dw) = energy_hist(ref["energy"], e_pool, w, lo, hi)
            with open(os.path.join(a.out, f"energy_{system}_{coords}.dat"), "w") as f:
                f.write("E md raw rw\n")
                for row in zip(c, dm, dr, dw):
                    f.write(" ".join(f"{v:.6g}" for v in row) + "\n")
        mac(f"Pep{stag}ELo", f"{lo:.0f}"); mac(f"Pep{stag}EHi", f"{hi:.0f}")
        # Cartesian minus internal log Z against the methyl symmetry factor
        dz = [summary[f"{system}_cartesian_s{s}"]["log_Z"] - summary[f"{system}_internal_s{s}"]["log_Z"] for s in SEEDS]
        mac(f"Pep{stag}DLogZ", cell(dz, 3))
        mac(f"Pep{stag}DLogZExpected", f"{METHYLS[system] * math.log(2):.3f}")
        mac(f"Pep{stag}Methyls", str(METHYLS[system]))
        summary[f"{system}_basins"] = {k: v.tolist() for k, v in basins.items()}
        summary[f"{system}_residues"] = names
        if R > 1:
            for k in range(R):
                for b, bname in enumerate(("$\\alpha_R$", "$\\beta$/PPII", "$\\alpha_L$")):
                    vals = [basins[key][k, b] for key in ("md", "internal_raw", "internal_rw", "cartesian_raw", "cartesian_rw")]
                    basin_rows.append(" & ".join([names[k] if b == 0 else "", bname] + [f"${v:.3f}$" for v in vals]) + " \\\\")
        mac(f"Pep{stag}NSamples", f"{per[0]['n'] / 1e6:.0f}M")

    costs = wandb_costs(a.wandb_group, a.wandb_entity) if a.wandb_group else {}
    if costs:
        summary["costs"] = costs
        for system, stag in SYSTEMS.items():
            for coords, ctag in COORDS.items():
                c = [costs[f"{system}_{coords}_s{s}"] for s in SEEDS if f"{system}_{coords}_s{s}" in costs]
                if c:
                    mac(f"Pep{stag}{ctag}Hours", f"{np.mean([x['hours'] for x in c]):.0f}")
                    ec = [x["energy_calls"] for x in c if x["energy_calls"]]
                    if ec:
                        mac(f"Pep{stag}{ctag}ECalls", f"{np.mean(ec) / 1e9:.1f}")
    mac("PepFECap", f"{FE_CAP:.0f}")
    mac("PepNSeeds", str(len(SEEDS)))
    with open(os.path.join(a.out, "numbers.tex"), "w") as f:
        f.write("% generated by digin/paper.py (digin-peptides) -- do not edit\n" + "\n".join(macros) + "\n")
    with open(os.path.join(a.out, "seeds.tex"), "w") as f:
        f.write("\n".join(seed_rows) + "\n")
    with open(os.path.join(a.out, "basins.tex"), "w") as f:
        f.write("\n".join(basin_rows) + "\n")
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=float)
    print("\n".join(macros), flush=True)


if __name__ == "__main__":
    main()
