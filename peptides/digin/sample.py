"""Draw samples of a trained run with their likelihoods and energies, for downstream evaluation.

    python -m digin.sample --run_dir runs/al4_cart_s0 --out samples/al4_cart_s0 --n 2000000

Loads the EMA model of the run's last checkpoint and writes to --out one .npy per field
(np.load(..., mmap_mode="r") reads them without loading everything):

  x         (n, atoms, 3) float32  configuration in nm, atoms in the order of topology.pdb, in the
                                   canonical frame (central CA at the origin, its N on +x, its C in
                                   the xy-plane)
  energy    (n,) float64           potential energy E(x) in kJ/mol (amber99sbildn + OBC)
  log_q     (n,) float64           log q(bins | order): exact log-likelihood of the drawn bins under
                                   the order it was drawn in (what the importance weights use)
  log_q_x   (n,) float64           the same as a density on the 3n - 6 frame coordinates of x:
                                   log_q - log(cell volume) - log|J(z)|, the same space for internal
                                   and Cartesian runs.  Their supports differ by the methyl hydrogens:
                                   internal boxes fix the cyclic order of the three H's, Cartesian
                                   boxes allow both, so a Cartesian model spreads over 2^(methyls)
                                   copies (log Z higher by log 8 for aldp, log 32 for al4)
  log_w     (n,) float64           importance log-weight -E/kT - log_q_x (+ the chirality penalty of
                                   Cartesian runs), unnormalized
  rama      (n, R, 2) float32      backbone (phi, psi) in radians of every residue in meta.json

and meta.json (n, kT, checkpoint step, definitions) and topology.pdb.  The draws are made in
parts of --part samples, each from its own seeded generator (independent of the training run's
final evaluation) and saved as it finishes; a restart skips the parts on disk.  meta.json is
written last, so its presence marks a complete output.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import signal
import sys
import time

import numpy as np
import torch

from digin import systems
from digin.model import any_order_sample
from digin.train import Trainer, ess_stats, parse

FIELDS = ("x", "energy", "log_q", "log_q_x", "log_w", "rama")
SUMMARY = ("run", "step", "n", "ess", "log_Z", "frac_chirality_penalized", "max_rel_dev_log_w_identity")


def find_run(name: str, root: str) -> str:
    """The output directory of a finished training job: <root>/*/<name> with result.json (newest)."""
    cands = [os.path.dirname(p) for p in glob.glob(os.path.join(root, "*", name, "result.json"))]
    if not cands:
        raise SystemExit(f"no finished run {name} under {root}")
    return max(cands, key=lambda d: os.path.getmtime(os.path.join(d, "result.json")))


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


def chirality_ok(tg, x: np.ndarray) -> np.ndarray:
    """(n,) True where every CA is L, i.e. where a Cartesian target's chirality penalty is off."""
    if not hasattr(tg, "chiral"):
        return np.ones(len(x), dtype=bool)
    q = tg.chiral.cpu().numpy()
    ok = np.ones(len(x), dtype=bool)
    for s in range(0, len(x), 1 << 18):
        xs = np.asarray(x[s:s + (1 << 18)], dtype=np.float64)
        ca = xs[:, q[:, 1]]
        vol = np.einsum("nij,nij->ni", xs[:, q[:, 0]] - ca, np.cross(xs[:, q[:, 2]] - ca, xs[:, q[:, 3]] - ca))
        ok[s:s + len(xs)] = (vol > 0).all(1)
    return ok


@torch.no_grad()
def draw(t: Trainer, n: int, generator) -> dict:
    tg = t.tg
    b, lq = any_order_sample(t.ema, n, tg.D, tg.L, t.device, generator)
    z = tg.dequantize(b, generator)
    x = tg.to_cartesian(z)
    lq = lq.double()
    energy = tg.ff(x).double()
    log_q_x = lq - tg.log_vol - tg.log_jacobian(z)
    return {"x": x.float(), "energy": energy, "log_q": lq, "log_q_x": log_q_x,
            "log_w": tg.log_p(z) - lq, "rama": tg.mol.rama(x).float()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", default=None, help="training output directory (or --run + --runs_root)")
    ap.add_argument("--run", default=None, help="job name, looked up under --runs_root")
    ap.add_argument("--runs_root", default=None, help="with --run: the directory holding */<run>/ outputs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=2_000_000)
    ap.add_argument("--part", type=int, default=65536)
    ap.add_argument("--chunk", type=int, default=4096, help="samples per network batch")
    ap.add_argument("--seed_offset", type=int, default=7_000_003)
    ap.add_argument("--allow_unfinished", action="store_true")
    s = ap.parse_args(argv)
    if not s.run_dir and not (s.run and s.runs_root):
        raise SystemExit("give --run_dir, or --run with --runs_root")
    run_dir = s.run_dir or find_run(s.run, s.runs_root)
    if os.path.exists(os.path.join(s.out, "meta.json")):
        meta = json.load(open(os.path.join(s.out, "meta.json")))
        remove_parts(os.path.join(s.out, "parts"))
        print(f"{s.out} is complete (meta.json):", json.dumps({k: meta.get(k) for k in SUMMARY}), flush=True)
        return
    saved = json.load(open(os.path.join(run_dir, "args.json")))
    a = parse(["--system", saved["system"], "--coords", saved["coords"], "--out", run_dir])
    vars(a).update({k: v for k, v in saved.items() if k != "device"})
    a.out, a.label_budget = run_dir, saved.get("label_budget", 0)
    t = Trainer(a)
    if not t.load():
        raise SystemExit(f"no checkpoint in {run_dir}")
    if t.step < a.steps and not s.allow_unfinished:
        raise SystemExit(f"{run_dir} is at step {t.step} < {a.steps}")
    tg = t.tg
    print(f"{run_dir}: {a.system} [{a.coords}] step {t.step}, {s.n} samples -> {s.out}", flush=True)

    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    pdir = os.path.join(s.out, "parts")
    os.makedirs(pdir, exist_ok=True)
    n_parts = (s.n + s.part - 1) // s.part
    t0 = time.time()
    for k in range(n_parts):
        path = os.path.join(pdir, f"part_{k:05d}.npz")
        if os.path.exists(path):
            continue
        m = min(s.part, s.n - k * s.part)
        g = torch.Generator(device=t.device).manual_seed(s.seed_offset * (a.seed + 1) + k)
        parts = [draw(t, min(s.chunk, m - i), g) for i in range(0, m, s.chunk)]
        tmp = os.path.join(pdir, f"tmp_{k:05d}.npz")
        np.savez(tmp, **{f: torch.cat([p[f] for p in parts]).cpu().numpy() for f in FIELDS})
        os.replace(tmp, path)
        print(f"  part {k + 1}/{n_parts} ({time.time() - t0:.0f} s)", flush=True)
        if stop["flag"]:
            print(f"SIGTERM: {k + 1}/{n_parts} parts saved", flush=True)
            sys.exit(143)

    files = [np.load(os.path.join(pdir, f"part_{k:05d}.npz")) for k in range(n_parts)]
    for f in FIELDS:
        np.save(os.path.join(s.out, f"{f}.npy"), np.concatenate([fl[f] for fl in files]))
    x, energy, log_q_x, log_w = (np.load(os.path.join(s.out, f"{f}.npy"), mmap_mode="r")
                                 for f in ("x", "energy", "log_q_x", "log_w"))
    assert len(energy) == s.n
    # log_w = -E/kT - log_q_x wherever the chirality penalty is off (relative: clashes reach 1e10 kT)
    ok = chirality_ok(tg, x)
    dev = np.abs(log_w + energy / tg.kT + log_q_x) / (1.0 + np.abs(energy / tg.kT))
    st_ = ess_stats(torch.as_tensor(np.array(log_w)))
    shutil.copyfile(systems.paths(a.system)[0], os.path.join(s.out, "topology.pdb"))
    meta = {"run": os.path.basename(os.path.normpath(run_dir)), "run_dir": run_dir, "step": t.step,
            "system": a.system, "coords": a.coords, "seed": a.seed, "n": s.n, "n_atoms": tg.mol.n,
            "rama_residues": tg.mol.rama_res, "kT_kJ_per_mol": tg.kT, "temperature_K": float(tg.ff.temperature),
            "sampling_seeds": f"part k: {s.seed_offset} * (seed + 1) + k, parts of {s.part}",
            "ess": st_["ess"], "log_Z": st_.get("log_Z"), "frac_chirality_penalized": float(1 - ok.mean()),
            "max_rel_dev_log_w_identity": float(dev[ok].max()) if ok.any() else None,
            "fields": {"x": "nm, canonical frame, atom order of topology.pdb", "energy": "kJ/mol",
                       "log_q": "log q(bins | sampled order)",
                       "log_q_x": "log density on the 3n-6 frame coordinates of x (Cartesian runs cover "
                                  "both cyclic orders of each methyl's H's, internal runs one)",
                       "log_w": "-E/kT - log_q_x (+ chirality penalty), unnormalized",
                       "rama": "(phi, psi) radians"},
            "seconds": time.time() - t0}
    with open(os.path.join(s.out, "meta.json.tmp"), "w") as f:
        json.dump(meta, f, indent=1)
    os.replace(os.path.join(s.out, "meta.json.tmp"), os.path.join(s.out, "meta.json"))
    remove_parts(pdir)
    print("=== DONE", json.dumps({k: meta[k] for k in SUMMARY}), flush=True)


if __name__ == "__main__":
    main()
