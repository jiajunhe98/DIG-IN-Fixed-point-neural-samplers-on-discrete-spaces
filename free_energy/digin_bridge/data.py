"""FEAT's ALDP reference data: MD of the vacuum (lambda = 0) and the implicit
solvent (lambda = 1) state, from the Hugging Face dataset JJHE/FEAT (ALDP.zip).

    aldp_0.0_align.pt   vacuum  MD, (N, 22, 3) positions in nm, aligned
    aldp_1_align.pt     solvent MD, same layout

Only the positions are extracted; the force files of the archive are not
needed.  The given endpoint-0 samples are vacuum frames (all of them for
aldps, those in S_b for aldpt); the endpoint-1 frames (solvent, or vacuum in
S_a) enter only the evaluation and the reference checks.

    python -m digin_bridge.data --task aldps     # download, convert, check against FEAT's dF
"""
from __future__ import annotations

import argparse
import math
import os
import zipfile

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(os.path.dirname(HERE), "data", "feat_aldp")
FILES = {0: "aldp_0.0_align.pt", 1: "aldp_1_align.pt"}
# log Z_1 / Z_0, FEAT's MBAR references (He et al. 2025):
# aldps log Z_solvent / Z_vacuum (Table 1, +- 0.01), aldpt log Z_a / Z_b (F.1.4, +- 0.05)
REF = {"aldps": 29.43, "aldpt": -4.25}


def ensure(root: str | None = None) -> str:
    """A directory holding both position files (default data/feat_aldp); downloads them on first use."""
    root = root or DATA
    if all(os.path.exists(os.path.join(root, f)) for f in FILES.values()):
        return root
    from huggingface_hub import hf_hub_download
    os.makedirs(root, exist_ok=True)
    z = zipfile.ZipFile(hf_hub_download(repo_id="JJHE/FEAT", filename="ALDP.zip", repo_type="dataset"))
    for f in FILES.values():
        name = next(n for n in z.namelist() if n.endswith(f))
        tmp = os.path.join(root, f + f".part{os.getpid()}")
        with z.open(name) as src, open(tmp, "wb") as out:
            out.write(src.read())
        os.replace(tmp, os.path.join(root, f))
    return root


def positions(state: int, root: str | None = None) -> np.ndarray:
    """(N, 22, 3) float64 positions in nm of state 0 (vacuum) or 1 (solvent)."""
    x = torch.load(os.path.join(ensure(root), FILES[state]), map_location="cpu")
    return x.double().reshape(len(x), 22, 3).numpy()


def internal(state: int, root: str | None = None) -> np.ndarray:
    """(N, 60) float64 internal coordinates, cached next to the positions."""
    from digin_bridge import zmatrix as zm
    root = ensure(root)
    cache = os.path.join(root, f"z_{state}.npy")
    if os.path.exists(cache):
        return np.load(cache)
    z = zm.to_internal(torch.as_tensor(positions(state, root))).numpy()
    tmp = cache + f".part{os.getpid()}.npy"
    np.save(tmp, z)
    os.replace(tmp, cache)
    return z


def endpoint_frames(tg, root: str | None = None) -> tuple[np.ndarray, np.ndarray, dict, np.ndarray]:
    """(endpoint-0 frames, endpoint-1 frames, fraction of each MD set inside the box, index of each
    endpoint-0 frame in FEAT's vacuum MD) for tg.task; frames are internal coordinates inside the
    box and inside their state."""
    sets = {0: internal(0, root), 1: internal(1, root) if tg.task == "aldps" else internal(0, root)}
    out, frac, idx = {}, {}, {}
    for k, z in sets.items():
        zt = torch.as_tensor(z)
        inside = tg.to_bins(zt)[1].cpu()                  # to_bins works on tg's device
        frac[k] = float(inside.double().mean())
        keep = (inside & tg.in_endpoint(zt, k)).numpy()
        out[k], idx[k] = z[keep], np.nonzero(keep)[0]
    return out[0], out[1], frac, idx[0]


def bar(w_f: np.ndarray, w_r: np.ndarray, iters: int = 200) -> float:
    """Bennett acceptance ratio for log Z1/Z0 from forward work w_f = U1 - U0 on
    state-0 samples and reverse work w_r = U0 - U1 on state-1 samples (kT)."""
    nf, nr = len(w_f), len(w_r)
    m = math.log(nf / nr)
    lo, hi = -1e3, 1e3
    for _ in range(iters):
        df = 0.5 * (lo + hi)               # candidate F1 - F0 = -log Z1/Z0
        a = -np.logaddexp(0.0, m + w_f - df)
        b = -np.logaddexp(0.0, -m + w_r + df)
        g = np.logaddexp.reduce(a) - np.logaddexp.reduce(b)       # increasing in df
        lo, hi = (lo, df) if g > 0 else (df, hi)
    return -0.5 * (lo + hi)


def main(argv=None):
    from digin_bridge.target import BridgeTarget
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="aldps", choices=["aldps", "aldpt"])
    ap.add_argument("--root", default=None)
    ap.add_argument("--n", type=int, default=20000, help="frames per state for the energy check")
    a = ap.parse_args(argv)
    root = ensure(a.root)
    tg = BridgeTarget(task=a.task)
    z0, z1, frac, _ = endpoint_frames(tg, root)
    for k, z in ((0, z0), (1, z1)):
        print(f"endpoint {k}: {len(z)} frames in the box and the state (MD set inside the box {frac[k]:.6f})")
    if a.task == "aldpt":
        # both states are pieces of one vacuum MD: their frame counts estimate Z_a / Z_b directly
        print(f"log Z1/Z0: frame counts {math.log(len(z1) / len(z0)):.3f} (correlated MD, rough); "
              f"FEAT MBAR {REF[a.task]}")
        return
    g = np.random.default_rng(0)
    du = {}
    for s, z in ((0, z0), (1, z1)):
        zs = torch.as_tensor(z[g.choice(len(z), min(a.n, len(z)), replace=False)])
        _, polar = tg.parts(zs)
        du[s] = (polar / tg.kT).numpy()          # U_1 - U_0
    lr_fwd = float(torch.logsumexp(torch.as_tensor(-du[0]), 0) - math.log(len(du[0])))
    lr_rev = -float(torch.logsumexp(torch.as_tensor(du[1]), 0) - math.log(len(du[1])))
    print(f"log Z1/Z0: Zwanzig from vacuum {lr_fwd:.3f}, from solvent {lr_rev:.3f}, "
          f"BAR {bar(du[0], -du[1]):.3f}; FEAT MBAR {REF[a.task]}")


if __name__ == "__main__":
    main()
