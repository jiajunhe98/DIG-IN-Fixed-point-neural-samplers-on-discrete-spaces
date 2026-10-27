"""Reference MD with the torch force field: batched Langevin (BAOAB) + parallel tempering.

The force field is the copy of amber99sbildn + OBC in forcefield.py (it matches
OpenMM per force group), so the reference samples precisely the sampler's
target.  All replicas live in one tensor, so a small molecule uses the GPU as a
whole; replica exchange along a geometric temperature ladder equilibrates the
basin populations at 300 K.  Hydrogen mass repartitioning allows 2 fs steps
without changing the configurational distribution.

    python -m digin.md run --system al4 --n_temps 8 --copies 64 --t_max 600 --burn_ps 500 --hours 12 \
        --out $AMLT_OUTPUT_DIR/md --publish /mnt/data/digin_aldp/al4_ref.npz
    python -m digin.md freeze <live.npz> <frozen.npz>      # fixed snapshot for evaluation

Saved per frame of the 300 K replicas: potential energy, per-residue (phi, psi), and
the internal coordinates (molecule.py) of every `z_every`-th frame.
"""
from __future__ import annotations

import argparse
import math
import os
import shutil
import time

import numpy as np
import torch

from digin import systems
from digin.forcefield import AmberOBC
from digin.molecule import Molecule

MASS = {"H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999}


def publish(src, dst):
    if not dst:
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copyfile(src, dst + ".part")
    shutil.copyfile(dst + ".part", dst)
    os.remove(dst + ".part")


def freeze(src: str, dst: str, tries: int = 10) -> None:
    """Copy a (possibly still growing) reference to a fixed file, retrying while it is being rewritten."""
    if os.path.exists(dst):
        r = np.load(dst)
        print(f"{dst} exists: {len(r['energy'])} frames; not overwritten", flush=True)
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    for t in range(tries):
        tmp = dst + ".part.npz"
        shutil.copyfile(src, tmp)
        try:
            r = np.load(tmp)
            n = len(r["energy"])
            assert r["rama"].shape[0] == n and r["z"].shape[0] > 0
        except Exception as e:  # noqa: BLE001  (caught the publisher mid-write)
            print(f"try {t}: {src} unreadable ({e}); retrying", flush=True)
            time.sleep(60)
            continue
        os.replace(tmp, dst)
        print(f"froze {src} -> {dst}: {n} frames, {r['z'].shape[0]} internal-coordinate frames", flush=True)
        return
    raise SystemExit(f"could not read {src}")


def run(a):
    os.makedirs(a.out, exist_ok=True)
    dev = torch.device(a.device if a.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(a.seed)

    mol = Molecule(a.system)
    ff64 = AmberOBC(systems.paths(a.system)[1]).to(dev)
    ff = AmberOBC(systems.paths(a.system)[1]).to(dev).float()
    n, nT, C, K = mol.n, a.n_temps, a.copies, a.swap_every
    assert a.save_every % K == 0, "save_every must be a multiple of swap_every"
    T = torch.tensor(np.geomspace(a.t_min, a.t_max, nT), dtype=torch.float32, device=dev)   # (nT,)
    kT = (systems.KB * T).view(nT, 1, 1, 1)
    beta = 1.0 / (systems.KB * T)
    mass = np.array([MASS[at["element"]] for at in mol.atoms])
    if a.h_mass > 0:
        for i, j in mol.bonds:
            for h, heavy in ((i, j), (j, i)):
                if mol.atoms[h]["element"] == "H" and mol.atoms[heavy]["element"] != "H":
                    mass[heavy] -= a.h_mass - MASS["H"]
                    mass[h] = a.h_mass
    m = torch.tensor(mass, dtype=torch.float32, device=dev).view(1, 1, n, 1)
    dt = a.dt_fs * 1e-3
    c1 = math.exp(-a.friction * dt)
    c2 = math.sqrt(1 - c1 * c1)
    sig = torch.sqrt(kT / m)                                                  # (nT, 1, n, 1)
    x0 = torch.as_tensor(mol.x_min, dtype=torch.float32, device=dev)
    # static state, updated in place (CUDA-graph friendly)
    x = x0.expand(nT, C, n, 3).clone() + 0.002 * torch.randn(nT, C, n, 3, device=dev)
    v = sig * torch.randn_like(x)
    F = torch.zeros_like(x)
    E = torch.zeros(nT, C, device=dev)

    def compute_force():
        xx = x.detach().requires_grad_(True)
        En = ff(xx.view(-1, n, 3)).view(nT, C)
        g, = torch.autograd.grad(En.sum(), xx)
        F.copy_(-g)
        E.copy_(En.detach())

    def one_step():
        v.add_(F / m, alpha=0.5 * dt)
        x.add_(v, alpha=0.5 * dt)
        v.mul_(c1).add_(sig * torch.randn_like(v), alpha=c2)
        x.add_(v, alpha=0.5 * dt)
        compute_force()
        v.add_(F / m, alpha=0.5 * dt)

    def k_steps():
        for _ in range(K):
            one_step()

    compute_force()
    graph = None
    if a.cuda_graph and dev.type == "cuda":
        try:
            s_ = torch.cuda.Stream()
            s_.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s_):
                for _ in range(3):
                    one_step()
            torch.cuda.current_stream().wait_stream(s_)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                k_steps()
            graph.replay()
            torch.cuda.synchronize()
            E_chk = E.clone()
            compute_force()            # eager recomputation at the replayed state must agree
            err = float((E - E_chk).abs().max())
            if not math.isfinite(err) or err > max(1e-2, 1e-5 * float(E.abs().max())):
                raise RuntimeError(f"graph/eager energy mismatch {err}")
            print(f"CUDA graph of {K} steps captured (check |dE| = {err:.2e})", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"CUDA graph capture failed, running eagerly: {e}", flush=True)
            graph = None
            x.copy_(x0.expand(nT, C, n, 3) + 0.002 * torch.randn(nT, C, n, 3, device=dev))
            v.copy_(sig * torch.randn_like(x))
            compute_force()

    rec = {"energy": [], "rama": [], "z": []}
    acc_n = torch.zeros(nT - 1, device=dev); acc_y = torch.zeros(nT - 1, device=dev)
    burn = int(a.burn_ps * 1000 / a.dt_fs)
    max_steps = burn + int(a.max_ns * 1e6 / a.dt_fs)
    t0 = time.time(); last_pub = t0; last_print = t0
    step, n_saved, n_reset, parity = 0, 0, 0, 0
    local = os.path.join(a.out, f"{a.system}_ref.npz")
    ci = torch.arange(C, device=dev).view(1, C).expand(nT, C)

    def dump():
        if not rec["energy"]:
            return
        out = {"energy": np.concatenate(rec["energy"]), "rama": np.concatenate(rec["rama"]),
               "z": np.concatenate(rec["z"]), "t_min": a.t_min, "ns_per_replica": (step - burn) * a.dt_fs * 1e-6}
        np.savez(local, **out)
        publish(local, a.publish)
        print(f"published {len(out['energy'])} frames ({out['ns_per_replica']:.2f} ns per replica)", flush=True)

    while True:
        if graph is not None:
            graph.replay()
        else:
            k_steps()
        step += K
        bad = ~torch.isfinite(E) | ~torch.isfinite(x).view(nT, C, -1).all(-1)
        if bad.any():
            n_reset += int(bad.sum())
            x[bad] = x0
            v[bad] = (sig * torch.randn_like(x))[bad]
            compute_force()
        if nT > 1:          # replica exchange between neighbouring temperatures, alternating parity
            perm = torch.arange(nT, device=dev).view(nT, 1).expand(nT, C).clone()
            for i in range(parity, nT - 1, 2):
                d = (beta[i] - beta[i + 1]) * (E[i] - E[i + 1])
                ok = torch.log(torch.rand(C, device=dev)) < d
                acc_n[i] += C; acc_y[i] += ok.sum()
                perm[i, ok] = i + 1; perm[i + 1, ok] = i
            parity ^= 1
            scale = torch.sqrt(T.view(nT, 1) / T[perm]).view(nT, C, 1, 1)
            x.copy_(x[perm, ci]); v.copy_(v[perm, ci] * scale); F.copy_(F[perm, ci]); E.copy_(E[perm, ci])
        if step > burn and step % a.save_every == 0:
            with torch.no_grad():
                xs = x[0].double()
                rec["energy"].append(ff64(xs).cpu().numpy())
                rec["rama"].append(mol.rama(xs).float().cpu().numpy())
                if n_saved % a.z_every == 0:
                    rec["z"].append(mol.to_internal(xs).float().cpu().numpy())
            n_saved += 1
        now = time.time()
        if now - last_print > a.print_every_s or step >= max_steps:
            ns = step * a.dt_fs * 1e-6
            ke_T = ((m * v * v).sum((-1, -2)) / (3 * n * systems.KB)).mean(1)
            print(f"step {step} ({ns:.3f} ns/replica, {ns / ((now - t0) / 86400):.1f} ns/day/replica, "
                  f"{nT * C} replicas, graph={graph is not None})  <E>_Tmin {E[0].mean():.1f}  "
                  f"T_kin {ke_T[0]:.0f}/{ke_T[-1]:.0f} K  "
                  f"swap acc {[round(float(y / max(nn, 1)), 2) for y, nn in zip(acc_y, acc_n)]}  resets {n_reset}",
                  flush=True)
            last_print = now
        if now - last_pub > a.publish_every_min * 60:
            dump(); last_pub = now
        if now - t0 > a.hours * 3600 or step >= max_steps:
            break
    dump()
    print("=== DONE", flush=True)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--system", required=True, choices=list(systems.SYSTEMS))
    r.add_argument("--n_temps", type=int, default=8)
    r.add_argument("--copies", type=int, default=64)
    r.add_argument("--t_min", type=float, default=300.0)
    r.add_argument("--t_max", type=float, default=600.0)
    r.add_argument("--dt_fs", type=float, default=2.0)
    r.add_argument("--h_mass", type=float, default=4.0,
                   help="hydrogen mass repartitioning (amu); 0 = off.  Masses do not enter the configurational "
                        "Boltzmann distribution, so the reference is unchanged; it only allows the 2 fs step")
    r.add_argument("--cuda_graph", type=int, default=1, help="replay swap_every steps as one CUDA graph")
    r.add_argument("--friction", type=float, default=1.0, help="1/ps")
    r.add_argument("--swap_every", type=int, default=50)
    r.add_argument("--save_every", type=int, default=500, help="steps between saved frames (1 ps at 2 fs)")
    r.add_argument("--z_every", type=int, default=10, help="keep internal coordinates of every k-th frame")
    r.add_argument("--burn_ps", type=float, default=200.0)
    r.add_argument("--hours", type=float, default=12.0)
    r.add_argument("--max_ns", type=float, default=1e9, help="per replica")
    r.add_argument("--publish_every_min", type=float, default=20.0)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--out", required=True)
    r.add_argument("--publish", default=None)
    r.add_argument("--device", default="auto")
    r.add_argument("--print_every_s", type=float, default=300.0)
    f = sub.add_parser("freeze")
    f.add_argument("src")
    f.add_argument("dst")
    a = ap.parse_args()
    if a.cmd == "freeze":
        freeze(a.src, a.dst)
    else:
        run(a)


if __name__ == "__main__":
    main()
