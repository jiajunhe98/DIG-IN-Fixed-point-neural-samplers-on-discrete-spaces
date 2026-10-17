"""Generate reference MCMC samples for a target and cache them on disk.

    python scripts/generate_reference.py --target ising --lattice 16 16 \
        --beta 0.4407 --n 8192 --method mixed
"""

from __future__ import annotations

import argparse

import torch

from dds.evaluation import get_reference_samples, reference_cache_key
from dds.targets import IsingModel, PottsModel
from dds.utils.metrics import summarize_observables


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", choices=["ising", "potts"], required=True)
    p.add_argument("--lattice", type=int, nargs="+", default=[16, 16])
    p.add_argument("--beta", type=float, required=True)
    p.add_argument("--J", type=float, default=1.0)
    p.add_argument("--h", type=float, default=0.0)
    p.add_argument("--q", type=int, default=3)
    p.add_argument("--n", type=int, default=8192)
    p.add_argument(
        "--method", default="mixed",
        choices=["metropolis", "heat_bath", "wolff", "swendsen_wang", "mixed"],
    )
    p.add_argument("--burn-in", type=int, default=1000)
    p.add_argument("--thin", type=int, default=10)
    p.add_argument("--cache-dir", default="data/reference")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    if args.target == "ising":
        target = IsingModel(tuple(args.lattice), beta=args.beta, J=args.J, h=args.h)
    else:
        target = PottsModel(tuple(args.lattice), q=args.q, beta=args.beta, J=args.J)

    x = get_reference_samples(
        target, n=args.n, cache_dir=args.cache_dir, method=args.method,
        burn_in=args.burn_in, thin=args.thin, device=args.device,
    )
    print(f"cached {x.shape[0]} samples ({reference_cache_key(target)})")
    for k, v in summarize_observables(target, x).items():
        print(f"  {k} = {v:.5f}")
    lz = target.exact_log_Z()
    if lz is not None:
        print(f"  exact log Z = {lz:.5f}")


if __name__ == "__main__":
    main()
