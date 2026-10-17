"""Stitch a sharded grid evaluation back into one sweep.

    python scripts/merge_alloy_shards.py <run_dir> [--keep-shards]

`dump_alloy_samples.py --shard i --n-shards k` writes one npz and one rows
file per shard, each carrying the indices of the state points it evaluated.
This puts them back in the sweep's own order and writes the same
`grid_samples.npz` and `grid.json` an unsharded run would have produced, so
everything downstream -- the report, the common-tangent construction -- is
unaware the evaluation was split.

Refuses to merge a partial set: a sweep missing state points would silently
produce isotherms with holes in delta_mu, and the phase diagram built from
them would be wrong in a way nothing downstream could detect.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--keep-shards", action="store_true")
    ap.add_argument("--allow-partial", action="store_true",
                    help="merge what is there; the sweep will have holes")
    a = ap.parse_args()

    npzs = sorted(glob.glob(os.path.join(a.run_dir, "grid_samples.shard*.npz")))
    if not npzs:
        raise SystemExit(f"no shards under {a.run_dir}")

    idx, parts, rows = [], [], []
    n_sites = system = note = None
    for p in npzs:
        # `np.load` on an npz holds the file open and reads members lazily, so
        # the arrays have to be materialised and the handle closed inside the
        # loop.  Leaving it to the garbage collector works on Linux and fails
        # on Windows, where the shard cleanup below then dies with "the
        # process cannot access the file because it is being used by another
        # process" -- after the merge has already been written, so the run
        # looks broken when only the tidy-up was.
        with np.load(p, allow_pickle=False) as z:
            if "point_index" not in z:
                raise SystemExit(f"{p} predates sharding: no point_index")
            idx.append(np.asarray(z["point_index"]))
            parts.append({k: np.asarray(z[k])
                          for k in ("x_packed", "log_w", "energy", "cond")})
            n_sites = int(z["n_sites"])
            system = str(z["system"]) if "system" in z else system
            note = str(z["note"]) if "note" in z else note
        m = re.search(r"shard(\d+)\.npz$", p)
        rp = os.path.join(a.run_dir, f"grid_rows.shard{m.group(1)}.json")
        rows.append(json.load(open(rp, encoding="utf-8"))
                    if os.path.exists(rp) else None)

    order = np.concatenate(idx)
    n = int(order.max()) + 1
    if len(order) != n or len(set(order.tolist())) != len(order):
        msg = (f"shards cover {len(order)} of {n} state points"
               f" ({n - len(set(order.tolist()))} missing or duplicated)")
        if not a.allow_partial:
            raise SystemExit(msg + "; refusing to merge a sweep with holes")
        print("WARNING: " + msg, flush=True)

    take = np.argsort(order, kind="stable")
    merged = {k: np.concatenate([p[k] for p in parts])[take]
              for k in ("x_packed", "log_w", "energy", "cond")}
    out = os.path.join(a.run_dir, "grid_samples.npz")
    np.savez_compressed(out, n_sites=np.int64(n_sites),
                        system=system or "", note=note or "", **merged)
    print(f"merged {len(npzs)} shards, {len(order)} state points, "
          f"{merged['log_w'].shape[1]} draws each -> {out} "
          f"({os.path.getsize(out) / 1e6:.1f} MB)")

    if all(r is not None for r in rows):
        flat = [r for part in rows for r in part]
        flat = [flat[i] for i in take]
        with open(os.path.join(a.run_dir, "grid.json"), "w", encoding="utf-8") as f:
            json.dump(flat, f)
        print(f"  wrote grid.json with {len(flat)} rows")

    if not a.keep_shards:
        # only after the merge is on disk and readable -- a cleanup that runs
        # before that can, and once did, delete the inputs while leaving the
        # output unverified
        with np.load(out, allow_pickle=False) as chk:
            assert chk["log_w"].shape[0] == len(order), "merged sweep is short"
        for p in npzs:
            os.remove(p)
        for p in glob.glob(os.path.join(a.run_dir, "grid_rows.shard*.json")):
            os.remove(p)
        print("  removed the shard files")


if __name__ == "__main__":
    main()
