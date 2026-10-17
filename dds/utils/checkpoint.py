"""Preemption-safe checkpointing, including on network-mounted run directories.

Preemptible cluster nodes can kill a job at any instant, including in the
middle of a write.  The obvious scheme -- write to a temp file and
`os.replace` -- is *not* available everywhere: object-storage mounts (e.g.
blobfuse, s3fs) do not provide an atomic rename.  Relying on one is the
failure that looks like it works locally and loses a multi-hour run on the
cluster.

What is used instead:

* **two alternating payload slots**, `ckpt_a.pt` / `ckpt_b.pt`, so the
  previous good checkpoint is never the file being overwritten;
* **a SHA-256 sidecar per slot**, written after its payload, so a slot that
  was cut short by a kill is detectably incomplete rather than silently
  truncated;
* **a pointer file written last**, naming the slot that just completed.
  Ordering is the whole mechanism: until the pointer is updated, the
  previous slot is still the one that gets loaded.

Loading verifies the pointed-at slot against its checksum and falls back to
the other slot when it fails, so any single interrupted write is survivable.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from typing import Any

import numpy as np
import torch

_SLOTS = ("a", "b")


def rng_state() -> dict:
    """Every RNG stream the training loop consumes."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def set_rng_state(state: dict):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") is not None and torch.cuda.is_available():
        cuda = state["cuda"]
        if len(cuda) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all([c.cpu() for c in cuda])


def _sha256(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _paths(base: str, slot: str) -> tuple[str, str]:
    stem = os.path.splitext(base)[0]
    return f"{stem}_{slot}.pt", f"{stem}_{slot}.sha256"


def _pointer_path(base: str) -> str:
    return os.path.splitext(base)[0] + ".ptr.json"


def _read_pointer(base: str) -> dict | None:
    p = _pointer_path(base)
    if not os.path.exists(p):
        return None
    try:
        with open(p) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _fsync(f):
    f.flush()
    os.fsync(f.fileno())


def save_checkpoint(base: str, payload: dict[str, Any]):
    """Write `payload` into the slot that is *not* currently pointed at."""
    ptr = _read_pointer(base)
    prev = ptr.get("slot") if ptr else None
    slot = _SLOTS[0] if prev != _SLOTS[0] else _SLOTS[1]
    data_path, sum_path = _paths(base, slot)

    # order matters: payload, then its checksum, then the pointer
    with open(data_path, "wb") as f:
        torch.save(payload, f)
        _fsync(f)
    digest = _sha256(data_path)
    with open(sum_path, "w") as f:
        f.write(digest)
        _fsync(f)
    with open(_pointer_path(base), "w") as f:
        json.dump({"slot": slot, "step": int(payload.get("step", -1)),
                   "sha256": digest}, f)
        _fsync(f)


def _load_slot(base: str, slot: str, map_location) -> dict | None:
    data_path, sum_path = _paths(base, slot)
    if not (os.path.exists(data_path) and os.path.exists(sum_path)):
        return None
    try:
        with open(sum_path) as f:
            expected = f.read().strip()
        if _sha256(data_path) != expected:
            print(f"[resume] slot {slot} fails its checksum; ignoring")
            return None
        return torch.load(data_path, map_location=map_location, weights_only=False)
    except Exception as e:
        print(f"[resume] slot {slot} unreadable ({e}); ignoring")
        return None


def load_checkpoint(base: str, map_location="cpu") -> tuple[dict | None, str | None]:
    """Load the newest verifiable checkpoint.

    Tries the pointed-at slot first, then the other one.  Returns
    (payload, description) or (None, None) when there is nothing to resume.
    """
    ptr = _read_pointer(base)
    order = list(_SLOTS)
    if ptr and ptr.get("slot") in _SLOTS:
        order = [ptr["slot"]] + [s for s in _SLOTS if s != ptr["slot"]]
    for slot in order:
        payload = _load_slot(base, slot, map_location)
        if payload is not None:
            return payload, _paths(base, slot)[0]
    return None, None
