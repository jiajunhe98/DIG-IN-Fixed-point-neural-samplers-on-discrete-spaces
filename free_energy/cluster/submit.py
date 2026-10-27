"""Submit to the Singularity cluster through amlt.

    # the 6 production runs: {aldps, aldpt} x seeds {0, 1, 2}, one setting
    python cluster/submit.py train bridge-v1 -d "DIG-IN bridge ALDP-S/ALDP-T, 3 seeds"
    python cluster/submit.py train bridge-v1 --tasks aldpt --seeds 0 --dry-run

    python cluster/submit.py test bridge-test -d "unit tests, OpenMM comparison, training smoke runs"

Writes a full amlt config (target, image, blob mount, the W&B key from ~/_netrc
as a job environment variable) to a temporary file and calls `amlt run` on it,
so no secret sits in the repository.  The cluster (target, image, storage,
managed identity) is read from cluster/amlt.yaml, which is not in git:
copy cluster/amlt.example.yaml and fill it in.  Jobs go round-robin over the SKUs.
Every job downloads FEAT's ALDP data from Hugging Face to node-local disk.
Output: $AMLT_OUTPUT_DIR/<job name>; W&B: project digin-bridge, group =
experiment.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

RUNS_ROOT = "/mnt/output/projects/digin-bridge/amlt-results"     # amlt's job outputs, seen from a job
SETUP = [
    "for i in 1 2 3; do python -m pip install --no-cache-dir wandb==0.28.2 matplotlib huggingface_hub && break; "
    "sleep 20; done",
    "python -c \"import torch, wandb, matplotlib, huggingface_hub; "
    "print('torch', torch.__version__, 'cuda', torch.cuda.is_available())\"",
]
SKUS = ["141G1-H200", "80G1-H100", "80G1-A100"]


def wandb_key() -> str:
    key = os.environ.get("WANDB_API_KEY")
    if key:
        return key
    import netrc
    for fn in ("~/_netrc", "~/.netrc"):
        p = os.path.expanduser(fn)
        if os.path.exists(p):
            auth = netrc.netrc(p).authenticators("api.wandb.ai")
            if auth:
                return auth[2]
    raise SystemExit("no W&B key: set WANDB_API_KEY or log in (wandb login)")


def cluster_config() -> dict:
    """Target, image, storage and identity of the amlt cluster, from cluster/amlt.yaml (not in git;
    cluster/amlt.example.yaml lists the keys)."""
    import yaml
    path = os.path.join(ROOT, "cluster", "amlt.yaml")
    if not os.path.exists(path):
        raise SystemExit(f"{path} is missing: copy cluster/amlt.example.yaml and fill in your cluster")
    with open(path) as f:
        return yaml.safe_load(f)


def job(name, sku, cmd, env, sla):
    return {"name": name, "sku": sku, "sla_tier": sla, "identity": "managed", "command": [cmd],
            "submit_args": {"env": dict(env)}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["train", "test", "paper"])
    ap.add_argument("experiment")
    ap.add_argument("--description", "-d", required=True, help="amlt prompts (and crashes in a non-console) without one")
    ap.add_argument("--tasks", nargs="+", default=["aldps", "aldpt"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--skus", nargs="+", default=SKUS)
    ap.add_argument("--prefix", default="")
    ap.add_argument("--group", default=None, help="W&B group (default: the experiment name)")
    ap.add_argument("--from_exp", default="bridge-v1", help="paper: the training experiment (its W&B group)")
    ap.add_argument("--sla", default="Basic", choices=["Basic", "Standard", "Premium"])
    ap.add_argument("--dry-run", action="store_true")
    a, rest = ap.parse_known_args()
    extra = " ".join(r for r in rest if r != "--")
    cfg = cluster_config()
    env = {"WANDB_API_KEY": wandb_key()}
    if cfg.get("uai"):
        env["_AZUREML_SINGULARITY_JOB_UAI"] = cfg["uai"]
    jobs = []
    if a.what == "train":
        for task in a.tasks:
            for k, s in enumerate(a.seeds):
                name = f"{a.prefix}{task}_s{s}"
                flags = f"--task {task} --seed {s} --wandb_group {a.group or a.experiment} {extra}"
                jobs.append(job(name, a.skus[k % len(a.skus)], f"bash cluster/run.sh {name} {flags}", env, a.sla))
    elif a.what == "paper":
        cmd = ("for i in 1 2 3; do python -c \"from digin_bridge import data; data.ensure('/tmp/feat_aldp')\" && break; "
               f"sleep 30; done && python -u -m digin_bridge.paper --runs_root {RUNS_ROOT} --data_root /tmp/feat_aldp "
               f"--out $$AMLT_OUTPUT_DIR/freeenergy --wandb_group {a.from_exp} {extra}")
        jobs.append(job("paper", a.skus[-1], cmd, env, a.sla))
    else:
        cmd = ("python -m pip install --no-cache-dir openmm pytest && "
               "FEAT_DATA=/tmp/feat_aldp python -m pytest tests -q -rs > $$AMLT_OUTPUT_DIR/pytest.txt 2>&1; "
               "s=$$?; cat $$AMLT_OUTPUT_DIR/pytest.txt; exit $$s")
        jobs.append(job("pytest", a.skus[-1], cmd, env, a.sla))
    doc = {"description": a.description, "target": cfg["target"],
           "environment": {"image": cfg["image"], "setup": SETUP},
           "storage": {"data": cfg["storage"]}, "code": {"local_dir": ROOT}, "jobs": jobs}
    import yaml
    text = yaml.safe_dump(doc, sort_keys=False, width=250)
    shown = text.replace(env["WANDB_API_KEY"], "***")
    print(shown)
    if a.dry_run:
        return
    fd, path = tempfile.mkstemp(suffix=".yaml", prefix="digin_bridge_amlt_")
    with os.fdopen(fd, "w") as f:
        f.write(text)
    try:
        cmd = ["amlt", "run", path, a.experiment, "-y", "-d", a.description]
        print("$", " ".join(cmd), flush=True)
        sys.exit(subprocess.call(cmd, cwd=ROOT))
    finally:
        os.remove(path)


if __name__ == "__main__":
    main()
