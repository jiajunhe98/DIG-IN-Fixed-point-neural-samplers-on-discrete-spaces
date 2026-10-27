"""Submit to the Singularity cluster through amlt.

    # the 12 runs: {aldp, al4} x {internal, cartesian} x seeds {0, 1, 2}, one setting
    python cluster/submit.py train digin-v1 -d "DIG-IN ALDP/AL4 internal + Cartesian, 3 seeds"
    python cluster/submit.py train digin-v1 --systems al4 --coords cartesian --seeds 0 --dry-run

    # 2M samples with likelihoods and energies of finished runs -> /mnt/data/digin_peptides/samples/digin-v1/<run>
    python cluster/submit.py sample digin-v1-samples --from_exp digin-v1 --runs aldp_internal_s0 al4_internal_s0 \
        -d "2M samples for the paper evaluations"

    # tables and figure data of the paper (data/peptides/) from the 2M-sample sets -> $AMLT_OUTPUT_DIR/peptides
    python cluster/submit.py paper digin-paper -d "paper tables and figures"

    python cluster/submit.py test digin-test -d "unit tests + OpenMM comparison"
    python cluster/submit.py freeze digin-ref-freeze -d "fixed MD references for the final evaluation"
    python cluster/submit.py md al4-md -d "AL4 reference MD" --system al4 -- --n_temps 8 --copies 64 \
        --t_max 600 --burn_ps 500 --hours 12

Writes a full amlt config (target, image, blob mount, the W&B key from ~/_netrc
as a job environment variable) to a temporary file and calls `amlt run` on it,
so no secret sits in the repository.  The cluster (target, image, storage,
managed identity) is read from cluster/amlt.yaml, which is not in git:
copy cluster/amlt.example.yaml and fill it in.  Jobs go round-robin over the SKUs of
their configuration.  Output: $AMLT_OUTPUT_DIR/<job name>; W&B: project
digin-peptides, group = experiment.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SETUP = [
    "for i in 1 2 3; do python -m pip install --no-cache-dir wandb==0.28.2 matplotlib && break; sleep 20; done",
    "python -c \"import torch, wandb, matplotlib; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())\"",
]
# on the blob mounted at /mnt/data: the live MD outputs (digin.md run --publish), the fixed snapshots
# every final evaluation uses, and the sample sets; training outputs are amlt's job outputs
REF_LIVE = "/mnt/data/digin_aldp/{system}_ref.npz"
REF = "/mnt/data/digin_peptides/ref/{system}_ref.npz"
SAMPLES = "/mnt/data/digin_peptides/samples/{exp}/{run}"
RUNS_ROOT = "/mnt/output/projects/digin-peptides/amlt-results"
SKUS = {("al4", "cartesian"): ["141G1-H200", "80G1-H100"],
        ("aldp", "cartesian"): ["80G1-H100", "141G1-H200", "80G1-A100"],
        ("al4", "internal"): ["80G1-A100", "80G1-H100", "141G1-H200"],
        ("aldp", "internal"): ["80G1-A100", "40G1-A100", "80G1-H100"]}


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
    ap.add_argument("what", choices=["train", "sample", "paper", "test", "freeze", "md"])
    ap.add_argument("experiment")
    ap.add_argument("--description", "-d", required=True, help="amlt prompts (and crashes in a non-console) without one")
    ap.add_argument("--systems", nargs="+", default=["aldp", "al4"])
    ap.add_argument("--coords", nargs="+", default=["internal", "cartesian"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--system", default="al4", help="md: the system")
    ap.add_argument("--from_exp", default="digin-v1", help="sample: the training experiment")
    ap.add_argument("--runs", nargs="+", default=None, help="sample: job names of finished training runs")
    ap.add_argument("--n", type=int, default=2_000_000, help="sample: samples per run")
    ap.add_argument("--skus", nargs="+", default=None, help="override the per-configuration SKU lists")
    ap.add_argument("--prefix", default="")
    ap.add_argument("--group", default=None, help="W&B group (default: the experiment name)")
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
        for system in a.systems:
            for coords in a.coords:
                skus = a.skus or SKUS[(system, coords)]
                for k, s in enumerate(a.seeds):
                    name = f"{a.prefix}{system}_{coords}_s{s}"
                    flags = (f"--system {system} --coords {coords} --seed {s} --wandb_group {a.group or a.experiment} "
                             f"--ref {REF.format(system=system)} {extra}")
                    jobs.append(job(name, skus[k % len(skus)], f"bash cluster/run.sh {name} {flags}", env, a.sla))
    elif a.what == "sample":
        if not a.runs:
            raise SystemExit("sample: name the finished runs with --runs")
        for k, run in enumerate(a.runs):
            system, coords = run.split("_")[:2]
            skus = a.skus or SKUS[(system, coords)]
            cmd = (f"python -u -m digin.sample --run {run} --runs_root {RUNS_ROOT} --n {a.n} "
                   f"--out {SAMPLES.format(exp=a.from_exp, run=run)} {extra}")
            jobs.append(job(f"sample_{run}", skus[k % len(skus)], cmd, env, a.sla))
    elif a.what == "paper":
        cmd = (f"python -u -m digin.paper --samples {SAMPLES.format(exp=a.from_exp, run='')} "
               f"--ref_dir {os.path.dirname(REF)} --out $$AMLT_OUTPUT_DIR/peptides --wandb_group {a.from_exp} {extra}")
        jobs.append(job("paper", (a.skus or ["80G1-A100"])[0], cmd, env, a.sla))
    elif a.what == "test":
        cmd = ("python -m pip install --no-cache-dir openmm pytest && "
               "python -m pytest tests -q -rs > $$AMLT_OUTPUT_DIR/pytest.txt 2>&1; s=$$?; "
               "cat $$AMLT_OUTPUT_DIR/pytest.txt; exit $$s")
        jobs.append(job("pytest", (a.skus or ["80G1-A100"])[0], cmd, env, a.sla))
    elif a.what == "freeze":
        cmd = " && ".join(f"python -u -m digin.md freeze {REF_LIVE.format(system=s)} {REF.format(system=s)}"
                          for s in a.systems)
        jobs.append(job("freeze", (a.skus or ["80G1-A100"])[0], cmd, env, a.sla))
    else:
        cmd = (f"python -u -m digin.md run --system {a.system} --out $$AMLT_OUTPUT_DIR/md "
               f"--publish {REF_LIVE.format(system=a.system)} {extra}")
        jobs.append(job(f"{a.system}_md", (a.skus or ["40G1-A100"])[0], cmd, env, a.sla))
    doc = {"description": a.description, "target": cfg["target"],
           "environment": {"image": cfg["image"], "setup": SETUP},
           "storage": {"data": cfg["storage"]}, "code": {"local_dir": ROOT}, "jobs": jobs}
    import yaml
    text = yaml.safe_dump(doc, sort_keys=False, width=250)
    shown = text.replace(env["WANDB_API_KEY"], "***")
    print(shown)
    if a.dry_run:
        return
    fd, path = tempfile.mkstemp(suffix=".yaml", prefix="digin_amlt_")
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
