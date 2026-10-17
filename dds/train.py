"""Training entry point.

    python -m dds.train method=dsi_masked_ce target=ising25_critical
    python -m dds.train method=leaps target=potts16_ordered wandb.enabled=true

Every evaluation is appended to `history.jsonl` in the run directory
together with the training wall time, the evaluation wall time and the
number of target evaluations spent so far, so a run's whole training curve
-- every criterion against every budget -- survives in one file that a
preempted job extends rather than restarts.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import signal
import time

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from dds.eval_alloy import (compare_phase_diagram, evaluate_alloy_grid,
                            phase_diagram_from_rows)
from dds.evaluation import evaluate, get_reference_samples
from dds.methods import build_method
from dds.targets import build_target
from dds.targets.base import CALL_KINDS
from dds.utils import metrics
from dds.utils.checkpoint import load_checkpoint, rng_state, save_checkpoint, set_rng_state

HISTORY_FILE = "history.jsonl"


class _Preemption:
    """Cooperative SIGTERM handling.

    Cluster schedulers send SIGTERM before killing a preempted job.  Dying inside
    `torch.save` would leave a truncated checkpoint, so the handler only
    raises a flag; the training loop checkpoints at the next step boundary
    and exits.  Without this the job still resumes -- from the last periodic
    checkpoint -- it just throws away the work since then.
    """

    def __init__(self):
        self.requested = False
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, self._handle)
            except (ValueError, OSError):
                pass  # not on the main thread, or unsupported on this platform

    def _handle(self, signum, frame):
        print(f"[preempt] caught signal {signum}; "
              f"checkpointing at the next step boundary", flush=True)
        self.requested = True


def wandb_run_id(run_dir: str) -> str:
    """Stable W&B run id for a run directory.

    A preempted job must reattach to the same W&B run, or one training curve
    fragments into a new run per restart.  Deriving the id from the run
    directory makes it stable even when the job dies before writing its
    first checkpoint, and avoids `wandb.util.generate_id`, which does not
    exist across wandb versions -- it is absent in both the cluster image
    and wandb 0.28.
    """
    return hashlib.sha1(os.path.abspath(run_dir).encode()).hexdigest()[:16]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------
# Evaluation history
# ----------------------------------------------------------------------
def load_history(path: str, max_step: int | None = None) -> list[dict]:
    """Rows of the history file, deduplicated by step (the last write wins)
    and truncated to `max_step`.

    Truncation is what makes the file consistent with the checkpoint after
    a preemption: an evaluation can land between two checkpoints, so a row
    can exist for a step the resumed run has yet to reach.  The row is then
    dropped and rewritten when the run gets there again.
    """
    rows: dict[int, dict] = {}
    if not os.path.exists(path):
        return []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue  # a line cut short by a kill mid-write
            if max_step is not None and r["step"] > max_step:
                continue
            rows[int(r["step"])] = r
    return [rows[k] for k in sorted(rows)]


def write_history(path: str, rows: list[dict]):
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, path)


def append_history(path: str, row: dict):
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _numeric(d: dict, prefix: str = "") -> dict:
    return {f"{prefix}{k}": (float(v) if isinstance(v, (int, float, bool)) else v)
            for k, v in d.items() if isinstance(v, (int, float, bool, str))}


@hydra.main(config_path="../configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    set_seed(cfg.seed)
    device = cfg.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    target = build_target(OmegaConf.to_container(cfg.target, resolve=True))
    method_cfg = OmegaConf.to_container(cfg.method, resolve=True)
    method_name = method_cfg.pop("name")
    # LEAPS / FrOGS pin their own backbone; everything else takes the model group
    model_name = "builtin" if "model" in method_cfg else cfg.model.name
    if "model" not in method_cfg:
        method_cfg["model"] = OmegaConf.to_container(cfg.model, resolve=True)
    method = build_method(method_name, target, method_cfg, device=device)

    # An amortized family is not one distribution, so a single reference set
    # and a single exact log Z do not describe it: sampling spans the whole
    # box while those describe one point in it.  Such runs are scored by
    # evaluate_alloy_grid, state point by state point, instead.
    amortized = bool(getattr(target, "amortized", False))
    exact_log_Z = None if amortized else target.exact_log_Z()
    x_ref, ref_summary = None, None
    if cfg.reference.n_samples > 0 and not amortized:
        x_ref = get_reference_samples(
            target,
            n=cfg.reference.n_samples,
            cache_dir=to_absolute_path(cfg.reference.cache_dir),
            method=cfg.reference.method,
            burn_in=cfg.reference.burn_in,
            thin=cfg.reference.thin,
            device=device,
        ).to(device)
        # the reference's observables are the denominators of every
        # comparison and do not change: computed once, not once per eval
        ref_summary = metrics.reference_summary(target, x_ref)

    def run_eval(n_samples: int, final: bool = False) -> dict:
        """Whichever evaluation actually describes this target.

        For an amortized family `n_samples` is the budget *per state point*,
        not in total: the grid is scored point by point.  Intermediate
        evaluations use the smaller `eval_samples` budget and only the final
        one pays for `grid.n_per_point`, since a mid-training curve needs to
        show a trend while the final number is the reported comparison.
        """
        if not amortized:
            return evaluate(method, n_samples, x_ref, exact_log_Z, ref=ref_summary,
                            keep_samples=bool(final and cfg.get("save_final_samples",
                                                                True)))
        if getattr(target, "grid_kind", "ising") == "alloy":
            # the alloy families carry packaged references and an exact log Z
            # on the whole plane, so they get their own evaluator
            summary, rows = evaluate_alloy_grid(
                method,
                n_per_point=cfg.grid.n_per_point if final else n_samples,
                n_T=cfg.grid.n_T,
                n_dmu=cfg.grid.n_dmu,
                T_range=(tuple(cfg.grid.T_range)
                         if cfg.grid.get("T_range") else None),
                dmu_range=(tuple(cfg.grid.dmu_range)
                           if cfg.grid.get("dmu_range") else None),
                use_reference_grid=bool(final and cfg.grid.get("final_reference_grid")),
                save_path=("grid_samples.npz"
                           if final and cfg.grid.get("save_samples") else None),
            )
            pd = phase_diagram_from_rows(rows)
            summary.update(compare_phase_diagram(pd, target))
            run_eval.rows = rows
            run_eval.phase_diagram = pd
            return summary
        raise ValueError(f"no grid evaluation for {type(target).__name__}")
    run_eval.rows = []
    run_eval.phase_diagram = []

    # ------------------------------------------------------------------
    # Resume.  The run directory is the unit of identity, so a preempted job
    # must be relaunched with the SAME hydra.run.dir -- the default timestamp
    # directory would silently start a fresh run instead of continuing.
    # ------------------------------------------------------------------
    preempt = _Preemption()
    ckpt_path = os.path.abspath("checkpoint.pt")
    history_path = os.path.abspath(HISTORY_FILE)
    start_step, wandb_id = 0, None
    # cumulative cost accounting, carried across preemptions in the checkpoint
    timing = {"train_s": 0.0, "eval_s": 0.0, "n_evals": 0}
    calls_eval = {k: 0 for k in CALL_KINDS}  # target calls spent on evaluation
    state, which = load_checkpoint(ckpt_path, map_location=device)
    if state is not None:
        method.load_state_dict(state["method"])
        start_step = int(state["step"])
        wandb_id = state.get("wandb_id")
        if state.get("rng") is not None:
            set_rng_state(state["rng"])
        timing.update(state.get("timing", {}))
        calls_eval.update(state.get("calls_eval", {}))
        target.n_calls.update(state.get("target_calls", {}))
        print(f"[resume] {which}: continuing from step {start_step}/{cfg.n_steps}")
        if start_step >= cfg.n_steps:
            print("[resume] training already complete; going straight to eval")
    # bring the history in line with the checkpoint (see load_history)
    history = load_history(history_path, max_step=start_step)
    write_history(history_path, history)

    run = None
    if cfg.wandb.enabled:
        import wandb

        # Offline unless a key is actually present.  Recent wandb clients
        # demand an API key inside `init` even when WANDB_MODE says offline,
        # and env_defaults did not always reach the process, so the mode is
        # passed explicitly rather than inferred from the environment.
        #
        # Ask the client whether it can resolve a key rather than looking only
        # at WANDB_API_KEY: a developer machine usually keeps it in ~/.netrc,
        # where the environment check finds nothing and every local run would
        # silently log offline.  Cluster images have no netrc, so they still
        # go offline unless the variable is set, which is what we want.
        def _resolve_mode() -> str:
            if cfg.wandb.get("mode"):
                return cfg.wandb.mode
            if os.environ.get("WANDB_MODE"):
                return os.environ["WANDB_MODE"]
            try:
                return "online" if wandb.api.api_key else "offline"
            except Exception:  # noqa: BLE001 - never let this decide the run
                return "offline"

        mode = _resolve_mode()
        # resume="allow" reattaches to the same W&B run after a preemption, so
        # one training curve stays one curve instead of fragmenting into a new
        # run per restart
        wandb_id = wandb_id or wandb_run_id(os.getcwd())
        try:
            run = wandb.init(
                project=cfg.wandb.project,
                entity=cfg.wandb.entity,
                name=cfg.wandb.name or f"{method_name}_{cfg.target.name}",
                tags=list(cfg.wandb.tags),
                config=OmegaConf.to_container(cfg, resolve=True),
                id=wandb_id,
                resume="allow",
                mode=mode,
            )
        except Exception as e:  # noqa: BLE001 - reporting is never fatal
            # Logging is reporting.  A run that trains fine must not die
            # because a reporting backend refused to start: without this
            # guard a missing key costs the whole job, three retries deep.
            print(f"[wandb] init failed ({e!r}); continuing without it",
                  flush=True)
            run = None

    n_params = sum(
        p.numel() for attr in ("net", "controller", "f_net")
        if hasattr(method, attr)
        for p in getattr(method, attr).parameters()
    )
    print(OmegaConf.to_yaml(cfg))
    print(f"target: {type(target).__name__} D={target.D} N={target.N} "
          f"beta={target.beta}  exact log Z: {exact_log_Z}")
    print(f"method: {method_name} ({n_params} params) on {device}")

    def sync():
        if device != "cpu":
            torch.cuda.synchronize()

    # Wall-clock is attributed by interval: everything between the end of
    # one evaluation and the start of the next is training (including the
    # checkpoint writes, which are negligible), the evaluation itself is
    # evaluation.  `mark` is the start of the current interval.
    mark = time.time()

    def fold_train():
        nonlocal mark
        sync()
        now = time.time()
        timing["train_s"] += now - mark
        mark = now

    def checkpoint(step: int):
        fold_train()
        save_checkpoint(ckpt_path, {
            "step": step,
            "method": method.state_dict(),
            "rng": rng_state(),
            "wandb_id": wandb_id,
            "timing": dict(timing),
            "calls_eval": dict(calls_eval),
            "target_calls": dict(target.n_calls),
            "n_params": int(n_params),
        })
        # The acquisition map is 256 numbers a side; writing it beside the
        # checkpoint keeps it in the mirrored run directory, where the report
        # can draw it, instead of only inside a 63 MB checkpoint.
        acq = getattr(method, "acq", None)
        if acq is not None:
            try:
                with open("acquisition.json", "w", encoding="utf-8") as f:
                    json.dump({"step": step, **acq.snapshot()}, f)
            except Exception as e:  # noqa: BLE001 - diagnostics are never fatal
                print(f"[acq] snapshot failed, continuing: {e!r}", flush=True)


    def cost_row() -> dict:
        train_calls = {k: target.n_calls[k] - calls_eval[k] for k in CALL_KINDS}
        return {
            "wall_train_s": timing["train_s"],
            "wall_eval_s": timing["eval_s"],
            "wall_total_s": timing["train_s"] + timing["eval_s"],
            "n_evals": timing["n_evals"],
            **{f"target_calls_{k}": target.n_calls[k] for k in CALL_KINDS},
            **{f"target_calls_train_{k}": train_calls[k] for k in CALL_KINDS},
            "target_calls_train_total": sum(train_calls.values()),
        }

    def do_eval(step: int, train_metrics: dict, n_samples: int, final: bool = False) -> dict:
        """Evaluate, then append the row (cost, train metrics, criteria)."""
        nonlocal mark
        fold_train()
        before = dict(target.n_calls)
        ev = run_eval(n_samples, final=final)
        sync()
        now = time.time()
        timing["eval_s"] += now - mark
        timing["n_evals"] += 1
        mark = now
        for k in CALL_KINDS:
            calls_eval[k] += target.n_calls[k] - before[k]
        samples = ev.pop("_samples", None)
        row = {"step": int(step), "final": bool(final), **cost_row(),
               **_numeric(train_metrics, "train/"),
               **{k.removeprefix("eval/"): v for k, v in ev.items()}}
        if samples is not None:
            # the draw the final row describes, kept so the distributional
            # plots (energy histogram before/after reweighting, g(r), the
            # composition histogram) can be rebuilt without shipping a
            # checkpoint -- and without a second round of energy calls, which
            # would make the reported cost disagree with the history
            xs, lw = samples
            store = xs.to(torch.int16) if int(target.N) <= 32767 else xs
            torch.save({"x": store, "log_w": lw, "N": int(target.N),
                        "D": int(target.D)}, "samples.pt")
        history.append(row)
        append_history(history_path, row)
        print(f"  [eval {step}] " + " ".join(
            f"{k.removeprefix('eval/')}={v:.4g}" for k, v in ev.items()
            if isinstance(v, float)) + f"  (train {timing['train_s']:.0f}s, "
            f"eval {timing['eval_s']:.0f}s, calls {cost_row()['target_calls_train_total']})",
            flush=True)
        if run is not None:
            run.log({**ev, **{f"cost/{k}": v for k, v in cost_row().items()}}, step=step)
        return ev

    train_metrics: dict = {}
    if state is None:
        # setup (exact log Z by enumeration, scoring the reference) is not
        # the method's cost: the meter starts at zero with training
        for k in CALL_KINDS:
            target.n_calls[k] = 0
        mark = time.time()
    if (start_step == 0 and cfg.get("eval_at_start", False) and cfg.eval_every > 0
            and not any(r["step"] == 0 for r in history)):
        do_eval(0, {}, cfg.eval_samples)
        checkpoint(0)

    t0 = time.time()
    for step in range(start_step + 1, cfg.n_steps + 1):
        train_metrics = method.step()
        if step % cfg.log_every == 0:
            msg = " ".join(
                f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                for k, v in train_metrics.items()
            )
            print(f"[{step}/{cfg.n_steps}] {msg} ({(time.time()-t0):.0f}s)", flush=True)
            if run is not None:
                run.log({f"train/{k}": v for k, v in train_metrics.items()
                         if isinstance(v, (int, float))}, step=step)
        evaluated = False
        if cfg.eval_every > 0 and step % cfg.eval_every == 0 and step < cfg.n_steps:
            do_eval(step, train_metrics, cfg.eval_samples)
            evaluated = True
        if evaluated or (cfg.checkpoint_every > 0 and step % cfg.checkpoint_every == 0):
            # always checkpoint right after an evaluation, so a kill can never
            # separate a history row from the state that produced it
            checkpoint(step)
        if preempt.requested:
            # Save and exit non-zero so the scheduler retries the job; the
            # relaunch finds this checkpoint and continues from `step`.
            checkpoint(step)
            print(f"[preempt] checkpointed at step {step}; exiting for requeue",
                  flush=True)
            if run is not None:
                run.finish(exit_code=1)
            raise SystemExit(17)

    # Checkpoint before the final evaluation, not after: that evaluation can
    # take hours at large D, and a wall-clock kill during it would otherwise
    # discard the entire training run.
    checkpoint(cfg.n_steps)
    # final evaluation with a larger sample budget
    ev = do_eval(cfg.n_steps, train_metrics, cfg.final_eval_samples, final=True)
    print("FINAL " + " ".join(f"{k.removeprefix('eval/')}={v:.5g}" for k, v in ev.items()
                              if isinstance(v, float)))
    if run_eval.rows:
        # the per-state-point numbers are the actual comparison against the
        # published figure, so they are written out whether or not W&B is on
        with open("grid.json", "w") as f:
            json.dump(run_eval.rows, f, indent=2)
    if getattr(run_eval, "phase_diagram", None):
        with open("phase_diagram.json", "w") as f:
            json.dump(run_eval.phase_diagram, f, indent=2)
    checkpoint(cfg.n_steps)

    def log_final(run):
        """W&B reporting for a finished run.

        Guarded, and deliberately *after* the artifacts are on disk: this is
        reporting, and reporting must never destroy a completed run.  A
        wandb.Table stages media through a temp directory and then moves it
        into the run directory, which is exactly the kind of thing that fails
        for environment reasons -- it failed here on a path-form mismatch --
        and without the guard the job dies holding a fully trained model,
        gets retried, redoes the whole final evaluation, and is eventually
        marked failed with the work stranded in the checkpoint.
        """
        if run is None:
            return
        try:
            run.log({k.replace("eval/", "final/"): v for k, v in ev.items()})
            if run_eval.rows:
                import wandb

                cols = sorted({k for r in run_eval.rows for k in r})
                run.log({"grid/table": wandb.Table(
                    columns=cols,
                    data=[[r.get(c) for c in cols] for r in run_eval.rows],
                )})
        except Exception as e:  # noqa: BLE001 - reporting is never fatal
            print(f"[wandb] final logging failed, continuing: {e!r}")

    log_final(run)
    # machine-readable summary for sweep aggregation
    with open("result.json", "w") as f:
        json.dump({
            "method": method_name,
            "model": model_name,
            "target": cfg.target.name,
            # config-group name, which distinguishes targets sharing a family
            # name at different temperatures (ising25_ordered / _critical / ...)
            "target_cfg": HydraConfig.get().runtime.choices.get("target", cfg.target.name),
            # likewise for methods sharing a class: DIG-IN, MDNS and PDNS all
            # report method="dsi" and are only told apart by their overrides
            "method_cfg": HydraConfig.get().runtime.choices.get("method", method_name),
            # the alloy families read their cell from the packaged cluster
            # tables rather than from the config, so take it off the target
            "lattice_shape": list(getattr(target, "lattice_shape",
                                          cfg.target.get("lattice_shape", []))),
            # an amortized family has no single beta: it reports the box it
            # covers instead, so the two kinds of run stay distinguishable in
            # an aggregated sweep rather than one of them crashing here
            "beta": float(cfg.target.beta) if "beta" in cfg.target else None,
            "T_range": (list(cfg.target.T_range)
                        if "T_range" in cfg.target else None),
            "dmu_range": (list(cfg.target.dmu_range)
                          if "dmu_range" in cfg.target else None),
            "seed": int(cfg.seed),
            "n_steps": int(cfg.n_steps),
            "n_params": int(n_params),
            # `wall_time_s` is the training time of this run across all of
            # its attempts; the cost breakdown carries the rest
            "wall_time_s": timing["train_s"],
            **cost_row(),
            "exact_log_Z": exact_log_Z,
            **{k.removeprefix("eval/"): v for k, v in ev.items()},
        }, f, indent=2)
    if run is not None:
        try:
            run.finish()
        except Exception as e:  # noqa: BLE001 - the run is already complete
            print(f"[wandb] finish failed, continuing: {e!r}")
    return ev


if __name__ == "__main__":
    main()
