"""Unattended three-day BG-PDR-FM GPU queue.

The queue is intentionally conservative: it waits for already-running jobs,
runs the expected post-train evaluation on GPU, then launches one follow-up job
per GPU group. It does not kill active jobs and it records each step under
``logs/bg_pdr_fm/unattended_3day_queue``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from bg_pdr_fm.runtime import configure_checkpoint_worker_environment

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QUEUE_DIR = REPO_ROOT / "logs" / "bg_pdr_fm" / "unattended_3day_queue"
DEFAULT_PYTHON = "/public/home/xuyinghao/miniconda3/envs/seg/bin/python"

PIXEL_RUN = REPO_ROOT / "logs" / "bg_pdr_fm" / "residual_train_smooth_hc64_nogate_pixelfm_film_predbg_e100"
PIXEL_E100_CONFIG = REPO_ROOT / "bg_pdr_fm" / "configs" / "openfwi_lmdb_residual_smooth_hc64_nogate_pixelfm_film_predbg_e100.yaml"
PIXEL_E100_EVAL_MB10_CONFIG = (
    REPO_ROOT / "bg_pdr_fm" / "configs" / "openfwi_lmdb_residual_smooth_hc64_nogate_pixelfm_film_predbg_e100_eval_mb10.yaml"
)

AUTO_AE_RUN = REPO_ROOT / "logs" / "bg_pdr_fm" / "aaai27" / "formal" / "adapted_auto_linear_ae_pretrain"
AUTO_AE_CONFIG = REPO_ROOT / "bg_pdr_fm" / "configs" / "experiments" / "aaai27" / "adapted_auto_linear_ae_pretrain.yaml"
AUTO_INVERSE_CONFIG = REPO_ROOT / "bg_pdr_fm" / "configs" / "experiments" / "aaai27" / "adapted_auto_linear_multimodal.yaml"
AUTO_INVERSE_EVAL_CONFIG = (
    REPO_ROOT / "bg_pdr_fm" / "configs" / "experiments" / "aaai27" / "formal_adapted_auto_linear_multimodal.yaml"
)
AUTO_INVERSE_RUN = REPO_ROOT / "logs" / "bg_pdr_fm" / "aaai27" / "formal" / "adapted_auto_linear_multimodal"


@dataclass(frozen=True)
class QueueJob:
    name: str
    run_dir: Path
    train_pid: int | None
    kind: str
    train_config: Path
    eval_config: Path | None
    eval_gpu: str
    next_train_config: Path | None
    gpu_group: str = ""


def _log(message: str) -> None:
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)


def _resolve(path: Path | str) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def _pid_alive(pid: int | None) -> bool:
    return pid is not None and Path(f"/proc/{pid}").exists()


def _read_pid_file(path: Path) -> int | None:
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError as exc:
        raise ValueError(f"Invalid pid file {path}: {text!r}") from exc


def _read_existing_pid(run_dir: Path) -> int | None:
    for name in ("launcher.pid", "train.pid"):
        pid = _read_pid_file(run_dir / name)
        if pid is not None:
            return pid
    return None


def _run_has_checkpoint(run_dir: Path, kind: str) -> bool:
    if kind == "benchmark":
        return (run_dir / "checkpoints" / "last.ckpt").is_file()
    return (run_dir / "stage_checkpoints" / "residual_last.ckpt").is_file()


def _load_means(summary_path: Path) -> dict[str, float] | None:
    if not summary_path.is_file():
        return None
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    means = data.get("means")
    if not isinstance(means, dict):
        return None
    parsed: dict[str, float] = {}
    for key, value in means.items():
        try:
            parsed[key] = float(value)
        except (TypeError, ValueError):
            continue
    return parsed


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _append_event(queue_dir: Path, event: dict[str, Any]) -> None:
    queue_dir.mkdir(parents=True, exist_ok=True)
    event = {"time": datetime.now().isoformat(timespec="seconds"), **event}
    with (queue_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")


def _command_env(gpus: str) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["PYTHONUNBUFFERED"] = "1"
    if gpus:
        env["CUDA_VISIBLE_DEVICES"] = str(gpus)
        env["HIP_VISIBLE_DEVICES"] = str(gpus)
    return configure_checkpoint_worker_environment(env)


def _run_blocking(command: list[str], *, gpus: str, log_path: Path, queue_dir: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _append_event(queue_dir, {"event": "run_blocking_start", "gpus": gpus, "command": command, "log": str(log_path)})
    _log(f"running on CUDA_VISIBLE_DEVICES={gpus}: {' '.join(command)}")
    with log_path.open("ab", buffering=0) as log_file:
        result = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            env=_command_env(gpus),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=False,
        )
    _append_event(queue_dir, {"event": "run_blocking_done", "code": result.returncode, "log": str(log_path)})
    return int(result.returncode)


def _launch_detached(
    command: list[str],
    *,
    gpus: str,
    run_dir: Path,
    log_name: str,
    queue_dir: Path,
) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / log_name
    pid_path = run_dir / "launcher.pid"
    _append_event(queue_dir, {"event": "launch_start", "gpus": gpus, "command": command, "log": str(log_path)})
    _log(f"launching on CUDA_VISIBLE_DEVICES={gpus}: {' '.join(command)}")
    with log_path.open("ab", buffering=0) as log_file:
        process = subprocess.Popen(
            command,
            cwd=str(REPO_ROOT),
            env=_command_env(gpus),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    pid_path.write_text(f"{process.pid}\n", encoding="utf-8")
    launch = {"pid": process.pid, "pid_file": str(pid_path), "log": str(log_path), "command": command, "gpus": gpus}
    _append_event(queue_dir, {"event": "launch_done", **launch})
    return launch


def _wait_for_pid(pid: int | None, poll_seconds: int, queue_dir: Path, name: str) -> None:
    if pid is None:
        _append_event(queue_dir, {"event": "wait_skip_no_pid", "job": name})
        return
    while _pid_alive(pid):
        _log(f"waiting for {name} pid={pid}")
        time.sleep(max(1, poll_seconds))
    _append_event(queue_dir, {"event": "pid_finished", "job": name, "pid": pid})


def _evaluate_residual_mb10(eval_config: Path, eval_gpu: str, queue_dir: Path, python: str) -> dict[str, Any]:
    summary = PIXEL_RUN / "evaluation_mb10" / "summary.json"
    if not summary.is_file():
        code = _run_blocking(
            [python, "-m", "bg_pdr_fm.evaluation.evaluate_bg_pdr_fm", "--config", str(eval_config)],
            gpus=eval_gpu,
            log_path=queue_dir / "pixelfm_e100_mb10_eval.log",
            queue_dir=queue_dir,
        )
        if code != 0:
            raise RuntimeError(f"Pixel FM mb10 evaluation failed with exit code {code}.")
    means = _load_means(summary)
    if means is None:
        raise FileNotFoundError(f"Pixel FM mb10 evaluation summary missing or invalid: {summary}")
    return {"summary": str(summary), "means": means}


def _evaluate_benchmark(config: Path, eval_gpu: str, queue_dir: Path, log_name: str, python: str) -> dict[str, Any]:
    code = _run_blocking(
        [python, "-m", "bg_pdr_fm.evaluation.compare_experiments", "--config", str(config)],
        gpus=eval_gpu,
        log_path=queue_dir / log_name,
        queue_dir=queue_dir,
    )
    if code != 0:
        raise RuntimeError(f"Benchmark evaluation failed with exit code {code}: {config}")
    summary = REPO_ROOT / "logs" / "bg_pdr_fm" / "aaai27" / "eval_adapted_auto_linear_multimodal" / "adapted_auto_linear_multimodal" / "summary.json"
    means = _load_means(summary)
    if means is None:
        raise FileNotFoundError(f"Adapted Auto-Linear evaluation summary missing or invalid: {summary}")
    return {"summary": str(summary), "means": means}


def next_pixel_followup(run_dir: Path, *, ssim: float | None, mae_l: float | None) -> dict[str, Any]:
    return {
        "action": "stop_no_ae_route",
        "reason": "No-AE pixel-space FM is no longer the main route; keep autoencoder latent FM as the production direction.",
        "source_run": str(run_dir),
        "train_config": None,
        "output_dir": None,
        "ssim": ssim,
        "mae_l": mae_l,
    }


def next_auto_linear_followup(run_dir: Path) -> dict[str, Any]:
    return {
        "action": "launch_inverse_training",
        "reason": "AE pretrain finished; inverse-linear stage is the next adapted Auto-Linear step.",
        "source_run": str(run_dir),
        "train_config": AUTO_INVERSE_CONFIG,
        "output_dir": AUTO_INVERSE_RUN,
    }


def build_plan(jobs: list[QueueJob]) -> dict[str, Any]:
    plan_jobs: list[dict[str, Any]] = []
    for job in jobs:
        status = "running" if _pid_alive(job.train_pid) else "finished_or_unknown"
        plan_jobs.append(
            {
                "name": job.name,
                "kind": job.kind,
                "current": {
                    "run_dir": str(job.run_dir),
                    "train_pid": job.train_pid,
                    "status": status,
                    "gpu_group": job.gpu_group,
                    "train_config": str(job.train_config),
                },
                "next": {
                    "eval_config": job.eval_config,
                    "eval_gpu": job.eval_gpu,
                    "train_config": job.next_train_config,
                },
            }
        )
    return {"created_at": datetime.now().isoformat(timespec="seconds"), "jobs": plan_jobs}


def format_summary_markdown(state: dict[str, Any]) -> str:
    lines = ["# BG-PDR-FM Unattended Queue Summary", ""]
    for job in state.get("jobs", []):
        current = job.get("current", {})
        next_step = job.get("next", {})
        lines.extend(
            [
                f"## {job.get('name', 'unknown')}",
                f"- status: {current.get('status', 'unknown')}",
                f"- gpu_group: {current.get('gpu_group', '')}",
                f"- next_action: {next_step.get('action', next_step.get('eval_config', ''))}",
                "",
            ]
        )
        if "eval" in job:
            means = job["eval"].get("means", {})
            lines.append("- eval:")
            for key in ("ssim", "mae", "mae_l", "mae_h"):
                if key in means:
                    lines.append(f"  - {key}: {float(means[key]):.6f}")
            lines.append("")
    return "\n".join(lines)


def _write_summary(queue_dir: Path, state: dict[str, Any]) -> None:
    _write_json(queue_dir / "queue_state.json", state)
    (queue_dir / "summary.md").write_text(format_summary_markdown(state), encoding="utf-8")


def run_queue(
    *,
    queue_dir: Path,
    pixel_pid: int | None,
    auto_pid: int | None,
    python: str,
    pixel_gpus: str,
    auto_gpus: str,
    pixel_eval_gpu: str,
    auto_eval_gpu: str,
    poll_seconds: int,
    dry_run: bool,
    skip_wait: bool,
) -> dict[str, Any]:
    queue_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        QueueJob(
            name="pixelfm_e100",
            run_dir=PIXEL_RUN,
            train_pid=pixel_pid,
            kind="residual",
            train_config=PIXEL_E100_CONFIG,
            eval_config=PIXEL_E100_EVAL_MB10_CONFIG,
            eval_gpu=pixel_eval_gpu,
            next_train_config=None,
            gpu_group=pixel_gpus,
        ),
        QueueJob(
            name="adapted_auto_linear_ae_pretrain",
            run_dir=AUTO_AE_RUN,
            train_pid=auto_pid,
            kind="benchmark",
            train_config=AUTO_AE_CONFIG,
            eval_config=AUTO_INVERSE_EVAL_CONFIG,
            eval_gpu=auto_eval_gpu,
            next_train_config=AUTO_INVERSE_CONFIG,
            gpu_group=auto_gpus,
        ),
    ]
    state = build_plan(jobs)
    _write_summary(queue_dir, state)
    if dry_run:
        _append_event(queue_dir, {"event": "dry_run_complete"})
        return state

    # Pixel-space FM chain.
    if not skip_wait:
        _wait_for_pid(pixel_pid, poll_seconds, queue_dir, "pixelfm_e100")
    if not _run_has_checkpoint(PIXEL_RUN, "residual"):
        raise FileNotFoundError(f"Pixel FM residual checkpoint is missing: {PIXEL_RUN / 'stage_checkpoints' / 'residual_last.ckpt'}")
    pixel_eval = _evaluate_residual_mb10(PIXEL_E100_EVAL_MB10_CONFIG, pixel_eval_gpu, queue_dir, python)
    pixel_followup = next_pixel_followup(
        PIXEL_RUN,
        ssim=pixel_eval["means"].get("ssim"),
        mae_l=pixel_eval["means"].get("mae_l"),
    )
    state["jobs"][0]["eval"] = pixel_eval
    state["jobs"][0]["next"] = pixel_followup
    _write_summary(queue_dir, state)

    # Adapted Auto-Linear chain.
    if not skip_wait:
        _wait_for_pid(auto_pid, poll_seconds, queue_dir, "adapted_auto_linear_ae_pretrain")
    if not (AUTO_AE_RUN / "checkpoints" / "last.ckpt").is_file():
        raise FileNotFoundError(f"Adapted Auto-Linear AE checkpoint is missing: {AUTO_AE_RUN / 'checkpoints' / 'last.ckpt'}")
    auto_followup = next_auto_linear_followup(AUTO_AE_RUN)
    state["jobs"][1]["next"] = {**auto_followup, "train_config": str(auto_followup["train_config"]), "output_dir": str(auto_followup["output_dir"])}
    if not _pid_alive(_read_existing_pid(AUTO_INVERSE_RUN)) and not (AUTO_INVERSE_RUN / "checkpoints" / "last.ckpt").is_file():
        launch = _launch_detached(
            [python, "-m", "bg_pdr_fm.training.train_aaai27_benchmark", "--config", str(AUTO_INVERSE_CONFIG)],
            gpus=auto_gpus,
            run_dir=AUTO_INVERSE_RUN,
            log_name="train.log",
            queue_dir=queue_dir,
        )
        state["jobs"][1]["launch"] = launch
        if not skip_wait:
            _wait_for_pid(int(launch["pid"]), poll_seconds, queue_dir, "adapted_auto_linear_multimodal")
    if (AUTO_INVERSE_RUN / "checkpoints" / "last.ckpt").is_file():
        state["jobs"][1]["eval"] = _evaluate_benchmark(
            AUTO_INVERSE_EVAL_CONFIG,
            auto_eval_gpu,
            queue_dir,
            "adapted_auto_linear_formal_eval.log",
            python,
        )
    _write_summary(queue_dir, state)
    return state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-dir", type=Path, default=DEFAULT_QUEUE_DIR)
    parser.add_argument("--pixel-pid", type=int, default=None)
    parser.add_argument("--auto-pid", type=int, default=None)
    parser.add_argument("--pixel-gpus", default="7,0,1,2")
    parser.add_argument("--auto-gpus", default="3,4,5,6")
    parser.add_argument("--pixel-eval-gpu", default="7")
    parser.add_argument("--auto-eval-gpu", default="6")
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-wait", action="store_true")
    args = parser.parse_args(argv)

    pixel_pid = args.pixel_pid if args.pixel_pid is not None else _read_existing_pid(PIXEL_RUN)
    auto_pid = args.auto_pid if args.auto_pid is not None else _read_existing_pid(AUTO_AE_RUN)
    state = run_queue(
        queue_dir=_resolve(args.queue_dir),
        pixel_pid=pixel_pid,
        auto_pid=auto_pid,
        python=args.python,
        pixel_gpus=args.pixel_gpus,
        auto_gpus=args.auto_gpus,
        pixel_eval_gpu=args.pixel_eval_gpu,
        auto_eval_gpu=args.auto_eval_gpu,
        poll_seconds=args.poll_seconds,
        dry_run=args.dry_run,
        skip_wait=args.skip_wait,
    )
    print(format_summary_markdown(state))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
