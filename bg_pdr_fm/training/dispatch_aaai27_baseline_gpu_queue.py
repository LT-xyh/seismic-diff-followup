"""Dispatch AAAI27 external/adapted baselines onto idle single GPUs.

This queue is intentionally narrow: it launches each benchmark training config
on one idle GPU, skips runs that already have ``checkpoints/last.ckpt``, and
sends one completion mail after every queued job has either finished or failed.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from bg_pdr_fm.runtime import configure_checkpoint_worker_environment
from bg_pdr_fm.training.benchmark_config import load_benchmark_config
from bg_pdr_fm.training.monitor_run_and_mail import _send_mail, _read_smtp_config


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_QUEUE_DIR = REPO_ROOT / "logs" / "bg_pdr_fm" / "aaai27" / "baseline_gpu_queue"
DEFAULT_PYTHON = "/public/home/xuyinghao/miniconda3/envs/seg/bin/python"
ROCM_SMI = "/opt/dtk-25.04.2/bin/rocm-smi"


@dataclass(frozen=True)
class QueueJob:
    name: str
    config: Path
    run_dir: Path
    depends_on: str | None = None
    eval_config: Path | None = None
    eval_dir: Path | None = None


@dataclass
class RunningJob:
    job: QueueJob
    gpu: int
    pid: int
    log_path: Path
    start_time: str
    process: subprocess.Popen[Any] | None = None


JOBS = (
    QueueJob(
        name="adapted_inversion_net",
        config=REPO_ROOT / "bg_pdr_fm" / "configs" / "experiments" / "aaai27" / "adapted_inversion_net.yaml",
        run_dir=REPO_ROOT / "logs" / "bg_pdr_fm" / "aaai27" / "formal" / "adapted_inversion_net_official_e100",
        eval_config=REPO_ROOT / "bg_pdr_fm" / "configs" / "experiments" / "aaai27" / "formal_adapted_inversion_net.yaml",
        eval_dir=REPO_ROOT / "logs" / "bg_pdr_fm" / "aaai27" / "eval_adapted_inversion_net_official_e100" / "adapted_inversion_net_official_e100",
    ),
    QueueJob(
        name="adapted_upfwi",
        config=REPO_ROOT / "bg_pdr_fm" / "configs" / "experiments" / "aaai27" / "adapted_upfwi.yaml",
        run_dir=REPO_ROOT / "logs" / "bg_pdr_fm" / "aaai27" / "formal" / "adapted_upfwi_official_e100",
        eval_config=REPO_ROOT / "bg_pdr_fm" / "configs" / "experiments" / "aaai27" / "formal_adapted_upfwi.yaml",
        eval_dir=REPO_ROOT / "logs" / "bg_pdr_fm" / "aaai27" / "eval_adapted_upfwi_official_e100" / "adapted_upfwi_official_e100",
    ),
    QueueJob(
        name="sv_inv_net",
        config=REPO_ROOT / "bg_pdr_fm" / "configs" / "experiments" / "aaai27" / "sv_inv_net.yaml",
        run_dir=REPO_ROOT / "logs" / "bg_pdr_fm" / "aaai27" / "formal" / "sv_inv_net",
    ),
    QueueJob(
        name="velocity_gan",
        config=REPO_ROOT / "bg_pdr_fm" / "configs" / "experiments" / "aaai27" / "velocity_gan.yaml",
        run_dir=REPO_ROOT / "logs" / "bg_pdr_fm" / "aaai27" / "formal" / "velocity_gan_official_e100",
        eval_config=REPO_ROOT / "bg_pdr_fm" / "configs" / "experiments" / "aaai27" / "formal_velocity_gan.yaml",
        eval_dir=REPO_ROOT / "logs" / "bg_pdr_fm" / "aaai27" / "eval_velocity_gan_official_e100" / "velocity_gan_official_e100",
    ),
    QueueJob(
        name="conditional_ddpm",
        config=REPO_ROOT / "bg_pdr_fm" / "configs" / "experiments" / "aaai27" / "conditional_ddpm.yaml",
        run_dir=REPO_ROOT / "logs" / "bg_pdr_fm" / "aaai27" / "formal" / "conditional_ddpm",
    ),
    QueueJob(
        name="adapted_gfi",
        config=REPO_ROOT / "bg_pdr_fm" / "configs" / "experiments" / "aaai27" / "formal_adapted_gfi_multimodal.yaml",
        run_dir=REPO_ROOT / "logs" / "bg_pdr_fm" / "aaai27" / "formal" / "adapted_gfi_multimodal",
    ),
    QueueJob(
        name="adapted_auto_linear_ae_pretrain",
        config=REPO_ROOT / "bg_pdr_fm" / "configs" / "experiments" / "aaai27" / "adapted_auto_linear_ae_pretrain.yaml",
        run_dir=REPO_ROOT / "logs" / "bg_pdr_fm" / "aaai27" / "formal" / "adapted_auto_linear_ae_pretrain",
    ),
    QueueJob(
        name="adapted_auto_linear_multimodal",
        config=REPO_ROOT / "bg_pdr_fm" / "configs" / "experiments" / "aaai27" / "adapted_auto_linear_multimodal.yaml",
        run_dir=REPO_ROOT / "logs" / "bg_pdr_fm" / "aaai27" / "formal" / "adapted_auto_linear_multimodal",
        depends_on="adapted_auto_linear_ae_pretrain",
    ),
    QueueJob(
        name="adapted_auto_linear_mae_ae_pretrain",
        config=REPO_ROOT / "bg_pdr_fm" / "configs" / "experiments" / "aaai27" / "adapted_auto_linear_mae_ae_pretrain.yaml",
        run_dir=REPO_ROOT / "logs" / "bg_pdr_fm" / "aaai27" / "formal" / "adapted_auto_linear_mae_ae_pretrain",
    ),
    QueueJob(
        name="adapted_auto_linear_mae_multimodal",
        config=REPO_ROOT / "bg_pdr_fm" / "configs" / "experiments" / "aaai27" / "adapted_auto_linear_mae_multimodal.yaml",
        run_dir=REPO_ROOT / "logs" / "bg_pdr_fm" / "aaai27" / "formal" / "adapted_auto_linear_mae_multimodal",
        depends_on="adapted_auto_linear_mae_ae_pretrain",
    ),
    QueueJob(
        name="adapted_auto_linear_original_ae_pretrain",
        config=REPO_ROOT / "bg_pdr_fm" / "configs" / "experiments" / "aaai27" / "adapted_auto_linear_original_ae_pretrain.yaml",
        run_dir=REPO_ROOT / "logs" / "bg_pdr_fm" / "aaai27" / "formal" / "adapted_auto_linear_original_ae_pretrain",
    ),
    QueueJob(
        name="adapted_auto_linear_original_multimodal",
        config=REPO_ROOT / "bg_pdr_fm" / "configs" / "experiments" / "aaai27" / "adapted_auto_linear_original_multimodal.yaml",
        run_dir=REPO_ROOT / "logs" / "bg_pdr_fm" / "aaai27" / "formal" / "adapted_auto_linear_original_multimodal",
        depends_on="adapted_auto_linear_original_ae_pretrain",
    ),
)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _log(message: str) -> None:
    print(f"[{_now()}] {message}", flush=True)


def _pid_alive(pid: int | None) -> bool:
    if pid is None:
        return False
    status = Path(f"/proc/{pid}/status")
    if not status.is_file():
        return False
    try:
        for line in status.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("State:"):
                return "\tZ " not in line and " Z " not in line
    except OSError:
        return False
    return True


def _read_pid(path: Path) -> int | None:
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError:
        return None


def _run_finished(job: QueueJob) -> bool:
    checkpoint_exists = (job.run_dir / "checkpoints" / "last.ckpt").is_file()
    if not checkpoint_exists:
        return False
    if job.eval_dir is None:
        return True
    return (job.eval_dir / "summary.json").is_file()


def _existing_live_pid(job: QueueJob) -> int | None:
    for name in ("launcher.pid", "train.pid"):
        pid = _read_pid(job.run_dir / name)
        if _pid_alive(pid):
            return pid
    return None


def _parse_rocm_smi() -> dict[int, dict[str, float]]:
    result = subprocess.run(
        [ROCM_SMI, "--showmeminfo", "vram", "--showuse"],
        cwd=str(REPO_ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"rocm-smi failed with code {result.returncode}: {result.stdout.strip()}")
    stats: dict[int, dict[str, float]] = {}
    for line in result.stdout.splitlines():
        use_match = re.search(r"HCU\[(\d+)\]\s*: HCU use \(%\):\s*([0-9.]+)", line)
        if use_match:
            gpu = int(use_match.group(1))
            stats.setdefault(gpu, {})["util"] = float(use_match.group(2))
            continue
        total_match = re.search(r"HCU\[(\d+)\]\s*: vram Total Memory \(MiB\):\s*([0-9.]+)", line)
        if total_match:
            gpu = int(total_match.group(1))
            stats.setdefault(gpu, {})["total_mib"] = float(total_match.group(2))
            continue
        used_match = re.search(r"HCU\[(\d+)\]\s*: vram Total Used Memory \(MiB\):\s*([0-9.]+)", line)
        if used_match:
            gpu = int(used_match.group(1))
            used_mib = float(used_match.group(2))
            stats.setdefault(gpu, {})["used_mib"] = used_mib
            stats[gpu]["mem_mib"] = used_mib
    return stats


def required_free_mib(peak_allocated_mib: float) -> int:
    return int(math.ceil(float(peak_allocated_mib) * 1.25 + 2048.0))


def _gpus_with_free_vram(
    stats: dict[int, dict[str, float]],
    *,
    allowed: list[int],
    reserved: set[int],
    required_free_mib: float,
) -> list[int]:
    eligible: list[tuple[float, int]] = []
    for gpu in allowed:
        if gpu in reserved:
            continue
        values = stats.get(gpu, {})
        total = float(values.get("total_mib", 0.0))
        used = float(values.get("used_mib", values.get("mem_mib", total)))
        free = total - used
        if free >= float(required_free_mib):
            eligible.append((free, gpu))
    eligible.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [gpu for _, gpu in eligible]


def _idle_gpus(
    allowed: list[int],
    running: dict[str, RunningJob],
    *,
    max_util: float,
    max_mem_mib: float,
) -> list[int]:
    used_by_queue = {item.gpu for item in running.values() if _pid_alive(item.pid)}
    stats = _parse_rocm_smi()
    idle = []
    for gpu in sorted(allowed, reverse=True):
        if gpu in used_by_queue:
            continue
        values = stats.get(gpu, {})
        util = float(values.get("util", 100.0))
        mem_mib = float(values.get("mem_mib", 1_000_000.0))
        if util <= max_util and mem_mib <= max_mem_mib:
            idle.append(gpu)
    return idle


def _command_env(gpu: int) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["HIP_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["PYTHONUNBUFFERED"] = "1"
    return configure_checkpoint_worker_environment(env)


def _validate_config(job: QueueJob) -> dict[str, Any]:
    if not job.config.is_file():
        raise FileNotFoundError(f"Missing config: {job.config}")
    conf = load_benchmark_config(job.config)
    devices = int(OmegaConf.select(conf, "training.devices", default=1))
    batch_size = int(OmegaConf.select(conf, "training.batch_size", default=0))
    variant = str(OmegaConf.select(conf, "benchmark.variant", default=""))
    if devices != 1:
        raise ValueError(f"{job.name} must use training.devices: 1 for this queue, got {devices}")
    if batch_size <= 0:
        raise ValueError(f"{job.name} has invalid training.batch_size: {batch_size}")
    return {"variant": variant, "batch_size": batch_size, "devices": devices}


def _launch(job: QueueJob, gpu: int, python: str, queue_dir: Path) -> RunningJob:
    job.run_dir.mkdir(parents=True, exist_ok=True)
    log_path = job.run_dir / "train.log"
    pid_path = job.run_dir / "launcher.pid"
    if job.eval_config is None:
        command = [python, "-m", "bg_pdr_fm.training.train_aaai27_benchmark", "--config", str(job.config)]
    else:
        command = [
            python,
            "-m",
            "bg_pdr_fm.training.run_aaai27_official_pipeline",
            "--train-config",
            str(job.config),
            "--eval-config",
            str(job.eval_config),
            "--python",
            python,
        ]
    with log_path.open("ab", buffering=0) as handle:
        process = subprocess.Popen(
            command,
            cwd=str(REPO_ROOT),
            env=_command_env(gpu),
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    pid_path.write_text(f"{process.pid}\n", encoding="utf-8")
    running = RunningJob(job=job, gpu=gpu, pid=process.pid, log_path=log_path, start_time=_now())
    running.process = process
    _append_event(
        queue_dir,
        {
            "event": "launch",
            "job": job.name,
            "gpu": gpu,
            "pid": process.pid,
            "config": str(job.config),
            "run_dir": str(job.run_dir),
            "log": str(log_path),
        },
    )
    _log(f"launched {job.name} pid={process.pid} gpu={gpu}")
    return running


def _append_event(queue_dir: Path, event: dict[str, Any]) -> None:
    queue_dir.mkdir(parents=True, exist_ok=True)
    with (queue_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time": _now(), **event}, sort_keys=True) + "\n")


def _write_state(queue_dir: Path, state: dict[str, Any]) -> None:
    queue_dir.mkdir(parents=True, exist_ok=True)
    (queue_dir / "queue_state.json").write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _build_state(
    *,
    pending: list[str],
    running: dict[str, RunningJob],
    completed: dict[str, dict[str, Any]],
    failed: dict[str, dict[str, Any]],
    skipped: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "updated_at": _now(),
        "pending": pending,
        "running": {
            name: {
                "gpu": item.gpu,
                "pid": item.pid,
                "alive": _pid_alive(item.pid),
                "returncode": item.process.poll() if item.process is not None else None,
                "run_dir": str(item.job.run_dir),
                "log": str(item.log_path),
                "start_time": item.start_time,
            }
            for name, item in sorted(running.items())
        },
        "completed": completed,
        "failed": failed,
        "skipped": skipped,
    }


def _ready(job: QueueJob, completed: dict[str, Any], skipped: dict[str, Any], failed: dict[str, Any]) -> bool:
    if job.depends_on is None:
        return True
    return job.depends_on in completed or job.depends_on in skipped


def _send_completion_mail(mail_to: str, smtp_config: Path | None, state: dict[str, Any]) -> int:
    if not mail_to:
        return 0
    lines = [
        "AAAI27 baseline GPU queue finished.",
        f"time: {_now()}",
        "",
        f"completed: {', '.join(sorted(state.get('completed', {}))) or 'none'}",
        f"skipped: {', '.join(sorted(state.get('skipped', {}))) or 'none'}",
        f"failed: {', '.join(sorted(state.get('failed', {}))) or 'none'}",
        "",
        f"queue_state: {DEFAULT_QUEUE_DIR / 'queue_state.json'}",
    ]
    smtp_conf = _read_smtp_config(smtp_config)
    return _send_mail(mail_to, "[AAAI27] baseline GPU queue finished", "\n".join(lines) + "\n", smtp_conf)


def run_queue(
    *,
    queue_dir: Path,
    allowed_gpus: list[int],
    python: str,
    poll_seconds: int,
    max_util: float,
    max_mem_mib: float,
    max_concurrent: int,
    mail_to: str,
    smtp_config: Path | None,
    dry_run: bool,
    selected_jobs: set[str] | None = None,
    memory_requirements: dict[str, float] | None = None,
) -> dict[str, Any]:
    queue_dir.mkdir(parents=True, exist_ok=True)
    jobs = tuple(job for job in JOBS if selected_jobs is None or job.name in selected_jobs)
    if selected_jobs is not None:
        known = {job.name for job in JOBS}
        unknown = sorted(selected_jobs - known)
        if unknown:
            raise ValueError(f"Unknown queue job(s): {', '.join(unknown)}")
    validated = {job.name: _validate_config(job) for job in jobs}
    requirements = {str(name): float(value) for name, value in (memory_requirements or {}).items()}
    pending_jobs = list(jobs)
    running: dict[str, RunningJob] = {}
    completed: dict[str, dict[str, Any]] = {}
    failed: dict[str, dict[str, Any]] = {}
    skipped: dict[str, dict[str, Any]] = {}

    for job in list(pending_jobs):
        live_pid = _existing_live_pid(job)
        if live_pid is not None:
            running[job.name] = RunningJob(
                job=job,
                gpu=-1,
                pid=live_pid,
                log_path=job.run_dir / "train.log",
                start_time="preexisting",
            )
            pending_jobs.remove(job)
        elif _run_finished(job):
            skipped[job.name] = {"reason": "checkpoint_exists", "run_dir": str(job.run_dir), **validated[job.name]}
            pending_jobs.remove(job)

    state = _build_state(
        pending=[job.name for job in pending_jobs],
        running=running,
        completed=completed,
        failed=failed,
        skipped=skipped,
    )
    state["validated"] = validated
    state["allowed_gpus"] = allowed_gpus
    state["memory_requirements_mib"] = requirements
    _write_state(queue_dir, state)
    if dry_run:
        _append_event(queue_dir, {"event": "dry_run", "state": state})
        print(json.dumps(state, indent=2, sort_keys=True))
        return state

    while pending_jobs or running:
        for name, item in list(running.items()):
            returncode = item.process.poll() if item.process is not None else None
            if returncode is None and _pid_alive(item.pid):
                continue
            if _run_finished(item.job):
                completed[name] = {
                    "gpu": item.gpu,
                    "pid": item.pid,
                    "returncode": returncode,
                    "run_dir": str(item.job.run_dir),
                    "checkpoint": str(item.job.run_dir / "checkpoints" / "last.ckpt"),
                    "log": str(item.log_path),
                }
                _append_event(queue_dir, {"event": "completed", "job": name, "pid": item.pid})
                _log(f"completed {name} pid={item.pid}")
            else:
                failed[name] = {
                    "gpu": item.gpu,
                    "pid": item.pid,
                    "returncode": returncode,
                    "run_dir": str(item.job.run_dir),
                    "log": str(item.log_path),
                    "reason": "process_exited_without_last_checkpoint",
                }
                _append_event(queue_dir, {"event": "failed", "job": name, "pid": item.pid})
                _log(f"failed {name} pid={item.pid}")
            del running[name]

        for job in list(pending_jobs):
            if job.depends_on is not None and job.depends_on in failed:
                failed[job.name] = {
                    "run_dir": str(job.run_dir),
                    "reason": "dependency_failed",
                    "depends_on": job.depends_on,
                }
                _append_event(
                    queue_dir,
                    {"event": "failed", "job": job.name, "reason": "dependency_failed", "depends_on": job.depends_on},
                )
                _log(f"failed {job.name}; dependency failed: {job.depends_on}")
                pending_jobs.remove(job)

        launched_any = False
        launch_slots = max(0, int(max_concurrent) - len(running))
        ready_jobs = [
            job
            for job in pending_jobs
            if _ready(job, completed, skipped, failed) and _existing_live_pid(job) is None
        ]
        reserved = {item.gpu for item in running.values() if item.gpu >= 0 and _pid_alive(item.pid)}
        stats = _parse_rocm_smi() if any(job.name in requirements for job in ready_jobs) else None
        for job in ready_jobs:
            if launch_slots <= 0:
                break
            if job.name in requirements:
                candidates = _gpus_with_free_vram(
                    stats or {},
                    allowed=allowed_gpus,
                    reserved=reserved,
                    required_free_mib=requirements[job.name],
                )
            else:
                candidates = [
                    gpu
                    for gpu in _idle_gpus(
                        allowed_gpus,
                        running,
                        max_util=max_util,
                        max_mem_mib=max_mem_mib,
                    )
                    if gpu not in reserved
                ]
            if not candidates:
                continue
            gpu = candidates[0]
            pending_jobs.remove(job)
            running[job.name] = _launch(job, gpu, python, queue_dir)
            reserved.add(gpu)
            launch_slots -= 1
            launched_any = True

        state = _build_state(
            pending=[job.name for job in pending_jobs],
            running=running,
            completed=completed,
            failed=failed,
            skipped=skipped,
        )
        state["allowed_gpus"] = allowed_gpus
        state["validated"] = validated
        state["memory_requirements_mib"] = requirements
        _write_state(queue_dir, state)

        if pending_jobs or running:
            if not launched_any:
                _log(
                    "waiting; pending="
                    + ",".join(job.name for job in pending_jobs)
                    + " running="
                    + ",".join(running)
                )
            time.sleep(max(1, poll_seconds))

    mail_status = _send_completion_mail(mail_to, smtp_config, state)
    state["mail_status"] = mail_status
    _write_state(queue_dir, state)
    return state


def _parse_gpus(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def _load_memory_requirements(path: Path | None) -> dict[str, float]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    requirements: dict[str, float] = {}
    for name, value in payload.items():
        if isinstance(value, dict):
            value = value["required_free_mib"]
        requirements[str(name)] = float(value)
    return requirements


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-dir", type=Path, default=DEFAULT_QUEUE_DIR)
    parser.add_argument("--gpus", default="7,6,5,4,3,2,1,0")
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--max-util", type=float, default=5.0)
    parser.add_argument("--max-mem-mib", type=float, default=1024.0)
    parser.add_argument("--max-concurrent", type=int, default=1)
    parser.add_argument("--mail-to", default="")
    parser.add_argument("--smtp-config", type=Path, default=None)
    parser.add_argument("--jobs", default="", help="Comma-separated subset of queue job names to run.")
    parser.add_argument("--memory-requirements-json", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    state = run_queue(
        queue_dir=args.queue_dir if args.queue_dir.is_absolute() else REPO_ROOT / args.queue_dir,
        allowed_gpus=_parse_gpus(args.gpus),
        python=args.python,
        poll_seconds=args.poll_seconds,
        max_util=args.max_util,
        max_mem_mib=args.max_mem_mib,
        max_concurrent=args.max_concurrent,
        mail_to=args.mail_to,
        smtp_config=args.smtp_config,
        dry_run=args.dry_run,
        selected_jobs=set(args.jobs.split(",")) if args.jobs else None,
        memory_requirements=_load_memory_requirements(args.memory_requirements_json),
    )
    print(json.dumps(state, indent=2, sort_keys=True))
    return 0 if not state.get("failed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
