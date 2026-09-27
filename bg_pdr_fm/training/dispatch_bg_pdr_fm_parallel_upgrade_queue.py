"""Dispatch BG-PDR-FM upgrade jobs onto naturally idle multi-GPU groups.

This queue is scoped to the current full-modality route:

* stage-1 direct background h128 then h192;
* stage-2 diffusers cross-attention latent FM medium then large.

It never kills existing processes. It polls GPU utilization, waits for stable
idle groups, runs a short batch-size ramp, then launches the selected e100 job.
"""

from __future__ import annotations

import argparse
import json
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
from bg_pdr_fm.training.monitor_run_and_mail import _read_smtp_config, _send_mail


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PYTHON = "/public/home/xuyinghao/miniconda3/envs/seg/bin/python"
DEFAULT_QUEUE_DIR = REPO_ROOT / "logs" / "bg_pdr_fm" / "parallel_upgrade_queue"
ROCM_SMI = "/opt/dtk-25.04.2/bin/rocm-smi"
DEFAULT_BATCH_CANDIDATES = (256, 512, 768, 1000)


@dataclass(frozen=True)
class UpgradeJob:
    name: str
    base_config: Path
    run_dir: Path
    checkpoint_name: str
    depends_on: str | None = None


@dataclass
class RunningJob:
    job: UpgradeJob
    gpus: tuple[int, ...]
    pid: int
    log_path: Path
    start_time: str
    process: subprocess.Popen[Any] | None = None


DEFAULT_JOBS = (
    UpgradeJob(
        name="background_h128",
        base_config=REPO_ROOT / "bg_pdr_fm" / "configs" / "openfwi_lmdb_background_unet_direct_h128_l1l2_e100.yaml",
        run_dir=REPO_ROOT / "logs" / "bg_pdr_fm" / "background_train_unet_direct_h128_l1l2_e100",
        checkpoint_name="background_last.ckpt",
    ),
    UpgradeJob(
        name="background_h192",
        base_config=REPO_ROOT / "bg_pdr_fm" / "configs" / "openfwi_lmdb_background_unet_direct_h192_l1l2_e100.yaml",
        run_dir=REPO_ROOT / "logs" / "bg_pdr_fm" / "background_train_unet_direct_h192_l1l2_e100",
        checkpoint_name="background_last.ckpt",
        depends_on="background_h128",
    ),
    UpgradeJob(
        name="diffusers_crossattn_medium",
        base_config=REPO_ROOT
        / "bg_pdr_fm"
        / "configs"
        / "openfwi_lmdb_residual_unet_direct_bg_h64_l1l2_diffusers_crossattn_oraclebg_medium_e100.yaml",
        run_dir=REPO_ROOT
        / "logs"
        / "bg_pdr_fm"
        / "residual_train_unet_direct_bg_h64_l1l2_diffusers_crossattn_oraclebg_medium_e100",
        checkpoint_name="residual_last.ckpt",
    ),
    UpgradeJob(
        name="diffusers_crossattn_large",
        base_config=REPO_ROOT
        / "bg_pdr_fm"
        / "configs"
        / "openfwi_lmdb_residual_unet_direct_bg_h64_l1l2_diffusers_crossattn_oraclebg_large_e100.yaml",
        run_dir=REPO_ROOT
        / "logs"
        / "bg_pdr_fm"
        / "residual_train_unet_direct_bg_h64_l1l2_diffusers_crossattn_oraclebg_large_e100",
        checkpoint_name="residual_last.ckpt",
        depends_on="diffusers_crossattn_medium",
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


def _existing_live_pid(job: UpgradeJob) -> int | None:
    for name in ("launcher.pid", "train.pid"):
        pid = _read_pid(job.run_dir / name)
        if _pid_alive(pid):
            return pid
    return None


def _run_checkpoint(job: UpgradeJob) -> Path:
    return job.run_dir / "stage_checkpoints" / job.checkpoint_name


def _run_finished(job: UpgradeJob) -> bool:
    return _run_checkpoint(job).is_file()


def parse_rocm_smi_text(text: str) -> dict[int, dict[str, float]]:
    stats: dict[int, dict[str, float]] = {}
    for line in text.splitlines():
        use_match = re.search(r"HCU\[(\d+)\]\s*: HCU use \(%\):\s*([0-9.]+)", line)
        if use_match:
            gpu = int(use_match.group(1))
            stats.setdefault(gpu, {})["util"] = float(use_match.group(2))
            continue
        mem_match = re.search(r"HCU\[(\d+)\]\s*: vram Total Used Memory \(MiB\):\s*([0-9.]+)", line)
        if mem_match:
            gpu = int(mem_match.group(1))
            stats.setdefault(gpu, {})["mem_mib"] = float(mem_match.group(2))
    return stats


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
    return parse_rocm_smi_text(result.stdout)


def idle_gpus_from_stats(
    stats: dict[int, dict[str, float]],
    *,
    allowed_gpus: list[int],
    max_util: float,
    max_mem_mib: float,
    reserved_gpus: set[int] | None = None,
) -> list[int]:
    reserved = reserved_gpus or set()
    idle = []
    for gpu in sorted(allowed_gpus, reverse=True):
        if gpu in reserved:
            continue
        values = stats.get(gpu, {})
        util = float(values.get("util", 100.0))
        mem_mib = float(values.get("mem_mib", 1_000_000.0))
        if util <= max_util and mem_mib <= max_mem_mib:
            idle.append(gpu)
    return idle


def allocate_gpu_groups(idle_gpus: list[int], *, group_size: int, max_groups: int) -> list[tuple[int, ...]]:
    groups: list[tuple[int, ...]] = []
    for idx in range(0, len(idle_gpus), group_size):
        group = tuple(idle_gpus[idx : idx + group_size])
        if len(group) != group_size:
            break
        groups.append(group)
        if len(groups) >= max_groups:
            break
    return groups


def _append_event(queue_dir: Path, event: dict[str, Any]) -> None:
    queue_dir.mkdir(parents=True, exist_ok=True)
    with (queue_dir / "events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time": _now(), **event}, sort_keys=True, default=str) + "\n")


def _write_state(queue_dir: Path, state: dict[str, Any]) -> None:
    queue_dir.mkdir(parents=True, exist_ok=True)
    (queue_dir / "queue_state.json").write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _command_env(gpus: tuple[int, ...]) -> dict[str, str]:
    env = os.environ.copy()
    visible = ",".join(str(gpu) for gpu in gpus)
    env["CUDA_VISIBLE_DEVICES"] = visible
    env["HIP_VISIBLE_DEVICES"] = visible
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["PYTHONUNBUFFERED"] = "1"
    return configure_checkpoint_worker_environment(env)


def prepare_training_config(
    base_conf: Any,
    *,
    job_name: str,
    run_dir: Path,
    batch_size: int,
    devices: int,
    max_epochs: int,
    limit_train_batches: int | None,
    limit_val_batches: int | None,
) -> Any:
    conf = OmegaConf.create(OmegaConf.to_container(base_conf, resolve=False))
    OmegaConf.update(conf, "training.batch_size", int(batch_size), merge=True)
    OmegaConf.update(conf, "training.devices", int(devices), merge=True)
    OmegaConf.update(conf, "training.max_epochs", int(max_epochs), merge=True)
    OmegaConf.update(conf, "training.limit_train_batches", limit_train_batches, merge=True)
    OmegaConf.update(conf, "training.limit_val_batches", limit_val_batches, merge=True)
    OmegaConf.update(conf, "training.stage_checkpoint_path", str(run_dir / "stage_checkpoints"), merge=True)
    OmegaConf.update(conf, "training.logging.log_dir", str(run_dir / "lightning"), merge=True)
    OmegaConf.update(conf, "training.logging.log_version", job_name, merge=True)
    OmegaConf.update(conf, "training.logging.add_timestamp", False, merge=True)
    OmegaConf.update(conf, "training.checkpoint.dirpath", str(run_dir / "lightning" / "checkpoints"), merge=True)
    OmegaConf.update(conf, "training.checkpoint.filename", f"{job_name}-epoch_{{epoch}}-loss{{val/loss:.4f}}", merge=True)
    OmegaConf.update(conf, "diagnostics.output_dir", str(run_dir / "diagnostics"), merge=True)
    OmegaConf.update(conf, "evaluation.output_dir", str(run_dir / "evaluation"), merge=True)
    return conf


def _save_config(conf: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=conf, f=str(path))
    return path


def _run_command(command: list[str], *, gpus: tuple[int, ...], log_path: Path, queue_dir: Path, python: str) -> int:
    del python
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _append_event(queue_dir, {"event": "run_start", "gpus": gpus, "command": command, "log": str(log_path)})
    _log(f"running on {gpus}: {' '.join(command)}")
    with log_path.open("ab", buffering=0) as handle:
        result = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            env=_command_env(gpus),
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    _append_event(queue_dir, {"event": "run_done", "code": result.returncode, "log": str(log_path)})
    return int(result.returncode)


def _read_worker_summary(job: UpgradeJob) -> dict[str, Any] | None:
    path = job.run_dir / "queue_worker_summary.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _batch_ramp(
    job: UpgradeJob,
    *,
    gpus: tuple[int, ...],
    python: str,
    queue_dir: Path,
    batch_candidates: tuple[int, ...],
    ramp_steps: int,
    dry_run: bool,
) -> tuple[int, list[dict[str, Any]]]:
    base_conf = OmegaConf.load(job.base_config)
    results: list[dict[str, Any]] = []
    best = int(OmegaConf.select(base_conf, "training.batch_size", default=batch_candidates[0]))
    for batch_size in batch_candidates:
        ramp_dir = job.run_dir / "batch_ramp" / f"bs{batch_size}"
        ramp_conf = prepare_training_config(
            base_conf,
            job_name=f"{job.name}_ramp_bs{batch_size}",
            run_dir=ramp_dir,
            batch_size=int(batch_size),
            devices=len(gpus),
            max_epochs=1,
            limit_train_batches=int(ramp_steps),
            limit_val_batches=1,
        )
        config_path = _save_config(ramp_conf, queue_dir / "generated_configs" / f"{job.name}_ramp_bs{batch_size}.yaml")
        item = {"batch_size": int(batch_size), "config": str(config_path), "run_dir": str(ramp_dir)}
        if dry_run:
            item["status"] = "dry_run"
            best = int(batch_size)
            results.append(item)
            continue
        code = _run_command(
            [python, "-m", "bg_pdr_fm.training.train_bg_pdr_fm", "--config", str(config_path)],
            gpus=gpus,
            log_path=ramp_dir / "train.log",
            queue_dir=queue_dir,
            python=python,
        )
        item["returncode"] = code
        item["status"] = "ok" if code == 0 else "failed"
        results.append(item)
        if code == 0:
            best = int(batch_size)
            continue
        break
    return best, results


def _launch_formal(
    job: UpgradeJob,
    *,
    gpus: tuple[int, ...],
    python: str,
    queue_dir: Path,
    batch_size: int,
    dry_run: bool,
) -> RunningJob:
    base_conf = OmegaConf.load(job.base_config)
    formal_conf = prepare_training_config(
        base_conf,
        job_name=job.name,
        run_dir=job.run_dir,
        batch_size=int(batch_size),
        devices=len(gpus),
        max_epochs=int(OmegaConf.select(base_conf, "training.max_epochs", default=100)),
        limit_train_batches=None,
        limit_val_batches=None,
    )
    config_path = _save_config(formal_conf, job.run_dir / "train_config.generated.yaml")
    log_path = job.run_dir / "train.log"
    pid_path = job.run_dir / "launcher.pid"
    command = [python, "-m", "bg_pdr_fm.training.train_bg_pdr_fm", "--config", str(config_path)]
    if dry_run:
        return RunningJob(job=job, gpus=gpus, pid=-1, log_path=log_path, start_time="dry_run")
    job.run_dir.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab", buffering=0) as handle:
        process = subprocess.Popen(
            command,
            cwd=str(REPO_ROOT),
            env=_command_env(gpus),
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    pid_path.write_text(f"{process.pid}\n", encoding="utf-8")
    running = RunningJob(job=job, gpus=gpus, pid=process.pid, log_path=log_path, start_time=_now(), process=process)
    _append_event(
        queue_dir,
        {
            "event": "formal_launch",
            "job": job.name,
            "gpus": gpus,
            "pid": process.pid,
            "batch_size": int(batch_size),
            "config": str(config_path),
            "log": str(log_path),
        },
    )
    _log(f"launched {job.name} pid={process.pid} gpus={gpus} batch_size={batch_size}")
    return running


def _run_formal_blocking(
    job: UpgradeJob,
    *,
    gpus: tuple[int, ...],
    python: str,
    queue_dir: Path,
    batch_size: int,
) -> int:
    base_conf = OmegaConf.load(job.base_config)
    formal_conf = prepare_training_config(
        base_conf,
        job_name=job.name,
        run_dir=job.run_dir,
        batch_size=int(batch_size),
        devices=len(gpus),
        max_epochs=int(OmegaConf.select(base_conf, "training.max_epochs", default=100)),
        limit_train_batches=None,
        limit_val_batches=None,
    )
    config_path = _save_config(formal_conf, job.run_dir / "train_config.generated.yaml")
    return _run_command(
        [python, "-m", "bg_pdr_fm.training.train_bg_pdr_fm", "--config", str(config_path)],
        gpus=gpus,
        log_path=job.run_dir / "train.log",
        queue_dir=queue_dir,
        python=python,
    )


def run_worker(
    job: UpgradeJob,
    *,
    queue_dir: Path,
    gpus: tuple[int, ...],
    python: str,
    batch_candidates: tuple[int, ...],
    ramp_steps: int,
) -> int:
    job.run_dir.mkdir(parents=True, exist_ok=True)
    summary_path = job.run_dir / "queue_worker_summary.json"
    _, ramp_results = _batch_ramp(
        job,
        gpus=gpus,
        python=python,
        queue_dir=queue_dir,
        batch_candidates=batch_candidates,
        ramp_steps=ramp_steps,
        dry_run=False,
    )
    best_batch = best_successful_batch(ramp_results)
    summary: dict[str, Any] = {
        "job": job.name,
        "gpus": list(gpus),
        "ramp_results": ramp_results,
        "best_batch": best_batch,
        "time": _now(),
    }
    if best_batch is None:
        summary["status"] = "batch_ramp_failed"
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _append_event(queue_dir, {"event": "batch_ramp_failed", "job": job.name, "results": ramp_results})
        return 2
    _append_event(queue_dir, {"event": "batch_ramp_done", "job": job.name, "best_batch": best_batch, "results": ramp_results})
    train_code = _run_formal_blocking(
        job,
        gpus=gpus,
        python=python,
        queue_dir=queue_dir,
        batch_size=best_batch,
    )
    summary["train_returncode"] = train_code
    summary["checkpoint"] = str(_run_checkpoint(job))
    summary["status"] = "completed" if train_code == 0 and _run_checkpoint(job).is_file() else "failed"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if summary["status"] == "completed" else 1


def build_worker_command(
    job: UpgradeJob,
    *,
    python: str,
    queue_dir: Path,
    gpus: tuple[int, ...],
    batch_candidates: tuple[int, ...],
    ramp_steps: int,
) -> list[str]:
    return [
        python,
        "-m",
        "bg_pdr_fm.training.dispatch_bg_pdr_fm_parallel_upgrade_queue",
        "--worker-job",
        job.name,
        "--worker-gpus",
        ",".join(str(gpu) for gpu in gpus),
        "--queue-dir",
        str(queue_dir),
        "--python",
        python,
        "--batch-candidates",
        ",".join(str(batch) for batch in batch_candidates),
        "--ramp-steps",
        str(int(ramp_steps)),
    ]


def _launch_worker(
    job: UpgradeJob,
    *,
    gpus: tuple[int, ...],
    python: str,
    queue_dir: Path,
    batch_candidates: tuple[int, ...],
    ramp_steps: int,
) -> RunningJob:
    job.run_dir.mkdir(parents=True, exist_ok=True)
    log_path = job.run_dir / "queue_worker.log"
    pid_path = job.run_dir / "launcher.pid"
    command = build_worker_command(
        job,
        python=python,
        queue_dir=queue_dir,
        gpus=gpus,
        batch_candidates=batch_candidates,
        ramp_steps=ramp_steps,
    )
    with log_path.open("ab", buffering=0) as handle:
        process = subprocess.Popen(
            command,
            cwd=str(REPO_ROOT),
            env=_command_env(gpus),
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    pid_path.write_text(f"{process.pid}\n", encoding="utf-8")
    running = RunningJob(job=job, gpus=gpus, pid=process.pid, log_path=log_path, start_time=_now(), process=process)
    _append_event(
        queue_dir,
        {
            "event": "worker_launch",
            "job": job.name,
            "gpus": gpus,
            "pid": process.pid,
            "command": command,
            "log": str(log_path),
        },
    )
    _log(f"launched worker {job.name} pid={process.pid} gpus={gpus}")
    return running


def dependency_ready(
    job: UpgradeJob,
    completed: dict[str, Any],
    skipped: dict[str, Any],
    failed: dict[str, Any],
) -> bool:
    if job.depends_on is None:
        return True
    del failed
    return job.depends_on in completed or job.depends_on in skipped


def skip_failed_dependency_jobs(
    pending: list[UpgradeJob],
    *,
    skipped: dict[str, dict[str, Any]],
    failed: dict[str, dict[str, Any]],
) -> None:
    changed = True
    while changed:
        changed = False
        failed_or_blocked = {
            name
            for name, info in skipped.items()
            if str(info.get("reason", "")) == "dependency_failed"
        } | set(failed)
        for job in list(pending):
            if job.depends_on in failed_or_blocked:
                skipped[job.name] = {
                    "reason": "dependency_failed",
                    "dependency": job.depends_on,
                    "run_dir": str(job.run_dir),
                }
                pending.remove(job)
                changed = True


def best_successful_batch(results: list[dict[str, Any]]) -> int | None:
    best: int | None = None
    for item in results:
        if item.get("status") == "ok" or item.get("status") == "dry_run":
            best = int(item["batch_size"])
    return best


def _build_state(
    *,
    pending: list[UpgradeJob],
    running: dict[str, RunningJob],
    completed: dict[str, dict[str, Any]],
    failed: dict[str, dict[str, Any]],
    skipped: dict[str, dict[str, Any]],
    selected_batches: dict[str, int],
) -> dict[str, Any]:
    return {
        "updated_at": _now(),
        "pending": [job.name for job in pending],
        "running": {
            name: {
                "gpus": list(item.gpus),
                "pid": item.pid,
                "alive": _pid_alive(item.pid) if item.pid > 0 else False,
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
        "selected_batches": selected_batches,
    }


def _send_completion_mail(mail_to: str, smtp_config: Path | None, state: dict[str, Any]) -> int:
    if not mail_to:
        return 0
    lines = [
        "BG-PDR-FM parallel upgrade queue finished.",
        f"time: {_now()}",
        "",
        f"completed: {', '.join(sorted(state.get('completed', {}))) or 'none'}",
        f"skipped: {', '.join(sorted(state.get('skipped', {}))) or 'none'}",
        f"failed: {', '.join(sorted(state.get('failed', {}))) or 'none'}",
        f"selected_batches: {state.get('selected_batches', {})}",
        "",
        f"queue_state: {DEFAULT_QUEUE_DIR / 'queue_state.json'}",
    ]
    smtp_conf = _read_smtp_config(smtp_config)
    return _send_mail(mail_to, "[BG-PDR-FM] parallel upgrade queue finished", "\n".join(lines) + "\n", smtp_conf)


def run_queue(
    *,
    queue_dir: Path,
    allowed_gpus: list[int],
    group_size: int,
    max_groups: int,
    python: str,
    poll_seconds: int,
    max_util: float,
    max_mem_mib: float,
    batch_candidates: tuple[int, ...],
    ramp_steps: int,
    mail_to: str,
    smtp_config: Path | None,
    dry_run: bool,
    selected_jobs: set[str] | None = None,
) -> dict[str, Any]:
    queue_dir.mkdir(parents=True, exist_ok=True)
    jobs = [job for job in DEFAULT_JOBS if selected_jobs is None or job.name in selected_jobs]
    known = {job.name for job in DEFAULT_JOBS}
    if selected_jobs:
        unknown = sorted(selected_jobs - known)
        if unknown:
            raise ValueError(f"Unknown queue job(s): {', '.join(unknown)}")
    for job in jobs:
        if not job.base_config.is_file():
            raise FileNotFoundError(f"Missing config for {job.name}: {job.base_config}")

    pending = list(jobs)
    running: dict[str, RunningJob] = {}
    completed: dict[str, dict[str, Any]] = {}
    failed: dict[str, dict[str, Any]] = {}
    skipped: dict[str, dict[str, Any]] = {}
    selected_batches: dict[str, int] = {}

    for job in list(pending):
        live_pid = _existing_live_pid(job)
        if live_pid is not None:
            running[job.name] = RunningJob(
                job=job,
                gpus=(),
                pid=live_pid,
                log_path=job.run_dir / "train.log",
                start_time="preexisting",
            )
            pending.remove(job)
        elif _run_finished(job):
            skipped[job.name] = {"reason": "checkpoint_exists", "checkpoint": str(_run_checkpoint(job))}
            pending.remove(job)

    while pending or running:
        skip_failed_dependency_jobs(pending, skipped=skipped, failed=failed)
        for name, item in list(running.items()):
            returncode = item.process.poll() if item.process is not None else None
            if returncode is None and _pid_alive(item.pid):
                continue
            summary = _read_worker_summary(item.job) or {}
            if _run_finished(item.job):
                completed[name] = {
                    "returncode": returncode,
                    "pid": item.pid,
                    "gpus": list(item.gpus),
                    "checkpoint": str(_run_checkpoint(item.job)),
                    "run_dir": str(item.job.run_dir),
                    "log": str(item.log_path),
                    "worker_summary": summary,
                }
            else:
                failed[name] = {
                    "returncode": returncode,
                    "pid": item.pid,
                    "gpus": list(item.gpus),
                    "reason": str(summary.get("status") or "process_exited_without_stage_checkpoint"),
                    "run_dir": str(item.job.run_dir),
                    "log": str(item.log_path),
                    "worker_summary": summary,
                }
            del running[name]
        skip_failed_dependency_jobs(pending, skipped=skipped, failed=failed)

        reserved = {gpu for item in running.values() for gpu in item.gpus}
        stats = _parse_rocm_smi() if not dry_run else {gpu: {"util": 0.0, "mem_mib": 0.0} for gpu in allowed_gpus}
        idle = idle_gpus_from_stats(stats, allowed_gpus=allowed_gpus, max_util=max_util, max_mem_mib=max_mem_mib, reserved_gpus=reserved)
        groups = allocate_gpu_groups(idle, group_size=group_size, max_groups=max_groups - len(running))

        launched_any = False
        for gpus in groups:
            ready_jobs = [
                job
                for job in pending
                if dependency_ready(job, completed, skipped, failed) and _existing_live_pid(job) is None
            ]
            if not ready_jobs:
                break
            job = ready_jobs[0]
            pending.remove(job)
            if dry_run:
                ramp_results = [
                    {"batch_size": int(batch), "status": "dry_run", "run_dir": str(job.run_dir / "batch_ramp" / f"bs{batch}")}
                    for batch in batch_candidates
                ]
                best_batch = best_successful_batch(ramp_results)
                assert best_batch is not None
                selected_batches[job.name] = best_batch
                completed[job.name] = {
                    "returncode": None,
                    "pid": -1,
                    "gpus": list(gpus),
                    "checkpoint": str(_run_checkpoint(job)),
                    "run_dir": str(job.run_dir),
                    "log": str(job.run_dir / "queue_worker.log"),
                    "worker_summary": {"status": "dry_run", "ramp_results": ramp_results, "best_batch": best_batch},
                }
            else:
                running[job.name] = _launch_worker(
                    job,
                    gpus=gpus,
                    python=python,
                    queue_dir=queue_dir,
                    batch_candidates=batch_candidates,
                    ramp_steps=ramp_steps,
                )
            launched_any = True

        state = _build_state(
            pending=pending,
            running=running,
            completed=completed,
            failed=failed,
            skipped=skipped,
            selected_batches=selected_batches,
        )
        state["allowed_gpus"] = allowed_gpus
        state["group_size"] = group_size
        _write_state(queue_dir, state)
        if dry_run:
            return state
        if pending or running:
            if not launched_any:
                _log("waiting; pending=" + ",".join(job.name for job in pending) + " running=" + ",".join(running))
            time.sleep(max(1, poll_seconds))

    mail_status = _send_completion_mail(mail_to, smtp_config, state)
    state["mail_status"] = mail_status
    _write_state(queue_dir, state)
    return state


def _parse_gpus(text: str) -> list[int]:
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def _parse_batches(text: str) -> tuple[int, ...]:
    batches = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    if not batches:
        raise ValueError("At least one batch size candidate is required.")
    return batches


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-dir", type=Path, default=DEFAULT_QUEUE_DIR)
    parser.add_argument("--gpus", default="7,6,5,4,3,2,1,0")
    parser.add_argument("--group-size", type=int, default=3)
    parser.add_argument("--max-groups", type=int, default=2)
    parser.add_argument("--python", default=DEFAULT_PYTHON)
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--max-util", type=float, default=5.0)
    parser.add_argument("--max-mem-mib", type=float, default=1024.0)
    parser.add_argument("--batch-candidates", default="256,512,768,1000")
    parser.add_argument("--ramp-steps", type=int, default=20)
    parser.add_argument("--mail-to", default="")
    parser.add_argument("--smtp-config", type=Path, default=None)
    parser.add_argument("--jobs", default="", help="Comma-separated subset of job names to run.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--worker-job", default="", help=argparse.SUPPRESS)
    parser.add_argument("--worker-gpus", default="", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.worker_job:
        jobs = {job.name: job for job in DEFAULT_JOBS}
        if args.worker_job not in jobs:
            raise ValueError(f"Unknown worker job: {args.worker_job}")
        worker_gpus = tuple(_parse_gpus(args.worker_gpus))
        if not worker_gpus:
            raise ValueError("--worker-gpus is required in worker mode.")
        return run_worker(
            jobs[args.worker_job],
            queue_dir=args.queue_dir if args.queue_dir.is_absolute() else REPO_ROOT / args.queue_dir,
            gpus=worker_gpus,
            python=args.python,
            batch_candidates=_parse_batches(args.batch_candidates),
            ramp_steps=int(args.ramp_steps),
        )

    state = run_queue(
        queue_dir=args.queue_dir if args.queue_dir.is_absolute() else REPO_ROOT / args.queue_dir,
        allowed_gpus=_parse_gpus(args.gpus),
        group_size=int(args.group_size),
        max_groups=int(args.max_groups),
        python=args.python,
        poll_seconds=int(args.poll_seconds),
        max_util=float(args.max_util),
        max_mem_mib=float(args.max_mem_mib),
        batch_candidates=_parse_batches(args.batch_candidates),
        ramp_steps=int(args.ramp_steps),
        mail_to=args.mail_to,
        smtp_config=args.smtp_config,
        dry_run=bool(args.dry_run),
        selected_jobs=set(args.jobs.split(",")) if args.jobs else None,
    )
    print(json.dumps(state, indent=2, sort_keys=True))
    return 0 if not state.get("failed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
