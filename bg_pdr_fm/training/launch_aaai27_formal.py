"""Launch AAAI27 formal benchmark jobs on selected GPUs.

This helper keeps process management out of shell one-liners: every job gets a
separate CUDA_VISIBLE_DEVICES value, stdout log, and pid file.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from bg_pdr_fm.runtime import configure_checkpoint_worker_environment

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "bg_pdr_fm" / "configs" / "experiments" / "aaai27"
LAUNCH_DIR = REPO_ROOT / "logs" / "bg_pdr_fm" / "aaai27" / "formal" / "launch"


@dataclass(frozen=True)
class JobSpec:
    name: str
    config: str


MAIN_TABLE_JOBS = (
    JobSpec("sv_inv_net", "formal_sv_inv_net.yaml"),
    JobSpec("velocity_gan", "formal_velocity_gan.yaml"),
    JobSpec("conditional_ddpm", "formal_conditional_ddpm.yaml"),
    JobSpec("adapted_gfi", "formal_adapted_gfi_multimodal.yaml"),
    JobSpec("adapted_auto_linear", "formal_adapted_auto_linear_multimodal.yaml"),
    JobSpec("bg_pdr_fm", "formal_bg_pdr_fm.yaml"),
)

ABLATION_JOBS = (
    JobSpec("concat_fm_matched", "formal_concat_fm_matched.yaml"),
    JobSpec("cncs_fm_matched", "formal_cncs_fm_matched.yaml"),
    JobSpec("bg_pdr_fm_no_gate", "formal_bg_pdr_fm_no_gate.yaml"),
)


def _select_jobs(names: list[str], include_bg: bool, include_matched: bool) -> list[JobSpec]:
    available = {job.name: job for job in (*MAIN_TABLE_JOBS, *ABLATION_JOBS)}
    if names:
        unknown = [name for name in names if name not in available]
        if unknown:
            raise ValueError(f"Unknown jobs {unknown}; available: {sorted(available)}")
        jobs = [available[name] for name in names]
    else:
        jobs = [job for job in MAIN_TABLE_JOBS if include_bg or job.name != "bg_pdr_fm"]
        if include_matched:
            jobs.extend(ABLATION_JOBS)
    return jobs


def launch_jobs(jobs: list[JobSpec], gpus: list[str], python: str, dry_run: bool = False) -> list[dict[str, str]]:
    if not gpus:
        raise ValueError("At least one GPU id is required.")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = LAUNCH_DIR / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    launched: list[dict[str, str]] = []
    for index, job in enumerate(jobs):
        gpu = str(gpus[index % len(gpus)])
        config_path = CONFIG_DIR / job.config
        if not config_path.is_file():
            raise FileNotFoundError(f"Missing benchmark config: {config_path}")
        log_path = run_dir / f"{job.name}.log"
        pid_path = run_dir / f"{job.name}.pid"
        command = [
            python,
            "-m",
            "bg_pdr_fm.training.train_aaai27_benchmark",
            "--config",
            str(config_path),
        ]
        launched.append({
            "job": job.name,
            "gpu": gpu,
            "config": str(config_path),
            "log": str(log_path),
            "pid": str(pid_path),
        })
        if dry_run:
            continue
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env.setdefault("PYTHONUNBUFFERED", "1")
        env = configure_checkpoint_worker_environment(env, LAUNCH_DIR / name / "tmp")
        with log_path.open("ab", buffering=0) as log_file:
            process = subprocess.Popen(
                command,
                cwd=str(REPO_ROOT),
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        pid_path.write_text(f"{process.pid}\n", encoding="utf-8")
    manifest = run_dir / "manifest.tsv"
    manifest.write_text(
        "job\tgpu\tpid\tconfig\tlog\n"
        + "\n".join(
            f"{item['job']}\t{item['gpu']}\t{Path(item['pid']).read_text().strip() if Path(item['pid']).exists() else ''}\t{item['config']}\t{item['log']}"
            for item in launched
        )
        + "\n",
        encoding="utf-8",
    )
    return launched


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch AAAI27 formal benchmark training jobs.")
    parser.add_argument("--gpus", nargs="+", default=["0", "1", "2", "3"])
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--jobs", nargs="*", default=[])
    parser.add_argument("--include-bg", action="store_true", help="Also launch formal_bg_pdr_fm.")
    parser.add_argument("--include-matched", action="store_true", help="Also launch matched/core ablation jobs.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    jobs = _select_jobs(args.jobs, include_bg=args.include_bg, include_matched=args.include_matched)
    launched = launch_jobs(jobs, args.gpus, args.python, dry_run=args.dry_run)
    for item in launched:
        print(f"{item['job']}\tgpu={item['gpu']}\tpid_file={item['pid']}\tlog={item['log']}")


if __name__ == "__main__":
    main()
