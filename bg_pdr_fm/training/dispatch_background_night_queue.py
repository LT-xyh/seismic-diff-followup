"""Dispatch the overnight BG-PDR-FM background follow-up run.

The dispatcher waits for the current pure HC64 probe to finish, reads its
Lightning CSV metrics, and launches exactly one follow-up background run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from bg_pdr_fm.runtime import configure_checkpoint_worker_environment

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIGS = {
    "to100": REPO_ROOT / "bg_pdr_fm" / "configs" / "openfwi_lmdb_background_ms_hc64_to100.yaml",
    "lr03": REPO_ROOT / "bg_pdr_fm" / "configs" / "openfwi_lmdb_background_ms_hc64_lr03.yaml",
    "highguard": REPO_ROOT / "bg_pdr_fm" / "configs" / "openfwi_lmdb_background_ms_hc64_highguard.yaml",
}
OUTPUT_DIRS = {
    "to100": REPO_ROOT / "logs" / "bg_pdr_fm" / "background_train_ms_hc64_to100",
    "lr03": REPO_ROOT / "logs" / "bg_pdr_fm" / "background_train_ms_hc64_lr03",
    "highguard": REPO_ROOT / "logs" / "bg_pdr_fm" / "background_train_ms_hc64_highguard",
}
QUEUE_DIR = REPO_ROOT / "logs" / "bg_pdr_fm" / "night_queue_hc64"


def _float_value(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    if value is None or value == "":
        raise KeyError(f"Missing metric {key!r}.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Metric {key!r} is not finite: {value!r}.")
    return result


def read_validation_rows(current_run: Path) -> list[dict[str, str]]:
    metric_paths = sorted(
        (current_run / "lightning").rglob("metrics.csv"),
        key=lambda path: path.stat().st_mtime,
    )
    if not metric_paths:
        raise FileNotFoundError(f"No Lightning metrics.csv found under {current_run / 'lightning'}.")
    with metric_paths[-1].open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("val/loss")]
    if not rows:
        raise ValueError(f"No validation rows found in {metric_paths[-1]}.")
    return rows


def _high_leak_sustained_up(rows: list[dict[str, Any]], threshold: float = 0.016, window: int = 5) -> bool:
    if len(rows) < window:
        return False
    values = [_float_value(row, "val/bg_high_leak") for row in rows[-window:]]
    return values[-1] > threshold and all(next_value > value for value, next_value in zip(values, values[1:]))


def choose_next_branch(latest: dict[str, Any], rows: list[dict[str, Any]]) -> str:
    rho_true = _float_value(latest, "val/rho_true")
    bg_high_leak = _float_value(latest, "val/bg_high_leak")
    if _high_leak_sustained_up(rows):
        return "highguard"
    if rho_true < 2.0 and bg_high_leak <= 0.016:
        return "to100"
    return "lr03"


def _wait_for_current_run(current_run: Path, poll_seconds: int) -> None:
    pid_path = current_run / "train.pid"
    if not pid_path.is_file():
        return
    text = pid_path.read_text(encoding="utf-8").strip()
    if not text:
        return
    pid = int(text)
    while Path(f"/proc/{pid}").exists():
        time.sleep(poll_seconds)


def _launch_followup(branch: str, gpus: str, python: str) -> dict[str, str]:
    config_path = CONFIGS[branch]
    output_dir = OUTPUT_DIRS[branch]
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing follow-up config: {config_path}")
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train.log"
    pid_path = output_dir / "train.pid"
    command = [python, "-m", "bg_pdr_fm.training.train_bg_pdr_fm", "--config", str(config_path)]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpus
    env.setdefault("PYTHONUNBUFFERED", "1")
    env = configure_checkpoint_worker_environment(env, output_dir / "tmp")
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
    return {
        "pid": str(process.pid),
        "pid_file": str(pid_path),
        "log": str(log_path),
        "config": str(config_path),
        "command": " ".join(command),
    }


def dispatch(
    current_run: Path,
    gpus: str,
    python: str,
    poll_seconds: int,
    dry_run: bool,
    skip_wait: bool,
) -> dict[str, Any]:
    if not skip_wait:
        _wait_for_current_run(current_run, poll_seconds)
    rows = read_validation_rows(current_run)
    latest = rows[-1]
    branch = choose_next_branch(latest, rows)
    current_stage_checkpoint = current_run / "stage_checkpoints" / "background_last.ckpt"
    if not dry_run and not current_stage_checkpoint.is_file():
        raise FileNotFoundError(f"Current background stage checkpoint is missing: {current_stage_checkpoint}")
    launch: dict[str, str] | None = None
    if not dry_run:
        launch = _launch_followup(branch, gpus=gpus, python=python)
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "current_run": str(current_run),
        "selected_branch": branch,
        "latest_metrics": {
            "epoch": latest.get("epoch", ""),
            "val_loss": _float_value(latest, "val/loss"),
            "val_rho_true": _float_value(latest, "val/rho_true"),
            "val_bg_high_leak": _float_value(latest, "val/bg_high_leak"),
            "val_bg_l1": _float_value(latest, "val/bg_l1"),
            "val_bg_ms_l1": _float_value(latest, "val/bg_ms_l1"),
        },
        "dry_run": dry_run,
        "launch": launch,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Dispatch the BG-PDR-FM HC64 overnight background queue.")
    parser.add_argument("--current-run", default="logs/bg_pdr_fm/background_train_ms_hc64")
    parser.add_argument("--gpus", default="4,5,6,7")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-wait", action="store_true")
    args = parser.parse_args()
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    decision = dispatch(
        current_run=(REPO_ROOT / args.current_run).resolve() if not Path(args.current_run).is_absolute() else Path(args.current_run),
        gpus=args.gpus,
        python=args.python,
        poll_seconds=args.poll_seconds,
        dry_run=args.dry_run,
        skip_wait=args.skip_wait,
    )
    decision_path = QUEUE_DIR / "decision.json"
    decision_path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
