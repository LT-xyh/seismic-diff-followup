"""Dispatch the follow-up queue for the wavelet HC64 background probe.

The dispatcher leaves the active probe untouched, waits for it to finish,
reads its final Lightning metrics, and launches one isolated follow-up run.
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

from omegaconf import OmegaConf

from bg_pdr_fm.runtime import configure_checkpoint_worker_environment

REPO_ROOT = Path(__file__).resolve().parents[2]
CURRENT_CONFIG = REPO_ROOT / "bg_pdr_fm" / "configs" / "openfwi_lmdb_background_wavelet_hc64.yaml"
CONFIGS = {
    "to100": REPO_ROOT / "bg_pdr_fm" / "configs" / "openfwi_lmdb_background_wavelet_hc64_to100.yaml",
    "to60": REPO_ROOT / "bg_pdr_fm" / "configs" / "openfwi_lmdb_background_wavelet_hc64_to60.yaml",
    "level2": REPO_ROOT / "bg_pdr_fm" / "configs" / "openfwi_lmdb_background_wavelet_l2_hc64.yaml",
}
OUTPUT_DIRS = {
    "to100": REPO_ROOT / "logs" / "bg_pdr_fm" / "background_train_wavelet_hc64_to100",
    "to60": REPO_ROOT / "logs" / "bg_pdr_fm" / "background_train_wavelet_hc64_to60",
    "level2": REPO_ROOT / "logs" / "bg_pdr_fm" / "background_train_wavelet_l2_hc64",
}
QUEUE_DIR = REPO_ROOT / "logs" / "bg_pdr_fm" / "wavelet_queue_hc64"


def _float_value(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    if value is None or value == "":
        raise KeyError(f"Missing metric {key!r}.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Metric {key!r} is not finite: {value!r}.")
    return result


def _resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (REPO_ROOT / path)


def read_validation_rows(current_run: Path) -> tuple[Path, list[dict[str, str]]]:
    metric_paths = sorted(
        (current_run / "lightning").rglob("metrics.csv"),
        key=lambda path: path.stat().st_mtime,
    )
    if not metric_paths:
        raise FileNotFoundError(f"No Lightning metrics.csv found under {current_run / 'lightning'}.")
    metrics_path = metric_paths[-1]
    with metrics_path.open(newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("val/loss")]
    if not rows:
        raise ValueError(f"No validation rows found in {metrics_path}.")
    return metrics_path, rows


def _strictly_recently_decreasing(rows: list[dict[str, Any]], key: str, window: int = 5, min_delta: float = 1e-3) -> bool:
    if len(rows) < window:
        return False
    values = [_float_value(row, key) for row in rows[-window:]]
    return values[-1] < values[0] - min_delta and values[-1] <= min(values[:-1])


def _recent_plateau(rows: list[dict[str, Any]], key: str, window: int = 5, min_improvement: float = 0.002) -> bool:
    if len(rows) < window:
        return False
    values = [_float_value(row, key) for row in rows[-window:]]
    return values[0] - values[-1] < min_improvement


def choose_next_branch(
    rows: list[dict[str, Any]],
    detail_leak_eps: float,
    rho_decrease_window: int,
    rho_min_delta: float,
    bg_l1_plateau_window: int,
    bg_l1_min_improvement: float,
) -> tuple[str, dict[str, Any]]:
    latest = rows[-1]
    rho_true = _float_value(latest, "val/rho_true")
    detail_leak = _float_value(latest, "val/bg_wavelet_detail_leak")
    bg_l1_plateau = _recent_plateau(
        rows,
        "val/bg_l1",
        window=bg_l1_plateau_window,
        min_improvement=bg_l1_min_improvement,
    )
    rho_decreasing = _strictly_recently_decreasing(
        rows,
        "val/rho_true",
        window=rho_decrease_window,
        min_delta=rho_min_delta,
    )

    reason: str
    if rho_true > 2.5:
        branch = "level2"
        reason = "rho_true > 2.5, pure one-level LL representation is not sufficient."
    elif bg_l1_plateau:
        branch = "level2"
        reason = "val/bg_l1 plateaued in the recent validation window."
    elif rho_true < 2.0 and detail_leak <= detail_leak_eps:
        branch = "to100"
        reason = "rho_true < 2.0 and wavelet detail leak is near zero."
    elif 2.0 <= rho_true <= 2.5 and rho_decreasing:
        branch = "to60"
        reason = "rho_true is in the gray zone but still clearly decreasing."
    else:
        branch = "level2"
        reason = "one-level LL probe did not satisfy any safe continuation rule."

    return branch, {
        "reason": reason,
        "rho_decreasing": rho_decreasing,
        "bg_l1_plateau": bg_l1_plateau,
        "thresholds": {
            "detail_leak_eps": detail_leak_eps,
            "rho_decrease_window": rho_decrease_window,
            "rho_min_delta": rho_min_delta,
            "bg_l1_plateau_window": bg_l1_plateau_window,
            "bg_l1_min_improvement": bg_l1_min_improvement,
        },
    }


def _read_pid(path: Path) -> int | None:
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError as exc:
        raise ValueError(f"Invalid pid file {path}: {text!r}") from exc


def _pid_exists(pid: int | None) -> bool:
    return pid is not None and Path(f"/proc/{pid}").exists()


def _active_current_processes(config_path: Path) -> list[int]:
    try:
        output = subprocess.check_output(["ps", "-eo", "pid=,cmd="], text=True)
    except subprocess.CalledProcessError:
        return []
    needles = {
        str(config_path),
        os.path.relpath(config_path, REPO_ROOT),
    }
    pids: list[int] = []
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        pid_text, _, cmd = stripped.partition(" ")
        if "train_bg_pdr_fm" not in cmd:
            continue
        if any(needle in cmd for needle in needles):
            pids.append(int(pid_text))
    return pids


def wait_for_current_run(current_run: Path, current_config: Path, poll_seconds: int) -> None:
    pid_paths = [current_run / "launcher.pid", current_run / "train.pid"]
    while True:
        live_pids = [pid for pid in (_read_pid(path) for path in pid_paths) if _pid_exists(pid)]
        live_pids.extend(_active_current_processes(current_config))
        if not sorted(set(live_pids)):
            return
        time.sleep(poll_seconds)


def _ensure_output_dir_available(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to launch into non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)


def _checkpoint_from_config(config_path: Path) -> Path | None:
    conf = OmegaConf.load(config_path)
    checkpoint = OmegaConf.select(conf, "training.load_stage_checkpoint", default=None)
    if checkpoint is None or str(checkpoint).strip() == "":
        return None
    return _resolve_path(str(checkpoint))


def _launch_followup(branch: str, gpus: str, python: str) -> dict[str, str]:
    config_path = CONFIGS[branch]
    output_dir = OUTPUT_DIRS[branch]
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing follow-up config: {config_path}")
    checkpoint = _checkpoint_from_config(config_path)
    if checkpoint is not None and not checkpoint.is_file():
        raise FileNotFoundError(f"Configured checkpoint is missing: {checkpoint}")
    _ensure_output_dir_available(output_dir)

    log_path = output_dir / "train.log"
    pid_path = output_dir / "launcher.pid"
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
        "output_dir": str(output_dir),
        "command": " ".join(command),
    }


def _latest_metrics(latest: dict[str, Any]) -> dict[str, Any]:
    fields = [
        ("val_loss", "val/loss"),
        ("val_rho_true", "val/rho_true"),
        ("val_bg_high_leak", "val/bg_high_leak"),
        ("val_bg_wavelet_detail_leak", "val/bg_wavelet_detail_leak"),
        ("val_bg_l1", "val/bg_l1"),
    ]
    metrics: dict[str, Any] = {"epoch": latest.get("epoch", "")}
    for out_key, source_key in fields:
        metrics[out_key] = _float_value(latest, source_key)
    if latest.get("val/bg_ms_l1"):
        metrics["val_bg_ms_l1"] = _float_value(latest, "val/bg_ms_l1")
    return metrics


def dispatch(
    current_run: Path,
    current_config: Path,
    gpus: str,
    python: str,
    poll_seconds: int,
    dry_run: bool,
    skip_wait: bool,
    detail_leak_eps: float,
    rho_decrease_window: int,
    rho_min_delta: float,
    bg_l1_plateau_window: int,
    bg_l1_min_improvement: float,
) -> dict[str, Any]:
    if not skip_wait:
        wait_for_current_run(current_run, current_config, poll_seconds)
    metrics_path, rows = read_validation_rows(current_run)
    branch, branch_info = choose_next_branch(
        rows,
        detail_leak_eps=detail_leak_eps,
        rho_decrease_window=rho_decrease_window,
        rho_min_delta=rho_min_delta,
        bg_l1_plateau_window=bg_l1_plateau_window,
        bg_l1_min_improvement=bg_l1_min_improvement,
    )
    current_stage_checkpoint = current_run / "stage_checkpoints" / "background_last.ckpt"
    if branch in {"to60", "to100"} and not dry_run and not current_stage_checkpoint.is_file():
        raise FileNotFoundError(f"Current background stage checkpoint is missing: {current_stage_checkpoint}")

    launch: dict[str, str] | None = None
    if not dry_run:
        launch = _launch_followup(branch, gpus=gpus, python=python)

    latest = rows[-1]
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "current_run": str(current_run),
        "current_config": str(current_config),
        "metrics_path": str(metrics_path),
        "validation_rows": len(rows),
        "selected_branch": branch,
        "branch_info": branch_info,
        "latest_metrics": _latest_metrics(latest),
        "dry_run": dry_run,
        "launch": launch,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Dispatch the BG-PDR-FM wavelet HC64 follow-up queue.")
    parser.add_argument("--current-run", default="logs/bg_pdr_fm/background_train_wavelet_hc64")
    parser.add_argument("--current-config", default=str(CURRENT_CONFIG.relative_to(REPO_ROOT)))
    parser.add_argument("--gpus", default="4,5,6,7")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--detail-leak-eps", type=float, default=1e-6)
    parser.add_argument("--rho-decrease-window", type=int, default=5)
    parser.add_argument("--rho-min-delta", type=float, default=1e-3)
    parser.add_argument("--bg-l1-plateau-window", type=int, default=5)
    parser.add_argument("--bg-l1-min-improvement", type=float, default=0.002)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-wait", action="store_true")
    args = parser.parse_args()

    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    decision = dispatch(
        current_run=_resolve_path(args.current_run),
        current_config=_resolve_path(args.current_config),
        gpus=args.gpus,
        python=args.python,
        poll_seconds=args.poll_seconds,
        dry_run=args.dry_run,
        skip_wait=args.skip_wait,
        detail_leak_eps=float(args.detail_leak_eps),
        rho_decrease_window=int(args.rho_decrease_window),
        rho_min_delta=float(args.rho_min_delta),
        bg_l1_plateau_window=int(args.bg_l1_plateau_window),
        bg_l1_min_improvement=float(args.bg_l1_min_improvement),
    )
    decision_path = QUEUE_DIR / "decision.json"
    decision_path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
