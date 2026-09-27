"""Dispatch the 24h follow-up queue for the smooth HC64 background probe."""

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
CURRENT_RUN = REPO_ROOT / "logs" / "bg_pdr_fm" / "background_train_smooth_hc64"
CURRENT_CONFIG = REPO_ROOT / "bg_pdr_fm" / "configs" / "openfwi_lmdb_background_smooth_hc64.yaml"
EVAL_CONFIG = REPO_ROOT / "bg_pdr_fm" / "configs" / "openfwi_lmdb_background_smooth_hc64_eval_mb10.yaml"
EVAL_SUMMARY = CURRENT_RUN / "evaluation_mb10" / "summary.json"
QUEUE_DIR = CURRENT_RUN / "queue"

CONFIGS = {
    "residual_probe": REPO_ROOT / "bg_pdr_fm" / "configs" / "openfwi_lmdb_residual_smooth_hc64_probe.yaml",
    "gate_residual_probe": REPO_ROOT / "bg_pdr_fm" / "configs" / "openfwi_lmdb_residual_smooth_hc64_gate_probe.yaml",
    "to60": REPO_ROOT / "bg_pdr_fm" / "configs" / "openfwi_lmdb_background_smooth_hc64_to60.yaml",
    "smooth24": REPO_ROOT / "bg_pdr_fm" / "configs" / "openfwi_lmdb_background_smooth24_hc64.yaml",
}
OUTPUT_DIRS = {
    "residual_probe": REPO_ROOT / "logs" / "bg_pdr_fm" / "residual_train_smooth_hc64_probe",
    "gate_residual_probe": REPO_ROOT / "logs" / "bg_pdr_fm" / "residual_train_smooth_hc64_gate_probe",
    "to60": REPO_ROOT / "logs" / "bg_pdr_fm" / "background_train_smooth_hc64_to60",
    "smooth24": REPO_ROOT / "logs" / "bg_pdr_fm" / "background_train_smooth24_hc64",
}
BRANCH_CHECKPOINTS = {
    "residual_probe": CURRENT_RUN / "stage_checkpoints" / "background_last.ckpt",
    "gate_residual_probe": CURRENT_RUN / "stage_checkpoints" / "background_last.ckpt",
    "to60": CURRENT_RUN / "stage_checkpoints" / "background_last.ckpt",
    "smooth24": REPO_ROOT / "logs" / "bg_pdr_fm" / "contrastive_train" / "stage_checkpoints" / "contrastive_last.ckpt",
}


def _log(message: str) -> None:
    print(f"[{datetime.now().strftime('%F %T')}] {message}", flush=True)


def _resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


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
    needles = {str(config_path), os.path.relpath(config_path, REPO_ROOT)}
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
    return sorted(set(pids))


def wait_for_current_run(current_run: Path, current_config: Path, poll_seconds: int) -> None:
    pid_paths = [current_run / "launcher.pid", current_run / "train.pid"]
    while True:
        live_pids = [pid for pid in (_read_pid(path) for path in pid_paths) if _pid_exists(pid)]
        live_pids.extend(_active_current_processes(current_config))
        live_pids = sorted(set(live_pids))
        if not live_pids:
            _log("current smooth_hc64 training appears finished")
            return
        _log(f"waiting for current smooth_hc64 training pids: {','.join(str(pid) for pid in live_pids[:8])}")
        time.sleep(poll_seconds)


def _float_value(row: dict[str, Any], key: str) -> float:
    value = row.get(key)
    if value is None or value == "":
        raise KeyError(f"Missing metric {key!r}.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Metric {key!r} is not finite: {value!r}.")
    return result


def _epoch_value(row: dict[str, Any]) -> int:
    value = row.get("epoch")
    if value is None or value == "":
        raise KeyError("Missing metric 'epoch'.")
    return int(float(value))


def read_validation_rows(current_run: Path) -> tuple[Path, list[dict[str, str]]]:
    metric_paths = sorted(
        (current_run / "lightning").rglob("metrics.csv"),
        key=lambda path: path.stat().st_mtime,
    )
    if not metric_paths:
        raise FileNotFoundError(f"No Lightning metrics.csv found under {current_run / 'lightning'}.")
    metrics_path = metric_paths[-1]
    rows: list[dict[str, str]] = []
    with metrics_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if not row.get("val/loss"):
                continue
            try:
                _float_value(row, "val/loss")
            except (KeyError, ValueError):
                continue
            rows.append(row)
    if not rows:
        raise ValueError(f"No validation rows found in {metrics_path}.")
    return metrics_path, rows


def _recently_decreasing(rows: list[dict[str, str]], key: str, window: int, min_delta: float) -> bool:
    if len(rows) < window:
        return False
    values = [_float_value(row, key) for row in rows[-window:]]
    return values[-1] < values[0] - min_delta and values[-1] <= min(values[:-1])


def _recent_plateau(rows: list[dict[str, str]], key: str, window: int, min_improvement: float) -> bool:
    if len(rows) < window:
        return False
    values = [_float_value(row, key) for row in rows[-window:]]
    return values[0] - values[-1] < min_improvement


def run_evaluation(eval_config: Path, eval_summary: Path, eval_gpu: str, python: str, force: bool) -> Path:
    if eval_summary.is_file() and not force:
        _log(f"using existing evaluation summary: {eval_summary}")
        return eval_summary
    checkpoint = CURRENT_RUN / "stage_checkpoints" / "background_last.ckpt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing smooth background checkpoint: {checkpoint}")
    eval_config = _resolve_path(eval_config)
    if not eval_config.is_file():
        raise FileNotFoundError(f"Missing evaluation config: {eval_config}")
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    log_path = QUEUE_DIR / "evaluation_mb10.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(eval_gpu)
    env["PYTHONPATH"] = str(REPO_ROOT)
    env.setdefault("PYTHONUNBUFFERED", "1")
    env = configure_checkpoint_worker_environment(env, QUEUE_DIR / "evaluation_tmp")
    command = [python, "-m", "bg_pdr_fm.evaluation.evaluate_bg_pdr_fm", "--config", str(eval_config)]
    _log(f"running mb10 evaluation on CUDA_VISIBLE_DEVICES={eval_gpu}")
    with log_path.open("ab", buffering=0) as log_file:
        subprocess.run(command, cwd=str(REPO_ROOT), env=env, stdout=log_file, stderr=subprocess.STDOUT, check=True)
    if not eval_summary.is_file():
        raise FileNotFoundError(f"Evaluation completed but summary is missing: {eval_summary}")
    return eval_summary


def read_summary(summary_path: Path) -> dict[str, float]:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    means = data.get("means")
    if not isinstance(means, dict):
        raise ValueError(f"Evaluation summary lacks a means object: {summary_path}")
    keys = ["rho_B", "bg_mae", "mae_l", "mae_h", "ssim", "epsilon_H_B"]
    parsed: dict[str, float] = {}
    for key in keys:
        value = means.get(key)
        if value is None:
            raise KeyError(f"Evaluation summary missing means.{key}: {summary_path}")
        parsed[key] = float(value)
        if not math.isfinite(parsed[key]):
            raise ValueError(f"Evaluation summary means.{key} is not finite: {value!r}")
    return parsed


def choose_branch(
    summary_means: dict[str, float],
    rows: list[dict[str, str]],
    loss_window: int,
    loss_min_delta: float,
    plateau_min_improvement: float,
) -> tuple[str, dict[str, Any]]:
    rho_b = summary_means["rho_B"]
    epsilon_h_b = summary_means["epsilon_H_B"]
    ssim = summary_means["ssim"]
    val_loss_decreasing = _recently_decreasing(rows, "val/loss", loss_window, loss_min_delta)
    val_loss_plateau = _recent_plateau(rows, "val/loss", loss_window, plateau_min_improvement)

    if rho_b < 1.8 and epsilon_h_b <= 0.01 and ssim >= 0.80:
        branch = "residual_probe"
        reason = "rho_B < 1.8, epsilon_H_B <= 0.01, and ssim >= 0.80."
    elif 1.8 <= rho_b < 2.2 and epsilon_h_b <= 0.01 and ssim >= 0.75:
        branch = "gate_residual_probe"
        reason = "rho_B is boundary-usable with low high-frequency leak and acceptable SSIM."
    elif rho_b >= 2.2 and val_loss_decreasing and epsilon_h_b <= 0.01:
        branch = "to60"
        reason = "rho_B is not ready, but recent validation loss is still clearly decreasing."
    elif rho_b >= 2.2 and epsilon_h_b > 0.01:
        branch = "smooth24"
        reason = "rho_B is not ready and high-frequency leakage exceeds the safe proxy threshold."
    elif rho_b >= 2.2 and (val_loss_plateau or ssim < 0.75):
        branch = "smooth24"
        reason = "smooth_hc64 appears plateaued or structurally weak; trying a 24x24 bottleneck."
    else:
        branch = "smooth24"
        reason = "no strict pass/continuation rule matched; treating this as a numeric failure fallback."

    return branch, {
        "reason": reason,
        "val_loss_decreasing": val_loss_decreasing,
        "val_loss_plateau": val_loss_plateau,
        "thresholds": {
            "loss_window": loss_window,
            "loss_min_delta": loss_min_delta,
            "plateau_min_improvement": plateau_min_improvement,
        },
    }


def _ensure_output_dir_available(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to launch into non-empty output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)


def launch_branch(branch: str, gpus: str, python: str, dry_run: bool) -> dict[str, str] | None:
    config_path = CONFIGS[branch]
    output_dir = OUTPUT_DIRS[branch]
    checkpoint = BRANCH_CHECKPOINTS[branch]
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing branch config: {config_path}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint required by branch {branch}: {checkpoint}")
    if dry_run:
        return None
    _ensure_output_dir_available(output_dir)
    log_path = output_dir / "train.log"
    pid_path = output_dir / "launcher.pid"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpus)
    env["PYTHONPATH"] = str(REPO_ROOT)
    env.setdefault("PYTHONUNBUFFERED", "1")
    env = configure_checkpoint_worker_environment(env, output_dir / "tmp")
    command = [python, "-m", "bg_pdr_fm.training.train_bg_pdr_fm", "--config", str(config_path)]
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
        "gpus": str(gpus),
    }


def dispatch(
    current_run: Path,
    current_config: Path,
    eval_config: Path,
    eval_summary: Path,
    gpus: str,
    eval_gpu: str,
    python: str,
    poll_seconds: int,
    dry_run: bool,
    skip_wait: bool,
    skip_eval: bool,
    force_eval: bool,
    loss_window: int,
    loss_min_delta: float,
    plateau_min_improvement: float,
    min_completed_epoch: int,
) -> dict[str, Any]:
    if not skip_wait:
        wait_for_current_run(current_run, current_config, poll_seconds)
    metrics_path, rows = read_validation_rows(current_run)
    latest = rows[-1]
    latest_epoch = _epoch_value(latest)
    if latest_epoch < min_completed_epoch:
        raise RuntimeError(
            f"Current smooth_hc64 run only reached epoch {latest_epoch}; "
            f"refusing to dispatch before epoch {min_completed_epoch}."
        )
    if not skip_eval:
        eval_summary = run_evaluation(eval_config, eval_summary, eval_gpu, python, force=force_eval)
    if not eval_summary.is_file():
        raise FileNotFoundError(f"Evaluation summary is missing: {eval_summary}")

    summary_means = read_summary(eval_summary)
    branch, branch_info = choose_branch(
        summary_means,
        rows,
        loss_window=loss_window,
        loss_min_delta=loss_min_delta,
        plateau_min_improvement=plateau_min_improvement,
    )
    launch = launch_branch(branch, gpus=gpus, python=python, dry_run=dry_run)
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "current_run": str(current_run),
        "current_config": str(current_config),
        "eval_config": str(eval_config),
        "eval_summary": str(eval_summary),
        "metrics_path": str(metrics_path),
        "validation_rows": len(rows),
        "selected_branch": branch,
        "branch_info": branch_info,
        "latest_validation": {
            "epoch": latest.get("epoch", ""),
            "val_loss": _float_value(latest, "val/loss"),
            "val_rho_true": _float_value(latest, "val/rho_true"),
            "val_bg_high_leak": _float_value(latest, "val/bg_high_leak"),
            "val_bg_l1": _float_value(latest, "val/bg_l1"),
        },
        "evaluation_means": summary_means,
        "dry_run": dry_run,
        "launch": launch,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Dispatch the smooth HC64 24h background/residual queue.")
    parser.add_argument("--current-run", default=str(CURRENT_RUN.relative_to(REPO_ROOT)))
    parser.add_argument("--current-config", default=str(CURRENT_CONFIG.relative_to(REPO_ROOT)))
    parser.add_argument("--eval-config", default=str(EVAL_CONFIG.relative_to(REPO_ROOT)))
    parser.add_argument("--eval-summary", default=str(EVAL_SUMMARY.relative_to(REPO_ROOT)))
    parser.add_argument("--gpus", default="0,3,4,7")
    parser.add_argument("--eval-gpu", default="0")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--loss-window", type=int, default=5)
    parser.add_argument("--loss-min-delta", type=float, default=0.002)
    parser.add_argument("--plateau-min-improvement", type=float, default=0.002)
    parser.add_argument("--min-completed-epoch", type=int, default=29)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-wait", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--force-eval", action="store_true")
    args = parser.parse_args()

    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    decision = dispatch(
        current_run=_resolve_path(args.current_run),
        current_config=_resolve_path(args.current_config),
        eval_config=_resolve_path(args.eval_config),
        eval_summary=_resolve_path(args.eval_summary),
        gpus=str(args.gpus),
        eval_gpu=str(args.eval_gpu),
        python=str(args.python),
        poll_seconds=int(args.poll_seconds),
        dry_run=bool(args.dry_run),
        skip_wait=bool(args.skip_wait),
        skip_eval=bool(args.skip_eval),
        force_eval=bool(args.force_eval),
        loss_window=int(args.loss_window),
        loss_min_delta=float(args.loss_min_delta),
        plateau_min_improvement=float(args.plateau_min_improvement),
        min_completed_epoch=int(args.min_completed_epoch),
    )
    decision_path = QUEUE_DIR / "decision.json"
    decision_path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
