"""Run a mid-training BG-PDR-FM evaluation at a target completed epoch."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]


def _log(message: str) -> None:
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)


def _pid_alive(pid: int | None) -> bool:
    return pid is not None and Path(f"/proc/{pid}").exists()


def _read_pid(path: Path | None) -> int | None:
    if path is None or not path.is_file():
        return None
    text = path.read_text(encoding="utf-8").strip()
    return int(text) if text else None


def _find_latest_metrics(run_dir: Path) -> Path | None:
    candidates = list((run_dir / "lightning").rglob("metrics.csv"))
    candidates.extend((run_dir / "diagnostics").glob("metrics.csv"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _last_validation_epoch(metrics_path: Path | None) -> int | None:
    if metrics_path is None or not metrics_path.is_file():
        return None
    last_epoch: int | None = None
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if not row.get("val/loss") and row.get("split") != "val":
                continue
            epoch_text = row.get("epoch")
            if not epoch_text:
                continue
            last_epoch = int(float(epoch_text))
    return last_epoch


def _read_means(summary_path: Path) -> dict[str, Any]:
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    means = data.get("means")
    if not isinstance(means, dict):
        raise KeyError(f"summary has no means dict: {summary_path}")
    return means


def _float(means: dict[str, Any], key: str) -> float:
    return float(means[key])


def _write_eval_config(base_config: Path, output_dir: Path, checkpoint_snapshot: Path) -> Path:
    from omegaconf import OmegaConf

    conf = OmegaConf.load(base_config)
    OmegaConf.update(conf, "evaluation.output_dir", str(output_dir), merge=True)
    OmegaConf.update(conf, "evaluation.checkpoints.residual", str(checkpoint_snapshot), merge=True)
    OmegaConf.update(conf, "evaluation.save_arrays", False, merge=True)
    OmegaConf.update(conf, "evaluation.save_panels", True, merge=True)
    eval_config = output_dir / "eval_config.yaml"
    output_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(conf, eval_config)
    return eval_config


def _run_eval(eval_config: Path, gpu: str, python: str, log_path: Path) -> int:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("MPLCONFIGDIR", "/tmp/mpl-bg-pdr-fm-epoch-eval")
    if gpu:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    command = [python, "-m", "bg_pdr_fm.evaluation.evaluate_bg_pdr_fm", "--config", str(eval_config)]
    _log(f"running mid-epoch eval on GPU {gpu or '<unset>'}: {' '.join(command)}")
    with log_path.open("ab", buffering=0) as handle:
        result = subprocess.run(command, cwd=str(REPO_ROOT), env=env, stdout=handle, stderr=subprocess.STDOUT)
    _log(f"mid-epoch eval exited with code {result.returncode}")
    return int(result.returncode)


def _stop_training(pid: int | None) -> None:
    if not _pid_alive(pid):
        return
    assert pid is not None
    try:
        pgid = os.getpgid(pid)
        _log(f"sending SIGTERM to training process group {pgid}")
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except PermissionError:
        _log(f"process-group stop failed; sending SIGTERM to pid {pid}")
        os.kill(pid, signal.SIGTERM)


def _decision(
    means: dict[str, Any],
    baseline: dict[str, Any],
    *,
    min_ssim_delta: float,
    max_mae_l_delta: float,
    max_mae_h_delta: float,
) -> dict[str, Any]:
    deltas = {
        "ssim": _float(means, "ssim") - _float(baseline, "ssim"),
        "mae_l": _float(means, "mae_l") - _float(baseline, "mae_l"),
        "mae_h": _float(means, "mae_h") - _float(baseline, "mae_h"),
    }
    clearly_worse = (
        deltas["ssim"] <= min_ssim_delta
        or deltas["mae_l"] >= max_mae_l_delta
        or deltas["mae_h"] >= max_mae_h_delta
    )
    promising = (
        deltas["ssim"] > 0.0
        or (deltas["mae_l"] < 0.0 and deltas["mae_h"] <= max_mae_h_delta)
    )
    return {
        "deltas": deltas,
        "clearly_worse": clearly_worse,
        "promising": promising,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--pid-file", type=Path, default=None)
    parser.add_argument("--train-pid", type=int, default=None)
    parser.add_argument("--target-completed-epochs", type=int, default=50)
    parser.add_argument("--eval-config", required=True, type=Path)
    parser.add_argument("--baseline-summary", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--eval-gpu", default="3")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--min-ssim-delta", type=float, default=-0.003)
    parser.add_argument("--max-mae-l-delta", type=float, default=0.003)
    parser.add_argument("--max-mae-h-delta", type=float, default=0.001)
    parser.add_argument("--stop-if-worse", action="store_true")
    args = parser.parse_args(argv)

    run_dir = args.run_dir
    train_pid = args.train_pid if args.train_pid is not None else _read_pid(args.pid_file)
    output_dir = args.output_dir or run_dir / f"evaluation_mb10_epoch{args.target_completed_epochs}"
    summary_path = output_dir / "summary.json"
    checkpoint = run_dir / "stage_checkpoints" / "residual_last.ckpt"
    checkpoint_snapshot = output_dir / f"residual_epoch{args.target_completed_epochs}_snapshot.ckpt"
    baseline = _read_means(args.baseline_summary)
    target_last_epoch = max(0, args.target_completed_epochs - 1)

    _log(
        "epoch monitor started: "
        f"run_dir={run_dir}, pid={train_pid}, target_completed_epochs={args.target_completed_epochs}"
    )
    while _pid_alive(train_pid):
        metrics_path = _find_latest_metrics(run_dir)
        last_epoch = _last_validation_epoch(metrics_path)
        _log(f"latest validation epoch={last_epoch if last_epoch is not None else 'NA'}")
        if last_epoch is not None and last_epoch >= target_last_epoch:
            break
        time.sleep(max(1, args.poll_seconds))

    if not checkpoint.is_file():
        raise FileNotFoundError(f"stage checkpoint is missing: {checkpoint}")
    if not summary_path.is_file():
        output_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(checkpoint, checkpoint_snapshot)
        eval_config = _write_eval_config(args.eval_config, output_dir, checkpoint_snapshot)
        eval_code = _run_eval(eval_config, args.eval_gpu, args.python, output_dir / "eval.log")
        if eval_code != 0:
            raise RuntimeError(f"mid-epoch eval failed with code {eval_code}")
    else:
        _log(f"using existing mid-epoch eval summary: {summary_path}")

    means = _read_means(summary_path)
    decision = _decision(
        means,
        baseline,
        min_ssim_delta=args.min_ssim_delta,
        max_mae_l_delta=args.max_mae_l_delta,
        max_mae_h_delta=args.max_mae_h_delta,
    )
    report = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "train_pid": train_pid,
        "target_completed_epochs": args.target_completed_epochs,
        "summary": str(summary_path),
        "baseline_summary": str(args.baseline_summary),
        "means": means,
        "baseline_means": baseline,
        **decision,
        "stop_if_worse": bool(args.stop_if_worse),
    }
    (output_dir / "epoch_eval_decision.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    lines = [
        f"target_completed_epochs: {args.target_completed_epochs}",
        f"summary: {summary_path}",
        f"ssim_delta: {decision['deltas']['ssim']:+.6f}",
        f"mae_l_delta: {decision['deltas']['mae_l']:+.6f}",
        f"mae_h_delta: {decision['deltas']['mae_h']:+.6f}",
        f"clearly_worse: {decision['clearly_worse']}",
        f"promising: {decision['promising']}",
        f"stop_if_worse: {bool(args.stop_if_worse)}",
        "",
    ]
    (output_dir / "epoch_eval_decision.txt").write_text("\n".join(lines), encoding="utf-8")
    if args.stop_if_worse and decision["clearly_worse"]:
        _stop_training(train_pid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
