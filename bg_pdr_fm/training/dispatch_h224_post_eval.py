"""Dispatch h224 post-training formal eval and contrastive checkpoint probe."""

from __future__ import annotations

import argparse
import json
import os
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


def _read_means(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"summary not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    means = data.get("means")
    if not isinstance(means, dict):
        raise KeyError(f"summary has no means dict: {path}")
    return means


def _run(command: list[str], *, gpu: str, log_path: Path) -> int:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("MPLCONFIGDIR", "/tmp/mpl-bg-pdr-fm-post-eval")
    if gpu:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    _log(f"running on GPU {gpu or '<unset>'}: {' '.join(command)}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("ab", buffering=0) as handle:
        result = subprocess.run(command, cwd=str(REPO_ROOT), env=env, stdout=handle, stderr=subprocess.STDOUT)
    _log(f"command exited with code {result.returncode}")
    return int(result.returncode)


def _format_delta(value: float | None) -> str:
    return "NA" if value is None else f"{value:+.6f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--pid-file", type=Path, default=None)
    parser.add_argument("--train-pid", type=int, default=None)
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--formal-config", type=Path, required=True)
    parser.add_argument("--formal-output-dir", type=Path, required=True)
    parser.add_argument("--formal-gpu", default="6")
    parser.add_argument("--current-summary", type=Path, required=True)
    parser.add_argument("--pairwise-config", type=Path, required=True)
    parser.add_argument("--pairwise-output-dir", type=Path, required=True)
    parser.add_argument("--probe-gpu", default="7")
    parser.add_argument("--concat-ssim", type=float, default=0.8678)
    parser.add_argument("--probe-window-low", type=float, default=0.8645)
    parser.add_argument("--probe-window-high", type=float, default=0.8680)
    parser.add_argument("--min-probe-ssim-gain", type=float, default=0.005)
    args = parser.parse_args(argv)

    train_pid = args.train_pid if args.train_pid is not None else _read_pid(args.pid_file)
    _log(f"post-eval dispatcher started for {args.run_dir}; train_pid={train_pid}")
    while _pid_alive(train_pid):
        time.sleep(max(1, args.poll_seconds))
    _log("training process is no longer alive")

    checkpoint = args.run_dir / "stage_checkpoints" / "residual_last.ckpt"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"h224 residual checkpoint missing: {checkpoint}")

    formal_summary = args.formal_output_dir / "summary.json"
    if not formal_summary.is_file():
        code = _run(
            [
                args.python,
                "-m",
                "bg_pdr_fm.evaluation.compare_experiments",
                "--config",
                str(args.formal_config),
            ],
            gpu=args.formal_gpu,
            log_path=args.formal_output_dir / "formal_full_eval.log",
        )
        if code != 0:
            raise RuntimeError(f"formal full eval failed with code {code}")
    else:
        _log(f"formal summary already exists: {formal_summary}")

    formal_means = _read_means(formal_summary)
    formal_ssim = float(formal_means["ssim"])
    status = "formal_passed_concat" if formal_ssim > args.concat_ssim else "formal_below_concat"
    run_probe = args.probe_window_low <= formal_ssim <= args.probe_window_high
    _log(f"h224 formal full SSIM={formal_ssim:.6f}; status={status}; run_probe={run_probe}")

    current_means: dict[str, Any] | None = None
    pairwise_means: dict[str, Any] | None = None
    probe_gain: float | None = None
    if run_probe:
        wait_deadline = time.time() + 3 * 60 * 60
        while not args.current_summary.is_file() and time.time() < wait_deadline:
            _log(f"waiting for current 10-batch summary: {args.current_summary}")
            time.sleep(max(30, args.poll_seconds))
        current_means = _read_means(args.current_summary)
        pairwise_summary = args.pairwise_output_dir / "summary.json"
        if not pairwise_summary.is_file():
            code = _run(
                [
                    args.python,
                    "-m",
                    "bg_pdr_fm.evaluation.evaluate_bg_pdr_fm",
                    "--config",
                    str(args.pairwise_config),
                ],
                gpu=args.probe_gpu,
                log_path=args.run_dir / "evaluation_pairwise_epoch77_after_train.log",
            )
            if code != 0:
                raise RuntimeError(f"pairwise checkpoint probe failed with code {code}")
        else:
            _log(f"pairwise probe summary already exists: {pairwise_summary}")
        pairwise_means = _read_means(pairwise_summary)
        probe_gain = float(pairwise_means["ssim"]) - float(current_means["ssim"])
        _log(f"pairwise probe SSIM delta={probe_gain:+.6f}")

    summary = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(args.run_dir),
        "formal_summary": str(formal_summary),
        "formal_means": formal_means,
        "concat_ssim": args.concat_ssim,
        "status": status,
        "contrastive_probe_ran": run_probe,
        "current_summary": str(args.current_summary),
        "pairwise_summary": str(args.pairwise_output_dir / "summary.json"),
        "pairwise_ssim_delta": probe_gain,
        "pairwise_probe_passed": (
            None if probe_gain is None else probe_gain >= args.min_probe_ssim_gain
        ),
    }
    out = args.run_dir / "post_h224_dispatch_summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "BG-PDR-FM h224 post-eval dispatch",
        f"formal_ssim: {formal_ssim:.6f}",
        f"concat_ssim: {args.concat_ssim:.6f}",
        f"status: {status}",
        f"contrastive_probe_ran: {run_probe}",
        f"pairwise_ssim_delta: {_format_delta(probe_gain)}",
        f"summary_json: {out}",
        "",
    ]
    (args.run_dir / "post_h224_dispatch_summary.txt").write_text("\n".join(lines), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
