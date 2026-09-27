"""Monitor a BG-PDR-FM run, optionally evaluate it, and send a mail summary."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import smtplib
import ssl
import subprocess
import sys
import time
from datetime import datetime
from email.message import EmailMessage
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]


def _log(message: str) -> None:
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)


def _pid_alive(pid: int | None) -> bool:
    return pid is not None and Path(f"/proc/{pid}").exists()


def _read_int_file(path: Path) -> int | None:
    if not path.is_file():
        return None
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return None
    return int(text)


def _find_latest_metrics(run_dir: Path) -> Path | None:
    metrics = sorted((run_dir / "lightning").rglob("metrics.csv"), key=lambda item: item.stat().st_mtime)
    return metrics[-1] if metrics else None


def _read_last_val(metrics_path: Path | None) -> dict[str, str] | None:
    if metrics_path is None or not metrics_path.is_file():
        return None
    last: dict[str, str] | None = None
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("val/loss"):
                last = row
    return last


def _read_max_epochs(config_path: Path | None) -> int | None:
    if config_path is None or not config_path.is_file():
        return None
    match = re.search(r"(?m)^\s*max_epochs:\s*(\d+)\s*$", config_path.read_text(encoding="utf-8"))
    return int(match.group(1)) if match else None


def _read_summary(summary_path: Path) -> dict[str, Any] | None:
    if not summary_path.is_file():
        return None
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    means = data.get("means")
    return means if isinstance(means, dict) else None


def _run_eval(eval_config: Path, eval_gpu: str, python: str, log_path: Path) -> int:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["PYTHONUNBUFFERED"] = "1"
    if eval_gpu:
        env["CUDA_VISIBLE_DEVICES"] = eval_gpu
    command = [python, "-m", "bg_pdr_fm.evaluation.evaluate_bg_pdr_fm", "--config", str(eval_config)]
    _log(f"running evaluation: CUDA_VISIBLE_DEVICES={eval_gpu or '<unset>'} {' '.join(command)}")
    with log_path.open("ab", buffering=0) as handle:
        result = subprocess.run(command, cwd=str(REPO_ROOT), env=env, stdout=handle, stderr=subprocess.STDOUT)
    _log(f"evaluation exited with code {result.returncode}")
    return int(result.returncode)


def _format_float(value: Any) -> str:
    try:
        return f"{float(value):.6f}"
    except (TypeError, ValueError):
        return "NA"


def _build_body(
    *,
    run_dir: Path,
    train_pid: int | None,
    max_epochs: int | None,
    last_val: dict[str, str] | None,
    summary: dict[str, Any] | None,
    checkpoint: Path,
    eval_summary: Path,
    eval_exit_code: int | None,
) -> tuple[str, str]:
    last_epoch = None
    val_loss = "NA"
    train_loss = "NA"
    if last_val is not None:
        epoch_text = last_val.get("epoch")
        if epoch_text:
            last_epoch = int(float(epoch_text))
        val_loss = _format_float(last_val.get("val/loss"))
        train_loss = _format_float(last_val.get("train/loss"))

    if last_epoch is not None and max_epochs is not None and last_epoch + 1 < max_epochs:
        status = "ENDED_EARLY"
    elif checkpoint.is_file():
        status = "FINISHED"
    else:
        status = "NO_CHECKPOINT"
    if eval_exit_code not in {None, 0}:
        status = f"{status}_EVAL_FAILED"

    lines = [
        f"BG-PDR-FM run monitor status: {status}",
        f"time: {datetime.now().isoformat(timespec='seconds')}",
        f"run_dir: {run_dir}",
        f"train_pid: {train_pid if train_pid is not None else 'NA'}",
        f"checkpoint: {checkpoint} ({'exists' if checkpoint.is_file() else 'missing'})",
        f"max_epochs: {max_epochs if max_epochs is not None else 'NA'}",
        f"last_val_epoch: {last_epoch if last_epoch is not None else 'NA'}",
        f"val/loss: {val_loss}",
        f"train/loss: {train_loss}",
        f"evaluation_summary: {eval_summary} ({'exists' if eval_summary.is_file() else 'missing'})",
    ]
    if eval_exit_code is not None:
        lines.append(f"evaluation_exit_code: {eval_exit_code}")
    if summary is not None:
        keys = ["mae", "rmse", "mse", "ssim", "mae_l", "mae_h", "bg_mae", "rho_B", "epsilon_H_B"]
        lines.append("")
        lines.append("evaluation means:")
        for key in keys:
            if key in summary:
                lines.append(f"  {key}: {_format_float(summary.get(key))}")
    return status, "\n".join(lines) + "\n"


def _read_smtp_config(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        host = os.environ.get("BG_PDR_FM_SMTP_HOST", "").strip()
        username = os.environ.get("BG_PDR_FM_SMTP_USERNAME", "").strip()
        password = os.environ.get("BG_PDR_FM_SMTP_PASSWORD", "")
        if not host and not username and not password:
            return None
        if not host or not username or not password:
            raise KeyError(
                "SMTP environment requires BG_PDR_FM_SMTP_HOST, "
                "BG_PDR_FM_SMTP_USERNAME, and BG_PDR_FM_SMTP_PASSWORD."
            )
        port = int(os.environ.get("BG_PDR_FM_SMTP_PORT", "465"))
        ssl_text = os.environ.get("BG_PDR_FM_SMTP_SSL", "1").strip().lower()
        return {
            "host": host,
            "port": port,
            "username": username,
            "password": password,
            "from": os.environ.get("BG_PDR_FM_SMTP_FROM", username),
            "ssl": ssl_text not in {"0", "false", "no", "off"},
        }
    if not path.is_file():
        raise FileNotFoundError(f"SMTP config not found: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    required = ["host", "port", "username", "password"]
    missing = [key for key in required if not data.get(key)]
    if missing:
        raise KeyError(f"SMTP config missing keys: {', '.join(missing)}")
    return data


def _send_smtp_mail(smtp_conf: dict[str, Any], mail_to: str, subject: str, body: str) -> int:
    message = EmailMessage()
    sender = str(smtp_conf.get("from") or smtp_conf["username"])
    message["Subject"] = subject
    message["From"] = sender
    message["To"] = mail_to
    message.set_content(body)
    host = str(smtp_conf["host"])
    port = int(smtp_conf["port"])
    username = str(smtp_conf["username"])
    password = str(smtp_conf["password"])
    use_ssl = bool(smtp_conf.get("ssl", port == 465))
    context = ssl.create_default_context()
    _log(f"sending SMTP mail to {mail_to} via {host}:{port}")
    if use_ssl:
        with smtplib.SMTP_SSL(host, port, timeout=60, context=context) as smtp:
            smtp.login(username, password)
            smtp.send_message(message)
    else:
        with smtplib.SMTP(host, port, timeout=60) as smtp:
            smtp.starttls(context=context)
            smtp.login(username, password)
            smtp.send_message(message)
    _log("SMTP mail sent")
    return 0


def _send_mail(mail_to: str, subject: str, body: str, smtp_conf: dict[str, Any] | None) -> int:
    if not mail_to:
        _log("mail recipient is empty; skipping mail")
        return 0
    if smtp_conf is not None:
        return _send_smtp_mail(smtp_conf, mail_to, subject, body)
    _log(f"sending mail to {mail_to}: {subject}")
    result = subprocess.run(
        ["mail", "-s", subject, mail_to],
        input=body,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    output = result.stdout.strip()
    if output:
        for line in output.splitlines():
            _log(f"mail output: {line}")
    failed_markers = ("postdrop:", "sendmail:", "No such file or directory", "Connection refused")
    if result.returncode == 0 and any(marker in output for marker in failed_markers):
        _log("mail command reported queue/transport errors despite exit code 0")
        return 1
    _log(f"mail exited with code {result.returncode}")
    return int(result.returncode)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--train-pid", type=int, default=None)
    parser.add_argument("--pid-file", type=Path, default=None)
    parser.add_argument("--no-wait", action="store_true")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--eval-config", type=Path, default=None)
    parser.add_argument("--eval-output-dir", type=Path, default=None)
    parser.add_argument("--eval-gpu", default="")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--mail-to", required=True)
    parser.add_argument("--smtp-config", type=Path, default=None)
    parser.add_argument("--poll-seconds", type=int, default=300)
    args = parser.parse_args(argv)

    run_dir = args.run_dir
    train_pid = args.train_pid
    if train_pid is None and args.pid_file is not None:
        train_pid = _read_int_file(args.pid_file)
    if train_pid is None and not args.no_wait:
        parser.error("provide --train-pid/--pid-file, or pass --no-wait for post-run mail mode")
    _log(f"monitor started for {run_dir}; pid={train_pid}")
    if train_pid is None:
        _log("--no-wait set; generating report immediately")
    else:
        while _pid_alive(train_pid):
            time.sleep(max(1, args.poll_seconds))
        _log("training process is no longer alive")

    eval_exit_code: int | None = None
    eval_summary = (args.eval_output_dir or run_dir / "evaluation_mb10") / "summary.json"
    checkpoint = run_dir / "stage_checkpoints" / "residual_last.ckpt"
    if args.eval_config is not None and checkpoint.is_file() and not eval_summary.is_file():
        eval_log = run_dir / "evaluation_after_train.log"
        eval_exit_code = _run_eval(args.eval_config, args.eval_gpu, args.python, eval_log)

    metrics_path = _find_latest_metrics(run_dir)
    last_val = _read_last_val(metrics_path)
    max_epochs = _read_max_epochs(args.config or args.eval_config)
    summary = _read_summary(eval_summary)
    status, body = _build_body(
        run_dir=run_dir,
        train_pid=train_pid,
        max_epochs=max_epochs,
        last_val=last_val,
        summary=summary,
        checkpoint=checkpoint,
        eval_summary=eval_summary,
        eval_exit_code=eval_exit_code,
    )
    report_path = run_dir / "mail_completion_summary.txt"
    report_path.write_text(body, encoding="utf-8")
    smtp_conf = _read_smtp_config(args.smtp_config)
    mail_status = _send_mail(args.mail_to, f"[BG-PDR-FM] {status}: {run_dir.name}", body, smtp_conf)
    return 0 if mail_status == 0 else mail_status


if __name__ == "__main__":
    raise SystemExit(main())
