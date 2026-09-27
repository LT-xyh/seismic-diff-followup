"""Run background-condition diagnostics and mail a compact report."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import smtplib
import subprocess
import sys
from email.message import EmailMessage
from typing import Any

from omegaconf import OmegaConf

from bg_pdr_fm.runtime import configure_checkpoint_worker_environment

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_WEAK_BG_MAE = 0.0474
MAIL_TO = "xuyinghao@s.upc.edu.cn"


def _env(gpu: int | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["PYTHONUNBUFFERED"] = "1"
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["HIP_VISIBLE_DEVICES"] = str(gpu)
    return configure_checkpoint_worker_environment(env)


def _run(command: list[str], *, log_path: Path, gpu: int | None = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(command, cwd=REPO_ROOT, env=_env(gpu), stdout=log, stderr=subprocess.STDOUT, text=True)
    return int(process.returncode)


def _popen(command: list[str], *, log_path: Path, gpu: int | None = None) -> subprocess.Popen[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("w", encoding="utf-8")
    return subprocess.Popen(command, cwd=REPO_ROOT, env=_env(gpu), stdout=log, stderr=subprocess.STDOUT, text=True)


def _write_config(src: Path, dst: Path, updates: dict[str, Any]) -> Path:
    conf = OmegaConf.load(src)
    for key, value in updates.items():
        OmegaConf.update(conf, key, value, merge=True)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(OmegaConf.to_yaml(conf, resolve=False), encoding="utf-8")
    return dst


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _weak_group(summary: dict[str, Any]) -> dict[str, Any]:
    for group in summary.get("worker_groups", []):
        datasets = str(group.get("datasets", ""))
        if "CurveVelB" in datasets or "CurveFaultB" in datasets:
            return group
    return {}


def _csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _mail(subject: str, body: str) -> bool:
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = MAIL_TO
    msg["To"] = MAIL_TO
    msg.set_content(body)
    with smtplib.SMTP_SSL("smtp.exmail.qq.com", 465, timeout=30) as smtp:
        smtp.login(MAIL_TO, "Xyh@030502")
        smtp.send_message(msg)
    return True


def _summarize_probe(root: Path) -> list[dict[str, Any]]:
    rows = []
    for name in ("linear_1x1", "mlp_1x1"):
        summary = _load_json(root / name / "summary.json")
        if not summary:
            rows.append({"name": name, "status": "missing"})
            continue
        weak_rows = [
            row for row in _csv_rows(root / name / "dataset_summary.csv")
            if row.get("dataset") in {"CurveVelB", "CurveFaultB"}
        ]
        weak_bg = ""
        if weak_rows:
            vals = [float(row["bg_mae"]) for row in weak_rows if row.get("bg_mae")]
            weak_bg = sum(vals) / len(vals)
        rows.append({
            "name": name,
            "status": "completed",
            "num_samples": summary.get("num_samples"),
            "bg_mae": summary.get("means", {}).get("bg_mae"),
            "ssim": summary.get("means", {}).get("ssim"),
            "weak_bg_mae": weak_bg,
            "path": str(root / name),
        })
    return rows


def _summary_markdown(rows: list[dict[str, Any]], failures: list[str], output_dir: Path) -> str:
    lines = ["# Background Condition Diagnostic Summary", ""]
    lines.append("| name | status | samples | bg_mae | ssim | weak_bg_mae | path |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for row in rows:
        lines.append(
            f"| {row.get('name','')} | {row.get('status','')} | {row.get('num_samples','')} | "
            f"{row.get('bg_mae','')} | {row.get('ssim','')} | {row.get('weak_bg_mae','')} | {row.get('path','')} |"
        )
    if failures:
        lines.extend(["", "## Failures", ""])
        lines.extend(f"- {failure}" for failure in failures)
    lines.extend(["", f"Baseline weak bg_mae reference: {BASELINE_WEAK_BG_MAE}", ""])
    text = "\n".join(lines)
    (output_dir / "diagnostic_summary.md").write_text(text, encoding="utf-8")
    with (output_dir / "diagnostic_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["name", "status", "num_samples", "bg_mae", "ssim", "weak_bg_mae", "path"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})
    return text


def _background_eval_config(train_config: Path, output_dir: Path, max_batches: int | None) -> Path:
    conf = OmegaConf.load(train_config)
    OmegaConf.update(conf, "evaluation.output_dir", str(output_dir), merge=True)
    OmegaConf.update(conf, "evaluation.max_batches", max_batches, merge=True)
    OmegaConf.update(conf, "evaluation.save_arrays", False, merge=True)
    OmegaConf.update(conf, "evaluation.save_panels", False, merge=True)
    path = output_dir / "eval_config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(OmegaConf.to_yaml(conf, resolve=False), encoding="utf-8")
    return path


def _run_parallel_eval(config: Path, output_dir: Path, gpus: list[int], max_batches: int | None, expected: int | None) -> int:
    command = [
        sys.executable,
        "-m",
        "bg_pdr_fm.evaluation.dispatch_parallel_eval",
        "--config",
        str(config),
        "--output-dir",
        str(output_dir),
        "--gpus",
        ",".join(str(gpu) for gpu in gpus),
        "--batch-candidates",
        "100,200,400",
        "--poll-seconds",
        "30",
    ]
    if max_batches is not None:
        command += ["--max-batches", str(max_batches)]
    if expected is not None:
        command += ["--expected-samples", str(expected)]
    return _run(command, log_path=output_dir / "dispatch.log")


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    gpus = [int(item) for item in str(args.gpus).split(",") if item.strip()]
    failures: list[str] = []
    rows: list[dict[str, Any]] = []
    print(f"[diagnostic] output_dir={output_dir} gpus={gpus}", flush=True)

    probe_conf = Path("bg_pdr_fm/configs/openfwi_lmdb_bg_feature_probe_linear_mlp.yaml")
    probe_root = output_dir / "linear_mlp_oracle_bg"
    probe_run_conf = _write_config(
        probe_conf,
        output_dir / "configs" / "probe.yaml",
        {
            "diagnostic.output_dir": str(probe_root),
            "evaluation.output_dir": str(probe_root),
            "evaluation.max_batches": 10,
            "training.max_epochs": 1,
            "training.limit_train_batches": 10,
            "training.num_workers": 0,
            "training.persistent_workers": False,
            "training.prefetch_factor": None,
        },
    )
    print("[diagnostic] running linear/mlp feature probe", flush=True)
    rc = _run(
        [sys.executable, "-m", "bg_pdr_fm.evaluation.background_feature_probe", "--config", str(probe_run_conf)],
        log_path=probe_root / "probe.log",
        gpu=gpus[0] if gpus else None,
    )
    if rc != 0:
        failures.append(f"linear/mlp probe failed rc={rc}; see {probe_root / 'probe.log'}")
    rows.extend(_summarize_probe(probe_root))

    train_specs = [
        (
            "raw_rms_bypass",
            Path("bg_pdr_fm/configs/openfwi_lmdb_background_unet_direct_h192_l1l2_raw_rms_bypass_e50.yaml"),
            gpus[1] if len(gpus) > 1 else gpus[0],
        ),
        (
            "numhead_tune",
            Path("bg_pdr_fm/configs/openfwi_lmdb_background_unet_direct_h192_l1l2_numhead_tune_e50.yaml"),
            gpus[2] if len(gpus) > 2 else gpus[0],
        ),
    ]
    pending_jobs: list[dict[str, Any]] = []
    for name, config, gpu in train_specs:
        run_dir = output_dir / name
        train_conf = _write_config(
            config,
            run_dir / "train_config.yaml",
            {
                "training.max_epochs": 3,
                "training.limit_train_batches": 120,
                "training.limit_val_batches": 20,
            },
        )
        print(f"[diagnostic] smoke {name} on gpu {gpu}", flush=True)
        smoke_conf = _write_config(
            config,
            run_dir / "smoke_config.yaml",
            {
                "training.max_epochs": 1,
                "training.limit_train_batches": 1,
                "training.limit_val_batches": 1,
                "training.num_workers": 0,
                "training.persistent_workers": False,
                "training.prefetch_factor": None,
                "training.stage_checkpoint_path": str(run_dir / "smoke_stage_checkpoints"),
                "training.checkpoint.dirpath": str(run_dir / "smoke_lightning" / "checkpoints"),
                "training.logging.log_dir": str(run_dir / "smoke_lightning"),
                "diagnostics.output_dir": str(run_dir / "smoke_diagnostics"),
            },
        )
        rc = _run([sys.executable, "-m", "bg_pdr_fm.training.train_bg_pdr_fm", "--config", str(smoke_conf)], log_path=run_dir / "smoke.log", gpu=gpu)
        if rc != 0:
            failures.append(f"{name} smoke failed rc={rc}; see {run_dir / 'smoke.log'}")
            rows.append({"name": name, "status": "smoke_failed", "path": str(run_dir)})
            continue
        print(f"[diagnostic] train {name} on gpu {gpu}", flush=True)
        process = _popen(
            [sys.executable, "-m", "bg_pdr_fm.training.train_bg_pdr_fm", "--config", str(train_conf)],
            log_path=run_dir / "train.log",
            gpu=gpu,
        )
        pending_jobs.append({"name": name, "run_dir": run_dir, "train_conf": train_conf, "gpu": gpu, "process": process})

    for job in pending_jobs:
        rc = int(job["process"].wait())
        name = str(job["name"])
        run_dir = Path(job["run_dir"])
        train_conf = Path(job["train_conf"])
        gpu = int(job["gpu"])
        if rc != 0:
            failures.append(f"{name} train failed rc={rc}; see {run_dir / 'train.log'}")
            rows.append({"name": name, "status": "train_failed", "path": str(run_dir)})
            continue
        print(f"[diagnostic] eval mb10 {name} on gpu {gpu}", flush=True)
        eval_dir = run_dir / "eval_mb10_parallel"
        eval_conf = _background_eval_config(train_conf, eval_dir, max_batches=10)
        rc = _run_parallel_eval(eval_conf, eval_dir, [gpu], max_batches=10, expected=None)
        summary = _load_json(eval_dir / "summary.json")
        if rc != 0 or not summary:
            failures.append(f"{name} mb10 eval failed rc={rc}; see {eval_dir}")
            rows.append({"name": name, "status": "eval_failed", "path": str(run_dir)})
            continue
        weak = _weak_group(summary)
        weak_bg = weak.get("bg_mae", "")
        rows.append({
            "name": name,
            "status": summary.get("status", "completed"),
            "num_samples": summary.get("num_samples"),
            "bg_mae": summary.get("means", {}).get("bg_mae"),
            "ssim": summary.get("means", {}).get("ssim"),
            "weak_bg_mae": weak_bg,
            "path": str(eval_dir),
        })
        if isinstance(weak_bg, (int, float)) and float(weak_bg) < BASELINE_WEAK_BG_MAE:
            print(f"[diagnostic] full eval {name}; weak_bg_mae={weak_bg}", flush=True)
            full_dir = run_dir / "eval_full_parallel"
            full_conf = _background_eval_config(train_conf, full_dir, max_batches=None)
            rc = _run_parallel_eval(full_conf, full_dir, gpus[: min(len(gpus), 3)], max_batches=None, expected=33600)
            full_summary = _load_json(full_dir / "summary.json")
            if rc == 0 and full_summary:
                weak_full = _weak_group(full_summary)
                rows.append({
                    "name": f"{name}_full",
                    "status": full_summary.get("status", "completed"),
                    "num_samples": full_summary.get("num_samples"),
                    "bg_mae": full_summary.get("means", {}).get("bg_mae"),
                    "ssim": full_summary.get("means", {}).get("ssim"),
                    "weak_bg_mae": weak_full.get("bg_mae", ""),
                    "path": str(full_dir),
                })
            else:
                failures.append(f"{name} full eval failed rc={rc}; see {full_dir}")

    rms_row = next((row for row in rows if row.get("name") == "raw_rms_bypass"), {})
    rms_improved = isinstance(rms_row.get("weak_bg_mae"), (int, float)) and float(rms_row["weak_bg_mae"]) < BASELINE_WEAK_BG_MAE
    all_config = Path("bg_pdr_fm/configs/openfwi_lmdb_background_unet_direct_h192_l1l2_raw_all_bypass_e50.yaml")
    all_run_dir = output_dir / "raw_all_bypass"
    if args.run_all_bypass or rms_improved:
        all_train_conf = _write_config(
            all_config,
            all_run_dir / "train_config.yaml",
            {
                "training.max_epochs": 3,
                "training.limit_train_batches": 120,
                "training.limit_val_batches": 20,
            },
        )
        print(f"[diagnostic] train raw_all_bypass on gpu {gpus[0]}", flush=True)
        rc = _run([sys.executable, "-m", "bg_pdr_fm.training.train_bg_pdr_fm", "--config", str(all_train_conf)], log_path=all_run_dir / "train.log", gpu=gpus[0])
        if rc == 0:
            all_eval_dir = all_run_dir / "eval_mb10_parallel"
            all_eval_conf = _background_eval_config(all_train_conf, all_eval_dir, max_batches=10)
            rc = _run_parallel_eval(all_eval_conf, all_eval_dir, [gpus[0]], max_batches=10, expected=None)
            summary = _load_json(all_eval_dir / "summary.json")
            weak = _weak_group(summary)
            rows.append({
                "name": "raw_all_bypass",
                "status": summary.get("status", "missing") if summary else "missing",
                "num_samples": summary.get("num_samples") if summary else "",
                "bg_mae": summary.get("means", {}).get("bg_mae") if summary else "",
                "ssim": summary.get("means", {}).get("ssim") if summary else "",
                "weak_bg_mae": weak.get("bg_mae", ""),
                "path": str(all_eval_dir),
            })
        else:
            failures.append(f"raw_all_bypass train failed rc={rc}; see {all_run_dir / 'train.log'}")

    report = _summary_markdown(rows, failures, output_dir)
    mail_ok = False
    try:
        _mail("BG-PDR-FM background condition diagnostics completed", report)
        mail_ok = True
    except Exception as exc:  # noqa: BLE001 - report mail failure in state.
        failures.append(f"mail failed: {exc}")
        _summary_markdown(rows, failures, output_dir)
    state = {"rows": rows, "failures": failures, "mail_sent": mail_ok}
    (output_dir / "diagnostic_state.json").write_text(json.dumps(state, indent=2), encoding="utf-8")
    return state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="logs/bg_pdr_fm/bg_feature_diagnostic/background_condition_diagnostics")
    parser.add_argument("--gpus", default="0,4,5")
    parser.add_argument("--run-all-bypass", action="store_true")
    args = parser.parse_args(argv)
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
