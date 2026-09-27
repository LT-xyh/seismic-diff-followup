"""Run AAAI27 benchmark checkpoints on Marmousi with the unified evaluator."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from statistics import mean
from typing import Any

from omegaconf import OmegaConf

from bg_pdr_fm.evaluation.compare_experiments import SUMMARY_METRIC_KEYS, run_benchmark_evaluation
from bg_pdr_fm.training.benchmark_config import load_benchmark_config


DEFAULT_OUTPUT_ROOT = Path("logs/bg_pdr_fm/aaai27/eval_marmousi_best")
DEFAULT_MARMOUSI_ROOT = Path("/public/home/xuyinghao/workspace/datasets/marmousi")

JOBS = [
    {
        "name": "smooth_dix",
        "config": "bg_pdr_fm/configs/experiments/aaai27/formal_smooth_dix.yaml",
        "checkpoint_hint": "",
    },
    {
        "name": "sv_inv_net",
        "config": "bg_pdr_fm/configs/experiments/aaai27/formal_sv_inv_net.yaml",
        "checkpoint_hint": "logs/bg_pdr_fm/aaai27/formal/sv_inv_net/checkpoints/last.ckpt",
    },
    {
        "name": "adapted_inversion_net",
        "config": "bg_pdr_fm/configs/experiments/aaai27/formal_adapted_inversion_net.yaml",
        "checkpoint_hint": "logs/bg_pdr_fm/aaai27/formal/adapted_inversion_net_official_e100/checkpoints/last.ckpt",
    },
    {
        "name": "adapted_upfwi",
        "config": "bg_pdr_fm/configs/experiments/aaai27/formal_adapted_upfwi.yaml",
        "checkpoint_hint": "logs/bg_pdr_fm/aaai27/formal/adapted_upfwi_official_e100/checkpoints/last.ckpt",
    },
    {
        "name": "velocity_gan",
        "config": "bg_pdr_fm/configs/experiments/aaai27/formal_velocity_gan.yaml",
        "checkpoint_hint": "logs/bg_pdr_fm/aaai27/formal/velocity_gan_official_e100/checkpoints/last.ckpt",
    },
    {
        "name": "conditional_ddpm",
        "config": "bg_pdr_fm/configs/experiments/aaai27/formal_conditional_ddpm.yaml",
        "checkpoint_hint": "logs/bg_pdr_fm/aaai27/formal/conditional_ddpm/checkpoints/last.ckpt",
    },
    {
        "name": "adapted_gfi",
        "config": "bg_pdr_fm/configs/experiments/aaai27/formal_adapted_gfi_multimodal.yaml",
        "checkpoint_hint": "logs/bg_pdr_fm/aaai27/formal/adapted_gfi_multimodal/checkpoints/last.ckpt",
    },
    {
        "name": "adapted_auto_linear",
        "config": "bg_pdr_fm/configs/experiments/aaai27/formal_adapted_auto_linear_multimodal.yaml",
        "checkpoint_hint": "logs/bg_pdr_fm/aaai27/formal/adapted_auto_linear_multimodal/checkpoints/last.ckpt",
    },
]
METHODS = JOBS


def _resolve_best_checkpoint(checkpoint_hint: str) -> Path:
    hint = Path(checkpoint_hint)
    checkpoint_dir = hint.parent if hint.suffix == ".ckpt" else hint
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    candidates: list[tuple[float, int, Path]] = []
    for ckpt in checkpoint_dir.glob("*.ckpt"):
        if ckpt.name.startswith("last"):
            continue
        match = re.search(r"-(\d+)-loss([0-9]+(?:\.[0-9]+)?)\.ckpt$", ckpt.name)
        if match:
            candidates.append((float(match.group(2)), int(match.group(1)), ckpt))
    if candidates:
        return min(candidates, key=lambda item: (item[0], -item[1]))[2]
    last = checkpoint_dir / "last.ckpt"
    if last.is_file():
        return last
    raise FileNotFoundError(f"No checkpoint candidates found under {checkpoint_dir}")


def _finite_float(value: str) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    if x != x:
        return None
    return x


def _read_dataset_mode_rows(path: Path, job_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append({"job": job_name, **row})
    return rows


def _average_rows(rows: list[dict[str, Any]], *, missing_only: bool) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for job in sorted({str(row["job"]) for row in rows}):
        selected = [row for row in rows if row["job"] == job and (not missing_only or row["missing_mode"] != "full")]
        item: dict[str, Any] = {
            "job": job,
            "dataset_name": "Marmousi",
            "missing_mode": "missing_only_average" if missing_only else "all_modes_average",
            "num_samples": sum(int(row["num_samples"]) for row in selected),
        }
        for key in SUMMARY_METRIC_KEYS:
            values = [x for row in selected if (x := _finite_float(str(row.get(key, "")))) is not None]
            item[key] = mean(values) if values else ""
        out.append(item)
    return out


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _dataset_average_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for job in sorted({str(row["job"]) for row in rows}):
        selected = [row for row in rows if row["job"] == job]
        item: dict[str, Any] = {
            "job": job,
            "dataset_name": "Marmousi",
            "num_samples": sum(int(row["num_samples"]) for row in selected),
        }
        for key in SUMMARY_METRIC_KEYS:
            values = [x for row in selected if (x := _finite_float(str(row.get(key, "")))) is not None]
            item[key] = mean(values) if values else ""
        out.append(item)
    return out


def _collect_available_rows(output_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for job in JOBS:
        path = output_root / str(job["name"]) / "dataset_missing_mode_summary.csv"
        if path.is_file():
            rows.extend(_read_dataset_mode_rows(path, str(job["name"])))
    return rows


def _write_global_outputs(output_root: Path, rows: list[dict[str, Any]]) -> None:
    csv_path = output_root / "marmousi_missing_mode_summary.csv"
    dataset_csv_path = output_root / "marmousi_dataset_summary.csv"
    dataset_mode_csv_path = output_root / "marmousi_dataset_missing_mode_summary.csv"
    fieldnames = ["job", "dataset_name", "missing_mode", "num_samples", *SUMMARY_METRIC_KEYS]
    _write_csv(csv_path, rows, fieldnames)
    _write_csv(dataset_mode_csv_path, rows, fieldnames)
    _write_csv(
        dataset_csv_path,
        _dataset_average_rows(rows),
        ["job", "dataset_name", "num_samples", *SUMMARY_METRIC_KEYS],
    )

    full_rows = [row for row in rows if row["missing_mode"] == "full"]
    missing_rows = _average_rows(rows, missing_only=True)
    lines = [
        "# AAAI27 Marmousi Evaluation",
        "",
        "External-domain evaluation using the unified benchmark evaluator. Do not average with OpenFWI.",
        "",
        "## Full Modality",
        "",
        "| Method | MAE | RMSE | SSIM | MAE_L | MAE_H | Inference Time |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(full_rows, key=lambda item: float(item["mae"])):
        lines.append(
            f"| {row['job']} | {float(row['mae']):.6f} | {float(row['rmse']):.6f} | "
            f"{float(row['ssim']):.6f} | {float(row['mae_l']):.6f} | {float(row['mae_h']):.6f} | "
            f"{float(row['inference_time']):.6f} |"
        )
    lines.extend(
        [
            "",
            "## Missing-Only Average",
            "",
            "| Method | MAE | RMSE | SSIM | MAE_L | MAE_H | Inference Time |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in sorted(missing_rows, key=lambda item: float(item["mae"])):
        lines.append(
            f"| {row['job']} | {float(row['mae']):.6f} | {float(row['rmse']):.6f} | "
            f"{float(row['ssim']):.6f} | {float(row['mae_l']):.6f} | {float(row['mae_h']):.6f} | "
            f"{float(row['inference_time']):.6f} |"
        )
    lines.extend(
        [
            "",
            f"Missing-mode CSV: `{csv_path}`",
            f"Dataset CSV: `{dataset_csv_path}`",
            f"Dataset-by-missing-mode CSV: `{dataset_mode_csv_path}`",
            "",
        ]
    )
    (output_root / "marmousi_summary.md").write_text("\n".join(lines), encoding="utf-8")


def _marmousi_conf(conf: Any, *, checkpoint: Path | None, output_dir: Path, root_dir: Path, max_batches: int | None) -> Any:
    OmegaConf.update(conf, "data.name", "marmousi", merge=True)
    OmegaConf.update(conf, "data.root_dir", str(root_dir), merge=True)
    OmegaConf.update(conf, "data.root_candidates", [str(root_dir)], merge=True)
    OmegaConf.update(conf, "data.openfwi_datasets", [], merge=True)
    OmegaConf.update(conf, "data.storage_backend", "npy", merge=True)
    OmegaConf.update(conf, "data.lmdb_root", "", merge=True)
    OmegaConf.update(conf, "evaluation.checkpoint", None if checkpoint is None else str(checkpoint), merge=True)
    OmegaConf.update(conf, "evaluation.output_dir", str(output_dir), merge=True)
    OmegaConf.update(conf, "evaluation.max_batches", max_batches, merge=True)
    OmegaConf.update(conf, "training.devices", 1, merge=True)
    OmegaConf.update(conf, "training.num_workers", 2, merge=True)
    OmegaConf.update(conf, "training.persistent_workers", True, merge=True)
    OmegaConf.update(conf, "training.prefetch_factor", 2, merge=True)
    return conf


def run(output_root: Path, marmousi_root: Path, max_batches: int | None) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    all_rows: list[dict[str, Any]] = []
    for job in JOBS:
        hint = str(job["checkpoint_hint"])
        checkpoint = None if hint.strip() == "" else _resolve_best_checkpoint(hint)
        output_dir = output_root / str(job["name"])
        print(f"=== Evaluating {job['name']} on Marmousi ===", flush=True)
        print(f"checkpoint: {checkpoint or 'analytic/no checkpoint'}", flush=True)
        conf = _marmousi_conf(
            load_benchmark_config(str(job["config"])),
            checkpoint=checkpoint,
            output_dir=output_dir,
            root_dir=marmousi_root,
            max_batches=max_batches,
        )
        summary = run_benchmark_evaluation(conf)
        all_rows = _collect_available_rows(output_root)
        manifest.append(
            {
                "job": job["name"],
                "config": job["config"],
                "checkpoint": "" if checkpoint is None else str(checkpoint),
                "output_dir": str(output_dir),
                "summary": summary,
            }
        )
        (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        _write_global_outputs(output_root, all_rows)
        print(f"=== Finished {job['name']}: {summary['num_samples']} rows ===", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate AAAI27 formal checkpoints on Marmousi.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--marmousi-root", default=str(DEFAULT_MARMOUSI_ROOT))
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--summarize-only", action="store_true", help="Rebuild global CSV/Markdown files from existing per-method outputs.")
    args = parser.parse_args()
    output_root = Path(args.output_root)
    if args.summarize_only:
        rows = _collect_available_rows(output_root)
        if not rows:
            raise FileNotFoundError(f"No per-method dataset summaries found under {output_root}")
        _write_global_outputs(output_root, rows)
        return
    run(output_root, Path(args.marmousi_root), args.max_batches)


if __name__ == "__main__":
    main()
