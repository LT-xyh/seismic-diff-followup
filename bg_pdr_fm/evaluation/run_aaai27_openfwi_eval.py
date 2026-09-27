"""Run AAAI27 benchmark checkpoints on OpenFWI with dataset-aware summaries."""

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


DEFAULT_OUTPUT_ROOT = Path("logs/bg_pdr_fm/aaai27/eval_openfwi_best")
DEFAULT_OPENFWI_ROOT = Path("/public/home/xuyinghao/workspace/datasets/openfwi")
DEFAULT_OPENFWI_LMDB_ROOT = Path("/public/home/xuyinghao/workspace/datasets/openfwi_lmdb")

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
    {
        "name": "bg_pdr_fm",
        "config": "bg_pdr_fm/configs/experiments/aaai27/formal_bg_pdr_fm.yaml",
        "checkpoint_hint": "logs/bg_pdr_fm/aaai27/formal/bg_pdr_fm/checkpoints/last.ckpt",
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


def _read_rows(path: Path, job_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append({"job": job_name, **row})
    return rows


def _selected_jobs(job_names: list[str] | None) -> list[dict[str, str]]:
    if not job_names:
        return JOBS
    requested = {name.strip() for name in job_names if name.strip()}
    known = {str(job["name"]) for job in JOBS}
    unknown = sorted(requested.difference(known))
    if unknown:
        raise ValueError(f"Unknown jobs {unknown}; expected subset of {sorted(known)}.")
    return [job for job in JOBS if str(job["name"]) in requested]


def _read_existing_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _upsert_manifest_entry(manifest: list[dict[str, Any]], entry: dict[str, Any]) -> list[dict[str, Any]]:
    kept = [item for item in manifest if item.get("job") != entry.get("job")]
    kept.append(entry)
    order = {str(job["name"]): idx for idx, job in enumerate(JOBS)}
    return sorted(kept, key=lambda item: order.get(str(item.get("job")), len(order)))


def _collect_available_rows(output_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    mode_rows: list[dict[str, Any]] = []
    dataset_mode_rows: list[dict[str, Any]] = []
    for job in JOBS:
        job_name = str(job["name"])
        output_dir = output_root / job_name
        mode_path = output_dir / "missing_mode_summary.csv"
        dataset_mode_path = output_dir / "dataset_missing_mode_summary.csv"
        if mode_path.is_file():
            rows = _read_rows(mode_path, job_name)
            for row in rows:
                row.setdefault("dataset_name", "OpenFWI")
            mode_rows.extend(rows)
        if dataset_mode_path.is_file():
            dataset_mode_rows.extend(_read_rows(dataset_mode_path, job_name))
    return mode_rows, dataset_mode_rows


def _average(rows: list[dict[str, Any]], group_keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(str(row[key]) for key in group_keys), []).append(row)

    out: list[dict[str, Any]] = []
    for group_key in sorted(groups):
        selected = groups[group_key]
        item: dict[str, Any] = {key: value for key, value in zip(group_keys, group_key, strict=True)}
        item["num_samples"] = sum(int(row["num_samples"]) for row in selected)
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


def _write_global_outputs(output_root: Path, mode_rows: list[dict[str, Any]], dataset_rows: list[dict[str, Any]]) -> None:
    mode_csv = output_root / "openfwi_missing_mode_summary.csv"
    dataset_csv = output_root / "openfwi_dataset_summary.csv"
    dataset_mode_csv = output_root / "openfwi_dataset_missing_mode_summary.csv"
    fieldnames = ["job", "dataset_name", "missing_mode", "num_samples", *SUMMARY_METRIC_KEYS]
    _write_csv(mode_csv, mode_rows, fieldnames)
    _write_csv(dataset_mode_csv, dataset_rows, fieldnames)

    dataset_averages = _average(dataset_rows, ["job", "dataset_name"])
    _write_csv(dataset_csv, dataset_averages, ["job", "dataset_name", "num_samples", *SUMMARY_METRIC_KEYS])

    full_rows = [row for row in mode_rows if row["missing_mode"] == "full"]
    missing_only_rows = _average(
        [row for row in mode_rows if row["missing_mode"] != "full"],
        ["job"],
    )
    lines = [
        "# AAAI27 OpenFWI Evaluation",
        "",
        "In-domain evaluation using the unified benchmark evaluator. OpenFWI subsets are summarized separately.",
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
    for row in sorted(missing_only_rows, key=lambda item: float(item["mae"])):
        lines.append(
            f"| {row['job']} | {float(row['mae']):.6f} | {float(row['rmse']):.6f} | "
            f"{float(row['ssim']):.6f} | {float(row['mae_l']):.6f} | {float(row['mae_h']):.6f} | "
            f"{float(row['inference_time']):.6f} |"
        )
    lines.extend(
        [
            "",
            f"Missing-mode CSV: `{mode_csv}`",
            f"Dataset CSV: `{dataset_csv}`",
            f"Dataset-by-missing-mode CSV: `{dataset_mode_csv}`",
            "",
        ]
    )
    (output_root / "openfwi_summary.md").write_text("\n".join(lines), encoding="utf-8")


def _openfwi_conf(
    conf: Any,
    *,
    checkpoint: Path | None,
    output_dir: Path,
    root_dir: Path,
    lmdb_root: Path,
    max_batches: int | None,
) -> Any:
    OmegaConf.update(conf, "data.name", "openfwi", merge=True)
    OmegaConf.update(conf, "data.root_dir", str(root_dir), merge=True)
    OmegaConf.update(conf, "data.root_candidates", [str(root_dir)], merge=True)
    OmegaConf.update(conf, "data.storage_backend", "lmdb", merge=True)
    OmegaConf.update(conf, "data.lmdb_root", str(lmdb_root), merge=True)
    OmegaConf.update(conf, "evaluation.checkpoint", None if checkpoint is None else str(checkpoint), merge=True)
    OmegaConf.update(conf, "evaluation.output_dir", str(output_dir), merge=True)
    OmegaConf.update(conf, "evaluation.max_batches", max_batches, merge=True)
    OmegaConf.update(conf, "training.devices", 1, merge=True)
    OmegaConf.update(conf, "training.num_workers", 4, merge=True)
    OmegaConf.update(conf, "training.persistent_workers", True, merge=True)
    OmegaConf.update(conf, "training.prefetch_factor", 3, merge=True)
    return conf


def run(
    output_root: Path,
    openfwi_root: Path,
    lmdb_root: Path,
    max_batches: int | None,
    job_names: list[str] | None,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    manifest = _read_existing_manifest(manifest_path)
    for job in _selected_jobs(job_names):
        hint = str(job["checkpoint_hint"])
        checkpoint = None if hint.strip() == "" else _resolve_best_checkpoint(hint)
        output_dir = output_root / str(job["name"])
        print(f"=== Evaluating {job['name']} on OpenFWI ===", flush=True)
        print(f"checkpoint: {checkpoint or 'analytic/no checkpoint'}", flush=True)
        conf = _openfwi_conf(
            load_benchmark_config(str(job["config"])),
            checkpoint=checkpoint,
            output_dir=output_dir,
            root_dir=openfwi_root,
            lmdb_root=lmdb_root,
            max_batches=max_batches,
        )
        summary = run_benchmark_evaluation(conf)
        manifest = _upsert_manifest_entry(
            manifest,
            {
                "job": job["name"],
                "config": job["config"],
                "checkpoint": "" if checkpoint is None else str(checkpoint),
                "output_dir": str(output_dir),
                "summary": summary,
            },
        )
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        mode_rows, dataset_mode_rows = _collect_available_rows(output_root)
        _write_global_outputs(output_root, mode_rows, dataset_mode_rows)
        print(f"=== Finished {job['name']}: {summary['num_samples']} rows ===", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate AAAI27 formal checkpoints on OpenFWI.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--openfwi-root", default=str(DEFAULT_OPENFWI_ROOT))
    parser.add_argument("--lmdb-root", default=str(DEFAULT_OPENFWI_LMDB_ROOT))
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument(
        "--jobs",
        nargs="+",
        default=None,
        help="Optional subset of jobs to run, e.g. --jobs bg_pdr_fm cncs_fm_strong.",
    )
    args = parser.parse_args()
    run(Path(args.output_root), Path(args.openfwi_root), Path(args.lmdb_root), args.max_batches, args.jobs)


if __name__ == "__main__":
    main()
