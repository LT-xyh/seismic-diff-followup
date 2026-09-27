"""Run explicit OpenFWI missing-modality diagnostics for AAAI27 methods."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from omegaconf import OmegaConf

from bg_pdr_fm.evaluation.compare_experiments import SUMMARY_METRIC_KEYS, run_benchmark_evaluation
from bg_pdr_fm.evaluation.missing_modalities import MissingModalityProtocol
from bg_pdr_fm.training.benchmark_config import load_benchmark_config


DEFAULT_OUTPUT_ROOT = Path("logs/bg_pdr_fm/aaai27/eval_missing_openfwi")
DEFAULT_OPENFWI_ROOT = Path("/public/home/xuyinghao/workspace/datasets/openfwi")
DEFAULT_OPENFWI_LMDB_ROOT = Path("/public/home/xuyinghao/workspace/datasets/openfwi_lmdb")
EXPECTED_OPENFWI_SAMPLES_PER_MODE = 33600


@dataclass(frozen=True)
class MissingBenchmarkJob:
    name: str
    config: str
    checkpoint_hint: str = ""


JOBS: tuple[MissingBenchmarkJob, ...] = (
    MissingBenchmarkJob("smooth_dix", "bg_pdr_fm/configs/experiments/aaai27/formal_smooth_dix.yaml"),
    MissingBenchmarkJob(
        "sv_inv_net",
        "bg_pdr_fm/configs/experiments/aaai27/formal_sv_inv_net.yaml",
        "logs/bg_pdr_fm/aaai27/formal/sv_inv_net/checkpoints/last.ckpt",
    ),
    MissingBenchmarkJob(
        "adapted_inversion_net",
        "bg_pdr_fm/configs/experiments/aaai27/formal_adapted_inversion_net.yaml",
        "logs/bg_pdr_fm/aaai27/formal/adapted_inversion_net/checkpoints/last.ckpt",
    ),
    MissingBenchmarkJob(
        "velocity_gan",
        "bg_pdr_fm/configs/experiments/aaai27/formal_velocity_gan.yaml",
        "logs/bg_pdr_fm/aaai27/formal/velocity_gan/checkpoints/last.ckpt",
    ),
    MissingBenchmarkJob(
        "conditional_ddpm",
        "bg_pdr_fm/configs/experiments/aaai27/formal_conditional_ddpm.yaml",
        "logs/bg_pdr_fm/aaai27/formal/conditional_ddpm/checkpoints/last.ckpt",
    ),
    MissingBenchmarkJob(
        "adapted_gfi",
        "bg_pdr_fm/configs/experiments/aaai27/formal_adapted_gfi_multimodal.yaml",
        "logs/bg_pdr_fm/aaai27/formal/adapted_gfi_multimodal/checkpoints/last.ckpt",
    ),
    MissingBenchmarkJob(
        "adapted_auto_linear",
        "bg_pdr_fm/configs/experiments/aaai27/formal_adapted_auto_linear_multimodal.yaml",
        "logs/bg_pdr_fm/aaai27/formal/adapted_auto_linear_multimodal/checkpoints/last.ckpt",
    ),
)


def _selected_jobs(job_names: list[str] | None) -> list[MissingBenchmarkJob]:
    if not job_names:
        return list(JOBS)
    requested = {name.strip() for name in job_names if name.strip()}
    known = {job.name for job in JOBS}
    unknown = sorted(requested.difference(known))
    if unknown:
        raise ValueError(f"Unknown jobs {unknown}; expected subset of {sorted(known)}.")
    return [job for job in JOBS if job.name in requested]


def _resolve_best_checkpoint(checkpoint_hint: str) -> Path | None:
    if not checkpoint_hint.strip():
        return None
    hint = Path(checkpoint_hint)
    checkpoint_dir = hint.parent if hint.suffix == ".ckpt" else hint
    if not checkpoint_dir.is_dir():
        return None
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
    return last if last.is_file() else None


def apply_openfwi_missing_overrides(
    conf: Any,
    *,
    checkpoint: Path | None,
    output_dir: Path,
    openfwi_root: Path,
    lmdb_root: Path,
    max_batches: int | None,
    protocol: MissingModalityProtocol,
    num_workers: int | None = None,
) -> Any:
    OmegaConf.update(conf, "data.name", "openfwi", merge=True)
    OmegaConf.update(conf, "data.root_dir", str(openfwi_root), merge=True)
    OmegaConf.update(conf, "data.root_candidates", [str(openfwi_root)], merge=True)
    OmegaConf.update(conf, "data.storage_backend", "lmdb", merge=True)
    OmegaConf.update(conf, "data.lmdb_root", str(lmdb_root), merge=True)
    OmegaConf.update(conf, "evaluation.checkpoint", None if checkpoint is None else str(checkpoint), merge=True)
    OmegaConf.update(conf, "evaluation.output_dir", str(output_dir), merge=True)
    OmegaConf.update(conf, "evaluation.max_batches", max_batches, merge=True)
    OmegaConf.update(conf, "evaluation.missing_modes", protocol.mode_names, merge=True)
    OmegaConf.update(conf, "training.devices", 1, merge=True)
    workers = num_workers
    if workers is None:
        workers = max(int(OmegaConf.select(conf, "training.num_workers", default=4)), 1)
    OmegaConf.update(conf, "training.num_workers", int(workers), merge=True)
    if int(workers) <= 0:
        OmegaConf.update(conf, "training.persistent_workers", False, merge=True)
        OmegaConf.update(conf, "training.prefetch_factor", None, merge=True)
    else:
        OmegaConf.update(conf, "training.persistent_workers", True, merge=True)
        OmegaConf.update(conf, "training.prefetch_factor", int(OmegaConf.select(conf, "training.prefetch_factor", default=3)), merge=True)
    return conf


def _finite_float(value: Any) -> float | None:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate_missing_outputs(
    output_dir: Path,
    protocol: MissingModalityProtocol,
    *,
    max_batches: int | None,
) -> dict[str, Any]:
    required = ["summary.json", "metrics.csv", "missing_mode_summary.csv", "dataset_missing_mode_summary.csv"]
    missing_files = [name for name in required if not (output_dir / name).is_file()]
    if missing_files:
        raise RuntimeError(f"{output_dir} missing required outputs: {missing_files}")

    rows = _read_csv(output_dir / "missing_mode_summary.csv")
    found_modes = [row.get("missing_mode", "") for row in rows]
    missing_modes = [mode for mode in protocol.mode_names if mode not in found_modes]
    extra_modes = [mode for mode in found_modes if mode not in protocol.mode_names]
    if missing_modes or extra_modes:
        raise RuntimeError(f"{output_dir} missing modes {missing_modes}; extra modes {extra_modes}.")

    expected_per_mode = None if max_batches is not None else EXPECTED_OPENFWI_SAMPLES_PER_MODE
    for row in rows:
        if expected_per_mode is not None and int(row["num_samples"]) != expected_per_mode:
            raise RuntimeError(
                f"{output_dir} mode {row['missing_mode']} has {row['num_samples']} samples; "
                f"expected {expected_per_mode}."
            )
        for key in ("mae", "rmse", "ssim", "mae_l", "mae_h"):
            if _finite_float(row.get(key)) is None:
                raise RuntimeError(f"{output_dir} mode {row['missing_mode']} has non-finite {key}: {row.get(key)!r}.")

    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    expected_total = None if max_batches is not None else EXPECTED_OPENFWI_SAMPLES_PER_MODE * protocol.expected_num_modes
    if expected_total is not None and int(summary.get("num_samples", -1)) != expected_total:
        raise RuntimeError(f"{output_dir} summary has {summary.get('num_samples')} samples; expected {expected_total}.")

    return {
        "status": "complete",
        "num_modes": len(rows),
        "num_samples": int(summary.get("num_samples", sum(int(row["num_samples"]) for row in rows))),
    }


def _read_existing_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _upsert_manifest_entry(manifest: list[dict[str, Any]], entry: dict[str, Any]) -> list[dict[str, Any]]:
    kept = [item for item in manifest if item.get("job") != entry.get("job")]
    kept.append(entry)
    order = {job.name: idx for idx, job in enumerate(JOBS)}
    return sorted(kept, key=lambda item: order.get(str(item.get("job")), len(order)))


def _read_job_rows(output_root: Path, filename: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for job in JOBS:
        path = output_root / job.name / filename
        if not path.is_file():
            continue
        for row in _read_csv(path):
            rows.append({"job": job.name, **row})
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _average(rows: list[dict[str, Any]], group_keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(tuple(str(row[key]) for key in group_keys), []).append(row)
    out: list[dict[str, Any]] = []
    for group_key in sorted(groups):
        selected = groups[group_key]
        item = {key: value for key, value in zip(group_keys, group_key, strict=True)}
        item["num_samples"] = sum(int(row["num_samples"]) for row in selected)
        for key in SUMMARY_METRIC_KEYS:
            values = [x for row in selected if (x := _finite_float(row.get(key))) is not None]
            item[key] = mean(values) if values else ""
        out.append(item)
    return out


def write_global_outputs(output_root: Path) -> None:
    mode_rows = _read_job_rows(output_root, "missing_mode_summary.csv")
    dataset_mode_rows = _read_job_rows(output_root, "dataset_missing_mode_summary.csv")
    fieldnames = ["job", "dataset_name", "missing_mode", "num_samples", *SUMMARY_METRIC_KEYS]

    for row in mode_rows:
        row.setdefault("dataset_name", "OpenFWI")
    _write_csv(output_root / "openfwi_missing_mode_summary.csv", mode_rows, fieldnames)
    _write_csv(output_root / "openfwi_dataset_missing_mode_summary.csv", dataset_mode_rows, fieldnames)
    _write_csv(
        output_root / "openfwi_dataset_summary.csv",
        _average(dataset_mode_rows, ["job", "dataset_name"]),
        ["job", "dataset_name", "num_samples", *SUMMARY_METRIC_KEYS],
    )

    missing_only = _average([row for row in mode_rows if row.get("missing_mode") != "full"], ["job"])
    lines = [
        "# AAAI27 OpenFWI Missing-Modality Diagnostic",
        "",
        "This diagnostic is explicit and separate from full-modality formal evaluation.",
        "",
        "## Missing-Only Average",
        "",
        "| Method | MAE | RMSE | SSIM | MAE_L | MAE_H |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(missing_only, key=lambda item: float(item["mae"]) if item.get("mae") != "" else float("inf")):
        lines.append(
            f"| {row['job']} | {float(row['mae']):.6f} | {float(row['rmse']):.6f} | "
            f"{float(row['ssim']):.6f} | {float(row['mae_l']):.6f} | {float(row['mae_h']):.6f} |"
        )
    lines.extend(
        [
            "",
            f"Mode CSV: `{output_root / 'openfwi_missing_mode_summary.csv'}`",
            f"Dataset-mode CSV: `{output_root / 'openfwi_dataset_missing_mode_summary.csv'}`",
            "",
        ]
    )
    (output_root / "openfwi_missing_summary.md").write_text("\n".join(lines), encoding="utf-8")


def run(
    *,
    output_root: Path,
    openfwi_root: Path,
    lmdb_root: Path,
    max_batches: int | None,
    job_names: list[str] | None,
    num_workers: int | None,
) -> None:
    protocol = MissingModalityProtocol.aaai27()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "run_manifest.json"
    manifest = _read_existing_manifest(manifest_path)

    for job in _selected_jobs(job_names):
        checkpoint = _resolve_best_checkpoint(job.checkpoint_hint)
        if job.checkpoint_hint and checkpoint is None:
            entry = {
                "job": job.name,
                "config": job.config,
                "checkpoint_hint": job.checkpoint_hint,
                "status": "skipped_missing_checkpoint",
            }
            manifest = _upsert_manifest_entry(manifest, entry)
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print(f"=== Skipped {job.name}: missing checkpoint {job.checkpoint_hint} ===", flush=True)
            continue

        output_dir = output_root / job.name
        print(f"=== Missing-modality OpenFWI eval: {job.name} ===", flush=True)
        conf = apply_openfwi_missing_overrides(
            load_benchmark_config(job.config),
            checkpoint=checkpoint,
            output_dir=output_dir,
            openfwi_root=openfwi_root,
            lmdb_root=lmdb_root,
            max_batches=max_batches,
            protocol=protocol,
            num_workers=num_workers,
        )
        summary = run_benchmark_evaluation(conf)
        validation = validate_missing_outputs(output_dir, protocol, max_batches=max_batches)
        manifest = _upsert_manifest_entry(
            manifest,
            {
                "job": job.name,
                "config": job.config,
                "checkpoint": "" if checkpoint is None else str(checkpoint),
                "output_dir": str(output_dir),
                "status": "complete",
                "validation": validation,
                "summary": summary,
            },
        )
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        write_global_outputs(output_root)
        print(f"=== Finished {job.name}: {validation['num_samples']} samples ===", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run explicit OpenFWI missing-modality diagnostics.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--openfwi-root", default=str(DEFAULT_OPENFWI_ROOT))
    parser.add_argument("--lmdb-root", default=str(DEFAULT_OPENFWI_LMDB_ROOT))
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--jobs", nargs="+", default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    args = parser.parse_args()
    run(
        output_root=Path(args.output_root),
        openfwi_root=Path(args.openfwi_root),
        lmdb_root=Path(args.lmdb_root),
        max_batches=args.max_batches,
        job_names=args.jobs,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()
