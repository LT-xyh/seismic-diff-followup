"""Run OpenFWI engineering-stress evaluation for completed AAAI27 methods."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import mean
from typing import Any

from omegaconf import OmegaConf

from bg_pdr_fm.evaluation.compare_experiments import SUMMARY_METRIC_KEYS, run_benchmark_evaluation
from bg_pdr_fm.evaluation.engineering_stress import get_engineering_stress_scenario
from bg_pdr_fm.evaluation.run_missing_modality_benchmark import MissingBenchmarkJob
from bg_pdr_fm.training.benchmark_config import load_benchmark_config


DEFAULT_OUTPUT_ROOT = Path("logs/bg_pdr_fm/aaai27/eval_engineering_mild_openfwi")
DEFAULT_OPENFWI_ROOT = Path("/public/home/xuyinghao/workspace/datasets/openfwi")
DEFAULT_OPENFWI_LMDB_ROOT = Path("/public/home/xuyinghao/workspace/datasets/openfwi_lmdb")
EXPECTED_OPENFWI_SAMPLES = 33600


ENGINEERING_JOBS: tuple[MissingBenchmarkJob, ...] = (
    MissingBenchmarkJob("smooth_dix", "bg_pdr_fm/configs/experiments/aaai27/formal_smooth_dix.yaml"),
    MissingBenchmarkJob(
        "mm_invnet",
        "bg_pdr_fm/configs/experiments/aaai27/formal_mm_invnet.yaml",
        "logs/bg_pdr_fm/aaai27/formal/mm_invnet/checkpoints/last.ckpt",
    ),
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
        "two_stage_ddpm",
        "bg_pdr_fm/configs/experiments/aaai27/formal_two_stage_ddpm.yaml",
        "logs/bg_pdr_fm/aaai27/formal/two_stage_ddpm/checkpoints/last.ckpt",
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
        return list(ENGINEERING_JOBS)
    requested = {name.strip() for name in job_names if name.strip()}
    known = {job.name for job in ENGINEERING_JOBS}
    unknown = sorted(requested.difference(known))
    if unknown:
        raise ValueError(f"Unknown jobs {unknown}; expected subset of {sorted(known)}.")
    return [job for job in ENGINEERING_JOBS if job.name in requested]


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


def apply_openfwi_engineering_overrides(
    conf: Any,
    *,
    checkpoint: Path | str | None,
    output_dir: Path | str,
    openfwi_root: Path | str,
    lmdb_root: Path | str,
    max_batches: int | None,
    num_workers: int | None,
    scenario: str,
    seed: int,
) -> Any:
    get_engineering_stress_scenario(scenario, seed=seed)
    OmegaConf.update(conf, "data.name", "openfwi", merge=True)
    OmegaConf.update(conf, "data.root_dir", str(openfwi_root), merge=True)
    OmegaConf.update(conf, "data.root_candidates", [str(openfwi_root)], merge=True)
    OmegaConf.update(conf, "data.storage_backend", "lmdb", merge=True)
    OmegaConf.update(conf, "data.lmdb_root", str(lmdb_root), merge=True)
    OmegaConf.update(conf, "evaluation.checkpoint", None if checkpoint is None else str(checkpoint), merge=True)
    OmegaConf.update(conf, "evaluation.output_dir", str(output_dir), merge=True)
    OmegaConf.update(conf, "evaluation.max_batches", max_batches, merge=True)
    OmegaConf.update(conf, "evaluation.missing_modes", ["full"], merge=True)
    OmegaConf.update(conf, "evaluation.engineering_stress.enabled", True, merge=True)
    OmegaConf.update(conf, "evaluation.engineering_stress.scenario", scenario, merge=True)
    OmegaConf.update(conf, "evaluation.engineering_stress.seed", int(seed), merge=True)
    OmegaConf.update(conf, "training.devices", 1, merge=True)
    workers = max(int(OmegaConf.select(conf, "training.num_workers", default=4)), 1) if num_workers is None else int(num_workers)
    OmegaConf.update(conf, "training.num_workers", workers, merge=True)
    if workers <= 0:
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


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _read_existing_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def _upsert_manifest_entry(manifest: list[dict[str, Any]], entry: dict[str, Any]) -> list[dict[str, Any]]:
    kept = [item for item in manifest if item.get("job") != entry.get("job")]
    kept.append(entry)
    order = {job.name: idx for idx, job in enumerate(ENGINEERING_JOBS)}
    return sorted(kept, key=lambda item: order.get(str(item.get("job")), len(order)))


def validate_engineering_outputs(output_dir: Path, *, max_batches: int | None) -> dict[str, Any]:
    required = ["summary.json", "metrics.csv", "missing_mode_summary.csv", "dataset_summary.csv"]
    missing_files = [name for name in required if not (output_dir / name).is_file()]
    if missing_files:
        raise RuntimeError(f"{output_dir} missing required outputs: {missing_files}")
    summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
    if summary.get("engineering_stress") is None:
        raise RuntimeError(f"{output_dir} summary missing engineering_stress metadata.")
    if list(summary.get("missing_modes", [])) != ["full"]:
        raise RuntimeError(f"{output_dir} expected only full missing mode, got {summary.get('missing_modes')!r}.")
    if max_batches is None and int(summary.get("num_samples", -1)) != EXPECTED_OPENFWI_SAMPLES:
        raise RuntimeError(f"{output_dir} summary has {summary.get('num_samples')} samples; expected {EXPECTED_OPENFWI_SAMPLES}.")
    rows = _read_csv(output_dir / "missing_mode_summary.csv")
    if len(rows) != 1 or rows[0].get("missing_mode") != "full":
        raise RuntimeError(f"{output_dir} expected one full row in missing_mode_summary.csv.")
    for key in ("mae", "rmse", "ssim", "mae_l", "mae_h"):
        if _finite_float(rows[0].get(key)) is None:
            raise RuntimeError(f"{output_dir} has non-finite {key}: {rows[0].get(key)!r}.")
    return {"status": "complete", "num_samples": int(summary.get("num_samples", 0))}


def _read_job_rows(output_root: Path, filename: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for job in ENGINEERING_JOBS:
        path = output_root / job.name / filename
        if not path.is_file():
            continue
        for row in _read_csv(path):
            rows.append({"job": job.name, **row})
    return rows


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


def _scenario_from_outputs(output_root: Path) -> str:
    for job in ENGINEERING_JOBS:
        summary_path = output_root / job.name / "summary.json"
        if not summary_path.is_file():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        stress = summary.get("engineering_stress")
        if isinstance(stress, dict) and stress.get("name"):
            return str(stress["name"])
    return "engineering_stress"


def write_global_outputs(output_root: Path, scenario: str | None = None) -> None:
    scenario_name = scenario or _scenario_from_outputs(output_root)
    mode_rows = _read_job_rows(output_root, "missing_mode_summary.csv")
    dataset_rows = _read_job_rows(output_root, "dataset_summary.csv")
    fieldnames = ["job", "missing_mode", "num_samples", *SUMMARY_METRIC_KEYS]
    summary_csv = output_root / f"openfwi_{scenario_name}_summary.csv"
    dataset_csv = output_root / f"openfwi_{scenario_name}_dataset_summary.csv"
    summary_md = output_root / f"{scenario_name}_summary.md"
    _write_csv(summary_csv, mode_rows, fieldnames)
    _write_csv(
        dataset_csv,
        dataset_rows,
        ["job", "dataset_name", "num_samples", *SUMMARY_METRIC_KEYS],
    )
    lines = [
        "# AAAI27 OpenFWI Engineering-Stress Evaluation",
        "",
        f"Scenario: `{scenario_name}`.",
        "",
        "| Method | MAE | RMSE | SSIM | MAE_L | MAE_H |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(mode_rows, key=lambda item: float(item["mae"]) if item.get("mae") != "" else float("inf")):
        lines.append(
            f"| {row['job']} | {float(row['mae']):.6f} | {float(row['rmse']):.6f} | "
            f"{float(row['ssim']):.6f} | {float(row['mae_l']):.6f} | {float(row['mae_h']):.6f} |"
        )
    lines.extend(
        [
            "",
            f"Dataset CSV: `{dataset_csv}`",
            f"Summary CSV: `{summary_csv}`",
            "",
        ]
    )
    summary_md.write_text("\n".join(lines), encoding="utf-8")


def run(
    *,
    output_root: Path,
    openfwi_root: Path,
    lmdb_root: Path,
    max_batches: int | None,
    job_names: list[str] | None,
    num_workers: int | None,
    scenario: str,
    seed: int,
) -> None:
    get_engineering_stress_scenario(scenario, seed=seed)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "run_manifest.json"
    manifest = _read_existing_manifest(manifest_path)

    for job in _selected_jobs(job_names):
        checkpoint = _resolve_best_checkpoint(job.checkpoint_hint)
        if job.checkpoint_hint and checkpoint is None:
            manifest = _upsert_manifest_entry(
                manifest,
                {
                    "job": job.name,
                    "config": job.config,
                    "checkpoint_hint": job.checkpoint_hint,
                    "status": "skipped_missing_checkpoint",
                },
            )
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print(f"=== Skipped {job.name}: missing checkpoint {job.checkpoint_hint} ===", flush=True)
            continue

        output_dir = output_root / job.name
        print(f"=== Engineering-stress OpenFWI eval: {job.name} ===", flush=True)
        conf = apply_openfwi_engineering_overrides(
            load_benchmark_config(job.config),
            checkpoint=checkpoint,
            output_dir=output_dir,
            openfwi_root=openfwi_root,
            lmdb_root=lmdb_root,
            max_batches=max_batches,
            num_workers=num_workers,
            scenario=scenario,
            seed=seed,
        )
        summary = run_benchmark_evaluation(conf)
        validation = validate_engineering_outputs(output_dir, max_batches=max_batches)
        manifest = _upsert_manifest_entry(
            manifest,
            {
                "job": job.name,
                "config": job.config,
                "checkpoint": "" if checkpoint is None else str(checkpoint),
                "output_dir": str(output_dir),
                "scenario": scenario,
                "seed": seed,
                "status": "complete",
                "validation": validation,
                "summary": summary,
            },
        )
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        write_global_outputs(output_root, scenario=scenario)
        print(f"=== Finished {job.name}: {validation['num_samples']} samples ===", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OpenFWI engineering-stress evaluation.")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--openfwi-root", default=str(DEFAULT_OPENFWI_ROOT))
    parser.add_argument("--lmdb-root", default=str(DEFAULT_OPENFWI_LMDB_ROOT))
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--jobs", nargs="+", default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--scenario", default="pstm20_rms_mild_h025_w050")
    parser.add_argument("--seed", type=int, default=2027)
    args = parser.parse_args()
    run(
        output_root=Path(args.output_root),
        openfwi_root=Path(args.openfwi_root),
        lmdb_root=Path(args.lmdb_root),
        max_batches=args.max_batches,
        job_names=args.jobs,
        num_workers=args.num_workers,
        scenario=args.scenario,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
