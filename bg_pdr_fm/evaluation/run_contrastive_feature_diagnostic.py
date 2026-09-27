"""Run contrastive checkpoint-swap diagnostics for a fixed BG-PDR-FM model.

This utility keeps the trained background and residual checkpoints fixed, swaps
only the contrastive encoder checkpoint, and then summarizes both downstream
10-batch reconstruction metrics and contrastive feature diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_BASE_EVAL_CONFIG = (
    "bg_pdr_fm/configs/openfwi_lmdb_residual_smooth_hc64_nogate_h256_e200_eval_mb10.yaml"
)
DEFAULT_CONTRASTIVE_CONFIG = "bg_pdr_fm/configs/openfwi_lmdb_contrastive.yaml"


@dataclass(frozen=True)
class CaseSpec:
    name: str
    checkpoint: str
    metrics_csv: str


CASES: tuple[CaseSpec, ...] = (
    CaseSpec(
        "current_last",
        "logs/bg_pdr_fm/contrastive_train/stage_checkpoints/contrastive_last.ckpt",
        "logs/bg_pdr_fm/contrastive_train/lightning/csv/"
        "contrastive_pairwise_recover_20260528_161613/metrics.csv",
    ),
    CaseSpec(
        "pairwise_epoch77",
        "logs/bg_pdr_fm/contrastive_train/lightning/checkpoints/pairwise-epoch77-top50.9635.ckpt",
        "logs/bg_pdr_fm/contrastive_train/lightning/csv/"
        "contrastive_pairwise_recover_20260528_161613/metrics.csv",
    ),
    CaseSpec(
        "anchor_epoch22",
        "logs/bg_pdr_fm/contrastive_train/lightning/checkpoints/contrastive-epoch_22-loss8.6160.ckpt",
        "logs/bg_pdr_fm/contrastive_train/lightning/csv/contrastive_20260525_134208/metrics.csv",
    ),
    CaseSpec(
        "recover_epoch4",
        "logs/bg_pdr_fm/contrastive_train/lightning/checkpoints/contrastive_recover-epoch_4-loss8.8320.ckpt",
        "logs/bg_pdr_fm/contrastive_train/lightning/csv/"
        "contrastive_retrieval_recover_20260527_101149/metrics.csv",
    ),
)

EVAL_METRICS = (
    "mae",
    "rmse",
    "mse",
    "ssim",
    "mae_l",
    "mae_h",
    "bg_mae",
    "epsilon_H_B",
    "rho_B",
    "rho_hat_B",
    "alpha_hat",
)


def _log(message: str) -> None:
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {message}", flush=True)


def _resolve_path(path: str | Path) -> Path:
    item = Path(path)
    return item if item.is_absolute() else REPO_ROOT / item


def _repo_relative(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _case_by_name(names: list[str] | None) -> list[CaseSpec]:
    if not names:
        return list(CASES)
    lookup = {case.name: case for case in CASES}
    unknown = [name for name in names if name not in lookup]
    if unknown:
        raise ValueError(f"Unknown diagnostic case(s): {', '.join(unknown)}")
    return [lookup[name] for name in names]


def _ensure_checkpoints(cases: list[CaseSpec]) -> None:
    missing: list[str] = []
    for case in cases:
        if not _resolve_path(case.checkpoint).is_file():
            missing.append(f"{case.name}: {case.checkpoint}")
    if missing:
        raise FileNotFoundError("Missing contrastive checkpoints:\n" + "\n".join(missing))


def _update_eval_config(
    *,
    base_config: Path,
    output_root: Path,
    case: CaseSpec,
    max_batches: int,
    dry_run: bool,
    save_panels: bool,
) -> Path:
    conf = OmegaConf.load(base_config)
    tag = "dryrun" if dry_run else "mb10"
    out_dir = output_root / ("h256_dryrun" if dry_run else "h256_mb10") / case.name
    config_dir = output_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    OmegaConf.update(conf, "training.devices", 1, merge=True)
    OmegaConf.update(conf, "training.logging.log_version", f"contrastive_feature_{tag}_{case.name}", merge=True)
    OmegaConf.update(conf, "evaluation.output_dir", _repo_relative(out_dir), merge=True)
    OmegaConf.update(conf, "evaluation.max_batches", int(max_batches), merge=True)
    OmegaConf.update(conf, "evaluation.save_arrays", False, merge=True)
    OmegaConf.update(conf, "evaluation.save_panels", bool(save_panels), merge=True)
    OmegaConf.update(conf, "evaluation.checkpoints.contrastive", case.checkpoint, merge=True)

    path = config_dir / f"{case.name}_{tag}.yaml"
    OmegaConf.save(config=conf, f=str(path))
    return path


def _run_command(command: list[str], *, gpu: str, log_path: Path, force_env: dict[str, str] | None = None) -> int:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("MPLCONFIGDIR", "/tmp/mpl-bg-pdr-fm-contrastive-diagnostic")
    if gpu:
        env["CUDA_VISIBLE_DEVICES"] = gpu
    if force_env:
        env.update(force_env)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _log(f"GPU {gpu or '<unset>'}: {' '.join(command)}")
    with log_path.open("ab", buffering=0) as handle:
        result = subprocess.run(command, cwd=str(REPO_ROOT), env=env, stdout=handle, stderr=subprocess.STDOUT)
    _log(f"exit {result.returncode}: {log_path}")
    return int(result.returncode)


def _run_eval(
    *,
    case: CaseSpec,
    gpu: str,
    python: str,
    config_path: Path,
    output_dir: Path,
    force: bool,
) -> int:
    summary_path = output_dir / "summary.json"
    if summary_path.is_file() and not force:
        _log(f"{case.name}: using existing eval summary {summary_path}")
        return 0
    return _run_command(
        [python, "-m", "bg_pdr_fm.evaluation.evaluate_bg_pdr_fm", "--config", _repo_relative(config_path)],
        gpu=gpu,
        log_path=output_dir / "eval.log",
    )


def _run_visualization(
    *,
    contrastive_config: Path,
    case: CaseSpec,
    gpu: str,
    python: str,
    output_dir: Path,
    split: str,
    max_samples: int,
    batch_size: int,
    force: bool,
) -> int:
    summary_path = output_dir / "summary.json"
    if summary_path.is_file() and not force:
        _log(f"{case.name}: using existing visualization summary {summary_path}")
        return 0
    return _run_command(
        [
            python,
            "-m",
            "bg_pdr_fm.evaluation.visualize_contrastive",
            "--config",
            _repo_relative(contrastive_config),
            "--checkpoint",
            case.checkpoint,
            "--metrics-csv",
            case.metrics_csv,
            "--split",
            split,
            "--max-samples",
            str(max_samples),
            "--batch-size",
            str(batch_size),
            "--output-dir",
            _repo_relative(output_dir),
        ],
        gpu=gpu,
        log_path=output_dir / "visualize.log",
    )


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _mean_csv_column(path: Path, column: str) -> float | None:
    if not path.is_file():
        return None
    values: list[float] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            value = row.get(column)
            if value not in {None, ""}:
                values.append(float(value))
    if not values:
        return None
    return float(sum(values) / len(values))


def _read_eval_row(case: CaseSpec, output_root: Path) -> dict[str, Any]:
    summary_path = output_root / "h256_mb10" / case.name / "summary.json"
    summary = _read_json(summary_path)
    row: dict[str, Any] = {
        "case": case.name,
        "contrastive_checkpoint": case.checkpoint,
        "summary": _repo_relative(summary_path),
        "num_samples": "",
        "status": "missing",
    }
    if summary is None:
        return row
    means = summary.get("means", {})
    row["num_samples"] = summary.get("num_samples", "")
    row["status"] = "ok"
    for key in EVAL_METRICS:
        row[key] = means.get(key, "")
    return row


def _read_visual_row(case: CaseSpec, output_root: Path, max_samples: int) -> dict[str, Any]:
    vis_dir = output_root / f"contrastive_vis_test{max_samples}" / case.name
    summary = _read_json(vis_dir / "summary.json")
    row: dict[str, Any] = {
        "case": case.name,
        "contrastive_checkpoint": case.checkpoint,
        "summary": _repo_relative(vis_dir / "summary.json"),
        "num_samples": "",
        "status": "missing",
    }
    if summary is None:
        return row
    means = summary.get("means", {})
    reliability = means.get("reliability", {}) if isinstance(means, dict) else {}
    row["num_samples"] = summary.get("num_samples", "")
    row["status"] = "ok"
    row["retrieval_top1_mean"] = means.get("retrieval_top1_mean", "")
    row["retrieval_top5_mean"] = means.get("retrieval_top5_mean", "")
    for key, value in reliability.items():
        row[f"reliability_{key}"] = value
    sample_metrics = vis_dir / "sample_metrics.csv"
    row["structural_alignment_mean"] = _mean_csv_column(sample_metrics, "structural_alignment_mean")
    row["numerical_alignment_mean"] = _mean_csv_column(sample_metrics, "numerical_alignment_mean")
    energy_csv = vis_dir / "features" / "feature_energy_ratios.csv"
    for key in (
        "numerical_low_energy_ratio",
        "numerical_high_energy_ratio",
        "structural_low_energy_ratio",
        "structural_high_energy_ratio",
        "unique_low_energy_ratio",
        "unique_high_energy_ratio",
    ):
        row[key] = _mean_csv_column(energy_csv, key)
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]], preferred_fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(preferred_fields)
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _float_or_none(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _add_deltas(rows: list[dict[str, Any]]) -> None:
    current = next((row for row in rows if row.get("case") == "current_last" and row.get("status") == "ok"), None)
    if current is None:
        return
    for row in rows:
        for key in ("ssim", "mae", "mae_l", "mae_h", "rho_B"):
            base = _float_or_none(current.get(key))
            value = _float_or_none(row.get(key))
            row[f"delta_{key}"] = "" if base is None or value is None else value - base


def _fmt(value: Any) -> str:
    number = _float_or_none(value)
    if number is None:
        return "NA"
    return f"{number:.6f}"


def _markdown_eval_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| case | samples | ssim | delta_ssim | mae | mae_l | mae_h | rho_B |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {case} | {samples} | {ssim} | {delta_ssim} | {mae} | {mae_l} | {mae_h} | {rho_B} |".format(
                case=row.get("case", ""),
                samples=row.get("num_samples", ""),
                ssim=_fmt(row.get("ssim")),
                delta_ssim=_fmt(row.get("delta_ssim")),
                mae=_fmt(row.get("mae")),
                mae_l=_fmt(row.get("mae_l")),
                mae_h=_fmt(row.get("mae_h")),
                rho_B=_fmt(row.get("rho_B")),
            )
        )
    return lines


def _markdown_visual_table(rows: list[dict[str, Any]]) -> list[str]:
    lines = [
        "| case | samples | retrieval_top1 | retrieval_top5 | struct_align | num_align | S_high | N_low |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {case} | {samples} | {top1} | {top5} | {sa} | {na} | {sh} | {nl} |".format(
                case=row.get("case", ""),
                samples=row.get("num_samples", ""),
                top1=_fmt(row.get("retrieval_top1_mean")),
                top5=_fmt(row.get("retrieval_top5_mean")),
                sa=_fmt(row.get("structural_alignment_mean")),
                na=_fmt(row.get("numerical_alignment_mean")),
                sh=_fmt(row.get("structural_high_energy_ratio")),
                nl=_fmt(row.get("numerical_low_energy_ratio")),
            )
        )
    return lines


def _visual_row(visual_rows: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    return next((row for row in visual_rows if row.get("case") == name and row.get("status") == "ok"), None)


def _decision_lines(eval_rows: list[dict[str, Any]], visual_rows: list[dict[str, Any]]) -> list[str]:
    current = next((row for row in eval_rows if row.get("case") == "current_last"), None)
    if current is None or current.get("status") != "ok":
        return ["- current_last summary is missing; no decision can be made."]
    lines: list[str] = []

    improving = []
    for row in eval_rows:
        if row.get("case") == "current_last" or row.get("status") != "ok":
            continue
        delta_ssim = _float_or_none(row.get("delta_ssim"))
        delta_mae_l = _float_or_none(row.get("delta_mae_l"))
        delta_mae_h = _float_or_none(row.get("delta_mae_h"))
        if delta_ssim is not None and delta_ssim >= 0.003:
            improving.append(row.get("case"))
        elif delta_mae_l is not None and delta_mae_l < 0 and delta_mae_h is not None and delta_mae_h < 0:
            improving.append(row.get("case"))

    pairwise = next((row for row in eval_rows if row.get("case") == "pairwise_epoch77"), None)
    if pairwise is not None and pairwise.get("status") == "ok":
        delta = _float_or_none(pairwise.get("delta_ssim"))
        pairwise_vis = _visual_row(visual_rows, "pairwise_epoch77")
        current_vis = _visual_row(visual_rows, "current_last")
        top5_delta = None
        if pairwise_vis is not None and current_vis is not None:
            pairwise_top5 = _float_or_none(pairwise_vis.get("retrieval_top5_mean"))
            current_top5 = _float_or_none(current_vis.get("retrieval_top5_mean"))
            if pairwise_top5 is not None and current_top5 is not None:
                top5_delta = pairwise_top5 - current_top5
        if delta is not None and delta <= 0:
            if top5_delta is None:
                lines.append(
                    f"- pairwise_epoch77 drops downstream SSIM by {_fmt(delta)}; retrieval alone is not a sufficient selector."
                )
            else:
                lines.append(
                    f"- pairwise_epoch77 raises retrieval_top5 by {_fmt(top5_delta)} but drops downstream SSIM by {_fmt(delta)}; retrieval alone is not a sufficient selector."
                )

    for name in ("anchor_epoch22", "recover_epoch4"):
        row = next((item for item in eval_rows if item.get("case") == name), None)
        vis = _visual_row(visual_rows, name)
        if row is None or row.get("status") != "ok":
            continue
        delta = _float_or_none(row.get("delta_ssim"))
        align_text = ""
        if vis is not None:
            struct_align = _float_or_none(vis.get("structural_alignment_mean"))
            num_align = _float_or_none(vis.get("numerical_alignment_mean"))
            if struct_align is not None and num_align is not None:
                align_text = f" despite high anchor alignment (S={_fmt(struct_align)}, N={_fmt(num_align)})"
        if delta is not None and delta < -0.10:
            lines.append(
                f"- {name} collapses fixed-load downstream SSIM by {_fmt(delta)}{align_text}; high anchor alignment is not plug-compatible with the trained adapters/residual."
            )

    if improving:
        lines.append(
            "- At least one checkpoint crosses the improvement threshold; run formal full eval for: "
            + ", ".join(str(item) for item in improving)
            + "."
        )
    else:
        lines.append(
            "- Checkpoint-swap deltas do not cross the +0.003 SSIM threshold; contrastive checkpoint choice is not the primary limiter in this probe."
        )
        lines.append("- Do not run formal full eval or restart contrastive training from these swaps alone.")
    return lines


def summarize(output_root: Path, cases: list[CaseSpec], visual_max_samples: int) -> dict[str, Path]:
    eval_rows = [_read_eval_row(case, output_root) for case in cases]
    _add_deltas(eval_rows)
    visual_rows = [_read_visual_row(case, output_root, visual_max_samples) for case in cases]

    eval_csv = output_root / "checkpoint_swap_summary.csv"
    visual_csv = output_root / "contrastive_visual_summary.csv"
    _write_csv(
        eval_csv,
        eval_rows,
        [
            "case",
            "status",
            "num_samples",
            "ssim",
            "delta_ssim",
            "mae",
            "delta_mae",
            "mae_l",
            "delta_mae_l",
            "mae_h",
            "delta_mae_h",
            "rho_B",
            "delta_rho_B",
            "bg_mae",
            "contrastive_checkpoint",
            "summary",
        ],
    )
    _write_csv(
        visual_csv,
        visual_rows,
        [
            "case",
            "status",
            "num_samples",
            "retrieval_top1_mean",
            "retrieval_top5_mean",
            "structural_alignment_mean",
            "numerical_alignment_mean",
            "numerical_low_energy_ratio",
            "structural_high_energy_ratio",
            "contrastive_checkpoint",
            "summary",
        ],
    )

    md_path = output_root / "contrastive_feature_diagnostic_summary.md"
    lines = [
        "# Contrastive Feature Diagnostic",
        "",
        "Fixed model: `smooth_hc64 + No-Gate residual h256 e200`.",
        "Only the contrastive checkpoint is swapped; background and residual checkpoints stay fixed.",
        "",
        "## Downstream 10-Batch Eval",
        "",
        *_markdown_eval_table(eval_rows),
        "",
        "## Contrastive Feature Diagnostics",
        "",
        *_markdown_visual_table(visual_rows),
        "",
        "## Decision",
        "",
        *_decision_lines(eval_rows, visual_rows),
        "",
        "## Caveat",
        "",
        "This probe keeps trained background/residual adapters fixed. A positive swap is strong evidence that contrastive features matter; a negative swap does not fully rule out benefits from retraining adapters or residual FM with that encoder.",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {"eval_csv": eval_csv, "visual_csv": visual_csv, "markdown": md_path}


def run_case(
    *,
    case: CaseSpec,
    gpu: str,
    args: argparse.Namespace,
    base_eval_config: Path,
    contrastive_config: Path,
    output_root: Path,
) -> dict[str, Any]:
    dry_config = _update_eval_config(
        base_config=base_eval_config,
        output_root=output_root,
        case=case,
        max_batches=args.dry_run_batches,
        dry_run=True,
        save_panels=False,
    )
    eval_config = _update_eval_config(
        base_config=base_eval_config,
        output_root=output_root,
        case=case,
        max_batches=args.max_batches,
        dry_run=False,
        save_panels=args.save_panels,
    )
    dry_dir = output_root / "h256_dryrun" / case.name
    eval_dir = output_root / "h256_mb10" / case.name
    vis_dir = output_root / f"contrastive_vis_test{args.visual_max_samples}" / case.name

    result = {"case": case.name, "gpu": gpu, "dry_run": None, "eval": None, "visualization": None}
    if not args.skip_dry_run:
        result["dry_run"] = _run_eval(
            case=case,
            gpu=gpu,
            python=args.python,
            config_path=dry_config,
            output_dir=dry_dir,
            force=args.force,
        )
        if result["dry_run"] != 0:
            return result
    if not args.skip_eval:
        result["eval"] = _run_eval(
            case=case,
            gpu=gpu,
            python=args.python,
            config_path=eval_config,
            output_dir=eval_dir,
            force=args.force,
        )
        if result["eval"] != 0:
            return result
    if not args.skip_visualization:
        result["visualization"] = _run_visualization(
            contrastive_config=contrastive_config,
            case=case,
            gpu=gpu,
            python=args.python,
            output_dir=vis_dir,
            split=args.split,
            max_samples=args.visual_max_samples,
            batch_size=args.visual_batch_size,
            force=args.force,
        )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-eval-config", default=DEFAULT_BASE_EVAL_CONFIG)
    parser.add_argument("--contrastive-config", default=DEFAULT_CONTRASTIVE_CONFIG)
    parser.add_argument("--output-root", default="logs/bg_pdr_fm/contrastive_feature_diagnostic")
    parser.add_argument("--cases", nargs="*", choices=[case.name for case in CASES], default=None)
    parser.add_argument("--gpus", default="7,6,5,4")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-batches", type=int, default=10)
    parser.add_argument("--dry-run-batches", type=int, default=1)
    parser.add_argument("--visual-max-samples", type=int, default=300)
    parser.add_argument("--visual-batch-size", type=int, default=32)
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--save-panels", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-dry-run", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--skip-visualization", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args(argv)

    cases = _case_by_name(args.cases)
    _ensure_checkpoints(cases)
    base_eval_config = _resolve_path(args.base_eval_config)
    contrastive_config = _resolve_path(args.contrastive_config)
    output_root = _resolve_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if not base_eval_config.is_file():
        raise FileNotFoundError(f"Base eval config not found: {base_eval_config}")
    if not contrastive_config.is_file():
        raise FileNotFoundError(f"Contrastive config not found: {contrastive_config}")

    results: list[dict[str, Any]] = []
    if not args.summarize_only:
        gpu_list = [item.strip() for item in args.gpus.split(",") if item.strip()]
        if not gpu_list:
            gpu_list = [""]
        if args.parallel:
            with ThreadPoolExecutor(max_workers=min(len(cases), len(gpu_list))) as executor:
                futures = []
                for idx, case in enumerate(cases):
                    gpu = gpu_list[idx % len(gpu_list)]
                    futures.append(
                        executor.submit(
                            run_case,
                            case=case,
                            gpu=gpu,
                            args=args,
                            base_eval_config=base_eval_config,
                            contrastive_config=contrastive_config,
                            output_root=output_root,
                        )
                    )
                for future in as_completed(futures):
                    results.append(future.result())
        else:
            for idx, case in enumerate(cases):
                results.append(
                    run_case(
                        case=case,
                        gpu=gpu_list[idx % len(gpu_list)],
                        args=args,
                        base_eval_config=base_eval_config,
                        contrastive_config=contrastive_config,
                        output_root=output_root,
                    )
                )

    summary_paths = summarize(output_root, cases, args.visual_max_samples)
    run_summary = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "cases": [case.name for case in cases],
        "results": results,
        "summary_paths": {key: _repo_relative(value) for key, value in summary_paths.items()},
    }
    (output_root / "run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    _log(f"wrote {summary_paths['markdown']}")
    failed = any(
        value not in {None, 0}
        for result in results
        for key, value in result.items()
        if key not in {"case", "gpu"}
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
