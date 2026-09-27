"""Oracle upper-bound diagnostics for BG-PDR-FM full-modality reconstruction."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


METRIC_KEYS = (
    "mae",
    "rmse",
    "mse",
    "ssim",
    "mae_l",
    "mae_h",
    "rho_B",
    "bg_mae",
    "epsilon_H_B",
    "E_R_over_E_V",
)

ORACLE_PATHS = (
    "baseline",
    "oracle_background_current_residual",
    "oracle_latent_residual",
    "oracle_background_oracle_latent",
)


def finite_or_nan(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def mean_dict(rows: Iterable[dict[str, Any]], keys: Iterable[str]) -> dict[str, float]:
    means: dict[str, float] = {}
    rows = list(rows)
    for key in keys:
        values = [finite_or_nan(row.get(key, float("nan"))) for row in rows]
        values = [value for value in values if not math.isnan(value)]
        means[key] = float(sum(values) / len(values)) if values else float("nan")
    return means


def build_diagnosis(overall: dict[str, dict[str, float]]) -> dict[str, str | float]:
    baseline = overall.get("baseline", {})
    oracle_bg = overall.get("oracle_background_current_residual", {})
    oracle_latent = overall.get("oracle_latent_residual", {})
    oracle_combo = overall.get("oracle_background_oracle_latent", {})
    baseline_ssim = finite_or_nan(baseline.get("ssim", float("nan")))
    oracle_bg_ssim = finite_or_nan(oracle_bg.get("ssim", float("nan")))
    oracle_latent_ssim = finite_or_nan(oracle_latent.get("ssim", float("nan")))
    oracle_combo_ssim = finite_or_nan(oracle_combo.get("ssim", float("nan")))
    bg_gain = oracle_bg_ssim - baseline_ssim
    latent_gain = oracle_latent_ssim - baseline_ssim

    if oracle_latent_ssim < 0.95:
        bottleneck = "latent_codec"
        recommendation = (
            "VAE/latent ceiling is below the 0.95 target. Stop residual tuning and evaluate a higher-fidelity "
            "codec or larger latent grid before more generator training."
        )
    elif oracle_bg_ssim >= 0.93 and bg_gain >= 0.03:
        bottleneck = "stage1_background"
        recommendation = (
            "Stage 1 background is the main bottleneck. Rebuild the background estimator or background target "
            "rather than increasing residual capacity."
        )
    elif oracle_combo_ssim >= 0.95 and oracle_bg_ssim < 0.93:
        bottleneck = "residual_fm_or_conditioning"
        recommendation = (
            "Latent bandwidth is sufficient, but current residual FM does not exploit it. Focus on residual "
            "backend or condition injection."
        )
    else:
        bottleneck = "mixed_or_dataset_specific"
        recommendation = (
            "No single global bottleneck is decisive. Inspect dataset-wise oracle rows, especially CurveVelB "
            "and FlatVelB, before launching training."
        )

    return {
        "primary_bottleneck": bottleneck,
        "recommendation": recommendation,
        "baseline_ssim": baseline_ssim,
        "oracle_background_ssim": oracle_bg_ssim,
        "oracle_latent_ssim": oracle_latent_ssim,
        "oracle_combo_ssim": oracle_combo_ssim,
        "oracle_background_gain": bg_gain,
        "oracle_latent_gain": latent_gain,
    }


def _lazy_import_runtime_modules() -> dict[str, Any]:
    import torch
    from omegaconf import OmegaConf

    from bg_pdr_fm.data import batch_to_device
    from bg_pdr_fm.evaluation.benchmark_metrics import compute_velocity_metrics, residual_energy_ratio
    from bg_pdr_fm.evaluation.evaluate_bg_pdr_fm import (
        build_evaluation_loader,
        load_evaluation_checkpoints,
    )
    from bg_pdr_fm.lightning import BGPDRFMLightning
    from bg_pdr_fm.lightning.stage_losses import true_residual_low_frequency_ratio
    from bg_pdr_fm.runtime import configure_torch_runtime, configured_torch_device

    return {
        "torch": torch,
        "OmegaConf": OmegaConf,
        "batch_to_device": batch_to_device,
        "compute_velocity_metrics": compute_velocity_metrics,
        "residual_energy_ratio": residual_energy_ratio,
        "build_evaluation_loader": build_evaluation_loader,
        "load_evaluation_checkpoints": load_evaluation_checkpoints,
        "BGPDRFMLightning": BGPDRFMLightning,
        "true_residual_low_frequency_ratio": true_residual_low_frequency_ratio,
        "configure_torch_runtime": configure_torch_runtime,
        "configured_torch_device": configured_torch_device,
    }


def _conf_get(omega_conf: Any, conf: Any, path: str, default: Any) -> Any:
    return omega_conf.select(conf, path, default=default)


def _set_conf_value(omega_conf: Any, conf: Any, path: str, value: Any) -> None:
    omega_conf.update(conf, path, value, merge=False)


def apply_cli_overrides(update_fn: Any, conf: Any, output_dir: str | Path | None, max_batches: int | None) -> None:
    if output_dir is not None:
        update_fn(conf, "evaluation.output_dir", str(output_dir), merge=False)
    if max_batches is not None:
        update_fn(conf, "evaluation.max_batches", int(max_batches), merge=False)


def batch_seed(base_seed: int | None, batch_idx: int) -> int | None:
    if base_seed is None:
        return None
    return int(base_seed) + int(batch_idx)


def _metadata_int(batch: Any, key: str, item_idx: int, default: int) -> int:
    value = batch.metadata.get(key)
    if value is None:
        return int(default)
    return int(round(float(value[item_idx].detach().view(-1)[0].cpu())))


def _dataset_name_lookup(conf: Any, dataset: Any, omega_conf: Any) -> dict[int, str]:
    names = getattr(dataset, "dataset_names", None)
    if names is not None:
        return {idx: str(name) for idx, name in enumerate(tuple(names))}
    data_name = str(_conf_get(omega_conf, conf, "data.name", "synthetic"))
    if data_name == "marmousi":
        return {0: "Marmousi"}
    return {0: data_name}


def _tensor_to_float(tensor: Any, item_idx: int) -> float:
    return float(tensor[item_idx].detach().cpu())


def _predict_oracle_paths(model: Any, batch: Any, torch: Any) -> dict[str, dict[str, Any]]:
    _, conditions, bg, gate = model._common_forward(batch)
    if conditions is None or bg is None or gate is None:
        raise RuntimeError("Oracle diagnostics require a generation-stage BG-PDR-FM model.")
    if model.codec is None or model.residual is None:
        raise RuntimeError("Oracle diagnostics require codec and residual modules.")

    oracle_bg = model.filter.lowpass(batch.depth_vel)
    z_model_bg = model.codec.encode(bg.bg_hat, deterministic=True)
    z_oracle_bg = model.codec.encode(oracle_bg, deterministic=True)
    z_full = model.codec.encode(batch.depth_vel, deterministic=True)
    cond_s = conditions.structural
    out_hw = batch.depth_vel.shape[-2:]
    x_size = tuple(z_model_bg.shape)

    z_current_residual = model.residual.sample(
        cond_s,
        bg.bg_hat.detach(),
        x_size=x_size,
        steps=model.residual_num_inference_steps,
        z_bg=z_model_bg,
    )
    z_oracle_bg_current_residual = model.residual.sample(
        cond_s,
        oracle_bg.detach(),
        x_size=x_size,
        steps=model.residual_num_inference_steps,
        z_bg=z_oracle_bg,
    )

    baseline_v = model.codec.decode(z_model_bg + z_current_residual, out_hw=out_hw)
    oracle_bg_current_v = model.codec.decode(z_oracle_bg + z_oracle_bg_current_residual, out_hw=out_hw)
    oracle_latent_v = model.codec.decode(z_full, out_hw=out_hw)
    oracle_combo_v = model.codec.decode(z_full, out_hw=out_hw)

    zero_like = torch.zeros_like(gate.alpha_hat)
    return {
        "baseline": {
            "velocity_hat": baseline_v,
            "bg_hat": bg.bg_hat,
            "rho_hat_b": gate.rho_hat_b,
            "alpha_hat": gate.alpha_hat,
        },
        "oracle_background_current_residual": {
            "velocity_hat": oracle_bg_current_v,
            "bg_hat": oracle_bg,
            "rho_hat_b": gate.rho_hat_b,
            "alpha_hat": zero_like,
        },
        "oracle_latent_residual": {
            "velocity_hat": oracle_latent_v,
            "bg_hat": bg.bg_hat,
            "rho_hat_b": gate.rho_hat_b,
            "alpha_hat": zero_like,
        },
        "oracle_background_oracle_latent": {
            "velocity_hat": oracle_combo_v,
            "bg_hat": oracle_bg,
            "rho_hat_b": gate.rho_hat_b,
            "alpha_hat": zero_like,
        },
    }


def _metric_tensors_for_prediction(
    *,
    batch: Any,
    prediction: dict[str, Any],
    model: Any,
    modules: dict[str, Any],
) -> dict[str, Any]:
    compute_velocity_metrics = modules["compute_velocity_metrics"]
    residual_energy_ratio = modules["residual_energy_ratio"]
    true_residual_low_frequency_ratio = modules["true_residual_low_frequency_ratio"]
    velocity_hat = prediction["velocity_hat"]
    bg_hat = prediction["bg_hat"]
    target = batch.depth_vel
    metrics = compute_velocity_metrics(velocity_hat, target, model.filter)
    error = velocity_hat - target
    mse = (error ** 2).flatten(1).mean(dim=1)
    low, _ = model.anchors(target)
    bg_high = model.filter.highpass(bg_hat)
    rho_b = true_residual_low_frequency_ratio(model.filter, target, bg_hat)
    bg_mae = (bg_hat - low).abs().flatten(1).mean(dim=1)
    e_ratio = residual_energy_ratio(target, bg_hat)
    return {
        "mae": metrics["mae"],
        "rmse": metrics["rmse"],
        "mse": mse,
        "ssim": metrics["ssim"],
        "mae_l": metrics["mae_l"],
        "mae_h": metrics["mae_h"],
        "rho_B": rho_b,
        "bg_mae": bg_mae,
        "epsilon_H_B": bg_high.abs().flatten(1).mean(dim=1),
        "E_R_over_E_V": e_ratio,
        "rho_hat_B": prediction["rho_hat_b"],
        "alpha_hat": prediction["alpha_hat"],
    }


def _row_from_metric_tensors(
    *,
    path_name: str,
    sample_index: int,
    batch_idx: int,
    item_idx: int,
    batch: Any,
    dataset_lookup: dict[int, str],
    metric_tensors: dict[str, Any],
    elapsed_seconds: float,
) -> dict[str, Any]:
    dataset_id = _metadata_int(batch, "dataset_id", item_idx, 0)
    dataset_sample_index = _metadata_int(batch, "sample_index", item_idx, sample_index)
    dataset_name = dataset_lookup.get(dataset_id, f"dataset_{dataset_id}")
    return {
        "sample_index": sample_index,
        "path": path_name,
        "dataset_name": dataset_name,
        "dataset_id": dataset_id,
        "dataset_sample_index": dataset_sample_index,
        "batch_index": batch_idx,
        "item_index": item_idx,
        "mae": _tensor_to_float(metric_tensors["mae"], item_idx),
        "rmse": _tensor_to_float(metric_tensors["rmse"], item_idx),
        "mse": _tensor_to_float(metric_tensors["mse"], item_idx),
        "ssim": _tensor_to_float(metric_tensors["ssim"], item_idx),
        "mae_l": _tensor_to_float(metric_tensors["mae_l"], item_idx),
        "mae_h": _tensor_to_float(metric_tensors["mae_h"], item_idx),
        "rho_B": _tensor_to_float(metric_tensors["rho_B"], item_idx),
        "bg_mae": _tensor_to_float(metric_tensors["bg_mae"], item_idx),
        "epsilon_H_B": _tensor_to_float(metric_tensors["epsilon_H_B"], item_idx),
        "E_R_over_E_V": _tensor_to_float(metric_tensors["E_R_over_E_V"], item_idx),
        "rho_hat_B": _tensor_to_float(metric_tensors["rho_hat_B"], item_idx),
        "alpha_hat": _tensor_to_float(metric_tensors["alpha_hat"], item_idx),
        "inference_time": float(elapsed_seconds / max(int(batch.depth_vel.shape[0]), 1)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def group_rows(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = tuple(str(row[field]) for field in keys)
        buckets.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for key in sorted(buckets):
        group = buckets[key]
        out = {field: value for field, value in zip(keys, key, strict=True)}
        out["num_samples"] = len(group)
        out.update(mean_dict(group, METRIC_KEYS))
        output.append(out)
    return output


def rows_for_path(rows: list[dict[str, Any]], path_name: str) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("path") == path_name]


def _overall_by_path(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for path_name in ORACLE_PATHS:
        path_rows = [row for row in rows if row["path"] == path_name]
        values = mean_dict(path_rows, METRIC_KEYS)
        values["num_samples"] = float(len(path_rows))
        result[path_name] = values
    return result


def write_markdown_report(
    path: Path,
    *,
    summary: dict[str, Any],
    dataset_summary_rows: list[dict[str, Any]],
) -> None:
    diagnosis = summary["diagnosis"]
    overall = summary["overall_by_path"]
    lines = [
        "# BG-PDR-FM Oracle Upper-Bound Diagnostic",
        "",
        f"- Primary bottleneck: `{diagnosis['primary_bottleneck']}`",
        f"- Recommendation: {diagnosis['recommendation']}",
        f"- Num samples per path: `{int(overall['baseline']['num_samples'])}`",
        "",
        "## Overall",
        "",
        "| path | ssim | mae | mae_l | mae_h | rho_B | bg_mae |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for path_name in ORACLE_PATHS:
        row = overall[path_name]
        lines.append(
            f"| {path_name} | {row['ssim']:.6f} | {row['mae']:.6f} | {row['mae_l']:.6f} | "
            f"{row['mae_h']:.6f} | {row['rho_B']:.6f} | {row['bg_mae']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Weak Subsets",
            "",
            "| dataset | path | samples | ssim | mae_l | mae_h | rho_B |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in dataset_summary_rows:
        if row["dataset_name"] not in {"CurveVelB", "FlatVelB"}:
            continue
        lines.append(
            f"| {row['dataset_name']} | {row['path']} | {row['num_samples']} | {row['ssim']:.6f} | "
            f"{row['mae_l']:.6f} | {row['mae_h']:.6f} | {row['rho_B']:.6f} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_oracle_upper_bound(
    config_path: str | Path,
    output_dir: str | Path | None = None,
    max_batches: int | None = None,
    seed: int | None = 1234,
) -> dict[str, Any]:
    modules = _lazy_import_runtime_modules()
    torch = modules["torch"]
    omega_conf = modules["OmegaConf"]
    conf = omega_conf.load(config_path)
    apply_cli_overrides(omega_conf.update, conf, output_dir, max_batches)
    out_dir = Path(str(_conf_get(omega_conf, conf, "evaluation.output_dir", "logs/bg_pdr_fm/oracle_upper_bound")))
    out_dir.mkdir(parents=True, exist_ok=True)

    modules["configure_torch_runtime"](_conf_get(omega_conf, conf, "training.matmul_precision", None))
    loader = modules["build_evaluation_loader"](conf)
    model = modules["BGPDRFMLightning"](conf)
    checkpoint_info = modules["load_evaluation_checkpoints"](model, conf)
    device = modules["configured_torch_device"](
        _conf_get(omega_conf, conf, "training.accelerator", "auto"),
        _conf_get(omega_conf, conf, "training.devices", "auto"),
    )
    model.eval().to(device)
    dataset_lookup = _dataset_name_lookup(conf, loader.dataset, omega_conf)
    max_batches = _conf_get(omega_conf, conf, "evaluation.max_batches", None)
    max_batches = None if max_batches is None else int(max_batches)

    rows: list[dict[str, Any]] = []
    sample_index = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            batch = modules["batch_to_device"](model._as_batch(batch), device)
            current_seed = batch_seed(seed, batch_idx)
            if current_seed is not None:
                torch.manual_seed(current_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(current_seed)
            if torch.cuda.is_available() and str(device).startswith("cuda"):
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            predictions = _predict_oracle_paths(model, batch, torch)
            if torch.cuda.is_available() and str(device).startswith("cuda"):
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start
            metric_tensors = {
                path_name: _metric_tensors_for_prediction(
                    batch=batch,
                    prediction=prediction,
                    model=model,
                    modules=modules,
                )
                for path_name, prediction in predictions.items()
            }
            for item_idx in range(batch.depth_vel.shape[0]):
                for path_name in ORACLE_PATHS:
                    rows.append(
                        _row_from_metric_tensors(
                            path_name=path_name,
                            sample_index=sample_index,
                            batch_idx=batch_idx,
                            item_idx=item_idx,
                            batch=batch,
                            dataset_lookup=dataset_lookup,
                            metric_tensors=metric_tensors[path_name],
                            elapsed_seconds=elapsed,
                        )
                    )
                sample_index += 1

    metric_fields = [
        "sample_index",
        "path",
        "dataset_name",
        "dataset_id",
        "dataset_sample_index",
        "batch_index",
        "item_index",
        *METRIC_KEYS,
        "rho_hat_B",
        "alpha_hat",
        "inference_time",
    ]
    write_csv(out_dir / "metrics.csv", rows, metric_fields)
    path_summary_rows = group_rows(rows, ("path",))
    dataset_summary_rows = group_rows(rows, ("dataset_name", "path"))
    write_csv(out_dir / "path_summary.csv", path_summary_rows, ["path", "num_samples", *METRIC_KEYS])
    write_csv(
        out_dir / "dataset_summary.csv",
        dataset_summary_rows,
        ["dataset_name", "path", "num_samples", *METRIC_KEYS],
    )
    for path_name in ORACLE_PATHS:
        path_dir = out_dir / path_name
        path_rows = rows_for_path(rows, path_name)
        path_dataset_rows = group_rows(path_rows, ("dataset_name",))
        path_overall = mean_dict(path_rows, METRIC_KEYS)
        path_summary = {
            "path": path_name,
            "num_samples": len(path_rows),
            "means": path_overall,
        }
        write_csv(path_dir / "metrics.csv", path_rows, metric_fields)
        write_csv(path_dir / "dataset_summary.csv", path_dataset_rows,
                  ["dataset_name", "num_samples", *METRIC_KEYS])
        (path_dir / "summary.json").write_text(json.dumps(path_summary, indent=2), encoding="utf-8")

    overall = _overall_by_path(rows)
    summary = {
        "num_samples": sample_index,
        "num_prediction_rows": len(rows),
        "paths": list(ORACLE_PATHS),
        "overall_by_path": overall,
        "diagnosis": build_diagnosis(overall),
        "checkpoints": checkpoint_info,
        "config_path": str(config_path),
        "seed": seed,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown_report(out_dir / "oracle_upper_bound_report.md", summary=summary,
                          dataset_summary_rows=dataset_summary_rows)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run BG-PDR-FM oracle upper-bound diagnostics.")
    parser.add_argument("--config", required=True, help="Evaluation config YAML.")
    parser.add_argument("--output-dir", default=None, help="Override evaluation.output_dir.")
    parser.add_argument("--max-batches", type=int, default=None, help="Override evaluation.max_batches.")
    parser.add_argument("--seed", type=int, default=1234, help="Base seed for stochastic residual sampling.")
    args = parser.parse_args()
    summary = run_oracle_upper_bound(args.config, args.output_dir, args.max_batches, args.seed)
    print(json.dumps(summary["diagnosis"], indent=2))


if __name__ == "__main__":
    main()
