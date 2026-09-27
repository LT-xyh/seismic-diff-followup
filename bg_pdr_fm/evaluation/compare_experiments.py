"""Unified AAAI27 benchmark evaluator."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from bg_pdr_fm.data import batch_to_device, collate_bg_samples, validate_bg_batch
from bg_pdr_fm.evaluation.benchmark_metrics import (
    compute_velocity_metrics,
    count_parameters,
    count_trainable_parameters,
    residual_energy_ratio,
    residual_energy_ratio_l2,
    time_prediction,
    transport_target_ratio,
)
from bg_pdr_fm.evaluation.engineering_stress import apply_engineering_stress, engineering_stress_from_config
from bg_pdr_fm.evaluation.missing_modalities import AAAI27_MISSING_MODES, apply_missing_modality_mode
from bg_pdr_fm.lightning import AAAI27BenchmarkLightning
from bg_pdr_fm.lightning.stage_losses import true_residual_low_frequency_ratio
from bg_pdr_fm.runtime import configure_torch_runtime, configured_torch_device
from bg_pdr_fm.training.benchmark_config import load_benchmark_config
from bg_pdr_fm.training.train_bg_pdr_fm import build_dataset


BENCHMARK_METRIC_FIELDNAMES = [
    "sample_index",
    "dataset_name",
    "dataset_id",
    "dataset_sample_index",
    "batch_index",
    "item_index",
    "method",
    "capacity_tier",
    "missing_mode",
    "stress_scenario",
    "mae",
    "rmse",
    "ssim",
    "mae_l",
    "mae_h",
    "rho_B",
    "rho_hat_B",
    "E_R_over_E_V",
    "residual_energy_ratio_l2",
    "transport_target_ratio",
    "params",
    "trainable_params",
    "generator_params",
    "trainable_generator_params",
    "inference_time",
    "residual_num_inference_steps",
]

SUMMARY_METRIC_KEYS = [
    "mae",
    "rmse",
    "ssim",
    "mae_l",
    "mae_h",
    "rho_B",
    "rho_hat_B",
    "E_R_over_E_V",
    "residual_energy_ratio_l2",
    "transport_target_ratio",
    "inference_time",
]

OFFICIAL_CHECKPOINT_PREFIXES = {
    "adapted_inversion_net": "adapted_inversion_net.",
    "adapted_upfwi": "adapted_upfwi.",
    "velocity_gan": "velocity_gan.",
}


def _conf_get(conf: Any, path: str, default: Any) -> Any:
    return OmegaConf.select(conf, path, default=default)


def _validate_official_checkpoint_state(
    model: torch.nn.Module,
    state_dict: dict[str, Any],
    *,
    variant: str,
    path: Path,
) -> None:
    prefix = OFFICIAL_CHECKPOINT_PREFIXES.get(variant)
    if prefix is None:
        return
    expected = {key: value for key, value in model.state_dict().items() if key.startswith(prefix)}
    if not expected:
        raise RuntimeError(f"Official baseline {variant!r} has no model keys under {prefix!r}.")
    missing = sorted(set(expected) - set(state_dict))
    if missing:
        preview = ", ".join(missing[:3])
        raise RuntimeError(
            f"Official baseline checkpoint {path} is missing {len(missing)} critical {variant} keys "
            f"(for example: {preview}). Legacy or partial checkpoints are not accepted."
        )
    mismatched = [
        key
        for key, expected_value in expected.items()
        if not hasattr(state_dict[key], "shape") or tuple(state_dict[key].shape) != tuple(expected_value.shape)
    ]
    if mismatched:
        preview = ", ".join(mismatched[:3])
        raise RuntimeError(
            f"Official baseline checkpoint {path} has {len(mismatched)} critical {variant} shape mismatches "
            f"(for example: {preview}). Legacy checkpoints are not accepted."
        )


def _load_checkpoint(model: torch.nn.Module, conf: Any) -> dict[str, Any]:
    checkpoint_path = _conf_get(conf, "evaluation.checkpoint", None)
    allow_untrained = bool(_conf_get(conf, "evaluation.allow_untrained", False))
    variant = str(_conf_get(conf, "benchmark.variant", "")).lower()
    if checkpoint_path is None or str(checkpoint_path).strip() == "":
        if variant == "smooth_dix":
            return {"mode": "analytic", "path": ""}
        if allow_untrained:
            return {"mode": "untrained", "path": ""}
        raise FileNotFoundError("Set evaluation.checkpoint or evaluation.allow_untrained: true for smoke evaluation.")
    path = Path(str(checkpoint_path))
    if not path.is_file():
        raise FileNotFoundError(f"Benchmark checkpoint not found: {path}")
    state = torch.load(path, map_location="cpu")
    state_dict = state.get("state_dict", state) if isinstance(state, dict) else state
    if not isinstance(state_dict, dict):
        raise TypeError(f"Benchmark checkpoint {path} did not contain a state dict.")
    _validate_official_checkpoint_state(model, state_dict, variant=variant, path=path)
    incompatible = model.load_state_dict(state_dict, strict=False)
    return {
        "mode": "checkpoint",
        "path": str(path),
        "loaded_keys": len(state_dict),
        "missing_keys": len(incompatible.missing_keys),
        "unexpected_keys": len(incompatible.unexpected_keys),
    }


def build_benchmark_loader(conf: Any) -> DataLoader:
    split = str(_conf_get(conf, "evaluation.split", "test"))
    dataset_split = "val" if split == "valid" else split
    dataset = build_dataset(conf, dataset_split)
    batch_size = int(_conf_get(conf, "training.batch_size", 2))
    num_workers = int(_conf_get(conf, "training.num_workers", 0))
    loader_kwargs = dict(
        num_workers=num_workers,
        pin_memory=bool(_conf_get(conf, "training.pin_memory", True)),
    )
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(_conf_get(conf, "training.persistent_workers", True))
        loader_kwargs["prefetch_factor"] = int(_conf_get(conf, "training.prefetch_factor", 3))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_bg_samples, **loader_kwargs)
    first_batch = next(iter(loader))
    validate_bg_batch(first_batch, context="aaai27 benchmark first batch")
    return loader


def _missing_modes(conf: Any) -> list[str]:
    modes = _conf_get(conf, "evaluation.missing_modes", None)
    if modes is None:
        return list(AAAI27_MISSING_MODES)
    modes = list(modes)
    unknown = [mode for mode in modes if mode not in AAAI27_MISSING_MODES]
    if unknown:
        raise ValueError(f"Unknown missing modes {unknown}; expected subset of {list(AAAI27_MISSING_MODES)}.")
    return modes


def _mean(values: list[float]) -> float:
    clean = [value for value in values if not math.isnan(value)]
    if not clean:
        return float("nan")
    return float(sum(clean) / len(clean))


def _dataset_name_lookup(conf: Any, dataset: Any) -> dict[int, str]:
    names = getattr(dataset, "dataset_names", None)
    if names is not None:
        return {idx: str(name) for idx, name in enumerate(tuple(names))}
    data_name = str(_conf_get(conf, "data.name", "synthetic"))
    if data_name == "marmousi":
        return {0: "Marmousi"}
    return {0: data_name}


def _metadata_int(batch: Any, key: str, item_idx: int, default: int) -> int:
    value = batch.metadata.get(key)
    if value is None:
        return int(default)
    return int(round(float(value[item_idx].detach().view(-1)[0].cpu())))


def _empty_metric_lists() -> dict[str, list[float]]:
    return {key: [] for key in SUMMARY_METRIC_KEYS}


def _update_summary_bucket(bucket: dict[str, list[float]], row: dict[str, Any]) -> None:
    for key in SUMMARY_METRIC_KEYS:
        bucket[key].append(float(row[key]))


def _mean_dict(values: dict[str, list[float]]) -> dict[str, float]:
    return {key: _mean(items) for key, items in values.items()}


def _write_group_summary_csv(
    path: Path,
    group_fieldnames: list[str],
    groups: dict[tuple[str, ...], dict[str, list[float]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group_key in sorted(groups):
        values = groups[group_key]
        row: dict[str, Any] = {
            field: value for field, value in zip(group_fieldnames, group_key, strict=True)
        }
        row["num_samples"] = len(values["mae"])
        row.update(_mean_dict(values))
        rows.append(row)
    fieldnames = [*group_fieldnames, "num_samples", *SUMMARY_METRIC_KEYS]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def run_benchmark_evaluation(conf: Any) -> dict[str, Any]:
    configure_torch_runtime(_conf_get(conf, "training.matmul_precision", None))
    output_dir = Path(str(_conf_get(conf, "evaluation.output_dir", "logs/bg_pdr_fm/aaai27/evaluation")))
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.csv"
    if metrics_path.exists():
        metrics_path.unlink()

    model = AAAI27BenchmarkLightning(conf)
    model_info = model.benchmark_metadata()
    checkpoint_info = _load_checkpoint(model, conf)
    generator_params = int(model_info["generator_params"])
    trainable_generator_params = int(model_info["trainable_generator_params"])
    params = int(model_info.get("reported_params", generator_params))
    trainable_params = trainable_generator_params
    device = configured_torch_device(
        _conf_get(conf, "training.accelerator", "auto"),
        _conf_get(conf, "training.devices", "auto"),
    )
    model.eval().to(device)
    loader = build_benchmark_loader(conf)
    dataset_lookup = _dataset_name_lookup(conf, loader.dataset)
    max_batches = _conf_get(conf, "evaluation.max_batches", None)
    max_batches = None if max_batches is None else int(max_batches)
    stress_scenario = engineering_stress_from_config(conf)
    stress_name = "" if stress_scenario is None else stress_scenario.name
    has_background = model.variant in {"two_stage_ddpm", "bg_pdr_fm"}
    totals = _empty_metric_lists()
    missing_mode_totals: dict[tuple[str, ...], dict[str, list[float]]] = {}
    dataset_totals: dict[tuple[str, ...], dict[str, list[float]]] = {}
    dataset_mode_totals: dict[tuple[str, str], dict[str, list[float]]] = {}
    sample_index = 0
    flush_every_rows = int(_conf_get(conf, "evaluation.flush_every_rows", 4096))
    with metrics_path.open("w", newline="", encoding="utf-8") as handle, torch.no_grad():
        writer = csv.DictWriter(handle, fieldnames=BENCHMARK_METRIC_FIELDNAMES)
        writer.writeheader()
        for missing_mode in _missing_modes(conf):
            for batch_idx, batch in enumerate(loader):
                if max_batches is not None and batch_idx >= max_batches:
                    break
                batch = apply_missing_modality_mode(batch, missing_mode)
                if stress_scenario is not None:
                    batch = apply_engineering_stress(batch, stress_scenario, batch_index=batch_idx)
                batch = batch_to_device(batch, device)
                timed = time_prediction(lambda: model.predict_batch(batch), device=device)
                prediction = timed.prediction
                metrics = compute_velocity_metrics(prediction.velocity_hat, batch.depth_vel, model.filter)
                rho_true = (
                    true_residual_low_frequency_ratio(model.filter, batch.depth_vel, prediction.bg_hat)
                    if has_background
                    else torch.full((batch.depth_vel.shape[0],), float("nan"), device=device)
                )
                energy_ratio = residual_energy_ratio(batch.depth_vel, prediction.bg_hat if has_background else None)
                energy_ratio_l2 = residual_energy_ratio_l2(
                    batch.depth_vel,
                    prediction.bg_hat if has_background else None,
                )
                target_ratio = transport_target_ratio(
                    batch.depth_vel,
                    prediction.bg_hat if has_background else None,
                )
                per_item_time = timed.elapsed_seconds / max(int(batch.depth_vel.shape[0]), 1)
                for item_idx in range(batch.depth_vel.shape[0]):
                    dataset_id = _metadata_int(batch, "dataset_id", item_idx, 0)
                    dataset_sample_index = _metadata_int(batch, "sample_index", item_idx, sample_index)
                    dataset_name = dataset_lookup.get(dataset_id, f"dataset_{dataset_id}")
                    row = {
                        "sample_index": sample_index,
                        "dataset_name": dataset_name,
                        "dataset_id": dataset_id,
                        "dataset_sample_index": dataset_sample_index,
                        "batch_index": batch_idx,
                        "item_index": item_idx,
                        "method": model.variant,
                        "capacity_tier": model.capacity_tier,
                        "missing_mode": missing_mode,
                        "stress_scenario": stress_name,
                        "mae": float(metrics["mae"][item_idx].detach().cpu()),
                        "rmse": float(metrics["rmse"][item_idx].detach().cpu()),
                        "ssim": float(metrics["ssim"][item_idx].detach().cpu()),
                        "mae_l": float(metrics["mae_l"][item_idx].detach().cpu()),
                        "mae_h": float(metrics["mae_h"][item_idx].detach().cpu()),
                        "rho_B": float(rho_true[item_idx].detach().cpu()),
                        "rho_hat_B": float(prediction.rho_hat_b[item_idx].detach().cpu()),
                        "E_R_over_E_V": float(energy_ratio[item_idx].detach().cpu()),
                        "residual_energy_ratio_l2": float(energy_ratio_l2[item_idx].detach().cpu()),
                        "transport_target_ratio": float(target_ratio[item_idx].detach().cpu()),
                        "params": params,
                        "trainable_params": trainable_params,
                        "generator_params": generator_params,
                        "trainable_generator_params": trainable_generator_params,
                        "inference_time": float(per_item_time),
                        "residual_num_inference_steps": int(model.residual_num_inference_steps),
                    }
                    writer.writerow(row)
                    _update_summary_bucket(totals, row)
                    _update_summary_bucket(
                        missing_mode_totals.setdefault((missing_mode,), _empty_metric_lists()),
                        row,
                    )
                    _update_summary_bucket(
                        dataset_totals.setdefault((dataset_name,), _empty_metric_lists()),
                        row,
                    )
                    _update_summary_bucket(
                        dataset_mode_totals.setdefault((dataset_name, missing_mode), _empty_metric_lists()),
                        row,
                    )
                    sample_index += 1
                    if flush_every_rows > 0 and sample_index % flush_every_rows == 0:
                        handle.flush()
        handle.flush()
    missing_mode_rows = _write_group_summary_csv(
        output_dir / "missing_mode_summary.csv",
        ["missing_mode"],
        missing_mode_totals,
    )
    dataset_rows = _write_group_summary_csv(output_dir / "dataset_summary.csv", ["dataset_name"], dataset_totals)
    dataset_mode_rows = _write_group_summary_csv(
        output_dir / "dataset_missing_mode_summary.csv",
        ["dataset_name", "missing_mode"],
        dataset_mode_totals,
    )
    summary = {
        "num_samples": sample_index,
        "method": model.variant,
        "capacity_tier": model.capacity_tier,
        "means": _mean_dict(totals),
        "missing_mode_means": missing_mode_rows,
        "dataset_means": dataset_rows,
        "dataset_missing_mode_means": dataset_mode_rows,
        "params": params,
        "trainable_params": trainable_params,
        "generator_params": generator_params,
        "trainable_generator_params": trainable_generator_params,
        "missing_modes": _missing_modes(conf),
        "engineering_stress": None
        if stress_scenario is None
        else {
            "name": stress_scenario.name,
            "pstm_snr_db": stress_scenario.pstm_snr_db,
            "rms_scale_max": stress_scenario.rms_scale_max,
            "rms_smooth_noise_std_frac": stress_scenario.rms_smooth_noise_std_frac,
            "rms_lowfreq_bias_std_frac": stress_scenario.rms_lowfreq_bias_std_frac,
            "horizon_missing_prob": stress_scenario.horizon_missing_prob,
            "well_missing_prob": stress_scenario.well_missing_prob,
            "seed": stress_scenario.seed,
        },
        "checkpoint": checkpoint_info,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(config_path: str) -> dict[str, Any]:
    conf = load_benchmark_config(config_path)
    return run_benchmark_evaluation(conf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate AAAI27 benchmark variants with a unified schema.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
