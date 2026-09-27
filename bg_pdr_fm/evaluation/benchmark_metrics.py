"""Shared metrics for AAAI27 benchmark comparisons."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from collections.abc import Mapping, Sequence
from typing import Any, Callable

import torch
import torch.nn.functional as F

from bg_pdr_fm.models.filters import LowHighPassFilter


METRIC_SCHEMA = ("mae", "rmse", "ssim", "mae_l", "mae_h")
METRIC_SAMPLE_COUNT_FIELDS = ("expected_records", "evaluated_records", "metric_rows")


BENCHMARK_METRIC_FIELDS = (
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
    "inference_time",
    "missing_mode",
)


@dataclass
class TimedPrediction:
    prediction: object
    elapsed_seconds: float


def validate_metric_row(row: Mapping[str, Any]) -> dict[str, float]:
    """Validate the paper-defined metrics for one evaluated sample."""
    validated: dict[str, float] = {}
    for metric_name in METRIC_SCHEMA:
        try:
            value = float(row[metric_name])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"missing or invalid core metric {metric_name}") from exc
        if not math.isfinite(value):
            raise ValueError(f"non-finite core metric {metric_name}: {value}")
        validated[metric_name] = value
    return validated


def validate_metric_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Validate every metric row and return its explicit row count."""
    for row in rows:
        validate_metric_row(row)
    return {"metric_rows": len(rows)}


def count_trainable_parameters(module: torch.nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad))


def count_parameters(module: torch.nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters()))


def _global_ssim(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_f = pred.flatten(1)
    target_f = target.flatten(1)
    mu_x = pred_f.mean(dim=1)
    mu_y = target_f.mean(dim=1)
    var_x = pred_f.var(dim=1, unbiased=False)
    var_y = target_f.var(dim=1, unbiased=False)
    cov = ((pred_f - mu_x[:, None]) * (target_f - mu_y[:, None])).mean(dim=1)
    value_range = (target_f.amax(dim=1) - target_f.amin(dim=1)).clamp_min(1.0)
    c1 = (0.01 * value_range) ** 2
    c2 = (0.03 * value_range) ** 2
    return ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / (
        (mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2)
    ).clamp_min(1e-8)


def compute_freq_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    filter_module: LowHighPassFilter,
) -> dict[str, torch.Tensor]:
    error = pred - target
    low = filter_module.lowpass(error)
    high = filter_module.highpass(error)
    return {
        "mae_l": low.abs().flatten(1).mean(dim=1),
        "mae_h": high.abs().flatten(1).mean(dim=1),
    }


def compute_velocity_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    filter_module: LowHighPassFilter,
) -> dict[str, torch.Tensor]:
    error = pred - target
    mse = (error ** 2).flatten(1).mean(dim=1)
    metrics = {
        "mae": error.abs().flatten(1).mean(dim=1),
        "rmse": mse.sqrt(),
        "ssim": _global_ssim(pred, target),
    }
    metrics.update(compute_freq_metrics(pred, target, filter_module))
    return metrics


def residual_energy_ratio(target: torch.Tensor, background: torch.Tensor | None) -> torch.Tensor:
    e_v = target.abs().flatten(1).mean(dim=1).clamp_min(1e-8)
    if background is None:
        return torch.ones_like(e_v)
    e_r = (target - background).abs().flatten(1).mean(dim=1)
    return e_r / e_v


def residual_energy_ratio_l2(
    target: torch.Tensor,
    background: torch.Tensor | None,
    eps: float = 1e-8,
) -> torch.Tensor:
    target_energy = target.square().flatten(1).mean(dim=1)
    if background is None:
        return torch.ones_like(target_energy)
    residual_energy = (target - background).square().flatten(1).mean(dim=1)
    return residual_energy / (target_energy + eps)


def transport_target_ratio(
    target: torch.Tensor,
    background: torch.Tensor | None,
) -> torch.Tensor:
    target_energy = target.square().flatten(1).mean(dim=1)
    if background is None:
        return torch.ones_like(target_energy)
    residual_energy = (target - background).square().flatten(1).mean(dim=1)
    return (residual_energy + 1.0) / (target_energy + 1.0)


def time_prediction(fn: Callable[[], object], device: torch.device | str | None = None) -> TimedPrediction:
    if device is not None and torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    prediction = fn()
    if device is not None and torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize(device)
    return TimedPrediction(prediction=prediction, elapsed_seconds=time.perf_counter() - start)
