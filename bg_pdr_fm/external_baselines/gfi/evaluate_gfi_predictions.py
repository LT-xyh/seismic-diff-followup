"""Evaluate official GFI velocity predictions with BG-PDR-FM benchmark metrics.

This script is intentionally narrow: it does not train or reimplement GFI. It
only reads velocity prediction arrays exported by the official GFI code and
writes the same core metric schema used by the AAAI27 benchmark tables.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch

from bg_pdr_fm.evaluation.benchmark_metrics import compute_velocity_metrics
from bg_pdr_fm.models.filters import LowHighPassFilter


METRIC_FIELDS = ("mae", "rmse", "ssim", "mae_l", "mae_h")


def _load_velocity_array(path: Path, role: str) -> torch.Tensor:
    if not path.is_file():
        raise FileNotFoundError(f"{role} array not found: {path}")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    shape = tuple(int(dim) for dim in array.shape)
    if len(shape) == 3:
        array = array[:, None, :, :]
        shape = tuple(int(dim) for dim in array.shape)
    if len(shape) != 4 or shape[1:] != (1, 70, 70):
        raise ValueError(f"{role} array must have shape [N,1,70,70] or [N,70,70], got {shape}")
    return torch.from_numpy(np.array(array, copy=True)).float()


def _batched_indices(num_samples: int, batch_size: int, max_samples: int | None) -> Iterable[range]:
    limit = num_samples if max_samples is None else min(num_samples, int(max_samples))
    for start in range(0, limit, batch_size):
        yield range(start, min(start + batch_size, limit))


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def evaluate_arrays(
    prediction_path: Path,
    target_path: Path,
    output_dir: Path,
    *,
    batch_size: int = 128,
    max_samples: int | None = None,
    filter_kernel_size: int = 5,
) -> dict[str, object]:
    pred = _load_velocity_array(prediction_path, "prediction")
    target = _load_velocity_array(target_path, "target")
    if pred.shape[0] != target.shape[0]:
        raise ValueError(f"prediction/target sample counts differ: {pred.shape[0]} vs {target.shape[0]}")

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.csv"
    filter_module = LowHighPassFilter(kernel_size=filter_kernel_size)
    totals: dict[str, list[float]] = {key: [] for key in METRIC_FIELDS}
    rows: list[dict[str, object]] = []

    with metrics_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["sample_index", *METRIC_FIELDS]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for indices in _batched_indices(pred.shape[0], batch_size, max_samples):
            idx = list(indices)
            metrics = compute_velocity_metrics(pred[idx], target[idx], filter_module)
            for item_offset, sample_index in enumerate(idx):
                row: dict[str, object] = {"sample_index": int(sample_index)}
                for key in METRIC_FIELDS:
                    value = float(metrics[key][item_offset].detach().cpu())
                    row[key] = value
                    totals[key].append(value)
                writer.writerow(row)
                rows.append(row)

    means = {key: _mean(values) for key, values in totals.items()}
    summary: dict[str, object] = {
        "method": "gfi_external",
        "prediction_path": str(prediction_path),
        "target_path": str(target_path),
        "num_samples": len(rows),
        "array_shape": list(pred.shape),
        "metrics_path": str(metrics_path),
        "means": means,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred", type=Path, required=True, help="GFI prediction array [N,1,70,70] or [N,70,70].")
    parser.add_argument("--target", type=Path, required=True, help="Velocity target array [N,1,70,70] or [N,70,70].")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for metrics.csv and summary.json.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--filter-kernel-size", type=int, default=5)
    args = parser.parse_args()
    summary = evaluate_arrays(
        args.pred,
        args.target,
        args.output_dir,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        filter_kernel_size=args.filter_kernel_size,
    )
    print(json.dumps(summary["means"], indent=2))


if __name__ == "__main__":
    main()
