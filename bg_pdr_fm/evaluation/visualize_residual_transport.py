"""Generate the publication-facing residual-transport Figure 4.

The module keeps the two evidence sources separate:

* panel (a) uses one deterministically selected best-MAE held-out example per
  OpenFWI subset and bounded checkpoint replay for the branch decomposition;
* panel (b) uses the complete manifest-backed per-sample metric table for the
  transport-energy ratio ``q_T``.

The best-case examples are intentionally not treated as representative.  All
selection and provenance decisions are written to CSV/JSON beside the figure.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

from bg_pdr_fm.data import batch_to_device, collate_bg_samples
from bg_pdr_fm.data.normalization_profiles import get_normalization_profile
from bg_pdr_fm.evaluation.evaluate_bg_pdr_fm import _load_full_checkpoint
from bg_pdr_fm.lightning import BGPDRFMLightning
from bg_pdr_fm.runtime import configure_torch_runtime, configured_torch_device
from bg_pdr_fm.training.train_bg_pdr_fm import build_dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
SUBSET_ORDER = (
    "FlatVelA",
    "FlatVelB",
    "CurveVelA",
    "CurveVelB",
    "FlatFaultA",
    "FlatFaultB",
    "CurveFaultA",
    "CurveFaultB",
)

VELOCITY_KEYS = ("target", "background", "final")
RESIDUAL_KEYS = ("target_residual", "generated_residual")

BACKGROUND_GREEN = "#178F48"
STRUCTURE_BLUE = "#1E63D5"
DIAGNOSTIC_PURPLE = "#8357A6"
ERROR_RED = "#C44E52"
NEUTRAL_GRAY = "#8A929B"
FONT_FAMILY = "DejaVu Serif"


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _read_rows(path: str | Path) -> list[dict[str, str]]:
    path = _resolve_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"CSV artifact does not exist: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _float(row: Mapping[str, Any], key: str) -> float:
    value = row.get(key)
    if value is None or str(value).strip() == "":
        raise ValueError(f"Required numeric field {key!r} is missing from row: {row}")
    return float(value)


def _int(row: Mapping[str, Any], key: str) -> int:
    value = row.get(key)
    if value is None or str(value).strip() == "":
        raise ValueError(f"Required integer field {key!r} is missing from row: {row}")
    return int(float(value))


def _metric_qt(row: Mapping[str, Any]) -> float:
    value = row.get("transport_target_ratio", row.get("q_T", row.get("q_t")))
    if value is None or str(value).strip() == "":
        raise ValueError(f"Required q_T metric is missing from row: {row}")
    return float(value)


def _metric_background_mae(row: Mapping[str, Any]) -> float:
    value = row.get("bg_mae", row.get("background_mae"))
    if value is None or str(value).strip() == "":
        raise ValueError(f"Required background MAE metric is missing from row: {row}")
    return float(value)


def _identity(row: Mapping[str, Any]) -> tuple[int, str, int]:
    return (_int(row, "dataset_id"), str(row.get("dataset_name", "")), _int(row, "source_sample_index"))


def compute_q_t(target: np.ndarray | torch.Tensor, background: np.ndarray | torch.Tensor) -> float | np.ndarray:
    """Compute the samplewise theorem ratio in the normalized coordinates.

    A two-dimensional input is interpreted as one sample.  For an input with
    three or more dimensions, axis zero is interpreted as the sample axis.
    """
    target_array = np.asarray(target.detach().cpu() if torch.is_tensor(target) else target, dtype=np.float64)
    background_array = np.asarray(
        background.detach().cpu() if torch.is_tensor(background) else background,
        dtype=np.float64,
    )
    if target_array.shape != background_array.shape:
        raise ValueError(
            f"target and background must have identical shapes, got {target_array.shape} and {background_array.shape}"
        )
    if target_array.ndim < 2:
        raise ValueError(f"target/background must be at least two-dimensional, got {target_array.shape}")
    if target_array.ndim == 2:
        target_energy = np.mean(np.square(target_array))
        residual_energy = np.mean(np.square(target_array - background_array))
    else:
        flattened_target = target_array.reshape(target_array.shape[0], -1)
        flattened_residual = (target_array - background_array).reshape(target_array.shape[0], -1)
        target_energy = np.mean(np.square(flattened_target), axis=1)
        residual_energy = np.mean(np.square(flattened_residual), axis=1)
    value = (residual_energy + 1.0) / (target_energy + 1.0)
    if np.any(~np.isfinite(value)) or np.any(value <= 0.0):
        raise ValueError(f"q_T must be finite and positive, got {value}")
    if np.ndim(value) == 0:
        return float(value)
    return np.asarray(value, dtype=np.float64)


def validate_replay_metrics(
    target: np.ndarray,
    background: np.ndarray,
    final: np.ndarray,
    metric_row: Mapping[str, Any],
    *,
    atol: float = 5e-5,
    rtol: float = 5e-4,
    strict_mae: bool = True,
) -> dict[str, float]:
    """Cross-check a replay against the corresponding formal metric row."""
    target = np.asarray(target, dtype=np.float64)
    background = np.asarray(background, dtype=np.float64)
    final = np.asarray(final, dtype=np.float64)
    if target.shape != background.shape or target.shape != final.shape:
        raise ValueError("Replay target, background, and final arrays must have identical shapes.")
    if not all(np.all(np.isfinite(array)) for array in (target, background, final)):
        raise ValueError("Replay arrays contain non-finite values.")
    q_t_replay = float(compute_q_t(target, background))
    mae_replay = float(np.mean(np.abs(final - target)))
    q_t_formal = _metric_qt(metric_row)
    mae_formal = _float(metric_row, "mae")
    if not np.isclose(q_t_replay, q_t_formal, atol=atol, rtol=rtol):
        raise ValueError(f"Replay q_T {q_t_replay} disagrees with formal q_T {q_t_formal}")
    mae_matches = bool(np.isclose(mae_replay, mae_formal, atol=atol, rtol=rtol))
    if strict_mae and not mae_matches:
        raise ValueError(f"Replay MAE {mae_replay} disagrees with formal MAE {mae_formal}")
    return {
        "q_t_replay": q_t_replay,
        "q_t_formal": q_t_formal,
        "q_t_delta": q_t_replay - q_t_formal,
        "mae_replay": mae_replay,
        "mae_formal": mae_formal,
        "mae_delta": mae_replay - mae_formal,
        "mae_matches_formal": float(mae_matches),
    }


def _selection_key(row: Mapping[str, Any]) -> tuple[float, float, float, int]:
    return (_float(row, "mae"), _float(row, "rmse"), -_float(row, "ssim"), _int(row, "source_sample_index"))


def select_best_cases(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Select one lowest-MAE row per subset using deterministic tie breakers."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        dataset_name = str(row.get("dataset_name", "")).strip()
        if dataset_name not in SUBSET_ORDER:
            continue
        grouped[dataset_name].append(dict(row))
    missing = [name for name in SUBSET_ORDER if not grouped.get(name)]
    if missing:
        raise ValueError(f"Formal metrics are missing OpenFWI subsets: {missing}")
    selected: dict[str, dict[str, Any]] = {}
    for dataset_name in SUBSET_ORDER:
        selected[dataset_name] = min(grouped[dataset_name], key=_selection_key)
    return selected


def validate_heldout_identity(
    manifest_path: str | Path,
    metrics_path: str | Path,
    *,
    expected_count: int = 33_600,
) -> dict[str, Any]:
    """Validate exact identity equality between manifest and metric rows."""
    manifest_rows = _read_rows(manifest_path)
    metric_rows = _read_rows(metrics_path)

    def unique_identities(rows: Sequence[Mapping[str, Any]], label: str) -> set[tuple[int, str, int]]:
        identities = [_identity(row) for row in rows]
        if len(set(identities)) != len(identities):
            duplicates = len(identities) - len(set(identities))
            raise ValueError(f"duplicate held-out identities in {label}: {duplicates}")
        return set(identities)

    manifest_ids = unique_identities(manifest_rows, "manifest")
    metric_ids = unique_identities(metric_rows, "metrics")
    if len(manifest_rows) != int(expected_count):
        raise ValueError(f"held-out manifest has {len(manifest_rows)} rows; expected {expected_count}")
    if len(metric_rows) != int(expected_count):
        raise ValueError(f"merged metrics has {len(metric_rows)} rows; expected {expected_count}")
    if manifest_ids != metric_ids:
        missing = sorted(manifest_ids - metric_ids)[:5]
        unexpected = sorted(metric_ids - manifest_ids)[:5]
        raise ValueError(f"held-out identity mismatch; missing={missing}, unexpected={unexpected}")
    dataset_counts = {name: sum(1 for row in manifest_rows if row.get("dataset_name") == name) for name in SUBSET_ORDER}
    if any(count == 0 for count in dataset_counts.values()):
        raise ValueError(f"held-out manifest does not contain all OpenFWI subsets: {dataset_counts}")
    return {
        "manifest_rows": len(manifest_rows),
        "metric_rows": len(metric_rows),
        "unique_identities": len(manifest_ids),
        "dataset_counts": dataset_counts,
        "missing_test": 0,
        "unexpected": 0,
    }


def validate_split_identity(
    train_rows: Sequence[Mapping[str, Any]],
    val_rows: Sequence[Mapping[str, Any]],
    test_rows: Sequence[Mapping[str, Any]],
    manifest_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Verify split disjointness and that the manifest is exactly the test set."""

    def identity_set(rows: Sequence[Mapping[str, Any]], label: str) -> set[tuple[int, str, int]]:
        identities = [_identity(row) for row in rows]
        if len(set(identities)) != len(identities):
            raise ValueError(f"duplicate identities in {label}")
        return set(identities)

    train_ids = identity_set(train_rows, "train")
    val_ids = identity_set(val_rows, "val")
    test_ids = identity_set(test_rows, "test")
    manifest_ids = identity_set(manifest_rows, "manifest")
    train_val_overlap = train_ids & val_ids
    train_test_overlap = train_ids & test_ids
    val_test_overlap = val_ids & test_ids
    if train_val_overlap or train_test_overlap or val_test_overlap:
        raise ValueError(
            "split overlap detected: "
            f"train_val={len(train_val_overlap)}, "
            f"train_test={len(train_test_overlap)}, "
            f"val_test={len(val_test_overlap)}"
        )
    if test_ids != manifest_ids:
        raise ValueError(
            "test/manifest identity mismatch: "
            f"missing={len(manifest_ids - test_ids)}, unexpected={len(test_ids - manifest_ids)}"
        )
    return {
        "train_count": len(train_ids),
        "val_count": len(val_ids),
        "test_count": len(test_ids),
        "manifest_count": len(manifest_ids),
        "overlap_train": 0,
        "overlap_val": 0,
        "train_val_overlap": 0,
        "test_manifest_missing": 0,
        "test_manifest_unexpected": 0,
    }


def shared_color_limits(cases: Iterable[Mapping[str, np.ndarray]]) -> dict[str, tuple[float, float]]:
    """Get global velocity, residual, and error limits for all selected tiles."""
    velocity_values: list[np.ndarray] = []
    residual_values: list[np.ndarray] = []
    error_values: list[np.ndarray] = []
    for case in cases:
        velocity_values.extend(np.asarray(case[key], dtype=np.float64).ravel() for key in VELOCITY_KEYS)
        residual_values.extend(np.asarray(case[key], dtype=np.float64).ravel() for key in RESIDUAL_KEYS)
        error_values.append(np.asarray(case["absolute_error"], dtype=np.float64).ravel())
    if not velocity_values or not residual_values or not error_values:
        raise ValueError("At least one complete case is required to compute color limits.")
    velocity = np.concatenate(velocity_values)
    residual = np.concatenate(residual_values)
    errors = np.concatenate(error_values)
    if not all(np.all(np.isfinite(values)) for values in (velocity, residual, errors)):
        raise ValueError("Color-limit inputs contain non-finite values.")
    residual_bound = float(np.max(np.abs(residual)))
    return {
        "velocity": (float(np.min(velocity)), float(np.max(velocity))),
        "residual": (-residual_bound, residual_bound),
        "error": (0.0, float(np.max(errors))),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        names: list[str] = []
        for row in rows:
            for key in row:
                if key not in names:
                    names.append(key)
        fieldnames = names
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    raise TypeError(f"Object is not JSON serializable: {type(value).__name__}")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default), encoding="utf-8")


def _bootstrap_mean_ci(values: np.ndarray, *, seed: int, repeats: int = 2000) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("Cannot bootstrap an empty sample.")
    rng = np.random.default_rng(int(seed))
    means = np.empty(int(repeats), dtype=np.float64)
    chunk = 32
    for start in range(0, int(repeats), chunk):
        count = min(chunk, int(repeats) - start)
        indices = rng.integers(0, values.size, size=(count, values.size))
        means[start : start + count] = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _summarize_qt(
    qt_rows: Sequence[Mapping[str, Any]],
    *,
    bootstrap_seed: int,
    bootstrap_repeats: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    def row_qt(row: Mapping[str, Any]) -> float:
        # Accept the legacy lowercase field when reading hand-built test rows,
        # but write the paper-facing artifact with the theorem's q_T spelling.
        value = row.get("q_T", row.get("q_t"))
        if value is None or str(value).strip() == "":
            raise ValueError(f"q_T is missing from row: {row}")
        return float(value)

    grouped: dict[str, list[float]] = defaultdict(list)
    for row in qt_rows:
        name = str(row.get("dataset_name", ""))
        if name not in SUBSET_ORDER:
            continue
        value = row_qt(row)
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"Invalid q_T for {name}: {value}")
        grouped[name].append(value)
    missing = [name for name in SUBSET_ORDER if not grouped.get(name)]
    if missing:
        raise ValueError(f"q_T rows are missing subsets: {missing}")
    summary: list[dict[str, Any]] = []
    for name in SUBSET_ORDER:
        values = np.asarray(grouped[name], dtype=np.float64)
        ci = _bootstrap_mean_ci(values, seed=bootstrap_seed + SUBSET_ORDER.index(name), repeats=bootstrap_repeats)
        summary.append(
            {
                "dataset_name": name,
                "n": int(values.size),
                "mean_q_t": float(values.mean()),
                "median_q_t": float(np.median(values)),
                "std_q_t": float(values.std(ddof=1)) if values.size > 1 else 0.0,
                "fraction_q_t_lt_1": float(np.mean(values < 1.0)),
                "mean_q_t_ci_low": ci[0],
                "mean_q_t_ci_high": ci[1],
            }
        )
    all_values = np.asarray([value for name in SUBSET_ORDER for value in grouped[name]], dtype=np.float64)
    global_ci = _bootstrap_mean_ci(all_values, seed=bootstrap_seed, repeats=bootstrap_repeats)
    global_summary = {
        "n": int(all_values.size),
        "mean_q_t": float(all_values.mean()),
        "median_q_t": float(np.median(all_values)),
        "std_q_t": float(all_values.std(ddof=1)) if all_values.size > 1 else 0.0,
        "fraction_q_t_lt_1": float(np.mean(all_values < 1.0)),
        "mean_q_t_ci_low": global_ci[0],
        "mean_q_t_ci_high": global_ci[1],
        "bootstrap_seed": int(bootstrap_seed),
        "bootstrap_repeats": int(bootstrap_repeats),
        "aggregation": "sample-level bootstrap over per-record q_T",
    }
    return summary, global_summary


def _canonicalize_qt_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    canonical: list[dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        if "q_T" not in row or str(row.get("q_T", "")).strip() == "":
            if "q_t" not in row or str(row.get("q_t", "")).strip() == "":
                raise ValueError(f"q_T is missing from row: {row}")
            row["q_T"] = row["q_t"]
        row.pop("q_t", None)
        canonical.append(row)
    return canonical


def _safe_limits(limits: tuple[float, float], *, symmetric: bool = False, minimum_span: float = 1e-8) -> tuple[float, float]:
    low, high = map(float, limits)
    if symmetric:
        bound = max(abs(low), abs(high), minimum_span)
        return -bound, bound
    if high - low < minimum_span:
        center = (high + low) / 2.0
        return center - minimum_span / 2.0, center + minimum_span / 2.0
    return low, high


def _draw_tile(
    ax: plt.Axes,
    image: np.ndarray,
    *,
    cmap: str,
    limits: tuple[float, float],
    title: str | None,
    show_x: bool,
    show_y: bool,
    show_ylabel: bool = False,
) -> mpl.image.AxesImage:
    height, width = np.asarray(image).shape[-2:]
    artist = ax.imshow(
        np.asarray(image),
        cmap=cmap,
        vmin=limits[0],
        vmax=limits[1],
        origin="upper",
        interpolation="nearest",
        aspect="equal",
    )
    if title:
        ax.set_title(title, fontsize=5.8, pad=2.0, fontweight="normal")
    ax.tick_params(labelsize=6.5, length=2, pad=1)
    if not show_x:
        ax.set_xticks([])
    else:
        ax.set_xticks([0, width - 1])
        ax.set_xticklabels(["0", f"{width - 1:g}"], fontsize=6.5)
        ax.set_xlabel("x", fontsize=6.5, labelpad=1)
    if not show_y:
        ax.set_yticks([])
    else:
        ax.set_yticks([0, height - 1])
        ax.set_yticklabels(["0", f"{height - 1:g}"], fontsize=6.5)
        if show_ylabel:
            ax.set_ylabel("z", fontsize=6.5, labelpad=1)
    for spine in ax.spines.values():
        spine.set_linewidth(0.35)
        spine.set_color("#525866")
    return artist


def _plot_panel_a(
    fig: plt.Figure,
    spec: Any,
    best_cases: Mapping[str, Mapping[str, Any]],
    limits: Mapping[str, tuple[float, float]],
) -> tuple[list[mpl.image.AxesImage], list[mpl.image.AxesImage], list[mpl.image.AxesImage]]:
    axes = np.empty((4, 13), dtype=object)
    for row_idx in range(4):
        for col_idx in range(13):
            axes[row_idx, col_idx] = fig.add_subplot(spec[row_idx, col_idx])
            if col_idx == 6:
                axes[row_idx, col_idx].axis("off")
    velocity_artists: list[mpl.image.AxesImage] = []
    residual_artists: list[mpl.image.AxesImage] = []
    error_artists: list[mpl.image.AxesImage] = []
    columns = (
        ("target", "V", "velocity"),
        ("background", r"$\hat{B}$", "velocity"),
        ("target_residual", r"$V-\hat{B}$", "residual"),
        ("generated_residual", r"$\hat{R}$", "residual"),
        ("final", r"$\hat{B}+\hat{R}$", "velocity"),
        ("absolute_error", r"$|\hat{V}-V|$", "error"),
    )
    block_starts = (0, 7)
    block_names = (("Velocity subsets", SUBSET_ORDER[:4]), ("Fault subsets", SUBSET_ORDER[4:]))
    for start, (block_label, names) in zip(block_starts, block_names):
        for row_idx, dataset_name in enumerate(names):
            case = best_cases[dataset_name]
            for col_idx, (key, title, category) in enumerate(columns):
                ax = axes[row_idx, start + col_idx]
                artist = _draw_tile(
                    ax,
                    np.asarray(case[key]),
                    cmap={"velocity": "viridis", "residual": "coolwarm", "error": "magma"}[category],
                    limits=limits[category],
                    title=title if row_idx == 0 else None,
                    show_x=row_idx == 3,
                    show_y=start == 0 and col_idx == 0,
                    show_ylabel=start == 0 and col_idx == 0 and row_idx == 1,
                )
                if category == "velocity":
                    velocity_artists.append(artist)
                elif category == "residual":
                    residual_artists.append(artist)
                else:
                    error_artists.append(artist)
            axes[row_idx, start].text(
                -0.75,
                0.5,
                dataset_name,
                transform=axes[row_idx, start].transAxes,
                fontsize=6.5,
                color="#30343B",
                ha="right",
                va="center",
            )
            metric_lines = [f"formal {float(case['mae']):.4f}"]
            if "replay_mae" in case:
                metric_lines.append(f"replay {float(case['replay_mae']):.4f}")
            metric_lines.append(f"$q_T$ {float(case['q_t']):.3f}")
            metric_text = "\n".join(metric_lines)
            axes[row_idx, start + 5].text(
                0.97,
                0.04,
                metric_text,
                transform=axes[row_idx, start + 5].transAxes,
                fontsize=6.5,
                color="white",
                ha="right",
                va="bottom",
                bbox={"facecolor": "#20242A", "alpha": 0.48, "edgecolor": "none", "pad": 1.0},
            )
    fig.text(0.013, 0.982, "(a) Formal best-MAE identities: fixed-seed residual replay", fontsize=9.0, fontweight="bold", va="top")
    return velocity_artists, residual_artists, error_artists


def _plot_panel_b(
    fig: plt.Figure,
    spec: Any,
    qt_rows: Sequence[Mapping[str, Any]],
    *,
    global_summary: Mapping[str, Any] | None = None,
    panel_label: str | None = None,
) -> plt.Axes:
    ax = fig.add_subplot(spec)
    values_by_subset = [
        np.asarray(
            [float(row.get("q_T", row.get("q_t"))) for row in qt_rows if row.get("dataset_name") == name],
            dtype=np.float64,
        )
        for name in SUBSET_ORDER
    ]
    if any(values.size == 0 for values in values_by_subset):
        raise ValueError("Panel (b) requires q_T values for all eight subsets.")
    positions = np.arange(1, len(SUBSET_ORDER) + 1)
    violin = ax.violinplot(values_by_subset, positions=positions, widths=0.78, showextrema=False, showmedians=True)
    for body, color in zip(violin["bodies"], (BACKGROUND_GREEN, BACKGROUND_GREEN, BACKGROUND_GREEN, BACKGROUND_GREEN, STRUCTURE_BLUE, STRUCTURE_BLUE, STRUCTURE_BLUE, STRUCTURE_BLUE)):
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.24)
    violin["cmedians"].set_color("#30343B")
    violin["cmedians"].set_linewidth(0.9)
    box = ax.boxplot(
        values_by_subset,
        positions=positions,
        widths=0.18,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#20242A", "linewidth": 0.9},
        whiskerprops={"color": "#525866", "linewidth": 0.55},
        capprops={"color": "#525866", "linewidth": 0.55},
    )
    for patch in box["boxes"]:
        patch.set_facecolor("white")
        patch.set_edgecolor("#525866")
        patch.set_linewidth(0.55)
    ax.axhline(1.0, color=DIAGNOSTIC_PURPLE, linestyle=(0, (3, 2)), linewidth=0.85, label=r"$q_T=1$")
    fractions = [float(np.mean(values < 1.0)) for values in values_by_subset]
    for position, values, fraction in zip(positions, values_by_subset, fractions):
        ax.text(
            position,
            1.018,
            rf"$q_T<1$: {fraction:.0%}",
            fontsize=6.2,
            ha="center",
            va="bottom",
            color="#404650",
        )
    ax.set_xticks(positions)
    ax.set_xticklabels([name.replace("Curve", "C-").replace("Flat", "F-") for name in SUBSET_ORDER], rotation=22, ha="right", fontsize=6.5)
    ax.set_ylabel(r"normalized transport burden $q_T$", fontsize=7.0)
    ax.set_xlabel("OpenFWI subset", fontsize=7.0)
    ax.tick_params(axis="y", labelsize=6.5, length=2)
    ax.grid(axis="y", color="#9AA1AA", alpha=0.22, linewidth=0.5)
    ax.legend(loc="upper right", fontsize=6.5, frameon=False)
    prefix = f"{panel_label} " if panel_label else ""
    ax.set_title(
        f"{prefix}Full-test transport-energy condition | all 33,600 held-out records",
        fontsize=8.5,
        loc="left",
        pad=3,
    )
    ax.set_ylim(min(0.5, float(min(np.min(values) for values in values_by_subset)) - 0.02), 1.13)
    global_values = np.asarray([value for values in values_by_subset for value in values], dtype=np.float64)
    if global_summary is None:
        global_ci = _bootstrap_mean_ci(global_values, seed=2027, repeats=400)
        global_fraction = float(np.mean(global_values < 1.0))
        global_mean = float(np.mean(global_values))
    else:
        global_ci = (float(global_summary["mean_q_t_ci_low"]), float(global_summary["mean_q_t_ci_high"]))
        global_fraction = float(global_summary["fraction_q_t_lt_1"])
        global_mean = float(global_summary["mean_q_t"])
    ax.text(
        0.01,
        1.085,
        f"global $q_T<1$: {global_fraction:.1%}  |  "
        f"mean $q_T$: {global_mean:.3f} "
        f"[{global_ci[0]:.3f}, {global_ci[1]:.3f}]",
        transform=ax.get_yaxis_transform(),
        ha="left",
        va="bottom",
        fontsize=6.5,
        color="#454B55",
    )
    for spine in ax.spines.values():
        spine.set_linewidth(0.55)
        spine.set_color("#525866")
    return ax


def _case_row(dataset_name: str, case: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "dataset_name": dataset_name,
        "dataset_id": case.get("dataset_id", ""),
        "source_sample_index": case.get("source_sample_index", ""),
        "checkpoint": case.get("checkpoint", ""),
        "checkpoint_sha256": case.get("checkpoint_sha256", ""),
        "mae": case.get("mae", ""),
        "rmse": case.get("rmse", ""),
        "ssim": case.get("ssim", ""),
        "q_T": case.get("q_t", ""),
        "background_mae": case.get("background_mae", ""),
        "replay_mae": case.get("replay_mae", ""),
        "replay_mae_delta": case.get("replay_mae_delta", ""),
        "displayed_prediction_source": case.get("displayed_prediction_source", ""),
        "selection_rank": case.get("selection_rank", 1),
        "normalization_profile": case.get("normalization_profile", ""),
        "normalization_mode": case.get("normalization_mode", ""),
    }


def plot_figure4(
    output_dir: str | Path,
    *,
    best_cases: Mapping[str, Mapping[str, Any]],
    qt_rows: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any] | None = None,
    bootstrap_seed: int = 2027,
    bootstrap_repeats: int = 2000,
) -> list[Path]:
    """Render Figure 4 and write raw summary assets."""
    output_dir = _resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    missing = [name for name in SUBSET_ORDER if name not in best_cases]
    if missing:
        raise ValueError(f"Figure 4 needs one best case per subset; missing {missing}")
    best_cases = {name: dict(best_cases[name]) for name in SUBSET_ORDER}
    limits = shared_color_limits(best_cases.values())
    qt_rows = _canonicalize_qt_rows(qt_rows)
    subset_summary, global_summary = _summarize_qt(
        qt_rows,
        bootstrap_seed=bootstrap_seed,
        bootstrap_repeats=bootstrap_repeats,
    )
    _write_csv(output_dir / "best_case_selection.csv", [_case_row(name, best_cases[name]) for name in SUBSET_ORDER])
    _write_csv(output_dir / "qt_samples.csv", qt_rows)
    _write_csv(output_dir / "qt_subset_summary.csv", subset_summary)

    arrays: dict[str, np.ndarray] = {}
    for name in SUBSET_ORDER:
        case = best_cases[name]
        for key in ("target", "background", "target_residual", "generated_residual", "final", "absolute_error"):
            arrays[f"{name}__{key}"] = np.asarray(case[key], dtype=np.float32)
    np.savez_compressed(output_dir / "replay_arrays.npz", **arrays)

    metadata_payload = dict(metadata or {})
    metadata_payload.update(
        {
            "figure": "Figure 4",
            "title": "Empirical Characterization of Background-Guided Residual Transport",
            "subset_order": list(SUBSET_ORDER),
            "panel_a_selection": "argmin (MAE, RMSE, -SSIM, source_sample_index) per subset",
            "panel_a_is_best_case_not_representative": True,
            "panel_a_replay_note": "Identities are selected from formal metrics; residual/final tiles are fixed-seed bounded replays because the formal evaluator did not save prediction arrays or sampler RNG states.",
            "main_figure_content": "full-test q_T distribution only",
            "branch_decomposition_in_main_figure": False,
            "panel_b_record_count": len(qt_rows),
            "q_t_definition": "(mean((V - B_hat)^2) + 1) / (mean(V^2) + 1)",
            "q_t_summary": global_summary,
            "q_t_subset_summary": subset_summary,
            "bootstrap": {
                "seed": int(bootstrap_seed),
                "repeats": int(bootstrap_repeats),
                "aggregation": "sample-level bootstrap over per-record q_T",
            },
            "color_limits": {key: list(value) for key, value in limits.items()},
            "normalization_conversion": metadata_payload.get("normalization_conversion", {}),
            "color_normalization": {
                "velocity": "shared physical-unit sequential viridis scale",
                "residual": "shared physical-unit zero-centered symmetric coolwarm scale",
                "error": "shared physical-unit sequential magma scale beginning at zero",
            },
        }
    )
    _write_json(output_dir / "metadata.json", metadata_payload)

    plt.rcParams.update({"font.family": FONT_FAMILY, "axes.unicode_minus": False})
    fig = plt.figure(figsize=(7.16, 2.80), dpi=160, facecolor="white")
    grid = fig.add_gridspec(1, 1, left=0.095, right=0.985, top=0.89, bottom=0.20)
    _plot_panel_b(fig, grid[0], qt_rows, global_summary=global_summary)
    fig_paths = [
        output_dir / "figure4_residual_transport.png",
        output_dir / "figure4_residual_transport.pdf",
        output_dir / "figure4_residual_transport.svg",
    ]
    for path in fig_paths:
        fig.savefig(path, dpi=300 if path.suffix == ".png" else None, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return fig_paths


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _denormalize_velocity(array: np.ndarray, profile_name: str, *, mode: str = "-1_1") -> np.ndarray:
    if mode in {"", "none", "null"}:
        return np.asarray(array, dtype=np.float32)
    profile = get_normalization_profile(profile_name)
    low, high = profile.ranges["depth_vel"]
    if mode == "-1_1":
        return (low + (np.asarray(array, dtype=np.float32) + 1.0) * 0.5 * (high - low)).astype(np.float32)
    if mode == "01":
        return (low + np.asarray(array, dtype=np.float32) * (high - low)).astype(np.float32)
    raise ValueError(f"Unsupported normalization mode for replay: {mode!r}")


def _normalization_conversion(profile_name: str, mode: str | None) -> dict[str, Any]:
    resolved_mode = "none" if mode is None else str(mode)
    profile = get_normalization_profile(profile_name)
    low, high = profile.ranges["depth_vel"]
    if resolved_mode in {"none", "None", ""}:
        formula = "physical_value = stored_value"
    elif resolved_mode == "-1_1":
        formula = "physical_value = low + (stored_value + 1) * 0.5 * (high - low)"
    elif resolved_mode == "01":
        formula = "physical_value = low + stored_value * (high - low)"
    else:
        raise ValueError(f"Unsupported normalization mode for metadata: {resolved_mode!r}")
    return {
        "profile": str(profile_name),
        "field": "depth_vel",
        "mode": resolved_mode,
        "physical_range_m_per_s": [float(low), float(high)],
        "formula": formula,
        "residual_conversion": "physical residuals are formed after affine denormalization",
    }


def _physical_case(
    target: np.ndarray,
    background: np.ndarray,
    residual_hat: np.ndarray,
    final: np.ndarray,
    *,
    profile_name: str,
    normalization_mode: str | None,
) -> dict[str, np.ndarray]:
    mode = "none" if normalization_mode is None else str(normalization_mode)
    if mode in {"none", "None", ""}:
        target_physical = np.asarray(target, dtype=np.float32)
        background_physical = np.asarray(background, dtype=np.float32)
        final_physical = np.asarray(final, dtype=np.float32)
        generated_residual_physical = np.asarray(residual_hat, dtype=np.float32)
    else:
        target_physical = _denormalize_velocity(target, profile_name, mode=mode)
        background_physical = _denormalize_velocity(background, profile_name, mode=mode)
        final_physical = _denormalize_velocity(final, profile_name, mode=mode)
        generated_residual_physical = final_physical - background_physical
    target_residual_physical = target_physical - background_physical
    return {
        "target": target_physical,
        "background": background_physical,
        "target_residual": target_residual_physical,
        "generated_residual": generated_residual_physical,
        "final": final_physical,
        "absolute_error": np.abs(final_physical - target_physical),
    }


def replay_selected_cases(
    *,
    config_path: str | Path,
    checkpoint_path: str | Path,
    selected_cases: Mapping[str, Mapping[str, Any]],
    output_dir: str | Path,
    inference_seed: int = 2027,
    batch_size: int = 8,
    manifest_rows: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Replay only the eight selected records with the formal checkpoint."""
    config_path = _resolve_path(config_path)
    checkpoint_path = _resolve_path(checkpoint_path)
    output_dir = _resolve_path(output_dir)
    if not config_path.is_file():
        raise FileNotFoundError(f"Replay config does not exist: {config_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Replay checkpoint does not exist: {checkpoint_path}")
    conf = OmegaConf.load(config_path)
    OmegaConf.update(conf, "data.post_split_datasets", None, merge=True)
    OmegaConf.update(conf, "evaluation.checkpoints.full", str(checkpoint_path), merge=True)
    OmegaConf.update(conf, "training.num_workers", 0, merge=True)
    OmegaConf.update(conf, "training.persistent_workers", False, merge=True)
    OmegaConf.update(conf, "training.prefetch_factor", 2, merge=True)
    OmegaConf.update(conf, "training.batch_size", int(batch_size), merge=True)
    configure_torch_runtime(OmegaConf.select(conf, "training.matmul_precision", default="medium"))

    # The dataset creates the exact global split first, then exposes all test
    # records. Filtering by source identity here cannot change split membership.
    dataset = build_dataset(conf, "test")
    split_audit: dict[str, Any] | None = None
    if manifest_rows is not None:
        train_dataset = build_dataset(conf, "train")
        val_dataset = build_dataset(conf, "val")

        def dataset_identity_rows(dataset_object: Any) -> list[dict[str, Any]]:
            return [
                {
                    "dataset_id": int(record["dataset_id"]),
                    "dataset_name": str(record["dataset_name"]),
                    "source_sample_index": int(record["sample_index"]),
                }
                for record in dataset_object.records
            ]

        split_audit = validate_split_identity(
            dataset_identity_rows(train_dataset),
            dataset_identity_rows(val_dataset),
            dataset_identity_rows(dataset),
            manifest_rows,
        )
        for split_dataset in (train_dataset, val_dataset):
            close = getattr(split_dataset, "close", None)
            if callable(close):
                close()
    positions: list[int] = []
    for name in SUBSET_ORDER:
        wanted = selected_cases[name]
        wanted_identity = (int(wanted["dataset_id"]), name, int(wanted["source_sample_index"]))
        matches = [
            position
            for position, record in enumerate(dataset.records)
            if (int(record["dataset_id"]), str(record["dataset_name"]), int(record["sample_index"])) == wanted_identity
        ]
        if len(matches) != 1:
            raise ValueError(f"Replay identity {wanted_identity} matched {len(matches)} dataset records")
        positions.append(matches[0])
    loader = DataLoader(
        Subset(dataset, positions),
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_bg_samples,
    )
    device = configured_torch_device(
        OmegaConf.select(conf, "training.accelerator", default="auto"),
        OmegaConf.select(conf, "training.devices", default=1),
    )
    model = BGPDRFMLightning(conf)
    checkpoint_info = _load_full_checkpoint(model, checkpoint_path)
    if checkpoint_info["missing_keys"] or checkpoint_info["unexpected_keys"] or checkpoint_info["skipped_shape_keys"]:
        raise RuntimeError(f"Formal checkpoint did not load cleanly: {checkpoint_info}")
    model.eval().to(device)
    random.seed(int(inference_seed))
    np.random.seed(int(inference_seed))
    torch.manual_seed(int(inference_seed))

    profile_name = str(OmegaConf.select(conf, "data.normalization_profile", default="auto"))
    if profile_name == "auto":
        profile_name = "openfwi"
    normalization_mode = OmegaConf.select(conf, "data.use_normalize", default="-1_1")
    cases: dict[str, dict[str, Any]] = {}
    raw_arrays: dict[str, np.ndarray] = {}
    normalized_arrays: dict[str, np.ndarray] = {}
    replay_metric_checks: dict[str, dict[str, float]] = {}
    selected_by_identity = {
        (int(case["dataset_id"]), str(name), int(case["source_sample_index"])): str(name)
        for name, case in selected_cases.items()
    }
    if len(selected_by_identity) != len(SUBSET_ORDER):
        raise ValueError("Selected replay identities must be unique across the eight subsets")
    selected_by_id_index = {
        (dataset_id, source_index): name
        for (dataset_id, _dataset_name, source_index), name in selected_by_identity.items()
    }
    if len(selected_by_id_index) != len(selected_by_identity):
        raise ValueError("Selected replay identities collide on dataset_id/source_sample_index")
    with torch.no_grad():
        for batch in loader:
            batch = batch_to_device(BGPDRFMLightning._as_batch(batch), device)
            prediction = model.predict_batch(batch)
            target = batch.depth_vel.detach().float().cpu().numpy()[:, 0]
            background = prediction.bg_hat.detach().float().cpu().numpy()[:, 0]
            residual_hat = prediction.residual_hat.detach().float().cpu().numpy()[:, 0]
            final = prediction.velocity_hat.detach().float().cpu().numpy()[:, 0]
            if not np.allclose(final, background + residual_hat, atol=2e-5, rtol=2e-5):
                raise RuntimeError("Replay violated V_hat = B_hat + R_hat within tolerance")
            metadata = getattr(batch, "metadata", {})
            dataset_ids = metadata.get("dataset_id")
            source_indices = metadata.get("sample_index")
            if dataset_ids is None or source_indices is None:
                raise RuntimeError("Replay batch is missing dataset_id/sample_index metadata")
            for item_idx in range(batch.depth_vel.shape[0]):
                dataset_id = int(dataset_ids[item_idx].detach().cpu().item())
                source_sample_index = int(source_indices[item_idx].detach().cpu().item())
                name = selected_by_id_index.get((dataset_id, source_sample_index))
                if name is None:
                    raise RuntimeError(
                        "Replay batch contains an unexpected selected identity: "
                        f"dataset_id={dataset_id}, source_sample_index={source_sample_index}"
                    )
                selected = dict(selected_cases[name])
                replay_metric_checks[name] = validate_replay_metrics(
                    target[item_idx],
                    background[item_idx],
                    final[item_idx],
                    selected,
                    strict_mae=False,
                )
                case = _physical_case(
                    target[item_idx],
                    background[item_idx],
                    residual_hat[item_idx],
                    final[item_idx],
                    profile_name=profile_name,
                    normalization_mode=normalization_mode,
                )
                case.update(
                    {
                        "dataset_name": name,
                        "dataset_id": dataset_id,
                        "source_sample_index": source_sample_index,
                        "mae": _float(selected, "mae"),
                        "rmse": _float(selected, "rmse"),
                        "ssim": _float(selected, "ssim"),
                        "q_t": _metric_qt(selected),
                        "background_mae": _metric_background_mae(selected),
                        "checkpoint": str(checkpoint_path),
                        "checkpoint_sha256": _sha256(checkpoint_path),
                        "normalization_profile": profile_name,
                        "normalization_mode": normalization_mode,
                        "selection_rank": 1,
                        "replay_mae": replay_metric_checks[name]["mae_replay"],
                        "replay_mae_delta": replay_metric_checks[name]["mae_delta"],
                        "displayed_prediction_source": "fixed-seed bounded replay",
                    }
                )
                cases[name] = case
                for key in ("target", "background", "target_residual", "generated_residual", "final", "absolute_error"):
                    raw_arrays[f"{name}__{key}"] = np.asarray(case[key], dtype=np.float32)
                normalized_arrays[f"{name}__target"] = np.asarray(target[item_idx], dtype=np.float32)
                normalized_arrays[f"{name}__background"] = np.asarray(background[item_idx], dtype=np.float32)
                normalized_arrays[f"{name}__target_residual"] = np.asarray(
                    target[item_idx] - background[item_idx], dtype=np.float32
                )
                normalized_arrays[f"{name}__generated_residual"] = np.asarray(residual_hat[item_idx], dtype=np.float32)
                normalized_arrays[f"{name}__final"] = np.asarray(final[item_idx], dtype=np.float32)
                normalized_arrays[f"{name}__absolute_error"] = np.asarray(
                    np.abs(final[item_idx] - target[item_idx]), dtype=np.float32
                )
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "replay_arrays.npz", **raw_arrays)
    np.savez_compressed(output_dir / "replay_arrays_normalized.npz", **normalized_arrays)
    replay_metadata = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_info": checkpoint_info,
        "config": str(config_path),
        "inference_seed": int(inference_seed),
        "euler_steps": int(OmegaConf.select(conf, "model.residual_num_inference_steps", default=50)),
        "normalization_profile": profile_name,
        "normalization_mode": normalization_mode,
        "normalization_conversion": _normalization_conversion(profile_name, normalization_mode),
        "dataset_split": "global test then identity filter",
        "split_audit": split_audit,
        "replay_metric_checks": replay_metric_checks,
        "physical_arrays": str(output_dir / "replay_arrays.npz"),
        "normalized_arrays": str(output_dir / "replay_arrays_normalized.npz"),
        "selected_identities": {
            name: {
                "dataset_id": cases[name]["dataset_id"],
                "source_sample_index": cases[name]["source_sample_index"],
            }
            for name in SUBSET_ORDER
        },
    }
    _write_json(output_dir / "replay_metadata.json", replay_metadata)
    return cases, replay_metadata


def build_figure4_from_formal(
    *,
    config_path: str | Path,
    checkpoint_path: str | Path,
    manifest_path: str | Path,
    metrics_path: str | Path,
    output_dir: str | Path,
    inference_seed: int = 2027,
    bootstrap_seed: int = 2027,
    bootstrap_repeats: int = 2000,
) -> list[Path]:
    manifest_path = _resolve_path(manifest_path)
    metrics_path = _resolve_path(metrics_path)
    checkpoint_path = _resolve_path(checkpoint_path)
    audit = validate_heldout_identity(manifest_path, metrics_path, expected_count=33_600)
    metric_rows = _read_rows(metrics_path)
    manifest_rows = _read_rows(manifest_path)
    selected = select_best_cases(metric_rows)
    expected = {
        "FlatVelA": 14135,
        "FlatVelB": 17501,
        "CurveVelA": 10700,
        "CurveVelB": 15941,
        "FlatFaultA": 6511,
        "FlatFaultB": 12648,
        "CurveFaultA": 1220,
        "CurveFaultB": 4692,
    }
    observed = {name: _int(selected[name], "source_sample_index") for name in SUBSET_ORDER}
    if observed != expected:
        raise RuntimeError(f"Live best-case selection differs from the audited expected selection: {observed}")
    qt_rows = [
        {
            "dataset_id": _int(row, "dataset_id"),
            "dataset_name": str(row["dataset_name"]),
            "source_sample_index": _int(row, "source_sample_index"),
            "q_T": _metric_qt(row),
        }
        for row in metric_rows
    ]
    cases, replay_metadata = replay_selected_cases(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        selected_cases=selected,
        output_dir=output_dir,
        inference_seed=inference_seed,
        manifest_rows=manifest_rows,
    )
    metadata = {
        "formal_manifest": str(manifest_path),
        "formal_metrics": str(metrics_path),
        "heldout_audit": audit,
        "selected_source_indices": observed,
        "normalization_conversion": replay_metadata["normalization_conversion"],
        "replay": replay_metadata,
        "provenance_note": "Panel (a) is deterministic best-MAE qualitative evidence, not a representative sample.",
    }
    return plot_figure4(
        output_dir,
        best_cases=cases,
        qt_rows=qt_rows,
        metadata=metadata,
        bootstrap_seed=bootstrap_seed,
        bootstrap_repeats=bootstrap_repeats,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Render AAAI Figure 4 from formal BG-PDR-FM held-out artifacts.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--inference-seed", type=int, default=2027)
    parser.add_argument("--bootstrap-seed", type=int, default=2027)
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    args = parser.parse_args()
    paths = build_figure4_from_formal(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        manifest_path=args.manifest,
        metrics_path=args.metrics,
        output_dir=args.output_dir,
        inference_seed=args.inference_seed,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_repeats=args.bootstrap_repeats,
    )
    print(json.dumps({"figure_paths": [str(path) for path in paths]}, indent=2))


if __name__ == "__main__":
    main()
