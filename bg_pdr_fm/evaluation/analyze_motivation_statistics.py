"""Generate training-free motivation statistics and Figure 1 assets.

This script intentionally does not construct or load a neural network.  It
uses the OpenFWI dataset adapter only to obtain its deterministic model-ready
observations and normalized targets, then applies kernel statistics and a
sample-Gram eigendecomposition.

Example:
    conda run -n seg python -m bg_pdr_fm.evaluation.analyze_motivation_statistics \
        --samples-per-subset 1000 \
        --output-dir docs/paper/AAAI2027/figures/figure1_motivation
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import platform
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpecFromSubplotSpec

from bg_pdr_fm.data.datasets import OpenFWIBGDataset
from bg_pdr_fm.evaluation.motivation_statistics import (
    cka_with_bandwidths,
    centered_normalized_cka,
    effective_dimension,
    flatten_dimension_normalize,
    low_high_decompose,
    sample_balanced_records,
)


OPENFWI_SUBSETS = (
    "FlatVelA",
    "FlatVelB",
    "CurveVelA",
    "CurveVelB",
    "FlatFaultA",
    "FlatFaultB",
    "CurveFaultA",
    "CurveFaultB",
)
MODALITIES = ("migrated_image", "horizon", "rms_vel", "well_log_mask")
MODALITY_LABELS = {
    "migrated_image": "PoSTM",
    "horizon": "Horizon",
    "rms_vel": "RMS velocity",
    "well_log_mask": "Well log + mask",
}
COMPONENTS = ("background", "structure")
COMPONENT_LABELS = {"background": "Background B", "structure": "Structure S"}
BACKGROUND_COLOR = "#2A9D6F"
STRUCTURE_COLOR = "#D55E00"
SHUFFLE_COLOR = "#9A9A9A"
TEXT_COLOR = "#252525"
SUBSET_LABELS = {name: name.replace("Vel", "Vel-").replace("Fault", "Fault-") for name in OPENFWI_SUBSETS}


def _float(value: Any) -> float:
    return float(np.asarray(value).item())


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}.")


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unavailable"


def _git_revision(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def _subset_display_name(name: str) -> str:
    return SUBSET_LABELS.get(name, name)


def _record_identity(row: Mapping[str, Any]) -> tuple[str, int]:
    return str(row["dataset_name"]), int(row["sample_index"])


def _to_spatial_array(value: Any, *, name: str) -> np.ndarray:
    if torch.is_tensor(value):
        array = value.detach().cpu().numpy()
    else:
        array = np.asarray(value)
    array = np.asarray(array, dtype=np.float64)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim < 2:
        raise ValueError(f"{name} must have at least two spatial dimensions, got {array.shape}.")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values.")
    return array


def _load_subset_arrays(dataset: OpenFWIBGDataset, rows: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    """Load one balanced subset in the same normalized form exposed by the loader."""

    values: dict[str, list[np.ndarray]] = {
        "velocity": [],
        "migrated_image": [],
        "horizon": [],
        "rms_vel": [],
        "well_log_mask": [],
    }
    for row in rows:
        position = int(row["global_train_position"])
        item = dataset[position]
        velocity = _to_spatial_array(item["depth_vel"], name="depth_vel")
        if velocity.shape != (70, 70):
            raise ValueError(f"Expected depth_vel to be 70x70, got {velocity.shape} for {_record_identity(row)}.")
        values["velocity"].append(velocity)
        values["migrated_image"].append(_to_spatial_array(item["migrated_image"], name="migrated_image"))
        values["horizon"].append(_to_spatial_array(item["horizon"], name="horizon"))
        values["rms_vel"].append(_to_spatial_array(item["rms_vel"], name="rms_vel"))
        well_log = _to_spatial_array(item["well_log"], name="well_log")
        well_mask = _to_spatial_array(item["well_mask"], name="well_mask")
        if well_log.shape != well_mask.shape:
            raise ValueError(f"well_log and well_mask shapes differ: {well_log.shape} vs {well_mask.shape}.")
        values["well_log_mask"].append(np.stack([well_log, well_mask], axis=0))

    return {key: np.stack(items, axis=0) for key, items in values.items()}


def _summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    q1, median, q3 = np.quantile(array, [0.25, 0.5, 0.75])
    return {
        "mean": float(array.mean()),
        "median": float(median),
        "std": float(array.std(ddof=0)),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
        "count": int(array.size),
    }


def _compute_subset_statistics(
    subset: str,
    arrays: Mapping[str, np.ndarray],
    *,
    subset_index: int,
    analysis_seed: int,
    lowpass_kernel: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, np.ndarray], dict[str, Any]]:
    velocity = np.asarray(arrays["velocity"], dtype=np.float64)
    background, structure = low_high_decompose(velocity, kernel_size=lowpass_kernel)
    decomposition_error = float(np.max(np.abs(velocity - (background + structure))))
    if decomposition_error >= 1e-6:
        raise AssertionError(f"B+S reconstruction error for {subset} is {decomposition_error:.3e}.")

    representations: dict[str, np.ndarray] = {}
    representation_config: dict[str, Any] = {}
    for modality in MODALITIES:
        representations[modality], representation_config[modality] = flatten_dimension_normalize(
            arrays[modality]
        )
    target_matrices = {
        "background": background.reshape(background.shape[0], -1),
        "structure": structure.reshape(structure.shape[0], -1),
    }
    target_config = {name: {"feature_count": int(matrix.shape[1]), "centered": False} for name, matrix in target_matrices.items()}

    shuffle_rng = np.random.default_rng(int(analysis_seed) + 1009 * (subset_index + 1))
    shuffle_permutation = shuffle_rng.permutation(velocity.shape[0])
    alignment_rows: list[dict[str, Any]] = []
    for modality in MODALITIES:
        for component in COMPONENTS:
            target = target_matrices[component]
            matched_cka, bandwidth_modality, bandwidth_target = cka_with_bandwidths(
                representations[modality], target
            )
            shuffled_target = target[shuffle_permutation]
            shuffled_cka, _, bandwidth_shuffled_target = cka_with_bandwidths(
                representations[modality], shuffled_target
            )
            linear_cka = centered_normalized_cka(representations[modality], target, kernel="linear")
            shuffled_linear_cka = centered_normalized_cka(
                representations[modality], shuffled_target, kernel="linear"
            )
            alignment_rows.append(
                {
                    "scope": "subset",
                    "subset": subset,
                    "modality": modality,
                    "component": component,
                    "rbf_cka": matched_cka,
                    "linear_cka": linear_cka,
                    "shuffled_rbf_cka": shuffled_cka,
                    "shuffled_linear_cka": shuffled_linear_cka,
                    "bandwidth_modality": bandwidth_modality,
                    "bandwidth_component": bandwidth_target,
                    "bandwidth_shuffled_component": bandwidth_shuffled_target,
                    "n_samples": int(velocity.shape[0]),
                    "subset_index": subset_index,
                }
            )

    rank_rows: list[dict[str, Any]] = []
    curves: dict[str, np.ndarray] = {}
    eigenvalues: dict[str, np.ndarray] = {}
    for component, matrix in target_matrices.items():
        result = effective_dimension(matrix)
        rank_rows.append(
            {
                "scope": "subset",
                "subset": subset,
                "component": component,
                "effective_rank": result.effective_rank,
                "rank": result.rank,
                "n_samples": int(velocity.shape[0]),
                "feature_count": int(matrix.shape[1]),
                "decomposition_max_abs_error": decomposition_error,
            }
        )
        curves[component] = result.explained_variance
        eigenvalues[component] = result.eigenvalues

    metadata = {
        "subset": subset,
        "n_samples": int(velocity.shape[0]),
        "decomposition_max_abs_error": decomposition_error,
        "representation_config": representation_config,
        "target_config": target_config,
        "shuffle_permutation": shuffle_permutation.tolist(),
    }
    return alignment_rows, rank_rows, {**curves, **{f"{key}_eigenvalues": value for key, value in eigenvalues.items()}}, metadata


def _add_alignment_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = list(rows)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["modality"]), str(row["component"]))].append(row)
    for (modality, component), group in sorted(grouped.items()):
        summary_row: dict[str, Any] = {
            "scope": "all_subsets_summary",
            "subset": "ALL",
            "modality": modality,
            "component": component,
            "n_samples": sum(int(row["n_samples"]) for row in group),
            "subset_index": "",
        }
        for field in ("rbf_cka", "linear_cka", "shuffled_rbf_cka", "shuffled_linear_cka"):
            stats = _summary([float(row[field]) for row in group])
            for name, value in stats.items():
                if name != "count":
                    summary_row[f"{field}_{name}"] = value
        output.append(summary_row)
    return output


def _add_rank_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = list(rows)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["component"])].append(row)
    for component, group in sorted(grouped.items()):
        rank_stats = _summary([float(row["effective_rank"]) for row in group])
        output.append(
            {
                "scope": "all_subsets_summary",
                "subset": "ALL",
                "component": component,
                "effective_rank": rank_stats["median"],
                "effective_rank_mean": rank_stats["mean"],
                "effective_rank_median": rank_stats["median"],
                "effective_rank_std": rank_stats["std"],
                "effective_rank_q1": rank_stats["q1"],
                "effective_rank_q3": rank_stats["q3"],
                "effective_rank_iqr": rank_stats["iqr"],
                "rank": int(round(np.median([int(row["rank"]) for row in group]))),
                "n_samples": sum(int(row["n_samples"]) for row in group),
                "feature_count": group[0]["feature_count"],
                "decomposition_max_abs_error": max(float(row["decomposition_max_abs_error"]) for row in group),
            }
        )
    return output


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for field in row:
            if field not in fieldnames:
                fieldnames.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def load_motivation_artifacts(input_dir: str | Path) -> dict[str, Any]:
    """Read the frozen statistics used by the publication-only v2 renderer."""

    root = Path(input_dir).resolve()
    required = (
        "alignment_by_subset.csv",
        "effective_rank_by_subset.csv",
        "explained_variance_curves.npz",
        "analysis_config.json",
    )
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing motivation artifacts under {root}: {missing}")
    with (root / "alignment_by_subset.csv").open("r", newline="", encoding="utf-8") as handle:
        alignment_rows = [dict(row) for row in csv.DictReader(handle)]
    with (root / "effective_rank_by_subset.csv").open("r", newline="", encoding="utf-8") as handle:
        rank_rows = [dict(row) for row in csv.DictReader(handle)]
    with np.load(root / "explained_variance_curves.npz") as data:
        curves = {key: np.asarray(data[key]).copy() for key in data.files}
    config = json.loads((root / "analysis_config.json").read_text(encoding="utf-8"))
    return {
        "input_dir": root,
        "alignment_rows": alignment_rows,
        "rank_rows": rank_rows,
        "curves": curves,
        "config": config,
    }


def extract_v2_render_data(artifacts: Mapping[str, Any]) -> dict[str, Any]:
    """Extract plot arrays without recomputing or transforming statistics."""

    alignment_rows = [
        row for row in artifacts["alignment_rows"] if row.get("scope") == "all_subsets_summary"
    ]
    expected_alignment_keys = {(modality, component) for modality in MODALITIES for component in COMPONENTS}
    alignment_lookup = {
        (str(row["modality"]), str(row["component"])): float(row["rbf_cka_median"])
        for row in alignment_rows
    }
    if set(alignment_lookup) != expected_alignment_keys:
        raise ValueError(
            "alignment_by_subset.csv must contain exactly one all-subsets median for every modality/component pair."
        )
    alignment_matrix = np.asarray(
        [[alignment_lookup[(modality, component)] for component in COMPONENTS] for modality in MODALITIES],
        dtype=np.float64,
    )

    rank_rows = [row for row in artifacts["rank_rows"] if row.get("scope") == "subset"]
    rank_lookup = {
        (str(row["subset"]), str(row["component"])): float(row["effective_rank"])
        for row in rank_rows
    }
    expected_rank_keys = {(subset, component) for subset in OPENFWI_SUBSETS for component in COMPONENTS}
    if set(rank_lookup) != expected_rank_keys:
        raise ValueError("effective_rank_by_subset.csv must contain exactly 16 subset/component rows.")
    rank_pairs = np.asarray(
        [[rank_lookup[(subset, component)] for component in COMPONENTS] for subset in OPENFWI_SUBSETS],
        dtype=np.float64,
    )

    curves = artifacts["curves"]
    for key in ("background_curves", "structure_curves"):
        if key not in curves or curves[key].ndim != 2 or curves[key].shape[0] != len(OPENFWI_SUBSETS):
            raise ValueError(f"explained_variance_curves.npz has an invalid {key!r} array.")
    if curves["background_curves"].shape != curves["structure_curves"].shape:
        raise ValueError("Background and structure explained-variance curves must have the same shape.")
    return {
        "alignment_matrix": alignment_matrix,
        "rank_pairs": rank_pairs,
        "curves": {
            "background": np.asarray(curves["background_curves"], dtype=np.float64),
            "structure": np.asarray(curves["structure_curves"], dtype=np.float64),
        },
        "subsets": OPENFWI_SUBSETS,
        "modalities": MODALITIES,
        "config": artifacts["config"],
    }


def _plot_alignment_panel(
    ax: plt.Axes,
    alignment_rows: Sequence[Mapping[str, Any]],
    *,
    panel_label: str = "(a)",
    add_colorbar: bool = True,
    colorbar_ax: plt.Axes | None = None,
) -> None:
    summary = {
        (str(row["modality"]), str(row["component"])): float(row["rbf_cka_median"])
        for row in alignment_rows
        if row.get("scope") == "all_subsets_summary"
    }
    matrix = np.asarray(
        [[summary[(modality, component)] for component in COMPONENTS] for modality in MODALITIES],
        dtype=np.float64,
    )
    image = ax.imshow(matrix, cmap="cividis", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(COMPONENTS)), ["Background B", "Structure S"], fontsize=7.5)
    ax.set_yticks(range(len(MODALITIES)), [MODALITY_LABELS[name] for name in MODALITIES], fontsize=7.5)
    ax.tick_params(length=0, pad=2)
    ax.set_title("Data-level kernel alignment", fontsize=8.5, pad=5, color=TEXT_COLOR)
    ax.text(-0.16, 1.06, panel_label, transform=ax.transAxes, fontsize=9, fontweight="bold", color=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            color = "white" if matrix[row_index, column_index] > 0.58 else TEXT_COLOR
            ax.text(column_index, row_index, f"{matrix[row_index, column_index]:.2f}", ha="center", va="center", fontsize=8, color=color)
    ax.set_xlabel("Target component", fontsize=7.5, labelpad=4)
    if add_colorbar:
        cbar = ax.figure.colorbar(image, ax=ax if colorbar_ax is None else None, cax=colorbar_ax, fraction=0.046, pad=0.04)
        cbar.set_label("RBF CKA", fontsize=7.5, labelpad=2)
        cbar.ax.tick_params(labelsize=7.5, length=2)


def _plot_effective_panel(
    ax: plt.Axes,
    curves: Mapping[str, np.ndarray],
    effective_rank_by_subset: Mapping[str, Mapping[str, float]],
    *,
    panel_label: str = "(b)",
) -> None:
    background_curves = np.asarray(curves["background"], dtype=np.float64)
    structure_curves = np.asarray(curves["structure"], dtype=np.float64)
    x = np.arange(1, background_curves.shape[1] + 1)
    background_median = np.nanmedian(background_curves, axis=0)
    structure_median = np.nanmedian(structure_curves, axis=0)
    background_q1, background_q3 = np.nanquantile(background_curves, [0.25, 0.75], axis=0)
    structure_q1, structure_q3 = np.nanquantile(structure_curves, [0.25, 0.75], axis=0)
    ax.plot(x, background_median, color=BACKGROUND_COLOR, linewidth=1.5, label="Background B")
    ax.fill_between(x, background_q1, background_q3, color=BACKGROUND_COLOR, alpha=0.16, linewidth=0)
    ax.plot(x, structure_median, color=STRUCTURE_COLOR, linewidth=1.5, label="Structure S")
    ax.fill_between(x, structure_q1, structure_q3, color=STRUCTURE_COLOR, alpha=0.16, linewidth=0)
    ax.set_xscale("log")
    ax.set_xlim(1, max(2, int(x[-1])))
    ax.set_ylim(0.0, 1.02)
    if int(x[-1]) >= 10:
        tick_values = [value for value in (1, 10, 100, 1000, 10000) if value <= int(x[-1])]
    else:
        tick_values = sorted(set([1, max(2, int(x[-1] // 2)), int(x[-1])]))
    ax.set_xticks(tick_values)
    ax.set_xticklabels([str(value) for value in tick_values])
    ax.set_xlabel("Principal components (log scale)", fontsize=7.5)
    ax.set_ylabel("Cumulative explained variance", fontsize=7.5)
    ax.tick_params(labelsize=7.5, length=2)
    ax.set_title("Effective degrees of freedom", fontsize=8.5, pad=5, color=TEXT_COLOR)
    ax.text(-0.16, 1.06, panel_label, transform=ax.transAxes, fontsize=9, fontweight="bold", color=TEXT_COLOR)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.5)
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), fontsize=7.5, frameon=False, handlelength=1.8)
    inset = ax.inset_axes([0.59, 0.16, 0.36, 0.38])
    subset_names = tuple(effective_rank_by_subset.keys())
    for index, subset in enumerate(subset_names):
        values = effective_rank_by_subset[subset]
        inset.plot([0, 1], [values["background"], values["structure"]], color="#B7B7B7", linewidth=0.65, zorder=1)
        inset.scatter(0, values["background"], s=8, color=BACKGROUND_COLOR, zorder=2)
        inset.scatter(1, values["structure"], s=8, color=STRUCTURE_COLOR, zorder=2)
    inset.set_xticks([0, 1], ["B", "S"], fontsize=7.5)
    inset.set_ylabel(r"$r_{\mathrm{eff}}$", fontsize=7.5, labelpad=1)
    inset.tick_params(axis="y", labelsize=7.5, length=2)
    inset.grid(axis="y", color="#E6E6E6", linewidth=0.4)
    inset.set_title(r"Paired $r_{\mathrm{eff}}$", fontsize=7.5, pad=1)


def _save_figure(fig: plt.Figure, output_stem: Path, *, dpi: int = 600) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(output_stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def _plot_all(
    output_dir: Path,
    alignment_rows: Sequence[Mapping[str, Any]],
    curves: Mapping[str, np.ndarray],
    effective_rank_by_subset: Mapping[str, Mapping[str, float]],
) -> None:
    alignment_fig, alignment_ax = plt.subplots(figsize=(3.25, 2.75), constrained_layout=False)
    _plot_alignment_panel(alignment_ax, alignment_rows)
    alignment_fig.subplots_adjust(left=0.25, right=0.87, bottom=0.22, top=0.88)
    _save_figure(alignment_fig, output_dir / "modality_target_alignment")

    dimension_fig, dimension_ax = plt.subplots(figsize=(3.25, 2.75), constrained_layout=False)
    _plot_effective_panel(
        dimension_ax,
        {"background": curves["background_curves"], "structure": curves["structure_curves"]},
        effective_rank_by_subset,
    )
    dimension_fig.subplots_adjust(left=0.17, right=0.97, bottom=0.22, top=0.88)
    _save_figure(dimension_fig, output_dir / "component_effective_dimension")

    combined_fig = plt.figure(figsize=(6.9, 2.95), constrained_layout=False)
    grid = combined_fig.add_gridspec(1, 2, width_ratios=(1.0, 1.27), wspace=0.43)
    alignment_ax = combined_fig.add_subplot(grid[0, 0])
    dimension_ax = combined_fig.add_subplot(grid[0, 1])
    _plot_alignment_panel(alignment_ax, alignment_rows, add_colorbar=True)
    _plot_effective_panel(
        dimension_ax,
        {"background": curves["background_curves"], "structure": curves["structure_curves"]},
        effective_rank_by_subset,
    )
    combined_fig.subplots_adjust(left=0.075, right=0.985, bottom=0.22, top=0.88, wspace=0.42)
    _save_figure(combined_fig, output_dir / "motivation_statistics_combined")


V2_FONT_FAMILY = "DejaVu Sans"
V2_AXIS_COLOR = "#666666"
V2_GRID_COLOR = "#D9D9D9"
V2_PAIR_COLOR = "#B5B5B5"


def _configure_v2_style() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": V2_FONT_FAMILY,
            "mathtext.fontset": "stix",
            "axes.labelcolor": TEXT_COLOR,
            "axes.titlecolor": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "axes.edgecolor": V2_AXIS_COLOR,
            "savefig.facecolor": "white",
        }
    )


def _v2_cell_text_color(cmap: Any, value: float) -> str:
    rgba = np.asarray(cmap(float(np.clip(value, 0.0, 1.0))))
    luminance = float(0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2])
    return "white" if luminance < 0.50 else TEXT_COLOR


def _v2_nested_grid(fig: plt.Figure, subplot_spec: Any, *, width_ratios: Sequence[float], wspace: float) -> Any:
    if subplot_spec is None:
        return fig.add_gridspec(1, len(width_ratios), width_ratios=width_ratios, wspace=wspace)
    return GridSpecFromSubplotSpec(
        1,
        len(width_ratios),
        subplot_spec=subplot_spec,
        width_ratios=width_ratios,
        wspace=wspace,
    )


def _draw_v2_alignment_panel(
    fig: plt.Figure,
    data: Mapping[str, Any],
    *,
    subplot_spec: Any = None,
    colorbar_label_position: str = "right",
) -> None:
    nested = _v2_nested_grid(fig, subplot_spec, width_ratios=(0.90, 0.06), wspace=0.12)
    ax = fig.add_subplot(nested[0, 0])
    colorbar_ax = fig.add_subplot(nested[0, 1])
    matrix = np.asarray(data["alignment_matrix"], dtype=np.float64)
    image = ax.imshow(matrix, cmap="cividis", vmin=0.0, vmax=1.0, aspect="auto", interpolation="none")
    ax.set_title("(a) Modality-component alignment", loc="left", fontsize=8.8, pad=5, fontweight="normal")
    ax.set_xticks([0, 1], [r"Background $B$", r"Structure $S$"], fontsize=7.5)
    ax.set_yticks(range(len(MODALITIES)), [MODALITY_LABELS[name] for name in MODALITIES], fontsize=7.5)
    ax.tick_params(length=0, pad=2)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks(np.arange(-0.5, matrix.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, matrix.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.9)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = float(matrix[row_index, column_index])
            ax.text(
                column_index,
                row_index,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8.0,
                color=_v2_cell_text_color(image.cmap, value),
            )
    colorbar = fig.colorbar(image, cax=colorbar_ax)
    if colorbar_label_position not in {"left", "right"}:
        raise ValueError("colorbar_label_position must be 'left' or 'right'.")
    colorbar.ax.yaxis.set_label_position(colorbar_label_position)
    colorbar.set_label("Median RBF CKA", fontsize=7.5, labelpad=4)
    colorbar.set_ticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    colorbar.ax.tick_params(labelsize=7.5, length=2, pad=2)


def _draw_v2_effective_panel(
    fig: plt.Figure,
    data: Mapping[str, Any],
    *,
    subplot_spec: Any = None,
) -> None:
    nested = _v2_nested_grid(fig, subplot_spec, width_ratios=(0.76, 0.24), wspace=0.46)
    main_ax = fig.add_subplot(nested[0, 0])
    rank_ax = fig.add_subplot(nested[0, 1])
    curves = data["curves"]
    background_curves = np.asarray(curves["background"], dtype=np.float64)
    structure_curves = np.asarray(curves["structure"], dtype=np.float64)
    x = np.arange(1, background_curves.shape[1] + 1)
    background_median = np.nanmedian(background_curves, axis=0)
    structure_median = np.nanmedian(structure_curves, axis=0)
    background_q1, background_q3 = np.nanquantile(background_curves, [0.25, 0.75], axis=0)
    structure_q1, structure_q3 = np.nanquantile(structure_curves, [0.25, 0.75], axis=0)
    main_ax.plot(x, background_median, color=BACKGROUND_COLOR, linewidth=1.8, label="Background B", zorder=3)
    main_ax.fill_between(x, background_q1, background_q3, color=BACKGROUND_COLOR, alpha=0.15, linewidth=0, zorder=1)
    main_ax.plot(x, structure_median, color=STRUCTURE_COLOR, linewidth=1.8, label="Structure S", zorder=3)
    main_ax.fill_between(x, structure_q1, structure_q3, color=STRUCTURE_COLOR, alpha=0.15, linewidth=0, zorder=1)
    main_ax.set_title("(b) Component effective dimension", loc="left", fontsize=8.8, pad=5, fontweight="normal")
    main_ax.set_xscale("log")
    main_ax.set_xlim(1, max(1000, int(x[-1])))
    main_ax.set_ylim(0.0, 1.02)
    main_ax.set_xticks([1, 10, 100, 1000])
    main_ax.set_xticklabels(["1", "10", "100", "1000"], fontsize=7.5)
    main_ax.set_xlabel("Principal components", fontsize=7.5, labelpad=3)
    main_ax.set_ylabel("Cumulative explained variance", fontsize=7.5, labelpad=3)
    main_ax.tick_params(axis="y", labelsize=7.5, length=2)
    main_ax.grid(axis="y", color=V2_GRID_COLOR, linewidth=0.5, alpha=0.8)
    main_ax.grid(axis="x", visible=False)
    for side in ("top", "right"):
        main_ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        main_ax.spines[side].set_color(V2_AXIS_COLOR)
        main_ax.spines[side].set_linewidth(0.65)
    main_ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.01, 0.995),
        ncol=2,
        fontsize=7.5,
        frameon=False,
        handlelength=1.6,
        columnspacing=1.0,
        handletextpad=0.35,
        borderaxespad=0.0,
    )

    rank_pairs = np.asarray(data["rank_pairs"], dtype=np.float64)
    for pair in rank_pairs:
        rank_ax.plot([0, 1], pair, color=V2_PAIR_COLOR, linewidth=0.75, alpha=0.85, zorder=1)
    rank_ax.scatter(
        np.zeros(rank_pairs.shape[0]),
        rank_pairs[:, 0],
        color=BACKGROUND_COLOR,
        s=16,
        edgecolors="white",
        linewidths=0.3,
        zorder=3,
    )
    rank_ax.scatter(
        np.ones(rank_pairs.shape[0]),
        rank_pairs[:, 1],
        color=STRUCTURE_COLOR,
        s=16,
        edgecolors="white",
        linewidths=0.3,
        zorder=3,
    )
    rank_ax.set_title("Effective rank", fontsize=7.5, pad=4, loc="left", x=0.05)
    rank_ax.set_xlim(-0.25, 1.25)
    rank_ax.set_xticks([0, 1], ["B", "S"], fontsize=7.5)
    rank_ax.set_yscale("log")
    rank_ax.set_ylim(1.0, 550.0)
    rank_ax.set_yticks([1, 10, 100, 500])
    rank_ax.set_yticklabels(["1", "10", "100", "500"], fontsize=7.5)
    rank_ax.set_ylabel("")
    rank_ax.text(
        0.98,
        0.50,
        r"$r_{\mathrm{eff}}$",
        transform=rank_ax.transAxes,
        rotation=90,
        ha="right",
        va="center",
        fontsize=7.5,
        color=TEXT_COLOR,
    )
    rank_ax.tick_params(length=2)
    rank_ax.grid(axis="y", color=V2_GRID_COLOR, linewidth=0.45, alpha=0.8)
    rank_ax.grid(axis="x", visible=False)
    for side in ("top", "right"):
        rank_ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        rank_ax.spines[side].set_color(V2_AXIS_COLOR)
        rank_ax.spines[side].set_linewidth(0.65)


def _save_v2_figure(fig: plt.Figure, output_stem: Path, *, dpi: int = 600) -> None:
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".pdf"), dpi=dpi)
    fig.savefig(output_stem.with_suffix(".png"), dpi=dpi)
    plt.close(fig)


def render_v2_from_artifacts(input_dir: str | Path, output_dir: str | Path | None = None) -> dict[str, str]:
    """Render the six v2 assets from frozen CSV/NPZ/JSON statistics only."""

    _configure_v2_style()
    artifacts = load_motivation_artifacts(input_dir)
    data = extract_v2_render_data(artifacts)
    output = Path(output_dir or input_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)

    alignment_fig = plt.figure(figsize=(4.00, 2.70))
    alignment_grid = alignment_fig.add_gridspec(1, 1)
    _draw_v2_alignment_panel(alignment_fig, data, subplot_spec=alignment_grid[0, 0])
    alignment_fig.subplots_adjust(left=0.22, right=0.90, bottom=0.14, top=0.84)
    _save_v2_figure(alignment_fig, output / "modality_target_alignment_v2")

    effective_fig = plt.figure(figsize=(4.20, 2.70))
    effective_grid = effective_fig.add_gridspec(1, 1)
    _draw_v2_effective_panel(effective_fig, data, subplot_spec=effective_grid[0, 0])
    effective_fig.subplots_adjust(left=0.16, right=0.98, bottom=0.17, top=0.82)
    _save_v2_figure(effective_fig, output / "component_effective_dimension_v2")

    combined_fig = plt.figure(figsize=(7.00, 2.70))
    combined_grid = combined_fig.add_gridspec(1, 2, width_ratios=(0.44, 0.56), wspace=0.20)
    _draw_v2_alignment_panel(
        combined_fig,
        data,
        subplot_spec=combined_grid[0, 0],
        colorbar_label_position="left",
    )
    _draw_v2_effective_panel(combined_fig, data, subplot_spec=combined_grid[0, 1])
    combined_fig.subplots_adjust(left=0.125, right=0.965, bottom=0.18, top=0.82, wspace=0.20)
    _save_v2_figure(combined_fig, output / "motivation_statistics_combined_v2")
    return {
        "input_dir": str(Path(input_dir).resolve()),
        "output_dir": str(output),
        "alignment_pdf": str(output / "modality_target_alignment_v2.pdf"),
        "alignment_png": str(output / "modality_target_alignment_v2.png"),
        "effective_pdf": str(output / "component_effective_dimension_v2.pdf"),
        "effective_png": str(output / "component_effective_dimension_v2.png"),
        "combined_pdf": str(output / "motivation_statistics_combined_v2.pdf"),
        "combined_png": str(output / "motivation_statistics_combined_v2.png"),
    }


def _write_report(
    path: Path,
    *,
    config: Mapping[str, Any],
    alignment_rows: Sequence[Mapping[str, Any]],
    rank_rows: Sequence[Mapping[str, Any]],
    subset_metadata: Mapping[str, Mapping[str, Any]],
) -> None:
    summary_alignment = [row for row in alignment_rows if row.get("scope") == "all_subsets_summary"]
    summary_rank = [row for row in rank_rows if row.get("scope") == "all_subsets_summary"]
    lines = [
        "# Training-free motivation statistics",
        "",
        "This report was generated from normalized OpenFWI observations and targets without constructing or loading a neural network, checkpoint, optimizer, or learned dimensionality-reduction method.",
        "",
        "## Protocol",
        "",
        f"- Global split: 70/20/10, split seed `{config['split_seed']}`; only the global training split was read.",
        f"- Analysis seed: `{config['analysis_seed']}`; well seed: `{config['well_seed']}` with deterministic well generation.",
        f"- Subsets: {', '.join(config['subsets'])}.",
        f"- Samples per subset requested: `{config['samples_per_subset']}`; selected counts: `{config['selected_counts']}`.",
        f"- Low-pass operator: stride-one `{config['lowpass_kernel']}x{config['lowpass_kernel']}` average pooling with zero padding and padding included in the divisor.",
        f"- Maximum numerical error in `V = B + S`: `{config['max_decomposition_error']:.3e}`.",
        "",
        "## Figure 1a: data-level kernel alignment",
        "",
        "The entries below are median RBF CKA values across the eight subsets. They are reported as data-level kernel-alignment or dependence proxies, not mutual information, causality, or identifiable semantic roles.",
        "",
        "| Modality | Background B | Structure S |",
        "|---|---:|---:|",
    ]
    lookup = {(str(row["modality"]), str(row["component"])): row for row in summary_alignment}
    for modality in MODALITIES:
        lines.append(
            f"| {MODALITY_LABELS[modality]} | {float(lookup[(modality, 'background')]['rbf_cka_median']):.4f} | {float(lookup[(modality, 'structure')]['rbf_cka_median']):.4f} |"
        )
    lines.extend(["", "Matched versus fixed-shuffle controls:", "", "| Modality | Component | Matched median | Shuffled median | Difference |", "|---|---|---:|---:|---:|"])
    for row in summary_alignment:
        matched = float(row["rbf_cka_median"])
        shuffled = float(row["shuffled_rbf_cka_median"])
        lines.append(
            f"| {MODALITY_LABELS[str(row['modality'])]} | {COMPONENT_LABELS[str(row['component'])]} | {matched:.4f} | {shuffled:.4f} | {matched - shuffled:.4f} |"
        )

    lines.extend(["", "## Figure 1b: effective dimension", "", "The effective rank is the participation-ratio summary of the nonzero sample-covariance spectrum. It measures intrinsic variability/effective degrees of freedom and is not conditional entropy.", "", "| Component | Median $r_{\\mathrm{eff}}$ | Mean | Std | IQR |", "|---|---:|---:|---:|---:|"])
    for row in summary_rank:
        lines.append(
            f"| {COMPONENT_LABELS[str(row['component'])]} | {float(row['effective_rank_median']):.3f} | {float(row['effective_rank_mean']):.3f} | {float(row['effective_rank_std']):.3f} | {float(row['effective_rank_iqr']):.3f} |"
        )

    per_subset_rank = [row for row in rank_rows if row.get("scope") == "subset"]
    lines.extend(["", "### Per-subset effective ranks", "", "| Subset | Background B | Structure S | Structure minus background |", "|---|---:|---:|---:|"])
    by_subset: dict[str, dict[str, float]] = defaultdict(dict)
    for row in per_subset_rank:
        by_subset[str(row["subset"])][str(row["component"])] = float(row["effective_rank"])
    for subset in OPENFWI_SUBSETS:
        background = by_subset[subset]["background"]
        structure = by_subset[subset]["structure"]
        lines.append(f"| {_subset_display_name(subset)} | {background:.3f} | {structure:.3f} | {structure - background:.3f} |")

    differences = [by_subset[subset]["structure"] - by_subset[subset]["background"] for subset in OPENFWI_SUBSETS]
    alignment_differences = []
    for modality in MODALITIES:
        group = [row for row in alignment_rows if row.get("scope") == "subset" and row["modality"] == modality]
        background_values = {str(row["subset"]): float(row["rbf_cka"]) for row in group if row["component"] == "background"}
        structure_values = {str(row["subset"]): float(row["rbf_cka"]) for row in group if row["component"] == "structure"}
        alignment_differences.extend(structure_values[name] - background_values[name] for name in OPENFWI_SUBSETS)
    lines.extend([
        "",
        "## Interpretation",
        "",
        f"- Structure effective rank exceeds background effective rank in `{sum(value > 0 for value in differences)}/{len(differences)}` subsets; median difference is `{np.median(differences):.3f}`.",
        f"- Across all modality/subset pairs, structure-minus-background alignment is positive in `{sum(value > 0 for value in alignment_differences)}/{len(alignment_differences)}` cases.",
        "- These results support or weaken the proposed information-source and degree-of-freedom asymmetries only at the level of the stated data statistics. They do not prove representation identifiability, causal modality responsibility, or conditional-entropy ordering.",
        "",
        "## Limitations",
        "",
        "- The analysis uses a fixed training-only sample and fixed median-heuristic bandwidths; no sample, axis, or bandwidth was adjusted after seeing the result.",
        "- Well logs and masks are concatenated as one raw information block; the mask is not discarded.",
        "- The effective-dimension calculation intentionally avoids per-pixel whitening and per-sample unit-norm scaling, so its values describe the normalized fields under the specified protocol.",
        "",
        "## Provenance",
        "",
        f"- Per-subset metadata and shuffle permutations are recorded in `analysis_config.json`; the complete selected-record manifest is `analysis_manifest.csv`.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--render-v2",
        action="store_true",
        help="Render publication-ready v2 figures from existing CSV/NPZ/JSON artifacts without recomputing statistics.",
    )
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("docs/paper/AAAI2027/figures/figure1_motivation"),
        help="Directory containing the frozen motivation statistics used by --render-v2.",
    )
    parser.add_argument(
        "--v2-output-dir",
        type=Path,
        default=None,
        help="Optional output directory for v2 assets; defaults to --artifact-dir.",
    )
    parser.add_argument("--root-dir", default="/public/home/xuyinghao/workspace/datasets/openfwi")
    parser.add_argument("--lmdb-root", default="/public/home/xuyinghao/workspace/datasets/openfwi_lmdb")
    parser.add_argument("--storage-backend", choices=("auto", "lmdb", "npy"), default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("docs/paper/AAAI2027/figures/figure1_motivation"))
    parser.add_argument("--samples-per-subset", type=int, default=1000)
    parser.add_argument("--analysis-seed", type=int, default=2027)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--well-seed", type=int, default=1234)
    parser.add_argument("--lowpass-kernel", type=int, default=5)
    return parser


def run_analysis(args: argparse.Namespace) -> dict[str, Any]:
    repo_root = Path(__file__).resolve().parents[2]
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.time()

    dataset = OpenFWIBGDataset(
        root_dir=args.root_dir,
        datasets=OPENFWI_SUBSETS,
        split="train",
        use_normalize="-1_1",
        target_shape=None,
        required_modalities=("depth_vel", "migrated_image", "horizon", "rms_vel"),
        split_fractions=(0.7, 0.2, 0.1),
        split_seed=int(args.split_seed),
        well_count_range=(0, 3),
        well_seed=int(args.well_seed),
        well_random=False,
        storage_backend=args.storage_backend,
        lmdb_root=args.lmdb_root,
        normalization_profile="openfwi",
        normalize_clamp=True,
    )
    try:
        record_rows: list[dict[str, Any]] = []
        for position, record in enumerate(dataset.records):
            row = dict(record)
            row["split"] = "train"
            row["global_train_position"] = int(position)
            record_rows.append(row)
        selected = sample_balanced_records(
            record_rows,
            OPENFWI_SUBSETS,
            samples_per_subset=int(args.samples_per_subset),
            seed=int(args.analysis_seed),
        )
        if not selected:
            raise RuntimeError("No global-training records were selected for the motivation analysis.")
        selected_by_subset: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in selected:
            selected_by_subset[str(row["dataset_name"])].append(row)
        missing = [name for name in OPENFWI_SUBSETS if not selected_by_subset.get(name)]
        if missing:
            raise RuntimeError(f"The global training split has no selected records for subsets: {missing}")

        manifest_rows: list[dict[str, Any]] = []
        for subset in OPENFWI_SUBSETS:
            for selection_rank, row in enumerate(selected_by_subset[subset]):
                manifest_rows.append(
                    {
                        "analysis_seed": int(args.analysis_seed),
                        "split": "global_train",
                        "split_seed": int(args.split_seed),
                        "well_seed": int(args.well_seed),
                        "dataset_id": int(row["dataset_id"]),
                        "dataset_name": subset,
                        "dataset_label": _subset_display_name(subset),
                        "source_sample_index": int(row["sample_index"]),
                        "global_train_position": int(row["global_train_position"]),
                        "selection_rank": int(selection_rank),
                        "record_id": f"{subset}:{int(row['sample_index'])}",
                    }
                )
        _write_csv(output_dir / "analysis_manifest.csv", manifest_rows)

        alignment_rows: list[dict[str, Any]] = []
        rank_rows: list[dict[str, Any]] = []
        curve_data: dict[str, list[np.ndarray]] = {"background": [], "structure": []}
        eigenvalue_data: dict[str, list[np.ndarray]] = {"background": [], "structure": []}
        subset_metadata: dict[str, dict[str, Any]] = {}
        max_decomposition_error = 0.0
        for subset_index, subset in enumerate(OPENFWI_SUBSETS):
            rows = selected_by_subset[subset]
            arrays = _load_subset_arrays(dataset, rows)
            subset_alignment, subset_rank, subset_curves, metadata = _compute_subset_statistics(
                subset,
                arrays,
                subset_index=subset_index,
                analysis_seed=int(args.analysis_seed),
                lowpass_kernel=int(args.lowpass_kernel),
            )
            alignment_rows.extend(subset_alignment)
            rank_rows.extend(subset_rank)
            curve_data["background"].append(subset_curves["background"])
            curve_data["structure"].append(subset_curves["structure"])
            eigenvalue_data["background"].append(subset_curves["background_eigenvalues"])
            eigenvalue_data["structure"].append(subset_curves["structure_eigenvalues"])
            subset_metadata[subset] = metadata
            max_decomposition_error = max(max_decomposition_error, float(metadata["decomposition_max_abs_error"]))
    finally:
        dataset.close()

    alignment_rows_with_summary = _add_alignment_summary(alignment_rows)
    rank_rows_with_summary = _add_rank_summary(rank_rows)
    _write_csv(output_dir / "alignment_by_subset.csv", alignment_rows_with_summary)
    _write_csv(output_dir / "effective_rank_by_subset.csv", rank_rows_with_summary)

    curve_arrays = {
        "subsets": np.asarray(OPENFWI_SUBSETS),
        "background_curves": np.asarray(curve_data["background"], dtype=np.float64),
        "structure_curves": np.asarray(curve_data["structure"], dtype=np.float64),
        "background_eigenvalues": np.asarray(eigenvalue_data["background"], dtype=np.float64),
        "structure_eigenvalues": np.asarray(eigenvalue_data["structure"], dtype=np.float64),
        "principal_components": np.arange(1, len(curve_data["background"][0]) + 1, dtype=np.int64),
    }
    np.savez_compressed(output_dir / "explained_variance_curves.npz", **curve_arrays)

    effective_rank_by_subset: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rank_rows:
        effective_rank_by_subset[str(row["subset"])][str(row["component"])] = float(row["effective_rank"])
    _plot_all(output_dir, alignment_rows_with_summary, curve_arrays, effective_rank_by_subset)

    selected_counts = {subset: len(selected_by_subset[subset]) for subset in OPENFWI_SUBSETS}
    train_available_counts = {
        subset: sum(str(row["dataset_name"]) == subset for row in record_rows) for subset in OPENFWI_SUBSETS
    }
    config: dict[str, Any] = {
        "analysis_name": "training_free_motivation_statistics",
        "script": str(Path(__file__).resolve()),
        "repo_root": str(repo_root),
        "git_revision": _git_revision(repo_root),
        "created_unix": time.time(),
        "duration_seconds": time.time() - start_time,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": {name: _package_version(name) for name in ("numpy", "matplotlib", "torch")},
        "subsets": list(OPENFWI_SUBSETS),
        "split": "global_train",
        "split_seed": int(args.split_seed),
        "split_fractions": [0.7, 0.2, 0.1],
        "analysis_seed": int(args.analysis_seed),
        "well_seed": int(args.well_seed),
        "well_count_range": [0, 3],
        "samples_per_subset": int(args.samples_per_subset),
        "available_train_counts": train_available_counts,
        "selected_counts": selected_counts,
        "root_dir": str(Path(args.root_dir).resolve()),
        "lmdb_root": str(Path(args.lmdb_root).resolve()),
        "storage_backend_requested": str(args.storage_backend),
        "normalization": {"profile": "openfwi", "mode": "-1_1", "clamp": True, "target_shape": None},
        "modalities": {
            "migrated_image": "dataloader normalized PoSTM array",
            "horizon": "dataloader normalized horizon array",
            "rms_vel": "dataloader normalized RMS velocity array",
            "well_log_mask": "concatenation of dataloader normalized well_log and well_mask; mask retained",
        },
        "lowpass_kernel": int(args.lowpass_kernel),
        "lowpass_boundary": "constant zero padding; count_include_pad=True; stride=1",
        "max_decomposition_error": max_decomposition_error,
        "cka": {
            "main": "RBF CKA / centered normalized HSIC",
            "bandwidth": "median non-zero pairwise Euclidean distance per representation and subset",
            "dimension_normalization": "flatten then divide by sqrt(feature_count) for modality and target alignment only",
            "linear_robustness_check": True,
            "fixed_shuffle_control": "one seeded permutation per subset, applied to target rows",
        },
        "effective_dimension": {
            "matrix": "centered X X^T/(N-1)",
            "feature_standardization": False,
            "sample_unit_norm": False,
            "negative_eigenvalue_handling": "clip numerical negatives to zero",
            "metric": "participation-ratio effective rank",
        },
        "subset_metadata": subset_metadata,
    }
    (output_dir / "analysis_config.json").write_text(json.dumps(config, indent=2, default=_json_default) + "\n", encoding="utf-8")
    _write_report(
        output_dir / "analysis_report.md",
        config=config,
        alignment_rows=alignment_rows_with_summary,
        rank_rows=rank_rows_with_summary,
        subset_metadata=subset_metadata,
    )
    return config


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "mathtext.fontset": "stix",
            "axes.labelcolor": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "axes.edgecolor": "#666666",
            "savefig.facecolor": "white",
        }
    )
    if args.render_v2:
        outputs = render_v2_from_artifacts(args.artifact_dir, args.v2_output_dir)
        print(json.dumps(outputs, indent=2))
        return 0
    config = run_analysis(args)
    print(json.dumps({"output_dir": str(Path(args.output_dir).resolve()), "selected_counts": config["selected_counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
