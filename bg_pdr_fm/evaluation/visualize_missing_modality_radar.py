"""Render the four-panel OpenFWI missing-input modality radar figure."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "logs/bg_pdr_fm/aaai27/eval_missing_modalities_radar_global_heldout/radar_dataset_mode_summary.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "docs/paper/AAAI2027/figures/missing_modality_radar"
RADAR_METHOD_ORDER = (
    "InversionNet",
    "VelocityGAN",
    "UPFWI",
    "Auto-Linear",
    "Latent U-Net (Large)",
    "PD-BG-RFM",
)
RADAR_MODE_ORDER = ("w/o well_log", "w/o horizon", "w/o PSTM", "RMS only")
RADAR_DATASET_ORDER = (
    "FlatVelA",
    "FlatVelB",
    "CurveVelA",
    "CurveVelB",
    "FlatFaultA",
    "FlatFaultB",
    "CurveFaultA",
    "CurveFaultB",
)
RADAR_DATASET_LABELS = {
    "FlatVelA": "FVA",
    "FlatVelB": "FVB",
    "CurveVelA": "CVA",
    "CurveVelB": "CVB",
    "FlatFaultA": "FFA",
    "FlatFaultB": "FFB",
    "CurveFaultA": "CFA",
    "CurveFaultB": "CFB",
}
RADAR_MODE_LABELS = {
    "w/o well_log": "w/o Well",
    "w/o horizon": "w/o Horizon",
    "w/o PSTM": "w/o PoSTM",
    "RMS only": "RMS Only",
}
RADAR_RADIAL_RANGE = (0.50, 1.00)
RADAR_RADIAL_TICKS = (0.50, 0.60, 0.70, 0.80, 0.90, 1.00)
RADAR_FOCUS_RANGE = (0.85, 1.00)
RADAR_FOCUS_TICKS = (0.85, 0.90, 0.95, 1.00)
RADAR_PANEL_RANGES = {
    "w/o well_log": RADAR_FOCUS_RANGE,
    "w/o horizon": RADAR_RADIAL_RANGE,
    "w/o PSTM": RADAR_FOCUS_RANGE,
    "RMS only": RADAR_RADIAL_RANGE,
}
RADAR_PANEL_TICKS = {
    "w/o well_log": RADAR_FOCUS_TICKS,
    "w/o horizon": RADAR_RADIAL_TICKS,
    "w/o PSTM": RADAR_FOCUS_TICKS,
    "RMS only": RADAR_RADIAL_TICKS,
}
METHOD_SLUGS = {
    "InversionNet": "inversion_net",
    "VelocityGAN": "velocity_gan",
    "UPFWI": "upfwi",
    "Auto-Linear": "auto_linear",
    "Latent U-Net (Large)": "latent_unet_large",
    "PD-BG-RFM": "pd_bg_rfm",
}
METHOD_STYLES = {
    "InversionNet": {"color": "#0072B2", "linestyle": "-", "marker": "o"},
    "VelocityGAN": {"color": "#E69F00", "linestyle": "--", "marker": "s"},
    "UPFWI": {"color": "#009E73", "linestyle": "-.", "marker": "^"},
    "Auto-Linear": {"color": "#CC79A7", "linestyle": ":", "marker": "D"},
    "Latent U-Net (Large)": {"color": "#56B4E9", "linestyle": (0, (5, 1)), "marker": "v"},
    "PD-BG-RFM": {"color": "#D55E00", "linestyle": "-", "marker": "*"},
}
METRIC_SOURCE_FIELDS = ("method", "missing_mode", "dataset_name", "num_samples", "ssim")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_radar_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate the exact method × scenario × subset grid used by the figure."""

    expected = {
        (method, mode, dataset)
        for method in RADAR_METHOD_ORDER
        for mode in RADAR_MODE_ORDER
        for dataset in RADAR_DATASET_ORDER
    }
    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        key = (
            str(row.get("method", "")),
            str(row.get("missing_mode", "")),
            str(row.get("dataset_name", "")),
        )
        if key not in expected:
            raise ValueError(f"unexpected radar row identity: {key}")
        if key in seen:
            raise ValueError(f"duplicate radar row identity: {key}")
        seen.add(key)
        try:
            count = int(row["num_samples"])
            ssim = float(row["ssim"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid radar row values for {key}") from exc
        if count <= 0:
            raise ValueError(f"non-positive radar sample count for {key}: {count}")
        if not math.isfinite(ssim) or not 0.0 <= ssim <= 1.0:
            raise ValueError(f"SSIM must be finite and in [0, 1] for {key}: {ssim}")
    missing = expected.difference(seen)
    if missing:
        raise ValueError(f"radar source is missing {len(missing)} grid rows: {sorted(missing)[:3]}")
    return {
        "status": "passed",
        "num_rows": len(rows),
        "num_methods": len(RADAR_METHOD_ORDER),
        "num_modes": len(RADAR_MODE_ORDER),
        "num_datasets": len(RADAR_DATASET_ORDER),
    }


def load_radar_rows(path: str | Path) -> list[dict[str, str]]:
    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Radar source CSV does not exist: {source}")
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing_fields = sorted(set(METRIC_SOURCE_FIELDS).difference(reader.fieldnames or ()))
        if missing_fields:
            raise ValueError(f"Radar source CSV is missing fields: {missing_fields}")
        rows = [dict(row) for row in reader]
    validate_radar_rows(rows)
    return rows


def _matrix(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, list[float]]]:
    values = {
        mode: {method: [0.0] * len(RADAR_DATASET_ORDER) for method in RADAR_METHOD_ORDER}
        for mode in RADAR_MODE_ORDER
    }
    dataset_index = {name: index for index, name in enumerate(RADAR_DATASET_ORDER)}
    for row in rows:
        method = str(row["method"])
        mode = str(row["missing_mode"])
        dataset = str(row["dataset_name"])
        values[mode][method][dataset_index[dataset]] = float(row["ssim"])
    return values


def _write_source_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(METRIC_SOURCE_FIELDS),
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in METRIC_SOURCE_FIELDS})


def _checkpoint_manifest(eval_root: Path) -> dict[str, Any]:
    checkpoints: dict[str, Any] = {}
    for method in RADAR_METHOD_ORDER:
        run_manifest_path = eval_root / METHOD_SLUGS[method] / "run_manifest.json"
        if not run_manifest_path.is_file():
            raise FileNotFoundError(f"Missing method run manifest: {run_manifest_path}")
        payload = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        checkpoints[method] = {
            "path": payload.get("checkpoint", {}).get("path", ""),
            "sha256": payload.get("checkpoint", {}).get("sha256", ""),
            "run_manifest": str(run_manifest_path.resolve()),
        }
    return checkpoints


def _git_commit() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip()


def render_radar_figure(rows: Sequence[Mapping[str, Any]], output_dir: str | Path) -> dict[str, str]:
    validate_radar_rows(rows)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    matrix = _matrix(rows)
    theta = np.linspace(0.0, 2.0 * np.pi, len(RADAR_DATASET_ORDER), endpoint=False)
    theta_closed = np.concatenate([theta, theta[:1]])
    labels = [RADAR_DATASET_LABELS[name] for name in RADAR_DATASET_ORDER]

    plt.rcParams.update({"font.family": "serif", "axes.unicode_minus": False})
    figure, axes = plt.subplots(
        1,
        len(RADAR_MODE_ORDER),
        subplot_kw={"polar": True},
        figsize=(7.15, 3.15),
        dpi=160,
    )
    axes = np.asarray(axes).reshape(-1)
    handles: list[Line2D] = []
    for panel_index, (axis, mode) in enumerate(zip(axes, RADAR_MODE_ORDER)):
        panel_range = RADAR_PANEL_RANGES[mode]
        panel_ticks = RADAR_PANEL_TICKS[mode]
        axis.set_theta_offset(np.pi / 2.0)
        axis.set_theta_direction(-1)
        axis.set_ylim(*panel_range)
        axis.set_yticks(panel_ticks)
        axis.set_yticklabels(
            tuple(f"{tick:.2f}" for tick in panel_ticks),
            fontsize=5.2,
            color="#555555",
        )
        axis.set_xticks(theta)
        axis.set_xticklabels(labels, fontsize=4.5)
        axis.tick_params(axis="x", pad=0)
        axis.grid(color="#B8B8B8", linewidth=0.45, alpha=0.75)
        axis.spines["polar"].set_color("#666666")
        axis.spines["polar"].set_linewidth(0.55)
        axis.set_title(
            f"({chr(ord('a') + panel_index)}) {RADAR_MODE_LABELS[mode]}",
            fontsize=7.6,
            pad=10,
            fontweight="bold",
        )
        for method in RADAR_METHOD_ORDER:
            style = METHOD_STYLES[method]
            values = np.asarray(matrix[mode][method] + [matrix[mode][method][0]], dtype=float)
            line_width = 2.0 if method == "PD-BG-RFM" else 1.0
            marker_size = 3.3 if method == "PD-BG-RFM" else 2.0
            line = axis.plot(
                theta_closed,
                values,
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=line_width,
                marker=style["marker"],
                markersize=marker_size,
                markerfacecolor="white",
                markeredgewidth=0.45,
                markeredgecolor=style["color"],
                alpha=1.0 if method == "PD-BG-RFM" else 0.84,
                zorder=5 if method == "PD-BG-RFM" else 3,
            )[0]
            if panel_index == 0:
                handles.append(line)

    figure.legend(
        handles,
        list(RADAR_METHOD_ORDER),
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=3,
        frameon=False,
        fontsize=6.0,
        handlelength=2.2,
        columnspacing=1.2,
    )
    figure.subplots_adjust(left=0.015, right=0.985, top=0.84, bottom=0.235, wspace=0.55)
    pdf_path = output / "missing_modality_radar_4panel.pdf"
    png_path = output / "missing_modality_radar_4panel.png"
    figure.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    figure.savefig(png_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return {"pdf": str(pdf_path), "png": str(png_path)}


def build_radar_figure(
    *,
    input_csv: str | Path = DEFAULT_INPUT,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    input_path = Path(input_csv).resolve()
    output = Path(output_dir).resolve()
    rows = load_radar_rows(input_path)
    figure_paths = render_radar_figure(rows, output)
    source_path = output / "missing_modality_radar_source.csv"
    _write_source_csv(source_path, rows)
    eval_root = input_path.parent
    validation_path = eval_root / "evaluation_validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8")) if validation_path.is_file() else {}
    manifest = {
        "status": "passed",
        "input_csv": str(input_path),
        "input_csv_sha256": _sha256(input_path),
        "canonical_manifest": validation.get("canonical_manifest", ""),
        "canonical_manifest_sha256": validation.get("canonical_manifest_sha256", ""),
        "git_commit": _git_commit(),
        "methods": list(RADAR_METHOD_ORDER),
        "datasets": list(RADAR_DATASET_ORDER),
        "dataset_labels": RADAR_DATASET_LABELS,
        "modes": list(RADAR_MODE_ORDER),
        "radial_range": {mode: list(RADAR_PANEL_RANGES[mode]) for mode in RADAR_MODE_ORDER},
        "radial_ticks": {mode: list(RADAR_PANEL_TICKS[mode]) for mode in RADAR_MODE_ORDER},
        "radial_scale": "panel-specific zoomed raw SSIM display; values are not normalized",
        "metric": "SSIM",
        "num_rows": len(rows),
        "checkpoint_provenance": _checkpoint_manifest(eval_root),
        "figure_paths": figure_paths,
        "source_csv": str(source_path),
        "source_csv_sha256": _sha256(source_path),
    }
    manifest_path = output / "missing_modality_radar_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"figure_paths": figure_paths, "source_csv": str(source_path), "manifest": str(manifest_path)}


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    args = parser.parse_args(argv)
    result = build_radar_figure(input_csv=args.input_csv, output_dir=args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
