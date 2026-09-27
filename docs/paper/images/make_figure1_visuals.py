"""Export editable visual assets for Figure 1.

This script creates representative OpenFWI thumbnails for the Figure 1
motivation panel:

  - asymmetric multimodal evidence: PSTM, horizon, RMS velocity, well log
  - physical decomposition: V, low-frequency background B, high-frequency S

Single-tile outputs are pure 1:1 image tiles: no margins, titles, tick labels,
or colorbars. Combined panel drafts keep lightweight labels for layout review.

Example:

    python docs/paper/images/make_figure1_visuals.py

    python docs/paper/images/make_figure1_visuals.py \
      --dataset CurveFaultA \
      --sample-index 27006 \
      --out-dir docs/paper/images/figure1_visuals
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_LMDB_ROOT = Path("/public/home/xuyinghao/workspace/datasets/openfwi_lmdb")
DEFAULT_SOURCE_ROOT = Path("/public/home/xuyinghao/workspace/datasets/openfwi")
DEFAULT_OUT_DIR = Path(__file__).resolve().parent / "figure1_visuals"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="CurveFaultA", help="OpenFWI subset name.")
    parser.add_argument("--sample-index", type=int, default=27006, help="Sample index within the subset.")
    parser.add_argument("--lmdb-root", type=Path, default=DEFAULT_LMDB_ROOT, help="OpenFWI LMDB root.")
    parser.add_argument(
        "--source-root",
        type=Path,
        default=None,
        help="Optional OpenFWI .npy root. If provided, it is used before LMDB.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Directory for generated assets.")
    parser.add_argument("--lowpass-kernel", type=int, default=5, help="Odd kernel size matching LowHighPassFilter.")
    parser.add_argument(
        "--lowpass-pad-mode",
        choices=("edge", "reflect", "constant"),
        default="edge",
        help=(
            "Padding used for the Figure 1 low/high-frequency visualization. "
            "'edge' avoids zero-padding border artifacts; 'constant' matches avg_pool2d padding."
        ),
    )
    parser.add_argument(
        "--pstm-crop",
        default="none",
        help="Optional PSTM vertical crop as start:stop, e.g. 0:350. Use 'none' to keep the full image.",
    )
    parser.add_argument(
        "--well-columns",
        default="auto",
        help="Comma-separated well columns, or 'auto' for sparse deterministic columns.",
    )
    parser.add_argument("--dpi", type=int, default=240, help="Raster output DPI.")
    parser.add_argument(
        "--formats",
        default="png,pdf",
        help="Comma-separated output formats supported by matplotlib, e.g. png,pdf,svg.",
    )
    parser.add_argument(
        "--no-combined",
        action="store_true",
        help="Only export individual thumbnails; skip combined panel drafts.",
    )
    return parser.parse_args()


def require_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "This script requires matplotlib. Install it in the active environment, "
            "for example: python -m pip install matplotlib"
        ) from exc
    return plt, TwoSlopeNorm


def _squeeze_image(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    while x.ndim > 2 and x.shape[0] == 1:
        x = x[0]
    if x.ndim == 3:
        x = np.mean(x, axis=0)
    if x.ndim != 2:
        raise ValueError(f"Expected a 2D image after squeezing, got shape {x.shape}.")
    return np.asarray(x, dtype=np.float32)


def apply_vertical_crop(image: np.ndarray, spec: str) -> np.ndarray:
    spec = str(spec).strip().lower()
    if spec in {"", "none", "full"}:
        return image
    if ":" not in spec:
        raise ValueError("--pstm-crop must be 'none' or a start:stop range.")
    start_raw, stop_raw = spec.split(":", 1)
    start = 0 if not start_raw else int(start_raw)
    stop = image.shape[0] if not stop_raw else int(stop_raw)
    start = max(0, min(start, image.shape[0]))
    stop = max(start + 1, min(stop, image.shape[0]))
    return image[start:stop, :]


def _lmdb_sample_key(sample_index: int, modality: str) -> bytes:
    return f"{int(sample_index):08d}:{modality}".encode("ascii")


def _load_from_lmdb(lmdb_root: Path, dataset: str, sample_index: int, modality: str) -> np.ndarray:
    try:
        import lmdb
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "LMDB reading requires the lmdb package. Install it with: python -m pip install lmdb\n"
            "Alternatively pass --source-root to read the original .npy files."
        ) from exc

    lmdb_path = lmdb_root / f"{dataset}.lmdb"
    manifest_path = lmdb_path / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing LMDB manifest: {manifest_path}")

    env = lmdb.open(str(lmdb_path), readonly=True, lock=False, readahead=True, meminit=False)
    try:
        with env.begin(write=False) as txn:
            value = txn.get(_lmdb_sample_key(sample_index, modality))
        if value is None:
            raise KeyError(f"Missing sample {sample_index} modality {modality!r} in {lmdb_path}.")
        return np.asarray(np.load(io.BytesIO(value), allow_pickle=False))
    finally:
        env.close()


def _candidate_npy_paths(source_root: Path, dataset: str, sample_index: int, modality: str) -> Iterable[Path]:
    modality_dir = source_root / dataset / modality
    yield modality_dir / f"{sample_index}.npy"
    yield modality_dir / f"{sample_index:08d}.npy"
    yield modality_dir / f"{sample_index:06d}.npy"


def _load_from_source(source_root: Path, dataset: str, sample_index: int, modality: str) -> np.ndarray:
    for path in _candidate_npy_paths(source_root, dataset, sample_index, modality):
        if path.is_file():
            return np.asarray(np.load(path, allow_pickle=False))
    raise FileNotFoundError(
        f"Could not find {dataset}/{modality}/{sample_index}.npy under {source_root}. "
        "Use --lmdb-root or check the sample index."
    )


def load_modality(args: argparse.Namespace, modality: str) -> np.ndarray:
    if args.source_root is not None:
        return _squeeze_image(_load_from_source(args.source_root, args.dataset, args.sample_index, modality))
    return _squeeze_image(_load_from_lmdb(args.lmdb_root, args.dataset, args.sample_index, modality))


def lowpass_avg(x: np.ndarray, kernel_size: int, pad_mode: str = "edge") -> np.ndarray:
    if kernel_size % 2 == 0:
        raise ValueError("--lowpass-kernel must be odd.")
    pad = kernel_size // 2
    if pad_mode not in {"edge", "reflect", "constant"}:
        raise ValueError("--lowpass-pad-mode must be one of 'edge', 'reflect', or 'constant'.")
    padded = np.pad(np.asarray(x, dtype=np.float32), ((pad, pad), (pad, pad)), mode=pad_mode)
    windows = np.lib.stride_tricks.sliding_window_view(padded, (kernel_size, kernel_size))
    return windows.mean(axis=(-2, -1)).astype(np.float32)


def parse_well_columns(spec: str, width: int) -> list[int]:
    if spec.strip().lower() == "auto":
        # Sparse but visible columns for a paper thumbnail.
        return sorted({max(0, min(width - 1, int(round(width * frac)))) for frac in (0.28, 0.62)})
    columns = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        col = int(item)
        if col < 0 or col >= width:
            raise ValueError(f"Well column {col} is outside image width {width}.")
        columns.append(col)
    if not columns:
        raise ValueError("--well-columns produced no columns.")
    return sorted(set(columns))


def build_well_log(depth_velocity: np.ndarray, columns: list[int]) -> tuple[np.ndarray, np.ndarray]:
    well_log = np.full_like(depth_velocity, np.nan, dtype=np.float32)
    well_mask = np.zeros_like(depth_velocity, dtype=np.float32)
    for col in columns:
        well_log[:, col] = depth_velocity[:, col]
        well_mask[:, col] = 1.0
    return well_log, well_mask


def horizon_display_image(horizon: np.ndarray) -> np.ndarray:
    """Render missing horizon pixels as white and horizon lines as black."""
    return np.where(np.asarray(horizon) > 0, 0.0, 1.0).astype(np.float32)


def masked_cmap(name: str, bad_color: str = "white"):
    plt, _ = require_matplotlib()
    cmap = plt.get_cmap(name).copy()
    cmap.set_bad(bad_color)
    return cmap


def robust_limits(x: np.ndarray, *, symmetric: bool = False, lower: float = 1.0, upper: float = 99.0) -> tuple[float, float]:
    finite = np.asarray(x)[np.isfinite(x)]
    if finite.size == 0:
        return 0.0, 1.0
    if symmetric:
        vmax = float(np.percentile(np.abs(finite), upper))
        vmax = vmax if vmax > 0 else 1.0
        return -vmax, vmax
    lo, hi = (float(v) for v in np.percentile(finite, [lower, upper]))
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def add_axis_text(
    ax,
    *,
    x_label: str = "lateral position",
    y_label: str = "depth",
    color: str = "0.18",
    fontsize: int = 8,
) -> None:
    ax.text(
        0.5,
        -0.065,
        x_label,
        ha="center",
        va="top",
        transform=ax.transAxes,
        fontsize=fontsize,
        color=color,
        clip_on=False,
    )
    ax.text(
        -0.055,
        0.5,
        y_label,
        ha="right",
        va="center",
        rotation=90,
        transform=ax.transAxes,
        fontsize=fontsize,
        color=color,
        clip_on=False,
    )


def style_axis(
    ax,
    title: str,
    subtitle: str | None = None,
    *,
    x_label: str = "lateral position",
    y_label: str = "depth",
) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.text(
        0.5,
        1.145,
        title,
        ha="center",
        va="bottom",
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        color="black",
        clip_on=False,
    )
    if subtitle:
        ax.text(
            0.5,
            1.065,
            subtitle,
            ha="center",
            va="bottom",
            transform=ax.transAxes,
            fontsize=7.5,
            color="0.30",
        )
    add_axis_text(ax, x_label=x_label, y_label=y_label)


def show_image(
    ax,
    image: np.ndarray,
    *,
    cmap: str,
    title: str,
    subtitle: str | None = None,
    symmetric: bool = False,
    x_label: str = "lateral position",
    y_label: str = "depth",
):
    plt, TwoSlopeNorm = require_matplotlib()
    vmin, vmax = robust_limits(image, symmetric=symmetric)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax) if symmetric else None
    cmap_obj = masked_cmap(cmap, "white") if np.ma.isMaskedArray(image) else cmap
    ax.imshow(image, cmap=cmap_obj, aspect="auto", vmin=None if norm else vmin, vmax=None if norm else vmax, norm=norm)
    style_axis(ax, title, subtitle, x_label=x_label, y_label=y_label)


def show_image_tile(ax, image: np.ndarray, *, cmap: str, symmetric: bool = False) -> None:
    plt, TwoSlopeNorm = require_matplotlib()
    vmin, vmax = robust_limits(image, symmetric=symmetric)
    norm = TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax) if symmetric else None
    cmap_obj = masked_cmap(cmap, "white") if np.ma.isMaskedArray(image) else cmap
    ax.imshow(image, cmap=cmap_obj, aspect="auto", vmin=None if norm else vmin, vmax=None if norm else vmax, norm=norm)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def save_current(fig, base_path: Path, formats: list[str], dpi: int) -> list[str]:
    written = []
    for fmt in formats:
        out = base_path.with_suffix(f".{fmt}")
        fig.savefig(out, dpi=dpi)
        written.append(str(out))
    return written


def save_single_image(
    image: np.ndarray,
    *,
    out_dir: Path,
    name: str,
    title: str,
    subtitle: str,
    cmap: str,
    formats: list[str],
    dpi: int,
    symmetric: bool = False,
    x_label: str = "lateral position",
    y_label: str = "depth",
) -> list[str]:
    plt, _ = require_matplotlib()
    del title, subtitle, x_label, y_label
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    show_image_tile(ax, image, cmap=cmap, symmetric=symmetric)
    fig.patch.set_facecolor("white")
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0)
    paths = save_current(fig, out_dir / name, formats, dpi)
    plt.close(fig)
    return paths


def make_asymmetry_panel(assets: dict[str, np.ndarray], out_dir: Path, formats: list[str], dpi: int) -> list[str]:
    plt, _ = require_matplotlib()
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2))
    fig.patch.set_facecolor("white")
    fig.suptitle("(a) Asymmetric multimodal evidence", fontsize=13, fontweight="bold", y=0.985)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.81, bottom=0.13, wspace=0.12, hspace=0.62)

    show_image(
        axes[0, 0],
        assets["pstm"],
        cmap="gray",
        title="PSTM",
        subtitle="dense reflector texture",
        y_label="time/depth",
    )
    show_image(
        axes[0, 1],
        assets["horizon"],
        cmap="gray",
        title="Horizon",
        subtitle="sparse interface prior",
    )
    show_image(
        axes[1, 0],
        assets["rms"],
        cmap="jet",
        title="RMS velocity",
        subtitle="smooth numerical trend",
        y_label="time/depth",
    )
    show_image(
        axes[1, 1],
        assets["well_log"],
        cmap="jet",
        title="Well log",
        subtitle="sparse absolute calibration",
    )

    fig.text(0.50, 0.495, "structural cues", ha="center", fontsize=10, color="#087f5b", fontweight="bold")
    fig.text(0.50, 0.035, "numerical background cues", ha="center", fontsize=10, color="#1c4fa3", fontweight="bold")
    paths = save_current(fig, out_dir / "figure1a_multimodal_asymmetry_panel", formats, dpi)
    plt.close(fig)
    return paths


def make_decomposition_panel(assets: dict[str, np.ndarray], out_dir: Path, formats: list[str], dpi: int) -> list[str]:
    plt, _ = require_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 2.8))
    fig.patch.set_facecolor("white")
    fig.suptitle("(b) Physical decomposition", fontsize=13, fontweight="bold", y=0.985)
    fig.subplots_adjust(left=0.055, right=0.985, top=0.73, bottom=0.22, wspace=0.14)
    show_image(axes[0], assets["depth_vel"], cmap="jet", title="Velocity V", subtitle="target field")
    show_image(axes[1], assets["low"], cmap="jet", title="Background B", subtitle="low-frequency P_L(V)")
    show_image(axes[2], assets["high"], cmap="RdBu_r", title="Structure S", subtitle="high-frequency P_H(V)", symmetric=True)
    fig.text(0.50, 0.025, "V = B + S", ha="center", fontsize=11, fontweight="bold")
    paths = save_current(fig, out_dir / "figure1b_physical_decomposition_panel", formats, dpi)
    plt.close(fig)
    return paths


def write_metadata(path: Path, args: argparse.Namespace, assets: dict[str, np.ndarray], well_columns: list[int]) -> None:
    def asset_stats(value) -> dict[str, object]:
        if np.ma.isMaskedArray(value):
            finite = np.asarray(value.compressed(), dtype=np.float32)
            shape = list(value.shape)
        else:
            array = np.asarray(value, dtype=np.float32)
            finite = array[np.isfinite(array)]
            shape = list(array.shape)
        if finite.size == 0:
            return {"shape": shape, "finite_min": None, "finite_max": None, "finite_mean": None}
        return {
            "shape": shape,
            "finite_min": float(finite.min()),
            "finite_max": float(finite.max()),
            "finite_mean": float(finite.mean()),
        }

    metadata = {
        "dataset": args.dataset,
        "sample_index": int(args.sample_index),
        "lmdb_root": str(args.lmdb_root),
        "source_root": None if args.source_root is None else str(args.source_root),
        "lowpass_kernel": int(args.lowpass_kernel),
        "lowpass_pad_mode": str(args.lowpass_pad_mode),
        "pstm_crop": str(args.pstm_crop),
        "well_columns": [int(c) for c in well_columns],
        "assets": {key: asset_stats(value) for key, value in assets.items()},
        "display_policy": {
            "single_tiles": "1:1 pure image tiles with plt.subplots_adjust(top=1, bottom=0, right=1, left=0)",
            "horizon": "missing/background pixels are white; horizon lines are black",
            "well_log": "unobserved pixels are white; observed well columns use jet velocity colors",
        },
    }
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    formats = [fmt.strip().lower().lstrip(".") for fmt in args.formats.split(",") if fmt.strip()]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    depth = load_modality(args, "depth_vel")
    pstm = apply_vertical_crop(load_modality(args, "migrated_image"), args.pstm_crop)
    horizon = load_modality(args, "horizon")
    rms = load_modality(args, "rms_vel")
    low = lowpass_avg(depth, args.lowpass_kernel, args.lowpass_pad_mode)
    high = depth - low
    well_columns = parse_well_columns(args.well_columns, depth.shape[-1])
    well_log, _ = build_well_log(depth, well_columns)
    horizon_display = horizon_display_image(horizon)
    well_display = np.ma.masked_invalid(well_log)

    assets = {
        "pstm": pstm,
        "horizon": horizon_display,
        "rms": rms,
        "well_log": well_display,
        "depth_vel": depth,
        "low": low,
        "high": high,
    }

    written: list[str] = []
    written += save_single_image(
        pstm,
        out_dir=args.out_dir,
        name="pstm_dense_reflector_texture",
        title="PSTM",
        subtitle="dense reflector texture",
        cmap="gray",
        formats=formats,
        dpi=args.dpi,
        y_label="time/depth",
    )
    written += save_single_image(
        horizon_display,
        out_dir=args.out_dir,
        name="horizon_sparse_interfaces",
        title="Horizon",
        subtitle="sparse interface prior",
        cmap="gray",
        formats=formats,
        dpi=args.dpi,
    )
    written += save_single_image(
        rms,
        out_dir=args.out_dir,
        name="rms_smooth_numerical_trend",
        title="RMS velocity",
        subtitle="smooth numerical trend",
        cmap="jet",
        formats=formats,
        dpi=args.dpi,
        y_label="time/depth",
    )
    written += save_single_image(
        well_log,
        out_dir=args.out_dir,
        name="well_log_sparse_absolute_calibration",
        title="Well log",
        subtitle="sparse absolute calibration",
        cmap="jet",
        formats=formats,
        dpi=args.dpi,
    )
    written += save_single_image(
        depth,
        out_dir=args.out_dir,
        name="velocity_target_v",
        title="Velocity V",
        subtitle="target field",
        cmap="jet",
        formats=formats,
        dpi=args.dpi,
    )
    written += save_single_image(
        low,
        out_dir=args.out_dir,
        name="low_frequency_background_b",
        title="Background B",
        subtitle="low-frequency P_L(V)",
        cmap="jet",
        formats=formats,
        dpi=args.dpi,
    )
    written += save_single_image(
        high,
        out_dir=args.out_dir,
        name="high_frequency_structure_s",
        title="Structure S",
        subtitle="high-frequency P_H(V)",
        cmap="RdBu_r",
        formats=formats,
        dpi=args.dpi,
        symmetric=True,
    )
    if not args.no_combined:
        written += make_asymmetry_panel(assets, args.out_dir, formats, args.dpi)
        written += make_decomposition_panel(assets, args.out_dir, formats, args.dpi)

    metadata_path = args.out_dir / "figure1_visuals_metadata.json"
    write_metadata(metadata_path, args, assets, well_columns)
    written.append(str(metadata_path))

    print("Generated Figure 1 visual assets:")
    for path in written:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
