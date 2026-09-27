"""Export drawing-ready Figure 1 tiles and colorbars.

These assets are intended for manual layout in Visio/Inkscape/Illustrator.
PoSTM and RMS are exported as 1:1 visual thumbnails by design, while metadata
records their original array shapes.
"""

from __future__ import annotations

import io
import json
from pathlib import Path

import lmdb
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm


HERE = Path(__file__).resolve().parent
LMDB_ROOT = Path("/public/home/xuyinghao/workspace/datasets/openfwi_lmdb")
DATASET = "FlatFaultB"
SAMPLE_INDEX = 27006
OUT_DIR = HERE / "figure1_drawing_tiles"

VEL_MIN = 1600.0
VEL_MAX = 4400.0
HIGH_ABS = 800.0


def squeeze_image(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    while x.ndim > 2 and x.shape[0] == 1:
        x = x[0]
    if x.ndim == 3:
        x = x.mean(axis=0)
    return x.astype(np.float32)


def load_modality(name: str) -> np.ndarray:
    env = lmdb.open(str(LMDB_ROOT / f"{DATASET}.lmdb"), readonly=True, lock=False, readahead=True, meminit=False)
    try:
        with env.begin() as txn:
            raw = txn.get(f"{SAMPLE_INDEX:08d}:{name}".encode("ascii"))
        if raw is None:
            raise KeyError(f"Missing {name}")
        return squeeze_image(np.load(io.BytesIO(raw), allow_pickle=False))
    finally:
        env.close()


def lowpass_avg(x: np.ndarray, kernel: int = 5) -> np.ndarray:
    pad = kernel // 2
    padded = np.pad(x, ((pad, pad), (pad, pad)), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (kernel, kernel))
    return windows.mean(axis=(-2, -1)).astype(np.float32)


def save_tile(array: np.ndarray, name: str, cmap: str, norm, *, masked_white: bool = False) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(3.0, 3.0), dpi=300)
    if masked_white:
        cmap_obj = plt.get_cmap(cmap).copy()
        cmap_obj.set_bad("white")
        data = np.ma.masked_invalid(array)
    else:
        cmap_obj = cmap
        data = array
    ax.imshow(data, cmap=cmap_obj, norm=norm, aspect="auto", interpolation="nearest")
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(OUT_DIR / f"{name}.png", dpi=300)
    plt.close(fig)


def save_colorbar(name: str, cmap: str, norm, label: str, ticks: list[float], ticklabels: list[str]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(0.75, 3.0), dpi=300)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, cax=ax, orientation="vertical", ticks=ticks)
    cb.ax.set_yticklabels(ticklabels)
    cb.set_label(label, fontsize=8)
    cb.ax.tick_params(labelsize=7, length=2, width=0.5)
    cb.outline.set_linewidth(0.5)
    fig.savefig(OUT_DIR / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> int:
    depth = load_modality("depth_vel")
    pstm = load_modality("migrated_image")
    horizon = load_modality("horizon")
    rms = load_modality("rms_vel")
    low = lowpass_avg(depth)
    high = depth - low
    well = np.full_like(depth, np.nan)
    well_columns = [20, 43]
    for col in well_columns:
        well[:, col] = depth[:, col]

    velocity_norm = Normalize(VEL_MIN, VEL_MAX)
    residual_norm = TwoSlopeNorm(vmin=-HIGH_ABS, vcenter=0, vmax=HIGH_ABS)

    save_tile(pstm, "pstm_1to1", "gray", Normalize(-250, 180))
    save_tile(rms, "rms_1to1", "jet", velocity_norm)
    save_tile(np.where(horizon > 0, 0.0, 1.0), "horizon_1to1", "gray", Normalize(0, 1))
    save_tile(well, "well_1to1", "jet", velocity_norm, masked_white=True)
    save_tile(depth, "velocity_1to1", "jet", velocity_norm)
    save_tile(low, "background_1to1", "jet", velocity_norm)
    save_tile(high, "structure_1to1", "RdBu_r", residual_norm)

    save_colorbar(
        "colorbar_rms_velocity",
        "jet",
        velocity_norm,
        "RMS velocity (m/s)",
        [VEL_MIN, 3000, VEL_MAX],
        ["1600", "3000", "4400"],
    )
    save_colorbar(
        "colorbar_velocity",
        "jet",
        velocity_norm,
        "Velocity (m/s)",
        [VEL_MIN, 3000, VEL_MAX],
        ["1600", "3000", "4400"],
    )
    save_colorbar(
        "colorbar_structure",
        "RdBu_r",
        residual_norm,
        "Structure residual (m/s)",
        [-HIGH_ABS, 0, HIGH_ABS],
        ["-800", "0", "800"],
    )

    metadata = {
        "dataset": DATASET,
        "sample_index": SAMPLE_INDEX,
        "note": "PoSTM and RMS are exported as 1:1 visual thumbnails for manual drawing; original shapes are recorded below.",
        "original_shapes": {
            "pstm": list(pstm.shape),
            "rms": list(rms.shape),
            "horizon": list(horizon.shape),
            "well": list(well.shape),
            "velocity": list(depth.shape),
        },
        "colorbars": {
            "rms_velocity": {"range_m_per_s": [VEL_MIN, VEL_MAX], "ticks": [VEL_MIN, 3000, VEL_MAX]},
            "velocity": {"range_m_per_s": [VEL_MIN, VEL_MAX], "ticks": [VEL_MIN, 3000, VEL_MAX]},
            "structure": {"range_m_per_s": [-HIGH_ABS, HIGH_ABS], "ticks": [-HIGH_ABS, 0, HIGH_ABS]},
        },
        "finite_ranges": {
            "depth_vel": [float(np.nanmin(depth)), float(np.nanmax(depth))],
            "rms": [float(np.nanmin(rms)), float(np.nanmax(rms))],
            "well": [float(np.nanmin(well)), float(np.nanmax(well))],
            "structure": [float(np.nanmin(high)), float(np.nanmax(high))],
        },
        "well_columns": well_columns,
    }
    (OUT_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote drawing assets to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
