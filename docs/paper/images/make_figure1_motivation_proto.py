"""Generate Figure 1 motivation prototypes for the AAAI manuscript.

Version v3 addresses the manuscript-facing requirements:
  - data tiles have lightweight axes
  - velocity/decomposition tiles have compact colorbars
  - layout spacing is tighter than the pure-LaTeX table
  - the physical decomposition panel includes explicit formulas
  - fonts are unified with Microsoft YaHei when available
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import Normalize
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


HERE = Path(__file__).resolve().parent
TILE_DIR = HERE / "figure1_visuals"
OUT_BASE = HERE / "figure1_motivation_v4"

STRUCT = "#2563eb"
BACK = "#16a34a"
INK = "#111827"
MUTED = "#64748b"
PANEL = "#cfd8e3"
PANEL_FILL = "#fcfdff"


def configure_fonts() -> None:
    candidates = ["Microsoft YaHei", "Noto Sans CJK SC", "Arial", "DejaVu Sans"]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.family"] = name
            break
    plt.rcParams["mathtext.fontset"] = "dejavusans"
    plt.rcParams["axes.unicode_minus"] = False


def load_tile(name: str):
    path = TILE_DIR / f"{name}.png"
    if not path.exists():
        raise FileNotFoundError(path)
    return mpimg.imread(path)


def panel(ax, xy, wh, title: str) -> None:
    x, y = xy
    w, h = wh
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.006,rounding_size=0.006",
            linewidth=0.7,
            edgecolor=PANEL,
            facecolor=PANEL_FILL,
            zorder=0,
        )
    )
    ax.text(x + 0.014, y + h - 0.030, title, ha="left", va="center", fontsize=9.6, fontweight="bold", color=INK)


def add_tile(
    fig,
    rect: tuple[float, float, float, float],
    image_name: str,
    *,
    title: str,
    subtitle: str,
    subtitle_color: str,
    axis_kind: str = "depth",
):
    x, y, w, h = rect
    ax = fig.add_axes(rect, zorder=4)
    ax.imshow(load_tile(image_name), aspect="equal", interpolation="hanning")
    if axis_kind == "time":
        ax.set_xlabel("x", fontsize=5.7, labelpad=0)
        ax.set_ylabel("t", fontsize=5.7, labelpad=0)
    elif axis_kind == "depth":
        ax.set_xlabel("x", fontsize=5.7, labelpad=0)
        ax.set_ylabel("z", fontsize=5.7, labelpad=0)
    else:
        ax.set_xlabel("x", fontsize=5.7, labelpad=0)
        ax.set_ylabel("", fontsize=5.7, labelpad=0)
    ax.set_xticks([0, load_tile(image_name).shape[1] - 1])
    ax.set_yticks([0, load_tile(image_name).shape[0] - 1])
    ax.set_xticklabels(["0", "1"], fontsize=5.2)
    ax.set_yticklabels(["0", "1"], fontsize=5.2)
    ax.tick_params(length=1.6, width=0.45, pad=1)
    for spine in ax.spines.values():
        spine.set_linewidth(0.55)
        spine.set_edgecolor("#aeb9c7")

    fig.text(x + w / 2, y - 0.030, title, ha="center", va="top", fontsize=7.4, fontweight="bold", color=INK)
    fig.text(x + w / 2, y - 0.056, subtitle, ha="center", va="top", fontsize=6.1, color=subtitle_color)
    return ax


def add_cbar(fig, rect, cmap: str, label: str, norm: Normalize) -> None:
    cax = fig.add_axes(rect, zorder=5)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, cax=cax, orientation="vertical")
    cb.outline.set_linewidth(0.35)
    cb.ax.tick_params(labelsize=5.0, length=1.5, width=0.35, pad=1)
    cb.set_label(label, fontsize=5.7, labelpad=1)


def arrow(ax, start, end, color=MUTED, lw=1.0, rad=0.0, scale=10) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=scale,
            linewidth=lw,
            color=color,
            connectionstyle=f"arc3,rad={rad}",
            zorder=3,
        )
    )


def tag(ax, xy, text: str, color: str, width: float) -> None:
    x, y = xy
    ax.add_patch(
        FancyBboxPatch(
            (x - width / 2, y - 0.013),
            width,
            0.026,
            boxstyle="round,pad=0.003,rounding_size=0.005",
            linewidth=0,
            facecolor=color,
            alpha=0.085,
            zorder=1,
        )
    )
    ax.text(x, y, text, ha="center", va="center", fontsize=7.0, fontweight="bold", color=color, zorder=2)


def main() -> int:
    configure_fonts()

    fig = plt.figure(figsize=(13.6, 4.15), dpi=300)
    canvas = fig.add_axes([0, 0, 1, 1], zorder=1)
    canvas.set_xlim(0, 1)
    canvas.set_ylim(0, 1)
    canvas.axis("off")
    fig.patch.set_facecolor("white")

    left_xy = (0.030, 0.125)
    left_wh = (0.430, 0.800)
    right_xy = (0.545, 0.125)
    right_wh = (0.430, 0.800)
    panel(canvas, left_xy, left_wh, "(a) Heterogeneous evidence")
    panel(canvas, right_xy, right_wh, "(b) Physical decomposition")

    tile_w = 0.098
    tile_h = 0.235
    tag(canvas, (0.245, 0.835), "structural cues", STRUCT, width=0.132)
    add_tile(fig, (0.130, 0.570, tile_w, tile_h), "pstm_dense_reflector_texture", title="PoSTM", subtitle="reflectors", subtitle_color=STRUCT, axis_kind="time")
    add_tile(fig, (0.310, 0.570, tile_w, tile_h), "horizon_sparse_interfaces", title="Horizon", subtitle="interfaces", subtitle_color=STRUCT)
    tag(canvas, (0.245, 0.485), "numerical cues", BACK, width=0.125)
    add_tile(fig, (0.130, 0.220, tile_w, tile_h), "rms_smooth_numerical_trend", title="RMS velocity", subtitle="trend", subtitle_color=BACK, axis_kind="time")
    add_tile(fig, (0.310, 0.220, tile_w, tile_h), "well_log_sparse_absolute_calibration", title="Well log", subtitle="scale", subtitle_color=BACK)

    # Shared compact colorbar for velocity-like input cues.
    add_cbar(fig, (0.425, 0.220, 0.008, 0.235), "jet", "velocity", Normalize(0, 1))

    # Right panel: paper-style triangular decomposition with formulas.
    rt_w = 0.105
    rt_h = 0.248
    add_tile(fig, (0.708, 0.660, rt_w, rt_h), "velocity_target_v", title=r"Target $V$", subtitle="full field", subtitle_color=MUTED)
    add_tile(fig, (0.623, 0.218, rt_w, rt_h), "low_frequency_background_b", title=r"$B=P_L(V)$", subtitle="low-pass", subtitle_color=BACK)
    add_tile(fig, (0.797, 0.218, rt_w, rt_h), "high_frequency_structure_s", title=r"$S=P_H(V)$", subtitle="high-pass", subtitle_color=STRUCT)

    canvas.text(0.760, 0.535, r"$V=P_L(V)+P_H(V)$", ha="center", va="center", fontsize=11.8, color=INK)
    canvas.text(0.760, 0.490, r"$=B+S$", ha="center", va="center", fontsize=11.8, color=INK)
    arrow(canvas, (0.728, 0.620), (0.682, 0.482), color=BACK, lw=1.05, rad=0.05, scale=10)
    arrow(canvas, (0.792, 0.620), (0.846, 0.482), color=STRUCT, lw=1.05, rad=-0.05, scale=10)

    add_cbar(fig, (0.920, 0.660, 0.008, 0.248), "jet", "velocity", Normalize(0, 1))
    add_cbar(fig, (0.920, 0.218, 0.008, 0.248), "RdBu_r", "residual", Normalize(-1, 1))

    arrow(canvas, (0.472, 0.525), (0.532, 0.525), color="#94a3b8", lw=0.9, scale=9)
    canvas.text(0.502, 0.554, "motivation", ha="center", va="bottom", fontsize=6.8, color=MUTED)

    for ext in ("pdf", "png"):
        fig.savefig(OUT_BASE.with_suffix(f".{ext}"), dpi=300)
    plt.close(fig)
    print(f"Wrote {OUT_BASE.with_suffix('.pdf')}")
    print(f"Wrote {OUT_BASE.with_suffix('.png')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
