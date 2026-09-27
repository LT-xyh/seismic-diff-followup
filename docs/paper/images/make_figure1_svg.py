"""Generate Figure 1 as an editable SVG with real axes and colorbars.

The script reloads the selected OpenFWI sample, exports scientific raster tiles
with fixed physical color mapping, then embeds those tiles into an SVG layout.
"""

from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from xml.sax.saxutils import escape

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
TILE_DIR = HERE / "figure1_svg_tiles"
OUT = HERE / "figure1_motivation.svg"

W, H = 1600, 560
FONT = "Microsoft YaHei, Noto Sans CJK SC, Arial, sans-serif"
INK = "#111827"
MUTED = "#64748b"
PANEL = "#cfd8e3"
PANEL_FILL = "#fcfdff"
STRUCT = "#2563eb"
BACK = "#16a34a"
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
    lmdb_path = LMDB_ROOT / f"{DATASET}.lmdb"
    env = lmdb.open(str(lmdb_path), readonly=True, lock=False, readahead=True, meminit=False)
    try:
        with env.begin() as txn:
            raw = txn.get(f"{SAMPLE_INDEX:08d}:{name}".encode("ascii"))
        if raw is None:
            raise KeyError(f"Missing {name} for {DATASET}:{SAMPLE_INDEX}")
        return squeeze_image(np.load(io.BytesIO(raw), allow_pickle=False))
    finally:
        env.close()


def lowpass_avg(x: np.ndarray, kernel: int = 5) -> np.ndarray:
    pad = kernel // 2
    padded = np.pad(x, ((pad, pad), (pad, pad)), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, (kernel, kernel))
    return windows.mean(axis=(-2, -1)).astype(np.float32)


def save_tile(
    array: np.ndarray,
    name: str,
    cmap: str,
    norm,
    *,
    masked_white: bool = False,
    figsize: tuple[float, float] = (2.0, 2.0),
) -> None:
    TILE_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    if masked_white:
        cmap_obj = plt.get_cmap(cmap).copy()
        cmap_obj.set_bad("white")
        image = np.ma.masked_invalid(array)
    else:
        cmap_obj = cmap
        image = array
    ax.imshow(image, cmap=cmap_obj, norm=norm, aspect="auto", interpolation="nearest")
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(TILE_DIR / f"{name}.png", dpi=300)
    plt.close(fig)


def generate_tiles() -> dict[str, object]:
    depth = load_modality("depth_vel")
    pstm = load_modality("migrated_image")
    horizon = load_modality("horizon")
    rms = load_modality("rms_vel")
    low = lowpass_avg(depth, 5)
    high = depth - low
    well = np.full_like(depth, np.nan)
    for col in (20, 43):
        well[:, col] = depth[:, col]

    save_tile(pstm, "pstm", "gray", Normalize(-250, 180), figsize=(1.35, 2.4))
    save_tile(np.where(horizon > 0, 0.0, 1.0), "horizon", "gray", Normalize(0, 1))
    save_tile(rms, "rms", "jet", Normalize(VEL_MIN, VEL_MAX), figsize=(1.35, 2.4))
    save_tile(well, "well", "jet", Normalize(VEL_MIN, VEL_MAX), masked_white=True)
    save_tile(depth, "velocity", "jet", Normalize(VEL_MIN, VEL_MAX))
    save_tile(low, "background", "jet", Normalize(VEL_MIN, VEL_MAX))
    save_tile(high, "structure", "RdBu_r", TwoSlopeNorm(vmin=-HIGH_ABS, vcenter=0, vmax=HIGH_ABS))

    metadata = {
        "dataset": DATASET,
        "sample_index": SAMPLE_INDEX,
        "velocity_colorbar_m_per_s": [VEL_MIN, VEL_MAX],
        "residual_colorbar_m_per_s": [-HIGH_ABS, HIGH_ABS],
        "shapes": {
            "pstm": list(pstm.shape),
            "horizon": list(horizon.shape),
            "rms": list(rms.shape),
            "well": list(well.shape),
            "velocity": list(depth.shape),
        },
        "finite_ranges": {
            "depth_vel": [float(np.nanmin(depth)), float(np.nanmax(depth))],
            "rms": [float(np.nanmin(rms)), float(np.nanmax(rms))],
            "well": [float(np.nanmin(well)), float(np.nanmax(well))],
            "high": [float(np.nanmin(high)), float(np.nanmax(high))],
        },
    }
    (TILE_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def tile_href(name: str) -> str:
    path = TILE_DIR / f"{name}.png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def text(x, y, content, size=14, weight="400", fill=INK, anchor="middle", style="") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-family="{FONT}" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}" {style}>{escape(content)}</text>\n'
    )


def panel(x, y, w, h, title) -> str:
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{PANEL_FILL}" '
        f'stroke="{PANEL}" stroke-width="1.2"/>\n'
        + text(x + 26, y + 38, title, size=21, weight="500", anchor="start")
    )


def tag(x, y, w, label, color) -> str:
    return (
        f'<rect x="{x - w/2:.1f}" y="{y - 13:.1f}" width="{w:.1f}" height="26" rx="5" '
        f'fill="{color}" opacity="0.09"/>\n'
        + text(x, y + 5, label, size=14, weight="700", fill=color)
    )


def axes(x, y, w, h, *, x_max: int, y_max: int, y_label: str) -> str:
    return (
        f'<line x1="{x:.1f}" y1="{y+h:.1f}" x2="{x+w:.1f}" y2="{y+h:.1f}" stroke="#334155" stroke-width="0.75"/>\n'
        f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{x:.1f}" y2="{y+h:.1f}" stroke="#334155" stroke-width="0.75"/>\n'
        + text(x - 6, y + 5, "0", size=9, fill=INK, anchor="end")
        + text(x - 6, y + h + 4, str(y_max), size=9, fill=INK, anchor="end")
        + text(x + 1, y + h + 15, "0", size=9, fill=INK)
        + text(x + w, y + h + 15, str(x_max), size=9, fill=INK)
        + text(x + w / 2, y + h + 30, "lateral index", size=9, fill=INK)
        + f'<text x="{x - 20:.1f}" y="{y + h/2:.1f}" text-anchor="middle" transform="rotate(-90 {x - 20:.1f} {y + h/2:.1f})" '
        f'font-family="{FONT}" font-size="9" fill="{INK}">{escape(y_label)}</text>\n'
    )


def image_tag(name: str, x: float, y: float, w: float, h: float) -> str:
    href = tile_href(name)
    return (
        f'<image href="{href}" xlink:href="{href}" x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" preserveAspectRatio="none"/>\n'
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" fill="none" stroke="#aeb9c7" stroke-width="1"/>\n'
    )


def tile_block(name, x, y, w, h, title, subtitle, color, *, x_max, y_max, y_label) -> str:
    return (
        axes(x, y, w, h, x_max=x_max, y_max=y_max, y_label=y_label)
        + image_tag(name, x, y, w, h)
        + text(x + w / 2, y + h + 48, title, size=13, weight="700")
        + text(x + w / 2, y + h + 66, subtitle, size=11, fill=color)
    )


def arrow(x1, y1, x2, y2, color=MUTED, width=2.0) -> str:
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}" marker-end="url(#{color.strip("#")}-arrow)"/>\n'


def colorbar(x, y, h, cmap, label, ticks) -> str:
    grad_id = f"grad-{label.replace(' ', '-')}-{x}-{y}"
    if cmap == "jet":
        stops = [(0, "#0000a8"), (0.25, "#00a6ff"), (0.50, "#7dff00"), (0.75, "#ffae00"), (1, "#b00000")]
    else:
        stops = [(0, "#053061"), (0.5, "#f7f7f7"), (1, "#67001f")]
    stop_svg = "".join(f'<stop offset="{int(o*100)}%" stop-color="{c}"/>' for o, c in stops)
    out = [
        f'<defs><linearGradient id="{grad_id}" x1="0" y1="1" x2="0" y2="0">{stop_svg}</linearGradient></defs>',
        f'<rect x="{x}" y="{y}" width="13" height="{h}" fill="url(#{grad_id})" stroke="#334155" stroke-width="0.6"/>',
    ]
    for rel, label_text in ticks:
        yy = y + h * (1 - rel)
        out.append(f'<line x1="{x+13}" y1="{yy:.1f}" x2="{x+18}" y2="{yy:.1f}" stroke="#334155" stroke-width="0.6"/>')
        out.append(text(x + 22, yy + 3, label_text, size=9, anchor="start"))
    out.append(
        f'<text x="{x+52}" y="{y+h/2}" text-anchor="middle" transform="rotate(-90 {x+52} {y+h/2})" '
        f'font-family="{FONT}" font-size="11" fill="{INK}">{escape(label)}</text>'
    )
    return "\n".join(out) + "\n"


def math_text() -> str:
    return (
        '<text x="1175" y="276" text-anchor="middle" font-family="Microsoft YaHei, Noto Sans CJK SC, Arial, sans-serif" font-size="25" fill="#111827">'
        '<tspan font-style="italic">V</tspan><tspan> = P</tspan><tspan baseline-shift="sub" font-size="15">L</tspan><tspan>(V) + P</tspan><tspan baseline-shift="sub" font-size="15">H</tspan><tspan>(V)</tspan>'
        '</text>\n'
        + text(1175, 310, "=  B  +  S", size=24, weight="500")
    )


def main() -> int:
    meta = generate_tiles()
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        "<defs>",
        f'<marker id="{MUTED.strip("#")}-arrow" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto"><path d="M0,0 L10,4 L0,8 Z" fill="{MUTED}"/></marker>',
        f'<marker id="{BACK.strip("#")}-arrow" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto"><path d="M0,0 L10,4 L0,8 Z" fill="{BACK}"/></marker>',
        f'<marker id="{STRUCT.strip("#")}-arrow" markerWidth="10" markerHeight="8" refX="9" refY="4" orient="auto"><path d="M0,0 L10,4 L0,8 Z" fill="{STRUCT}"/></marker>',
        "</defs>",
        '<rect width="100%" height="100%" fill="white"/>',
        panel(30, 35, 700, 470, "(a) Heterogeneous evidence"),
        panel(870, 35, 700, 470, "(b) Physical decomposition"),
        tag(346, 88, 165, "structural cues", STRUCT),
        tile_block("pstm", 150, 105, 92, 170, "PoSTM", "full 1000 samples", STRUCT, x_max=69, y_max=999, y_label="time sample"),
        tile_block("horizon", 405, 130, 122, 122, "Horizon", "interfaces", STRUCT, x_max=69, y_max=69, y_label="depth index"),
        tag(346, 305, 155, "numerical cues", BACK),
        tile_block("rms", 150, 322, 92, 170, "RMS velocity", "full 1000 samples", BACK, x_max=69, y_max=999, y_label="time sample"),
        tile_block("well", 405, 345, 122, 122, "Well log", "scale", BACK, x_max=69, y_max=69, y_label="depth index"),
        colorbar(635, 345, 122, "jet", "velocity (m/s)", [(0, "1600"), (0.5, "3000"), (1, "4400")]),
        arrow(750, 270, 850, 270, MUTED, 1.8),
        text(800, 252, "motivation", size=13, fill=MUTED),
        tile_block("velocity", 1110, 70, 128, 128, "Target V", "full field", MUTED, x_max=69, y_max=69, y_label="depth index"),
        tile_block("background", 980, 322, 128, 128, "B = P_L(V)", "low-pass", BACK, x_max=69, y_max=69, y_label="depth index"),
        tile_block("structure", 1245, 322, 128, 128, "S = P_H(V)", "high-pass", STRUCT, x_max=69, y_max=69, y_label="depth index"),
        colorbar(1445, 70, 128, "jet", "velocity (m/s)", [(0, "1600"), (0.5, "3000"), (1, "4400")]),
        colorbar(1445, 322, 128, "residual (m/s)", "residual (m/s)", [(0, "-800"), (0.5, "0"), (1, "800")]),
        math_text(),
        arrow(1155, 220, 1045, 302, BACK, 2.2),
        arrow(1228, 220, 1310, 302, STRUCT, 2.2),
        f'<!-- metadata: {escape(json.dumps(meta))} -->',
        "</svg>\n",
    ]
    OUT.write_text("\n".join(svg), encoding="utf-8")
    print(f"Wrote {OUT}")
    print(f"Wrote {TILE_DIR / 'metadata.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
