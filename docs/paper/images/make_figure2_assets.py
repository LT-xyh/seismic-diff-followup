"""Generate editable Figure 2 assets for the PD-BG-RFM method overview.

The output is a drawing package rather than a final camera-ready figure.  Each
concept panel is exported as SVG for manual editing in Visio/Inkscape, with PNG
previews rendered by rsvg-convert when available.
"""

from __future__ import annotations

import base64
import json
import shutil
import subprocess
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


HERE = Path(__file__).resolve().parent
FIG1_TILES = HERE / "figure1_drawing_tiles"
OUT = HERE / "figure2_assets"
PANELS = OUT / "concept_panels"
DATA_TILES = OUT / "data_tiles"
COLORBARS = OUT / "colorbars"
PREVIEWS = OUT / "previews"
DOCS = OUT / "docs"
FEATURE_MAPS = OUT / "feature_maps"
TRANSPORT_BURDEN = OUT / "transport_burden"

FONT = "'Microsoft YaHei', 'Noto Sans CJK SC', Arial, sans-serif"
TEXT = "#111827"
MUTED = "#64748B"
BORDER = "#CBD5E1"
GREEN = "#16A34A"
GREEN_LIGHT = "#F0FDF4"
GREEN_BORDER = "#BBF7D0"
BLUE = "#2563EB"
BLUE_LIGHT = "#EFF6FF"
BLUE_BORDER = "#BFDBFE"
ORANGE = "#F59E0B"
ORANGE_LIGHT = "#FFF7ED"
RED = "#DC2626"
RED_LIGHT = "#FEF2F2"
RED_BORDER = "#FECACA"
GRAY_LIGHT = "#F8FAFC"
RESIDUAL_ABS = 800.0


def svg_header(width: int, height: int) -> str:
    return f"""<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<defs>
  <marker id="arrow" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
    <path d="M0,0 L0,6 L9,3 z" fill="{MUTED}" />
  </marker>
  <marker id="arrow-blue" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
    <path d="M0,0 L0,6 L9,3 z" fill="{BLUE}" />
  </marker>
  <marker id="arrow-green" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
    <path d="M0,0 L0,6 L9,3 z" fill="{GREEN}" />
  </marker>
  <marker id="arrow-orange" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
    <path d="M0,0 L0,6 L9,3 z" fill="{ORANGE}" />
  </marker>
  <marker id="arrow-red" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto" markerUnits="strokeWidth">
    <path d="M0,0 L0,6 L9,3 z" fill="{RED}" />
  </marker>
  <style>
    .title {{ font: 700 18px {FONT}; fill: {TEXT}; }}
    .subtitle {{ font: 500 12px {FONT}; fill: {MUTED}; }}
    .label {{ font: 600 13px {FONT}; fill: {TEXT}; }}
    .small {{ font: 500 11px {FONT}; fill: {MUTED}; }}
    .formula {{ font: 600 13px {FONT}; fill: {TEXT}; }}
    .tiny {{ font: 500 9px {FONT}; fill: {MUTED}; }}
  </style>
</defs>
"""


def write_svg(path: Path, width: int, height: int, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg_header(width, height) + body + "\n</svg>\n", encoding="utf-8")


def rect(x: int, y: int, w: int, h: int, fill: str, stroke: str = BORDER, rx: int = 8, extra: str = "") -> str:
    return f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="1.2" {extra}/>'


def line(x1: int, y1: int, x2: int, y2: int, color: str = MUTED, marker: str = "arrow", extra: str = "") -> str:
    width_attr = "" if "stroke-width" in extra else ' stroke-width="1.8"'
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}"{width_attr} marker-end="url(#{marker})" {extra}/>'


def text(x: int, y: int, content: str, cls: str = "label", anchor: str = "start", color: str | None = None) -> str:
    style = f' fill="{color}"' if color else ""
    return f'<text x="{x}" y="{y}" class="{cls}" text-anchor="{anchor}"{style}>{content}</text>'


def embedded_png_href(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def image_tag(href: str, x: int, y: int, width: int, height: int) -> str:
    return (
        f'<image href="{href}" xlink:href="{href}" x="{x}" y="{y}" '
        f'width="{width}" height="{height}" preserveAspectRatio="xMidYMid meet"/>'
    )


def circle(cx: int, cy: int, r: int, fill: str, stroke: str, cls: str = "") -> str:
    return f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="1.1" class="{cls}"/>'


def save_residual_style_tile(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(2.0, 2.0), dpi=220)
    ax.imshow(
        array,
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-RESIDUAL_ABS, vcenter=0.0, vmax=RESIDUAL_ABS),
        interpolation="nearest",
    )
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def make_transport_field_tiles() -> None:
    structure_path = DATA_TILES / "structure_1to1.png"
    if not structure_path.exists():
        return
    # Use the existing residual tile only for visual structure.  Convert it to a
    # signed proxy so the intermediate panel has coherent spatial texture rather
    # than arbitrary embedding points.
    img = np.asarray(Image.open(structure_path).convert("RGB"), dtype=np.float32) / 255.0
    red_blue_proxy = (img[..., 0] - img[..., 2]) * RESIDUAL_ABS
    rng = np.random.default_rng(20270709)
    raw_noise = rng.normal(0.0, 1.0, size=red_blue_proxy.shape).astype(np.float32)
    noise = raw_noise * (RESIDUAL_ABS * 0.42)
    intermediate = 0.45 * noise + 0.55 * red_blue_proxy

    save_residual_style_tile(noise, DATA_TILES / "noise_field.png")
    save_residual_style_tile(intermediate, DATA_TILES / "interpolated_residual_field.png")


def make_condition_space_cluster() -> None:
    body = f"""
<rect x="0" y="0" width="620" height="390" fill="white"/>
{text(24, 36, "Evidence-to-Contract Mapping", "title")}
{text(24, 56, "Use actual data views to show why C_B and C_S carry different roles", "subtitle")}
{rect(34, 82, 250, 224, GREEN_LIGHT, GREEN_BORDER)}
{text(159, 112, "Background interface C_B", "label", "middle", GREEN)}
<image href="../data_tiles/rms_1to1.png" x="58" y="134" width="82" height="82" preserveAspectRatio="xMidYMid meet"/>
<image href="../data_tiles/well_1to1.png" x="178" y="134" width="82" height="82" preserveAspectRatio="xMidYMid meet"/>
<image href="../data_tiles/background_1to1.png" x="118" y="218" width="82" height="82" preserveAspectRatio="xMidYMid meet"/>
{rect(58, 134, 82, 82, "none", BORDER, 3)}
{rect(178, 134, 82, 82, "none", BORDER, 3)}
{rect(118, 218, 82, 82, "none", GREEN_BORDER, 3)}
{text(99, 230, "RMS", "tiny", "middle")}
{text(219, 230, "Well", "tiny", "middle")}
{text(159, 316, "smooth numerical trend", "small", "middle")}
{rect(336, 82, 250, 224, BLUE_LIGHT, BLUE_BORDER)}
{text(461, 112, "Structural interface C_S", "label", "middle", BLUE)}
<image href="../data_tiles/pstm_1to1.png" x="360" y="134" width="82" height="82" preserveAspectRatio="xMidYMid meet"/>
<image href="../data_tiles/horizon_1to1.png" x="480" y="134" width="82" height="82" preserveAspectRatio="xMidYMid meet"/>
<image href="../data_tiles/structure_1to1.png" x="420" y="218" width="82" height="82" preserveAspectRatio="xMidYMid meet"/>
{rect(360, 134, 82, 82, "none", BORDER, 3)}
{rect(480, 134, 82, 82, "none", BORDER, 3)}
{rect(420, 218, 82, 82, "none", BLUE_BORDER, 3)}
{text(401, 230, "PoSTM", "tiny", "middle")}
{text(521, 230, "Horizon", "tiny", "middle")}
{text(461, 316, "localized structural residual", "small", "middle")}
{line(284, 188, 336, 188, ORANGE, "arrow-orange", 'stroke-dasharray="6 4"')}
{text(310, 176, "contract", "tiny", "middle", ORANGE)}
{text(310, 216, "L_symile + L_A + L_O", "tiny", "middle", ORANGE)}
{text(40, 354, "No t-SNE/UMAP: this panel uses real examples and target-derived training views.", "tiny")}
"""
    write_svg(PANELS / "condition_space_cluster.svg", 620, 390, body)


def make_anchor_calibration() -> None:
    body = f"""
<rect x="0" y="0" width="620" height="330" fill="white"/>
{text(24, 36, "Training-Only Anchor Calibration", "title")}
{text(24, 56, "Anchors are derived from V during training and never used at inference", "subtitle")}
{rect(42, 96, 156, 108, GREEN_LIGHT, GREEN_BORDER)}
{text(120, 128, "A_B = W_L(V)", "formula", "middle", GREEN)}
{text(120, 154, "low-frequency", "small", "middle")}
{text(120, 176, "background anchor", "small", "middle")}
{rect(42, 226, 156, 74, ORANGE_LIGHT, ORANGE, 8, 'stroke-dasharray="7 5"')}
{text(120, 256, "training only", "label", "middle", ORANGE)}
{text(120, 278, "not in inference", "small", "middle")}
{rect(420, 96, 156, 108, BLUE_LIGHT, BLUE_BORDER)}
{text(498, 128, "A_S = W_H(V)", "formula", "middle", BLUE)}
{text(498, 154, "high-frequency", "small", "middle")}
{text(498, 176, "structure anchor", "small", "middle")}
{rect(252, 102, 118, 78, "white", BORDER)}
{text(311, 132, "C_B", "formula", "middle", GREEN)}
{text(311, 156, "C_S", "formula", "middle", BLUE)}
{line(198, 148, 252, 140, ORANGE, "arrow-orange", 'stroke-dasharray="7 5"')}
{line(420, 148, 370, 152, ORANGE, "arrow-orange", 'stroke-dasharray="7 5"')}
{text(310, 220, "L_A calibrates physical semantics", "label", "middle")}
{text(310, 246, "L_symile preserves cross-modal consistency", "small", "middle")}
{text(310, 270, "L_rel handles missing or unreliable modalities", "small", "middle")}
"""
    write_svg(PANELS / "anchor_calibration.svg", 620, 330, body)


def make_residual_transport() -> None:
    noise_href = embedded_png_href(DATA_TILES / "noise_field.png")
    interp_href = embedded_png_href(DATA_TILES / "interpolated_residual_field.png")
    residual_href = embedded_png_href(DATA_TILES / "structure_1to1.png")
    body = f"""
<rect x="0" y="0" width="700" height="360" fill="white"/>
{text(24, 36, "Residual-Space Flow Transport", "title")}
{text(24, 56, "A field-to-field transport view, not a point-cloud embedding", "subtitle")}
{rect(42, 96, 144, 144, "white", BORDER, 4)}
{image_tag(noise_href, 42, 96, 144, 144)}
{rect(42, 96, 144, 144, "none", BORDER, 4)}
{text(114, 266, "xi ~ N(0,I)", "formula", "middle")}
{text(114, 288, "noise field", "small", "middle")}
{line(200, 168, 268, 168, BLUE, "arrow-blue")}
{rect(282, 96, 144, 144, BLUE_LIGHT, BLUE_BORDER, 4)}
{image_tag(interp_href, 282, 96, 144, 144)}
{rect(282, 96, 144, 144, "none", BLUE_BORDER, 4)}
{text(354, 266, "R_t", "formula", "middle", BLUE)}
{text(354, 288, "(1-t)xi + tR_hatB", "small", "middle")}
{line(440, 168, 508, 168, BLUE, "arrow-blue")}
{rect(522, 96, 144, 144, BLUE_LIGHT, BLUE_BORDER, 4)}
{image_tag(residual_href, 522, 96, 144, 144)}
{rect(522, 96, 144, 144, "none", BLUE_BORDER, 4)}
{text(594, 266, "hat R", "formula", "middle", BLUE)}
{text(594, 288, "structured residual", "small", "middle")}
{text(354, 326, "target vector: u_t = R_hatB - xi", "formula", "middle")}
{text(354, 348, "conditioned by C_S and psi_B(hat B)", "small", "middle")}
"""
    write_svg(PANELS / "residual_transport.svg", 700, 360, body)


def make_transport_burden_assets() -> None:
    TRANSPORT_BURDEN.mkdir(parents=True, exist_ok=True)
    noise_href = embedded_png_href(DATA_TILES / "noise_field.png")
    interp_href = embedded_png_href(DATA_TILES / "interpolated_residual_field.png")
    residual_href = embedded_png_href(DATA_TILES / "structure_1to1.png")
    velocity_href = embedded_png_href(DATA_TILES / "velocity_1to1.png")

    three_stage = f"""
<rect x="0" y="0" width="900" height="360" fill="white"/>
{text(24, 36, "Residual-space Flow Matching", "title")}
{text(24, 58, "Three-stage transport sketch inside the residual generator", "subtitle")}
{rect(34, 86, 178, 178, "white", BORDER, 5)}
{image_tag(noise_href, 34, 86, 178, 178)}
{rect(34, 86, 178, 178, "none", BORDER, 5)}
{text(123, 290, "xi ~ N(0,I)", "formula", "middle")}
{text(123, 312, "source noise field", "small", "middle")}
{line(232, 174, 314, 174, BLUE, "arrow-blue")}
{rect(330, 86, 178, 178, BLUE_LIGHT, BLUE_BORDER, 5)}
{image_tag(interp_href, 330, 86, 178, 178)}
{rect(330, 86, 178, 178, "none", BLUE_BORDER, 5)}
{text(419, 290, "R_t = (1-t)xi + tR_hatB", "formula", "middle", BLUE)}
{text(419, 312, "linear path in residual space", "small", "middle")}
{line(528, 174, 610, 174, BLUE, "arrow-blue")}
{rect(626, 86, 178, 178, BLUE_LIGHT, BLUE_BORDER, 5)}
{image_tag(residual_href, 626, 86, 178, 178)}
{rect(626, 86, 178, 178, "none", BLUE_BORDER, 5)}
{text(715, 290, "hat R", "formula", "middle", BLUE)}
{text(715, 312, "generated structured residual", "small", "middle")}
{rect(312, 28, 276, 38, BLUE_LIGHT, BLUE_BORDER, 18)}
{text(450, 52, "condition: C_S + psi_B(hat B)", "formula", "middle", BLUE)}
{text(450, 342, "FM target vector: u_R = R_hatB - xi", "formula", "middle")}
"""
    write_svg(TRANSPORT_BURDEN / "residual_transport_3stage.svg", 900, 360, three_stage)

    burden = f"""
<rect x="0" y="0" width="980" height="430" fill="white"/>
{text(24, 36, "Why Transport in Residual Space?", "title")}
{text(24, 58, "The generator transports to the background-unexplained component instead of the full field.", "subtitle")}
{rect(46, 92, 888, 128, RED_LIGHT, RED_BORDER, 8)}
{text(84, 128, "Full-field FM", "label", color=RED)}
{rect(244, 116, 76, 76, "white", BORDER, 4)}
{image_tag(noise_href, 244, 116, 76, 76)}
{text(282, 208, "xi", "small", "middle")}
{line(344, 154, 650, 154, RED, "arrow-red", 'stroke-width="4.0"')}
{text(497, 140, "long target vector", "small", "middle", RED)}
{rect(676, 116, 76, 76, "white", BORDER, 4)}
{image_tag(velocity_href, 676, 116, 76, 76)}
{text(714, 208, "V", "small", "middle")}
{text(816, 150, "u_V = V - xi", "formula", "middle", RED)}
{text(816, 178, "full background + structure", "small", "middle")}
{rect(46, 244, 888, 128, BLUE_LIGHT, BLUE_BORDER, 8)}
{text(84, 280, "Residual FM", "label", color=BLUE)}
{rect(244, 268, 76, 76, "white", BORDER, 4)}
{image_tag(noise_href, 244, 268, 76, 76)}
{text(282, 360, "xi", "small", "middle")}
{line(344, 306, 540, 306, BLUE, "arrow-blue", 'stroke-width="2.4"')}
{text(442, 292, "smaller target vector", "small", "middle", BLUE)}
{rect(566, 268, 76, 76, "white", BLUE_BORDER, 4)}
{image_tag(residual_href, 566, 268, 76, 76)}
{text(604, 360, "R_hatB", "small", "middle")}
{text(742, 302, "u_R = R_hatB - xi", "formula", "middle", BLUE)}
{text(742, 330, "background-unexplained residual", "small", "middle")}
{rect(122, 384, 736, 32, "white", BORDER, 16)}
{text(490, 406, "E||u_R||^2 &lt; E||u_V||^2  if  E||V-hat B||^2 &lt; E||V||^2", "formula", "middle")}
"""
    write_svg(TRANSPORT_BURDEN / "full_vs_residual_burden.svg", 980, 430, burden)

    readme = """# Residual Transport Burden Assets

These panels are reusable materials for the Residual FM part of Fig. 2.

- `residual_transport_3stage.svg/png`: three-stage residual-space transport
  sketch: source noise, interpolated residual state, generated residual.
- `full_vs_residual_burden.svg/png`: theoretical comparison between full-field
  FM and residual FM. Use it as a small explanatory inset, not as a main
  architecture block.

Recommended wording in the final figure:

```text
Residual-space transport reduces the target-vector norm under the stated
condition, rather than guaranteeing better performance unconditionally.
```
"""
    (TRANSPORT_BURDEN / "README.md").write_text(readme, encoding="utf-8")


def make_background_residual_composition() -> None:
    body = f"""
<rect x="0" y="0" width="820" height="300" fill="white"/>
{text(24, 36, "Background-Guided Composition", "title")}
{text(24, 56, "Deterministic background plus generated structured residual", "subtitle")}
<image href="../data_tiles/background_1to1.png" x="54" y="92" width="142" height="142" preserveAspectRatio="xMidYMid meet"/>
{rect(54, 92, 142, 142, "none", GREEN_BORDER, 4)}
{text(125, 258, "hat B = f_B(C_B)", "formula", "middle", GREEN)}
{text(226, 176, "+", "title", "middle")}
<image href="../data_tiles/structure_1to1.png" x="258" y="92" width="142" height="142" preserveAspectRatio="xMidYMid meet"/>
{rect(258, 92, 142, 142, "none", BLUE_BORDER, 4)}
{text(329, 258, "hat R", "formula", "middle", BLUE)}
{text(430, 176, "=", "title", "middle")}
<image href="../data_tiles/velocity_1to1.png" x="462" y="92" width="142" height="142" preserveAspectRatio="xMidYMid meet"/>
{rect(462, 92, 142, 142, "none", BORDER, 4)}
{text(533, 258, "hat V = hat B + hat R", "formula", "middle")}
{rect(646, 100, 138, 116, ORANGE_LIGHT, ORANGE, 8, 'stroke-dasharray="7 5"')}
{text(715, 132, "Training target", "label", "middle", ORANGE)}
{text(715, 158, "R_hatB = V - hat B", "formula", "middle")}
{text(715, 184, "not an oracle", "small", "middle")}
{text(715, 204, "low-pass residual", "small", "middle")}
"""
    write_svg(PANELS / "background_residual_composition.svg", 820, 300, body)


def make_training_only_box() -> None:
    body = f"""
<rect x="0" y="0" width="620" height="250" fill="white"/>
{rect(28, 32, 564, 184, ORANGE_LIGHT, ORANGE, 12, 'stroke-dasharray="8 5"')}
{text(50, 68, "Training-only information", "title", color=ORANGE)}
{text(50, 96, "Used to define supervision and contract losses; unavailable at inference.", "subtitle")}
{text(70, 132, "Anchors: A_B = W_L(V),  A_S = W_H(V)", "formula")}
{text(70, 162, "Contract: L_C = L_symile + L_A + L_O + L_freq + L_rel + L_pair", "formula")}
{text(70, 192, "Residual target: R_hatB = V - hat B", "formula")}
{text(50, 232, "Suggested placement: side band or dashed overlay, never on the inference path.", "tiny")}
"""
    write_svg(PANELS / "training_only_box.svg", 620, 250, body)


def make_diagnostic_hooks() -> None:
    body = f"""
<rect x="0" y="0" width="640" height="210" fill="white"/>
{text(24, 36, "Diagnostic Hooks", "title")}
{text(24, 56, "These metrics verify whether the claimed decomposition is followed", "subtitle")}
{rect(48, 92, 158, 76, GREEN_LIGHT, GREEN_BORDER)}
{text(127, 122, "eta_B", "formula", "middle", GREEN)}
{text(127, 146, "background role", "small", "middle")}
{rect(242, 92, 158, 76, BLUE_LIGHT, BLUE_BORDER)}
{text(321, 122, "eta_S", "formula", "middle", BLUE)}
{text(321, 146, "structural role", "small", "middle")}
{rect(436, 92, 158, 76, GRAY_LIGHT, BORDER)}
{text(515, 122, "rho_B", "formula", "middle")}
{text(515, 146, "residual leakage", "small", "middle")}
{line(206, 130, 242, 130)}
{line(400, 130, 436, 130)}
{text(322, 194, "Use as a small footer: design -> measurable checks", "tiny", "middle")}
"""
    write_svg(PANELS / "diagnostic_hooks.svg", 640, 210, body)


def make_figure2_draft() -> None:
    body = f"""
<rect x="0" y="0" width="1500" height="760" fill="white"/>
{text(34, 42, "PD-BG-RFM: physics-decoupled residual transport", "title")}
{text(34, 66, "Solid arrows: inference computation. Dashed orange arrows: training-only supervision.", "subtitle")}
{rect(34, 112, 260, 500, GRAY_LIGHT, BORDER)}
{text(164, 144, "Multimodal evidence", "label", "middle")}
<image href="data_tiles/pstm_1to1.png" x="60" y="172" width="92" height="92"/>
<image href="data_tiles/horizon_1to1.png" x="174" y="172" width="92" height="92"/>
<image href="data_tiles/rms_1to1.png" x="60" y="310" width="92" height="92"/>
<image href="data_tiles/well_1to1.png" x="174" y="310" width="92" height="92"/>
{text(106, 282, "PoSTM", "tiny", "middle")}
{text(220, 282, "Horizon", "tiny", "middle")}
{text(106, 420, "RMS", "tiny", "middle")}
{text(220, 420, "Well", "tiny", "middle")}
{rect(78, 472, 172, 70, "white", BORDER)}
{text(164, 502, "modality encoder", "label", "middle")}
{text(164, 526, "1x1 role heads", "small", "middle")}
{line(294, 362, 370, 362)}
{rect(370, 112, 360, 500, "white", BORDER)}
{text(550, 144, "Physics-decoupled contract", "label", "middle")}
<image href="concept_panels/condition_space_cluster.svg" x="396" y="170" width="308" height="194"/>
{rect(408, 394, 282, 92, ORANGE_LIGHT, ORANGE, 8, 'stroke-dasharray="7 5"')}
{text(549, 426, "A_B=W_L(V), A_S=W_H(V)", "formula", "middle")}
{text(549, 454, "contract losses are training only", "small", "middle")}
{line(730, 362, 806, 362)}
{rect(806, 112, 280, 500, "white", BORDER)}
{text(946, 144, "Background responsibility", "label", "middle")}
{rect(854, 190, 184, 72, GREEN_LIGHT, GREEN_BORDER)}
{text(946, 222, "C_B -> f_B -> hat B", "formula", "middle", GREEN)}
{text(946, 246, "deterministic branch", "small", "middle")}
<image href="data_tiles/background_1to1.png" x="876" y="300" width="140" height="140"/>
{text(946, 466, "hat B", "formula", "middle", GREEN)}
{line(1086, 362, 1162, 362)}
{rect(1162, 112, 304, 500, "white", BORDER)}
{text(1314, 144, "Residual-space transport", "label", "middle")}
{rect(1192, 184, 244, 74, BLUE_LIGHT, BLUE_BORDER)}
{text(1314, 216, "C_S + psi_B(hat B)", "formula", "middle", BLUE)}
{text(1314, 240, "conditions residual FM", "small", "middle")}
<image href="concept_panels/residual_transport.svg" x="1190" y="286" width="246" height="126"/>
{text(1314, 454, "hat V = hat B + hat R", "formula", "middle")}
{rect(56, 650, 1388, 70, "white", BORDER)}
{text(96, 682, "Diagnostics:", "label")}
{text(212, 682, "eta_B, eta_S check condition roles; rho_B checks low-frequency residual leakage.", "small")}
{text(212, 706, "These are evaluation diagnostics, not extra inference inputs.", "small")}
"""
    write_svg(OUT / "figure2_draft.svg", 1500, 760, body)


def copy_base_assets() -> None:
    DATA_TILES.mkdir(parents=True, exist_ok=True)
    COLORBARS.mkdir(parents=True, exist_ok=True)
    for name in [
        "pstm_1to1.png",
        "horizon_1to1.png",
        "rms_1to1.png",
        "well_1to1.png",
        "velocity_1to1.png",
        "background_1to1.png",
        "structure_1to1.png",
        "metadata.json",
    ]:
        src = FIG1_TILES / name
        if src.exists():
            shutil.copy2(src, DATA_TILES / name)
    for name in ["colorbar_velocity.png", "colorbar_structure.png"]:
        src = FIG1_TILES / name
        if src.exists():
            shutil.copy2(src, COLORBARS / name)


def render_previews() -> list[str]:
    PREVIEWS.mkdir(parents=True, exist_ok=True)
    converter = shutil.which("rsvg-convert")
    rendered: list[str] = []
    if converter is None:
        return rendered
    svgs = sorted(PANELS.glob("*.svg")) + [OUT / "figure2_draft.svg"]
    for svg in svgs:
        out = PREVIEWS / f"{svg.stem}.png"
        subprocess.run([converter, "-o", str(out), str(svg)], check=True)
        rendered.append(str(out.relative_to(OUT)))
    for svg in sorted(TRANSPORT_BURDEN.glob("*.svg")):
        out = TRANSPORT_BURDEN / f"{svg.stem}.png"
        subprocess.run([converter, "-o", str(out), str(svg)], check=True)
        rendered.append(str(out.relative_to(OUT)))
    return rendered


def write_docs(rendered: list[str]) -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    manifest = {
        "purpose": "Editable Fig. 2 asset package for PD-BG-RFM method overview.",
        "semantic_colors": {
            "background_condition": GREEN,
            "structural_condition": BLUE,
            "training_only": ORANGE,
            "neutral_border": BORDER,
        },
        "solid_arrows": "Inference computation.",
        "dashed_orange_arrows": "Training-only supervision or target construction.",
        "concept_panels": [p.name for p in sorted(PANELS.glob("*.svg"))],
        "data_tiles": [p.name for p in sorted(DATA_TILES.glob("*"))],
        "feature_maps": [p.name for p in sorted(FEATURE_MAPS.glob("*"))] if FEATURE_MAPS.exists() else [],
        "colorbars": [p.name for p in sorted(COLORBARS.glob("*"))],
        "transport_burden": [p.name for p in sorted(TRANSPORT_BURDEN.glob("*"))],
        "previews": rendered,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    readme = """# Figure 2 Asset Package

This directory contains editable assets for the PD-BG-RFM method overview figure.
The goal is to support a top-conference style method-as-argument figure rather
than a literal layer-by-layer architecture diagram.

## Recommended Figure 2 Story

Use the final figure to show this chain:

```text
Multimodal evidence
-> Physics-decoupled condition contract
-> Background responsibility assignment
-> Residual-space Flow Matching
-> Velocity reconstruction
```

The figure should distinguish inference computation from training-only
supervision:

- Solid arrows: inference path.
- Dashed orange arrows/boxes: training-only anchors, losses, and residual target
  construction.
- Never put `V`, `A_B`, `A_S`, or `R_{hat B}=V-hat B` on the inference path.

## Asset Map

- `concept_panels/condition_space_cluster.svg`: actual-data contract panel
  showing how RMS/well evidence supports `C_B` and PoSTM/horizon evidence
  supports `C_S`. Despite the filename, this is not a t-SNE/UMAP plot.
- `concept_panels/anchor_calibration.svg`: training-only frequency-anchor
  calibration (`A_B=W_L(V)`, `A_S=W_H(V)`).
- `concept_panels/residual_transport.svg`: residual Flow Matching principle
  (`R_t=(1-t)xi+tR_{hat B}`).
- `concept_panels/background_residual_composition.svg`: background/residual
  composition (`hat V=hat B+hat R`) using reusable tiles.
- `concept_panels/training_only_box.svg`: compact dashed side box for
  training-only information.
- `concept_panels/diagnostic_hooks.svg`: small footer for `eta_B`, `eta_S`,
  and `rho_B`.
- `transport_burden/residual_transport_3stage.svg`: compact three-stage sketch
  for the Residual FM module (`xi -> R_t -> hat R`).
- `transport_burden/full_vs_residual_burden.svg`: theoretical inset comparing
  full-field FM and residual FM target-vector norms under the stated condition.
- `data_tiles/`: copied Fig. 1 tiles for PoSTM, horizon, RMS, well, velocity,
  background, and structure.
- `feature_maps/`: real `C_B` and `C_S` feature-map visualizations exported
  from the trained encoder checkpoint by `../make_cb_cs_feature_maps.py`.
  Use `cb_cs_feature_panel.png` for a compact paper panel, or use the
  individual aggregate/channel tiles for manual Visio layout.
- `colorbars/`: copied velocity and structure colorbars.
- `figure2_draft.svg`: rough full-layout starting point; use this as a
  composition guide, not as the final figure.
- `previews/`: PNG renderings for quick inspection.

## Visio Layout Prompt

Create a wide two-column AAAI method overview figure titled:
`PD-BG-RFM: Physics-Decoupled Background-Guided Residual Flow Matching`.

Use four left-to-right stages:

1. Multimodal evidence: PoSTM, horizon, RMS velocity, well log.
2. Physics-decoupled condition contract: show `C_B` in green and `C_S` in blue,
   using the actual-data contract panel. `C_B` should be visually tied to
   RMS/well/background, while `C_S` should be tied to PoSTM/horizon/structure.
   Add dashed orange arrows from training-only anchors `A_B=W_L(V)` and
   `A_S=W_H(V)`.
   If space permits, replace or augment this stage with
   `feature_maps/cb_cs_feature_panel.png` to show the actual learned
   `C_B/C_S` feature maps.
3. Background responsibility: show `C_B -> f_B -> hat B` as a deterministic
   green branch.
4. Residual transport and reconstruction: show `C_S + psi_B(hat B)` guiding
   residual Flow Matching, then `hat V = hat B + hat R`.

Add a small diagnostic footer:
`eta_B, eta_S -> condition-role checks; rho_B -> residual leakage check`.

Keep U-Net internals, channel counts, optimizer details, and full loss
expansions out of the main figure.
"""
    (OUT / "README.md").write_text(readme, encoding="utf-8")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    PANELS.mkdir(parents=True, exist_ok=True)
    PREVIEWS.mkdir(parents=True, exist_ok=True)
    TRANSPORT_BURDEN.mkdir(parents=True, exist_ok=True)
    copy_base_assets()
    make_transport_field_tiles()
    make_condition_space_cluster()
    make_anchor_calibration()
    make_residual_transport()
    make_transport_burden_assets()
    make_background_residual_composition()
    make_training_only_box()
    make_diagnostic_hooks()
    make_figure2_draft()
    rendered = render_previews()
    write_docs(rendered)
    print(f"Wrote Figure 2 assets to {OUT}")
    print(f"Rendered {len(rendered)} PNG previews")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
