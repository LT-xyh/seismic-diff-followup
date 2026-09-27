"""Post-hoc registration diagnostics for rendered residual-flow assets.

The PD-BG-RFM implementation is residual Flow Matching, not spatial warping.
This module therefore treats any displacement field as an image-registration
diagnostic estimated after the fact.  The default inputs are the independent
rendered proxy assets used by the Figure 2 transport sketch; they are not
checkpoint states and cannot support a model-internal RFW claim.

Example
-------
conda run -n seg python -m bg_pdr_fm.evaluation.visualize_posthoc_rfw
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_DIR = REPO_ROOT / "docs" / "paper" / "images" / "figure2_assets" / "transport_burden" / "subfigures"
DEFAULT_NOISE = DEFAULT_SOURCE_DIR / "stage1_noise_xi_900.png"
DEFAULT_INTERMEDIATE = DEFAULT_SOURCE_DIR / "stage2_interpolated_Rt_900.png"
DEFAULT_TARGET = DEFAULT_SOURCE_DIR / "stage3_target_residual_Rhat_900.png"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "docs" / "paper" / "AAAI2027" / "figures" / "posthoc_rfw_diagnostic"

SOURCE_KIND = "independent_rendered_raster_proxy"
FOLDING_THRESHOLD = 0.01
DEFAULT_MAX_REGISTRATION_SIZE = 300


def _resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def decode_rendered_field(path: str | Path) -> np.ndarray:
    """Decode the signed scalar proxy encoded by the diverging render.

    The source assets were rendered with ``RdBu_r`` and a common zero-centered
    range.  Red-minus-blue retains the signed spatial structure without
    pretending that the PNG stores the original model tensor.
    """
    from PIL import Image

    resolved = _resolve_path(path)
    with Image.open(resolved) as image:
        rgba = np.asarray(image.convert("RGBA"), dtype=np.float32) / 255.0
    alpha = rgba[..., 3]
    if np.any(alpha <= 0.0):
        raise ValueError(f"Rendered source contains transparent pixels: {resolved}")
    field = rgba[..., 0] - rgba[..., 2]
    field = np.asarray(field, dtype=np.float32)
    if field.ndim != 2 or not np.isfinite(field).all():
        raise ValueError(f"Could not decode a finite 2D scalar field from {resolved}")
    if float(np.ptp(field)) <= 1e-8:
        raise ValueError(f"Rendered source is effectively constant: {resolved}")
    return field


def _resize_field(field: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if tuple(field.shape) == tuple(shape):
        return np.asarray(field, dtype=np.float32)
    from skimage.transform import resize

    return np.asarray(
        resize(field, shape, order=1, mode="reflect", anti_aliasing=True, preserve_range=True),
        dtype=np.float32,
    )


def _inspect_composite_attachment(path: str | Path | None) -> dict[str, Any]:
    """Record the supplied composite attachment without using it for decoding."""
    if path is None:
        return {"provided": False, "used_for_field_extraction": False}

    from PIL import Image

    resolved = _resolve_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Missing composite attachment: {resolved}")
    with Image.open(resolved) as image:
        shape = [int(image.height), int(image.width)]
        mode = image.mode
    return {
        "provided": True,
        "path": str(resolved),
        "sha256": _sha256(resolved),
        "shape": shape,
        "mode": mode,
        "used_for_field_extraction": False,
        "reason": "Independent 900 x 900 subfigure assets were found in the repository, so no composite crop was used.",
    }


def _registration_view(field: np.ndarray, max_size: int) -> tuple[np.ndarray, tuple[float, float]]:
    height, width = field.shape
    largest = max(height, width)
    if largest <= int(max_size):
        return field.astype(np.float32, copy=False), (1.0, 1.0)
    scale = float(max_size) / float(largest)
    shape = (max(16, int(round(height * scale))), max(16, int(round(width * scale))))
    return _resize_field(field, shape), (height / shape[0], width / shape[1])


def estimate_tvl1_displacement(
    moving: np.ndarray,
    reference: np.ndarray,
    *,
    max_size: int = DEFAULT_MAX_REGISTRATION_SIZE,
    attachment: float = 15.0,
    tightness: float = 0.3,
    num_warp: int = 5,
    num_iter: int = 10,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Estimate D for ``warp(moving, x + D(x)) = reference`` using TV-L1."""
    from skimage.registration import optical_flow_tvl1
    from skimage.transform import resize

    if moving.shape != reference.shape:
        raise ValueError(f"moving and reference must have the same shape, got {moving.shape} and {reference.shape}")
    moving_small, scale = _registration_view(moving, max_size)
    reference_small = _resize_field(reference, moving_small.shape)
    flow_small = optical_flow_tvl1(
        reference_small,
        moving_small,
        attachment=float(attachment),
        tightness=float(tightness),
        num_warp=int(num_warp),
        num_iter=int(num_iter),
        prefilter=True,
        dtype=np.float32,
    )
    flow = np.stack(
        [
            resize(flow_small[0], moving.shape, order=1, mode="edge", anti_aliasing=False, preserve_range=True) * scale[0],
            resize(flow_small[1], moving.shape, order=1, mode="edge", anti_aliasing=False, preserve_range=True) * scale[1],
        ],
        axis=0,
    ).astype(np.float32)
    return flow, {
        "method": "skimage.registration.optical_flow_tvl1",
        "reference": "R_Bhat",
        "moving": "R_t",
        "mapping": "G(x)=x+D(x); R_warp(x)=S(R_t,G(x))",
        "registration_max_size": int(max_size),
        "registration_view_shape": [int(v) for v in moving_small.shape],
        "scale_to_full_resolution": [float(v) for v in scale],
        "attachment": float(attachment),
        "tightness": float(tightness),
        "num_warp": int(num_warp),
        "num_iter": int(num_iter),
        "prefilter": True,
    }


def warp_with_displacement(field: np.ndarray, displacement: np.ndarray) -> np.ndarray:
    """Sample ``field`` at ``G(x)=x+D(x)`` using bilinear interpolation."""
    from skimage.transform import warp

    if displacement.shape != (2, *field.shape):
        raise ValueError(f"displacement must have shape (2,H,W), got {displacement.shape} for {field.shape}")
    rows, cols = np.meshgrid(
        np.arange(field.shape[0], dtype=np.float32),
        np.arange(field.shape[1], dtype=np.float32),
        indexing="ij",
    )
    coordinates = np.stack([rows + displacement[0], cols + displacement[1]], axis=0)
    return np.asarray(
        warp(field, coordinates, order=1, mode="edge", preserve_range=True),
        dtype=np.float32,
    )


def jacobian_determinant(displacement: np.ndarray) -> np.ndarray:
    """Return det(J_G) for G(row,col)=(row,col)+D(row,col)."""
    if displacement.ndim != 3 or displacement.shape[0] != 2:
        raise ValueError(f"displacement must have shape (2,H,W), got {displacement.shape}")
    d_row_dr, d_row_dc = np.gradient(displacement[0].astype(np.float64), edge_order=1)
    d_col_dr, d_col_dc = np.gradient(displacement[1].astype(np.float64), edge_order=1)
    determinant = (1.0 + d_row_dr) * (1.0 + d_col_dc) - d_row_dc * d_col_dr
    return np.asarray(determinant, dtype=np.float32)


def _similarity(reference: np.ndarray, candidate: np.ndarray) -> float:
    from skimage.metrics import structural_similarity

    data_range = max(float(np.ptp(reference)), 1e-6)
    return float(structural_similarity(reference, candidate, data_range=data_range))


def _image_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    return {
        "mae": float(np.mean(np.abs(candidate - reference))),
        "rmse": float(np.sqrt(np.mean(np.square(candidate - reference)))),
        "ssim": _similarity(reference, candidate),
    }


def registration_decision(
    *,
    before_mae: float,
    after_mae: float,
    before_ssim: float,
    after_ssim: float,
    folding_fraction: float,
    folding_threshold: float = FOLDING_THRESHOLD,
    source_kind: str = SOURCE_KIND,
) -> dict[str, Any]:
    """Apply the explicit RFW gate and keep model-internal claims separate."""
    errors_improve = bool(after_mae < before_mae)
    structure_improves = bool(after_ssim > before_ssim)
    no_severe_folding = bool(folding_fraction <= float(folding_threshold))
    posthoc_supported = bool(errors_improve and structure_improves and no_severe_folding)
    raw_model_arrays = source_kind == "raw_model_array"
    return {
        "errors_improve": errors_improve,
        "structure_improves": structure_improves,
        "no_severe_folding": no_severe_folding,
        "posthoc_registration_supported": posthoc_supported,
        "model_internal_rfw_supported": bool(posthoc_supported and raw_model_arrays),
        "source_kind": source_kind,
        "folding_threshold": float(folding_threshold),
        "interpretation": (
            "Post-hoc registration is numerically supported for this image pair, but the source is not a raw model state."
            if posthoc_supported and not raw_model_arrays
            else "The image pair does not pass the post-hoc registration gate."
            if not posthoc_supported
            else "The source is a raw model array and passes the post-hoc gate; this still does not establish an internal warp operator."
        ),
    }


def _grid_coordinates(shape: tuple[int, int], step: int) -> tuple[np.ndarray, np.ndarray]:
    rows = np.arange(0, shape[0], max(1, int(step)), dtype=np.float32)
    cols = np.arange(0, shape[1], max(1, int(step)), dtype=np.float32)
    return rows, cols


def _draw_grid(ax: Any, shape: tuple[int, int], *, displacement: np.ndarray | None, step: int, color: str) -> None:
    rows, cols = _grid_coordinates(shape, step)
    if displacement is None:
        for row in rows:
            ax.plot([0, shape[1] - 1], [row, row], color=color, linewidth=0.55, alpha=0.75)
        for col in cols:
            ax.plot([col, col], [0, shape[0] - 1], color=color, linewidth=0.55, alpha=0.75)
        return
    for row in rows:
        col_values = np.arange(shape[1], dtype=np.float32)
        row_values = np.full_like(col_values, row)
        ax.plot(col_values + displacement[1, int(row), :], row_values + displacement[0, int(row), :], color=color, linewidth=0.55, alpha=0.75)
    for col in cols:
        row_values = np.arange(shape[0], dtype=np.float32)
        col_values = np.full_like(row_values, col)
        ax.plot(col_values + displacement[1, :, int(col)], row_values + displacement[0, :, int(col)], color=color, linewidth=0.55, alpha=0.75)


def _style_image_axis(ax: Any, title: str, *, field: np.ndarray, vmin: float, vmax: float, cmap: str = "RdBu_r") -> Any:
    image = ax.imshow(field, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=7.2, loc="left", pad=3)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#CBD5E1")
        spine.set_linewidth(0.6)
    return image


def _format_common_figure(fig: Any, *, title: str, note: str) -> None:
    fig.suptitle(title, x=0.02, y=0.995, ha="left", fontsize=10.5, fontweight="bold")
    fig.text(0.02, 0.005, note, ha="left", va="bottom", fontsize=6.2, color="#596270")


def _plot_error_comparison(ax: Any, before: np.ndarray, after: np.ndarray, *, vmax: float, before_mae: float, after_mae: float) -> None:
    gap = max(2, before.shape[1] // 30)
    comparison = np.concatenate([before, np.full((before.shape[0], gap), np.nan), after], axis=1)
    ax.imshow(comparison, cmap="magma", vmin=0.0, vmax=vmax, interpolation="nearest")
    ax.axvline(before.shape[1] - 0.5, color="white", linewidth=0.5)
    ax.set_title("(h) Absolute error", fontsize=6.5, loc="left", pad=3)
    left_center = before.shape[1] * 0.5
    right_center = before.shape[1] + gap + after.shape[1] * 0.5
    ax.text(left_center, 0.06 * before.shape[0], "before", color="white", fontsize=5.6, ha="center", va="top")
    ax.text(right_center, 0.06 * before.shape[0], "after", color="white", fontsize=5.6, ha="center", va="top")
    ax.text(left_center, before.shape[0] - 5, f"MAE {before_mae:.3f}", color="white", fontsize=5.6, ha="center", va="bottom")
    ax.text(right_center, after.shape[0] - 5, f"MAE {after_mae:.3f}", color="white", fontsize=5.6, ha="center", va="bottom")
    ax.set_xticks([])
    ax.set_yticks([])


def _add_horizontal_colorbar(fig: Any, scalar_mappable: Any, rect: tuple[float, float, float, float], label: str) -> Any:
    """Add a compact top colorbar without changing the 2 x 4 panel geometry."""
    cax = fig.add_axes(rect)
    colorbar = fig.colorbar(scalar_mappable, cax=cax, orientation="horizontal")
    colorbar.ax.tick_params(labelsize=5.2, length=1.5, pad=1)
    colorbar.ax.xaxis.set_label_position("top")
    colorbar.set_label(label, fontsize=5.6, labelpad=1)
    return colorbar


def plot_rfw_diagnostic(
    *,
    moving: np.ndarray,
    reference: np.ndarray,
    warped: np.ndarray,
    displacement: np.ndarray,
    output_path: str | Path,
    before_metrics: Mapping[str, float],
    after_metrics: Mapping[str, float],
    grid_step: int = 45,
) -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    field_limit = max(float(np.max(np.abs(moving))), float(np.max(np.abs(reference))), float(np.max(np.abs(warped))), 1e-6)
    error_before = np.abs(moving - reference)
    error_after = np.abs(warped - reference)
    error_limit = max(float(np.max(error_before)), float(np.max(error_after)), 1e-6)
    magnitude = np.sqrt(np.square(displacement[0]) + np.square(displacement[1]))

    fig = plt.figure(figsize=(7.0, 4.65), facecolor="white")
    grid = fig.add_gridspec(2, 4, left=0.035, right=0.985, top=0.80, bottom=0.115, wspace=0.17, hspace=0.42)
    axes = np.asarray([[fig.add_subplot(grid[r, c]) for c in range(4)] for r in range(2)])
    _style_image_axis(axes[0, 0], r"(a) Intermediate $R_t$", field=moving, vmin=-field_limit, vmax=field_limit)
    _style_image_axis(axes[0, 1], r"(b) Target $R_{\hat B}$", field=reference, vmin=-field_limit, vmax=field_limit)
    displacement_image = _style_image_axis(
        axes[0, 2], r"(c) Displacement $\|\mathbf{D}(x)\|_2$", field=magnitude,
        vmin=0.0, vmax=max(float(magnitude.max()), 1e-6), cmap="viridis"
    )
    _style_image_axis(axes[1, 0], r"(d) Displacement vectors over $R_t$", field=moving, vmin=-field_limit, vmax=field_limit)
    rows, cols = np.meshgrid(np.arange(0, moving.shape[0], grid_step), np.arange(0, moving.shape[1], grid_step), indexing="ij")
    axes[1, 0].quiver(
        cols, rows, displacement[1, rows, cols], displacement[0, rows, cols],
        color="#111827", angles="xy", scale_units="xy", scale=0.25,
        width=0.0035, headwidth=3.2, headlength=4.0, headaxislength=3.5,
    )
    _style_image_axis(axes[1, 1], "(e) Regular coordinate grid", field=np.zeros_like(moving), vmin=-1.0, vmax=1.0, cmap="Greys")
    axes[1, 1].images[-1].set_alpha(0.0)
    _draw_grid(axes[1, 1], moving.shape, displacement=None, step=grid_step, color="#2563EB")
    _style_image_axis(axes[1, 2], r"(f) Warped coordinate grid $G(x)$", field=np.zeros_like(moving), vmin=-1.0, vmax=1.0, cmap="Greys")
    axes[1, 2].images[-1].set_alpha(0.0)
    _draw_grid(axes[1, 2], moving.shape, displacement=displacement, step=grid_step, color="#C2410C")
    _style_image_axis(axes[1, 3], r"(g) Warped result $R_{\mathrm{warp}}$", field=warped, vmin=-field_limit, vmax=field_limit)
    _plot_error_comparison(axes[0, 3], error_before, error_after, vmax=error_limit, before_mae=float(before_metrics["mae"]), after_mae=float(after_metrics["mae"]))
    _format_common_figure(fig, title="Post-hoc Spatial Registration Diagnostic", note="Post-hoc registration, not a model-internal warp. Quiver arrows are enlarged for readability. Source: independent rendered proxy assets.")
    field_sm = plt.cm.ScalarMappable(norm=Normalize(-field_limit, field_limit), cmap="RdBu_r")
    field_sm.set_array([])
    _add_horizontal_colorbar(fig, field_sm, (0.055, 0.885, 0.235, 0.022), "signed proxy field")
    mag_sm = plt.cm.ScalarMappable(norm=Normalize(0.0, max(float(magnitude.max()), 1e-6)), cmap="viridis")
    mag_sm.set_array([])
    _add_horizontal_colorbar(fig, mag_sm, (0.385, 0.885, 0.215, 0.022), "pixels")
    error_sm = plt.cm.ScalarMappable(norm=Normalize(0.0, error_limit), cmap="magma")
    error_sm.set_array([])
    _add_horizontal_colorbar(fig, error_sm, (0.705, 0.885, 0.215, 0.022), "absolute error")
    return fig


def plot_trajectory_fallback(
    *,
    noise: np.ndarray,
    moving: np.ndarray,
    reference: np.ndarray,
    output_path: str | Path,
) -> Any:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    field_limit = max(float(np.max(np.abs(noise))), float(np.max(np.abs(moving))), float(np.max(np.abs(reference))), 1e-6)
    update = reference - noise
    t_values = (0.25, 0.50, 0.75)
    reference_path = [((1.0 - t) * noise + t * reference) for t in t_values]
    adjacent = np.abs(np.stack([moving - noise, reference - moving], axis=0))
    adjacent_limit = max(float(adjacent.max()), 1e-6)

    fig = plt.figure(figsize=(7.0, 4.65), facecolor="white")
    grid = fig.add_gridspec(2, 4, left=0.035, right=0.985, top=0.91, bottom=0.075, wspace=0.20, hspace=0.36)
    axes = np.asarray([[fig.add_subplot(grid[r, c]) for c in range(4)] for r in range(2)])
    _style_image_axis(axes[0, 0], "(a) Source noise xi", field=noise, vmin=-field_limit, vmax=field_limit)
    _style_image_axis(axes[0, 1], "(b) Provided intermediate proxy R_t", field=moving, vmin=-field_limit, vmax=field_limit)
    _style_image_axis(axes[0, 2], "(c) Residual target proxy R_Bhat", field=reference, vmin=-field_limit, vmax=field_limit)
    _style_image_axis(axes[0, 3], "(d) Value change u_t = R_Bhat - xi", field=update, vmin=-field_limit, vmax=field_limit)
    for axis, value, index in zip(axes[1], reference_path, (0.25, 0.50, 0.75)):
        _style_image_axis(axis, f"t={index:.2f}: linear reference path", field=value, vmin=-field_limit, vmax=field_limit)
    axes[1, 3].imshow(np.concatenate([adjacent[0], np.full((adjacent.shape[1], 2), np.nan), adjacent[1]], axis=1), cmap="magma", vmin=0.0, vmax=adjacent_limit, interpolation="nearest")
    axes[1, 3].set_title("(h) Adjacent absolute changes", fontsize=7.2, loc="left", pad=3)
    axes[1, 3].set_xticks([])
    axes[1, 3].set_yticks([])
    _format_common_figure(fig, title="Residual Flow Matching Trajectory", note="Rendered proxy states; the reference t-path is a visualization aid, not a recorded checkpoint trajectory. No spatial warp is implied.")
    sm = plt.cm.ScalarMappable(norm=plt.Normalize(-field_limit, field_limit), cmap="RdBu_r")
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist()[:4] + axes.ravel().tolist()[4:3:-1], fraction=0.025, pad=0.025)
    cbar.ax.tick_params(labelsize=6, length=2)
    cbar.set_label("signed rendered field", fontsize=6.2, labelpad=2)
    return fig


def _save_figure(fig: Any, output_dir: Path, stem: str) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "pdf": output_dir / f"{stem}.pdf",
        "png": output_dir / f"{stem}.png",
        "svg": output_dir / f"{stem}.svg",
    }
    fig.savefig(paths["pdf"], bbox_inches="tight", pad_inches=0.03)
    fig.savefig(paths["png"], dpi=300, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(paths["svg"], bbox_inches="tight", pad_inches=0.03)
    return {key: str(value) for key, value in paths.items()}


def _read_source_fields(
    noise_path: Path,
    moving_path: Path,
    target_path: Path,
    *,
    composite_path: str | Path | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    paths = {"noise": noise_path, "intermediate": moving_path, "target": target_path}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name} source image: {path}")
    fields = {name: decode_rendered_field(path) for name, path in paths.items()}
    target_shape = fields["target"].shape
    for name in tuple(fields):
        fields[name] = _resize_field(fields[name], target_shape)
    source_metadata = {
        "source_kind": SOURCE_KIND,
        "raw_scalar_arrays_found": False,
        "crop_used": False,
        "composite_attachment": _inspect_composite_attachment(composite_path),
        "scalar_decode": "signed red-minus-blue proxy from the common RdBu_r rendered PNGs",
        "generation_script": str(REPO_ROOT / "docs" / "paper" / "images" / "make_figure2_assets.py"),
        "generation_note": "The repository generator constructs noise and intermediate as a signed visual proxy from the structure tile; no checkpoint R_t tensor is stored.",
        "sources": {
            name: {
                "path": str(path),
                "sha256": _sha256(path),
                "shape": [int(v) for v in fields[name].shape],
            }
            for name, path in paths.items()
        },
    }
    return fields, source_metadata


def run_diagnostic(
    *,
    noise_path: str | Path = DEFAULT_NOISE,
    intermediate_path: str | Path = DEFAULT_INTERMEDIATE,
    target_path: str | Path = DEFAULT_TARGET,
    composite_path: str | Path | None = None,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    max_registration_size: int = DEFAULT_MAX_REGISTRATION_SIZE,
    folding_threshold: float = FOLDING_THRESHOLD,
    grid_step: int = 45,
    attachment: float = 15.0,
    tightness: float = 0.3,
    num_warp: int = 5,
    num_iter: int = 10,
) -> dict[str, Any]:
    output = _resolve_path(output_dir)
    fields, source_metadata = _read_source_fields(
        _resolve_path(noise_path),
        _resolve_path(intermediate_path),
        _resolve_path(target_path),
        composite_path=composite_path,
    )
    moving = fields["intermediate"]
    reference = fields["target"]
    before_metrics = _image_metrics(reference, moving)
    displacement, registration_metadata = estimate_tvl1_displacement(
        moving,
        reference,
        max_size=max_registration_size,
        attachment=attachment,
        tightness=tightness,
        num_warp=num_warp,
        num_iter=num_iter,
    )
    warped = warp_with_displacement(moving, displacement)
    after_metrics = _image_metrics(reference, warped)
    determinant = jacobian_determinant(displacement)
    magnitude = np.sqrt(np.square(displacement[0]) + np.square(displacement[1]))
    folding = determinant <= 0.0
    decision = registration_decision(
        before_mae=before_metrics["mae"],
        after_mae=after_metrics["mae"],
        before_ssim=before_metrics["ssim"],
        after_ssim=after_metrics["ssim"],
        folding_fraction=float(np.mean(folding)),
        folding_threshold=folding_threshold,
        source_kind=source_metadata["source_kind"],
    )
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "registration_arrays.npz",
        noise=fields["noise"],
        intermediate=fields["intermediate"],
        target=fields["target"],
        displacement=displacement,
        warped=warped,
        jacobian_determinant=determinant,
    )
    if decision["posthoc_registration_supported"]:
        figure = plot_rfw_diagnostic(
            moving=moving,
            reference=reference,
            warped=warped,
            displacement=displacement,
            output_path=output / "posthoc_rfw_diagnostic.pdf",
            before_metrics=before_metrics,
            after_metrics=after_metrics,
            grid_step=grid_step,
        )
        figure_outputs = _save_figure(figure, output, "posthoc_rfw_diagnostic")
        import matplotlib.pyplot as plt

        plt.close(figure)
        figure_mode = "posthoc_spatial_registration_diagnostic"
    else:
        figure = plot_trajectory_fallback(noise=fields["noise"], moving=moving, reference=reference, output_path=output / "posthoc_rfw_diagnostic.pdf")
        figure_outputs = _save_figure(figure, output, "posthoc_rfw_diagnostic")
        import matplotlib.pyplot as plt

        plt.close(figure)
        fallback_outputs = _save_figure(plot_trajectory_fallback(noise=fields["noise"], moving=moving, reference=reference, output_path=output / "residual_flow_matching_trajectory.pdf"), output, "residual_flow_matching_trajectory")
        import matplotlib.pyplot as plt

        plt.close("all")
        figure_outputs["fallback_pdf"] = fallback_outputs["pdf"]
        figure_outputs["fallback_png"] = fallback_outputs["png"]
        figure_outputs["fallback_svg"] = fallback_outputs["svg"]
        figure_mode = "residual_flow_matching_trajectory_fallback"
    jacobian_summary = {
        "min": float(np.min(determinant)),
        "p01": float(np.quantile(determinant, 0.01)),
        "median": float(np.median(determinant)),
        "p99": float(np.quantile(determinant, 0.99)),
        "max": float(np.max(determinant)),
        "folding_fraction": float(np.mean(folding)),
        "folding_pixels": int(np.sum(folding)),
        "total_pixels": int(folding.size),
    }
    metrics = {
        "figure_mode": figure_mode,
        "figure_title": "Post-hoc Spatial Registration Diagnostic" if decision["posthoc_registration_supported"] else "Residual Flow Matching Trajectory",
        "source": source_metadata,
        "registration": registration_metadata,
        "before": before_metrics,
        "after": after_metrics,
        "improvement": {
            "mae_absolute": float(before_metrics["mae"] - after_metrics["mae"]),
            "mae_relative": float((before_metrics["mae"] - after_metrics["mae"]) / max(before_metrics["mae"], 1e-12)),
            "ssim_absolute": float(after_metrics["ssim"] - before_metrics["ssim"]),
        },
        "displacement": {
            "mean_magnitude_pixels": float(np.mean(magnitude)),
            "median_magnitude_pixels": float(np.median(magnitude)),
            "max_magnitude_pixels": float(np.max(magnitude)),
            "p95_magnitude_pixels": float(np.quantile(magnitude, 0.95)),
        },
        "jacobian": jacobian_summary,
        "decision": decision,
        "outputs": figure_outputs,
        "no_paper_tex_modified": True,
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    _write_readme(output, metrics)
    return metrics


def _write_readme(output: Path, metrics: Mapping[str, Any]) -> None:
    source = metrics["source"]
    decision = metrics["decision"]
    before = metrics["before"]
    after = metrics["after"]
    displacement = metrics["displacement"]
    jacobian = metrics["jacobian"]
    mode = metrics["figure_mode"]
    lines = [
        "# Post-hoc Spatial Registration Diagnostic",
        "",
        f"Figure mode: `{mode}`.",
        "",
        "This artifact is explicitly a post-hoc image-registration diagnostic. PD-BG-RFM implements Residual Flow Matching, not a spatial-warping layer; the displacement field is not a model output.",
        "",
        "## Provenance",
        "",
        f"- Source kind: `{source['source_kind']}`.",
        f"- Raw scalar model arrays found: `{source['raw_scalar_arrays_found']}`.",
        f"- Crop used: `{source['crop_used']}`.",
        f"- Composite attachment inspected: `{source['composite_attachment'].get('provided', False)}`.",
        (
            f"- Composite attachment: `{source['composite_attachment']['path']}`; it was not used for field extraction."
            if source["composite_attachment"].get("provided")
            else ""
        ),
        f"- Scalar decoding: {source['scalar_decode']}.",
        f"- Generator note: {source['generation_note']}",
        "- The three PNGs are independent rendered proxy assets; they are not checkpoint trajectory states.",
        "",
        "## Registration",
        "",
        "The TV-L1 estimator computes D for `G(x)=x+D(x)`, and the warped result is sampled as `R_warp(x)=S(R_t,G(x))` with bilinear interpolation.",
        f"- Before: MAE `{before['mae']:.6f}`, SSIM `{before['ssim']:.6f}`.",
        f"- After: MAE `{after['mae']:.6f}`, SSIM `{after['ssim']:.6f}`.",
        f"- Mean displacement magnitude: `{displacement['mean_magnitude_pixels']:.4f}` pixels.",
        f"- Maximum displacement magnitude: `{displacement['max_magnitude_pixels']:.4f}` pixels.",
        f"- Jacobian determinant: min `{jacobian['min']:.6f}`, median `{jacobian['median']:.6f}`, max `{jacobian['max']:.6f}`.",
        f"- Folding fraction (det(J) <= 0): `{jacobian['folding_fraction']:.6%}`.",
        "",
        "## Decision",
        "",
        f"- Error improvement: `{decision['errors_improve']}`.",
        f"- Structural similarity improvement: `{decision['structure_improves']}`.",
        f"- No severe folding: `{decision['no_severe_folding']}` with threshold `{decision['folding_threshold']:.4f}`.",
        f"- Post-hoc registration supported for this pair: `{decision['posthoc_registration_supported']}`.",
        f"- Model-internal RFW supported: `{decision['model_internal_rfw_supported']}`.",
        f"- Interpretation: {decision['interpretation']}",
        "",
        "When the registration gate fails, the primary figure is replaced by a Residual Flow Matching Trajectory view. Its reference linear-path panels are visualization aids, not recorded model states.",
        "",
        "No `paper.tex` file was modified by this diagnostic.",
    ]
    (output / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--noise", default=str(DEFAULT_NOISE))
    parser.add_argument("--intermediate", default=str(DEFAULT_INTERMEDIATE))
    parser.add_argument("--target", default=str(DEFAULT_TARGET))
    parser.add_argument("--composite", default=None, help="Optional composite attachment to record as inspected provenance.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--max-registration-size", type=int, default=DEFAULT_MAX_REGISTRATION_SIZE)
    parser.add_argument("--folding-threshold", type=float, default=FOLDING_THRESHOLD)
    parser.add_argument("--grid-step", type=int, default=45)
    parser.add_argument("--attachment", type=float, default=15.0)
    parser.add_argument("--tightness", type=float, default=0.3)
    parser.add_argument("--num-warp", type=int, default=5)
    parser.add_argument("--num-iter", type=int, default=10)
    args = parser.parse_args()
    metrics = run_diagnostic(
        noise_path=args.noise,
        intermediate_path=args.intermediate,
        target_path=args.target,
        composite_path=args.composite,
        output_dir=args.output_dir,
        max_registration_size=args.max_registration_size,
        folding_threshold=args.folding_threshold,
        grid_step=args.grid_step,
        attachment=args.attachment,
        tightness=args.tightness,
        num_warp=args.num_warp,
        num_iter=args.num_iter,
    )
    print(json.dumps({
        "figure_mode": metrics["figure_mode"],
        "posthoc_registration_supported": metrics["decision"]["posthoc_registration_supported"],
        "model_internal_rfw_supported": metrics["decision"]["model_internal_rfw_supported"],
        "before": metrics["before"],
        "after": metrics["after"],
        "folding_fraction": metrics["jacobian"]["folding_fraction"],
        "output_dir": str(_resolve_path(args.output_dir)),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
