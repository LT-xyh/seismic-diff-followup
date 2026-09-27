"""Generate auditable UMAP/t-SNE diagnostics for condition embeddings.

The main figure is intentionally diagnostic rather than causal evidence. It uses
the modality-specific projector spaces optimized by the contrastive objective and
keeps the sample membership, reducer settings, and checkpoint provenance explicit.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib
import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

from bg_pdr_fm.data import STANDARD_MODALITIES, batch_to_device
from bg_pdr_fm.evaluation.visualize_condition_contract import (
    MODALITY_LABELS,
    ROLE_TO_ANCHOR,
    ROLE_TO_HEAD,
    _batch_record_ids,
    _loader,
    _prepare_model,
    _resolve_path,
    _slice_batch,
    _to_numpy,
    collect_seed_diagnostics,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
ROLE_NAMES = ("background", "structural")
DEFAULT_SUBSETS = (
    "FlatVelA",
    "FlatVelB",
    "CurveVelA",
    "CurveVelB",
    "FlatFaultA",
    "FlatFaultB",
    "CurveFaultA",
    "CurveFaultB",
)
EXPECTED_TEST_ROWS = 33_600
DEFAULT_SAMPLE_SEED = 42
DEFAULT_REDUCER_SEEDS = (42, 3407, 7777)
GREEN = "#178F48"
BLUE = "#1E63D5"
ANCHOR_GOLD = "#C18F00"
MODALITY_COLORS = {
    "migrated_image": "#0072B2",
    "horizon": "#D55E00",
    "rms_vel": "#009E73",
    "well_log": "#CC79A7",
}
MODALITY_MARKERS = {
    "migrated_image": "o",
    "horizon": "^",
    "rms_vel": "s",
    "well_log": "D",
}


@dataclass(frozen=True)
class ViewData:
    name: str
    features: np.ndarray
    labels: np.ndarray
    roles: np.ndarray
    modalities: np.ndarray
    is_anchor: np.ndarray
    selected_rows: np.ndarray
    dataset_ids: np.ndarray
    dataset_names: np.ndarray
    source_sample_indices: np.ndarray
    label_semantics: str


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _resolve_path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _l2_normalize(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    if np.any(norms <= 0) or not np.all(np.isfinite(array)):
        raise ValueError("Embedding matrix contains non-finite or zero-norm rows.")
    return array / norms


def stratified_complete_indices(
    dataset_ids: np.ndarray,
    sample_indices: np.ndarray,
    availability: np.ndarray,
    dataset_names: Sequence[str],
    *,
    per_subset: int = 128,
    seed: int = DEFAULT_SAMPLE_SEED,
) -> np.ndarray:
    """Select a fixed number of complete-modality records per dataset."""
    dataset_ids = np.asarray(dataset_ids).reshape(-1)
    sample_indices = np.asarray(sample_indices).reshape(-1)
    availability = np.asarray(availability, dtype=bool)
    if dataset_ids.size != sample_indices.size or availability.shape[0] != dataset_ids.size:
        raise ValueError("dataset_ids, sample_indices, and availability must share the row count.")
    if availability.ndim != 2 or availability.shape[1] != len(STANDARD_MODALITIES):
        raise ValueError(
            f"availability must have shape (N, {len(STANDARD_MODALITIES)}), got {availability.shape}."
        )
    if int(per_subset) < 1:
        raise ValueError("per_subset must be positive.")

    rng = np.random.default_rng(int(seed))
    complete = np.all(availability, axis=1)
    selected: list[np.ndarray] = []
    for dataset_id, dataset_name in enumerate(dataset_names):
        candidates = np.flatnonzero(complete & (dataset_ids == int(dataset_id)))
        if candidates.size < int(per_subset):
            raise ValueError(
                f"{dataset_name} has only {candidates.size} complete-modality records; "
                f"need {int(per_subset)}."
            )
        chosen = rng.choice(candidates, size=int(per_subset), replace=False)
        selected.append(np.sort(chosen.astype(np.int64)))
    return np.concatenate(selected, axis=0)


def knn_overlap(coordinates_a: np.ndarray, coordinates_b: np.ndarray, *, k: int = 15) -> float:
    """Return mean k-nearest-neighbor set overlap between two layouts."""
    a = np.asarray(coordinates_a, dtype=np.float64)
    b = np.asarray(coordinates_b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2 or a.shape[0] < 3:
        raise ValueError("Coordinate arrays must have the same 2-D shape with at least three rows.")
    neighbors = min(int(k), a.shape[0] - 1)
    if neighbors < 1:
        raise ValueError("k must leave at least one neighbor.")
    indices_a = NearestNeighbors(n_neighbors=neighbors + 1).fit(a).kneighbors(return_distance=False)[:, 1:]
    indices_b = NearestNeighbors(n_neighbors=neighbors + 1).fit(b).kneighbors(return_distance=False)[:, 1:]
    overlaps = [len(set(left).intersection(right)) / float(neighbors) for left, right in zip(indices_a, indices_b)]
    value = float(np.mean(overlaps))
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"Invalid kNN overlap: {value}")
    return value


def compute_embedding_metrics(
    original: np.ndarray,
    labels: Sequence[str],
    coordinates: np.ndarray,
    *,
    trustworthiness_neighbors: int = 15,
) -> dict[str, float]:
    """Compute qualitative-layout and original-space diagnostics."""
    original = _l2_normalize(original)
    coordinates = np.asarray(coordinates, dtype=np.float64)
    labels = np.asarray(labels)
    if coordinates.shape[0] != original.shape[0] or coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("coordinates must have shape (N, 2) matching original.")
    if np.unique(labels).size < 2:
        raise ValueError("At least two labels are required for silhouette diagnostics.")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError("2-D coordinates contain NaN or Inf.")
    neighbors = min(int(trustworthiness_neighbors), max(2, (original.shape[0] - 1) // 2))
    metrics = {
        "trustworthiness": float(trustworthiness(original, coordinates, n_neighbors=neighbors, metric="cosine")),
        "silhouette_original": float(silhouette_score(original, labels, metric="cosine")),
        "silhouette_2d": float(silhouette_score(coordinates, labels, metric="euclidean")),
    }
    if not all(np.isfinite(value) for value in metrics.values()):
        raise ValueError(f"Non-finite embedding metrics: {metrics}")
    return metrics


def _read_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = _resolve_path(path)
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != EXPECTED_TEST_ROWS:
        raise ValueError(f"Expected {EXPECTED_TEST_ROWS} manifest rows, found {len(rows)}.")
    identities = [
        (int(row["dataset_id"]), str(row["dataset_name"]), int(row["source_sample_index"]))
        for row in rows
    ]
    if len(set(identities)) != len(identities):
        raise ValueError("Held-out manifest contains duplicate identities.")
    counts = {name: 0 for name in DEFAULT_SUBSETS}
    for _, name, _ in identities:
        if name not in counts:
            raise ValueError(f"Unexpected OpenFWI subset in manifest: {name}")
        counts[name] += 1
    if any(value == 0 for value in counts.values()):
        raise ValueError(f"Manifest is missing OpenFWI subsets: {counts}")
    return {
        "path": str(manifest_path),
        "rows": len(rows),
        "unique_identities": len(identities),
        "dataset_counts": counts,
        "sha256": _sha256_file(manifest_path),
        "identities": identities,
    }


def _cache_key(role: str, modality: str) -> str:
    return f"{role}__{modality}"


def _cache_from_diagnostics(data: Any) -> dict[str, Any]:
    cache: dict[str, Any] = {
        "dataset_ids": np.asarray(data.dataset_ids, dtype=np.int64),
        "sample_indices": np.asarray(data.sample_indices, dtype=np.int64),
        "availability": np.asarray(data.availability, dtype=bool),
        "dataset_names": np.asarray(data.dataset_names, dtype="U32"),
    }
    for role in ROLE_NAMES:
        cache[f"anchor__{role}"] = _l2_normalize(data.anchor_embeddings[role])
        for modality in STANDARD_MODALITIES:
            cache[_cache_key(role, modality)] = _l2_normalize(data.embeddings[role][modality])
    return cache


@torch.inference_mode()
def _collect_projector_cache(
    *,
    config_path: str | Path,
    checkpoint_path: str | Path,
    seed: int,
    device: str | torch.device,
    batch_size: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Stream only projected embeddings, releasing large feature maps per batch."""
    resolved_device = torch.device(device)
    model, conf, checkpoint_info = _prepare_model(
        _resolve_path(config_path), _resolve_path(checkpoint_path), resolved_device
    )
    loader, _ = _loader(conf, "test", int(batch_size))
    embedding_parts: dict[tuple[str, str], list[np.ndarray]] = {
        (role, modality): [] for role in ROLE_NAMES for modality in STANDARD_MODALITIES
    }
    anchor_parts: dict[str, list[np.ndarray]] = {role: [] for role in ROLE_NAMES}
    dataset_parts: list[np.ndarray] = []
    sample_parts: list[np.ndarray] = []
    availability_parts: list[np.ndarray] = []
    dataset_names = [str(name) for name in conf.data.openfwi_datasets]
    seen = 0
    for batch_number, cpu_batch in enumerate(loader, start=1):
        batch = batch_to_device(cpu_batch, resolved_device)
        features = model.encoder(batch, return_all_heads=True)
        if features.all_heads is None:
            raise RuntimeError("Condition embedding extraction requires all modality role heads.")
        anchor_bundle = model.wavelet_anchor(batch.depth_vel)
        role_anchors = {
            "background": anchor_bundle.anchor_num,
            "structural": anchor_bundle.anchor_struct,
        }
        for role in ROLE_NAMES:
            anchor_parts[role].append(
                _to_numpy(model.contrastive_loss._project(ROLE_TO_ANCHOR[role], role_anchors[role]))
            )
            for modality in STANDARD_MODALITIES:
                head = features.all_heads[modality][ROLE_TO_HEAD[role]]
                embedding_parts[(role, modality)].append(
                    _to_numpy(model.contrastive_loss._project(modality, head))
                )
        dataset_ids, source_indices = _batch_record_ids(cpu_batch, start_row=seen)
        dataset_parts.append(dataset_ids)
        sample_parts.append(source_indices)
        availability_parts.append(
            _to_numpy((cpu_batch.modality_mask * cpu_batch.modality_quality).gt(0)).astype(bool)
        )
        seen += int(cpu_batch.depth_vel.shape[0])
        del anchor_bundle, features, batch, cpu_batch
        if resolved_device.type == "cuda" and batch_number % 8 == 0:
            torch.cuda.empty_cache()
            gc.collect()
        if batch_number == 1 or batch_number % 50 == 0:
            print(f"[condition-embeddings] batches={batch_number} rows={seen}", flush=True)
    if seen == 0:
        raise RuntimeError("Projector cache extraction collected zero test rows.")
    cache: dict[str, Any] = {
        "dataset_ids": np.concatenate(dataset_parts).astype(np.int64),
        "sample_indices": np.concatenate(sample_parts).astype(np.int64),
        "availability": np.concatenate(availability_parts).astype(bool),
        "dataset_names": np.asarray(dataset_names, dtype="U32"),
    }
    for role in ROLE_NAMES:
        cache[f"anchor__{role}"] = _l2_normalize(np.concatenate(anchor_parts[role], axis=0))
        for modality in STANDARD_MODALITIES:
            cache[_cache_key(role, modality)] = _l2_normalize(
                np.concatenate(embedding_parts[(role, modality)], axis=0)
            )
    _validate_cache(cache, expected_rows=seen)
    checkpoint_info = dict(checkpoint_info)
    checkpoint_info["streamed_rows"] = int(seen)
    checkpoint_info["streamed_batch_size"] = int(batch_size)
    checkpoint_info["streamed_seed"] = int(seed)
    return cache, checkpoint_info


def _validate_cache(cache: Mapping[str, Any], *, expected_rows: int = EXPECTED_TEST_ROWS) -> None:
    dataset_ids = np.asarray(cache["dataset_ids"])
    sample_indices = np.asarray(cache["sample_indices"])
    availability = np.asarray(cache["availability"])
    if dataset_ids.shape != (expected_rows,) or sample_indices.shape != (expected_rows,):
        raise ValueError("Embedding cache does not contain the formal test row count.")
    if availability.shape != (expected_rows, len(STANDARD_MODALITIES)):
        raise ValueError(f"Unexpected availability shape: {availability.shape}")
    if len(np.unique(np.stack([dataset_ids, sample_indices], axis=1), axis=0)) != expected_rows:
        raise ValueError("Embedding cache contains duplicate source identities.")
    for role in ROLE_NAMES:
        for modality in STANDARD_MODALITIES:
            values = np.asarray(cache[_cache_key(role, modality)])
            if values.shape[0] != expected_rows or values.ndim != 2:
                raise ValueError(f"Invalid shape for {role}/{modality}: {values.shape}")
            _l2_normalize(values)
        anchor = np.asarray(cache[f"anchor__{role}"])
        if anchor.shape[0] != expected_rows or anchor.ndim != 2:
            raise ValueError(f"Invalid shape for {role} anchor: {anchor.shape}")
        _l2_normalize(anchor)


def save_embedding_cache(path: str | Path, cache: Mapping[str, Any]) -> None:
    _validate_cache(cache)
    target = _resolve_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **dict(cache))


def load_embedding_cache(path: str | Path) -> dict[str, np.ndarray]:
    source = _resolve_path(path)
    with np.load(source, allow_pickle=False) as loaded:
        cache = {name: loaded[name] for name in loaded.files}
    _validate_cache(cache)
    return cache


def build_views(cache: Mapping[str, Any], selected_rows: np.ndarray, dataset_names: Sequence[str]) -> dict[str, ViewData]:
    """Build the three common-space views used by the six-panel figure."""
    selected_rows = np.asarray(selected_rows, dtype=np.int64)
    dataset_ids = np.asarray(cache["dataset_ids"], dtype=np.int64)[selected_rows]
    source_indices = np.asarray(cache["sample_indices"], dtype=np.int64)[selected_rows]
    selected_names = np.asarray([dataset_names[int(value)] for value in dataset_ids], dtype="U32")
    blocks: list[np.ndarray] = []
    roles: list[np.ndarray] = []
    modalities: list[np.ndarray] = []
    anchors: list[np.ndarray] = []
    row_ids: list[np.ndarray] = []
    dataset_id_blocks: list[np.ndarray] = []
    dataset_name_blocks: list[np.ndarray] = []
    source_blocks: list[np.ndarray] = []
    for role in ROLE_NAMES:
        for modality in STANDARD_MODALITIES:
            blocks.append(np.asarray(cache[_cache_key(role, modality)])[selected_rows])
            roles.append(np.full(selected_rows.size, role, dtype="U16"))
            modalities.append(np.full(selected_rows.size, modality, dtype="U32"))
            anchors.append(np.zeros(selected_rows.size, dtype=bool))
            row_ids.append(selected_rows)
            dataset_id_blocks.append(dataset_ids)
            dataset_name_blocks.append(selected_names)
            source_blocks.append(source_indices)
    joint = ViewData(
        name="joint_roles",
        features=np.concatenate(blocks, axis=0),
        labels=np.concatenate(roles),
        roles=np.concatenate(roles),
        modalities=np.concatenate(modalities),
        is_anchor=np.concatenate(anchors),
        selected_rows=np.concatenate(row_ids),
        dataset_ids=np.concatenate(dataset_id_blocks),
        dataset_names=np.concatenate(dataset_name_blocks),
        source_sample_indices=np.concatenate(source_blocks),
        label_semantics="role",
    )

    role_views: dict[str, ViewData] = {"joint_roles": joint}
    for role in ROLE_NAMES:
        role_blocks = [np.asarray(cache[_cache_key(role, modality)])[selected_rows] for modality in STANDARD_MODALITIES]
        role_blocks.append(np.asarray(cache[f"anchor__{role}"])[selected_rows])
        role_labels = [np.full(selected_rows.size, modality, dtype="U32") for modality in STANDARD_MODALITIES]
        role_labels.append(np.full(selected_rows.size, "anchor", dtype="U32"))
        role_modalities = role_labels.copy()
        role_anchors = [np.zeros(selected_rows.size, dtype=bool) for _ in STANDARD_MODALITIES]
        role_anchors.append(np.ones(selected_rows.size, dtype=bool))
        role_views[role] = ViewData(
            name=role,
            features=np.concatenate(role_blocks, axis=0),
            labels=np.concatenate(role_labels),
            roles=np.full(selected_rows.size * 5, role, dtype="U16"),
            modalities=np.concatenate(role_modalities),
            is_anchor=np.concatenate(role_anchors),
            selected_rows=np.tile(selected_rows, 5),
            dataset_ids=np.tile(dataset_ids, 5),
            dataset_names=np.tile(selected_names, 5),
            source_sample_indices=np.tile(source_indices, 5),
            label_semantics="modality_with_training_only_anchor",
        )
    return role_views


def _reduce(features: np.ndarray, algorithm: str, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    normalized = _l2_normalize(features)
    components = min(50, normalized.shape[1], normalized.shape[0] - 1)
    pca = PCA(n_components=components, whiten=False, svd_solver="auto", random_state=int(seed))
    reduced = _l2_normalize(pca.fit_transform(normalized))
    if algorithm == "umap":
        try:
            umap_module = importlib.import_module("umap")
        except ImportError as exc:
            raise RuntimeError("UMAP requires umap-learn in conda environment seg.") from exc
        reducer = umap_module.UMAP(
            n_components=2,
            n_neighbors=30,
            min_dist=0.15,
            metric="cosine",
            random_state=int(seed),
            n_jobs=1,
        )
        coordinates = reducer.fit_transform(reduced)
    elif algorithm == "tsne":
        perplexity = min(30.0, max(5.0, (reduced.shape[0] - 1) / 3.0))
        try:
            from openTSNE import TSNE as OpenTSNE
        except ImportError as exc:
            raise RuntimeError("t-SNE requires openTSNE in conda environment seg.") from exc
        reducer = OpenTSNE(
            n_components=2,
            perplexity=perplexity,
            metric="cosine",
            initialization="pca",
            learning_rate="auto",
            n_iter=2000,
            random_state=int(seed),
            n_jobs=8,
            negative_gradient_method="fft",
            verbose=False,
        )
        coordinates = np.asarray(reducer.fit(reduced), dtype=np.float64)
    else:
        raise ValueError(f"Unknown reduction algorithm: {algorithm}")
    if not np.all(np.isfinite(coordinates)):
        raise ValueError(f"{algorithm} produced non-finite coordinates.")
    return np.asarray(coordinates, dtype=np.float64), {
        "pca_components": int(components),
        "pca_explained_variance": float(np.sum(pca.explained_variance_ratio_)),
        "reducer_seed": int(seed),
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _plot_view(ax: Any, view: ViewData, coordinates: np.ndarray, *, view_name: str) -> None:
    if view_name == "joint_roles":
        for role, color in (("background", GREEN), ("structural", BLUE)):
            for modality in STANDARD_MODALITIES:
                mask = (view.roles == role) & (view.modalities == modality)
                ax.scatter(
                    coordinates[mask, 0],
                    coordinates[mask, 1],
                    s=8,
                    alpha=0.30,
                    color=color,
                    marker=MODALITY_MARKERS[modality],
                    linewidths=0,
                    rasterized=True,
                )
    else:
        for modality in STANDARD_MODALITIES:
            mask = (view.modalities == modality) & (~view.is_anchor)
            ax.scatter(
                coordinates[mask, 0],
                coordinates[mask, 1],
                s=9,
                alpha=0.30,
                color=MODALITY_COLORS[modality],
                marker=MODALITY_MARKERS[modality],
                linewidths=0,
                rasterized=True,
            )
        anchor_mask = view.is_anchor
        ax.scatter(
            coordinates[anchor_mask, 0],
            coordinates[anchor_mask, 1],
            s=22,
            alpha=0.75,
            color=ANCHOR_GOLD,
            marker="*",
            edgecolors="black",
            linewidths=0.25,
            rasterized=True,
        )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_color("#7A838C")
        spine.set_linewidth(0.6)


def _add_joint_legend(ax: Any) -> None:
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=GREEN, markeredgecolor="none", label="Background"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=BLUE, markeredgecolor="none", label="Structural"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=5.5, frameon=False, handletextpad=0.3)


def _add_role_legend(ax: Any) -> None:
    handles = [
        Line2D([0], [0], marker=MODALITY_MARKERS[modality], color="none", markerfacecolor=MODALITY_COLORS[modality], markeredgecolor="none", label=MODALITY_LABELS[modality])
        for modality in STANDARD_MODALITIES
    ]
    handles.append(Line2D([0], [0], marker="*", color="none", markerfacecolor=ANCHOR_GOLD, markeredgecolor="black", label="Anchor"))
    ax.legend(handles=handles, loc="upper right", fontsize=5.2, frameon=False, handletextpad=0.25, ncol=2)


def render_main_figure(
    output_path: str | Path,
    views: Mapping[str, ViewData],
    canonical_coordinates: Mapping[tuple[str, str], np.ndarray],
    metric_rows: Sequence[Mapping[str, Any]],
) -> None:
    output_path = _resolve_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"], "font.size": 7})
    figure, axes = plt.subplots(2, 3, figsize=(10.4, 6.0), squeeze=False)
    view_order = ("joint_roles", "background", "structural")
    view_titles = ("Joint role space", "Background-role space", "Structural-role space")
    algorithms = (("umap", "UMAP"), ("tsne", "t-SNE"))
    for row, (algorithm, algorithm_title) in enumerate(algorithms):
        for col, (view_name, title) in enumerate(zip(view_order, view_titles)):
            axis = axes[row, col]
            view = views[view_name]
            coordinates = canonical_coordinates[(algorithm, view_name)]
            _plot_view(axis, view, coordinates, view_name=view_name)
            if row == 0:
                axis.set_title(title, fontsize=8.2, fontweight="bold", pad=4)
            if col == 0:
                axis.text(-0.12, 0.5, algorithm_title, transform=axis.transAxes, rotation=90, va="center", ha="center", fontsize=8, fontweight="bold")
            relevant = [row_data for row_data in metric_rows if row_data["algorithm"] == algorithm and row_data["view"] == view_name and row_data["reducer_seed"] == 42]
            if relevant:
                row_data = relevant[0]
                axis.text(0.02, 0.02, f"T={float(row_data['trustworthiness']):.2f}  S={float(row_data['silhouette_original']):.2f}", transform=axis.transAxes, fontsize=5.4, color="#4C5660")
            if row == 0 and col == 0:
                _add_joint_legend(axis)
            if row == 0 and col in (1, 2):
                _add_role_legend(axis)
    figure.text(0.5, 0.008, "Points are fixed held-out sample-role representations; 2-D layouts are qualitative. Metrics are computed in the original projector space.", ha="center", fontsize=6.4)
    figure.savefig(output_path, bbox_inches="tight", facecolor="white")
    figure.savefig(output_path.with_suffix(".png"), dpi=600, bbox_inches="tight", facecolor="white")
    figure.savefig(output_path.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(figure)


def render_dataset_audit(
    output_path: str | Path,
    view: ViewData,
    canonical_coordinates: Mapping[str, np.ndarray],
) -> None:
    output_path = _resolve_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    palette = plt.get_cmap("tab10")
    dataset_ids = np.unique(view.dataset_ids)
    figure, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), squeeze=False)
    for axis, algorithm in zip(axes[0], ("umap", "tsne")):
        coordinates = canonical_coordinates[algorithm]
        for position, dataset_id in enumerate(dataset_ids):
            mask = view.dataset_ids == dataset_id
            axis.scatter(coordinates[mask, 0], coordinates[mask, 1], s=7, alpha=0.24, color=palette(position % 10), rasterized=True, label=str(view.dataset_names[mask][0]))
        axis.set_title(algorithm.upper() if algorithm == "umap" else "t-SNE", fontsize=8, fontweight="bold")
        axis.set_xticks([])
        axis.set_yticks([])
        axis.legend(fontsize=5.2, frameon=False, ncol=2, loc="upper right")
        for spine in axis.spines.values():
            spine.set_color("#7A838C")
            spine.set_linewidth(0.6)
    figure.suptitle("Dataset-colored audit view; not used as the main evidence", fontsize=9, fontweight="bold")
    figure.savefig(output_path, bbox_inches="tight", facecolor="white")
    figure.savefig(output_path.with_suffix(".png"), dpi=400, bbox_inches="tight", facecolor="white")
    figure.savefig(output_path.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _package_version(name: str) -> str | None:
    try:
        module = importlib.import_module(name)
    except ImportError:
        return None
    return str(getattr(module, "__version__", "available"))


def generate_assets(
    *,
    config_path: str | Path,
    checkpoint_path: str | Path,
    manifest_path: str | Path,
    output_dir: str | Path,
    cache_path: str | Path,
    device: str | torch.device,
    batch_size: int = 64,
    per_subset: int = 128,
    sample_seed: int = DEFAULT_SAMPLE_SEED,
    reducer_seeds: Sequence[int] = DEFAULT_REDUCER_SEEDS,
    force_extract: bool = False,
) -> dict[str, Any]:
    output_dir = _resolve_path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = _resolve_path(checkpoint_path)
    config_path = _resolve_path(config_path)
    manifest_path = _resolve_path(manifest_path)
    manifest_audit = _read_manifest(manifest_path)
    if force_extract or not _resolve_path(cache_path).exists():
        cache, checkpoint_info = _collect_projector_cache(
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            device=device,
            batch_size=batch_size,
            seed=sample_seed,
        )
        save_embedding_cache(cache_path, cache)
    else:
        cache = load_embedding_cache(cache_path)
        checkpoint_info = {"loaded_from_cache": True}
    _validate_cache(cache)
    dataset_names = [str(value) for value in np.asarray(cache["dataset_names"]).tolist()]
    if tuple(dataset_names) != DEFAULT_SUBSETS:
        raise ValueError(f"Unexpected OpenFWI ordering: {dataset_names}")
    cache_ids = {(int(dataset_id), int(sample_index)) for dataset_id, sample_index in zip(cache["dataset_ids"], cache["sample_indices"])}
    manifest_ids = {(int(dataset_id), int(sample_index)) for dataset_id, _, sample_index in manifest_audit["identities"]}
    if cache_ids != manifest_ids:
        raise ValueError("Embedding cache identities do not match the formal held-out manifest.")
    selected = stratified_complete_indices(
        cache["dataset_ids"],
        cache["sample_indices"],
        cache["availability"],
        dataset_names,
        per_subset=per_subset,
        seed=sample_seed,
    )
    views = build_views(cache, selected, dataset_names)
    provenance_rows: list[dict[str, Any]] = []
    for order, row in enumerate(selected.tolist()):
        dataset_id = int(cache["dataset_ids"][row])
        provenance_rows.append(
            {
                "sample_order": order,
                "cache_row": row,
                "dataset_id": dataset_id,
                "dataset_name": dataset_names[dataset_id],
                "source_sample_index": int(cache["sample_indices"][row]),
                "complete_modalities": True,
                "availability": ";".join(str(int(value)) for value in cache["availability"][row]),
                "sample_seed": int(sample_seed),
            }
        )
    _write_csv(output_dir / "provenance.csv", provenance_rows, tuple(provenance_rows[0].keys()))

    if 42 not in tuple(int(value) for value in reducer_seeds):
        raise ValueError("Reducer seeds must include canonical seed 42.")
    canonical_seed = 42
    canonical_coordinates: dict[tuple[str, str], np.ndarray] = {}
    reduced_by_seed: dict[tuple[str, str, int], np.ndarray] = {}
    metric_rows: list[dict[str, Any]] = []
    for algorithm in ("umap", "tsne"):
        for view_name, view in views.items():
            for reducer_seed in reducer_seeds:
                print(
                    f"[condition-embeddings] reducer={algorithm}/{view_name}/seed={int(reducer_seed)}",
                    flush=True,
                )
                coordinates, reduction_info = _reduce(view.features, algorithm, int(reducer_seed))
                reduced_by_seed[(algorithm, view_name, int(reducer_seed))] = coordinates
                labels = view.labels
                metrics = compute_embedding_metrics(view.features, labels, coordinates)
                row = {
                    "algorithm": algorithm,
                    "view": view_name,
                    "reducer_seed": int(reducer_seed),
                    "label_semantics": view.label_semantics,
                    "n_points": int(view.features.shape[0]),
                    **reduction_info,
                    **metrics,
                }
                if int(reducer_seed) == canonical_seed:
                    canonical_coordinates[(algorithm, view_name)] = coordinates
                metric_rows.append(row)
            canonical = canonical_coordinates[(algorithm, view_name)]
            for row in metric_rows:
                if row["algorithm"] == algorithm and row["view"] == view_name and row["reducer_seed"] != canonical_seed:
                    other = reduced_by_seed[(algorithm, view_name, int(row["reducer_seed"]))]
                    row["knn_overlap_to_canonical"] = knn_overlap(canonical, other, k=15)
    metric_fields = (
        "algorithm",
        "view",
        "reducer_seed",
        "label_semantics",
        "n_points",
        "pca_components",
        "pca_explained_variance",
        "trustworthiness",
        "silhouette_original",
        "silhouette_2d",
        "knn_overlap_to_canonical",
    )
    _write_csv(output_dir / "metrics.csv", metric_rows, metric_fields)

    coordinate_rows: list[dict[str, Any]] = []
    for (algorithm, view_name), coordinates in canonical_coordinates.items():
        view = views[view_name]
        for point, (x_value, y_value) in enumerate(coordinates):
            coordinate_rows.append(
                {
                    "algorithm": algorithm,
                    "view": view_name,
                    "reducer_seed": canonical_seed,
                    "point": point,
                    "role": str(view.roles[point]),
                    "modality": str(view.modalities[point]),
                    "is_anchor": bool(view.is_anchor[point]),
                    "cache_row": int(view.selected_rows[point]),
                    "dataset_id": int(view.dataset_ids[point]),
                    "dataset_name": str(view.dataset_names[point]),
                    "source_sample_index": int(view.source_sample_indices[point]),
                    "x": float(x_value),
                    "y": float(y_value),
                }
            )
    _write_csv(output_dir / "coordinates.csv", coordinate_rows, tuple(coordinate_rows[0].keys()))
    render_main_figure(output_dir / "condition_embedding_clusters.pdf", views, canonical_coordinates, metric_rows)
    render_dataset_audit(
        output_dir / "condition_embedding_dataset_audit.pdf",
        views["joint_roles"],
        {algorithm: canonical_coordinates[(algorithm, "joint_roles")] for algorithm in ("umap", "tsne")},
    )
    metadata = {
        "evidence_scope": "single_checkpoint_diagnostic",
        "causal_ablation": False,
        "causal_statement": "UMAP/t-SNE layouts are qualitative representation diagnostics, not a causal contrastive-loss ablation.",
        "inputs": {
            "checkpoint": str(checkpoint_path),
            "config": str(config_path),
            "manifest": str(manifest_path),
            "checkpoint_sha256": _sha256_file(checkpoint_path),
            "config_sha256": _sha256_file(config_path),
            "manifest_sha256": _sha256_file(manifest_path),
        },
        "checkpoint": checkpoint_info,
        "manifest_audit": {key: value for key, value in manifest_audit.items() if key != "identities"},
        "sampling": {
            "complete_modalities_only": True,
            "per_subset": int(per_subset),
            "selected_rows": int(selected.size),
            "subsets": dataset_names,
            "seed": int(sample_seed),
            "selection_rule": "fixed stratified random sample from formal held-out records; no metric or visual filtering",
        },
        "representations": {
            "joint_roles": "four modality-specific projected numerical/structural heads",
            "background": "projected numerical role heads plus projected training-only background anchor",
            "structural": "projected structural role heads plus projected training-only structural anchor",
            "projector_space": "trained modality/anchor contrastive projectors",
            "inference_claim": "anchors are diagnostic quantities and unavailable at inference",
        },
        "reduction": {
            "pca_components": 50,
            "pca_whiten": False,
            "umap": {"metric": "cosine", "n_neighbors": 30, "min_dist": 0.15, "seed": canonical_seed},
            "tsne": {"backend": "openTSNE", "metric": "cosine", "perplexity": 30, "init": "pca", "learning_rate": "auto", "max_iter": 2000, "negative_gradient_method": "fft", "seed": canonical_seed},
            "stability_seeds": [int(value) for value in reducer_seeds],
        },
        "dependency_versions": {
            "python": platform.python_version(),
            "numpy": _package_version("numpy"),
            "scipy": _package_version("scipy"),
            "sklearn": _package_version("sklearn"),
            "umap": _package_version("umap"),
            "openTSNE": _package_version("openTSNE"),
            "torch": _package_version("torch"),
        },
        "outputs": {
            "figure_pdf": str(output_dir / "condition_embedding_clusters.pdf"),
            "figure_png": str(output_dir / "condition_embedding_clusters.png"),
            "figure_svg": str(output_dir / "condition_embedding_clusters.svg"),
            "dataset_audit_pdf": str(output_dir / "condition_embedding_dataset_audit.pdf"),
            "coordinates_csv": str(output_dir / "coordinates.csv"),
            "metrics_csv": str(output_dir / "metrics.csv"),
            "provenance_csv": str(output_dir / "provenance.csv"),
            "cache": str(_resolve_path(cache_path)),
        },
        "paper_recommendation": "appendix_diagnostic_only",
        "visual_review": {
            "main_text_candidate": False,
            "reason": "Joint role separation is stable, but structural cross-modal modality silhouette is negative; retain as an appendix diagnostic rather than causal main-text evidence.",
            "canonical_original_space_silhouette": {
                "joint_roles": next(row["silhouette_original"] for row in metric_rows if row["algorithm"] == "umap" and row["view"] == "joint_roles" and row["reducer_seed"] == canonical_seed),
                "background_modality": next(row["silhouette_original"] for row in metric_rows if row["algorithm"] == "umap" and row["view"] == "background" and row["reducer_seed"] == canonical_seed),
                "structural_modality": next(row["silhouette_original"] for row in metric_rows if row["algorithm"] == "umap" and row["view"] == "structural" and row["reducer_seed"] == canonical_seed),
            },
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return metadata


def _parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not seeds:
        raise argparse.ArgumentTypeError("At least one reducer seed is required.")
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", default="docs/paper/AAAI2027/figures/condition_embedding_clusters")
    parser.add_argument("--cache", default="logs/bg_pdr_fm/aaai27/diagnostics/condition_embedding_clusters/embedding_cache.npz")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--per-subset", type=int, default=128)
    parser.add_argument("--sample-seed", type=int, default=DEFAULT_SAMPLE_SEED)
    parser.add_argument("--reducer-seeds", type=_parse_seeds, default=DEFAULT_REDUCER_SEEDS)
    parser.add_argument("--force-extract", action="store_true")
    args = parser.parse_args()
    generate_assets(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        cache_path=args.cache,
        device=args.device,
        batch_size=args.batch_size,
        per_subset=args.per_subset,
        sample_seed=args.sample_seed,
        reducer_seeds=args.reducer_seeds,
        force_extract=args.force_extract,
    )


if __name__ == "__main__":
    main()
