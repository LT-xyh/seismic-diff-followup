"""Post-training contrastive diagnostics for BG-PDR-FM."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from bg_pdr_fm.data import STANDARD_MODALITIES, batch_to_device, collate_bg_samples, validate_bg_batch
from bg_pdr_fm.evaluation.evaluate_bg_pdr_fm import CONTRASTIVE_PREFIXES, _load_prefixed_checkpoint
from bg_pdr_fm.lightning import BGPDRFMLightning
from bg_pdr_fm.runtime import configure_torch_runtime, configured_torch_device
from bg_pdr_fm.training.train_bg_pdr_fm import build_dataset


DEFAULT_CONFIG = "bg_pdr_fm/configs/openfwi_lmdb_contrastive.yaml"
DEFAULT_CHECKPOINT = "logs/bg_pdr_fm/contrastive_train/lightning/checkpoints/contrastive-epoch_22-loss8.6160.ckpt"
DEFAULT_METRICS_CSV = "logs/bg_pdr_fm/contrastive_train/lightning/csv/contrastive_20260525_134208/metrics.csv"
DEFAULT_OUTPUT_DIR = (
    "logs/bg_pdr_fm/contrastive_train/diagnostics/"
    "contrastive_20260525_134208/best_val_epoch22_test300"
)
SPACES = ("structural", "numerical")
ANCHOR_BY_SPACE = {"structural": "anchor_struct", "numerical": "anchor_num"}
ANCHOR_LABEL_BY_SPACE = {"structural": "anchor_struct", "numerical": "anchor_num"}


@dataclass
class CollectedContrastiveData:
    embeddings: dict[str, dict[str, np.ndarray]]
    anchor_embeddings: dict[str, np.ndarray]
    reliability: np.ndarray
    availability: np.ndarray
    dataset_ids: np.ndarray
    sample_indices: np.ndarray
    modality_names: list[str]
    sample_images: dict[str, np.ndarray]
    feature_panel: dict[str, np.ndarray]
    anchor_panel: dict[str, np.ndarray]
    energy_rows: list[dict[str, float]]


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _load_metric_rows(metrics_csv: Path) -> list[dict[str, Any]]:
    with metrics_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows: list[dict[str, Any]] = []
        for row in reader:
            parsed: dict[str, Any] = {}
            for key, value in row.items():
                if key in {"epoch", "step"}:
                    number = _read_float(value)
                    parsed[key] = int(number) if number is not None else None
                else:
                    parsed[key] = _read_float(value)
            rows.append(parsed)
    return rows


def _metric_series(rows: list[dict[str, Any]], key: str) -> tuple[np.ndarray, np.ndarray]:
    xs: list[int] = []
    ys: list[float] = []
    for row in rows:
        value = row.get(key)
        epoch = row.get("epoch")
        if value is None or epoch is None:
            continue
        xs.append(int(epoch))
        ys.append(float(value))
    return np.asarray(xs, dtype=np.int64), np.asarray(ys, dtype=np.float64)


def _plot_train_val_metric(
    rows: list[dict[str, Any]],
    metric: str,
    output_path: Path,
    title: str,
    ylabel: str | None = None,
) -> bool:
    train_x, train_y = _metric_series(rows, f"train/{metric}")
    val_x, val_y = _metric_series(rows, f"val/{metric}")
    if train_y.size == 0 and val_y.size == 0:
        return False
    _ensure_dir(output_path.parent)
    fig, ax = plt.subplots(figsize=(7, 4))
    if train_y.size:
        ax.plot(train_x, train_y, label="train", linewidth=1.8)
    if val_y.size:
        ax.plot(val_x, val_y, label="val", linewidth=1.8)
    ax.set_title(title)
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylabel or metric)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return True


def _plot_grouped_reliability_curves(rows: list[dict[str, Any]], output_path: Path) -> bool:
    _ensure_dir(output_path.parent)
    fig, axes = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    plotted = False
    for ax, prefix in zip(axes, ("train", "val")):
        for modality in STANDARD_MODALITIES:
            for space, style in (("structural", "-"), ("numerical", "--")):
                key = f"{prefix}/reliability_{modality}_{space}"
                x, y = _metric_series(rows, key)
                if y.size == 0:
                    continue
                plotted = True
                ax.plot(x, y, style, linewidth=1.4, label=f"{modality} {space[:3]}")
        ax.set_title(prefix)
        ax.set_ylabel("reliability")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.25)
        ax.legend(ncol=2, fontsize=8)
    axes[-1].set_xlabel("epoch")
    fig.suptitle("Reliability Curves")
    fig.tight_layout()
    if plotted:
        fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return plotted


def write_training_curves(metrics_csv: Path, output_dir: Path) -> list[str]:
    rows = _load_metric_rows(metrics_csv)
    curve_dir = _ensure_dir(output_dir / "curves")
    specs = (
        ("loss", "Loss", "loss.png"),
        ("symile_structural", "Symile Structural", "symile_structural.png"),
        ("symile_numerical", "Symile Numerical", "symile_numerical.png"),
        ("pairwise_structural", "Pairwise Structural", "pairwise_structural.png"),
        ("pairwise_numerical", "Pairwise Numerical", "pairwise_numerical.png"),
        ("pairwise_retrieval_top1", "Pairwise Retrieval Top-1", "pairwise_retrieval_top1.png"),
        ("pairwise_retrieval_top5", "Pairwise Retrieval Top-5", "pairwise_retrieval_top5.png"),
        (
            "pairwise_retrieval_top5_numerical_well_log_rms_vel",
            "Well Log/RMS Numerical Top-5",
            "pairwise_retrieval_top5_numerical_well_log_rms_vel.png",
        ),
        (
            "pairwise_alignment_numerical_well_log_rms_vel",
            "Well Log/RMS Numerical Alignment",
            "pairwise_alignment_numerical_well_log_rms_vel.png",
        ),
        ("anchor", "Anchor Loss", "anchor.png"),
        ("structural_anchor_alignment", "Structural Anchor Alignment", "structural_anchor_alignment.png"),
        ("numerical_anchor_alignment", "Numerical Anchor Alignment", "numerical_anchor_alignment.png"),
        ("sn_ortho", "S/N Orthogonality", "sn_ortho.png"),
        ("unique_ortho", "Unique Orthogonality", "unique_ortho.png"),
    )
    files: list[str] = []
    for metric, title, filename in specs:
        path = curve_dir / filename
        if _plot_train_val_metric(rows, metric, path, title):
            files.append(str(path))
    rel_path = curve_dir / "reliability.png"
    if _plot_grouped_reliability_curves(rows, rel_path):
        files.append(str(rel_path))
    return files


def _to_image(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    while arr.ndim > 2:
        arr = arr[0]
    return arr


def _imshow(ax: plt.Axes, image: np.ndarray, title: str, cmap: str = "viridis") -> None:
    ax.imshow(_to_image(image), cmap=cmap)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def _save_image_grid(images: dict[str, np.ndarray], output_path: Path, cmap: str = "viridis", columns: int = 4) -> None:
    _ensure_dir(output_path.parent)
    names = list(images.keys())
    rows = math.ceil(len(names) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(3.2 * columns, 3.0 * rows))
    axes_arr = np.asarray(axes).reshape(-1)
    for ax, name in zip(axes_arr, names):
        _imshow(ax, images[name], name, cmap=cmap)
    for ax in axes_arr[len(names):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _batch_image_array(x: torch.Tensor) -> np.ndarray:
    arr = x.detach().float().cpu().numpy()
    if arr.ndim == 4:
        return arr[:, 0]
    if arr.ndim == 3:
        return arr
    raise ValueError(f"Expected BCHW or BHW image tensor, got shape {arr.shape}.")


def _cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sum(a * b, axis=1)


def _feature_low_high_ratio(model: BGPDRFMLightning, x: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    low = model.filter.lowpass(x.detach())
    high = model.filter.highpass(x.detach())
    low_energy = low.square().flatten(1).mean(dim=1)
    high_energy = high.square().flatten(1).mean(dim=1)
    total = (low_energy + high_energy).clamp_min(1e-8)
    return (low_energy / total).cpu().numpy(), (high_energy / total).cpu().numpy()


def _build_loader(conf: Any, split: str, batch_size: int) -> DataLoader:
    dataset = build_dataset(conf, split)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=collate_bg_samples)
    first_batch = next(iter(loader))
    validate_bg_batch(first_batch, context="contrastive visualization first batch")
    return loader


def _load_model(conf: Any, checkpoint: Path, device: torch.device) -> tuple[BGPDRFMLightning, dict[str, Any]]:
    OmegaConf.update(conf, "training.stage", "contrastive", merge=True)
    model = BGPDRFMLightning(conf)
    loaded = _load_prefixed_checkpoint(model, checkpoint, CONTRASTIVE_PREFIXES, "contrastive")
    model.eval().to(device)
    return model, loaded


def _extract_embeddings(
    model: BGPDRFMLightning,
    features: Any,
    anchors: Any,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, np.ndarray]]:
    if features.all_heads is None:
        raise RuntimeError("Contrastive visualization requires per-modality heads.")
    embeddings: dict[str, dict[str, np.ndarray]] = {space: {} for space in SPACES}
    anchor_embeddings: dict[str, np.ndarray] = {}
    for space in SPACES:
        for modality in STANDARD_MODALITIES:
            projected = model.contrastive_loss._project(modality, features.all_heads[modality][space])
            embeddings[space][modality] = projected.detach().float().cpu().numpy()
        anchor_name = ANCHOR_BY_SPACE[space]
        anchor_tensor = anchors.anchor_struct if space == "structural" else anchors.anchor_num
        projected_anchor = model.contrastive_loss._project(anchor_name, anchor_tensor)
        anchor_embeddings[space] = projected_anchor.detach().float().cpu().numpy()
    return embeddings, anchor_embeddings


def collect_contrastive_data(
    conf: Any,
    checkpoint: Path,
    split: str,
    max_samples: int,
    batch_size: int,
) -> tuple[CollectedContrastiveData, dict[str, Any], torch.device]:
    configure_torch_runtime(OmegaConf.select(conf, "training.matmul_precision", default=None))
    device = configured_torch_device(
        OmegaConf.select(conf, "training.accelerator", default="auto"),
        OmegaConf.select(conf, "training.devices", default="auto"),
    )
    loader = _build_loader(conf, split, batch_size)
    model, loaded = _load_model(conf, checkpoint, device)

    embeddings: dict[str, dict[str, list[np.ndarray]]] = {
        space: {modality: [] for modality in STANDARD_MODALITIES} for space in SPACES
    }
    anchor_embeddings: dict[str, list[np.ndarray]] = {space: [] for space in SPACES}
    reliability_rows: list[np.ndarray] = []
    availability_rows: list[np.ndarray] = []
    dataset_ids: list[np.ndarray] = []
    sample_indices: list[np.ndarray] = []
    sample_image_parts: dict[str, list[np.ndarray]] = {
        "depth_vel": [],
        "migrated_image": [],
        "horizon": [],
        "rms_vel": [],
        "well_log": [],
        "N": [],
        "S": [],
        "U": [],
        "anchor_struct": [],
        "anchor_num": [],
    }
    feature_panel: dict[str, np.ndarray] = {}
    anchor_panel: dict[str, np.ndarray] = {}
    energy_rows: list[dict[str, float]] = []

    sample_count = 0
    with torch.no_grad():
        for batch in loader:
            if sample_count >= max_samples:
                break
            batch = batch_to_device(BGPDRFMLightning._as_batch(batch), device)
            if sample_count + batch.depth_vel.shape[0] > max_samples:
                keep = max_samples - sample_count
                batch = _slice_batch(batch, keep)
            features = model.encoder(batch, return_all_heads=True)
            anchors = model.wavelet_anchor(batch.depth_vel)
            batch_embeddings, batch_anchor_embeddings = _extract_embeddings(model, features, anchors)

            availability = (batch.modality_mask * batch.modality_quality).gt(0).float().detach().cpu().numpy()
            reliability_rows.append(features.reliability.detach().float().cpu().numpy())
            availability_rows.append(availability)
            for space in SPACES:
                for modality in STANDARD_MODALITIES:
                    embeddings[space][modality].append(batch_embeddings[space][modality])
                anchor_embeddings[space].append(batch_anchor_embeddings[space])

            metadata = batch.metadata
            dataset_ids.append(metadata.get("dataset_id", torch.full((batch.depth_vel.shape[0],), -1, device=device)).cpu().numpy())
            sample_indices.append(
                metadata.get("sample_index", torch.arange(batch.depth_vel.shape[0], device=device)).cpu().numpy()
            )

            n_low, n_high = _feature_low_high_ratio(model, features.numerical)
            s_low, s_high = _feature_low_high_ratio(model, features.structural)
            u_low, u_high = _feature_low_high_ratio(model, features.unique)
            sample_image_parts["depth_vel"].append(_batch_image_array(batch.depth_vel))
            sample_image_parts["migrated_image"].append(_batch_image_array(batch.migrated_image))
            sample_image_parts["horizon"].append(_batch_image_array(batch.horizon))
            sample_image_parts["rms_vel"].append(_batch_image_array(batch.rms_vel))
            sample_image_parts["well_log"].append(_batch_image_array(batch.well_log))
            sample_image_parts["N"].append(_batch_image_array(features.numerical[:, :1]))
            sample_image_parts["S"].append(_batch_image_array(features.structural[:, :1]))
            sample_image_parts["U"].append(_batch_image_array(features.unique[:, :1]))
            sample_image_parts["anchor_struct"].append(_batch_image_array(anchors.anchor_struct))
            sample_image_parts["anchor_num"].append(_batch_image_array(anchors.anchor_num))
            for item_idx in range(batch.depth_vel.shape[0]):
                energy_rows.append(
                    {
                        "numerical_low_energy_ratio": float(n_low[item_idx]),
                        "numerical_high_energy_ratio": float(n_high[item_idx]),
                        "structural_low_energy_ratio": float(s_low[item_idx]),
                        "structural_high_energy_ratio": float(s_high[item_idx]),
                        "unique_low_energy_ratio": float(u_low[item_idx]),
                        "unique_high_energy_ratio": float(u_high[item_idx]),
                    }
                )

            if not feature_panel:
                feature_panel = {
                    "depth_vel": batch.depth_vel[:1].detach().float().cpu().numpy(),
                    "migrated_image": batch.migrated_image[:1].detach().float().cpu().numpy(),
                    "horizon": batch.horizon[:1].detach().float().cpu().numpy(),
                    "rms_vel": batch.rms_vel[:1].detach().float().cpu().numpy(),
                    "well_log": batch.well_log[:1].detach().float().cpu().numpy(),
                    "N": features.numerical[:1, :1].detach().float().cpu().numpy(),
                    "S": features.structural[:1, :1].detach().float().cpu().numpy(),
                    "U": features.unique[:1, :1].detach().float().cpu().numpy(),
                }
                anchor_panel = {
                    "depth_vel": batch.depth_vel[:1].detach().float().cpu().numpy(),
                    "anchor_struct": anchors.anchor_struct[:1].detach().float().cpu().numpy(),
                    "anchor_num": anchors.anchor_num[:1].detach().float().cpu().numpy(),
                    "N": features.numerical[:1, :1].detach().float().cpu().numpy(),
                    "S": features.structural[:1, :1].detach().float().cpu().numpy(),
                }
            sample_count += int(batch.depth_vel.shape[0])

    final_embeddings = {
        space: {modality: np.concatenate(parts, axis=0) for modality, parts in by_modality.items()}
        for space, by_modality in embeddings.items()
    }
    final_anchor_embeddings = {space: np.concatenate(parts, axis=0) for space, parts in anchor_embeddings.items()}
    data = CollectedContrastiveData(
        embeddings=final_embeddings,
        anchor_embeddings=final_anchor_embeddings,
        reliability=np.concatenate(reliability_rows, axis=0),
        availability=np.concatenate(availability_rows, axis=0),
        dataset_ids=np.concatenate(dataset_ids, axis=0).astype(np.int64),
        sample_indices=np.concatenate(sample_indices, axis=0).astype(np.int64),
        modality_names=list(STANDARD_MODALITIES),
        sample_images={key: np.concatenate(parts, axis=0) for key, parts in sample_image_parts.items()},
        feature_panel=feature_panel,
        anchor_panel=anchor_panel,
        energy_rows=energy_rows,
    )
    return data, loaded, device


def _slice_batch(batch: Any, keep: int) -> Any:
    return type(batch)(
        depth_vel=batch.depth_vel[:keep],
        migrated_image=batch.migrated_image[:keep],
        horizon=batch.horizon[:keep],
        rms_vel=batch.rms_vel[:keep],
        well_log=batch.well_log[:keep],
        well_mask=batch.well_mask[:keep],
        modality_mask=batch.modality_mask[:keep],
        modality_quality=batch.modality_quality[:keep],
        metadata={key: value[:keep] for key, value in batch.metadata.items()},
    )


def _save_sample_metrics(data: CollectedContrastiveData, output_path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    for i in range(data.reliability.shape[0]):
        row: dict[str, float] = {
            "sample_row": float(i),
            "dataset_id": float(data.dataset_ids[i]),
            "sample_index": float(data.sample_indices[i]),
        }
        for modality_idx, modality in enumerate(data.modality_names):
            row[f"availability_{modality}"] = float(data.availability[i, modality_idx])
            row[f"reliability_{modality}_structural"] = float(data.reliability[i, modality_idx, 0])
            row[f"reliability_{modality}_numerical"] = float(data.reliability[i, modality_idx, 1])
            row[f"alignment_{modality}_structural"] = float(
                np.dot(data.embeddings["structural"][modality][i], data.anchor_embeddings["structural"][i])
            )
            row[f"alignment_{modality}_numerical"] = float(
                np.dot(data.embeddings["numerical"][modality][i], data.anchor_embeddings["numerical"][i])
            )
        row["structural_alignment_mean"] = float(
            _masked_row_mean(
                np.stack(
                    [
                        _cosine_rows(data.embeddings["structural"][m], data.anchor_embeddings["structural"])
                        for m in data.modality_names
                    ],
                    axis=1,
                )[i],
                data.availability[i],
            )
        )
        row["numerical_alignment_mean"] = float(
            _masked_row_mean(
                np.stack(
                    [
                        _cosine_rows(data.embeddings["numerical"][m], data.anchor_embeddings["numerical"])
                        for m in data.modality_names
                    ],
                    axis=1,
                )[i],
                data.availability[i],
            )
        )
        rows.append(row)

    _ensure_dir(output_path.parent)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _masked_row_mean(values: np.ndarray, mask: np.ndarray) -> float:
    observed = mask > 0.5
    if not observed.any():
        return float("nan")
    return float(values[observed].mean())


def _plot_embedding_projection(data: CollectedContrastiveData, output_dir: Path, method: str, space: str) -> str | None:
    labels: list[str] = []
    arrays: list[np.ndarray] = []
    for modality in data.modality_names:
        arrays.append(data.embeddings[space][modality])
        labels.extend([modality] * data.embeddings[space][modality].shape[0])
    arrays.append(data.anchor_embeddings[space])
    labels.extend([ANCHOR_LABEL_BY_SPACE[space]] * data.anchor_embeddings[space].shape[0])
    x = np.concatenate(arrays, axis=0)
    if x.shape[0] < 3:
        return None
    if method == "pca":
        coords = PCA(n_components=2, random_state=42).fit_transform(x)
    elif method == "tsne":
        perplexity = max(2, min(30, (x.shape[0] - 1) // 3))
        coords = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto", random_state=42).fit_transform(x)
    else:
        raise ValueError(f"Unknown projection method: {method}")

    output_path = _ensure_dir(output_dir / "embeddings") / f"{space}_{method}.png"
    fig, ax = plt.subplots(figsize=(7, 6))
    unique_labels = list(dict.fromkeys(labels))
    for label in unique_labels:
        idx = np.asarray([item == label for item in labels])
        marker = "*" if label.startswith("anchor_") else "o"
        ax.scatter(coords[idx, 0], coords[idx, 1], s=16 if marker == "o" else 44, alpha=0.75, label=label, marker=marker)
    ax.set_title(f"{space.title()} {method.upper()} Projection")
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return str(output_path)


def _similarity_matrix(data: CollectedContrastiveData, space: str) -> tuple[list[str], np.ndarray]:
    names = [*data.modality_names, ANCHOR_LABEL_BY_SPACE[space]]
    vectors = [data.embeddings[space][name] for name in data.modality_names]
    vectors.append(data.anchor_embeddings[space])
    matrix = np.zeros((len(names), len(names)), dtype=np.float32)
    for i, a in enumerate(vectors):
        for j, b in enumerate(vectors):
            mask = _item_mask(data, names[i]) & _item_mask(data, names[j])
            matrix[i, j] = float(np.mean(np.sum(a[mask] * b[mask], axis=1))) if mask.any() else float("nan")
    return names, matrix


def _item_mask(data: CollectedContrastiveData, item_name: str) -> np.ndarray:
    if item_name.startswith("anchor_"):
        return np.ones(data.availability.shape[0], dtype=bool)
    modality_idx = data.modality_names.index(item_name)
    return data.availability[:, modality_idx] > 0.5


def _plot_heatmap(names: list[str], matrix: np.ndarray, output_path: Path, title: str) -> None:
    _ensure_dir(output_path.parent)
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(names)), labels=names, rotation=40, ha="right")
    ax.set_yticks(np.arange(len(names)), labels=names)
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_similarity_distributions(data: CollectedContrastiveData, space: str, output_path: Path) -> None:
    positives: list[np.ndarray] = []
    negatives: list[np.ndarray] = []
    anchor = data.anchor_embeddings[space]
    for modality_idx, modality in enumerate(data.modality_names):
        emb = data.embeddings[space][modality]
        observed = data.availability[:, modality_idx] > 0.5
        if observed.any():
            positives.append(_cosine_rows(emb[observed], anchor[observed]))
            shuffled = np.roll(anchor, 1, axis=0)
            negatives.append(_cosine_rows(emb[observed], shuffled[observed]))
    pos = np.concatenate(positives) if positives else np.asarray([], dtype=np.float32)
    neg = np.concatenate(negatives) if negatives else np.asarray([], dtype=np.float32)
    _ensure_dir(output_path.parent)
    fig, ax = plt.subplots(figsize=(6, 4))
    if pos.size:
        ax.hist(pos, bins=30, alpha=0.65, label="positive", density=True)
    if neg.size:
        ax.hist(neg, bins=30, alpha=0.65, label="shuffled negative", density=True)
    ax.set_title(f"{space.title()} Positive vs Negative Similarity")
    ax.set_xlabel("cosine similarity")
    ax.set_ylabel("density")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _retrieval_metrics(data: CollectedContrastiveData, space: str, output_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    targets = {**{modality: data.embeddings[space][modality] for modality in data.modality_names}, ANCHOR_LABEL_BY_SPACE[space]: data.anchor_embeddings[space]}
    queries = targets
    for query_name, query_vectors in queries.items():
        for target_name, target_vectors in targets.items():
            if query_name == target_name:
                continue
            valid = _item_mask(data, query_name) & _item_mask(data, target_name)
            if not valid.any():
                continue
            query_subset = query_vectors[valid]
            target_subset = target_vectors[valid]
            n = query_subset.shape[0]
            scores = query_subset @ target_subset.T
            ranks = np.argsort(-scores, axis=1)
            truth = np.arange(n)[:, None]
            top1 = float((ranks[:, :1] == truth).any(axis=1).mean())
            top5 = float((ranks[:, : min(5, n)] == truth).any(axis=1).mean())
            rows.append(
                {
                    "space": space,
                    "query": query_name,
                    "target": target_name,
                    "top1": top1,
                    "top5": top5,
                    "num_queries": n,
                }
            )
    _ensure_dir(output_path.parent)
    mode = "a" if output_path.exists() else "w"
    with output_path.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["space", "query", "target", "top1", "top5", "num_queries"])
        if mode == "w":
            writer.writeheader()
        writer.writerows(rows)
    return rows


def _plot_retrieval_examples(data: CollectedContrastiveData, space: str, output_path: Path, top_k: int = 5) -> None:
    query_name = "migrated_image" if space == "structural" else "rms_vel"
    target_name = ANCHOR_LABEL_BY_SPACE[space]
    query_vectors = data.embeddings[space][query_name]
    target_vectors = data.anchor_embeddings[space]
    scores = query_vectors @ target_vectors.T
    valid_query = np.nonzero(_item_mask(data, query_name))[0]
    if valid_query.size == 0:
        return
    query_indices = valid_query[np.linspace(0, valid_query.size - 1, num=min(4, valid_query.size), dtype=int)]
    _ensure_dir(output_path.parent)
    fig, axes = plt.subplots(len(query_indices), top_k + 1, figsize=(2.2 * (top_k + 1), 2.3 * len(query_indices)))
    axes_arr = np.asarray(axes)
    if axes_arr.ndim == 1:
        axes_arr = axes_arr[None, :]
    for row_idx, sample_idx in enumerate(query_indices):
        _imshow(axes_arr[row_idx, 0], data.sample_images[query_name][sample_idx], f"{query_name}\nrow {sample_idx}")
        top = np.argsort(-scores[sample_idx])[:top_k]
        for col_idx, target_idx in enumerate(top, start=1):
            is_true = target_idx == sample_idx
            title = f"{target_name}\nrow {target_idx}\n{scores[sample_idx, target_idx]:.2f}"
            if is_true:
                title += "\nmatch"
            _imshow(axes_arr[row_idx, col_idx], data.sample_images["depth_vel"][target_idx], title, cmap="jet")
    fig.suptitle(f"{space.title()} Retrieval Examples")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_reliability_bars(data: CollectedContrastiveData, output_path: Path) -> None:
    means = data.reliability.mean(axis=0)
    x = np.arange(len(data.modality_names))
    width = 0.38
    _ensure_dir(output_path.parent)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width / 2, means[:, 0], width, label="structural")
    ax.bar(x + width / 2, means[:, 1], width, label="numerical")
    ax.set_xticks(x, labels=data.modality_names, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("mean reliability")
    ax.set_title("Mean Reliability by Modality")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_reliability_heatmap(data: CollectedContrastiveData, output_path: Path) -> None:
    labels = [f"{m[:4]}_S" for m in data.modality_names] + [f"{m[:4]}_N" for m in data.modality_names]
    values = np.concatenate([data.reliability[:, :, 0], data.reliability[:, :, 1]], axis=1)
    _ensure_dir(output_path.parent)
    fig, ax = plt.subplots(figsize=(8, 5))
    image = ax.imshow(values, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(labels)), labels=labels, rotation=35, ha="right")
    ax.set_ylabel("sample row")
    ax.set_title("Reliability Heatmap")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_reliability_alignment_scatter(data: CollectedContrastiveData, output_path: Path) -> None:
    _ensure_dir(output_path.parent)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, space, rel_idx in zip(axes, SPACES, (0, 1)):
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        anchor = data.anchor_embeddings[space]
        for modality_idx, modality in enumerate(data.modality_names):
            observed = data.availability[:, modality_idx] > 0.5
            if not observed.any():
                continue
            xs.append(data.reliability[observed, modality_idx, rel_idx])
            ys.append(_cosine_rows(data.embeddings[space][modality][observed], anchor[observed]))
        if xs:
            ax.scatter(np.concatenate(xs), np.concatenate(ys), s=16, alpha=0.65)
        ax.set_title(space)
        ax.set_xlabel("reliability")
        ax.set_ylabel("anchor cosine")
        ax.grid(True, alpha=0.25)
    fig.suptitle("Reliability vs Anchor Alignment")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_energy_ratios(data: CollectedContrastiveData, output_path: Path) -> None:
    keys = [
        "numerical_low_energy_ratio",
        "numerical_high_energy_ratio",
        "structural_low_energy_ratio",
        "structural_high_energy_ratio",
        "unique_low_energy_ratio",
        "unique_high_energy_ratio",
    ]
    means = [np.mean([row[key] for row in data.energy_rows]) for key in keys]
    _ensure_dir(output_path.parent)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(np.arange(len(keys)), means)
    ax.set_xticks(np.arange(len(keys)), labels=[key.replace("_energy_ratio", "") for key in keys], rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("mean ratio")
    ax.set_title("Feature Low/High Energy Ratios")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _write_energy_csv(data: CollectedContrastiveData, output_path: Path) -> None:
    _ensure_dir(output_path.parent)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(data.energy_rows[0].keys()) if data.energy_rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data.energy_rows)


def _failure_indices(sample_rows: list[dict[str, float]], key: str, count: int = 8) -> list[int]:
    rows = sorted(sample_rows, key=lambda row: row.get(key, float("inf")))
    return [int(row["sample_row"]) for row in rows[:count]]


def _disagreement_indices(sample_rows: list[dict[str, float]], count: int = 8) -> list[int]:
    rows = sorted(
        sample_rows,
        key=lambda row: abs(
            np.nanmean([row[f"reliability_{m}_structural"] for m in STANDARD_MODALITIES])
            - row.get("structural_alignment_mean", float("nan"))
        ),
        reverse=True,
    )
    return [int(row["sample_row"]) for row in rows[:count]]


def _plot_failure_table(sample_rows: list[dict[str, float]], output_path: Path) -> None:
    structural = _failure_indices(sample_rows, "structural_alignment_mean")
    numerical = _failure_indices(sample_rows, "numerical_alignment_mean")
    disagreement = _disagreement_indices(sample_rows)
    lines = [
        "Structural low alignment: " + ", ".join(map(str, structural)),
        "Numerical low alignment: " + ", ".join(map(str, numerical)),
        "Reliability/alignment disagreement: " + ", ".join(map(str, disagreement)),
    ]
    _ensure_dir(output_path.parent)
    fig, ax = plt.subplots(figsize=(9, 2.5))
    ax.text(0.02, 0.75, "\n".join(lines), ha="left", va="top", fontsize=11)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_case_panel(data: CollectedContrastiveData, indices: list[int], output_path: Path, title: str) -> None:
    indices = indices[:8]
    columns = ["depth_vel", "migrated_image", "horizon", "rms_vel", "well_log", "anchor_struct", "anchor_num", "N", "S", "U"]
    _ensure_dir(output_path.parent)
    fig, axes = plt.subplots(len(indices), len(columns), figsize=(2.1 * len(columns), 2.1 * len(indices)))
    axes_arr = np.asarray(axes)
    if axes_arr.ndim == 1:
        axes_arr = axes_arr[None, :]
    for row_idx, sample_idx in enumerate(indices):
        for col_idx, name in enumerate(columns):
            ax = axes_arr[row_idx, col_idx]
            cmap = "jet" if name in {"depth_vel", "rms_vel"} else "viridis"
            ax.imshow(data.sample_images[name][sample_idx], cmap=cmap)
            if row_idx == 0:
                ax.set_title(name, fontsize=8)
            if col_idx == 0:
                ax.set_ylabel(f"row {sample_idx}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_diagnostic_plots(data: CollectedContrastiveData, output_dir: Path) -> tuple[list[str], list[dict[str, Any]]]:
    files: list[str] = []
    retrieval_rows: list[dict[str, Any]] = []
    sample_rows = _save_sample_metrics(data, output_dir / "sample_metrics.csv")
    retrieval_csv = output_dir / "retrieval" / "retrieval_metrics.csv"
    if retrieval_csv.exists():
        retrieval_csv.unlink()

    _save_image_grid(data.feature_panel, output_dir / "features" / "feature_panel.png", cmap="viridis", columns=4)
    files.append(str(output_dir / "features" / "feature_panel.png"))
    _save_image_grid(data.anchor_panel, output_dir / "features" / "anchor_alignment_panel.png", cmap="viridis", columns=5)
    files.append(str(output_dir / "features" / "anchor_alignment_panel.png"))
    _plot_energy_ratios(data, output_dir / "features" / "feature_energy_ratios.png")
    files.append(str(output_dir / "features" / "feature_energy_ratios.png"))
    _write_energy_csv(data, output_dir / "features" / "feature_energy_ratios.csv")

    for space in SPACES:
        for method in ("pca", "tsne"):
            path = _plot_embedding_projection(data, output_dir, method, space)
            if path is not None:
                files.append(path)
        names, matrix = _similarity_matrix(data, space)
        heatmap_path = output_dir / "similarity" / f"{space}_cosine_heatmap.png"
        _plot_heatmap(names, matrix, heatmap_path, f"{space.title()} Mean Cosine Similarity")
        files.append(str(heatmap_path))
        dist_path = output_dir / "similarity" / f"{space}_positive_negative_similarity.png"
        _plot_similarity_distributions(data, space, dist_path)
        files.append(str(dist_path))
        retrieval_rows.extend(_retrieval_metrics(data, space, retrieval_csv))
        retrieval_path = output_dir / "retrieval" / f"{space}_retrieval_examples.png"
        _plot_retrieval_examples(data, space, retrieval_path)
        files.append(str(retrieval_path))

    _plot_reliability_bars(data, output_dir / "reliability" / "mean_reliability.png")
    files.append(str(output_dir / "reliability" / "mean_reliability.png"))
    _plot_reliability_heatmap(data, output_dir / "reliability" / "reliability_heatmap.png")
    files.append(str(output_dir / "reliability" / "reliability_heatmap.png"))
    _plot_reliability_alignment_scatter(data, output_dir / "reliability" / "reliability_vs_alignment.png")
    files.append(str(output_dir / "reliability" / "reliability_vs_alignment.png"))
    _plot_failure_table(sample_rows, output_dir / "features" / "failure_case_indices.png")
    files.append(str(output_dir / "features" / "failure_case_indices.png"))
    failure_panels = (
        (
            _failure_indices(sample_rows, "structural_alignment_mean"),
            output_dir / "features" / "failure_structural_alignment.png",
            "Lowest Structural Alignment Cases",
        ),
        (
            _failure_indices(sample_rows, "numerical_alignment_mean"),
            output_dir / "features" / "failure_numerical_alignment.png",
            "Lowest Numerical Alignment Cases",
        ),
        (
            _disagreement_indices(sample_rows),
            output_dir / "features" / "failure_reliability_alignment_disagreement.png",
            "Reliability/Alignment Disagreement Cases",
        ),
    )
    for indices, path, title in failure_panels:
        _plot_case_panel(data, indices, path, title)
        files.append(str(path))
    return files, retrieval_rows


def _summary_means(data: CollectedContrastiveData, retrieval_rows: list[dict[str, Any]]) -> dict[str, Any]:
    reliability = {}
    for modality_idx, modality in enumerate(data.modality_names):
        reliability[f"{modality}_structural"] = float(data.reliability[:, modality_idx, 0].mean())
        reliability[f"{modality}_numerical"] = float(data.reliability[:, modality_idx, 1].mean())
    return {
        "reliability": reliability,
        "retrieval_top1_mean": float(np.mean([row["top1"] for row in retrieval_rows])) if retrieval_rows else float("nan"),
        "retrieval_top5_mean": float(np.mean([row["top5"] for row in retrieval_rows])) if retrieval_rows else float("nan"),
    }


def run_visualization(
    *,
    config: str | Path = DEFAULT_CONFIG,
    checkpoint: str | Path = DEFAULT_CHECKPOINT,
    metrics_csv: str | Path = DEFAULT_METRICS_CSV,
    split: str = "test",
    max_samples: int = 300,
    batch_size: int = 32,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
) -> dict[str, Any]:
    config_path = Path(config)
    checkpoint_path = Path(checkpoint)
    metrics_path = Path(metrics_csv)
    output_path = _ensure_dir(Path(output_dir))
    conf = OmegaConf.load(config_path)
    OmegaConf.update(conf, "evaluation.split", split, merge=True)
    OmegaConf.update(conf, "training.batch_size", int(batch_size), merge=True)

    plot_files: list[str] = []
    if metrics_path.is_file():
        plot_files.extend(write_training_curves(metrics_path, output_path))
    data, loaded_checkpoint, device = collect_contrastive_data(
        conf,
        checkpoint_path,
        split=split,
        max_samples=int(max_samples),
        batch_size=int(batch_size),
    )
    diagnostic_files, retrieval_rows = write_diagnostic_plots(data, output_path)
    plot_files.extend(diagnostic_files)
    summary = {
        "config": str(config_path),
        "checkpoint": str(checkpoint_path),
        "metrics_csv": str(metrics_path),
        "split": split,
        "max_samples": int(max_samples),
        "batch_size": int(batch_size),
        "num_samples": int(data.reliability.shape[0]),
        "device": str(device),
        "loaded_checkpoint": loaded_checkpoint,
        "means": _summary_means(data, retrieval_rows),
        "plot_files": plot_files,
    }
    (output_path / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description="Generate BG-PDR-FM contrastive diagnostic figures.")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--metrics-csv", default=DEFAULT_METRICS_CSV)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-samples", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)
    summary = run_visualization(
        config=args.config,
        checkpoint=args.checkpoint,
        metrics_csv=args.metrics_csv,
        split=args.split,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    main(sys.argv[1:])
