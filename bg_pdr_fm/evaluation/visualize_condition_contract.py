"""Generate the Figure 3 condition-contract diagnostics from trained checkpoints.

The module deliberately distinguishes three representation levels:

* cross-modal retrieval and anchor calibration use modality-specific role heads;
* role separation uses the fused background/structural conditions;
* spatial probes are frozen-encoder, post-hoc visualizations only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from bg_pdr_fm.data import STANDARD_MODALITIES, batch_to_device, collate_bg_samples
from bg_pdr_fm.evaluation.evaluate_bg_pdr_fm import _load_full_checkpoint
from bg_pdr_fm.evaluation.visualize_residual_transport import validate_split_identity
from bg_pdr_fm.lightning import BGPDRFMLightning
from bg_pdr_fm.training.train_bg_pdr_fm import build_dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
ROLE_TO_HEAD = {"background": "numerical", "structural": "structural"}
ROLE_TO_ANCHOR = {"background": "anchor_num", "structural": "anchor_struct"}
FORMAL_FULL_SEEDS = (2027, 3407, 7777)
EXPECTED_OPENFWI_SUBSETS = 8
MODALITY_LABELS = {
    "migrated_image": "PoSTM",
    "horizon": "Horizon",
    "rms_vel": "RMS",
    "well_log": "Well",
}
GREEN = "#178F48"
BLUE = "#1E63D5"
PURPLE = "#8357A6"
GRAY = "#8A929B"


@dataclass
class SeedDiagnostics:
    seed: int
    checkpoint: str
    checkpoint_info: dict[str, Any]
    dataset_ids: np.ndarray
    sample_indices: np.ndarray
    availability: np.ndarray
    embeddings: dict[str, dict[str, np.ndarray]]
    anchor_embeddings: dict[str, np.ndarray]
    eta_b: np.ndarray
    eta_s: np.ndarray
    chi_bs: np.ndarray
    routing_background: np.ndarray
    routing_structural: np.ndarray
    dataset_names: list[str]


@dataclass
class ProbeWeights:
    background: np.ndarray
    structural: np.ndarray
    background_alpha: float
    structural_alpha: float
    background_validation_mse: float
    structural_validation_mse: float


def _resolve_path(path: str | Path) -> Path:
    item = Path(path)
    return item if item.is_absolute() else REPO_ROOT / item


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with _resolve_path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().float().cpu().numpy()


def strict_derangement_indices(size: int, offset: int = 1) -> np.ndarray:
    """Return a deterministic permutation without fixed points."""
    if size < 2:
        raise ValueError("A strict derangement requires at least two samples.")
    shift = int(offset) % int(size)
    if shift == 0:
        shift = 1
    return (np.arange(size, dtype=np.int64) + shift) % int(size)


def build_subset_galleries(
    dataset_ids: np.ndarray,
    all_modalities_available: np.ndarray,
    *,
    gallery_size: int,
    galleries_per_subset: int,
    seed: int,
) -> list[np.ndarray]:
    """Sample fixed-size galleries inside individual OpenFWI subsets."""
    dataset_ids = np.asarray(dataset_ids, dtype=np.int64)
    available = np.asarray(all_modalities_available, dtype=bool)
    if dataset_ids.ndim != 1 or available.shape != dataset_ids.shape:
        raise ValueError("dataset_ids and all_modalities_available must be aligned one-dimensional arrays.")
    if gallery_size < 2 or galleries_per_subset < 1:
        raise ValueError("gallery_size must be at least two and galleries_per_subset must be positive.")
    rng = np.random.default_rng(int(seed))
    galleries: list[np.ndarray] = []
    for dataset_id in sorted(np.unique(dataset_ids).tolist()):
        candidates = np.flatnonzero((dataset_ids == dataset_id) & available)
        if candidates.size < gallery_size:
            raise ValueError(
                f"Dataset id {dataset_id} has {candidates.size} complete-modality records; "
                f"need at least {gallery_size}."
            )
        for _ in range(galleries_per_subset):
            galleries.append(np.sort(rng.choice(candidates, size=gallery_size, replace=False)).astype(np.int64))
    return galleries


def _normalized_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)


def _recall_at_one(query: np.ndarray, key: np.ndarray) -> float:
    if query.shape != key.shape or query.ndim != 2 or query.shape[0] < 2:
        raise ValueError("Recall@1 requires aligned N x D embeddings with N >= 2.")
    scores = _normalized_rows(query) @ _normalized_rows(key).T
    return float((scores.argmax(axis=1) == np.arange(scores.shape[0])).mean())


def symmetric_recall_at_one(embeddings: Mapping[str, np.ndarray], gallery_indices: np.ndarray) -> np.ndarray:
    """Return a symmetric off-diagonal Recall@1 matrix for one gallery."""
    names = list(embeddings)
    matrix = np.full((len(names), len(names)), np.nan, dtype=np.float64)
    gallery = np.asarray(gallery_indices, dtype=np.int64)
    for left_idx, left_name in enumerate(names):
        for right_idx in range(left_idx + 1, len(names)):
            right_name = names[right_idx]
            left = np.asarray(embeddings[left_name])[gallery]
            right = np.asarray(embeddings[right_name])[gallery]
            score = 0.5 * (_recall_at_one(left, right) + _recall_at_one(right, left))
            matrix[left_idx, right_idx] = score
            matrix[right_idx, left_idx] = score
    return matrix


def _role_separation_per_sample(
    background: torch.Tensor,
    structural: torch.Tensor,
    *,
    kernel_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if background.shape != structural.shape or background.ndim != 4:
        raise ValueError("Background and structural conditions must be aligned BCHW tensors.")
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer.")
    low_b = F.avg_pool2d(background, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    low_s = F.avg_pool2d(structural, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
    high_b = background - low_b
    high_s = structural - low_s
    low_b_mag = low_b.abs().flatten(1).mean(dim=1)
    high_b_mag = high_b.abs().flatten(1).mean(dim=1)
    low_s_mag = low_s.abs().flatten(1).mean(dim=1)
    high_s_mag = high_s.abs().flatten(1).mean(dim=1)
    eta_b = low_b_mag / (low_b_mag + high_b_mag).clamp_min(1e-8)
    eta_s = high_s_mag / (low_s_mag + high_s_mag).clamp_min(1e-8)
    b_vector = F.adaptive_avg_pool2d(background, 1).flatten(1)
    s_vector = F.adaptive_avg_pool2d(structural, 1).flatten(1)
    chi_bs = F.cosine_similarity(b_vector, s_vector, dim=1, eps=1e-8).abs()
    return eta_b, eta_s, chi_bs


def role_separation_metrics(background: torch.Tensor, structural: torch.Tensor, kernel_size: int = 5) -> dict[str, float]:
    """Compute fused-condition diagnostics used by Figure 3(c)."""
    eta_b, eta_s, chi_bs = _role_separation_per_sample(background, structural, kernel_size=kernel_size)
    return {
        "eta_b": float(eta_b.mean().detach().cpu()),
        "eta_s": float(eta_s.mean().detach().cpu()),
        "chi_bs": float(chi_bs.mean().detach().cpu()),
    }


def _prepare_model(config_path: Path, checkpoint_path: Path, device: torch.device) -> tuple[BGPDRFMLightning, DictConfig, dict[str, Any]]:
    conf = OmegaConf.load(config_path)
    _set = OmegaConf.update
    _set(conf, "training.stage", "joint_full", merge=True)
    _set(conf, "training.load_stage_checkpoint", None, merge=True)
    _set(conf, "training.joint.warm_start_checkpoints", {"contrastive": None, "background": None, "residual": None}, merge=True)
    model = BGPDRFMLightning(conf).to(device)
    info = _load_full_checkpoint(model, checkpoint_path)
    info["sha256"] = _sha256_file(checkpoint_path)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = state.get("state_dict", {}) if isinstance(state, dict) else {}
    required = ("encoder.", "contrastive_loss.projectors.")
    missing_prefixes = [prefix for prefix in required if not any(key.startswith(prefix) for key in state_dict)]
    if missing_prefixes:
        raise KeyError(f"Checkpoint {checkpoint_path} lacks required Figure 3 prefixes: {missing_prefixes}.")
    if int(info["loaded_keys"]) == 0:
        raise RuntimeError(f"Checkpoint {checkpoint_path} loaded zero compatible parameters.")
    info.update(
        {
            "epoch": None if not isinstance(state, dict) else state.get("epoch"),
            "state_dict_keys": len(state_dict),
            "required_prefixes": list(required),
            "required_prefixes_present": [
                prefix for prefix in required if any(key.startswith(prefix) for key in state_dict)
            ],
        }
    )
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, conf, info


def _loader(conf: DictConfig, split: str, batch_size: int) -> tuple[DataLoader, Any]:
    dataset = build_dataset(conf, split)
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_bg_samples,
    )
    return loader, dataset


def _slice_batch(batch: Any, count: int) -> Any:
    if count == int(batch.depth_vel.shape[0]):
        return batch
    return type(batch)(
        depth_vel=batch.depth_vel[:count],
        migrated_image=batch.migrated_image[:count],
        horizon=batch.horizon[:count],
        rms_vel=batch.rms_vel[:count],
        well_log=batch.well_log[:count],
        well_mask=batch.well_mask[:count],
        modality_mask=batch.modality_mask[:count],
        modality_quality=batch.modality_quality[:count],
        metadata={key: value[:count] for key, value in batch.metadata.items()},
    )


def _batch_record_ids(batch: Any, *, start_row: int) -> tuple[np.ndarray, np.ndarray]:
    """Return source identities, with deterministic fallbacks for synthetic smoke data."""
    batch_size = int(batch.depth_vel.shape[0])
    metadata = getattr(batch, "metadata", {})
    dataset_tensor = metadata.get("dataset_id")
    sample_tensor = metadata.get("sample_index")
    dataset_ids = (
        np.full(batch_size, -1, dtype=np.int64)
        if dataset_tensor is None
        else _to_numpy(dataset_tensor).astype(np.int64).reshape(-1)
    )
    sample_indices = (
        np.arange(start_row, start_row + batch_size, dtype=np.int64)
        if sample_tensor is None
        else _to_numpy(sample_tensor).astype(np.int64).reshape(-1)
    )
    return dataset_ids, sample_indices


@torch.no_grad()
def collect_seed_diagnostics(
    *,
    config_path: str | Path,
    checkpoint_path: str | Path,
    seed: int,
    device: str | torch.device,
    batch_size: int = 64,
    max_samples: int | None = None,
) -> SeedDiagnostics:
    """Collect model-native role-head and fused-condition diagnostics on test data."""
    config_path = _resolve_path(config_path)
    checkpoint_path = _resolve_path(checkpoint_path)
    resolved_device = torch.device(device)
    model, conf, checkpoint_info = _prepare_model(config_path, checkpoint_path, resolved_device)
    loader, _ = _loader(conf, "test", batch_size)
    dataset_names = [str(name) for name in OmegaConf.select(conf, "data.openfwi_datasets", default=[])]
    embeddings: dict[str, dict[str, list[np.ndarray]]] = {
        role: {modality: [] for modality in STANDARD_MODALITIES} for role in ROLE_TO_HEAD
    }
    anchors: dict[str, list[np.ndarray]] = {role: [] for role in ROLE_TO_HEAD}
    dataset_ids: list[np.ndarray] = []
    sample_indices: list[np.ndarray] = []
    availability: list[np.ndarray] = []
    eta_b_values: list[np.ndarray] = []
    eta_s_values: list[np.ndarray] = []
    chi_values: list[np.ndarray] = []
    routing_background_values: list[np.ndarray] = []
    routing_structural_values: list[np.ndarray] = []
    seen = 0

    for cpu_batch in loader:
        if max_samples is not None and seen >= int(max_samples):
            break
        take = int(cpu_batch.depth_vel.shape[0])
        if max_samples is not None:
            take = min(take, int(max_samples) - seen)
        cpu_batch = _slice_batch(cpu_batch, take)
        batch = batch_to_device(cpu_batch, resolved_device)
        features = model.encoder(batch, return_all_heads=True)
        if features.all_heads is None:
            raise RuntimeError("Condition-contract diagnostics require all modality role heads.")
        available = (batch.modality_mask * batch.modality_quality).gt(0)[:, model.encoder.modality_indices]
        routing_structural, routing_background = model.encoder.routing_weights(features.reliability, available)
        anchor_bundle = model.wavelet_anchor(batch.depth_vel)
        role_anchors = {
            "background": anchor_bundle.anchor_num,
            "structural": anchor_bundle.anchor_struct,
        }
        for role, head_name in ROLE_TO_HEAD.items():
            anchors[role].append(_to_numpy(model.contrastive_loss._project(ROLE_TO_ANCHOR[role], role_anchors[role])))
            for modality in STANDARD_MODALITIES:
                head = features.all_heads[modality][head_name]
                embeddings[role][modality].append(_to_numpy(model.contrastive_loss._project(modality, head)))
        eta_b, eta_s, chi_bs = _role_separation_per_sample(
            features.numerical,
            features.structural,
            kernel_size=int(OmegaConf.select(conf, "model.lowpass_kernel", default=5)),
        )
        eta_b_values.append(_to_numpy(eta_b))
        eta_s_values.append(_to_numpy(eta_s))
        chi_values.append(_to_numpy(chi_bs))
        routing_background_values.append(_to_numpy(routing_background))
        routing_structural_values.append(_to_numpy(routing_structural))
        batch_dataset_ids, batch_sample_indices = _batch_record_ids(cpu_batch, start_row=seen)
        dataset_ids.append(batch_dataset_ids)
        sample_indices.append(batch_sample_indices)
        availability.append(_to_numpy((cpu_batch.modality_mask * cpu_batch.modality_quality).gt(0)).astype(bool))
        seen += take

    if seen == 0:
        raise RuntimeError("Condition-contract diagnostics collected zero test samples.")
    return SeedDiagnostics(
        seed=int(seed),
        checkpoint=str(checkpoint_path),
        checkpoint_info=checkpoint_info,
        dataset_ids=np.concatenate(dataset_ids),
        sample_indices=np.concatenate(sample_indices),
        availability=np.concatenate(availability),
        embeddings={role: {modality: np.concatenate(parts) for modality, parts in by_modality.items()} for role, by_modality in embeddings.items()},
        anchor_embeddings={role: np.concatenate(parts) for role, parts in anchors.items()},
        eta_b=np.concatenate(eta_b_values),
        eta_s=np.concatenate(eta_s_values),
        chi_bs=np.concatenate(chi_values),
        routing_background=np.concatenate(routing_background_values),
        routing_structural=np.concatenate(routing_structural_values),
        dataset_names=dataset_names,
    )


def _mean_matrix(matrices: Iterable[np.ndarray]) -> np.ndarray:
    items = list(matrices)
    if not items:
        raise ValueError("Expected at least one matrix.")
    stacked = np.stack(items, axis=0)
    valid = ~np.isnan(stacked)
    result = np.full(stacked.shape[1:], np.nan, dtype=np.float64)
    np.divide(np.nansum(stacked, axis=0), valid.sum(axis=0), out=result, where=valid.sum(axis=0) > 0)
    return result


def _record_is_available(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"", "0", "false", "no", "nan"}


def _read_identity_csv(path: str | Path, *, required_fields: Sequence[str]) -> list[dict[str, Any]]:
    """Read manifest/metrics identities without discarding provenance columns."""
    resolved = _resolve_path(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"Identity CSV not found: {resolved}")
    with resolved.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        missing = sorted(set(required_fields) - fields)
        if missing:
            raise KeyError(f"{resolved} is missing required identity fields: {missing}")
        rows: list[dict[str, Any]] = []
        for raw in reader:
            row = dict(raw)
            row["dataset_id"] = int(raw["dataset_id"])
            row["dataset_name"] = str(raw["dataset_name"])
            row["source_sample_index"] = int(raw["source_sample_index"])
            rows.append(row)
    return rows


def _identity_set(rows: Sequence[Mapping[str, Any]]) -> set[tuple[int, str, int]]:
    identities = {
        (int(row["dataset_id"]), str(row["dataset_name"]), int(row["source_sample_index"]))
        for row in rows
    }
    if len(identities) != len(rows):
        raise ValueError("Duplicate dataset/sample identities found in formal Figure 3 records.")
    return identities


def _validate_formal_full_seed_set(seeds: Sequence[int]) -> tuple[int, ...]:
    normalized = tuple(sorted(int(seed) for seed in seeds))
    expected = tuple(sorted(FORMAL_FULL_SEEDS))
    if normalized != expected:
        raise ValueError(f"Formal Figure 3 diagnostics require Full seeds {expected}, got {normalized}.")
    return expected


def select_best_records_by_mean_ssim(
    records: Sequence[Mapping[str, Any]],
    *,
    required_seeds: Sequence[int] = (2027, 3407, 7777),
) -> list[dict[str, Any]]:
    """Select one complete-modality record per dataset by mean Full SSIM.

    The selection is deterministic: all required seeds must report the same
    dataset/sample identity, the largest mean SSIM wins, and source sample
    index breaks exact ties.  This function intentionally does not inspect
    routing weights, preventing the visualization from selecting a sample
    because its routing already looks favorable.
    """
    seeds = tuple(int(seed) for seed in required_seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("required_seeds must contain unique model seeds.")
    grouped: dict[tuple[int, int], dict[str, Any]] = {}
    for raw in records:
        if not _record_is_available(raw.get("all_modalities_available", False)):
            continue
        seed = int(raw["seed"])
        if seed not in seeds:
            continue
        key = (int(raw["dataset_id"]), int(raw["source_sample_index"]))
        score = float(raw["ssim"])
        if not np.isfinite(score):
            raise ValueError(f"Non-finite SSIM for record {key} and seed {seed}.")
        entry = grouped.setdefault(
            key,
            {
                "dataset_id": key[0],
                "dataset_name": str(raw.get("dataset_name", key[0])),
                "source_sample_index": key[1],
                "all_modalities_available": True,
                "seed_ssim": {},
            },
        )
        if seed in entry["seed_ssim"]:
            raise ValueError(f"Duplicate SSIM record for dataset/sample {key} and seed {seed}.")
        entry["seed_ssim"][seed] = score

    candidates_by_dataset: dict[int, list[dict[str, Any]]] = {}
    for entry in grouped.values():
        if set(entry["seed_ssim"]) != set(seeds):
            continue
        entry["mean_ssim"] = float(np.mean([entry["seed_ssim"][seed] for seed in seeds]))
        candidates_by_dataset.setdefault(int(entry["dataset_id"]), []).append(entry)

    selected: list[dict[str, Any]] = []
    for dataset_id, candidates in candidates_by_dataset.items():
        winner = min(
            candidates,
            key=lambda item: (-float(item["mean_ssim"]), int(item["source_sample_index"])),
        )
        selected.append(
            {
                **winner,
                "seed_ssim": {str(seed): float(winner["seed_ssim"][seed]) for seed in seeds},
                "required_seeds": list(seeds),
                "selection_rule": "highest mean Full SSIM over complete-modality held-out records; source_sample_index breaks ties",
            }
        )
    return sorted(selected, key=lambda item: int(item["dataset_id"]))


def load_evaluation_ssim(path: str | Path, *, seed: int) -> list[dict[str, Any]]:
    """Read per-sample SSIM and provenance from one formal evaluation CSV."""
    path = _resolve_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation metrics CSV not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"dataset_id", "dataset_name", "source_sample_index", "ssim"}
        missing = sorted(required.difference(reader.fieldnames or ()))
        if missing:
            raise KeyError(f"{path} is missing Figure 3 selection fields: {missing}")
        rows: list[dict[str, Any]] = []
        for raw in reader:
            rows.append(
                {
                    "seed": int(seed),
                    "dataset_id": int(raw["dataset_id"]),
                    "dataset_name": str(raw["dataset_name"]),
                    "source_sample_index": int(raw["source_sample_index"]),
                    "ssim": float(raw["ssim"]),
                    "all_modalities_available": _record_is_available(raw.get("all_modalities_available", True)),
                }
            )
    return rows


def _dataset_identity_rows(dataset: Any) -> list[dict[str, Any]]:
    records = getattr(dataset, "records", None)
    if records is None:
        raise TypeError("Formal split auditing requires a dataset exposing source records.")
    return [
        {
            "dataset_id": int(record["dataset_id"]),
            "dataset_name": str(record["dataset_name"]),
            "source_sample_index": int(record["sample_index"]),
        }
        for record in records
    ]


def audit_formal_openfwi_split(
    conf: DictConfig,
    *,
    manifest_path: str | Path | None = None,
    expected_test_count: int = 33_600,
) -> dict[str, Any]:
    """Audit global OpenFWI split membership before selecting Figure 3 samples."""
    data_name = str(OmegaConf.select(conf, "data.name", default=""))
    if data_name != "openfwi":
        raise ValueError(f"Formal Figure 3 split auditing requires data.name=openfwi, got {data_name!r}.")
    datasets: dict[str, Any] = {}
    try:
        for split in ("train", "val", "test"):
            datasets[split] = build_dataset(conf, split)
        rows = {split: _dataset_identity_rows(dataset) for split, dataset in datasets.items()}
        manifest_rows = (
            _read_identity_csv(
                manifest_path,
                required_fields=("dataset_id", "dataset_name", "source_sample_index"),
            )
            if manifest_path is not None
            else rows["test"]
        )
        audit = validate_split_identity(rows["train"], rows["val"], rows["test"], manifest_rows)
        if int(audit["test_count"]) != int(expected_test_count):
            raise ValueError(
                f"Formal test split has {audit['test_count']} records; expected {expected_test_count}."
            )
        return {
            **audit,
            "split_seed": int(OmegaConf.select(conf, "data.split_seed", default=42)),
            "well_seed": int(OmegaConf.select(conf, "data.well_seed", default=1234)),
            "dataset_names": [str(name) for name in OmegaConf.select(conf, "data.openfwi_datasets", default=[])],
            "manifest_path": None if manifest_path is None else str(_resolve_path(manifest_path)),
        }
    finally:
        for dataset in datasets.values():
            close = getattr(dataset, "close", None)
            if callable(close):
                close()


def retrieval_matrix(
    data: SeedDiagnostics,
    *,
    role: str,
    gallery_size: int,
    galleries_per_subset: int,
    seed: int,
) -> np.ndarray:
    if role not in ROLE_TO_HEAD:
        raise ValueError(f"Unknown role {role!r}.")
    galleries = build_subset_galleries(
        data.dataset_ids,
        data.availability.all(axis=1),
        gallery_size=gallery_size,
        galleries_per_subset=galleries_per_subset,
        seed=seed,
    )
    return _mean_matrix(symmetric_recall_at_one(data.embeddings[role], gallery) for gallery in galleries)


def _anchor_scores(data: SeedDiagnostics, role: str, shuffled: bool) -> np.ndarray:
    anchor = data.anchor_embeddings[role]
    if shuffled:
        anchor = anchor[strict_derangement_indices(anchor.shape[0])]
    values = np.zeros(anchor.shape[0], dtype=np.float32)
    counts = np.zeros(anchor.shape[0], dtype=np.float32)
    for modality_idx, modality in enumerate(STANDARD_MODALITIES):
        observed = data.availability[:, modality_idx]
        cosine = np.sum(data.embeddings[role][modality] * anchor, axis=1)
        values[observed] += cosine[observed]
        counts[observed] += 1.0
    valid = counts > 0
    return values[valid] / counts[valid]


def _bootstrap_mean_difference(matched: np.ndarray, shuffled: np.ndarray, *, seed: int, repeats: int = 1000) -> tuple[float, float, float]:
    if matched.shape != shuffled.shape:
        raise ValueError("Matched and shuffled arrays must be aligned.")
    delta = np.asarray(matched, dtype=np.float64) - np.asarray(shuffled, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        means[index] = delta[rng.integers(0, delta.size, size=delta.size)].mean()
    return float(delta.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _dataset_name(data: SeedDiagnostics, dataset_id: int) -> str:
    if 0 <= int(dataset_id) < len(data.dataset_names):
        return str(data.dataset_names[int(dataset_id)])
    return str(dataset_id)


def single_checkpoint_role_rows(data: SeedDiagnostics) -> list[dict[str, Any]]:
    """Return sample-level fused-condition diagnostics for the single-checkpoint mode."""
    rows: list[dict[str, Any]] = []
    for row_index, (dataset_id, source_index) in enumerate(zip(data.dataset_ids, data.sample_indices)):
        rows.append(
            {
                "evidence_scope": "single_checkpoint_diagnostic",
                "dataset_id": int(dataset_id),
                "dataset_name": _dataset_name(data, int(dataset_id)),
                "source_sample_index": int(source_index),
                "sample_row": int(row_index),
                "eta_b": float(data.eta_b[row_index]),
                "eta_s": float(data.eta_s[row_index]),
                "chi_bs": float(data.chi_bs[row_index]),
            }
        )
    return rows


def _validate_routing_arrays(data: SeedDiagnostics) -> None:
    expected_shape = (data.dataset_ids.size, len(STANDARD_MODALITIES))
    for role, weights in (
        ("background", data.routing_background),
        ("structural", data.routing_structural),
    ):
        values = np.asarray(weights, dtype=np.float64)
        if values.shape != expected_shape:
            raise ValueError(f"{role} routing has shape {values.shape}; expected {expected_shape}.")
        if not np.isfinite(values).all() or (values < -1e-7).any():
            raise ValueError(f"{role} routing contains non-finite or negative weights.")
        unavailable = ~np.asarray(data.availability, dtype=bool)
        if not np.allclose(values[unavailable], 0.0, atol=1e-6):
            raise ValueError(f"{role} routing assigns nonzero weight to unavailable modalities.")
        available_count = np.asarray(data.availability, dtype=bool).sum(axis=1)
        sums = values.sum(axis=1)
        if not np.allclose(sums[available_count > 0], 1.0, atol=1e-5):
            raise ValueError(f"{role} routing is not normalized over available modalities.")


def aggregate_complete_modality_routing(
    data: SeedDiagnostics,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], list[dict[str, Any]]]:
    """Aggregate learned routing over every complete-modality record by subset.

    The raw rows are deliberately retained alongside the 8x4 summary so the
    published heatmap can be audited without reconstructing model outputs.
    """
    _validate_routing_arrays(data)
    complete = np.asarray(data.availability, dtype=bool).all(axis=1)
    dataset_ids = sorted(np.unique(data.dataset_ids).astype(int).tolist())
    matrices = {
        "background": np.full((len(dataset_ids), len(STANDARD_MODALITIES)), np.nan, dtype=np.float64),
        "structural": np.full((len(dataset_ids), len(STANDARD_MODALITIES)), np.nan, dtype=np.float64),
    }
    raw_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    weights_by_role = {
        "background": np.asarray(data.routing_background, dtype=np.float64),
        "structural": np.asarray(data.routing_structural, dtype=np.float64),
    }
    for matrix_row, dataset_id in enumerate(dataset_ids):
        selected = np.flatnonzero((data.dataset_ids == dataset_id) & complete)
        if selected.size == 0:
            raise ValueError(f"Dataset {dataset_id} has no complete-modality records for routing diagnostics.")
        dataset_name = _dataset_name(data, dataset_id)
        for role, weights in weights_by_role.items():
            subset_weights = weights[selected]
            matrices[role][matrix_row] = subset_weights.mean(axis=0)
            for modality_index, modality in enumerate(STANDARD_MODALITIES):
                summary_rows.append(
                    {
                        "record_type": "summary",
                        "evidence_scope": "single_checkpoint_diagnostic",
                        "dataset_id": int(dataset_id),
                        "dataset_name": dataset_name,
                        "source_sample_index": "",
                        "sample_row": "",
                        "role": role,
                        "modality": modality,
                        "weight": "",
                        "mean_weight": float(subset_weights[:, modality_index].mean()),
                        "std_weight": float(subset_weights[:, modality_index].std(ddof=1))
                        if subset_weights.shape[0] > 1
                        else 0.0,
                        "sample_count": int(selected.size),
                    }
                )
        for row_index in selected:
            for role, weights in weights_by_role.items():
                for modality, weight in zip(STANDARD_MODALITIES, weights[row_index]):
                    raw_rows.append(
                        {
                            "record_type": "raw",
                            "evidence_scope": "single_checkpoint_diagnostic",
                            "dataset_id": int(dataset_id),
                            "dataset_name": dataset_name,
                            "source_sample_index": int(data.sample_indices[row_index]),
                            "sample_row": int(row_index),
                            "role": role,
                            "modality": modality,
                            "weight": float(weight),
                            "mean_weight": "",
                            "std_weight": "",
                            "sample_count": int(selected.size),
                        }
                    )
    return matrices, raw_rows, summary_rows


def build_retrieval_gallery_provenance(
    data: SeedDiagnostics,
    *,
    gallery_size: int,
    galleries_per_subset: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Record the exact same-subset identities used by retrieval heatmaps."""
    galleries = build_subset_galleries(
        data.dataset_ids,
        data.availability.all(axis=1),
        gallery_size=gallery_size,
        galleries_per_subset=galleries_per_subset,
        seed=seed,
    )
    provenance: list[dict[str, Any]] = []
    for gallery_id, indices in enumerate(galleries):
        dataset_ids = np.unique(data.dataset_ids[indices])
        if dataset_ids.size != 1:
            raise ValueError("Retrieval gallery crosses OpenFWI subset boundaries.")
        dataset_id = int(dataset_ids[0])
        provenance.append(
            {
                "gallery_id": int(gallery_id),
                "dataset_id": dataset_id,
                "dataset_name": _dataset_name(data, dataset_id),
                "size": int(indices.size),
                "sample_rows": [int(value) for value in indices],
                "source_sample_indices": [int(value) for value in data.sample_indices[indices]],
            }
        )
    return provenance


def build_single_checkpoint_metadata(
    *,
    checkpoint_path: str | Path,
    config_path: str | Path,
    manifest_path: str | Path,
    metrics_path: str | Path,
    checkpoint_info: Mapping[str, Any],
    sample_counts: Mapping[str, int],
    split_audit: Mapping[str, Any],
    gallery_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Build provenance metadata for the non-causal Figure 3 evidence scope."""
    paths = {
        "checkpoint": _resolve_path(checkpoint_path),
        "config": _resolve_path(config_path),
        "manifest": _resolve_path(manifest_path),
        "metrics": _resolve_path(metrics_path),
    }
    checkpoint_info_json = {
        str(key): (None if value is None else int(value) if isinstance(value, np.integer) else value)
        for key, value in dict(checkpoint_info).items()
    }
    return {
        "evidence_scope": "single_checkpoint_diagnostic",
        "causal_ablation": False,
        "causal_statement": "Diagnostic evidence from one existing epoch-99 checkpoint; not a causal ablation.",
        "inputs": {name: str(path) for name, path in paths.items()},
        "input_sha256": {name: _sha256_file(path) for name, path in paths.items()},
        "checkpoint": {"path": str(paths["checkpoint"]), **checkpoint_info_json},
        "sample_counts": {str(key): int(value) for key, value in sample_counts.items()},
        "split_audit": dict(split_audit),
        "gallery": dict(gallery_config),
    }


def _write_retrieval_csv(output_dir: Path, matrices: Mapping[str, np.ndarray]) -> Path:
    output = output_dir / "retrieval_matrices.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["role", "query", "target", "recall_at_1"])
        writer.writeheader()
        labels = [MODALITY_LABELS[name] for name in STANDARD_MODALITIES]
        for role, matrix in matrices.items():
            for row, query in enumerate(labels):
                for column, target in enumerate(labels):
                    if row == column:
                        continue
                    writer.writerow({"role": role, "query": query, "target": target, "recall_at_1": float(matrix[row, column])})
    return output


def _write_anchor_csv(output_dir: Path, rows: list[dict[str, Any]]) -> Path:
    output = output_dir / "anchor_calibration.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["seed", "role", "sample_row", "matched", "shuffled"])
        writer.writeheader()
        writer.writerows(rows)
    return output


def _write_role_csv(output_dir: Path, rows: list[dict[str, Any]]) -> Path:
    output = output_dir / "role_separation.csv"
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["regime", "seed", "eta_b", "eta_s", "chi_bs"])
        writer.writeheader()
        writer.writerows(rows)
    return output


def _write_best_sample_csv(
    output_dir: Path,
    selected: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
    *,
    checkpoint_sha256: Mapping[int, str | None] | None = None,
) -> Path:
    output = output_dir / "best_sample_selection.csv"
    seed_fields = [f"ssim_seed_{int(seed)}" for seed in seeds]
    checkpoint_fields = [f"checkpoint_sha256_seed_{int(seed)}" for seed in seeds]
    availability_fields = [f"available_{MODALITY_LABELS[name]}" for name in STANDARD_MODALITIES]
    fields = [
        "dataset_id",
        "dataset_name",
        "source_sample_index",
        *seed_fields,
        *checkpoint_fields,
        "mean_ssim",
        "all_modalities_available",
        *availability_fields,
        "selection_rule",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in selected:
            writer.writerow(
                {
                    "dataset_id": int(row["dataset_id"]),
                    "dataset_name": str(row["dataset_name"]),
                    "source_sample_index": int(row["source_sample_index"]),
                    **{field: float(row["seed_ssim"][str(seed)]) for field, seed in zip(seed_fields, seeds)},
                    **{
                        field: str((checkpoint_sha256 or {}).get(int(seed)) or "")
                        for field, seed in zip(checkpoint_fields, seeds)
                    },
                    "mean_ssim": float(row["mean_ssim"]),
                    "all_modalities_available": bool(row["all_modalities_available"]),
                    **{
                        field: bool(value)
                        for field, value in zip(
                            availability_fields,
                            row.get("modality_availability", [True] * len(STANDARD_MODALITIES)),
                        )
                    },
                    "selection_rule": str(row["selection_rule"]),
                }
            )
    return output


def _write_routing_csv(output_dir: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    output = output_dir / "reliability_routing.csv"
    fields = [
        "dataset_id",
        "dataset_name",
        "source_sample_index",
        "mean_ssim",
        "seed",
        "role",
        "modality",
        "weight",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return output


def _write_single_anchor_csv(output_dir: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    output = output_dir / "anchor_calibration.csv"
    fields = [
        "evidence_scope",
        "dataset_id",
        "dataset_name",
        "source_sample_index",
        "sample_row",
        "role",
        "matched",
        "shuffled",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return output


def _write_single_role_csv(output_dir: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    output = output_dir / "role_separation.csv"
    fields = [
        "evidence_scope",
        "dataset_id",
        "dataset_name",
        "source_sample_index",
        "sample_row",
        "eta_b",
        "eta_s",
        "chi_bs",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return output


def _write_single_routing_csv(
    output_dir: Path,
    raw_rows: Sequence[Mapping[str, Any]],
    summary_rows: Sequence[Mapping[str, Any]],
) -> Path:
    output = output_dir / "reliability_routing.csv"
    fields = [
        "record_type",
        "evidence_scope",
        "dataset_id",
        "dataset_name",
        "source_sample_index",
        "sample_row",
        "role",
        "modality",
        "weight",
        "mean_weight",
        "std_weight",
        "sample_count",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([*raw_rows, *summary_rows])
    return output


def _write_gallery_provenance_csv(output_dir: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    output = output_dir / "retrieval_gallery_provenance.csv"
    fields = [
        "gallery_id",
        "dataset_id",
        "dataset_name",
        "position",
        "sample_row",
        "source_sample_index",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for gallery in rows:
            for position, (sample_row, source_index) in enumerate(
                zip(gallery["sample_rows"], gallery["source_sample_indices"])
            ):
                writer.writerow(
                    {
                        "gallery_id": int(gallery["gallery_id"]),
                        "dataset_id": int(gallery["dataset_id"]),
                        "dataset_name": str(gallery["dataset_name"]),
                        "position": int(position),
                        "sample_row": int(sample_row),
                        "source_sample_index": int(source_index),
                    }
                )
    return output


def build_routing_matrices(
    full: Sequence[SeedDiagnostics],
    selected: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    """Aggregate actual encoder routing weights for selected test identities."""
    if not full:
        raise ValueError("At least one Full diagnostic is required for routing matrices.")
    data_by_seed = {int(data.seed): data for data in full}
    seeds = tuple(int(seed) for seed in selected[0].get("required_seeds", data_by_seed)) if selected else tuple(data_by_seed)
    if set(seeds) != set(data_by_seed):
        raise ValueError(f"Selected routing seeds {seeds} do not match Full diagnostics {sorted(data_by_seed)}.")
    matrices = {
        "background": np.zeros((len(selected), len(STANDARD_MODALITIES)), dtype=np.float64),
        "structural": np.zeros((len(selected), len(STANDARD_MODALITIES)), dtype=np.float64),
    }
    detail_rows: list[dict[str, Any]] = []
    for row_index, selected_row in enumerate(selected):
        identity = (int(selected_row["dataset_id"]), int(selected_row["source_sample_index"]))
        per_role: dict[str, list[np.ndarray]] = {"background": [], "structural": []}
        for seed in seeds:
            data = data_by_seed[seed]
            matches = np.flatnonzero((data.dataset_ids == identity[0]) & (data.sample_indices == identity[1]))
            if matches.size != 1:
                raise ValueError(f"Expected one diagnostic row for {identity} and seed {seed}, found {matches.size}.")
            match = int(matches[0])
            if not data.availability[match].all():
                raise ValueError(f"Selected routing identity {identity} is not complete for seed {seed}.")
            per_role["background"].append(data.routing_background[match])
            per_role["structural"].append(data.routing_structural[match])
            for role, values in (("background", data.routing_background[match]), ("structural", data.routing_structural[match])):
                for modality, weight in zip(STANDARD_MODALITIES, values):
                    detail_rows.append(
                        {
                            "dataset_id": identity[0],
                            "dataset_name": str(selected_row["dataset_name"]),
                            "source_sample_index": identity[1],
                            "mean_ssim": float(selected_row["mean_ssim"]),
                            "seed": seed,
                            "role": role,
                            "modality": modality,
                            "weight": float(weight),
                        }
                    )
        for role in matrices:
            matrices[role][row_index] = np.mean(np.stack(per_role[role], axis=0), axis=0)
    return matrices, detail_rows


def _seed_metadata(data: SeedDiagnostics) -> dict[str, Any]:
    counts = {
        data.dataset_names[dataset_id] if 0 <= dataset_id < len(data.dataset_names) else str(dataset_id): int(count)
        for dataset_id, count in zip(*np.unique(data.dataset_ids, return_counts=True))
    }
    return {
        "seed": data.seed,
        "checkpoint": data.checkpoint,
        "checkpoint_info": data.checkpoint_info,
        "num_samples": int(data.dataset_ids.size),
        "complete_modality_samples": int(data.availability.all(axis=1).sum()),
        "dataset_counts": counts,
    }


def _configure_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.linewidth": 0.7,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
        }
    )


def _heatmap(ax: plt.Axes, matrix: np.ndarray, title: str, *, show_y_labels: bool = True) -> Any:
    image = ax.imshow(matrix * 100.0, cmap="viridis", vmin=0.0, vmax=100.0)
    labels = [MODALITY_LABELS[name] for name in STANDARD_MODALITIES]
    ax.set_xticks(range(len(labels)), labels=labels, rotation=38, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels if show_y_labels else [])
    ax.tick_params(axis="both", labelsize=5.7)
    ax.set_title(title, fontsize=6.8, pad=2)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            if row == col:
                ax.text(col, row, "--", ha="center", va="center", fontsize=7, color="#555555")
            else:
                color = "white" if matrix[row, col] > 0.52 else "black"
                ax.text(col, row, f"{100.0 * matrix[row, col]:.1f}", ha="center", va="center", fontsize=5.1, color=color)
    return image


def _violin(ax: plt.Axes, values: np.ndarray, position: float, color: str, label: str | None = None) -> None:
    result = ax.violinplot([values], positions=[position], widths=0.28, showmeans=False, showmedians=True, showextrema=False)
    for body in result["bodies"]:
        body.set_facecolor(color)
        body.set_edgecolor(color)
        body.set_alpha(0.62)
    result["cmedians"].set_color("#202124")
    result["cmedians"].set_linewidth(1.1)
    if label is not None:
        ax.scatter([], [], color=color, label=label)


def _draw_role_panel(ax: plt.Axes, rows: Sequence[Mapping[str, Any]]) -> None:
    metrics = (("eta_b", r"$\eta_B\uparrow$", GREEN), ("eta_s", r"$\eta_S\uparrow$", BLUE), ("chi_bs", r"$\chi_{BS}\downarrow$", PURPLE))
    regimes = ["full", "no_contract"]
    labels = {"full": "Full", "no_contract": "w/o contract"}
    for index, (key, title, color) in enumerate(metrics):
        for regime_offset, regime in enumerate(regimes):
            values = [float(row[key]) for row in rows if str(row["regime"]) == regime]
            x = index + (-0.14 if regime == "full" else 0.14)
            if not values:
                continue
            ax.scatter([x] * len(values), values, s=21, facecolors=color if regime == "full" else "white", edgecolors=color if regime == "full" else GRAY, linewidths=0.9, zorder=3)
            ax.errorbar(x, np.mean(values), yerr=np.std(values, ddof=1) if len(values) > 1 else 0.0, color=color if regime == "full" else GRAY, fmt="_", markersize=10, capsize=2, linewidth=1.0, zorder=4)
    ax.set_xlim(-0.55, 2.55)
    ax.set_ylim(-0.04, 1.0)
    ax.set_xticks(
        [0, 1, 2],
        labels=[
            r"$C_B$" + "\n" + r"$\eta_B\uparrow$",
            r"$C_S$" + "\n" + r"$\eta_S\uparrow$",
            "leakage\n" + r"$\chi_{BS}\downarrow$",
        ],
    )
    for label, (_, _, color) in zip(ax.get_xticklabels(), metrics):
        label.set_color(color)
        label.set_fontsize(5.5)
    ax.set_ylabel("test-set diagnostic")
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)
    ax.scatter([], [], color="#333333", label=labels["full"])
    ax.scatter([], [], facecolors="white", edgecolors=GRAY, label=labels["no_contract"])
    ax.legend(loc="lower left", fontsize=6.4, frameon=False)


def _plot_single_role_panel(
    figure: plt.Figure,
    slot: Any,
    rows: Sequence[Mapping[str, Any]],
    dataset_labels: Sequence[str],
    *,
    panel_label: str = "(b)",
) -> None:
    """Plot sample distributions for the fused conditions without a control regime."""
    subgrid = slot.subgridspec(4, 1, height_ratios=[0.18, 1.0, 1.0, 1.0], hspace=0.38)
    title_axis = figure.add_subplot(subgrid[0, 0])
    title_axis.axis("off")
    title_axis.text(
        0.0,
        0.45,
        f"{panel_label} Role specialization diagnostics",
        transform=title_axis.transAxes,
        ha="left",
        va="center",
        fontsize=7.4,
        fontweight="bold",
    )
    metric_specs = (
        ("eta_b", r"$\eta_B\uparrow$", GREEN),
        ("eta_s", r"$\eta_S\uparrow$", BLUE),
        ("chi_bs", r"$\chi_{BS}\downarrow$", PURPLE),
    )
    for row_index, (key, label, color) in enumerate(metric_specs, start=1):
        axis = figure.add_subplot(subgrid[row_index, 0])
        values_by_dataset = [
            np.asarray([float(item[key]) for item in rows if str(item["dataset_name"]) == str(name)], dtype=np.float64)
            for name in dataset_labels
        ]
        if any(values.size == 0 for values in values_by_dataset):
            raise ValueError(f"Missing role diagnostic values for one or more datasets in metric {key}.")
        result = axis.boxplot(
            values_by_dataset,
            positions=np.arange(len(dataset_labels)),
            widths=0.56,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#202124", "linewidth": 0.75},
            whiskerprops={"color": color, "linewidth": 0.65},
            capprops={"color": color, "linewidth": 0.65},
            boxprops={"edgecolor": color, "linewidth": 0.65},
        )
        for patch in result["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.35)
        global_median = float(np.median(np.concatenate(values_by_dataset)))
        axis.axhline(global_median, color=color, linestyle="--", linewidth=0.65, alpha=0.9)
        axis.text(
            0.99,
            0.82,
            f"global med. {global_median:.2f}",
            transform=axis.transAxes,
            ha="right",
            va="center",
            fontsize=4.5,
            color=color,
        )
        axis.set_ylabel(label, fontsize=5.6, color=color)
        axis.set_ylim(-0.02, 1.02)
        axis.set_xlim(-0.7, len(dataset_labels) - 0.3)
        axis.grid(axis="y", alpha=0.18, linewidth=0.45)
        axis.tick_params(axis="y", labelsize=4.8, length=1.5)
        if row_index == len(metric_specs):
            axis.set_xticks(np.arange(len(dataset_labels)), labels=list(dataset_labels), rotation=48, ha="right")
            axis.tick_params(axis="x", labelsize=4.0, length=1.5)
        else:
            axis.set_xticks([])


def _plot_single_checkpoint_figure3(
    *,
    output_dir: Path,
    anchor_values: Mapping[str, tuple[np.ndarray, np.ndarray]],
    role_rows: Sequence[Mapping[str, Any]],
    dataset_labels: Sequence[str],
) -> list[Path]:
    """Render the main-text anchor/role diagnostic without weak auxiliary panels."""
    _configure_plot_style()
    figure = plt.figure(figsize=(7.15, 3.05))
    grid = figure.add_gridspec(
        1,
        2,
        width_ratios=[1.0, 1.55],
        wspace=0.34,
    )
    axis_anchor = figure.add_subplot(grid[0, 0])
    positions = {"background": (0.83, 1.17), "structural": (1.83, 2.17)}
    summaries: list[str] = []
    for role, color in (("background", GREEN), ("structural", BLUE)):
        matched, shuffled = anchor_values[role]
        _violin(axis_anchor, matched, positions[role][0], color, "matched" if role == "background" else None)
        _violin(axis_anchor, shuffled, positions[role][1], GRAY, "shuffled" if role == "background" else None)
        delta, lower, upper = _bootstrap_mean_difference(
            matched,
            shuffled,
            seed=17 if role == "background" else 29,
        )
        summaries.append(
            f"{role[0].upper()} med={np.median(matched):.2f}; $\u0394$={delta:.2f} [{lower:.2f},{upper:.2f}]"
        )
    axis_anchor.set_title("(a) Physical anchor calibration", fontsize=8.0, pad=17)
    axis_anchor.text(
        0.5,
        1.01,
        "\n".join(summaries),
        transform=axis_anchor.transAxes,
        ha="center",
        va="bottom",
        fontsize=4.35,
        color="#3E4852",
    )
    axis_anchor.set_xticks([1.0, 2.0], labels=["background", "structure"])
    axis_anchor.tick_params(axis="both", labelsize=5.5)
    axis_anchor.set_ylabel("cosine similarity", fontsize=5.8)
    axis_anchor.grid(axis="y", alpha=0.2, linewidth=0.5)
    axis_anchor.legend(loc="lower left", fontsize=5.7, frameon=False)

    _plot_single_role_panel(
        figure,
        grid[0, 1],
        role_rows,
        dataset_labels,
        panel_label="(b)",
    )
    figure.subplots_adjust(left=0.075, right=0.985, top=0.86, bottom=0.19, wspace=0.34)
    outputs: list[Path] = []
    for suffix, dpi in (("png", 320), ("pdf", None), ("svg", None)):
        path = output_dir / f"figure3_condition_learning.{suffix}"
        kwargs: dict[str, Any] = {"bbox_inches": "tight"}
        if dpi is not None:
            kwargs["dpi"] = dpi
        figure.savefig(path, **kwargs)
        outputs.append(path)
    plt.close(figure)
    return outputs


def render_single_checkpoint_summary_from_csv(
    *,
    output_dir: str | Path,
    anchor_csv: str | Path,
    role_csv: str | Path,
    dataset_labels: Sequence[str] | None = None,
) -> list[Path]:
    """Regenerate the compact main-text figure from saved sample-level diagnostics."""
    output = _ensure_dir(_resolve_path(output_dir))
    with _resolve_path(anchor_csv).open(newline="", encoding="utf-8") as handle:
        anchor_rows = list(csv.DictReader(handle))
    with _resolve_path(role_csv).open(newline="", encoding="utf-8") as handle:
        raw_role_rows = list(csv.DictReader(handle))
    if not anchor_rows or not raw_role_rows:
        raise ValueError("Anchor and role CSVs must both contain sample-level rows.")
    anchor_values: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for role in ROLE_TO_HEAD:
        selected = [row for row in anchor_rows if str(row["role"]) == role]
        if not selected:
            raise ValueError(f"Anchor CSV contains no rows for role {role!r}.")
        anchor_values[role] = (
            np.asarray([float(row["matched"]) for row in selected], dtype=np.float64),
            np.asarray([float(row["shuffled"]) for row in selected], dtype=np.float64),
        )
    role_rows = [
        {
            **row,
            "dataset_id": int(row["dataset_id"]),
            "source_sample_index": int(row["source_sample_index"]),
            "eta_b": float(row["eta_b"]),
            "eta_s": float(row["eta_s"]),
            "chi_bs": float(row["chi_bs"]),
        }
        for row in raw_role_rows
    ]
    labels = list(dataset_labels or ())
    if not labels:
        labels = [
            str(next(row["dataset_name"] for row in role_rows if int(row["dataset_id"]) == dataset_id))
            for dataset_id in sorted({int(row["dataset_id"]) for row in role_rows})
        ]
    return _plot_single_checkpoint_figure3(
        output_dir=output,
        anchor_values=anchor_values,
        role_rows=role_rows,
        dataset_labels=labels,
    )


def _plot_routing_panel(
    figure: plt.Figure,
    slot: Any,
    routing_matrices: Mapping[str, np.ndarray] | None,
    routing_labels: Sequence[str] | None,
) -> None:
    subgrid = slot.subgridspec(2, 2, height_ratios=[0.16, 1.0], wspace=0.16, hspace=0.12)
    title_axis = figure.add_subplot(subgrid[0, :])
    title_axis.axis("off")
    title_axis.text(
        0.0,
        0.45,
        "(d) Role-specific reliability routing",
        transform=title_axis.transAxes,
        ha="left",
        va="center",
        fontsize=7.8,
        fontweight="bold",
    )
    if not routing_matrices or routing_labels is None:
        axis = figure.add_subplot(subgrid[1, :])
        axis.axis("off")
        axis.text(
            0.5,
            0.5,
            "Reliability routing requires completed Full evaluations.",
            ha="center",
            va="center",
            color=GRAY,
            fontsize=8.5,
        )
        return

    labels = [MODALITY_LABELS[name] for name in STANDARD_MODALITIES]
    for column, (role, cmap, color, title) in enumerate(
        (
            ("background", "YlGn", GREEN, r"Background weights $w^B$"),
            ("structural", "Blues", BLUE, r"Structural weights $w^S$"),
        )
    ):
        matrix = np.asarray(routing_matrices[role], dtype=np.float64)
        if matrix.shape != (len(routing_labels), len(labels)):
            raise ValueError(f"Routing matrix for {role} has shape {matrix.shape}, expected {(len(routing_labels), len(labels))}.")
        axis = figure.add_subplot(subgrid[1, column])
        image = axis.imshow(matrix, cmap=cmap, vmin=0.0, vmax=1.0, aspect="auto")
        axis.set_title(title, fontsize=7.0, color=color, pad=2)
        axis.set_xticks(range(len(labels)), labels=labels, rotation=28, ha="right")
        axis.tick_params(axis="x", labelsize=5.5)
        axis.tick_params(axis="y", labelsize=5.6)
        if column == 0:
            axis.set_yticks(range(len(routing_labels)), labels=list(routing_labels))
            axis.set_ylabel("OpenFWI subset", fontsize=6.0)
        else:
            axis.set_yticks(range(len(routing_labels)), labels=[])
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                value = float(matrix[row, col])
                text_color = "white" if value > 0.58 else "#202124"
                axis.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=5.1, color=text_color)
        colorbar = figure.colorbar(image, ax=axis, fraction=0.045, pad=0.025)
        colorbar.ax.tick_params(labelsize=5.0, length=1.5)
        colorbar.set_label("normalized weight", fontsize=5.2)


def _plot_spatial_panel(
    figure: plt.Figure,
    slot: Any,
    examples: Sequence[Mapping[str, Any]] | None,
) -> None:
    subgrid = slot.subgridspec(3, 5, height_ratios=[0.15, 1.0, 1.0], wspace=0.06, hspace=0.16)
    title_axis = figure.add_subplot(subgrid[0, :])
    title_axis.axis("off")
    title_axis.text(
        0.0,
        0.45,
        "(d) Spatial condition responses (frozen encoder; post-hoc probes)",
        transform=title_axis.transAxes,
        ha="left",
        va="center",
        fontsize=7.8,
        fontweight="bold",
    )
    columns = ["Target $V$", "Anchor $A_B$", "Probe$(C_B)$", "Anchor $A_S$", "Probe$(C_S)$"]
    if not examples:
        axis = figure.add_subplot(subgrid[1:, :])
        axis.axis("off")
        axis.text(0.5, 0.5, "Spatial probes require completed Full checkpoints.", ha="center", va="center", color=GRAY, fontsize=9)
        return
    background_values = np.concatenate([np.ravel(example[name]) for example in examples for name in ("target", "anchor_b", "probe_b")])
    structure_values = np.concatenate([np.ravel(example[name]) for example in examples for name in ("anchor_s", "probe_s")])
    bmin, bmax = np.quantile(background_values, [0.01, 0.99])
    smin, smax = np.quantile(structure_values, [0.01, 0.99])
    for row, example in enumerate(examples):
        values = [example["target"], example["anchor_b"], example["probe_b"], example["anchor_s"], example["probe_s"]]
        for col, value in enumerate(values):
            axis = figure.add_subplot(subgrid[row + 1, col])
            cmap, vmin, vmax = ("viridis", bmin, bmax) if col < 3 else ("magma", smin, smax)
            image = axis.imshow(value, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_linewidth(1.1)
                spine.set_color(GREEN if col in (1, 2) else BLUE if col in (3, 4) else "#4D5660")
            if row == 0:
                axis.set_title(columns[col], fontsize=7.1, color=GREEN if col in (1, 2) else BLUE if col in (3, 4) else "#202124", pad=2)
            if col == 0:
                axis.set_ylabel(str(example["dataset_name"]), fontsize=7.2)
            if row == 1 and col in (2, 4):
                cbar = figure.colorbar(image, ax=axis, orientation="horizontal", fraction=0.05, pad=0.10)
                cbar.ax.tick_params(labelsize=5, length=1.5)


def plot_spatial_probe_figure(output_dir: Path, examples: Sequence[Mapping[str, Any]]) -> list[Path]:
    """Export the legacy post-hoc probe visualization as an appendix asset."""
    output_dir = _ensure_dir(output_dir)
    _configure_plot_style()
    figure = plt.figure(figsize=(7.15, 3.1))
    slot = figure.add_gridspec(1, 1)[0, 0]
    _plot_spatial_panel(figure, slot, examples)
    figure.subplots_adjust(left=0.055, right=0.985, top=0.92, bottom=0.08)
    outputs: list[Path] = []
    for suffix, dpi in (("png", 320), ("pdf", None), ("svg", None)):
        path = output_dir / f"appendix_spatial_probes.{suffix}"
        kwargs: dict[str, Any] = {"bbox_inches": "tight"}
        if dpi is not None:
            kwargs["dpi"] = dpi
        figure.savefig(path, **kwargs)
        outputs.append(path)
    plt.close(figure)
    return outputs


def plot_figure3(
    *,
    output_dir: Path,
    retrieval_matrices: Mapping[str, np.ndarray],
    anchor_values: Mapping[str, tuple[np.ndarray, np.ndarray]],
    role_rows: Sequence[Mapping[str, Any]],
    spatial_examples: Sequence[Mapping[str, Any]] | None = None,
    routing_matrices: Mapping[str, np.ndarray] | None = None,
    routing_labels: Sequence[str] | None = None,
) -> list[Path]:
    _configure_plot_style()
    # Export at the physical width of an AAAI double-column figure.  The TeX
    # include then makes only a negligible scale adjustment, preserving the
    # intended 7--9 pt labels instead of shrinking a presentation-sized plot.
    figure = plt.figure(figsize=(7.15, 4.5))
    grid = figure.add_gridspec(
        2,
        3,
        height_ratios=[1.0, 1.32],
        width_ratios=[1.45, 1.0, 1.0],
        hspace=0.48,
        wspace=0.45,
    )
    heat_grid = grid[0, 0].subgridspec(1, 2, wspace=0.18)
    axis_b = figure.add_subplot(heat_grid[0, 0])
    axis_s = figure.add_subplot(heat_grid[0, 1])
    image = _heatmap(axis_b, retrieval_matrices["background"], "(a) Background R@1")
    _heatmap(axis_s, retrieval_matrices["structural"], "Structural R@1", show_y_labels=False)
    colorbar = figure.colorbar(image, ax=[axis_b, axis_s], fraction=0.042, pad=0.025)
    colorbar.set_label("R@1 (%)", fontsize=5.8)
    colorbar.ax.tick_params(labelsize=5.2)

    axis_anchor = figure.add_subplot(grid[0, 1])
    positions = {"background": (0.83, 1.17), "structural": (1.83, 2.17)}
    delta_summaries: list[str] = []
    for role, color in (("background", GREEN), ("structural", BLUE)):
        matched, shuffled = anchor_values[role]
        _violin(axis_anchor, matched, positions[role][0], color, "matched" if role == "background" else None)
        _violin(axis_anchor, shuffled, positions[role][1], GRAY, "shuffled" if role == "background" else None)
        delta, lower, upper = _bootstrap_mean_difference(matched, shuffled, seed=17 if role == "background" else 29)
        delta_summaries.append(f"{role[0].upper()}: $\\Delta$={delta:.3f} [{lower:.3f}, {upper:.3f}]")
    axis_anchor.set_title("(b) Anchor calibration", fontsize=7.4, pad=15)
    axis_anchor.text(
        0.5,
        1.01,
        "  ".join(delta_summaries),
        transform=axis_anchor.transAxes,
        ha="center",
        va="bottom",
        fontsize=4.7,
        color="#3E4852",
    )
    axis_anchor.set_xticks([1.0, 2.0], labels=["background", "structure"])
    axis_anchor.tick_params(axis="both", labelsize=5.8)
    axis_anchor.set_ylabel("cosine", fontsize=6.2)
    axis_anchor.grid(axis="y", alpha=0.2, linewidth=0.5)
    axis_anchor.legend(loc="lower left", fontsize=6.3, frameon=False)

    axis_role = figure.add_subplot(grid[0, 2])
    axis_role.set_title("(c) Role separation", fontsize=7.4, pad=3)
    _draw_role_panel(axis_role, role_rows)

    if routing_matrices is not None or spatial_examples is None:
        _plot_routing_panel(figure, grid[1, :], routing_matrices, routing_labels)
    else:
        _plot_spatial_panel(figure, grid[1, :], spatial_examples)
    figure.subplots_adjust(left=0.055, right=0.985, top=0.90, bottom=0.075, hspace=0.48, wspace=0.45)
    outputs = []
    for suffix, dpi in (("png", 320), ("pdf", None), ("svg", None)):
        path = output_dir / f"figure3_condition_learning.{suffix}"
        kwargs: dict[str, Any] = {"bbox_inches": "tight"}
        if dpi is not None:
            kwargs["dpi"] = dpi
        figure.savefig(path, **kwargs)
        outputs.append(path)
    plt.close(figure)
    return outputs


def _probe_design(condition: torch.Tensor) -> torch.Tensor:
    batch, channels, height, width = condition.shape
    rows = condition.permute(0, 2, 3, 1).reshape(batch * height * width, channels)
    return torch.cat([rows, torch.ones(rows.shape[0], 1, device=rows.device, dtype=rows.dtype)], dim=1)


@torch.no_grad()
def _probe_normal_equations(model: BGPDRFMLightning, loader: DataLoader, device: torch.device, max_batches: int | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    xtx: torch.Tensor | None = None
    xty_b: torch.Tensor | None = None
    xty_s: torch.Tensor | None = None
    for batch_idx, cpu_batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        batch = batch_to_device(cpu_batch, device)
        features = model.encoder(batch, return_all_heads=False)
        anchors = model.wavelet_anchor(batch.depth_vel)
        x = _probe_design(features.numerical).float()
        xs = _probe_design(features.structural).float()
        yb = anchors.anchor_num.reshape(-1, 1).float()
        ys = anchors.anchor_struct.reshape(-1, 1).float()
        current_xtx_b = x.T @ x
        current_xtx_s = xs.T @ xs
        if xtx is None:
            xtx = torch.stack([current_xtx_b, current_xtx_s]).cpu().double()
            xty_b = (x.T @ yb).cpu().double()
            xty_s = (xs.T @ ys).cpu().double()
        else:
            xtx[0] += current_xtx_b.cpu().double()
            xtx[1] += current_xtx_s.cpu().double()
            xty_b += (x.T @ yb).cpu().double()
            xty_s += (xs.T @ ys).cpu().double()
    if xtx is None or xty_b is None or xty_s is None:
        raise RuntimeError("Probe fitting received no batches.")
    return xtx, xty_b, xty_s


@torch.no_grad()
def _probe_mse(model: BGPDRFMLightning, loader: DataLoader, device: torch.device, weights_b: torch.Tensor, weights_s: torch.Tensor, max_batches: int | None) -> tuple[float, float]:
    squared_b = 0.0
    squared_s = 0.0
    count = 0
    for batch_idx, cpu_batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        batch = batch_to_device(cpu_batch, device)
        features = model.encoder(batch, return_all_heads=False)
        anchors = model.wavelet_anchor(batch.depth_vel)
        xb = _probe_design(features.numerical).double().cpu()
        xs = _probe_design(features.structural).double().cpu()
        prediction_b = xb @ weights_b
        prediction_s = xs @ weights_s
        target_b = anchors.anchor_num.reshape(-1, 1).double().cpu()
        target_s = anchors.anchor_struct.reshape(-1, 1).double().cpu()
        squared_b += float((prediction_b - target_b).square().sum())
        squared_s += float((prediction_s - target_s).square().sum())
        count += int(target_b.numel())
    if count == 0:
        raise RuntimeError("Probe validation received no batches.")
    return squared_b / count, squared_s / count


def fit_posthoc_probes(
    *,
    config_path: str | Path,
    checkpoint_path: str | Path,
    device: str | torch.device,
    batch_size: int = 64,
    max_batches: int | None = None,
) -> ProbeWeights:
    """Fit train-only ridge probes and select their ridge scale on validation."""
    resolved_device = torch.device(device)
    model, conf, _ = _prepare_model(_resolve_path(config_path), _resolve_path(checkpoint_path), resolved_device)
    train_loader, _ = _loader(conf, "train", batch_size)
    val_loader, _ = _loader(conf, "val", batch_size)
    xtx, xty_b, xty_s = _probe_normal_equations(model, train_loader, resolved_device, max_batches)
    candidates = (1e-6, 1e-4, 1e-2)
    candidate_weights: list[tuple[float, torch.Tensor, torch.Tensor, float, float]] = []
    for alpha in candidates:
        scale_b = torch.trace(xtx[0]) / xtx.shape[-1]
        scale_s = torch.trace(xtx[1]) / xtx.shape[-1]
        wb = torch.linalg.solve(xtx[0] + torch.eye(xtx.shape[-1], dtype=torch.float64) * alpha * scale_b, xty_b)
        ws = torch.linalg.solve(xtx[1] + torch.eye(xtx.shape[-1], dtype=torch.float64) * alpha * scale_s, xty_s)
        mse_b, mse_s = _probe_mse(model, val_loader, resolved_device, wb, ws, max_batches)
        candidate_weights.append((alpha, wb, ws, mse_b, mse_s))
    best_b = min(candidate_weights, key=lambda item: item[3])
    best_s = min(candidate_weights, key=lambda item: item[4])
    return ProbeWeights(
        background=best_b[1].numpy().reshape(-1),
        structural=best_s[2].numpy().reshape(-1),
        background_alpha=float(best_b[0]),
        structural_alpha=float(best_s[0]),
        background_validation_mse=float(best_b[3]),
        structural_validation_mse=float(best_s[4]),
    )


def _apply_probe(condition: torch.Tensor, weights: np.ndarray) -> torch.Tensor:
    coefficients = torch.as_tensor(weights[:-1], device=condition.device, dtype=condition.dtype).view(1, -1, 1, 1)
    bias = torch.as_tensor(weights[-1], device=condition.device, dtype=condition.dtype)
    return (condition * coefficients).sum(dim=1, keepdim=True) + bias


def _select_medoid_rows(data: SeedDiagnostics, dataset_name: str) -> tuple[int, int]:
    if dataset_name not in data.dataset_names:
        raise ValueError(f"Dataset {dataset_name!r} is not present in this evaluation configuration.")
    dataset_id = data.dataset_names.index(dataset_name)
    rows = np.flatnonzero(data.dataset_ids == dataset_id)
    if rows.size == 0:
        raise ValueError(f"No held-out rows found for dataset {dataset_name!r}.")
    diagnostics = np.stack([data.eta_b[rows], data.eta_s[rows], data.chi_bs[rows]], axis=1)
    center = np.median(diagnostics, axis=0)
    scale = np.maximum(np.median(np.abs(diagnostics - center), axis=0), 1e-6)
    row = rows[np.argmin(np.sum(((diagnostics - center) / scale) ** 2, axis=1))]
    return int(data.dataset_ids[row]), int(data.sample_indices[row])


@torch.no_grad()
def extract_spatial_examples(
    *,
    config_path: str | Path,
    checkpoint_path: str | Path,
    diagnostics: SeedDiagnostics,
    probes: ProbeWeights,
    device: str | torch.device,
    batch_size: int,
    datasets: Sequence[str] = ("FlatFaultB", "CurveFaultB"),
) -> list[dict[str, Any]]:
    """Export pre-registered medoid examples without selecting by visual quality."""
    desired = {_select_medoid_rows(diagnostics, dataset_name): dataset_name for dataset_name in datasets}
    resolved_device = torch.device(device)
    model, conf, _ = _prepare_model(_resolve_path(config_path), _resolve_path(checkpoint_path), resolved_device)
    loader, _ = _loader(conf, "test", batch_size)
    examples: dict[str, dict[str, Any]] = {}
    for cpu_batch in loader:
        ids = cpu_batch.metadata["dataset_id"].to(torch.int64).tolist()
        indices = cpu_batch.metadata["sample_index"].to(torch.int64).tolist()
        selected = [(offset, desired[(int(dataset_id), int(sample_index))]) for offset, (dataset_id, sample_index) in enumerate(zip(ids, indices)) if (int(dataset_id), int(sample_index)) in desired]
        if not selected:
            continue
        batch = batch_to_device(cpu_batch, resolved_device)
        features = model.encoder(batch, return_all_heads=False)
        anchors = model.wavelet_anchor(batch.depth_vel)
        probe_b = _apply_probe(features.numerical, probes.background)
        probe_s = _apply_probe(features.structural, probes.structural)
        for offset, dataset_name in selected:
            examples[dataset_name] = {
                "dataset_name": dataset_name,
                "dataset_id": int(ids[offset]),
                "sample_index": int(indices[offset]),
                "target": _to_numpy(batch.depth_vel[offset, 0]),
                "anchor_b": _to_numpy(anchors.anchor_num[offset, 0]),
                "probe_b": _to_numpy(probe_b[offset, 0]),
                "anchor_s": _to_numpy(anchors.anchor_struct[offset, 0]),
                "probe_s": _to_numpy(probe_s[offset, 0]),
            }
        if len(examples) == len(datasets):
            break
    missing = [dataset_name for dataset_name in datasets if dataset_name not in examples]
    if missing:
        raise RuntimeError(f"Could not recover selected spatial examples: {missing}.")
    return [examples[dataset_name] for dataset_name in datasets]


def _parse_checkpoint_specs(specification: str) -> list[tuple[int, Path]]:
    items = [item.strip() for item in specification.split(",") if item.strip()]
    parsed: list[tuple[int, Path]] = []
    for item in items:
        if "=" not in item:
            raise ValueError("Checkpoint specifications must use seed=path, separated by commas.")
        seed_text, path_text = item.split("=", 1)
        parsed.append((int(seed_text), _resolve_path(path_text)))
    return parsed


def _build_selection_records(
    full: Sequence[SeedDiagnostics],
    evaluation_metrics: Sequence[tuple[int, str | Path]],
    *,
    expected_test_count: int = 33_600,
) -> list[dict[str, Any]]:
    metric_specs = {int(seed): path for seed, path in evaluation_metrics}
    full_seeds = {int(data.seed) for data in full}
    if set(metric_specs) != full_seeds:
        raise ValueError(
            f"Full evaluation metrics must cover exactly Full seeds {sorted(full_seeds)}, got {sorted(metric_specs)}."
        )
    rows: list[dict[str, Any]] = []
    for data in full:
        metric_rows = load_evaluation_ssim(metric_specs[data.seed], seed=data.seed)
        if int(data.dataset_ids.size) != int(expected_test_count):
            raise ValueError(
                f"Full seed {data.seed} diagnostics have {data.dataset_ids.size} records; "
                f"expected {expected_test_count}."
            )
        if len(metric_rows) != int(expected_test_count):
            raise ValueError(
                f"Full seed {data.seed} metrics have {len(metric_rows)} records; "
                f"expected {expected_test_count}."
            )
        metrics_by_identity: dict[tuple[int, int], dict[str, Any]] = {}
        for row in metric_rows:
            identity = (int(row["dataset_id"]), int(row["source_sample_index"]))
            if identity in metrics_by_identity:
                raise ValueError(f"Duplicate evaluation metric identity {identity} in seed {data.seed}.")
            metrics_by_identity[identity] = row
        diagnostic_ids = {
            (int(dataset_id), int(sample_index))
            for dataset_id, sample_index in zip(data.dataset_ids, data.sample_indices)
        }
        metric_ids = set(metrics_by_identity)
        if diagnostic_ids != metric_ids:
            missing = sorted(diagnostic_ids - metric_ids)[:5]
            unexpected = sorted(metric_ids - diagnostic_ids)[:5]
            raise ValueError(
                f"Full seed {data.seed} metrics do not match diagnostic identities; "
                f"missing={missing}, unexpected={unexpected}."
            )
        for index, (dataset_id, sample_index) in enumerate(zip(data.dataset_ids, data.sample_indices)):
            identity = (int(dataset_id), int(sample_index))
            metric = metrics_by_identity.get(identity)
            if metric is None:
                raise KeyError(f"Missing SSIM metric for diagnostic identity {identity} and seed {data.seed}.")
            rows.append(
                {
                    "seed": data.seed,
                    "dataset_id": identity[0],
                    "dataset_name": str(metric["dataset_name"]),
                    "source_sample_index": identity[1],
                    "ssim": float(metric["ssim"]),
                    "all_modalities_available": bool(data.availability[index].all()),
                    "modality_availability": [bool(value) for value in data.availability[index]],
                }
            )
    return rows


def run_condition_contract_visualization(
    *,
    config: str | Path,
    full_checkpoints: Sequence[tuple[int, str | Path]],
    no_contract_checkpoints: Sequence[tuple[int, str | Path]],
    output_dir: str | Path,
    device: str | torch.device = "cpu",
    batch_size: int = 64,
    max_samples: int | None = None,
    gallery_size: int = 128,
    galleries_per_subset: int = 10,
    gallery_seed: int = 42,
    full_evaluation_metrics: Sequence[tuple[int, str | Path]] | None = None,
    expected_test_record_count: int = 33_600,
    fit_probes: bool = False,
    probe_max_batches: int | None = None,
) -> dict[str, Any]:
    """Generate publication assets after all required checkpoint paths are available."""
    output = _ensure_dir(_resolve_path(output_dir))
    config_path = _resolve_path(config)
    full = [
        collect_seed_diagnostics(
            config_path=config_path,
            checkpoint_path=checkpoint,
            seed=seed,
            device=device,
            batch_size=batch_size,
            max_samples=max_samples,
        )
        for seed, checkpoint in full_checkpoints
    ]
    if not full:
        raise ValueError("At least one Full checkpoint is required.")
    no_contract = [
        collect_seed_diagnostics(
            config_path=config_path,
            checkpoint_path=checkpoint,
            seed=seed,
            device=device,
            batch_size=batch_size,
            max_samples=max_samples,
        )
        for seed, checkpoint in no_contract_checkpoints
    ]
    if no_contract and {data.seed for data in full} != {data.seed for data in no_contract}:
        raise ValueError("Full and no-contract diagnostics must contain the same model seeds.")
    if max_samples is not None and int(max_samples) < gallery_size:
        raise ValueError("max_samples must be at least gallery_size.")
    matrices = {
        role: _mean_matrix(
            retrieval_matrix(
                data,
                role=role,
                gallery_size=gallery_size,
                galleries_per_subset=galleries_per_subset,
                seed=gallery_seed + data.seed,
            )
            for data in full
        )
        for role in ROLE_TO_HEAD
    }
    anchor_values: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    anchor_rows: list[dict[str, Any]] = []
    for role in ROLE_TO_HEAD:
        matched_parts: list[np.ndarray] = []
        shuffled_parts: list[np.ndarray] = []
        for data in full:
            matched = _anchor_scores(data, role, shuffled=False)
            shuffled = _anchor_scores(data, role, shuffled=True)
            matched_parts.append(matched)
            shuffled_parts.append(shuffled)
            anchor_rows.extend(
                {"seed": data.seed, "role": role, "sample_row": index, "matched": float(match), "shuffled": float(shuffle)}
                for index, (match, shuffle) in enumerate(zip(matched, shuffled))
            )
        anchor_values[role] = (np.concatenate(matched_parts), np.concatenate(shuffled_parts))
    role_rows = [
        {"regime": "full", "seed": data.seed, "eta_b": float(data.eta_b.mean()), "eta_s": float(data.eta_s.mean()), "chi_bs": float(data.chi_bs.mean())}
        for data in full
    ] + [
        {"regime": "no_contract", "seed": data.seed, "eta_b": float(data.eta_b.mean()), "eta_s": float(data.eta_s.mean()), "chi_bs": float(data.chi_bs.mean())}
        for data in no_contract
    ]
    selection_rows: list[dict[str, Any]] = []
    selected_samples: list[dict[str, Any]] = []
    routing_matrices: dict[str, np.ndarray] | None = None
    routing_rows: list[dict[str, Any]] = []
    routing_labels: list[str] | None = None
    routing_status = "pending_evaluation_metrics"
    full_metric_specs = list(full_evaluation_metrics or [])
    split_audit: dict[str, Any] | None = None
    if full_metric_specs:
        _validate_formal_full_seed_set([data.seed for data in full])
        if len(full[0].dataset_names) != EXPECTED_OPENFWI_SUBSETS:
            raise ValueError(
                f"Formal Figure 3 routing requires {EXPECTED_OPENFWI_SUBSETS} configured OpenFWI subsets, "
                f"got {len(full[0].dataset_names)}."
            )
        split_audit = audit_formal_openfwi_split(OmegaConf.load(config_path), expected_test_count=expected_test_record_count)
        selection_rows = _build_selection_records(
            full,
            full_metric_specs,
            expected_test_count=expected_test_record_count,
        )
        full_seeds = tuple(data.seed for data in full)
        selected_samples = select_best_records_by_mean_ssim(selection_rows, required_seeds=full_seeds)
        expected_dataset_ids = set(range(EXPECTED_OPENFWI_SUBSETS))
        selected_dataset_ids = {int(row["dataset_id"]) for row in selected_samples}
        if selected_dataset_ids != expected_dataset_ids:
            raise ValueError(
                f"Best-sample selection must cover all configured OpenFWI subsets {sorted(expected_dataset_ids)}, "
                f"got {sorted(selected_dataset_ids)}."
            )
        checkpoint_sha256 = {
            int(data.seed): str(data.checkpoint_info.get("sha256") or "")
            for data in full
        }
        for row in selected_samples:
            row["checkpoint_sha256"] = {
                str(seed): checkpoint_sha256.get(int(seed), "")
                for seed in full_seeds
            }
        routing_matrices, routing_rows = build_routing_matrices(full, selected_samples)
        routing_labels = [str(row["dataset_name"]) for row in selected_samples]
        routing_status = "complete"
    spatial_examples: list[dict[str, Any]] | None = None
    probes: ProbeWeights | None = None
    appendix_spatial_files: list[Path] = []
    if fit_probes:
        first = full[0]
        probes = fit_posthoc_probes(
            config_path=config_path,
            checkpoint_path=first.checkpoint,
            device=device,
            batch_size=batch_size,
            max_batches=probe_max_batches,
        )
        spatial_examples = extract_spatial_examples(
            config_path=config_path,
            checkpoint_path=first.checkpoint,
            diagnostics=first,
            probes=probes,
            device=device,
            batch_size=batch_size,
        )
        appendix_spatial_files = plot_spatial_probe_figure(output, spatial_examples)
    files = [
        _write_retrieval_csv(output, matrices),
        _write_anchor_csv(output, anchor_rows),
        _write_role_csv(output, role_rows),
        _write_best_sample_csv(
            output,
            selected_samples,
            tuple(data.seed for data in full),
            checkpoint_sha256={
                int(data.seed): str(data.checkpoint_info.get("sha256") or "")
                for data in full
            },
        ),
        _write_routing_csv(output, routing_rows),
        *appendix_spatial_files,
        *plot_figure3(
            output_dir=output,
            retrieval_matrices=matrices,
            anchor_values=anchor_values,
            role_rows=role_rows,
            spatial_examples=spatial_examples,
            routing_matrices=routing_matrices,
            routing_labels=routing_labels,
        ),
    ]
    metadata = {
        "config": str(config_path),
        "device": str(device),
        "batch_size": int(batch_size),
        "max_samples": max_samples,
        "gallery_size": int(gallery_size),
        "galleries_per_subset": int(galleries_per_subset),
        "gallery_seed": int(gallery_seed),
        "expected_test_record_count": int(expected_test_record_count),
        "split_audit": split_audit,
        "routing_status": routing_status,
        "full_evaluation_metrics": {
            str(seed): {
                "path": str(path),
                "sha256": _sha256_file(path),
                "rows": sum(1 for _ in Path(_resolve_path(path)).open(encoding="utf-8")) - 1,
            }
            for seed, path in full_metric_specs
        },
        "selected_samples": selected_samples,
        "appendix_spatial_probe_outputs": [str(path) for path in appendix_spatial_files],
        "chance_recall_at_1": 1.0 / float(gallery_size),
        "full_checkpoints": [_seed_metadata(data) for data in full],
        "no_contract_checkpoints": [_seed_metadata(data) for data in no_contract],
        "spatial_examples": [{key: value for key, value in example.items() if key not in {"target", "anchor_b", "probe_b", "anchor_s", "probe_s"}} for example in spatial_examples or []],
        "probes": None if probes is None else {
            "background_alpha": probes.background_alpha,
            "structural_alpha": probes.structural_alpha,
            "background_validation_mse": probes.background_validation_mse,
            "structural_validation_mse": probes.structural_validation_mse,
        },
        "outputs": [str(path) for path in files],
    }
    metadata_path = output / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    files.append(metadata_path)
    return metadata


def _audit_single_checkpoint_inputs(
    *,
    config_path: Path,
    manifest_path: Path,
    metrics_path: Path,
    expected_test_record_count: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate the canonical split and evaluation identities before inference."""
    conf = OmegaConf.load(config_path)
    split_audit = audit_formal_openfwi_split(
        conf,
        manifest_path=manifest_path,
        expected_test_count=expected_test_record_count,
    )
    manifest_rows = _read_identity_csv(
        manifest_path,
        required_fields=("dataset_id", "dataset_name", "source_sample_index"),
    )
    metric_rows = _read_identity_csv(
        metrics_path,
        required_fields=("dataset_id", "dataset_name", "source_sample_index"),
    )
    if len(metric_rows) != int(expected_test_record_count):
        raise ValueError(
            f"Formal merged metrics contain {len(metric_rows)} rows; expected {expected_test_record_count}."
        )
    manifest_ids = _identity_set(manifest_rows)
    metric_ids = _identity_set(metric_rows)
    if manifest_ids != metric_ids:
        missing = sorted(manifest_ids - metric_ids)[:5]
        unexpected = sorted(metric_ids - manifest_ids)[:5]
        raise ValueError(f"Manifest/metrics identity mismatch; missing={missing}, unexpected={unexpected}.")
    return split_audit, manifest_rows, metric_rows


def _diagnostic_identity_set(data: SeedDiagnostics) -> set[tuple[int, str, int]]:
    rows = [
        {
            "dataset_id": int(dataset_id),
            "dataset_name": _dataset_name(data, int(dataset_id)),
            "source_sample_index": int(source_index),
        }
        for dataset_id, source_index in zip(data.dataset_ids, data.sample_indices)
    ]
    return _identity_set(rows)


def _single_anchor_scores(data: SeedDiagnostics, role: str, shuffled: bool) -> tuple[np.ndarray, np.ndarray]:
    """Return anchor scores and the retained sample rows, including availability audit."""
    anchor = data.anchor_embeddings[role]
    if shuffled:
        anchor = anchor[strict_derangement_indices(anchor.shape[0])]
    values = np.zeros(anchor.shape[0], dtype=np.float32)
    counts = np.zeros(anchor.shape[0], dtype=np.float32)
    for modality_idx, modality in enumerate(STANDARD_MODALITIES):
        observed = data.availability[:, modality_idx]
        cosine = np.sum(data.embeddings[role][modality] * anchor, axis=1)
        values[observed] += cosine[observed]
        counts[observed] += 1.0
    valid = counts > 0
    return values[valid] / counts[valid], np.flatnonzero(valid)


def run_single_checkpoint_diagnostic(
    *,
    config: str | Path,
    checkpoint: str | Path,
    manifest: str | Path,
    metrics: str | Path,
    output_dir: str | Path,
    device: str | torch.device = "cpu",
    batch_size: int = 64,
    gallery_size: int = 128,
    galleries_per_subset: int = 10,
    gallery_seed: int = 42,
    expected_test_record_count: int = 33_600,
) -> dict[str, Any]:
    """Generate Figure 3 from one existing Full checkpoint without training."""
    output = _ensure_dir(_resolve_path(output_dir))
    config_path = _resolve_path(config)
    checkpoint_path = _resolve_path(checkpoint)
    manifest_path = _resolve_path(manifest)
    metrics_path = _resolve_path(metrics)
    split_audit, manifest_rows, metric_rows = _audit_single_checkpoint_inputs(
        config_path=config_path,
        manifest_path=manifest_path,
        metrics_path=metrics_path,
        expected_test_record_count=expected_test_record_count,
    )
    data = collect_seed_diagnostics(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        seed=0,
        device=device,
        batch_size=batch_size,
        max_samples=None,
    )
    if int(data.dataset_ids.size) != int(expected_test_record_count):
        raise ValueError(
            f"Checkpoint diagnostics collected {data.dataset_ids.size} records; expected {expected_test_record_count}."
        )
    manifest_ids = _identity_set(manifest_rows)
    diagnostic_ids = _diagnostic_identity_set(data)
    if diagnostic_ids != manifest_ids:
        missing = sorted(manifest_ids - diagnostic_ids)[:5]
        unexpected = sorted(diagnostic_ids - manifest_ids)[:5]
        raise ValueError(f"Diagnostic/manifest identity mismatch; missing={missing}, unexpected={unexpected}.")
    if int(data.checkpoint_info.get("epoch") or -1) != 99:
        raise ValueError(f"Figure 3 single-checkpoint mode requires epoch 99, got {data.checkpoint_info.get('epoch')!r}.")
    if len(data.dataset_names) != EXPECTED_OPENFWI_SUBSETS:
        raise ValueError(
            f"Figure 3 requires {EXPECTED_OPENFWI_SUBSETS} configured OpenFWI subsets, got {len(data.dataset_names)}."
        )

    gallery_provenance = build_retrieval_gallery_provenance(
        data,
        gallery_size=gallery_size,
        galleries_per_subset=galleries_per_subset,
        seed=gallery_seed,
    )
    expected_gallery_count = EXPECTED_OPENFWI_SUBSETS * int(galleries_per_subset)
    if len(gallery_provenance) != expected_gallery_count:
        raise ValueError(
            f"Expected {expected_gallery_count} retrieval galleries, got {len(gallery_provenance)}."
        )
    matrices = {
        role: retrieval_matrix(
            data,
            role=role,
            gallery_size=gallery_size,
            galleries_per_subset=galleries_per_subset,
            seed=gallery_seed,
        )
        for role in ROLE_TO_HEAD
    }
    for role, matrix in matrices.items():
        if matrix.shape != (len(STANDARD_MODALITIES), len(STANDARD_MODALITIES)):
            raise ValueError(f"{role} retrieval matrix has unexpected shape {matrix.shape}.")
        if not np.isnan(np.diag(matrix)).all() or not np.allclose(matrix, matrix.T, equal_nan=True):
            raise ValueError(f"{role} retrieval matrix is not symmetric with an empty diagonal.")
        off_diagonal = matrix[~np.eye(matrix.shape[0], dtype=bool)]
        if not np.isfinite(off_diagonal).all() or ((off_diagonal < 0) | (off_diagonal > 1)).any():
            raise ValueError(f"{role} retrieval matrix contains Recall@1 outside [0, 1].")

    anchor_values: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    anchor_rows: list[dict[str, Any]] = []
    bootstrap_settings: dict[str, Any] = {}
    for role in ROLE_TO_HEAD:
        matched, matched_rows = _single_anchor_scores(data, role, shuffled=False)
        shuffled, shuffled_rows = _single_anchor_scores(data, role, shuffled=True)
        if not np.array_equal(matched_rows, shuffled_rows) or matched.size != int(data.dataset_ids.size):
            raise ValueError(f"Anchor calibration did not produce one score for every held-out record in {role} role.")
        anchor_values[role] = (matched, shuffled)
        for row_index, (match, shuffle) in enumerate(zip(matched, shuffled)):
            anchor_rows.append(
                {
                    "evidence_scope": "single_checkpoint_diagnostic",
                    "dataset_id": int(data.dataset_ids[row_index]),
                    "dataset_name": _dataset_name(data, int(data.dataset_ids[row_index])),
                    "source_sample_index": int(data.sample_indices[row_index]),
                    "sample_row": int(row_index),
                    "role": role,
                    "matched": float(match),
                    "shuffled": float(shuffle),
                }
            )
        delta, lower, upper = _bootstrap_mean_difference(
            matched,
            shuffled,
            seed=17 if role == "background" else 29,
        )
        bootstrap_settings[role] = {
            "repeats": 1000,
            "seed": 17 if role == "background" else 29,
            "cluster": "sample_row",
            "delta_mean": delta,
            "delta_ci95": [lower, upper],
            "matched_median": float(np.median(matched)),
            "shuffled_median": float(np.median(shuffled)),
        }

    role_rows = single_checkpoint_role_rows(data)
    routing_matrices, routing_raw_rows, routing_summary_rows = aggregate_complete_modality_routing(data)
    if set(data.dataset_names) != set(str(row["dataset_name"]) for row in manifest_rows):
        raise ValueError("Configured OpenFWI subset names do not match formal manifest names.")
    if set(np.unique(data.dataset_ids).astype(int).tolist()) != set(range(EXPECTED_OPENFWI_SUBSETS)):
        raise ValueError("Formal routing diagnostics must contain all eight dataset ids.")
    routing_labels = [str(name) for name in data.dataset_names]

    files = [
        _write_retrieval_csv(output, matrices),
        _write_single_anchor_csv(output, anchor_rows),
        _write_single_role_csv(output, role_rows),
        _write_single_routing_csv(output, routing_raw_rows, routing_summary_rows),
        _write_gallery_provenance_csv(output, gallery_provenance),
        *_plot_single_checkpoint_figure3(
            output_dir=output,
            anchor_values=anchor_values,
            role_rows=role_rows,
            dataset_labels=routing_labels,
        ),
    ]
    dataset_counts = {
        _dataset_name(data, int(dataset_id)): int(count)
        for dataset_id, count in zip(*np.unique(data.dataset_ids, return_counts=True))
    }
    complete_counts = {
        _dataset_name(data, int(dataset_id)): int(
            ((data.dataset_ids == dataset_id) & data.availability.all(axis=1)).sum()
        )
        for dataset_id in sorted(np.unique(data.dataset_ids).astype(int).tolist())
    }
    metadata = build_single_checkpoint_metadata(
        checkpoint_path=checkpoint_path,
        config_path=config_path,
        manifest_path=manifest_path,
        metrics_path=metrics_path,
        checkpoint_info=data.checkpoint_info,
        sample_counts={
            "manifest": len(manifest_rows),
            "metrics": len(metric_rows),
            "diagnostics": int(data.dataset_ids.size),
            "complete_modality": int(data.availability.all(axis=1).sum()),
        },
        split_audit=split_audit,
        gallery_config={
            "gallery_count": len(gallery_provenance),
            "gallery_size": int(gallery_size),
            "galleries_per_subset": int(galleries_per_subset),
            "seed": int(gallery_seed),
            "same_subset": True,
            "complete_modalities_only": True,
            "provenance_csv": str(output / "retrieval_gallery_provenance.csv"),
        },
    )
    metadata.update(
        {
            "checkpoint_epoch": int(data.checkpoint_info["epoch"]),
            "dataset_names": routing_labels,
            "dataset_counts": dataset_counts,
            "complete_modality_counts": complete_counts,
            "anchor_calibration": {
                "projector_spaces": "training-objective role projectors",
                "matched_vs_strict_derangement": True,
                "derangement_offset": 1,
                "derangement_size": int(data.dataset_ids.size),
                "fixed_points": 0,
                "training_only_anchors": True,
                "bootstrap": bootstrap_settings,
            },
            "role_diagnostics": {
                "sample_level_rows": len(role_rows),
                "metrics": ["eta_B", "eta_S", "chi_BS"],
                "causal_interpretation": False,
            },
            "routing": {
                "source": "encoder forward routing_weights",
                "static_modality_priors_used": False,
                "aggregation": "mean over all complete-modality held-out records per dataset",
                "raw_rows": len(routing_raw_rows),
                "summary_rows": len(routing_summary_rows),
                "matrix_shapes": {role: list(matrix.shape) for role, matrix in routing_matrices.items()},
                "nonnegative": True,
                "available_modalities_sum_to_one": True,
                "missing_modalities_zero": True,
            },
            "outputs": [str(path) for path in files],
        }
    )
    metadata_path = output / "metadata.json"
    metadata["outputs"].append(str(metadata_path))
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Figure 3 condition-contract diagnostics.")
    parser.add_argument("--mode", choices=("formal", "single_checkpoint_diagnostic"), default="formal")
    parser.add_argument("--config", required=True)
    parser.add_argument("--full-checkpoints", default="", help="Comma-separated seed=checkpoint pairs.")
    parser.add_argument("--no-contract-checkpoints", default="", help="Comma-separated seed=checkpoint pairs.")
    parser.add_argument("--checkpoint", default="", help="One existing epoch-99 checkpoint for single mode.")
    parser.add_argument("--manifest", default="", help="Canonical held-out manifest for single mode.")
    parser.add_argument("--metrics", default="", help="Merged formal metrics CSV for single mode.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu", help="Use cuda:3 for a bounded visualization run.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--gallery-size", type=int, default=128)
    parser.add_argument("--galleries-per-subset", type=int, default=10)
    parser.add_argument("--gallery-seed", type=int, default=42)
    parser.add_argument(
        "--full-evaluation-metrics",
        default="",
        help="Comma-separated seed=metrics.csv pairs used for deterministic best-SSIM routing samples.",
    )
    parser.add_argument("--expected-test-record-count", type=int, default=33_600)
    parser.add_argument("--fit-probes", action="store_true")
    parser.add_argument("--probe-max-batches", type=int, default=None)
    args = parser.parse_args()
    if args.mode == "single_checkpoint_diagnostic":
        required = {"--checkpoint": args.checkpoint, "--manifest": args.manifest, "--metrics": args.metrics}
        missing = [name for name, value in required.items() if not str(value).strip()]
        if missing:
            parser.error(f"single_checkpoint_diagnostic requires: {', '.join(missing)}")
        if args.max_samples is not None:
            parser.error("single_checkpoint_diagnostic always requires the complete held-out set; omit --max-samples")
        run_single_checkpoint_diagnostic(
            config=args.config,
            checkpoint=args.checkpoint,
            manifest=args.manifest,
            metrics=args.metrics,
            output_dir=args.output_dir,
            device=args.device,
            batch_size=args.batch_size,
            gallery_size=args.gallery_size,
            galleries_per_subset=args.galleries_per_subset,
            gallery_seed=args.gallery_seed,
            expected_test_record_count=args.expected_test_record_count,
        )
        return
    if not args.full_checkpoints.strip():
        parser.error("formal mode requires --full-checkpoints")
    run_condition_contract_visualization(
        config=args.config,
        full_checkpoints=_parse_checkpoint_specs(args.full_checkpoints),
        no_contract_checkpoints=_parse_checkpoint_specs(args.no_contract_checkpoints) if args.no_contract_checkpoints else [],
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        gallery_size=args.gallery_size,
        galleries_per_subset=args.galleries_per_subset,
        gallery_seed=args.gallery_seed,
        full_evaluation_metrics=_parse_checkpoint_specs(args.full_evaluation_metrics) if args.full_evaluation_metrics else [],
        expected_test_record_count=args.expected_test_record_count,
        fit_probes=args.fit_probes,
        probe_max_batches=args.probe_max_batches,
    )


if __name__ == "__main__":
    main()
