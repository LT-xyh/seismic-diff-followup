"""Manifest-safe OpenFWI missing-modality evaluation for Table 2.

This runner deliberately keeps the six Table 2 methods in one protocol while
allowing the five benchmark wrappers and PD-BG-RFM to use their native model
implementations and normalization profiles.  It never changes the dataset
list before the global OpenFWI split is materialized.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from bg_pdr_fm.data import batch_to_device, collate_bg_samples, validate_bg_batch
from bg_pdr_fm.data.datasets import OpenFWIBGDataset
from bg_pdr_fm.evaluation.benchmark_metrics import (
    compute_velocity_metrics,
    count_parameters,
    count_trainable_parameters,
    time_prediction,
)
from bg_pdr_fm.evaluation.missing_modalities import MissingModalityProtocol
from bg_pdr_fm.evaluation.visualize_table2_qualitative import (
    EXPECTED_GLOBAL_HELD_OUT_RECORDS,
    MethodSpec,
    checkpoint_provenance,
    default_method_specs,
    load_manifest_metadata,
    load_method_openfwi_datasets,
    load_model_runtime,
)
from bg_pdr_fm.runtime import configure_torch_runtime


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "logs/bg_pdr_fm/aaai27/eval_table2_missing_modalities_global_heldout"
DEFAULT_PAPER_OUTPUT_ROOT = REPO_ROOT / "logs/bg_pdr_fm/aaai27/eval_table2_missing_modalities_rms_only_global_heldout"
DEFAULT_RADAR_OUTPUT_ROOT = REPO_ROOT / "logs/bg_pdr_fm/aaai27/eval_missing_modalities_radar_global_heldout"
DEFAULT_SINGLE_ONLY_OUTPUT_ROOT = REPO_ROOT / "logs/bg_pdr_fm/aaai27/eval_missing_modalities_single_only_global_heldout"
LEGACY_PROTOCOL_NAME = "legacy-pstm-only"
PAPER_PROTOCOL_NAME = "paper-rms-only"
RADAR_PROTOCOL_NAME = "paper-radar-v1"
SINGLE_ONLY_PROTOCOL_NAME = "paper-single-only-v1"
PAPER_COMMON_MODES = (
    "full",
    "w/o well_log",
    "w/o horizon",
    "w/o rms_vel",
    "w/o well+rms",
)
RADAR_REUSABLE_MODES = ("w/o well_log", "w/o horizon", "RMS only")
SINGLE_ONLY_REUSABLE_MODES = ("RMS only",)
CANONICAL_MANIFEST = (
    REPO_ROOT
    / "logs/bg_pdr_fm/joint_full_contrastive_bgfm_pixelfm_predbg_e100"
    / "evaluation_epoch99_global_heldout_full_parallel_bs64/held_out_manifest.csv"
)
CANONICAL_MANIFEST_SHA256 = "10b22bf60842528f843f07f8a374f9920677e1a0d097f53db83e61e100494876"
OPENFWI_DATASETS = (
    "FlatVelA",
    "FlatVelB",
    "CurveVelA",
    "CurveVelB",
    "FlatFaultA",
    "FlatFaultB",
    "CurveFaultA",
    "CurveFaultB",
)
EXPECTED_DATASET_COUNTS = {
    "CurveFaultA": 5440,
    "CurveFaultB": 5401,
    "CurveVelA": 2958,
    "CurveVelB": 3069,
    "FlatFaultA": 5471,
    "FlatFaultB": 5351,
    "FlatVelA": 2930,
    "FlatVelB": 2980,
}
EXPECTED_METHOD_NAMES = (
    "InversionNet",
    "VelocityGAN",
    "UPFWI",
    "Auto-Linear",
    "Latent U-Net (Large)",
    "PD-BG-RFM",
)
METRIC_KEYS = ("mae", "rmse", "ssim", "mae_l", "mae_h")
METHOD_SLUGS = {
    "InversionNet": "inversion_net",
    "VelocityGAN": "velocity_gan",
    "UPFWI": "upfwi",
    "Auto-Linear": "auto_linear",
    "Latent U-Net (Large)": "latent_unet_large",
    "PD-BG-RFM": "pd_bg_rfm",
}
EXPECTED_PARAMS = {
    "InversionNet": 24_409_123,
    "VelocityGAN": 25_589_126,
    "UPFWI": 18_996_259,
    "Auto-Linear": 35_470_032,
    "Latent U-Net (Large)": 35_123_429,
}


def protocol_from_name(name: str) -> MissingModalityProtocol:
    protocol_name = str(name).strip().lower()
    if protocol_name == LEGACY_PROTOCOL_NAME:
        return MissingModalityProtocol.aaai27()
    if protocol_name == PAPER_PROTOCOL_NAME:
        return MissingModalityProtocol.aaai27_rms_only()
    if protocol_name == RADAR_PROTOCOL_NAME:
        return MissingModalityProtocol.aaai27_radar()
    if protocol_name == SINGLE_ONLY_PROTOCOL_NAME:
        return MissingModalityProtocol.aaai27_single_modality()
    raise ValueError(f"Unknown missing-modality protocol {name!r}.")


def reusable_modes_for_protocol(protocol_name: str, protocol: MissingModalityProtocol) -> tuple[str, ...]:
    if protocol_name == PAPER_PROTOCOL_NAME:
        allowed = PAPER_COMMON_MODES
    elif protocol_name == RADAR_PROTOCOL_NAME:
        allowed = RADAR_REUSABLE_MODES
    elif protocol_name == SINGLE_ONLY_PROTOCOL_NAME:
        allowed = SINGLE_ONLY_REUSABLE_MODES
    else:
        allowed = ()
    return tuple(mode for mode in protocol.mode_names if mode in allowed)


@dataclass
class MetricAccumulator:
    count: int = 0
    sums: dict[str, float] | None = None

    def add(self, row: Mapping[str, Any]) -> None:
        if self.sums is None:
            self.sums = {key: 0.0 for key in METRIC_KEYS}
        self.count += 1
        for key in METRIC_KEYS:
            value = float(row[key])
            if not math.isfinite(value):
                raise ValueError(f"non-finite {key} while accumulating metrics: {value}")
            self.sums[key] += value

    def means(self) -> dict[str, float]:
        if self.count <= 0 or self.sums is None:
            return {key: float("nan") for key in METRIC_KEYS}
        return {key: self.sums[key] / self.count for key in METRIC_KEYS}


def _metric_row_fields() -> list[str]:
    return [
        "method",
        "dataset_id",
        "dataset_name",
        "source_sample_index",
        "missing_mode",
        *METRIC_KEYS,
        "params",
    ]


def method_specs(repo_root: str | Path = REPO_ROOT) -> tuple[MethodSpec, ...]:
    specs = tuple(default_method_specs(repo_root))
    by_name = {spec.display_name: spec for spec in specs}
    missing = [name for name in EXPECTED_METHOD_NAMES if name not in by_name]
    extra = sorted(set(by_name).difference(EXPECTED_METHOD_NAMES))
    if missing or extra:
        raise ValueError(f"Table 2 method registry mismatch: missing={missing}, extra={extra}")
    return tuple(by_name[name] for name in EXPECTED_METHOD_NAMES)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_path(repo_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _load_canonical_manifest(repo_root: Path) -> tuple[Path, dict[tuple[str, int], int]]:
    path = _resolve_path(repo_root, CANONICAL_MANIFEST)
    if not path.is_file():
        raise FileNotFoundError(f"Canonical held-out manifest not found: {path}")
    digest = _sha256(path)
    if digest != CANONICAL_MANIFEST_SHA256:
        raise ValueError(
            f"Canonical held-out manifest SHA256 mismatch: expected {CANONICAL_MANIFEST_SHA256}, got {digest}"
        )
    manifest = load_manifest_metadata(path, expected_count=EXPECTED_GLOBAL_HELD_OUT_RECORDS)
    counts = Counter(dataset_name for dataset_name, _ in manifest)
    if dict(sorted(counts.items())) != dict(sorted(EXPECTED_DATASET_COUNTS.items())):
        raise ValueError(f"Canonical dataset counts mismatch: {dict(sorted(counts.items()))}")
    return path, manifest


def validate_global_split_membership(
    repo_root: str | Path,
    manifest_path: str | Path,
    manifest: Mapping[tuple[str, int], int],
) -> dict[str, Any]:
    """Rebuild the all-subset split and prove the canonical test manifest is disjoint.

    This is an audit-only operation.  It does not provide records to an
    evaluator, so the formal runner still consumes the fixed manifest rather
    than constructing a new split for each method or worker.
    """

    root = Path(repo_root).resolve()
    base_config_path = root / "bg_pdr_fm/configs/experiments/aaai27/_base_formal.yaml"
    if not base_config_path.is_file():
        raise FileNotFoundError(f"Formal split config not found: {base_config_path}")
    data = OmegaConf.load(base_config_path).data
    dataset_names = tuple(str(name) for name in data.openfwi_datasets)
    common_kwargs = {
        "root_dir": str(data.root_dir),
        "root_candidates": list(data.root_candidates),
        "datasets": dataset_names,
        "use_normalize": str(data.use_normalize),
        "target_shape": None if data.target_shape is None else tuple(data.target_shape),
        "required_modalities": tuple(str(value) for value in data.required_modalities),
        "split_fractions": tuple(float(value) for value in data.split_fractions),
        "split_seed": int(data.split_seed),
        "well_count_range": tuple(int(value) for value in data.well_count_range),
        "well_seed": int(data.well_seed),
        "well_random": False,
        "storage_backend": str(data.storage_backend),
        "lmdb_root": str(data.lmdb_root),
        "normalization_profile": str(data.normalization_profile),
        "normalize_clamp": bool(data.normalize_clamp),
    }

    split_ids: dict[str, set[tuple[int, str, int]]] = {}
    datasets: list[OpenFWIBGDataset] = []
    try:
        for split in ("train", "val", "test"):
            dataset = OpenFWIBGDataset(split=split, **common_kwargs)
            datasets.append(dataset)
            identities = {
                (
                    int(record["dataset_id"]),
                    str(record["dataset_name"]),
                    int(record["sample_index"]),
                )
                for record in dataset.records
            }
            if len(identities) != len(dataset.records):
                raise ValueError(f"duplicate identities in reconstructed {split} split")
            split_ids[split] = identities
    finally:
        for dataset in datasets:
            dataset.close()

    manifest_ids = {
        (int(dataset_id), str(dataset_name), int(source_index))
        for (dataset_name, source_index), dataset_id in manifest.items()
    }
    train_val_overlap = split_ids["train"] & split_ids["val"]
    train_test_overlap = split_ids["train"] & split_ids["test"]
    val_test_overlap = split_ids["val"] & split_ids["test"]
    if train_val_overlap or train_test_overlap or val_test_overlap:
        raise ValueError(
            "global split overlap detected: "
            f"train_val={len(train_val_overlap)}, "
            f"train_test={len(train_test_overlap)}, "
            f"val_test={len(val_test_overlap)}"
        )
    if split_ids["test"] != manifest_ids:
        raise ValueError(
            "canonical manifest does not equal reconstructed global test split: "
            f"missing={len(manifest_ids - split_ids['test'])}, "
            f"unexpected={len(split_ids['test'] - manifest_ids)}"
        )
    return {
        "status": "passed",
        "manifest_path": str(Path(manifest_path).resolve()),
        "dataset_names": list(dataset_names),
        "split_seed": int(data.split_seed),
        "split_fractions": [float(value) for value in data.split_fractions],
        "train_count": len(split_ids["train"]),
        "val_count": len(split_ids["val"]),
        "test_count": len(split_ids["test"]),
        "manifest_count": len(manifest_ids),
        "train_val_overlap": 0,
        "train_test_overlap": 0,
        "val_test_overlap": 0,
        "test_manifest_missing": 0,
        "test_manifest_unexpected": 0,
    }


def _runtime_checkpoint(repo_root: Path, spec: MethodSpec) -> tuple[dict[str, Any], Path]:
    checkpoint = dict(checkpoint_provenance(spec))
    path = _resolve_path(repo_root, str(checkpoint["path"]))
    if not path.is_file():
        raise FileNotFoundError(f"{spec.display_name} checkpoint does not exist: {path}")
    checkpoint["path"] = str(path)
    checkpoint["sha256"] = _sha256(path)
    return checkpoint, path


def _batch_identity_keys(batch: Any, dataset_names: Sequence[str]) -> list[tuple[str, int]]:
    dataset_ids = batch.metadata.get("dataset_id")
    source_indices = batch.metadata.get("sample_index")
    if dataset_ids is None or source_indices is None:
        raise ValueError("Evaluation batch lacks dataset_id/sample_index metadata.")
    keys: list[tuple[str, int]] = []
    for item_idx in range(batch.depth_vel.shape[0]):
        dataset_id = int(round(float(dataset_ids[item_idx].detach().cpu().view(-1)[0])))
        source_index = int(round(float(source_indices[item_idx].detach().cpu().view(-1)[0])))
        if dataset_id < 0 or dataset_id >= len(dataset_names):
            raise ValueError(f"Invalid dataset_id {dataset_id} in evaluation batch.")
        keys.append((str(dataset_names[dataset_id]), source_index))
    return keys


def batch_sampling_seed(base_seed: int, keys: Sequence[tuple[str, int]]) -> int:
    """Derive a stable seed from the complete fixed-order record batch.

    The held-out loader order is identical for every missing mode.  Hashing
    all record identities in a batch therefore gives each corresponding
    sample the same random stream across modes without serializing Ours into
    one-sample inference.
    """

    payload = f"{int(base_seed)}|" + "|".join(f"{name}:{index}" for name, index in keys)
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big") % (2**31)


@contextlib.contextmanager
def _prediction_seed_scope(device: torch.device, seed: int | None):
    if seed is None:
        yield
        return
    cuda_devices: list[int] = []
    if device.type == "cuda" and device.index is not None:
        cuda_devices = [int(device.index)]
    with torch.random.fork_rng(devices=cuda_devices, enabled=True):
        torch.manual_seed(int(seed))
        if cuda_devices:
            torch.cuda.manual_seed(int(seed))
        yield


def validate_identity_rows(
    rows: Sequence[Mapping[str, Any]],
    manifest: Mapping[tuple[str, int], int],
    *,
    expected_modes: Sequence[str],
) -> dict[str, Any]:
    """Validate record identity, mode cardinality, and finite metric values."""

    expected_keys = set(manifest)
    seen: set[tuple[str, str, int]] = set()
    mode_keys: dict[str, set[tuple[str, int]]] = {mode: set() for mode in expected_modes}
    dataset_mode_counts: Counter[tuple[str, str]] = Counter()
    for row in rows:
        mode = str(row.get("missing_mode", ""))
        if mode not in mode_keys:
            raise ValueError(f"unexpected missing mode {mode!r}")
        dataset_name = str(row.get("dataset_name", ""))
        try:
            source_index = int(row["source_sample_index"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("missing or invalid source_sample_index") from exc
        identity = (dataset_name, source_index)
        duplicate_key = (mode, dataset_name, source_index)
        if duplicate_key in seen:
            raise ValueError(f"duplicate held-out identity for mode {mode}: {identity}")
        seen.add(duplicate_key)
        if identity not in expected_keys:
            raise ValueError(f"unexpected held-out identity for mode {mode}: {identity}")
        if row.get("dataset_id") not in {None, ""}:
            try:
                row_dataset_id = int(row["dataset_id"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"missing or invalid dataset_id for {identity}") from exc
            expected_dataset_id = int(manifest[identity])
            if row_dataset_id != expected_dataset_id:
                raise ValueError(
                    f"dataset_id mismatch for {identity}: expected {expected_dataset_id}, got {row_dataset_id}"
                )
        mode_keys[mode].add(identity)
        dataset_mode_counts[(dataset_name, mode)] += 1
        for metric in METRIC_KEYS:
            try:
                value = float(row[metric])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"missing metric {metric} for {identity}") from exc
            if not math.isfinite(value):
                raise ValueError(f"non-finite metric {metric} for {identity}: {value}")
    for mode, identities in mode_keys.items():
        missing = expected_keys.difference(identities)
        extra = identities.difference(expected_keys)
        if missing or extra:
            raise ValueError(
                f"mode {mode} does not match canonical manifest: missing={len(missing)}, extra={len(extra)}"
            )
    expected_dataset_counts = Counter(dataset_name for dataset_name, _ in manifest)
    for mode in expected_modes:
        for dataset_name, expected_count in expected_dataset_counts.items():
            actual = dataset_mode_counts[(dataset_name, mode)]
            if actual != expected_count:
                raise ValueError(
                    f"dataset/mode count mismatch for {dataset_name}/{mode}: "
                    f"expected {expected_count}, got {actual}"
                )
    return {
        "status": "passed",
        "num_rows": len(rows),
        "num_unique_records": len(expected_keys),
        "mode_counts": {mode: len(mode_keys[mode]) for mode in expected_modes},
        "dataset_counts": dict(sorted(expected_dataset_counts.items())),
        "dataset_mode_counts": {
            f"{dataset_name}|{mode}": count
            for (dataset_name, mode), count in sorted(dataset_mode_counts.items())
        },
    }


def load_reusable_common_mode_rows(
    source_dir: str | Path,
    *,
    method_name: str,
    manifest: Mapping[tuple[str, int], int],
    common_modes: Sequence[str],
    checkpoint_sha256: str,
    canonical_manifest_sha256: str,
) -> list[dict[str, str]]:
    """Load legacy rows only after proving their provenance and identities."""

    source = Path(source_dir).resolve()
    metrics_path = source / "metrics.csv"
    run_manifest_path = source / "run_manifest.json"
    if not metrics_path.is_file() or not run_manifest_path.is_file():
        raise FileNotFoundError(f"Reusable evaluation is incomplete under {source}")
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    if str(run_manifest.get("method", "")) != str(method_name):
        raise ValueError(
            f"Reusable method mismatch: expected {method_name}, got {run_manifest.get('method')!r}"
        )
    source_manifest_sha256 = str(run_manifest.get("canonical_manifest_sha256", ""))
    if source_manifest_sha256 != str(canonical_manifest_sha256):
        raise ValueError(
            "Reusable canonical manifest SHA256 mismatch: "
            f"expected {canonical_manifest_sha256}, got {source_manifest_sha256}"
        )
    source_checkpoint_sha256 = str(run_manifest.get("checkpoint", {}).get("sha256", ""))
    if source_checkpoint_sha256 != str(checkpoint_sha256):
        raise ValueError(
            "Reusable checkpoint SHA256 mismatch: "
            f"expected {checkpoint_sha256}, got {source_checkpoint_sha256}"
        )
    source_modes = tuple(str(mode) for mode in run_manifest.get("missing_modes", ()))
    missing_source_modes = sorted(set(common_modes).difference(source_modes))
    if missing_source_modes:
        raise ValueError(f"Reusable output is missing requested modes: {missing_source_modes}")
    rows = _load_rows(metrics_path)
    selected = [row for row in rows if str(row.get("missing_mode", "")) in set(common_modes)]
    for row in selected:
        if str(row.get("method", "")) != str(method_name):
            raise ValueError(f"Reusable metrics row has unexpected method: {row.get('method')!r}")
    validate_identity_rows(selected, manifest, expected_modes=tuple(common_modes))
    return selected


def recover_complete_mode_rows(
    rows: Sequence[Mapping[str, Any]],
    expected_modes: Sequence[str],
    *,
    samples_per_mode: int,
    manifest: Mapping[tuple[str, int], int] | None = None,
) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    """Keep only the contiguous complete mode prefix from an interrupted run.

    A mode is complete only when it has the expected number of unique records,
    finite metrics, and (when supplied) exactly the canonical held-out
    identities.  Once the first incomplete mode is encountered, all later
    rows are discarded because they cannot be safely resumed without knowing
    whether their preceding mode was complete.
    """

    if samples_per_mode <= 0:
        raise ValueError(f"samples_per_mode must be positive, got {samples_per_mode}")
    mode_rows: dict[str, list[dict[str, Any]]] = {mode: [] for mode in expected_modes}
    expected_mode_set = set(expected_modes)
    for row in rows:
        mode = str(row.get("missing_mode", ""))
        if mode not in expected_mode_set:
            raise ValueError(f"unexpected missing mode in resumable output: {mode!r}")
        mode_rows[mode].append(dict(row))

    expected_keys = set(manifest) if manifest is not None else None
    kept: list[dict[str, Any]] = []
    completed: list[str] = []
    for mode in expected_modes:
        candidate = mode_rows[mode]
        if len(candidate) != samples_per_mode:
            break
        identities: list[tuple[str, int]] = []
        valid = True
        for row in candidate:
            try:
                identity = (str(row.get("dataset_name", "")), int(row["source_sample_index"]))
            except (KeyError, TypeError, ValueError):
                valid = False
                break
            identities.append(identity)
            for metric in METRIC_KEYS:
                if metric not in row and expected_keys is None:
                    continue
                try:
                    value = float(row[metric])
                except (KeyError, TypeError, ValueError):
                    valid = False
                    break
                if not math.isfinite(value):
                    valid = False
                    break
            if not valid:
                break
        identity_set = set(identities)
        if not valid or len(identity_set) != samples_per_mode:
            break
        if expected_keys is not None and identity_set != expected_keys:
            break
        kept.extend(candidate)
        completed.append(mode)
    return kept, tuple(completed)


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _rewrite_metric_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    rewrite_path = path.with_name(path.name + ".rewrite")
    with rewrite_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_metric_row_fields())
        writer.writeheader()
        writer.writerows(rows)
    rewrite_path.replace(path)


def _write_group_summary(path: Path, groups: Mapping[Any, MetricAccumulator], params: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group_key in sorted(groups, key=lambda key: tuple(key) if isinstance(key, tuple) else (key,)):
        key_values = group_key if isinstance(group_key, tuple) else (group_key,)
        row = {f"group_{idx}": value for idx, value in enumerate(key_values)}
        row.update({"num_samples": groups[group_key].count, **groups[group_key].means(), "params": params})
        rows.append(row)
    return rows


def _write_summary_csv(path: Path, rows: list[dict[str, Any]], group_fields: Sequence[str]) -> None:
    fields = [*group_fields, "num_samples", *METRIC_KEYS, "params"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            output = {field: row.get(f"group_{idx}", row.get(field, "")) for idx, field in enumerate(group_fields)}
            output.update({field: row[field] for field in ["num_samples", *METRIC_KEYS, "params"]})
            writer.writerow(output)


def _parameter_accounting(runtime: Any, spec: MethodSpec) -> tuple[int, dict[str, int]]:
    if spec.kind == "benchmark":
        metadata = runtime.model.benchmark_metadata()
        params = int(metadata["reported_params"])
        accounting = {
            "inference_generator_params": int(metadata["generator_params"]),
            "discriminator_params": int(metadata.get("discriminator_params", 0)),
            "reported_params": params,
        }
    else:
        params = count_trainable_parameters(runtime.model)
        accounting = {
            "total_wrapper_params": count_parameters(runtime.model),
            "trainable_inference_params": params,
            "reported_params": params,
        }
    expected = EXPECTED_PARAMS.get(spec.display_name)
    if expected is not None and params != expected:
        raise ValueError(f"{spec.display_name} parameter count mismatch: expected {expected}, got {params}")
    return params, accounting


def _build_loader(dataset: Any, conf: Any, batch_size: int, num_workers: int) -> DataLoader:
    kwargs: dict[str, Any] = {
        "num_workers": int(num_workers),
        "pin_memory": bool(OmegaConf.select(conf, "training.pin_memory", default=True)),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(OmegaConf.select(conf, "training.persistent_workers", default=True))
        kwargs["prefetch_factor"] = int(OmegaConf.select(conf, "training.prefetch_factor", default=3))
    loader = DataLoader(dataset, batch_size=int(batch_size), shuffle=False, collate_fn=collate_bg_samples, **kwargs)
    first_batch = next(iter(loader))
    validate_bg_batch(first_batch, context="Table 2 missing-modality first batch")
    return loader


def _write_canonical_manifest(path: Path, manifest: Mapping[tuple[str, int], int]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset_id", "dataset_name", "source_sample_index"])
        writer.writeheader()
        for (dataset_name, source_index), dataset_id in sorted(manifest.items()):
            writer.writerow(
                {
                    "dataset_id": dataset_id,
                    "dataset_name": dataset_name,
                    "source_sample_index": source_index,
                }
            )


def _write_run_manifest(
    output_dir: Path,
    *,
    spec: MethodSpec,
    config_path: Path,
    checkpoint: Mapping[str, Any],
    params: int,
    accounting: Mapping[str, int],
    batch_size: int,
    num_workers: int,
    assigned_gpu: str,
    manifest_path: Path,
    manifest_sha256: str,
    smoke: bool,
    protocol_name: str = LEGACY_PROTOCOL_NAME,
    protocol: MissingModalityProtocol | None = None,
    reused_modes: Sequence[str] = (),
    reuse_source: str = "",
) -> None:
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, capture_output=True, check=False
        ).stdout.strip()
    except OSError:
        git_commit = ""
    selected_protocol = protocol or protocol_from_name(protocol_name)
    payload = {
        "method": spec.display_name,
        "method_kind": "adapted_published_baseline" if spec.kind == "benchmark" else "ours",
        "config": str(config_path),
        "checkpoint": dict(checkpoint),
        "params": params,
        "parameter_accounting": dict(accounting),
        "batch_size": batch_size,
        "num_workers": num_workers,
        "assigned_gpu": assigned_gpu,
        "visible_gpu_environment": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "HIP_VISIBLE_DEVICES": os.environ.get("HIP_VISIBLE_DEVICES", ""),
        },
        "data_name": "openfwi",
        "openfwi_datasets": list(OPENFWI_DATASETS),
        "split": "test",
        "split_seed": 42,
        "split_fractions": [0.7, 0.2, 0.1],
        "canonical_manifest": str(manifest_path),
        "canonical_manifest_sha256": manifest_sha256,
        "protocol": protocol_name,
        "missing_modes": list(selected_protocol.mode_names),
        "reused_modes": list(reused_modes),
        "reuse_source": str(reuse_source),
        "base_seed": 2027,
        "sampling_seed_policy": "batch-keyed canonical record identities for paired modes",
        "smoke": bool(smoke),
        "git_commit": git_commit,
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def validate_method_output(
    output_dir: str | Path,
    manifest: Mapping[tuple[str, int], int],
    *,
    expected_modes: Sequence[str] | None = None,
) -> dict[str, Any]:
    output = Path(output_dir)
    modes = tuple(expected_modes or MissingModalityProtocol.aaai27().mode_names)
    required = (
        "metrics.csv",
        "summary.json",
        "missing_mode_summary.csv",
        "dataset_missing_mode_summary.csv",
        "held_out_manifest.csv",
        "run_manifest.json",
        "eval.log",
    )
    missing_files = [name for name in required if not (output / name).is_file()]
    if missing_files:
        raise ValueError(f"missing required output files: {missing_files}")
    rows = _load_rows(output / "metrics.csv")
    identity = validate_identity_rows(rows, manifest, expected_modes=modes)
    expected_rows = len(manifest) * len(modes)
    if len(rows) != expected_rows:
        raise ValueError(f"metrics row count mismatch: expected {expected_rows}, got {len(rows)}")
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    if int(summary.get("num_samples", -1)) != expected_rows:
        raise ValueError(f"summary num_samples mismatch: {summary.get('num_samples')} != {expected_rows}")
    summary_modes = tuple(str(mode) for mode in summary.get("missing_modes", ()))
    if summary_modes != modes:
        raise ValueError(f"summary missing_modes mismatch: expected {modes}, got {summary_modes}")
    run_manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    run_modes = tuple(str(mode) for mode in run_manifest.get("missing_modes", ()))
    if run_modes != modes:
        raise ValueError(f"run_manifest missing_modes mismatch: expected {modes}, got {run_modes}")
    manifest_rows = _load_rows(output / "held_out_manifest.csv")
    manifest_keys = {(str(row["dataset_name"]), int(row["source_sample_index"])) for row in manifest_rows}
    if manifest_keys != set(manifest):
        raise ValueError("method held_out_manifest.csv does not match canonical manifest")
    dataset_mode_rows = _load_rows(output / "dataset_missing_mode_summary.csv")
    if len(dataset_mode_rows) != len(EXPECTED_DATASET_COUNTS) * len(modes):
        raise ValueError("dataset_missing_mode_summary.csv has an incomplete row set")
    mode_rows = _load_rows(output / "missing_mode_summary.csv")
    if len(mode_rows) != len(modes) or {str(row.get("missing_mode", "")) for row in mode_rows} != set(modes):
        raise ValueError("missing_mode_summary.csv has an incomplete or unexpected mode set")
    return {"status": "passed", **identity, "expected_rows": expected_rows, "params": summary.get("params")}


def run_one_method(
    method_name: str,
    *,
    repo_root: str | Path = REPO_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    batch_size: int = 64,
    num_workers: int = 4,
    max_batches: int | None = None,
    assigned_gpu: str = "",
    device_name: str = "auto",
    smoke: bool = False,
    force: bool = False,
    resume: bool = False,
    protocol_name: str = LEGACY_PROTOCOL_NAME,
    reuse_common_modes_from: str | Path | None = None,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    output_base = Path(output_root).resolve()
    protocol = protocol_from_name(protocol_name)
    if smoke:
        output_base = output_base / "smoke"
    specs = method_specs(root)
    spec = next((item for item in specs if item.display_name == method_name), None)
    if spec is None:
        raise ValueError(f"Unknown Table 2 method {method_name!r}; expected {EXPECTED_METHOD_NAMES}")
    output_dir = output_base / METHOD_SLUGS[method_name]
    manifest_path, manifest = _load_canonical_manifest(root)
    if not smoke and not force:
        try:
            return {
                "status": "skipped",
                "method": method_name,
                **validate_method_output(output_dir, manifest, expected_modes=protocol.mode_names),
            }
        except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError):
            pass
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "eval.log"
    if not resume or not log_path.exists():
        log_path.write_text("", encoding="utf-8")

    def log(message: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}\n"
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(line)
        print(line, end="", flush=True)

    try:
        configure_torch_runtime("medium")
        log(
            f"Preparing {method_name}; protocol={protocol_name}, smoke={smoke}, resume={resume}, "
            f"batch_size={batch_size}, num_workers={num_workers}"
        )
        source_data = {
            "specs": specs,
            "manifest_path": manifest_path,
            "manifest_keys": set(manifest),
            "manifest_dataset_ids": manifest,
        }
        datasets, _, configs, _ = load_method_openfwi_datasets(source_data)
        dataset = datasets[method_name]
        conf = configs[method_name]
        checkpoint, checkpoint_path = _runtime_checkpoint(root, spec)
        runtime_device = torch.device("cpu") if device_name == "cpu" else torch.device("cuda:0")
        if runtime_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA evaluation requested but torch.cuda.is_available() is false")
        runtime = load_model_runtime(spec, conf, checkpoint, runtime_device)
        params, accounting = _parameter_accounting(runtime, spec)
        log(f"Loaded checkpoint {checkpoint_path} with params={params}, device={runtime_device}")
        loader = _build_loader(dataset, conf, batch_size, num_workers)
        dataset_names = tuple(getattr(dataset, "dataset_names", OPENFWI_DATASETS))
        metric_path = output_dir / "metrics.csv.tmp"
        global_acc = MetricAccumulator()
        mode_acc: dict[str, MetricAccumulator] = {mode: MetricAccumulator() for mode in protocol.mode_names}
        dataset_mode_acc: dict[tuple[str, str], MetricAccumulator] = {}
        completed_modes: tuple[str, ...] = ()
        reuse_source = ""
        if resume:
            source_path = metric_path if metric_path.is_file() else output_dir / "metrics.csv"
            existing_rows = _load_rows(source_path) if source_path.is_file() else []
            recovered_rows, completed_modes = recover_complete_mode_rows(
                existing_rows,
                protocol.mode_names,
                samples_per_mode=len(manifest),
                manifest=manifest,
            )
            _rewrite_metric_rows(metric_path, recovered_rows)
            log(
                f"Resume recovered modes={list(completed_modes)}; "
                f"discarded_rows={len(existing_rows) - len(recovered_rows)}"
            )
        elif reuse_common_modes_from:
            common_modes = reusable_modes_for_protocol(protocol_name, protocol)
            if not common_modes:
                raise ValueError(f"Protocol {protocol_name} has no reusable common modes.")
            reuse_root = Path(reuse_common_modes_from).resolve()
            reuse_source_dir = reuse_root / METHOD_SLUGS[method_name]
            recovered_rows = load_reusable_common_mode_rows(
                reuse_source_dir,
                method_name=method_name,
                manifest=manifest,
                common_modes=common_modes,
                checkpoint_sha256=str(checkpoint["sha256"]),
                canonical_manifest_sha256=_sha256(manifest_path),
            )
            completed_modes = common_modes
            reuse_source = str(reuse_root)
            _rewrite_metric_rows(metric_path, recovered_rows)
            log(f"Reused common modes={list(completed_modes)} from {reuse_source_dir}")
        recovered_rows = _load_rows(metric_path) if (resume or reuse_common_modes_from) else []
        for row in recovered_rows:
            global_acc.add(row)
            mode = str(row["missing_mode"])
            mode_acc[mode].add(row)
            dataset_mode_acc.setdefault((str(row["dataset_name"]), mode), MetricAccumulator()).add(row)
        row_count = len(recovered_rows)
        file_mode = "a" if (resume or reuse_common_modes_from) else "w"
        with metric_path.open(file_mode, newline="", encoding="utf-8") as handle, torch.no_grad():
            writer = csv.DictWriter(handle, fieldnames=_metric_row_fields())
            if not (resume or reuse_common_modes_from):
                writer.writeheader()
            for missing_mode in protocol.mode_names:
                if missing_mode in completed_modes:
                    log(f"Skipping recovered mode {missing_mode}")
                    continue
                log(f"Starting mode {missing_mode}")
                for batch_idx, raw_batch in enumerate(loader):
                    if max_batches is not None and batch_idx >= max_batches:
                        break
                    batch = protocol.apply(raw_batch, missing_mode)
                    keys = _batch_identity_keys(batch, dataset_names)
                    device_batch = batch_to_device(batch, runtime_device)
                    seed = batch_sampling_seed(2027, keys) if method_name == "PD-BG-RFM" else None
                    with _prediction_seed_scope(runtime_device, seed):
                        timed = time_prediction(
                            lambda: runtime.model.predict_batch(device_batch), device=runtime_device
                        )
                    prediction = timed.prediction
                    target = device_batch.depth_vel
                    if tuple(prediction.velocity_hat.shape) != tuple(target.shape):
                        raise ValueError(
                            f"{method_name} output shape {tuple(prediction.velocity_hat.shape)} "
                            f"does not match target {tuple(target.shape)}"
                        )
                    if not torch.isfinite(prediction.velocity_hat).all() or not torch.isfinite(target).all():
                        raise ValueError(f"{method_name} produced non-finite tensors in mode {missing_mode}")
                    metrics = compute_velocity_metrics(prediction.velocity_hat, target, runtime.filter)
                    for item_idx, (dataset_name, source_index) in enumerate(keys):
                        row = {
                            "method": method_name,
                            "dataset_id": int(round(float(device_batch.metadata["dataset_id"][item_idx].detach().cpu()))),
                            "dataset_name": dataset_name,
                            "source_sample_index": source_index,
                            "missing_mode": missing_mode,
                            "mae": float(metrics["mae"][item_idx].detach().cpu()),
                            "rmse": float(metrics["rmse"][item_idx].detach().cpu()),
                            "ssim": float(metrics["ssim"][item_idx].detach().cpu()),
                            "mae_l": float(metrics["mae_l"][item_idx].detach().cpu()),
                            "mae_h": float(metrics["mae_h"][item_idx].detach().cpu()),
                            "params": params,
                        }
                        writer.writerow(row)
                        global_acc.add(row)
                        mode_acc[missing_mode].add(row)
                        dataset_mode_acc.setdefault((dataset_name, missing_mode), MetricAccumulator()).add(row)
                        row_count += 1
                    if row_count % 4096 == 0:
                        handle.flush()
                log(f"Finished mode {missing_mode}; rows={row_count}")
        metric_path.replace(output_dir / "metrics.csv")
        if smoke:
            summary = {
                "status": "smoke",
                "method": method_name,
                "num_samples": row_count,
                "params": params,
                "checkpoint": checkpoint,
                "protocol": protocol_name,
                "missing_modes": list(protocol.mode_names),
            }
            (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            log(f"Smoke completed with {row_count} rows")
            return summary

        rows = _load_rows(output_dir / "metrics.csv")
        identity_result = validate_identity_rows(rows, manifest, expected_modes=protocol.mode_names)
        _write_canonical_manifest(output_dir / "held_out_manifest.csv", manifest)
        mode_rows = _write_group_summary(output_dir / "missing_mode_summary.tmp", mode_acc, params)
        dataset_mode_rows = _write_group_summary(output_dir / "dataset_missing_mode_summary.tmp", dataset_mode_acc, params)
        _write_summary_csv(output_dir / "missing_mode_summary.csv", mode_rows, ["missing_mode"])
        _write_summary_csv(output_dir / "dataset_missing_mode_summary.csv", dataset_mode_rows, ["dataset_name", "missing_mode"])
        (output_dir / "missing_mode_summary.tmp").unlink(missing_ok=True)
        (output_dir / "dataset_missing_mode_summary.tmp").unlink(missing_ok=True)
        _write_run_manifest(
            output_dir,
            spec=spec,
            config_path=_resolve_path(root, spec.config_path),
            checkpoint=checkpoint,
            params=params,
            accounting=accounting,
            batch_size=batch_size,
            num_workers=num_workers,
            assigned_gpu=assigned_gpu,
            manifest_path=manifest_path,
            manifest_sha256=_sha256(manifest_path),
            smoke=False,
            protocol_name=protocol_name,
            protocol=protocol,
            reused_modes=completed_modes,
            reuse_source=reuse_source,
        )
        summary = {
            "status": "passed",
            "method": method_name,
            "protocol": protocol_name,
            "num_samples": row_count,
            "params": params,
            "means": global_acc.means(),
            "missing_mode_means": mode_rows,
            "dataset_missing_mode_means": dataset_mode_rows,
            "missing_modes": list(protocol.mode_names),
            "reused_modes": list(completed_modes),
            "reuse_source": reuse_source,
            "checkpoint": checkpoint,
            "parameter_accounting": accounting,
            "held_out_validation": identity_result,
        }
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        result = validate_method_output(output_dir, manifest, expected_modes=protocol.mode_names)
        log(f"Formal evaluation passed: {result}")
        return {"method": method_name, **result}
    except Exception:
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(traceback.format_exc())
        raise


def _write_combined_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_radar_results(
    *,
    repo_root: str | Path = REPO_ROOT,
    output_root: str | Path = DEFAULT_RADAR_OUTPUT_ROOT,
) -> dict[str, Any]:
    """Aggregate the four selected radar scenarios without averaging subsets."""

    root = Path(repo_root).resolve()
    output = Path(output_root).resolve()
    manifest_path, manifest = _load_canonical_manifest(root)
    split_validation = validate_global_split_membership(root, manifest_path, manifest)
    protocol = MissingModalityProtocol.aaai27_radar()
    modes = tuple(protocol.mode_names)

    combined: list[dict[str, Any]] = []
    method_results: dict[str, Any] = {}
    for method_name in EXPECTED_METHOD_NAMES:
        method_dir = output / METHOD_SLUGS[method_name]
        method_results[method_name] = validate_method_output(
            method_dir,
            manifest,
            expected_modes=modes,
        )
        for row in _load_rows(method_dir / "dataset_missing_mode_summary.csv"):
            combined.append({"method": method_name, **row})

    dataset_mode_path = output / "radar_dataset_mode_summary.csv"
    _write_combined_csv(
        dataset_mode_path,
        combined,
        ["method", "dataset_name", "missing_mode", "num_samples", *METRIC_KEYS, "params"],
    )
    report_path = output / "radar_summary.md"
    lines = [
        "# OpenFWI Selected Missing-Input Radar Scenarios",
        "",
        "This output contains four selected scenarios for visualization; it does not replace the complete missing-modality table.",
        "",
        "The omitted `w/o rms_vel` condition remains in the paper-facing RMS-only benchmark output.",
        "",
        "| Scenario | Meaning |",
        "|---|---|",
        "| w/o well_log | PSTM + horizon + RMS |",
        "| w/o horizon | PSTM + RMS + well |",
        "| w/o PSTM | horizon + RMS + well |",
        "| RMS only | numerical-only RMS condition |",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    validation = {
        "status": "passed",
        "protocol": RADAR_PROTOCOL_NAME,
        "canonical_manifest": str(manifest_path),
        "canonical_manifest_sha256": _sha256(manifest_path),
        "global_split_validation": split_validation,
        "num_methods": len(EXPECTED_METHOD_NAMES),
        "num_modes": len(modes),
        "expected_records_per_method": len(manifest) * len(modes),
        "expected_source_rows": len(EXPECTED_METHOD_NAMES) * len(modes) * len(EXPECTED_DATASET_COUNTS),
        "reused_modes": list(RADAR_REUSABLE_MODES),
        "new_inferred_mode": "w/o PSTM",
        "dataset_mode_summary": str(dataset_mode_path),
        "method_results": method_results,
    }
    (output / "evaluation_validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return validation


def aggregate_global_results(
    *,
    repo_root: str | Path = REPO_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    protocol_name: str = LEGACY_PROTOCOL_NAME,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    output = Path(output_root).resolve()
    if protocol_name == RADAR_PROTOCOL_NAME:
        return aggregate_radar_results(repo_root=root, output_root=output)
    manifest_path, manifest = _load_canonical_manifest(root)
    split_validation = validate_global_split_membership(root, manifest_path, manifest)
    protocol = protocol_from_name(protocol_name)
    modes = tuple(protocol.mode_names)
    if protocol_name == PAPER_PROTOCOL_NAME:
        dataset_mode_path = output / "table2_rms_only_dataset_mode_summary.csv"
        mode_path = output / "table2_rms_only_mode_summary.csv"
        missing_average_path = output / "table2_rms_only_missing_average.csv"
        report_path = output / "table2_rms_only_summary.md"
    elif protocol_name == SINGLE_ONLY_PROTOCOL_NAME:
        dataset_mode_path = output / "single_only_dataset_mode_summary.csv"
        mode_path = output / "single_only_mode_summary.csv"
        missing_average_path = output / "single_only_dataset_average.csv"
        report_path = output / "single_only_summary.md"
    else:
        dataset_mode_path = output / "table2_missing_dataset_mode_summary.csv"
        mode_path = output / "table2_missing_mode_summary.csv"
        missing_average_path = output / "table2_missing_only_dataset_average.csv"
        report_path = output / "table2_missing_summary.md"

    for method_name in EXPECTED_METHOD_NAMES:
        method_dir = output / METHOD_SLUGS[method_name]
        validate_method_output(method_dir, manifest, expected_modes=modes)
    combined_fields = ["method", "dataset_name", "missing_mode", "num_samples", *METRIC_KEYS, "params"]
    combined: list[dict[str, Any]] = []
    for method_name in EXPECTED_METHOD_NAMES:
        method_dir = output / METHOD_SLUGS[method_name]
        for row in _load_rows(method_dir / "dataset_missing_mode_summary.csv"):
            combined.append({"method": method_name, **row})
    _write_combined_csv(dataset_mode_path, combined, combined_fields)

    mode_combined: list[dict[str, Any]] = []
    for method_name in EXPECTED_METHOD_NAMES:
        method_dir = output / METHOD_SLUGS[method_name]
        for row in _load_rows(method_dir / "missing_mode_summary.csv"):
            mode_combined.append({"method": method_name, **row})
    _write_combined_csv(mode_path, mode_combined, ["method", "missing_mode", "num_samples", *METRIC_KEYS, "params"])

    missing_average: list[dict[str, Any]] = []
    average_modes = tuple(mode for mode in modes if mode != "full")
    for method_name in EXPECTED_METHOD_NAMES:
        rows = [row for row in combined if row["method"] == method_name and row["missing_mode"] in average_modes]
        for dataset_name in sorted(EXPECTED_DATASET_COUNTS):
            subset = [row for row in rows if row["dataset_name"] == dataset_name]
            if len(subset) != len(average_modes):
                raise ValueError(f"Missing-mode average is incomplete for {method_name}/{dataset_name}")
            missing_average.append(
                {
                    "method": method_name,
                    "dataset_name": dataset_name,
                    "missing_modes_averaged": len(subset),
                    "num_samples": sum(int(row["num_samples"]) for row in subset),
                    **{key: sum(float(row[key]) for row in subset) / len(subset) for key in METRIC_KEYS},
                    "params": subset[0]["params"],
                }
            )
    _write_combined_csv(
        missing_average_path,
        missing_average,
        ["method", "dataset_name", "missing_modes_averaged", "num_samples", *METRIC_KEYS, "params"],
    )

    contribution_path: Path | None = None
    if protocol_name == PAPER_PROTOCOL_NAME:
        contribution_path = output / "modality_contribution_summary.csv"
        lookup = {(row["method"], row["dataset_name"], row["missing_mode"]): row for row in combined}
        contribution_rows: list[dict[str, Any]] = []
        for method_name in EXPECTED_METHOD_NAMES:
            for dataset_name in sorted(EXPECTED_DATASET_COUNTS):
                full = lookup[(method_name, dataset_name, "full")]
                structural = lookup[(method_name, dataset_name, "w/o well+rms")]
                numerical = lookup[(method_name, dataset_name, "RMS only")]
                ssim_full = float(full["ssim"])
                ssim_structural = float(structural["ssim"])
                ssim_numerical = float(numerical["ssim"])
                contribution_rows.append(
                    {
                        "method": method_name,
                        "dataset_name": dataset_name,
                        "num_samples": int(full["num_samples"]),
                        "ssim_full": ssim_full,
                        "ssim_rms_only": ssim_numerical,
                        "ssim_structural_only": ssim_structural,
                        "delta_rms": ssim_full - ssim_numerical,
                        "delta_struct": ssim_full - ssim_structural,
                        "synergy_ssim": ssim_full - max(ssim_numerical, ssim_structural),
                        "params": full["params"],
                    }
                )
        _write_combined_csv(
            contribution_path,
            contribution_rows,
            [
                "method",
                "dataset_name",
                "num_samples",
                "ssim_full",
                "ssim_rms_only",
                "ssim_structural_only",
                "delta_rms",
                "delta_struct",
                "synergy_ssim",
                "params",
            ],
        )

    lines = [
        "# Table 2 OpenFWI Missing-Modality Evaluation"
        if protocol_name == LEGACY_PROTOCOL_NAME
        else (
            "# OpenFWI Single-Modality Evaluation"
            if protocol_name == SINGLE_ONLY_PROTOCOL_NAME
            else "# Table 2 OpenFWI RMS-Only Missing-Modality Evaluation"
        ),
        "",
        "Protocol: global eight-subset 70/20/10 split with seed 42; each mode uses the exact canonical test records.",
        "",
        "The first five methods are adapted published architectures. PD-BG-RFM is the current method.",
    ]
    mode_labels = {
        "w/o well+rms": "w/o well+rms (Structural only: PSTM + horizon)",
        "RMS only": "RMS only (Numerical only)",
        "Horizon only": "Horizon only",
        "Well only": "Well only",
        "PSTM only": "PSTM only",
    }
    for mode in modes:
        lines.extend(["", f"## {mode_labels.get(mode, mode)}", ""])
        for dataset_name in sorted(EXPECTED_DATASET_COUNTS):
            lines.extend(
                [
                    f"### {dataset_name}",
                    "",
                    "| Method | MAE | RMSE | SSIM | MAE_L | MAE_H | Params | N |",
                    "|---|---:|---:|---:|---:|---:|---:|---:|",
                ]
            )
            selected = [
                row for row in combined if row["dataset_name"] == dataset_name and row["missing_mode"] == mode
            ]
            selected_by_method = {row["method"]: row for row in selected}
            for method_name in EXPECTED_METHOD_NAMES:
                row = selected_by_method[method_name]
                lines.append(
                    f"| {method_name} | {float(row['mae']):.6f} | {float(row['rmse']):.6f} | "
                    f"{float(row['ssim']):.6f} | {float(row['mae_l']):.6f} | {float(row['mae_h']):.6f} | "
                    f"{row['params']} | {row['num_samples']} |"
                )
    lines.extend(
        [
            "",
            "## Reporting Notes",
            "",
            f"- `{missing_average_path.name}` is a macro average over the selected single-modality conditions and does not replace the dataset-mode tables.",
            "- Metrics use the shared MAE, RMSE, SSIM, MAE_L, and MAE_H implementation.",
            "- No Trainable Params column is reported.",
        ]
    )
    if protocol_name == PAPER_PROTOCOL_NAME:
        lines.extend(
            [
                "",
                "- `w/o well+rms` is the structural-only condition.",
                "- `RMS only` is the numerical-only condition.",
                "- `PSTM only` is intentionally excluded from this paper-facing protocol; legacy results remain in the legacy output tree.",
            ]
        )
    if protocol_name == SINGLE_ONLY_PROTOCOL_NAME:
        lines.extend(
            [
                "",
                "- This is a single-modality diagnostic, not a full-modality or missing-modality average benchmark.",
                "- `RMS only`, `Horizon only`, `Well only`, and `PSTM only` use the same canonical held-out records.",
            ]
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    validation = {
        "status": "passed",
        "protocol": protocol_name,
        "canonical_manifest": str(_resolve_path(root, CANONICAL_MANIFEST)),
        "canonical_manifest_sha256": _sha256(_resolve_path(root, CANONICAL_MANIFEST)),
        "global_split_validation": split_validation,
        "num_methods": len(EXPECTED_METHOD_NAMES),
        "num_modes": len(modes),
        "expected_records_per_method": len(manifest) * len(modes),
        "excluded_legacy_mode": "PSTM only" if protocol_name == PAPER_PROTOCOL_NAME else None,
        "average_modes": list(average_modes),
        "modality_contribution_summary": str(contribution_path) if contribution_path else None,
        "method_results": {
            method_name: validate_method_output(output / METHOD_SLUGS[method_name], manifest, expected_modes=modes)
            for method_name in EXPECTED_METHOD_NAMES
        },
    }
    (output / "evaluation_validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return validation


def _gpu_probe() -> str:
    for command in (("rocm-smi", "--showuse", "--showmemuse"), ("hy-smi",)):
        try:
            result = subprocess.run(command, text=True, capture_output=True, check=False)
        except OSError:
            continue
        if result.returncode == 0:
            return result.stdout
    return "GPU status command unavailable"


def _child_env(gpu: int) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["HIP_VISIBLE_DEVICES"] = str(gpu)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONPATH"] = str(REPO_ROOT)
    return env


def run_all(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.output_root).resolve()
    protocol = protocol_from_name(args.protocol)
    output.mkdir(parents=True, exist_ok=True)
    methods = list(EXPECTED_METHOD_NAMES)
    pending = methods[:]
    requested_gpus = [int(value) for value in str(args.gpus).split(",") if value.strip()]
    if not requested_gpus:
        raise ValueError("--gpus must contain at least one GPU index")
    queue_log = output / "eval_queue.log"
    queue_log.write_text(_gpu_probe() + "\n", encoding="utf-8")
    running: dict[int, tuple[str, subprocess.Popen[str]]] = {}
    states: dict[str, dict[str, Any]] = {}
    while pending or running:
        for gpu in requested_gpus:
            if len(running) >= int(args.max_concurrent) or gpu in running or not pending:
                continue
            method_name = pending.pop(0)
            method_dir = output / METHOD_SLUGS[method_name]
            if not args.force:
                try:
                    manifest = _load_canonical_manifest(REPO_ROOT)[1]
                    result = validate_method_output(method_dir, manifest, expected_modes=protocol.mode_names)
                    states[method_name] = {"status": "skipped", **result}
                    continue
                except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError):
                    pass
            command = [
                sys.executable,
                "-m",
                "bg_pdr_fm.evaluation.run_table2_missing_modality_benchmark",
                "--method",
                method_name,
                "--output-root",
                str(output),
                "--batch-size",
                str(args.batch_size),
                "--num-workers",
                str(args.num_workers),
                "--device",
                "cuda:0",
                "--assigned-gpu",
                str(gpu),
                "--protocol",
                str(args.protocol),
            ]
            if args.max_batches is not None:
                command.extend(["--max-batches", str(args.max_batches)])
            if args.reuse_common_modes_from:
                command.extend(["--reuse-common-modes-from", str(args.reuse_common_modes_from)])
            if args.force:
                command.append("--force")
            if args.resume:
                command.append("--resume")
            process = subprocess.Popen(command, cwd=REPO_ROOT, env=_child_env(gpu))
            running[gpu] = (method_name, process)
            states[method_name] = {"status": "running", "gpu": gpu, "pid": process.pid}
        state = {"pending": pending, "running": states, "time": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
        (output / "eval_queue_state.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        for gpu, (method_name, process) in list(running.items()):
            returncode = process.poll()
            if returncode is None:
                continue
            states[method_name] = {"status": "completed" if returncode == 0 else "failed", "gpu": gpu, "returncode": returncode}
            del running[gpu]
        if running:
            time.sleep(max(1, int(args.poll_seconds)))
    failed = [name for name, state in states.items() if state.get("status") == "failed"]
    (output / "eval_queue_state.json").write_text(
        json.dumps(
            {
                "pending": list(pending),
                "running": states,
                "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if failed:
        raise RuntimeError(f"Table 2 missing-modality methods failed: {failed}")
    return aggregate_global_results(repo_root=REPO_ROOT, output_root=output, protocol_name=args.protocol)


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--method", choices=EXPECTED_METHOD_NAMES)
    group.add_argument("--all", action="store_true")
    parser.add_argument(
        "--protocol",
        choices=(LEGACY_PROTOCOL_NAME, PAPER_PROTOCOL_NAME, RADAR_PROTOCOL_NAME, SINGLE_ONLY_PROTOCOL_NAME),
        default=LEGACY_PROTOCOL_NAME,
    )
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--reuse-common-modes-from", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--gpus", default="7,6,5,4")
    parser.add_argument("--max-concurrent", type=int, default=4)
    parser.add_argument("--poll-seconds", type=int, default=10)
    parser.add_argument("--assigned-gpu", default="")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda:0"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.output_root is None:
        if args.protocol == PAPER_PROTOCOL_NAME:
            args.output_root = str(DEFAULT_PAPER_OUTPUT_ROOT)
        elif args.protocol == RADAR_PROTOCOL_NAME:
            args.output_root = str(DEFAULT_RADAR_OUTPUT_ROOT)
        elif args.protocol == SINGLE_ONLY_PROTOCOL_NAME:
            args.output_root = str(DEFAULT_SINGLE_ONLY_OUTPUT_ROOT)
        else:
            args.output_root = str(DEFAULT_OUTPUT_ROOT)
    if args.reuse_common_modes_from is None:
        if args.protocol == PAPER_PROTOCOL_NAME:
            args.reuse_common_modes_from = str(DEFAULT_OUTPUT_ROOT)
        elif args.protocol == RADAR_PROTOCOL_NAME:
            args.reuse_common_modes_from = str(DEFAULT_PAPER_OUTPUT_ROOT)
        elif args.protocol == SINGLE_ONLY_PROTOCOL_NAME:
            args.reuse_common_modes_from = str(DEFAULT_PAPER_OUTPUT_ROOT)
    if args.all:
        if args.smoke:
            raise ValueError("--all --smoke is not supported; launch each smoke method explicitly")
        return run_all(args)
    return run_one_method(
        args.method,
        output_root=args.output_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_batches=args.max_batches,
        assigned_gpu=args.assigned_gpu,
        device_name=args.device,
        smoke=args.smoke,
        force=args.force,
        resume=args.resume,
        protocol_name=args.protocol,
        reuse_common_modes_from=args.reuse_common_modes_from,
    )


if __name__ == "__main__":
    main()
