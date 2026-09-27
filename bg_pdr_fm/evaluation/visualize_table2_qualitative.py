"""Auditable qualitative selection helpers for the AAAI2027 Table 2 methods.

The module deliberately separates metric-based record selection from model
reconstruction.  The selection layer is deterministic, operates only on
record identities, and writes enough provenance for a later rendering pass to
be independently audited.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import csv
import json
import subprocess
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


RecordKey = tuple[str, int]
ADVANTAGE_CRITERIA = ("margin_l", "margin_h", "margin_s")
METRIC_COLUMNS = ("mae", "rmse", "ssim", "mae_l", "mae_h")
EXPECTED_GLOBAL_HELD_OUT_RECORDS = 33_600
VELOCITY_COLORMAP = "jet"
ERROR_COLORMAP = "inferno"
TABLE2_ERROR_LIMITS = (0.0, 750.0)
GRID_COORDINATE_SYSTEM = "x/z grid index"
REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_VARIANT_PREFIXES = {
    "InversionNet": "adapted_inversion_net.",
    "VelocityGAN": "velocity_gan.",
    "UPFWI": "adapted_upfwi.",
    "Auto-Linear": "adapted_auto_linear_original.",
    "Latent U-Net (Large)": "adapted_gfi_latent_unet.",
}


@dataclass(frozen=True)
class MethodSpec:
    """One Table 2 method and the formal artifacts that define it."""

    display_name: str
    kind: str
    config_path: Path
    metrics_path: Path
    summary_path: Path
    source_index_field: str


@dataclass
class ModelRuntime:
    """A fully loaded Table 2 checkpoint ready for single-record inference."""

    spec: MethodSpec
    model: Any
    filter: Any
    device: Any
    checkpoint: dict[str, Any]


def default_method_specs(repo_root: str | Path = REPO_ROOT) -> tuple[MethodSpec, ...]:
    root = Path(repo_root)
    aaai = root / "logs" / "bg_pdr_fm" / "aaai27"
    return (
        MethodSpec(
            "InversionNet",
            "benchmark",
            root / "bg_pdr_fm/configs/experiments/aaai27/formal_adapted_inversion_net.yaml",
            aaai / "eval_adapted_inversion_net_official_e100/adapted_inversion_net_official_e100/metrics.csv",
            aaai / "eval_adapted_inversion_net_official_e100/adapted_inversion_net_official_e100/summary.json",
            "dataset_sample_index",
        ),
        MethodSpec(
            "VelocityGAN",
            "benchmark",
            root / "bg_pdr_fm/configs/experiments/aaai27/formal_velocity_gan.yaml",
            aaai / "eval_velocity_gan_official_e100/velocity_gan_official_e100/metrics.csv",
            aaai / "eval_velocity_gan_official_e100/velocity_gan_official_e100/summary.json",
            "dataset_sample_index",
        ),
        MethodSpec(
            "UPFWI",
            "benchmark",
            root / "bg_pdr_fm/configs/experiments/aaai27/formal_adapted_upfwi.yaml",
            aaai / "eval_adapted_upfwi_official_e100/adapted_upfwi_official_e100/metrics.csv",
            aaai / "eval_adapted_upfwi_official_e100/adapted_upfwi_official_e100/summary.json",
            "dataset_sample_index",
        ),
        MethodSpec(
            "Auto-Linear",
            "benchmark",
            root / "bg_pdr_fm/configs/experiments/aaai27/formal_adapted_auto_linear_original_paper35m_multimodal.yaml",
            aaai / "eval_adapted_auto_linear_original_paper35m_multimodal/adapted_auto_linear_original_paper35m_multimodal/metrics.csv",
            aaai / "eval_adapted_auto_linear_original_paper35m_multimodal/adapted_auto_linear_original_paper35m_multimodal/summary.json",
            "dataset_sample_index",
        ),
        MethodSpec(
            "Latent U-Net (Large)",
            "benchmark",
            root / "bg_pdr_fm/configs/experiments/aaai27/formal_adapted_gfi_latent_unet_e100.yaml",
            aaai / "eval_adapted_gfi_latent_unet_e100_epoch95/adapted_gfi_latent_unet_e100_epoch95/metrics.csv",
            aaai / "eval_adapted_gfi_latent_unet_e100_epoch95/adapted_gfi_latent_unet_e100_epoch95/summary.json",
            "dataset_sample_index",
        ),
        MethodSpec(
            "PD-BG-RFM",
            "pdr",
            root / "bg_pdr_fm/configs/openfwi_lmdb_joint_full_contrastive_bgfm_pixelfm_predbg_e100.yaml",
            root / "logs/bg_pdr_fm/joint_full_contrastive_bgfm_pixelfm_predbg_e100/evaluation_epoch99_global_heldout_full_parallel_bs64/merged_metrics.csv",
            root / "logs/bg_pdr_fm/joint_full_contrastive_bgfm_pixelfm_predbg_e100/evaluation_epoch99_global_heldout_full_parallel_bs64/summary.json",
            "source_sample_index",
        ),
    )


def _key_from_row(row: Mapping[str, Any]) -> RecordKey:
    return str(row["dataset_name"]), int(row["source_sample_index"])


def load_metric_csv(
    path: str | Path,
    source_index_field: str,
    method_name: str,
) -> dict[RecordKey, dict[str, str]]:
    """Load the full-modality metrics keyed by source dataset identity."""

    metric_path = Path(path)
    if not metric_path.is_file():
        raise FileNotFoundError(f"{method_name} metrics file does not exist: {metric_path}")
    rows: dict[RecordKey, dict[str, str]] = {}
    with metric_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("missing_mode", "full") != "full":
                continue
            if not row.get("dataset_name") or row.get(source_index_field) in {None, ""}:
                raise ValueError(f"{method_name} has no usable record identity in {metric_path}.")
            key = (str(row["dataset_name"]), int(row[source_index_field]))
            if key in rows:
                raise ValueError(f"{method_name} has duplicate full-modality record identity {key}.")
            rows[key] = dict(row)
    if not rows:
        raise ValueError(f"{method_name} has no full-modality rows in {metric_path}.")
    return rows


def load_manifest_metadata(
    path: str | Path,
    *,
    expected_count: int | None = None,
) -> dict[RecordKey, int]:
    """Load canonical dataset IDs for the globally fixed held-out records."""

    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Held-out manifest does not exist: {manifest_path}")
    metadata: dict[RecordKey, int] = {}
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if not row.get("dataset_id") or not row.get("dataset_name") or row.get("source_sample_index") in {None, ""}:
                raise ValueError(f"Held-out manifest has no usable canonical identity: {manifest_path}")
            key = (str(row["dataset_name"]), int(row["source_sample_index"]))
            if key in metadata:
                raise ValueError(f"Held-out manifest has duplicate record identity {key}.")
            metadata[key] = int(row["dataset_id"])
    if not metadata:
        raise ValueError(f"Held-out manifest is empty: {manifest_path}")
    if expected_count is not None and len(metadata) != int(expected_count):
        raise ValueError(
            f"Held-out manifest has {len(metadata)} unique records; expected {int(expected_count)}."
        )
    return metadata


def load_manifest_keys(path: str | Path, *, expected_count: int | None = None) -> set[RecordKey]:
    """Load held-out keys while preserving an optional cardinality assertion."""

    return set(load_manifest_metadata(path, expected_count=expected_count))


def checkpoint_provenance(spec: MethodSpec) -> dict[str, Any]:
    """Read an evaluated checkpoint path and reject incomplete loading evidence."""

    if not spec.summary_path.is_file():
        raise FileNotFoundError(f"{spec.display_name} summary does not exist: {spec.summary_path}")
    summary = json.loads(spec.summary_path.read_text(encoding="utf-8"))
    if spec.kind == "benchmark":
        checkpoint = dict(summary.get("checkpoint") or {})
    else:
        full = [entry for entry in summary.get("checkpoints", []) if entry.get("mode") == "full"]
        if len(full) != 1:
            raise ValueError(f"{spec.display_name} summary must contain exactly one full checkpoint record.")
        checkpoint = dict(full[0])
    if not checkpoint.get("path"):
        raise ValueError(f"{spec.display_name} summary has no checkpoint path.")
    if int(checkpoint.get("missing_keys", 0)) != 0 or int(checkpoint.get("unexpected_keys", 0)) != 0:
        raise ValueError(f"{spec.display_name} checkpoint was not loaded completely: {checkpoint}")
    return checkpoint


def load_formal_metric_sources(repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    """Load and audit the six immutable formal Table 2 metric sources."""

    root = Path(repo_root)
    specs = default_method_specs(root)
    pd_spec = next(spec for spec in specs if spec.kind == "pdr")
    baseline_specs = tuple(spec for spec in specs if spec.kind == "benchmark")
    manifest_path = pd_spec.summary_path.parent / "held_out_manifest.csv"
    manifest_dataset_ids = load_manifest_metadata(
        manifest_path,
        expected_count=EXPECTED_GLOBAL_HELD_OUT_RECORDS,
    )
    manifest_keys = set(manifest_dataset_ids)
    baselines = {
        spec.display_name: load_metric_csv(spec.metrics_path, spec.source_index_field, spec.display_name)
        for spec in baseline_specs
    }
    pd_metrics = load_metric_csv(pd_spec.metrics_path, pd_spec.source_index_field, pd_spec.display_name)
    assert_exact_held_out_alignment(
        manifest_keys,
        baselines,
        pd_metrics,
        manifest_dataset_ids=manifest_dataset_ids,
    )
    return {
        "specs": specs,
        "manifest_path": manifest_path,
        "manifest_keys": manifest_keys,
        "manifest_dataset_ids": manifest_dataset_ids,
        "baselines": baselines,
        "pd_metrics": pd_metrics,
        "checkpoints": {spec.display_name: checkpoint_provenance(spec) for spec in specs},
    }


def _slug(value: str) -> str:
    return "".join(character.lower() if character.isalnum() else "_" for character in value).strip("_")


def selection_manifest_rows(
    *,
    selections: Mapping[str, Sequence[Mapping[str, Any]]],
    formal_rows: Sequence[Mapping[str, Any]],
    baselines: Mapping[str, Mapping[RecordKey, Mapping[str, Any]]],
    pd_metrics: Mapping[RecordKey, Mapping[str, Any]],
    checkpoints: Mapping[str, Mapping[str, Any]],
    manifest_dataset_ids: Mapping[RecordKey, int],
    base_seed: int,
) -> list[dict[str, Any]]:
    """Flatten automated case choices plus all formal metrics into audit rows."""

    formal_by_key = {_record_order(row): row for row in formal_rows}
    rows: list[dict[str, Any]] = []
    for group, cases in selections.items():
        for position, case in enumerate(cases):
            output: dict[str, Any] = {
                "selection_group": group,
                "selection_position": position,
                "selection_status": case["status"],
                "criterion": case["criterion"],
                "dataset_name": case.get("dataset_name", ""),
                "source_sample_index": case.get("source_sample_index"),
            }
            if case["status"] != "selected":
                rows.append(output)
                continue
            key = _record_order(case)
            formal = formal_by_key[key]
            if key not in manifest_dataset_ids:
                raise ValueError(f"Selection record {key} is absent from the held-out manifest metadata.")
            output.update({
                "dataset_id": int(manifest_dataset_ids[key]),
                "formal_margin_l": formal["margin_l"],
                "formal_margin_h": formal["margin_h"],
                "formal_margin_s": formal["margin_s"],
                "pd_visual_seed": record_seed(base_seed, key[0], key[1]),
            })
            for method_name, source in {**dict(baselines), "PD-BG-RFM": pd_metrics}.items():
                metrics = source[key]
                prefix = _slug(method_name)
                for metric in METRIC_COLUMNS:
                    output[f"formal_{prefix}_{metric}"] = metrics[metric]
                output[f"checkpoint_{prefix}"] = checkpoints[method_name]["path"]
            rows.append(output)
    return rows


def write_csv_rows(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({field for row in rows for field in row})
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_formal_selection_audit(
    output_dir: str | Path,
    source_data: Mapping[str, Any],
    *,
    base_seed: int,
) -> dict[str, Any]:
    """Persist a provisional, metric-only audit before any prediction reconstruction."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    formal_rows = build_margin_rows(baselines=source_data["baselines"], pd_metrics=source_data["pd_metrics"])
    selections = select_case_records(formal_rows)
    rows = selection_manifest_rows(
        selections=selections,
        formal_rows=formal_rows,
        baselines=source_data["baselines"],
        pd_metrics=source_data["pd_metrics"],
        checkpoints=source_data["checkpoints"],
        manifest_dataset_ids=source_data["manifest_dataset_ids"],
        base_seed=base_seed,
    )
    csv_path = output / "formal_candidate_selection.csv"
    write_csv_rows(csv_path, rows)
    positive_counts = {
        criterion: sum(float(row[criterion]) > 0 for row in formal_rows)
        for criterion in ADVANTAGE_CRITERIA
    }
    report = {
        "manifest_path": str(source_data["manifest_path"]),
        "num_manifest_records": len(source_data["manifest_keys"]),
        "positive_margin_counts": positive_counts,
        "base_seed": int(base_seed),
        "checkpoints": source_data["checkpoints"],
        "formal_candidate_selection_csv": str(csv_path),
    }
    (output / "formal_metric_audit.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# Table 2 Formal Metric Audit",
        "",
        f"- Held-out manifest: `{source_data['manifest_path']}`",
        f"- Exact aligned records: `{len(source_data['manifest_keys'])}`",
        f"- Positive low-frequency margins: `{positive_counts['margin_l']}`",
        f"- Positive high-frequency margins: `{positive_counts['margin_h']}`",
        f"- Positive SSIM margins: `{positive_counts['margin_s']}`",
        "- This file is metric-only candidate selection. Final panels must use re-inferred arrays and re-computed metrics.",
    ]
    (output / "formal_candidate_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"formal_rows": formal_rows, "selections": selections, "manifest_rows": rows, "report": report}


def _data_protocol_signature(conf: Any) -> dict[str, Any]:
    from omegaconf import OmegaConf

    fields = (
        "data.name",
        "data.use_normalize",
        "data.storage_backend",
        "data.lmdb_root",
        "data.normalization_profile",
        "data.well_seed",
        "data.split_seed",
        "data.split_fractions",
        "data.openfwi_datasets",
    )
    signature: dict[str, Any] = {}
    for field in fields:
        value = OmegaConf.select(conf, field)
        signature[field] = OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value
    return signature


def resolved_normalization_profile(conf: Any) -> str:
    """Mirror ``build_dataset`` so visualization uses each checkpoint's native input scale."""

    from omegaconf import OmegaConf

    profile = str(OmegaConf.select(conf, "data.normalization_profile", default="auto"))
    if profile != "auto":
        return profile
    dataset_name = str(OmegaConf.select(conf, "data.name", default="openfwi"))
    codec_type = str(OmegaConf.select(conf, "model.codec_type", default="simple")).lower()
    return "seismic_global" if codec_type in {"autoencoder", "vae"} else dataset_name


def velocity_to_physical(value: Any, profile_name: str, normalize_mode: str | None) -> np.ndarray:
    """Invert the exact velocity normalization used by an evaluated model without clipping predictions."""

    from bg_pdr_fm.data.normalization_profiles import get_normalization_profile

    array = np.asarray(value, dtype=np.float32)
    if normalize_mode is None:
        return array.copy()
    profile = get_normalization_profile(profile_name)
    minimum, maximum = profile.ranges["depth_vel"]
    if normalize_mode == "-1_1":
        return ((array + 1.0) * 0.5) * (maximum - minimum) + minimum
    if normalize_mode == "01":
        return array * (maximum - minimum) + minimum
    raise ValueError(f"Unsupported velocity normalization mode {normalize_mode!r}.")


def physical_to_openfwi_normalized(value: Any) -> np.ndarray:
    """Map physical velocity to the fixed OpenFWI [-1, 1] metric coordinate without clipping."""

    array = np.asarray(value, dtype=np.float32)
    return ((array - 1500.0) / 3000.0) * 2.0 - 1.0


def index_verified_dataset_records(
    records: Sequence[Mapping[str, Any]],
    manifest_dataset_ids: Mapping[RecordKey, int],
    method_name: str,
) -> dict[RecordKey, int]:
    """Index a test dataset only when every record matches the canonical manifest ID."""

    expected_keys = set(manifest_dataset_ids)
    indices: dict[RecordKey, int] = {}
    for index, record in enumerate(records):
        try:
            key = (str(record["dataset_name"]), int(record["sample_index"]))
            dataset_id = int(record["dataset_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"{method_name} dataset record at position {index} has no usable canonical identity."
            ) from exc
        if key in indices:
            raise ValueError(f"{method_name} dataset has duplicate record identity {key}.")
        expected_dataset_id = manifest_dataset_ids.get(key)
        if expected_dataset_id is not None and dataset_id != int(expected_dataset_id):
            raise ValueError(
                f"{method_name} dataset_id mismatch for {key}: "
                f"expected {int(expected_dataset_id)}, got {dataset_id}."
            )
        indices[key] = index
    extra = set(indices).difference(expected_keys)
    missing = expected_keys.difference(indices)
    if extra or missing:
        raise ValueError(
            f"{method_name} test dataset does not exactly match the held-out manifest: "
            f"extra={len(extra)} missing={len(missing)}."
        )
    return indices


def load_method_openfwi_datasets(
    source_data: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[RecordKey, int]], dict[str, Any], dict[str, str]]:
    """Build native-evaluation datasets for the same physical held-out records.

    Table 2 checkpoints use different codec-dependent normalization profiles.
    Each model must therefore receive its native normalized input, while their
    visual outputs are later converted to a common physical velocity scale.
    """

    from bg_pdr_fm.training.benchmark_config import load_benchmark_config
    from bg_pdr_fm.training.train_bg_pdr_fm import build_dataset

    specs = tuple(source_data["specs"])
    configs = {spec.display_name: load_benchmark_config(spec.config_path) for spec in specs}
    reference_signature = _data_protocol_signature(configs["PD-BG-RFM"])
    for method_name, conf in configs.items():
        if _data_protocol_signature(conf) != reference_signature:
            raise ValueError(f"{method_name} does not share the Table 2 OpenFWI input/split protocol.")
    manifest_dataset_ids = source_data["manifest_dataset_ids"]
    datasets: dict[str, Any] = {}
    record_indices: dict[str, dict[RecordKey, int]] = {}
    profiles: dict[str, str] = {}
    for method_name, conf in configs.items():
        dataset = build_dataset(conf, "test")
        indices = index_verified_dataset_records(dataset.records, manifest_dataset_ids, method_name)
        datasets[method_name] = dataset
        record_indices[method_name] = indices
        profiles[method_name] = resolved_normalization_profile(conf)
    return datasets, record_indices, configs, profiles


def _assert_complete_load(method_name: str, checkpoint: Mapping[str, Any]) -> None:
    missing = int(checkpoint.get("missing_keys", 0))
    unexpected = int(checkpoint.get("unexpected_keys", 0))
    skipped = int(checkpoint.get("skipped_shape_keys", 0))
    if missing or unexpected or skipped:
        raise RuntimeError(
            f"{method_name} checkpoint is incomplete: missing={missing}, unexpected={unexpected}, skipped={skipped}."
        )


def load_variant_state_dict(
    model: Any,
    state_dict: Mapping[str, Any],
    *,
    prefix: str,
    method_name: str,
) -> dict[str, Any]:
    """Strictly load one model's inference module while ignoring unrelated wrapper components."""

    current = model.state_dict()
    expected = {key: value for key, value in current.items() if key.startswith(prefix)}
    supplied = {key: value for key, value in state_dict.items() if key.startswith(prefix)}
    if not expected:
        raise RuntimeError(f"{method_name} has no current model state under {prefix!r}.")
    missing = sorted(set(expected).difference(supplied))
    unexpected = sorted(set(supplied).difference(expected))
    mismatched = [
        key
        for key in sorted(set(expected).intersection(supplied))
        if not hasattr(supplied[key], "shape") or tuple(supplied[key].shape) != tuple(expected[key].shape)
    ]
    if missing:
        raise RuntimeError(f"{method_name} checkpoint is missing critical {prefix} keys: {missing[:3]}")
    if unexpected:
        raise RuntimeError(f"{method_name} checkpoint has unexpected critical {prefix} keys: {unexpected[:3]}")
    if mismatched:
        raise RuntimeError(f"{method_name} checkpoint has critical {prefix} shape mismatches: {mismatched[:3]}")
    model.load_state_dict(supplied, strict=False)
    return {
        "mode": "variant",
        "loaded_keys": len(supplied),
        "missing_keys": 0,
        "unexpected_keys": 0,
        "skipped_shape_keys": 0,
    }


def load_model_runtime(
    spec: MethodSpec,
    conf: Any,
    checkpoint: Mapping[str, Any],
    device: Any,
) -> ModelRuntime:
    """Instantiate a Table 2 model using only the checkpoint recorded by formal evaluation."""

    import torch
    from omegaconf import OmegaConf

    if spec.kind == "benchmark":
        from bg_pdr_fm.lightning import AAAI27BenchmarkLightning

        model = AAAI27BenchmarkLightning(conf)
        state = torch.load(checkpoint["path"], map_location="cpu")
        state_dict = state.get("state_dict", state) if isinstance(state, dict) else state
        if not isinstance(state_dict, Mapping):
            raise TypeError(f"{spec.display_name} checkpoint has no state dictionary: {checkpoint['path']}")
        prefix = BENCHMARK_VARIANT_PREFIXES.get(spec.display_name)
        if prefix is None:
            raise RuntimeError(f"No inference state prefix is registered for {spec.display_name}.")
        loaded = load_variant_state_dict(
            model,
            state_dict,
            prefix=prefix,
            method_name=spec.display_name,
        )
        loaded["path"] = str(checkpoint["path"])
    elif spec.kind == "pdr":
        from bg_pdr_fm.evaluation.evaluate_bg_pdr_fm import load_evaluation_checkpoints
        from bg_pdr_fm.lightning import BGPDRFMLightning

        OmegaConf.update(conf, "evaluation.checkpoints.full", str(checkpoint["path"]), merge=False)
        model = BGPDRFMLightning(conf)
        loaded_items = load_evaluation_checkpoints(model, conf)
        if len(loaded_items) != 1 or loaded_items[0].get("mode") != "full":
            raise RuntimeError(f"{spec.display_name} did not load exactly one full checkpoint: {loaded_items}")
        loaded = loaded_items[0]
    else:
        raise ValueError(f"Unsupported Table 2 runtime kind {spec.kind!r}.")
    _assert_complete_load(spec.display_name, loaded)
    model.eval().to(device)
    return ModelRuntime(spec=spec, model=model, filter=model.filter, device=device, checkpoint=dict(loaded))


def _prediction_seed_scope(device: Any, seed: int | None):
    if seed is None:
        return contextlib.nullcontext()
    import torch

    devices: list[int] = []
    if getattr(device, "type", "") == "cuda" and getattr(device, "index", None) is not None:
        devices = [int(device.index)]

    @contextlib.contextmanager
    def scope():
        with torch.random.fork_rng(devices=devices, enabled=True):
            torch.manual_seed(int(seed))
            if devices:
                torch.cuda.manual_seed(int(seed))
            yield

    return scope()


def _metric_floats(prediction: Any, target: Any, metric_filter: Any) -> dict[str, float]:
    from bg_pdr_fm.evaluation.benchmark_metrics import compute_velocity_metrics

    values = compute_velocity_metrics(prediction, target, metric_filter)
    return {metric: float(values[metric][0].detach().cpu()) for metric in METRIC_COLUMNS}


def _predict_one_runtime(
    *,
    key: RecordKey,
    method_name: str,
    dataset: Any,
    record_indices: Mapping[RecordKey, int],
    runtime: ModelRuntime,
    normalization_profile: str,
    base_seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Run one checkpoint on one native-normalized record and return physical arrays."""

    import torch
    from bg_pdr_fm.data import batch_to_device, collate_bg_samples

    if key not in record_indices:
        raise KeyError(f"Record {key} is unavailable for {method_name}.")
    batch = batch_to_device(collate_bg_samples([dataset[record_indices[key]]]), runtime.device)
    seed = record_seed(base_seed, key[0], key[1]) if method_name == "PD-BG-RFM" else None
    with torch.no_grad(), _prediction_seed_scope(runtime.device, seed):
        prediction = runtime.model.predict_batch(batch)
    expected_shape = (1, 1, 70, 70)
    if tuple(prediction.velocity_hat.shape) != expected_shape or tuple(batch.depth_vel.shape) != expected_shape:
        raise ValueError(
            f"{method_name} produced {tuple(prediction.velocity_hat.shape)} for {key}; expected {expected_shape}."
        )
    prediction_native = prediction.velocity_hat.detach().float().cpu().numpy()
    target_native = batch.depth_vel.detach().float().cpu().numpy()
    if not np.isfinite(prediction_native).all() or not np.isfinite(target_native).all():
        raise ValueError(f"{method_name} produced non-finite values for {key}.")
    normalize_mode = getattr(dataset, "use_normalize", "-1_1")
    return (
        velocity_to_physical(target_native, normalization_profile, normalize_mode),
        velocity_to_physical(prediction_native, normalization_profile, normalize_mode),
        _metric_floats(prediction.velocity_hat, batch.depth_vel, runtime.filter),
    )


def _finalize_record_artifact(
    *,
    key: RecordKey,
    target_physical: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    native_metrics: Mapping[str, Mapping[str, float]],
    normalization_profiles: Mapping[str, str],
    base_seed: int,
) -> dict[str, Any]:
    """Compute common-scale metrics once every Table 2 prediction is available."""

    import torch
    from bg_pdr_fm.evaluation.benchmark_metrics import compute_velocity_metrics
    from bg_pdr_fm.models.filters import LowHighPassFilter

    expected_methods = set(normalization_profiles)
    if set(predictions) != expected_methods or set(native_metrics) != expected_methods:
        raise ValueError(f"Incomplete Table 2 predictions for {key}: got {sorted(predictions)}.")
    common_filter = LowHighPassFilter(kernel_size=5)
    target_common = torch.from_numpy(physical_to_openfwi_normalized(target_physical))
    common_metrics: dict[str, dict[str, float]] = {}
    for method_name, prediction_physical in predictions.items():
        prediction_common = torch.from_numpy(physical_to_openfwi_normalized(prediction_physical))
        values = compute_velocity_metrics(prediction_common, target_common, common_filter)
        common_metrics[method_name] = {metric: float(values[metric][0]) for metric in METRIC_COLUMNS}
    return {
        "target": target_physical,
        "predictions": dict(predictions),
        "metrics": common_metrics,
        "native_metrics": {name: dict(values) for name, values in native_metrics.items()},
        "normalization_profiles": dict(normalization_profiles),
        "pd_visual_seed": record_seed(base_seed, key[0], key[1]),
        "shape": list(target_physical.shape),
    }


def reconstruct_record(
    *,
    key: RecordKey,
    datasets: Mapping[str, Any],
    record_indices: Mapping[str, Mapping[RecordKey, int]],
    runtimes: Mapping[str, ModelRuntime],
    normalization_profiles: Mapping[str, str],
    base_seed: int,
) -> dict[str, Any]:
    """Re-infer all Table 2 models for one audited held-out record."""

    predictions: dict[str, np.ndarray] = {}
    native_metrics: dict[str, dict[str, float]] = {}
    target_physical: np.ndarray | None = None
    for method_name, runtime in runtimes.items():
        target_for_method, prediction_physical, metrics = _predict_one_runtime(
            key=key,
            method_name=method_name,
            dataset=datasets[method_name],
            record_indices=record_indices[method_name],
            runtime=runtime,
            normalization_profile=normalization_profiles[method_name],
            base_seed=base_seed,
        )
        if target_physical is None:
            target_physical = target_for_method
        elif not np.allclose(target_physical, target_for_method, rtol=1e-6, atol=1e-3):
            raise ValueError(f"Physical target mismatch while reconstructing {key} for {method_name}.")
        predictions[method_name] = prediction_physical
        native_metrics[method_name] = metrics
    assert target_physical is not None
    return _finalize_record_artifact(
        key=key,
        target_physical=target_physical,
        predictions=predictions,
        native_metrics=native_metrics,
        normalization_profiles=normalization_profiles,
        base_seed=base_seed,
    )


def reconstruct_records_sequentially(
    *,
    keys: Sequence[RecordKey],
    source_data: Mapping[str, Any],
    datasets: Mapping[str, Any],
    record_indices: Mapping[str, Mapping[RecordKey, int]],
    configs: Mapping[str, Any],
    normalization_profiles: Mapping[str, str],
    device: Any,
    base_seed: int,
) -> tuple[dict[RecordKey, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Load one full wrapper at a time to keep all six Table 2 checkpoints GPU-safe."""

    import torch

    requested = sorted(set(keys))
    partial: dict[RecordKey, dict[str, Any]] = {
        key: {"target": None, "predictions": {}, "native_metrics": {}}
        for key in requested
    }
    loaded_checkpoints: dict[str, dict[str, Any]] = {}
    for spec in source_data["specs"]:
        runtime = load_model_runtime(
            spec,
            configs[spec.display_name],
            source_data["checkpoints"][spec.display_name],
            device,
        )
        loaded_checkpoints[spec.display_name] = dict(runtime.checkpoint)
        try:
            for key in requested:
                target, prediction, native = _predict_one_runtime(
                    key=key,
                    method_name=spec.display_name,
                    dataset=datasets[spec.display_name],
                    record_indices=record_indices[spec.display_name],
                    runtime=runtime,
                    normalization_profile=normalization_profiles[spec.display_name],
                    base_seed=base_seed,
                )
                existing_target = partial[key]["target"]
                if existing_target is None:
                    partial[key]["target"] = target
                elif not np.allclose(existing_target, target, rtol=1e-6, atol=1e-3):
                    raise ValueError(f"Physical target mismatch for {key} in {spec.display_name}.")
                partial[key]["predictions"][spec.display_name] = prediction
                partial[key]["native_metrics"][spec.display_name] = native
        finally:
            del runtime
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    artifacts: dict[RecordKey, dict[str, Any]] = {}
    for key, values in partial.items():
        artifacts[key] = _finalize_record_artifact(
            key=key,
            target_physical=values["target"],
            predictions=values["predictions"],
            native_metrics=values["native_metrics"],
            normalization_profiles=normalization_profiles,
            base_seed=base_seed,
        )
    return artifacts, loaded_checkpoints


def merge_isolated_method_outputs(
    *,
    keys: Sequence[RecordKey],
    worker_outputs: Mapping[str, Mapping[str, Any]],
    normalization_profiles: Mapping[str, str],
    base_seed: int,
    manifest_dataset_ids: Mapping[RecordKey, int] | None = None,
) -> tuple[dict[RecordKey, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Merge one-process-per-method reconstructions with exact identity checks."""

    expected_methods = tuple(normalization_profiles)
    if set(worker_outputs) != set(expected_methods):
        raise ValueError(
            "Isolated reconstruction methods do not match the expected set: "
            f"expected={sorted(expected_methods)} got={sorted(worker_outputs)}."
        )
    requested = sorted(set(keys))
    if manifest_dataset_ids is not None:
        missing_metadata = set(requested).difference(manifest_dataset_ids)
        if missing_metadata:
            raise ValueError(
                "Isolated reconstruction contains records absent from canonical held-out manifest metadata: "
                f"missing={len(missing_metadata)}."
            )
    artifacts: dict[RecordKey, dict[str, Any]] = {}
    checkpoints: dict[str, dict[str, Any]] = {}
    for method_name in expected_methods:
        checkpoint = dict(worker_outputs[method_name].get("checkpoint") or {})
        _assert_complete_load(method_name, checkpoint)
        checkpoints[method_name] = checkpoint
        records = worker_outputs[method_name].get("records")
        if not isinstance(records, Mapping) or set(records) != set(requested):
            actual = set(records) if isinstance(records, Mapping) else set()
            raise ValueError(
                f"{method_name} isolated reconstruction records do not match the request: "
                f"extra={len(actual - set(requested))} missing={len(set(requested) - actual)}."
            )
        if manifest_dataset_ids is not None:
            for key in requested:
                dataset_id = records[key].get("dataset_id")
                if dataset_id in {None, ""}:
                    raise ValueError(f"{method_name} isolated reconstruction has no dataset_id for {key}.")
                if int(dataset_id) != int(manifest_dataset_ids[key]):
                    raise ValueError(
                        f"{method_name} isolated reconstruction dataset_id mismatch for {key}: "
                        f"expected {int(manifest_dataset_ids[key])}, got {int(dataset_id)}."
                    )

    for key in requested:
        target_physical: np.ndarray | None = None
        predictions: dict[str, np.ndarray] = {}
        native_metrics: dict[str, dict[str, float]] = {}
        for method_name in expected_methods:
            record = worker_outputs[method_name]["records"][key]
            target = np.asarray(record["target"], dtype=np.float32)
            prediction = np.asarray(record["prediction"], dtype=np.float32)
            if tuple(target.shape) != (1, 1, 70, 70) or tuple(prediction.shape) != (1, 1, 70, 70):
                raise ValueError(
                    f"{method_name} isolated reconstruction shape for {key} is "
                    f"target={tuple(target.shape)} prediction={tuple(prediction.shape)}; expected [1, 1, 70, 70]."
                )
            if not np.isfinite(target).all() or not np.isfinite(prediction).all():
                raise ValueError(f"{method_name} isolated reconstruction contains non-finite values for {key}.")
            if target_physical is None:
                target_physical = target
            elif not np.allclose(target_physical, target, rtol=1e-6, atol=1e-3):
                raise ValueError(f"Physical target mismatch for {key} in {method_name}.")
            predictions[method_name] = prediction
            native_metrics[method_name] = {
                metric: float(record["native_metrics"][metric]) for metric in METRIC_COLUMNS
            }
        assert target_physical is not None
        artifacts[key] = _finalize_record_artifact(
            key=key,
            target_physical=target_physical,
            predictions=predictions,
            native_metrics=native_metrics,
            normalization_profiles=normalization_profiles,
            base_seed=base_seed,
        )
    return artifacts, checkpoints


def write_isolated_worker_output(
    path: str | Path,
    *,
    method_name: str,
    checkpoint: Mapping[str, Any],
    records: Mapping[RecordKey, Mapping[str, Any]],
) -> None:
    """Persist one method's selected-record reconstruction without pickle payloads."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_records: list[dict[str, Any]] = []
    arrays: dict[str, np.ndarray] = {}
    for slot, key in enumerate(sorted(records)):
        record = records[key]
        target = np.asarray(record["target"], dtype=np.float32)
        prediction = np.asarray(record["prediction"], dtype=np.float32)
        if tuple(target.shape) != (1, 1, 70, 70) or tuple(prediction.shape) != (1, 1, 70, 70):
            raise ValueError(
                f"{method_name} worker output has invalid shape for {key}: "
                f"target={tuple(target.shape)} prediction={tuple(prediction.shape)}."
            )
        target_name = f"target_{slot}"
        prediction_name = f"prediction_{slot}"
        arrays[target_name] = target
        arrays[prediction_name] = prediction
        metadata_record = {
            "dataset_name": key[0],
            "source_sample_index": key[1],
            "target_array": target_name,
            "prediction_array": prediction_name,
            "native_metrics": {
                metric: float(record["native_metrics"][metric]) for metric in METRIC_COLUMNS
            },
        }
        if record.get("dataset_id") not in {None, ""}:
            metadata_record["dataset_id"] = int(record["dataset_id"])
        metadata_records.append(metadata_record)
    metadata = {
        "method_name": str(method_name),
        "checkpoint": dict(checkpoint),
        "records": metadata_records,
    }
    np.savez_compressed(
        output_path,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True, default=str)),
        **arrays,
    )


def read_isolated_worker_output(path: str | Path) -> dict[str, Any]:
    """Load an isolated worker result and reconstruct its tuple-keyed record map."""

    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Isolated reconstruction worker did not write {input_path}.")
    with np.load(input_path, allow_pickle=False) as archive:
        if "metadata" not in archive:
            raise ValueError(f"Isolated reconstruction output lacks metadata: {input_path}.")
        metadata = json.loads(str(archive["metadata"].item()))
        method_name = str(metadata.get("method_name", ""))
        if not method_name:
            raise ValueError(f"Isolated reconstruction output has no method name: {input_path}.")
        records: dict[RecordKey, dict[str, Any]] = {}
        for row in metadata.get("records", []):
            key = (str(row["dataset_name"]), int(row["source_sample_index"]))
            if key in records:
                raise ValueError(f"{method_name} worker output duplicates record identity {key}.")
            target_name = str(row["target_array"])
            prediction_name = str(row["prediction_array"])
            if target_name not in archive or prediction_name not in archive:
                raise ValueError(f"{method_name} worker output is missing arrays for {key}.")
            records[key] = {
                "target": np.asarray(archive[target_name], dtype=np.float32).copy(),
                "prediction": np.asarray(archive[prediction_name], dtype=np.float32).copy(),
                "native_metrics": {
                    metric: float(row["native_metrics"][metric]) for metric in METRIC_COLUMNS
                },
            }
            if row.get("dataset_id") not in {None, ""}:
                records[key]["dataset_id"] = int(row["dataset_id"])
    return {
        "method_name": method_name,
        "checkpoint": dict(metadata.get("checkpoint") or {}),
        "records": records,
    }


def build_isolated_worker_command(
    *,
    method_name: str,
    records_path: str | Path,
    output_path: str | Path,
    device: str,
    base_seed: int,
) -> list[str]:
    """Build the command for one short-lived, single-method reconstruction worker."""

    return [
        sys.executable,
        "-m",
        "bg_pdr_fm.evaluation.visualize_table2_qualitative",
        "--isolated-worker",
        "--worker-method",
        str(method_name),
        "--records-json",
        str(records_path),
        "--worker-output",
        str(output_path),
        "--device",
        str(device),
        "--base-seed",
        str(int(base_seed)),
    ]


def _write_record_keys(path: str | Path, keys: Sequence[RecordKey]) -> None:
    payload = [
        {"dataset_name": dataset_name, "source_sample_index": source_sample_index}
        for dataset_name, source_sample_index in sorted(set(keys))
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_record_keys(path: str | Path) -> list[RecordKey]:
    input_path = Path(path)
    rows = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Isolated worker record list must be a non-empty JSON list: {input_path}.")
    keys = [(str(row["dataset_name"]), int(row["source_sample_index"])) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Isolated worker record list has duplicate identities: {input_path}.")
    return keys


def load_method_configs_and_profiles(source_data: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, str]]:
    """Load immutable formal configs without keeping six test datasets resident."""

    from bg_pdr_fm.training.benchmark_config import load_benchmark_config

    specs = tuple(source_data["specs"])
    configs = {spec.display_name: load_benchmark_config(spec.config_path) for spec in specs}
    reference_signature = _data_protocol_signature(configs["PD-BG-RFM"])
    for method_name, conf in configs.items():
        if _data_protocol_signature(conf) != reference_signature:
            raise ValueError(f"{method_name} does not share the Table 2 OpenFWI input/split protocol.")
    return configs, {method_name: resolved_normalization_profile(conf) for method_name, conf in configs.items()}


def reconstruct_isolated_worker(
    *,
    method_name: str,
    keys: Sequence[RecordKey],
    device: str,
    base_seed: int,
) -> dict[str, Any]:
    """Run one formal checkpoint in a short-lived process and return physical arrays."""

    import torch
    from bg_pdr_fm.training.benchmark_config import load_benchmark_config
    from bg_pdr_fm.training.train_bg_pdr_fm import build_dataset

    specs = default_method_specs()
    spec = next((item for item in specs if item.display_name == method_name), None)
    if spec is None:
        raise ValueError(f"Unknown Table 2 method requested by isolated worker: {method_name!r}.")
    pd_spec = next(item for item in specs if item.kind == "pdr")
    manifest_dataset_ids = load_manifest_metadata(
        pd_spec.summary_path.parent / "held_out_manifest.csv",
        expected_count=EXPECTED_GLOBAL_HELD_OUT_RECORDS,
    )
    manifest_keys = set(manifest_dataset_ids)
    requested = sorted(set(keys))
    if not set(requested).issubset(manifest_keys):
        raise ValueError(f"{method_name} worker received records outside the formal held-out manifest.")
    checkpoint = checkpoint_provenance(spec)
    conf = load_benchmark_config(spec.config_path)
    dataset = build_dataset(conf, "test")
    runtime: ModelRuntime | None = None
    try:
        record_indices = index_verified_dataset_records(dataset.records, manifest_dataset_ids, method_name)
        profile = resolved_normalization_profile(conf)
        runtime = load_model_runtime(spec, conf, checkpoint, torch.device(device))
        records: dict[RecordKey, dict[str, Any]] = {}
        for key in requested:
            target, prediction, native_metrics = _predict_one_runtime(
                key=key,
                method_name=method_name,
                dataset=dataset,
                record_indices=record_indices,
                runtime=runtime,
                normalization_profile=profile,
                base_seed=base_seed,
            )
            records[key] = {
                "dataset_id": int(manifest_dataset_ids[key]),
                "target": target,
                "prediction": prediction,
                "native_metrics": native_metrics,
            }
        return {"method_name": method_name, "checkpoint": dict(runtime.checkpoint), "records": records}
    finally:
        if runtime is not None:
            del runtime
        close = getattr(dataset, "close", None)
        if callable(close):
            close()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def reconstruct_records_in_isolated_processes(
    *,
    keys: Sequence[RecordKey],
    source_data: Mapping[str, Any],
    normalization_profiles: Mapping[str, str],
    device: str,
    base_seed: int,
    worker_dir: str | Path,
) -> tuple[dict[RecordKey, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Reconstruct selected records with one operating-system process per method."""

    requested = sorted(set(keys))
    if not requested:
        return {}, {}
    directory = Path(worker_dir)
    directory.mkdir(parents=True, exist_ok=True)
    tag_payload = json.dumps(requested, separators=(",", ":"), sort_keys=True).encode("utf-8")
    tag = hashlib.sha256(tag_payload).hexdigest()[:12]
    records_path = directory / f"records_{tag}.json"
    _write_record_keys(records_path, requested)
    worker_outputs: dict[str, dict[str, Any]] = {}
    for spec in source_data["specs"]:
        slug = _slug(spec.display_name)
        output_path = directory / f"{tag}_{slug}.npz"
        log_path = directory / f"{tag}_{slug}.log"
        command = build_isolated_worker_command(
            method_name=spec.display_name,
            records_path=records_path,
            output_path=output_path,
            device=device,
            base_seed=base_seed,
        )
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(
                command,
                cwd=REPO_ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        if result.returncode != 0:
            raise RuntimeError(
                f"Isolated reconstruction worker failed for {spec.display_name} with return code "
                f"{result.returncode}; see {log_path}."
            )
        worker_output = read_isolated_worker_output(output_path)
        if worker_output["method_name"] != spec.display_name:
            raise ValueError(
                f"Worker output method mismatch: expected {spec.display_name}, got {worker_output['method_name']}."
            )
        worker_outputs[spec.display_name] = worker_output
    return merge_isolated_method_outputs(
        keys=requested,
        worker_outputs=worker_outputs,
        normalization_profiles=normalization_profiles,
        base_seed=base_seed,
        manifest_dataset_ids=source_data["manifest_dataset_ids"],
    )


def _ordered_advantage_candidates(
    rows: Sequence[Mapping[str, Any]],
    criterion: str,
    used_keys: set[RecordKey],
    used_datasets: set[str],
) -> list[dict[str, Any]]:
    candidates = [
        dict(row)
        for row in rows
        if float(row[criterion]) > 0 and _record_order(row) not in used_keys
    ]
    candidates.sort(key=lambda row: (-float(row[criterion]), *_record_order(row)))
    distinct = [row for row in candidates if str(row["dataset_name"]) not in used_datasets]
    return [*distinct, *(row for row in candidates if row not in distinct)]


def formal_candidate_keys(
    formal_rows: Sequence[Mapping[str, Any]],
    formal_selection: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    rank_offset: int,
    per_criterion: int,
) -> set[RecordKey]:
    """Return a fixed-rank candidate tranche plus per-subset and failure controls."""

    keys: set[RecordKey] = set()
    for criterion in ADVANTAGE_CRITERIA:
        ranked = [row for row in formal_rows if float(row[criterion]) > 0]
        ranked.sort(key=lambda row: (-float(row[criterion]), *_record_order(row)))
        keys.update(_record_order(row) for row in ranked[rank_offset:rank_offset + per_criterion])
    if rank_offset == 0:
        for group in ("representative", "failure"):
            for case in formal_selection[group]:
                if case.get("status") == "selected":
                    keys.add(_record_order(case))
    return keys


def select_rendered_advantage_cases(
    *,
    formal_rows: Sequence[Mapping[str, Any]],
    reconstruct: Any,
) -> tuple[list[dict[str, Any]], dict[RecordKey, dict[str, Any]]]:
    """Select formal-ranked advantage records only after fixed-seed re-inference confirms each margin."""

    selected: list[dict[str, Any]] = []
    artifacts: dict[RecordKey, dict[str, Any]] = {}
    used_keys: set[RecordKey] = set()
    used_datasets: set[str] = set()
    for criterion in ADVANTAGE_CRITERIA:
        chosen: dict[str, Any] | None = None
        for candidate in _ordered_advantage_candidates(formal_rows, criterion, used_keys, used_datasets):
            key = _record_order(candidate)
            artifact = artifacts.get(key)
            if artifact is None:
                try:
                    artifact = reconstruct(key)
                except KeyError:
                    continue
                artifacts[key] = artifact
            actual = rendered_margin_values(artifact["metrics"])
            if float(actual[criterion]) <= 0:
                continue
            chosen = _selected_case(candidate, criterion, "advantage")
            chosen.update({f"rendered_{name}": value for name, value in actual.items()})
            chosen["formal_rank_margin"] = float(candidate[criterion])
            used_keys.add(key)
            used_datasets.add(key[0])
            break
        if chosen is None:
            chosen = _missing_advantage_case(criterion)
        selected.append(chosen)
    return selected, artifacts


def add_rendered_margins(case: Mapping[str, Any], artifact: Mapping[str, Any]) -> dict[str, Any]:
    """Attach fixed-seed rendering metrics to a non-advantage selected case."""

    output = dict(case)
    output.update({f"rendered_{name}": value for name, value in rendered_margin_values(artifact["metrics"]).items()})
    return output


def save_reconstructed_arrays(output_dir: str | Path, artifacts: Mapping[RecordKey, Mapping[str, Any]]) -> list[str]:
    arrays_dir = Path(output_dir) / "arrays"
    arrays_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for key, artifact in sorted(artifacts.items()):
        payload = {"target": artifact["target"]}
        payload.update({_slug(name): array for name, array in artifact["predictions"].items()})
        path = arrays_dir / f"{key[0]}_{key[1]:06d}.npz"
        np.savez_compressed(path, **payload)
        paths.append(str(path))
    return paths


def assert_exact_held_out_alignment(
    manifest_keys: set[RecordKey],
    baselines: Mapping[str, Mapping[RecordKey, Mapping[str, Any]]],
    pd_metrics: Mapping[RecordKey, Mapping[str, Any]],
    *,
    manifest_dataset_ids: Mapping[RecordKey, int] | None = None,
) -> None:
    """Reject metric sources that do not exactly cover the held-out manifest."""

    sources: dict[str, Mapping[RecordKey, Mapping[str, Any]]] = {
        **dict(baselines),
        "PD-BG-RFM": pd_metrics,
    }
    if manifest_dataset_ids is not None and set(manifest_dataset_ids) != manifest_keys:
        raise ValueError("Held-out manifest key set does not match its dataset-ID metadata.")
    for name, rows in sources.items():
        keys = set(rows)
        extra = keys.difference(manifest_keys)
        missing = manifest_keys.difference(keys)
        if extra or missing:
            raise ValueError(f"{name}: extra={len(extra)} missing={len(missing)}")
        if manifest_dataset_ids is None:
            continue
        for key in sorted(keys):
            dataset_id = rows[key].get("dataset_id")
            if dataset_id in {None, ""}:
                raise ValueError(f"{name} has no dataset_id for held-out record {key}.")
            try:
                actual_dataset_id = int(dataset_id)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} has invalid dataset_id {dataset_id!r} for held-out record {key}.") from exc
            expected_dataset_id = int(manifest_dataset_ids[key])
            if actual_dataset_id != expected_dataset_id:
                raise ValueError(
                    f"{name} dataset_id mismatch for {key}: "
                    f"expected {expected_dataset_id}, got {actual_dataset_id}."
                )


def build_margin_rows(
    *,
    baselines: Mapping[str, Mapping[RecordKey, Mapping[str, Any]]],
    pd_metrics: Mapping[RecordKey, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Compute Table 2 advantage margins per exact record identity."""

    if not baselines:
        raise ValueError("At least one baseline is required to compute qualitative margins.")
    expected = set(pd_metrics)
    for name, source in baselines.items():
        if set(source) != expected:
            raise ValueError(f"{name} does not match the PD-BG-RFM metric identity set.")

    rows: list[dict[str, Any]] = []
    for dataset_name, source_sample_index in sorted(expected):
        key = (dataset_name, source_sample_index)
        pd = pd_metrics[key]
        baseline_values = {name: source[key] for name, source in baselines.items()}
        row = {
            "dataset_name": dataset_name,
            "source_sample_index": source_sample_index,
            "pd_mae": float(pd["mae"]),
            "pd_mae_l": float(pd["mae_l"]),
            "pd_mae_h": float(pd["mae_h"]),
            "pd_ssim": float(pd["ssim"]),
            "margin_l": min(float(metrics["mae_l"]) for metrics in baseline_values.values())
            - float(pd["mae_l"]),
            "margin_h": min(float(metrics["mae_h"]) for metrics in baseline_values.values())
            - float(pd["mae_h"]),
            "margin_s": float(pd["ssim"])
            - max(float(metrics["ssim"]) for metrics in baseline_values.values()),
        }
        rows.append(row)
    return rows


def rendered_margin_values(
    metrics_by_method: Mapping[str, Mapping[str, float]],
    pd_method_name: str = "PD-BG-RFM",
) -> dict[str, float]:
    """Compute the three disclosed margins from re-inferred per-record metrics."""

    if pd_method_name not in metrics_by_method:
        raise KeyError(f"Rendered metrics do not include {pd_method_name}.")
    baselines = {name: metrics for name, metrics in metrics_by_method.items() if name != pd_method_name}
    if not baselines:
        raise ValueError("Rendered margin computation requires at least one baseline.")
    pd = metrics_by_method[pd_method_name]
    return {
        "margin_l": min(float(metrics["mae_l"]) for metrics in baselines.values()) - float(pd["mae_l"]),
        "margin_h": min(float(metrics["mae_h"]) for metrics in baselines.values()) - float(pd["mae_h"]),
        "margin_s": float(pd["ssim"]) - max(float(metrics["ssim"]) for metrics in baselines.values()),
    }


def _record_order(row: Mapping[str, Any]) -> tuple[str, int]:
    return str(row["dataset_name"]), int(row["source_sample_index"])


def _selected_case(row: Mapping[str, Any], criterion: str, selection_group: str) -> dict[str, Any]:
    case = dict(row)
    case.update(
        {
            "criterion": criterion,
            "selection_group": selection_group,
            "status": "selected",
        }
    )
    return case


def _missing_advantage_case(criterion: str) -> dict[str, Any]:
    return {
        "dataset_name": "",
        "source_sample_index": None,
        "criterion": criterion,
        "selection_group": "advantage",
        "status": "no_positive_candidate",
    }


def _missing_failure_case(criterion: str) -> dict[str, Any]:
    return {
        "dataset_name": "",
        "source_sample_index": None,
        "criterion": criterion,
        "selection_group": "failure",
        "status": "no_negative_candidate",
    }


def select_case_records(rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Select criterion, per-subset maximum-SSIM-margin, and failure records automatically."""

    if not rows:
        raise ValueError("Cannot select qualitative records from an empty metric table.")
    normalized = [dict(row) for row in rows]
    keys = [_record_order(row) for row in normalized]
    if len(keys) != len(set(keys)):
        raise ValueError("Qualitative metric rows contain duplicate record identities.")

    advantage: list[dict[str, Any]] = []
    used_keys: set[RecordKey] = set()
    used_datasets: set[str] = set()
    for criterion in ADVANTAGE_CRITERIA:
        candidates = [
            row
            for row in normalized
            if float(row[criterion]) > 0 and _record_order(row) not in used_keys
        ]
        candidates.sort(key=lambda row: (-float(row[criterion]), *_record_order(row)))
        distinct_dataset = [row for row in candidates if str(row["dataset_name"]) not in used_datasets]
        if not candidates:
            advantage.append(_missing_advantage_case(criterion))
            continue
        chosen = distinct_dataset[0] if distinct_dataset else candidates[0]
        advantage.append(_selected_case(chosen, criterion, "advantage"))
        used_keys.add(_record_order(chosen))
        used_datasets.add(str(chosen["dataset_name"]))

    representative: list[dict[str, Any]] = []
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in normalized:
        by_dataset[str(row["dataset_name"])].append(row)
    for dataset_name in sorted(by_dataset):
        candidates = by_dataset[dataset_name]
        chosen = max(
            candidates,
            key=lambda row: (float(row["margin_s"]), -int(row["source_sample_index"])),
        )
        representative.append(_selected_case(chosen, "margin_s_max", "representative"))

    failure: list[dict[str, Any]] = []
    used_failure_keys: set[RecordKey] = set()
    for criterion in ADVANTAGE_CRITERIA:
        candidates = [
            row
            for row in normalized
            if float(row[criterion]) < 0 and _record_order(row) not in used_failure_keys
        ]
        candidates.sort(key=lambda row: (float(row[criterion]), *_record_order(row)))
        if not candidates:
            failure.append(_missing_failure_case(criterion))
            continue
        chosen = candidates[0]
        failure.append(_selected_case(chosen, criterion, "failure"))
        used_failure_keys.add(_record_order(chosen))

    return {
        "advantage": advantage,
        "representative": representative,
        "failure": failure,
    }


def record_seed(base_seed: int, dataset_name: str, source_sample_index: int) -> int:
    """Derive a stable 31-bit PD-BG-RFM sampling seed from a record identity."""

    payload = f"{int(base_seed)}:{dataset_name}:{int(source_sample_index)}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31)


def _image_2d(value: Any, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32).squeeze()
    if array.ndim != 2:
        raise ValueError(f"{label} must resolve to a 2D image, got shape {tuple(array.shape)}.")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} contains non-finite values.")
    return array


def format_ssim_annotation(value: float) -> str:
    """Format one finite per-image SSIM value for qualitative panels."""

    score = float(value)
    if not np.isfinite(score):
        raise ValueError("SSIM annotation requires a finite value.")
    return f"SSIM={score:.4f}"


def render_case_figure(
    *,
    cases: Sequence[Mapping[str, Any]],
    artifacts: Mapping[RecordKey, Mapping[str, Any]],
    method_names: Sequence[str],
    output_base: str | Path,
    title: str,
    velocity_limits: tuple[float, float] = (-1.0, 1.0),
    velocity_label: str = "Normalized velocity",
    error_label: str = "Absolute error",
    error_limits: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Render paired prediction/error rows with one scale shared by every panel."""

    visible_cases = [case for case in cases if case.get("status") == "selected"]
    if not visible_cases:
        raise ValueError("No selected qualitative cases are available for rendering.")
    if not method_names:
        raise ValueError("At least one method prediction is required for rendering.")

    images: list[np.ndarray] = []
    errors: list[np.ndarray] = []
    normalized_artifacts: dict[RecordKey, tuple[np.ndarray, dict[str, np.ndarray], dict[str, float]]] = {}
    for case in visible_cases:
        key = _record_order(case)
        artifact = artifacts.get(key)
        if artifact is None:
            raise KeyError(f"No reconstructed arrays found for selected record {key}.")
        target = _image_2d(artifact["target"], f"target for {key}")
        predictions = {
            method: _image_2d(artifact["predictions"][method], f"{method} prediction for {key}")
            for method in method_names
        }
        if any(prediction.shape != target.shape for prediction in predictions.values()):
            raise ValueError(f"Prediction and target shapes differ for {key}.")
        metric_values = artifact.get("metrics")
        if not isinstance(metric_values, Mapping):
            raise ValueError(f"No rendered per-method metrics are available for {key}.")
        ssim_values: dict[str, float] = {}
        for method in method_names:
            try:
                ssim_values[method] = float(metric_values[method]["ssim"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"No usable rendered SSIM is available for {method} on {key}.") from exc
            format_ssim_annotation(ssim_values[method])
        normalized_artifacts[key] = (target, predictions, ssim_values)
        images.extend([target, *predictions.values()])
        errors.extend([np.zeros_like(target), *(np.abs(prediction - target) for prediction in predictions.values())])

    velocity_limits = (float(velocity_limits[0]), float(velocity_limits[1]))
    if velocity_limits[0] >= velocity_limits[1]:
        raise ValueError(f"Velocity display limits must be increasing, got {velocity_limits}.")
    if error_limits is None:
        error_limits = (0.0, round(float(max(np.max(error) for error in errors)), 6))
    error_limits = (float(error_limits[0]), float(error_limits[1]))
    if error_limits[0] < 0 or error_limits[0] >= error_limits[1]:
        raise ValueError(f"Error display limits must satisfy 0 <= min < max, got {error_limits}.")

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    column_names = ("Ground Truth", *method_names)
    rows = len(visible_cases) * 2
    figure, axes = plt.subplots(rows, len(column_names), figsize=(2.0 * len(column_names), 2.2 * rows), squeeze=False)
    for column, name in enumerate(column_names):
        axes[0, column].set_title(name, fontsize=8, pad=5)

    for case_index, case in enumerate(visible_cases):
        row_prediction = case_index * 2
        row_error = row_prediction + 1
        key = _record_order(case)
        target, predictions, ssim_values = normalized_artifacts[key]
        criterion = str(case.get("criterion", ""))
        margin = case.get(f"rendered_{criterion}", case.get(criterion))
        label = f"{key[0]}\nindex {key[1]}\n{criterion}={float(margin):+.4f}" if margin is not None else f"{key[0]}\nindex {key[1]}"
        axes[row_prediction, 0].set_ylabel(label, fontsize=7)
        axes[row_error, 0].set_ylabel("absolute error", fontsize=7)
        plot_values = (target, *(predictions[name] for name in method_names))
        error_values = (np.zeros_like(target), *(np.abs(predictions[name] - target) for name in method_names))
        height, width = target.shape
        x_ticks = (0, width // 2, width - 1)
        y_ticks = (0, height // 2, height - 1)
        extent = (0, width - 1, height - 1, 0)
        for column, (velocity, error) in enumerate(zip(plot_values, error_values, strict=True)):
            prediction_axis = axes[row_prediction, column]
            error_axis = axes[row_error, column]
            prediction_axis.imshow(
                velocity,
                cmap=VELOCITY_COLORMAP,
                vmin=velocity_limits[0],
                vmax=velocity_limits[1],
                aspect="auto",
                extent=extent,
                origin="upper",
            )
            error_axis.imshow(
                error,
                cmap=ERROR_COLORMAP,
                vmin=error_limits[0],
                vmax=error_limits[1],
                aspect="auto",
                extent=extent,
                origin="upper",
            )
            ssim_label = format_ssim_annotation(1.0 if column == 0 else ssim_values[method_names[column - 1]])
            prediction_axis.text(
                0.02,
                0.97,
                ssim_label,
                transform=prediction_axis.transAxes,
                ha="left",
                va="top",
                fontsize=5.2,
                color="black",
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 0.15},
            )
            for axis in (prediction_axis, error_axis):
                axis.set_xticks(x_ticks)
                axis.set_yticks(y_ticks)
                axis.tick_params(axis="both", labelsize=5, length=2, pad=1)
            if column != 0:
                prediction_axis.tick_params(labelleft=False)
                error_axis.tick_params(labelleft=False)
            if row_error != rows - 1:
                prediction_axis.tick_params(labelbottom=False)
                error_axis.tick_params(labelbottom=False)
            if column == len(column_names) - 1:
                for axis in (prediction_axis, error_axis):
                    for spine in axis.spines.values():
                        spine.set_linewidth(2.0)
                        spine.set_edgecolor("#0f766e")

    velocity_map = ScalarMappable(norm=Normalize(*velocity_limits), cmap=VELOCITY_COLORMAP)
    error_map = ScalarMappable(norm=Normalize(*error_limits), cmap=ERROR_COLORMAP)
    velocity_map.set_array([])
    error_map.set_array([])
    figure.subplots_adjust(left=0.08, right=0.86, bottom=0.07, top=0.93, wspace=0.03, hspace=0.18)
    velocity_axis = figure.add_axes((0.89, 0.56, 0.012, 0.29))
    error_axis = figure.add_axes((0.89, 0.16, 0.012, 0.29))
    figure.colorbar(velocity_map, cax=velocity_axis, label=velocity_label)
    figure.colorbar(error_map, cax=error_axis, label=error_label)
    figure.text(0.47, 0.012, "x grid index", ha="center", va="bottom", fontsize=7)
    figure.text(0.006, 0.50, "z grid index", ha="left", va="center", rotation=90, fontsize=7)
    figure.suptitle(title, fontsize=10, y=1.01)

    base = Path(output_base)
    base.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = base.with_suffix(".pdf")
    png_path = base.with_suffix(".png")
    figure.savefig(pdf_path, bbox_inches="tight")
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return {
        "pdf": str(pdf_path),
        "png": str(png_path),
        "velocity_limits": velocity_limits,
        "error_limits": error_limits,
        "velocity_cmap": VELOCITY_COLORMAP,
        "error_cmap": ERROR_COLORMAP,
        "coordinate_system": GRID_COORDINATE_SYSTEM,
        "coordinate_ticks": (0, 35, 69),
    }


def _filename_slug(value: str) -> str:
    """Return a stable ASCII filename component without display-only punctuation."""

    slug = "".join(character.lower() if character.isalnum() else "_" for character in str(value))
    return slug.strip("_") or "unnamed"


def render_standalone_velocity_images(
    *,
    cases: Sequence[Mapping[str, Any]],
    artifacts: Mapping[RecordKey, Mapping[str, Any]],
    method_names: Sequence[str],
    output_dir: str | Path,
    velocity_limits: tuple[float, float],
) -> list[dict[str, Any]]:
    """Export one pixel-only velocity PNG for every selected case and method."""

    visible_cases = [case for case in cases if case.get("status") == "selected"]
    if not visible_cases:
        raise ValueError("No selected cases are available for standalone velocity export.")
    if not method_names:
        raise ValueError("At least one method prediction is required for standalone velocity export.")

    velocity_limits = (float(velocity_limits[0]), float(velocity_limits[1]))
    if velocity_limits[0] >= velocity_limits[1]:
        raise ValueError(f"Velocity display limits must be increasing, got {velocity_limits}.")

    from matplotlib import colormaps
    from matplotlib.colors import Normalize
    from PIL import Image

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    normalizer = Normalize(*velocity_limits, clip=True)
    color_map = colormaps[VELOCITY_COLORMAP]
    rows: list[dict[str, Any]] = []
    for case in visible_cases:
        key = _record_order(case)
        artifact = artifacts.get(key)
        if artifact is None:
            raise KeyError(f"No reconstructed arrays are available for standalone export {key}.")
        target = _image_2d(artifact["target"], f"target for standalone export {key}")
        metrics = artifact.get("metrics")
        if not isinstance(metrics, Mapping):
            raise ValueError(f"No rendered metrics are available for standalone export {key}.")
        images: list[tuple[str, str, np.ndarray, float]] = [("Ground Truth", "target", target, 1.0)]
        for method_name in method_names:
            try:
                prediction = _image_2d(
                    artifact["predictions"][method_name],
                    f"{method_name} prediction for standalone export {key}",
                )
                ssim = float(metrics[method_name]["ssim"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"No usable prediction or SSIM is available for {method_name} on {key}.") from exc
            images.append((method_name, "prediction", prediction, ssim))

        for method_name, image_role, image, ssim in images:
            filename = (
                f"{_filename_slug(key[0])}_{int(key[1]):06d}__"
                f"{_filename_slug(method_name)}__ssim_{ssim:.4f}.png"
            )
            path = destination / filename
            Image.fromarray(color_map(normalizer(image), bytes=True)).save(path)
            rows.append(
                {
                    "dataset_name": key[0],
                    "source_sample_index": int(key[1]),
                    "method": method_name,
                    "image_role": image_role,
                    "ssim": ssim,
                    "velocity_vmin_mps": velocity_limits[0],
                    "velocity_vmax_mps": velocity_limits[1],
                    "filename": filename,
                }
            )
    return rows


def render_standalone_error_images(
    *,
    cases: Sequence[Mapping[str, Any]],
    artifacts: Mapping[RecordKey, Mapping[str, Any]],
    method_names: Sequence[str],
    output_dir: str | Path,
    error_limits: tuple[float, float],
) -> list[dict[str, Any]]:
    """Export one pixel-only absolute-error PNG per method and selected case."""

    visible_cases = [case for case in cases if case.get("status") == "selected"]
    if not visible_cases:
        raise ValueError("No selected cases are available for standalone error export.")
    if not method_names:
        raise ValueError("At least one method prediction is required for standalone error export.")

    error_limits = (float(error_limits[0]), float(error_limits[1]))
    if error_limits[0] < 0 or error_limits[0] >= error_limits[1]:
        raise ValueError(f"Error display limits must satisfy 0 <= min < max, got {error_limits}.")

    from matplotlib import colormaps
    from matplotlib.colors import Normalize
    from PIL import Image

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    normalizer = Normalize(*error_limits, clip=True)
    color_map = colormaps[ERROR_COLORMAP]
    rows: list[dict[str, Any]] = []
    for case in visible_cases:
        key = _record_order(case)
        artifact = artifacts.get(key)
        if artifact is None:
            raise KeyError(f"No reconstructed arrays are available for standalone error export {key}.")
        target = _image_2d(artifact["target"], f"target for standalone error export {key}")
        metrics = artifact.get("metrics")
        if not isinstance(metrics, Mapping):
            raise ValueError(f"No rendered metrics are available for standalone error export {key}.")
        for method_name in method_names:
            try:
                prediction = _image_2d(
                    artifact["predictions"][method_name],
                    f"{method_name} prediction for standalone error export {key}",
                )
                method_metrics = metrics[method_name]
                mae = float(method_metrics["mae"])
                rmse = float(method_metrics["rmse"])
                ssim = float(method_metrics["ssim"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"No usable prediction or metrics are available for {method_name} on {key}.") from exc
            if prediction.shape != target.shape:
                raise ValueError(f"Prediction and target shapes differ for standalone error export {key}.")
            error = np.abs(prediction - target)
            filename = (
                f"{_filename_slug(key[0])}_{int(key[1]):06d}__"
                f"{_filename_slug(method_name)}__error__mae_{mae:.4f}.png"
            )
            path = destination / filename
            Image.fromarray(color_map(normalizer(error), bytes=True)).save(path)
            rows.append(
                {
                    "dataset_name": key[0],
                    "source_sample_index": int(key[1]),
                    "method": method_name,
                    "reference": "Ground Truth",
                    "mae": mae,
                    "rmse": rmse,
                    "ssim": ssim,
                    "error_vmin_mps": error_limits[0],
                    "error_vmax_mps": error_limits[1],
                    "filename": filename,
                }
            )
    return rows


def _assert_baseline_native_metrics(
    *,
    key: RecordKey,
    artifact: Mapping[str, Any],
    baselines: Mapping[str, Mapping[RecordKey, Mapping[str, Any]]],
    tolerance: float = 5e-4,
) -> None:
    """Ensure deterministic baseline re-inference still reproduces formal CSV metrics."""

    for method_name, formal_rows in baselines.items():
        native = artifact["native_metrics"][method_name]
        formal = formal_rows[key]
        for metric in METRIC_COLUMNS:
            difference = abs(float(native[metric]) - float(formal[metric]))
            if difference > tolerance:
                raise RuntimeError(
                    f"{method_name} native re-inference diverged from formal {metric} for {key}: "
                    f"{native[metric]} versus {formal[metric]} (difference={difference})."
                )


def _attach_reconstructed_metrics(case: Mapping[str, Any], artifact: Mapping[str, Any]) -> dict[str, Any]:
    output = add_rendered_margins(case, artifact)
    output["pd_visual_seed"] = int(artifact["pd_visual_seed"])
    output["output_shape"] = list(artifact["shape"])
    return output


def _all_selected_cases(selections: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    return [dict(case) for group in ("advantage", "representative", "failure") for case in selections[group]
            if case.get("status") == "selected"]


def assert_complete_panel_selections(selections: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    """Reject incomplete qualitative panels instead of silently rendering partial evidence."""

    required_counts = {"advantage": 3, "representative": 8, "failure": 3}
    for group, expected_count in required_counts.items():
        cases = list(selections.get(group, ()))
        if len(cases) != expected_count:
            raise RuntimeError(
                f"Incomplete {group} panel: expected {expected_count} cases, got {len(cases)}."
            )
        incomplete = [
            index for index, case in enumerate(cases)
            if case.get("status") != "selected"
        ]
        if incomplete:
            raise RuntimeError(
                f"Incomplete {group} panel: non-selected case positions {incomplete}."
            )


def chunk_cases(
    cases: Sequence[Mapping[str, Any]],
    *,
    max_cases_per_panel: int,
) -> list[list[dict[str, Any]]]:
    """Split a deterministic case list into readable figure panels without reordering it."""

    if max_cases_per_panel <= 0:
        raise ValueError("max_cases_per_panel must be positive.")
    normalized = [dict(case) for case in cases]
    return [normalized[start:start + max_cases_per_panel] for start in range(0, len(normalized), max_cases_per_panel)]


def _common_error_limits(
    cases: Sequence[Mapping[str, Any]],
    artifacts: Mapping[RecordKey, Mapping[str, Any]],
) -> tuple[float, float]:
    maximum = 0.0
    for case in cases:
        artifact = artifacts[_record_order(case)]
        target = np.asarray(artifact["target"], dtype=np.float32)
        for prediction in artifact["predictions"].values():
            maximum = max(maximum, float(np.max(np.abs(np.asarray(prediction, dtype=np.float32) - target))))
    return 0.0, round(maximum if maximum > 0 else 1.0, 6)


def final_selection_manifest_rows(
    *,
    selections: Mapping[str, Sequence[Mapping[str, Any]]],
    formal_rows: Sequence[Mapping[str, Any]],
    baselines: Mapping[str, Mapping[RecordKey, Mapping[str, Any]]],
    pd_metrics: Mapping[RecordKey, Mapping[str, Any]],
    checkpoints: Mapping[str, Mapping[str, Any]],
    manifest_dataset_ids: Mapping[RecordKey, int],
    base_seed: int,
    artifacts: Mapping[RecordKey, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Extend formal selection provenance with re-inferred native and common-scale metrics."""

    rows = selection_manifest_rows(
        selections=selections,
        formal_rows=formal_rows,
        baselines=baselines,
        pd_metrics=pd_metrics,
        checkpoints=checkpoints,
        manifest_dataset_ids=manifest_dataset_ids,
        base_seed=base_seed,
    )
    for row in rows:
        if row["selection_status"] != "selected":
            continue
        key = (str(row["dataset_name"]), int(row["source_sample_index"]))
        artifact = artifacts[key]
        row["pd_visual_seed"] = int(artifact["pd_visual_seed"])
        row["prediction_shape"] = json.dumps(artifact["shape"])
        row["rendered_common_margin_l"] = rendered_margin_values(artifact["metrics"])["margin_l"]
        row["rendered_common_margin_h"] = rendered_margin_values(artifact["metrics"])["margin_h"]
        row["rendered_common_margin_s"] = rendered_margin_values(artifact["metrics"])["margin_s"]
        for method_name, metrics in artifact["metrics"].items():
            prefix = _slug(method_name)
            row[f"normalization_profile_{prefix}"] = artifact["normalization_profiles"][method_name]
            for metric in METRIC_COLUMNS:
                row[f"rendered_common_{prefix}_{metric}"] = metrics[metric]
                row[f"rendered_native_{prefix}_{metric}"] = artifact["native_metrics"][method_name][metric]
    return rows


def write_selection_report(
    path: str | Path,
    *,
    source_data: Mapping[str, Any],
    selections: Mapping[str, Sequence[Mapping[str, Any]]],
    figures: Mapping[str, Mapping[str, Any]],
    error_limits: tuple[float, float],
    observed_error_limits: tuple[float, float],
) -> None:
    """Write a human-readable, paper-safe account of automatic case selection."""

    lines = [
        "# Table 2 Qualitative Visualization Report",
        "",
        "## Protocol",
        "",
        f"- Exact global held-out records audited: `{len(source_data['manifest_keys'])}`.",
        "- Criterion-selected advantage cases are retained only when fixed-seed re-inference preserves their positive formal margin.",
        "- Per-subset cases maximize the immutable formal SSIM margin and retain that formal winner even if its re-rendered margin changes under deterministic PD-BG-RFM sampling.",
        "- Baseline predictions use each checkpoint's native formal normalization; plotted velocities are converted to m/s.",
        "- Cross-method rendered margins use a common OpenFWI [-1, 1] metric coordinate after physical conversion.",
        "- PD-BG-RFM uses a deterministic seed derived from `(2027, dataset_name, source_sample_index)`.",
        f"- Shared display ranges: velocity `[1500, 4500]` m/s; absolute error `{error_limits}` m/s.",
        f"- Observed selected-panel absolute-error range is `{observed_error_limits}` m/s; values above the display maximum are clipped for visibility.",
        "- Velocity maps use the fixed `jet` colormap; absolute-error maps use the sequential `inferno` colormap; every velocity panel displays its per-image SSIM.",
        "- Spatial axes use `x/z grid index` with ticks 0, 35, and 69; no physical-distance unit is implied.",
        "",
        "## Automatically Selected Cases",
        "",
    ]
    group_titles = {
        "advantage": "Criterion-Selected Advantage",
        "representative": "Per-Subset Maximum Formal SSIM Margin",
        "failure": "Failure",
    }
    for group in ("advantage", "representative", "failure"):
        lines.extend([f"### {group_titles[group]}", ""])
        for case in selections[group]:
            if case.get("status") != "selected":
                lines.append(f"- `{case['criterion']}`: no qualifying record.")
                continue
            key = f"{case['dataset_name']}/{case['source_sample_index']}"
            dataset_id = source_data["manifest_dataset_ids"].get(_record_order(case))
            if dataset_id is not None:
                key += f" (dataset_id={int(dataset_id)})"
            details = [f"criterion={case['criterion']}"]
            for margin in ADVANTAGE_CRITERIA:
                if margin in case:
                    details.append(f"{margin}={float(case[margin]):+.6f}")
                rendered = f"rendered_{margin}"
                if rendered in case:
                    details.append(f"{rendered}={float(case[rendered]):+.6f}")
            lines.append(f"- `{key}`: {', '.join(details)}.")
        lines.append("")
    lines.extend(
        [
            "## Interpretation Limit",
            "",
            "The criterion-selected and per-subset maximum-formal-SSIM-margin panels are deliberate case selections, not claims of aggregate superiority. The per-subset rule retains the maximum formal margin even when it is non-positive; rendered margins are reported for audit, and the failure panel is retained alongside it.",
            "",
            "## Figure Files",
            "",
        ]
    )
    for name, result in figures.items():
        if "pdf" in result and "png" in result:
            lines.append(f"- `{name}`: `{result['pdf']}` and `{result['png']}`.")
        elif "directory" in result and "manifest" in result:
            lines.append(
                f"- `{name}`: `{result['num_images']}` coordinate-free PNGs in `{result['directory']}` "
                f"with metadata in `{result['manifest']}`."
            )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_qualitative_visualization(
    *,
    output_dir: str | Path,
    device: str = "cuda:0",
    base_seed: int = 2027,
) -> dict[str, Any]:
    """Run the end-to-end audited Table 2 visualization reconstruction pipeline."""

    import torch
    from bg_pdr_fm.runtime import configure_torch_runtime

    configure_torch_runtime("medium")
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("A GPU device was requested for Table 2 reconstruction but CUDA/ROCm is unavailable.")
    output = Path(output_dir)
    source_data = load_formal_metric_sources()
    formal_audit = write_formal_selection_audit(output, source_data, base_seed=base_seed)
    _, profiles = load_method_configs_and_profiles(source_data)
    artifacts: dict[RecordKey, dict[str, Any]] = {}
    loaded_checkpoints: dict[str, dict[str, Any]] = {}
    try:
        formal_selection = formal_audit["selections"]
        advantage: list[dict[str, Any]] = []
        candidate_chunk = 16
        for rank_offset in range(0, 128, candidate_chunk):
            candidate_keys = formal_candidate_keys(
                formal_audit["formal_rows"],
                formal_selection,
                rank_offset=rank_offset,
                per_criterion=candidate_chunk,
            )
            pending = sorted(candidate_keys.difference(artifacts))
            if pending:
                new_artifacts, new_checkpoints = reconstruct_records_in_isolated_processes(
                    keys=pending,
                    source_data=source_data,
                    normalization_profiles=profiles,
                    device=str(torch_device),
                    base_seed=base_seed,
                    worker_dir=output / "isolated_workers",
                )
                for key, artifact in new_artifacts.items():
                    _assert_baseline_native_metrics(
                        key=key,
                        artifact=artifact,
                        baselines=source_data["baselines"],
                    )
                artifacts.update(new_artifacts)
                loaded_checkpoints.update(new_checkpoints)

            advantage, _ = select_rendered_advantage_cases(
                formal_rows=formal_audit["formal_rows"],
                reconstruct=lambda key: artifacts[key],
            )
            if all(case.get("status") == "selected" for case in advantage):
                break

        def reconstruct(key: RecordKey) -> dict[str, Any]:
            if key not in artifacts:
                new_artifacts, new_checkpoints = reconstruct_records_in_isolated_processes(
                    keys=[key],
                    source_data=source_data,
                    normalization_profiles=profiles,
                    device=str(torch_device),
                    base_seed=base_seed,
                    worker_dir=output / "isolated_workers",
                )
                artifact = new_artifacts[key]
                _assert_baseline_native_metrics(key=key, artifact=artifact, baselines=source_data["baselines"])
                artifacts.update(new_artifacts)
                loaded_checkpoints.update(new_checkpoints)
            return artifacts[key]

        representative = []
        failure = []
        for case in formal_selection["representative"]:
            representative.append(_attach_reconstructed_metrics(case, reconstruct(_record_order(case))))
        for case in formal_selection["failure"]:
            if case["status"] == "selected":
                failure.append(_attach_reconstructed_metrics(case, reconstruct(_record_order(case))))
            else:
                failure.append(dict(case))
        selections = {"advantage": advantage, "representative": representative, "failure": failure}
        assert_complete_panel_selections(selections)
        selected_cases = _all_selected_cases(selections)
        if not selected_cases:
            raise RuntimeError("No qualitative cases remained after automatic selection.")
        computed_error_limits = _common_error_limits(selected_cases, artifacts)
        error_limits = TABLE2_ERROR_LIMITS
        method_names = tuple(spec.display_name for spec in source_data["specs"])
        figure_dir = REPO_ROOT / "docs/paper/AAAI2027/figures/table2_qualitative"
        figures = {
            "table2_advantage_cases": render_case_figure(
                cases=selections["advantage"], artifacts=artifacts, method_names=method_names,
                output_base=figure_dir / "table2_advantage_cases",
                title="Criterion-selected PD-BG-RFM advantage cases",
                velocity_limits=(1500.0, 4500.0), velocity_label="Velocity (m/s)",
                error_label="Absolute error (m/s)", error_limits=error_limits,
            ),
            "table2_representative_cases": render_case_figure(
                cases=selections["representative"], artifacts=artifacts, method_names=method_names,
                output_base=figure_dir / "table2_representative_cases",
                title="Per-subset maximum formal PD-BG-RFM SSIM-margin cases",
                velocity_limits=(1500.0, 4500.0), velocity_label="Velocity (m/s)",
                error_label="Absolute error (m/s)", error_limits=error_limits,
            ),
            "table2_failure_cases": render_case_figure(
                cases=selections["failure"], artifacts=artifacts, method_names=method_names,
                output_base=figure_dir / "table2_failure_cases",
                title="PD-BG-RFM failure cases selected by negative margin",
                velocity_limits=(1500.0, 4500.0), velocity_label="Velocity (m/s)",
                error_label="Absolute error (m/s)", error_limits=error_limits,
            ),
        }
        representative_panels = chunk_cases(selections["representative"], max_cases_per_panel=4)
        for panel_index, panel_cases in enumerate(representative_panels, start=1):
            figures[f"table2_representative_cases_{panel_index}"] = render_case_figure(
                cases=panel_cases,
                artifacts=artifacts,
                method_names=method_names,
                output_base=figure_dir / f"table2_representative_cases_{panel_index}",
                title=(
                    "Per-subset maximum formal PD-BG-RFM SSIM-margin cases "
                    f"({panel_index}/{len(representative_panels)})"
                ),
                velocity_limits=(1500.0, 4500.0),
                velocity_label="Velocity (m/s)",
                error_label="Absolute error (m/s)",
                error_limits=error_limits,
            )
        standalone_dir = figure_dir / "standalone_velocity"
        standalone_rows = render_standalone_velocity_images(
            cases=selections["representative"],
            artifacts=artifacts,
            method_names=method_names,
            output_dir=standalone_dir,
            velocity_limits=(1500.0, 4500.0),
        )
        standalone_manifest = figure_dir / "standalone_velocity_manifest.csv"
        write_csv_rows(standalone_manifest, standalone_rows)
        figures["standalone_velocity"] = {
            "directory": str(standalone_dir),
            "manifest": str(standalone_manifest),
            "num_images": len(standalone_rows),
        }
        standalone_error_dir = figure_dir / "standalone_error"
        standalone_error_rows = render_standalone_error_images(
            cases=selections["representative"],
            artifacts=artifacts,
            method_names=method_names,
            output_dir=standalone_error_dir,
            error_limits=error_limits,
        )
        standalone_error_manifest = figure_dir / "standalone_error_manifest.csv"
        write_csv_rows(standalone_error_manifest, standalone_error_rows)
        figures["standalone_error"] = {
            "directory": str(standalone_error_dir),
            "manifest": str(standalone_error_manifest),
            "num_images": len(standalone_error_rows),
        }
        array_paths = save_reconstructed_arrays(output, artifacts)
        manifest_rows = final_selection_manifest_rows(
            selections=selections,
            formal_rows=formal_audit["formal_rows"],
            baselines=source_data["baselines"],
            pd_metrics=source_data["pd_metrics"],
            checkpoints=source_data["checkpoints"],
            manifest_dataset_ids=source_data["manifest_dataset_ids"],
            base_seed=base_seed,
            artifacts=artifacts,
        )
        selection_manifest = output / "selection_manifest.csv"
        write_csv_rows(selection_manifest, manifest_rows)
        write_selection_report(
            output / "selection_report.md",
            source_data=source_data,
            selections=selections,
            figures=figures,
            error_limits=error_limits,
            observed_error_limits=computed_error_limits,
        )
        run_manifest = {
            "num_manifest_records": len(source_data["manifest_keys"]),
            "expected_global_held_out_records": EXPECTED_GLOBAL_HELD_OUT_RECORDS,
            "base_seed": int(base_seed),
            "device": str(torch_device),
            "normalization_profiles": profiles,
            "checkpoints": loaded_checkpoints,
            "selection_manifest": str(selection_manifest),
            "arrays": array_paths,
            "figures": figures,
            "error_limits_mps": error_limits,
            "observed_error_limits_mps": computed_error_limits,
            "advantage_complete": all(case.get("status") == "selected" for case in advantage),
        }
        (output / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return {"selections": selections, "artifacts": artifacts, "run_manifest": run_manifest}
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build auditable qualitative panels for AAAI2027 Table 2.")
    parser.add_argument(
        "--output-dir",
        default="logs/bg_pdr_fm/aaai27/table2_qualitative",
        help="Audit output directory; rendered paper figures are always written under docs/paper/AAAI2027/figures/.",
    )
    parser.add_argument("--device", default="cuda:0", help="Torch device for selected-record reconstruction.")
    parser.add_argument("--base-seed", type=int, default=2027, help="Base seed for record-keyed PD-BG-RFM sampling.")
    parser.add_argument("--audit-only", action="store_true", help="Write metric-only candidate audit without loading models.")
    parser.add_argument("--isolated-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--worker-method", help=argparse.SUPPRESS)
    parser.add_argument("--records-json", help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> dict[str, Any]:
    args = _parse_args()
    if args.isolated_worker:
        if not args.worker_method or not args.records_json or not args.worker_output:
            raise ValueError("An isolated worker requires --worker-method, --records-json, and --worker-output.")
        worker_result = reconstruct_isolated_worker(
            method_name=args.worker_method,
            keys=_read_record_keys(args.records_json),
            device=args.device,
            base_seed=args.base_seed,
        )
        write_isolated_worker_output(
            args.worker_output,
            method_name=worker_result["method_name"],
            checkpoint=worker_result["checkpoint"],
            records=worker_result["records"],
        )
        return {
            "worker": {
                "method_name": worker_result["method_name"],
                "num_records": len(worker_result["records"]),
                "output": str(args.worker_output),
            }
        }
    if args.audit_only:
        source_data = load_formal_metric_sources()
        result = write_formal_selection_audit(args.output_dir, source_data, base_seed=args.base_seed)
        return {"audit": result["report"]}
    return run_qualitative_visualization(output_dir=args.output_dir, device=args.device, base_seed=args.base_seed)


if __name__ == "__main__":
    result = main()
    print(json.dumps(result.get("run_manifest", result), indent=2, sort_keys=True))
