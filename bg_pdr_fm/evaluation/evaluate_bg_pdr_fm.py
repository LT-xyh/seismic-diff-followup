"""Checkpoint-first evaluation for BG-PDR-FM stage and full checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from bg_pdr_fm.data import batch_to_device, collate_bg_samples, validate_bg_batch
from bg_pdr_fm.diagnostics import append_metrics_csv, save_tensor_panel
from bg_pdr_fm.evaluation.benchmark_metrics import (
    compute_velocity_metrics,
    residual_energy_ratio_l2,
    transport_target_ratio,
    validate_metric_row,
)
from bg_pdr_fm.evaluation.run_manifest import (
    capture_evaluation_provenance,
    record_identity_key,
    write_run_manifest,
)
from bg_pdr_fm.lightning import BGPDRFMLightning
from bg_pdr_fm.lightning.stage_losses import true_residual_low_frequency_ratio
from bg_pdr_fm.runtime import configure_checkpoint_temp_dir, configure_torch_runtime, configured_torch_device
from bg_pdr_fm.training.benchmark_config import load_benchmark_config
from bg_pdr_fm.training.train_bg_pdr_fm import build_dataset


CONTRASTIVE_PREFIXES = (
    "encoder.",
    "adapters.",
    "rho_calibrator.",
    "num_to_anchor.",
    "struct_to_anchor.",
    "contrastive_loss.",
)
BACKGROUND_PREFIXES = ("adapters.", "background.", "rho_calibrator.")
RESIDUAL_PREFIXES = ("adapters.structural.", "rho_calibrator.", "residual.")


def _conf_get(conf: Any, path: str, default: Any) -> Any:
    value = OmegaConf.select(conf, path, default=default)
    return value


def _path_or_none(value: Any) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    return Path(text)


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    if not path.is_file():
        raise FileNotFoundError(f"BG-PDR-FM evaluation checkpoint not found: {path}")
    state = torch.load(path, map_location="cpu")
    if not isinstance(state, dict) or "state_dict" not in state:
        raise KeyError(f"BG-PDR-FM evaluation checkpoint {path} does not contain `state_dict`.")
    state_dict = state["state_dict"]
    if not isinstance(state_dict, dict):
        raise TypeError(f"BG-PDR-FM evaluation checkpoint {path} has a non-dict `state_dict`.")
    return state_dict


def _load_full_checkpoint(model: BGPDRFMLightning, path: Path) -> dict[str, Any]:
    state_dict = _load_state_dict(path)
    current_state = model.state_dict()
    compatible_state = {
        key: value
        for key, value in state_dict.items()
        if key not in current_state or tuple(value.shape) == tuple(current_state[key].shape)
    }
    incompatible = model.load_state_dict(compatible_state, strict=False)
    return {
        "mode": "full",
        "path": str(path),
        "loaded_keys": len(compatible_state),
        "skipped_shape_keys": len(state_dict) - len(compatible_state),
        "missing_keys": len(incompatible.missing_keys),
        "unexpected_keys": len(incompatible.unexpected_keys),
    }


def _filter_prefixes(state_dict: dict[str, torch.Tensor], prefixes: tuple[str, ...]) -> dict[str, torch.Tensor]:
    return {key: value for key, value in state_dict.items() if key.startswith(prefixes)}


def _filter_compatible_shapes(
    model: BGPDRFMLightning,
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    current_state = model.state_dict()
    return {
        key: value
        for key, value in state_dict.items()
        if key not in current_state or tuple(value.shape) == tuple(current_state[key].shape)
    }


def _load_prefixed_checkpoint(
    model: BGPDRFMLightning,
    path: Path,
    prefixes: tuple[str, ...],
    label: str,
) -> dict[str, Any]:
    state_dict = _load_state_dict(path)
    filtered = _filter_prefixes(state_dict, prefixes)
    if not filtered:
        raise KeyError(
            f"BG-PDR-FM evaluation checkpoint {path} has no keys for {label}; "
            f"expected prefixes: {', '.join(prefixes)}"
        )
    compatible = _filter_compatible_shapes(model, filtered)
    incompatible = model.load_state_dict(compatible, strict=False)
    return {
        "mode": label,
        "path": str(path),
        "loaded_keys": len(compatible),
        "skipped_shape_keys": len(filtered) - len(compatible),
        "missing_keys": len(incompatible.missing_keys),
        "unexpected_keys": len(incompatible.unexpected_keys),
    }


def _background_prefixes_for_config(conf: Any) -> tuple[str, ...]:
    prefixes = list(BACKGROUND_PREFIXES)
    condition_source = str(_conf_get(conf, "model.background_condition_source", "adapter_numerical")).lower()
    if condition_source == "encoder_numerical_raw_bypass":
        prefixes.append("background_raw_bypass.")
    if bool(_conf_get(conf, "training.tune_encoder_numerical_heads_for_background", False)):
        prefixes.extend(
            f"encoder.heads.{modality}.numerical."
            for modality in ("migrated_image", "horizon", "rms_vel", "well_log")
        )
    return tuple(prefixes)


def load_evaluation_checkpoints(model: BGPDRFMLightning, conf: Any) -> list[dict[str, Any]]:
    allow_untrained = bool(_conf_get(conf, "evaluation.allow_untrained", False))
    full_path = _path_or_none(_conf_get(conf, "evaluation.checkpoints.full", None))
    if full_path is not None:
        return [_load_full_checkpoint(model, full_path)]

    stage_specs = (
        ("contrastive", CONTRASTIVE_PREFIXES),
        ("background", _background_prefixes_for_config(conf)),
        ("residual", RESIDUAL_PREFIXES),
    )
    loaded: list[dict[str, Any]] = []
    for label, prefixes in stage_specs:
        path = _path_or_none(_conf_get(conf, f"evaluation.checkpoints.{label}", None))
        if path is not None:
            loaded.append(_load_prefixed_checkpoint(model, path, prefixes, label))
    if loaded or allow_untrained:
        return loaded
    raise FileNotFoundError(
        "BG-PDR-FM evaluation requires a checkpoint by default. Set `evaluation.checkpoints.full` "
        "or stage checkpoints, or set `evaluation.allow_untrained: true` for smoke-only evaluation."
    )


def build_evaluation_loader(conf: Any) -> DataLoader:
    split = str(_conf_get(conf, "evaluation.split", "test"))
    if split not in {"val", "valid", "test"}:
        raise ValueError(f"evaluation.split must be one of 'val', 'valid', or 'test', got {split!r}.")
    dataset_split = "val" if split == "valid" else split
    dataset = build_dataset(conf, dataset_split)
    batch_size = int(_conf_get(conf, "training.batch_size", 2))
    num_workers = int(_conf_get(conf, "training.num_workers", 4))
    loader_kwargs = dict(
        num_workers=num_workers,
        pin_memory=bool(_conf_get(conf, "training.pin_memory", True)),
    )
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(_conf_get(conf, "training.persistent_workers", True))
        loader_kwargs["prefetch_factor"] = int(_conf_get(conf, "training.prefetch_factor", 3))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_bg_samples,
                        **loader_kwargs)
    first_batch = next(iter(loader))
    validate_bg_batch(first_batch, context="evaluation_loader first batch")
    return loader


def _cpu_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().float().cpu().numpy()


def _save_prediction_arrays(output_dir: Path, sample_index: int, arrays: dict[str, torch.Tensor]) -> None:
    pred_dir = output_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    for name, tensor in arrays.items():
        np.save(pred_dir / f"{sample_index:06d}_{name}.npy", _cpu_numpy(tensor))


def _save_prediction_panel(output_dir: Path, sample_index: int, arrays: dict[str, torch.Tensor]) -> None:
    save_tensor_panel(
        {
            "V": arrays["target"],
            "B_hat": arrays["bg_hat"],
            "V_hat": arrays["velocity_hat"],
            "abs_error": arrays["error"].abs(),
            "P_H(B_hat)": arrays["bg_high"].abs(),
        },
        output_dir / "panels" / f"{sample_index:06d}.png",
    )


def _mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def evaluation_record_identity(
    batch: Any,
    *,
    item_idx: int,
    dataset_names: list[str],
    fallback_source_sample_index: int | None = None,
) -> dict[str, int | str]:
    metadata = getattr(batch, "metadata", {}) or {}
    dataset_ids = metadata.get("dataset_id")
    source_indices = metadata.get("sample_index")
    if dataset_ids is None or source_indices is None:
        fallback_index = item_idx if fallback_source_sample_index is None else fallback_source_sample_index
        if fallback_index < 0:
            raise ValueError("synthetic fallback source_sample_index must be non-negative")
        return {
            "dataset_id": -1,
            "dataset_name": "synthetic",
            "source_sample_index": int(fallback_index),
        }
    dataset_id = int(dataset_ids[item_idx].detach().cpu().item())
    source_sample_index = int(source_indices[item_idx].detach().cpu().item())
    if not dataset_names:
        return {
            "dataset_id": dataset_id,
            "dataset_name": "",
            "source_sample_index": source_sample_index,
        }
    if dataset_id < 0 or dataset_id >= len(dataset_names):
        raise ValueError(f"dataset_id {dataset_id} is outside configured OpenFWI datasets {dataset_names}.")
    return {
        "dataset_id": dataset_id,
        "dataset_name": dataset_names[dataset_id],
        "source_sample_index": source_sample_index,
    }


def emit_validated_evaluation_record(
    row: Mapping[str, Any],
    record_identity: Mapping[str, Any],
    seen_record_identities: set[tuple[int, str, int]],
    emit: Callable[[], None],
) -> None:
    """Validate one record before any evaluation artifact is emitted."""
    validate_metric_row(row)
    identity_key = record_identity_key(record_identity)
    if identity_key in seen_record_identities:
        raise ValueError(f"duplicate record identity: {identity_key}")
    seen_record_identities.add(identity_key)
    emit()


def _config_summary(conf: Any) -> dict[str, Any]:
    return {
        "data_name": str(_conf_get(conf, "data.name", "synthetic")),
        "split_seed": int(_conf_get(conf, "data.split_seed", 42)),
        "split_fractions": list(_conf_get(conf, "data.split_fractions", [0.7, 0.2, 0.1])),
        "split_scope_datasets": list(_conf_get(conf, "data.openfwi_datasets", [])),
        "post_split_datasets": list(_conf_get(conf, "data.post_split_datasets", [])),
        "stage": str(_conf_get(conf, "training.stage", "background")),
        "codec_type": str(_conf_get(conf, "model.codec_type", "simple")),
        "residual_backend": str(_conf_get(conf, "model.residual_backend", "simple_fm")),
        "residual_num_inference_steps": int(_conf_get(conf, "model.residual_num_inference_steps", 20)),
        "evaluation_split": str(_conf_get(conf, "evaluation.split", "test")),
        "structural_ablation": str(_conf_get(conf, "evaluation.structural_ablation", "none")),
        "residual_override": str(_conf_get(conf, "evaluation.residual_override", "none")),
    }


def run_evaluation(conf: Any) -> dict[str, Any]:
    configure_checkpoint_temp_dir(_conf_get(conf, "training.checkpoint_tmp_dir", None))
    configure_torch_runtime(_conf_get(conf, "training.matmul_precision", None))
    output_dir = Path(str(_conf_get(conf, "evaluation.output_dir", "logs/bg_pdr_fm/evaluation")))
    save_arrays = bool(_conf_get(conf, "evaluation.save_arrays", True))
    save_panels = bool(_conf_get(conf, "evaluation.save_panels", True))
    max_batches = _conf_get(conf, "evaluation.max_batches", None)
    max_batches = None if max_batches is None else int(max_batches)
    dataset_names = list(_conf_get(conf, "data.openfwi_datasets", []))

    loader = build_evaluation_loader(conf)
    model = BGPDRFMLightning(conf)
    loaded_checkpoints = load_evaluation_checkpoints(model, conf)
    checkpoint_paths = [checkpoint["path"] for checkpoint in loaded_checkpoints if "path" in checkpoint]
    provenance_snapshot = capture_evaluation_provenance(conf, checkpoint_paths)
    device = configured_torch_device(
        _conf_get(conf, "training.accelerator", "auto"),
        _conf_get(conf, "training.devices", "auto"),
    )
    model.eval().to(device)

    metrics_path = output_dir / "metrics.csv"
    if metrics_path.exists():
        metrics_path.unlink()

    metric_values: dict[str, list[float]] = {
        "mae": [],
        "rmse": [],
        "mse": [],
        "ssim": [],
        "mae_l": [],
        "mae_h": [],
        "bg_mae": [],
        "epsilon_H_B": [],
        "rho_B": [],
        "rho_hat_B": [],
        "residual_energy_ratio_l2": [],
        "transport_target_ratio": [],
        "alpha_hat": [],
    }
    record_ids: list[dict[str, int | str]] = []
    seen_record_identities: set[tuple[int, str, int]] = set()
    metric_row_count = 0
    sample_index = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            batch = BGPDRFMLightning._as_batch(batch)
            batch = batch_to_device(batch, device)
            # Project before prediction; depth_vel remains only for post-prediction metrics.
            prediction = model.predict_batch(batch.as_inference_batch())
            low, _ = model.anchors(batch.depth_vel)
            bg_high = model.filter.highpass(prediction.bg_hat)
            rho_true = true_residual_low_frequency_ratio(model.filter, batch.depth_vel, prediction.bg_hat)
            energy_ratio_l2 = residual_energy_ratio_l2(batch.depth_vel, prediction.bg_hat)
            target_ratio = transport_target_ratio(batch.depth_vel, prediction.bg_hat)
            error = prediction.velocity_hat - batch.depth_vel
            mse = (error ** 2).flatten(1).mean(dim=1)
            velocity_metrics = compute_velocity_metrics(prediction.velocity_hat, batch.depth_vel, model.filter)
            bg_mae = (prediction.bg_hat - low).abs().flatten(1).mean(dim=1)

            for item_idx in range(batch.depth_vel.shape[0]):
                arrays = {
                    "velocity_hat": prediction.velocity_hat[item_idx:item_idx + 1],
                    "bg_hat": prediction.bg_hat[item_idx:item_idx + 1],
                    "target": batch.depth_vel[item_idx:item_idx + 1],
                    "error": error[item_idx:item_idx + 1],
                    "residual_hat": prediction.residual_hat[item_idx:item_idx + 1],
                    "bg_high": bg_high[item_idx:item_idx + 1],
                }
                row = {
                    "sample_index": sample_index,
                    "batch_index": batch_idx,
                    "item_index": item_idx,
                    "mae": float(velocity_metrics["mae"][item_idx].detach().cpu()),
                    "rmse": float(velocity_metrics["rmse"][item_idx].detach().cpu()),
                    "mse": float(mse[item_idx].detach().cpu()),
                    "ssim": float(velocity_metrics["ssim"][item_idx].detach().cpu()),
                    "mae_l": float(velocity_metrics["mae_l"][item_idx].detach().cpu()),
                    "mae_h": float(velocity_metrics["mae_h"][item_idx].detach().cpu()),
                    "bg_mae": float(bg_mae[item_idx].detach().cpu()),
                    "epsilon_H_B": float(prediction.epsilon_h_b[item_idx].detach().cpu()),
                    "rho_B": float(rho_true[item_idx].detach().cpu()),
                    "rho_hat_B": float(prediction.rho_hat_b[item_idx].detach().cpu()),
                    "residual_energy_ratio_l2": float(energy_ratio_l2[item_idx].detach().cpu()),
                    "transport_target_ratio": float(target_ratio[item_idx].detach().cpu()),
                    "alpha_hat": float(prediction.alpha_hat[item_idx].detach().cpu()),
                }
                record_identity = evaluation_record_identity(
                    batch,
                    item_idx=item_idx,
                    dataset_names=dataset_names,
                    fallback_source_sample_index=sample_index,
                )
                row.update(record_identity)
                def emit_record_artifacts() -> None:
                    if save_arrays:
                        _save_prediction_arrays(output_dir, sample_index, arrays)
                    if save_panels:
                        _save_prediction_panel(output_dir, sample_index, arrays)
                    append_metrics_csv(metrics_path, row)

                emit_validated_evaluation_record(
                    row,
                    record_identity,
                    seen_record_identities,
                    emit_record_artifacts,
                )
                record_ids.append(record_identity)
                for key in metric_values:
                    metric_values[key].append(float(row[key]))
                metric_row_count += 1
                sample_index += 1

    if metric_row_count != len(record_ids):
        raise RuntimeError("metric row count does not match evaluated record identities")
    expected_records: int | None = None
    if max_batches is None:
        expected_records = len(loader.dataset)
        if sample_index != expected_records:
            raise ValueError(
                f"evaluation record count mismatch: expected {expected_records}, got {sample_index}"
            )
    trajectory_count = _conf_get(conf, "evaluation.trajectory_count", 1)
    write_run_manifest(
        output_dir,
        conf,
        checkpoint_paths,
        record_ids,
        trajectory_count=1 if trajectory_count is None else int(trajectory_count),
        literature_transcribed=bool(_conf_get(conf, "evaluation.literature_transcribed", False)),
        expected_records=expected_records,
        provenance_snapshot=provenance_snapshot,
    )

    summary = {
        "num_samples": sample_index,
        "means": {key: _mean(values) for key, values in metric_values.items()},
        "config": _config_summary(conf),
        "checkpoints": loaded_checkpoints,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main(config_path: str = "bg_pdr_fm/configs/bg_pdr_fm.yaml") -> dict[str, Any]:
    conf = load_benchmark_config(config_path)
    return run_evaluation(conf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate BG-PDR-FM checkpoints.")
    parser.add_argument("--config", default="bg_pdr_fm/configs/bg_pdr_fm.yaml")
    args = parser.parse_args()
    main(args.config)
