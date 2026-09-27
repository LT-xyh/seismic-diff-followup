"""Batch schema and collation helpers for BG-PDR-FM physical modalities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import torch


STANDARD_MODALITIES = ("migrated_image", "horizon", "rms_vel", "well_log")
REQUIRED_SAMPLE_KEYS = ("depth_vel", *STANDARD_MODALITIES, "well_mask")
REQUIRED_INFERENCE_KEYS = (*STANDARD_MODALITIES, "well_mask", "modality_mask", "modality_quality")
INFERENCE_FORBIDDEN_KEYS = frozenset(
    {
        "depth_vel",
        "target",
        "A_bg",
        "A_str",
        "low_anchor",
        "high_anchor",
        "oracle_residual",
        "oracle_latent",
        "P_L(V)",
        "P_H(V)",
        "rho_true",
        "rho_B",
    }
)


def _is_inference_forbidden_key(key: object) -> bool:
    key_text = str(key)
    return key_text in INFERENCE_FORBIDDEN_KEYS or any(
        marker in key_text.lower() for marker in ("target", "anchor", "oracle")
    )


@dataclass
class BGSampleBatch:
    depth_vel: torch.Tensor
    migrated_image: torch.Tensor
    horizon: torch.Tensor
    rms_vel: torch.Tensor
    well_log: torch.Tensor
    well_mask: torch.Tensor
    modality_mask: torch.Tensor
    modality_quality: torch.Tensor
    metadata: dict[str, torch.Tensor] = field(default_factory=dict)

    def as_model_inputs(self) -> dict[str, torch.Tensor]:
        return {
            "migrated_image": self.migrated_image,
            "horizon": self.horizon,
            "rms_vel": self.rms_vel,
            "well_log": self.well_log,
        }

    @property
    def output_hw(self) -> tuple[int, int]:
        return tuple(int(value) for value in self.depth_vel.shape[-2:])

    def as_inference_batch(self) -> BGInferenceBatch:
        """Drop training targets before entering the prediction boundary."""
        return BGInferenceBatch(
            migrated_image=self.migrated_image,
            horizon=self.horizon,
            rms_vel=self.rms_vel,
            well_log=self.well_log,
            well_mask=self.well_mask,
            modality_mask=self.modality_mask,
            modality_quality=self.modality_quality,
            metadata={
                key: value
                for key, value in self.metadata.items()
                if not _is_inference_forbidden_key(key)
            },
        )


@dataclass
class BGInferenceBatch:
    """Observed-only inputs for BG-PDR-FM prediction."""

    migrated_image: torch.Tensor
    horizon: torch.Tensor
    rms_vel: torch.Tensor
    well_log: torch.Tensor
    well_mask: torch.Tensor
    modality_mask: torch.Tensor
    modality_quality: torch.Tensor
    metadata: dict[str, torch.Tensor] = field(default_factory=dict)
    output_hw: tuple[int, int] = field(init=False)

    def __post_init__(self) -> None:
        self.output_hw = tuple(int(value) for value in self.horizon.shape[-2:])
        validate_bg_inference_batch(self)

    def as_model_inputs(self) -> dict[str, torch.Tensor]:
        return {
            "migrated_image": self.migrated_image,
            "horizon": self.horizon,
            "rms_vel": self.rms_vel,
            "well_log": self.well_log,
        }


def _ensure_chw(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 2:
        return x.unsqueeze(0)
    if x.ndim == 3:
        return x
    raise ValueError(f"Expected CHW or HW tensor, got shape {tuple(x.shape)}.")


def _stack_required(items: list[Mapping[str, torch.Tensor]], key: str) -> torch.Tensor:
    return torch.stack([_ensure_chw(item[key]).to(torch.float32) for item in items], dim=0)


def validate_bg_sample(item: Mapping[str, torch.Tensor], context: str = "sample") -> None:
    missing = [key for key in REQUIRED_SAMPLE_KEYS if key not in item]
    if missing:
        raise KeyError(f"BG-PDR-FM {context} is missing required keys: {missing}.")
    for key in REQUIRED_SAMPLE_KEYS:
        value = item[key]
        if not torch.is_tensor(value):
            raise TypeError(f"BG-PDR-FM {context} key {key!r} must be a tensor, got {type(value).__name__}.")
        if value.ndim not in (2, 3):
            raise ValueError(f"BG-PDR-FM {context} key {key!r} must be HW or CHW, got shape {tuple(value.shape)}.")


def _validate_inference_keys(keys, context: str) -> None:
    forbidden = sorted(str(key) for key in keys if _is_inference_forbidden_key(key))
    if forbidden:
        raise ValueError(
            f"BG-PDR-FM {context} must be observed-only; target-derived keys are forbidden: {forbidden}."
        )


def validate_bg_inference_sample(item: Mapping[str, torch.Tensor], context: str = "inference sample") -> None:
    _validate_inference_keys(item.keys(), context)
    missing = [key for key in REQUIRED_INFERENCE_KEYS if key not in item]
    if missing:
        raise KeyError(f"BG-PDR-FM {context} is missing required observed keys: {missing}.")
    for key in REQUIRED_INFERENCE_KEYS:
        value = item[key]
        if not torch.is_tensor(value):
            raise TypeError(f"BG-PDR-FM {context} key {key!r} must be a tensor, got {type(value).__name__}.")
        if key in {"modality_mask", "modality_quality"}:
            if value.ndim != 1:
                raise ValueError(f"BG-PDR-FM {context} key {key!r} must be M, got shape {tuple(value.shape)}.")
        elif value.ndim not in (2, 3):
            raise ValueError(f"BG-PDR-FM {context} key {key!r} must be HW or CHW, got shape {tuple(value.shape)}.")


def validate_bg_batch(batch: BGSampleBatch, context: str = "batch") -> None:
    tensors = {
        "depth_vel": batch.depth_vel,
        "migrated_image": batch.migrated_image,
        "horizon": batch.horizon,
        "rms_vel": batch.rms_vel,
        "well_log": batch.well_log,
        "well_mask": batch.well_mask,
    }
    for key, value in tensors.items():
        if value.ndim != 4:
            raise ValueError(f"BG-PDR-FM {context} key {key!r} must be BCHW, got {tuple(value.shape)}.")
        if value.dtype != torch.float32:
            raise TypeError(f"BG-PDR-FM {context} key {key!r} must be float32, got {value.dtype}.")
    depth_shape = tuple(batch.depth_vel.shape)
    for key in ("horizon", "well_log", "well_mask"):
        shape = tuple(tensors[key].shape)
        if shape != depth_shape:
            raise ValueError(
                f"BG-PDR-FM {context} key {key!r} must match depth_vel shape {depth_shape}, got {shape}."
            )
    if batch.migrated_image.shape[0] != batch.depth_vel.shape[0]:
        raise ValueError(
            f"BG-PDR-FM {context} migrated_image batch size must match depth_vel, "
            f"got {batch.migrated_image.shape[0]} and {batch.depth_vel.shape[0]}."
        )
    if batch.rms_vel.shape[0] != batch.depth_vel.shape[0]:
        raise ValueError(
            f"BG-PDR-FM {context} rms_vel batch size must match depth_vel, "
            f"got {batch.rms_vel.shape[0]} and {batch.depth_vel.shape[0]}."
        )
    expected = (batch.depth_vel.shape[0], len(STANDARD_MODALITIES))
    if tuple(batch.modality_mask.shape) != expected:
        raise ValueError(
            f"BG-PDR-FM {context} modality_mask must be {expected}, got {tuple(batch.modality_mask.shape)}."
        )
    if tuple(batch.modality_quality.shape) != expected:
        raise ValueError(
            f"BG-PDR-FM {context} modality_quality must be {expected}, got {tuple(batch.modality_quality.shape)}."
        )


def validate_bg_inference_batch(batch: BGInferenceBatch, context: str = "inference batch") -> None:
    tensors = {
        "migrated_image": batch.migrated_image,
        "horizon": batch.horizon,
        "rms_vel": batch.rms_vel,
        "well_log": batch.well_log,
        "well_mask": batch.well_mask,
    }
    for key, value in tensors.items():
        if value.ndim != 4:
            raise ValueError(f"BG-PDR-FM {context} key {key!r} must be BCHW, got {tuple(value.shape)}.")
        if value.dtype != torch.float32:
            raise TypeError(f"BG-PDR-FM {context} key {key!r} must be float32, got {value.dtype}.")
    observed_depth_shape = tuple(batch.horizon.shape)
    for key in ("well_log", "well_mask"):
        shape = tuple(tensors[key].shape)
        if shape != observed_depth_shape:
            raise ValueError(
                f"BG-PDR-FM {context} key {key!r} must match observed horizon shape {observed_depth_shape}, "
                f"got {shape}."
            )
    for key in ("migrated_image", "rms_vel"):
        if tensors[key].shape[0] != batch.horizon.shape[0]:
            raise ValueError(
                f"BG-PDR-FM {context} {key} batch size must match horizon, "
                f"got {tensors[key].shape[0]} and {batch.horizon.shape[0]}."
            )
    expected = (batch.horizon.shape[0], len(STANDARD_MODALITIES))
    if tuple(batch.modality_mask.shape) != expected:
        raise ValueError(
            f"BG-PDR-FM {context} modality_mask must be {expected}, got {tuple(batch.modality_mask.shape)}."
        )
    if tuple(batch.modality_quality.shape) != expected:
        raise ValueError(
            f"BG-PDR-FM {context} modality_quality must be {expected}, got {tuple(batch.modality_quality.shape)}."
        )
    if tuple(batch.output_hw) != tuple(batch.horizon.shape[-2:]):
        raise ValueError(
            f"BG-PDR-FM {context} output_hw must be derived from horizon, got {batch.output_hw}."
        )
    _validate_inference_keys(batch.metadata.keys(), f"{context} metadata")
    for key, value in batch.metadata.items():
        if not torch.is_tensor(value):
            raise TypeError(f"BG-PDR-FM {context} metadata {key!r} must be a tensor, got {type(value).__name__}.")


def collate_bg_samples(items: list[Mapping[str, torch.Tensor]]) -> BGSampleBatch:
    if not items:
        raise ValueError("Cannot collate an empty batch.")
    for i, item in enumerate(items):
        validate_bg_sample(item, context=f"sample[{i}]")

    batch_size = len(items)
    mask = []
    quality = []
    for item in items:
        item_mask = item.get("modality_mask")
        item_quality = item.get("modality_quality")
        if item_mask is None:
            item_mask = torch.ones(len(STANDARD_MODALITIES), dtype=torch.float32)
        if item_quality is None:
            item_quality = torch.ones(len(STANDARD_MODALITIES), dtype=torch.float32)
        mask.append(item_mask.to(torch.float32).view(-1))
        quality.append(item_quality.to(torch.float32).view(-1))

    metadata: dict[str, torch.Tensor] = {}
    skip_keys = {
        "depth_vel",
        "migrated_image",
        "horizon",
        "rms_vel",
        "well_log",
        "well_mask",
        "modality_mask",
        "modality_quality",
    }
    for key in items[0].keys():
        if key in skip_keys:
            continue
        values = [item[key] for item in items if key in item]
        if len(values) == batch_size and all(torch.is_tensor(value) for value in values):
            metadata[key] = torch.stack([value.to(torch.float32) for value in values], dim=0)

    batch = BGSampleBatch(
        depth_vel=_stack_required(items, "depth_vel"),
        migrated_image=_stack_required(items, "migrated_image"),
        horizon=_stack_required(items, "horizon"),
        rms_vel=_stack_required(items, "rms_vel"),
        well_log=_stack_required(items, "well_log"),
        well_mask=_stack_required(items, "well_mask"),
        modality_mask=torch.stack(mask, dim=0),
        modality_quality=torch.stack(quality, dim=0),
        metadata=metadata,
    )
    validate_bg_batch(batch)
    return batch


def collate_bg_inference_samples(items: list[Mapping[str, torch.Tensor]]) -> BGInferenceBatch:
    if not items:
        raise ValueError("Cannot collate an empty inference batch.")
    for i, item in enumerate(items):
        validate_bg_inference_sample(item, context=f"inference sample[{i}]")

    batch_size = len(items)
    metadata: dict[str, torch.Tensor] = {}
    skip_keys = set(REQUIRED_INFERENCE_KEYS)
    for key in items[0].keys():
        if key in skip_keys:
            continue
        values = [item[key] for item in items if key in item]
        if len(values) == batch_size and all(torch.is_tensor(value) for value in values):
            metadata[key] = torch.stack([value.to(torch.float32) for value in values], dim=0)

    return BGInferenceBatch(
        migrated_image=_stack_required(items, "migrated_image"),
        horizon=_stack_required(items, "horizon"),
        rms_vel=_stack_required(items, "rms_vel"),
        well_log=_stack_required(items, "well_log"),
        well_mask=_stack_required(items, "well_mask"),
        modality_mask=torch.stack([item["modality_mask"].to(torch.float32).view(-1) for item in items], dim=0),
        modality_quality=torch.stack(
            [item["modality_quality"].to(torch.float32).view(-1) for item in items], dim=0
        ),
        metadata=metadata,
    )


def inference_batch_from_mapping(batch: Mapping[str, torch.Tensor]) -> BGInferenceBatch:
    """Construct an observed-only batch from a public prediction mapping."""
    _validate_inference_keys(batch.keys(), "prediction mapping")
    missing = [key for key in REQUIRED_INFERENCE_KEYS if key not in batch]
    if missing:
        raise KeyError(f"BG-PDR-FM prediction mapping is missing required observed keys: {missing}.")
    first = batch["horizon"]
    if not torch.is_tensor(first):
        raise TypeError(f"BG-PDR-FM prediction mapping key 'horizon' must be a tensor, got {type(first).__name__}.")
    if first.ndim == 3:
        return collate_bg_inference_samples([batch])
    if first.ndim != 4:
        raise ValueError(f"BG-PDR-FM prediction mapping horizon must be BCHW or CHW, got {tuple(first.shape)}.")
    core_keys = set(REQUIRED_INFERENCE_KEYS)
    metadata = {key: value for key, value in batch.items() if key not in core_keys}
    return BGInferenceBatch(
        migrated_image=batch["migrated_image"],
        horizon=batch["horizon"],
        rms_vel=batch["rms_vel"],
        well_log=batch["well_log"],
        well_mask=batch["well_mask"],
        modality_mask=batch["modality_mask"],
        modality_quality=batch["modality_quality"],
        metadata=metadata,
    )


def batch_to_device(batch: BGSampleBatch | BGInferenceBatch, device: torch.device | str) -> BGSampleBatch | BGInferenceBatch:
    metadata = {key: value.to(device) for key, value in batch.metadata.items()}
    if isinstance(batch, BGInferenceBatch):
        return BGInferenceBatch(
            migrated_image=batch.migrated_image.to(device),
            horizon=batch.horizon.to(device),
            rms_vel=batch.rms_vel.to(device),
            well_log=batch.well_log.to(device),
            well_mask=batch.well_mask.to(device),
            modality_mask=batch.modality_mask.to(device),
            modality_quality=batch.modality_quality.to(device),
            metadata=metadata,
        )
    return BGSampleBatch(
        depth_vel=batch.depth_vel.to(device),
        migrated_image=batch.migrated_image.to(device),
        horizon=batch.horizon.to(device),
        rms_vel=batch.rms_vel.to(device),
        well_log=batch.well_log.to(device),
        well_mask=batch.well_mask.to(device),
        modality_mask=batch.modality_mask.to(device),
        modality_quality=batch.modality_quality.to(device),
        metadata=metadata,
    )
