"""Shared utilities for same-input adapted external baselines."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from bg_pdr_fm.data.batch import BGSampleBatch


def resize_channel(x: torch.Tensor, size: tuple[int, int], *, mode: str) -> torch.Tensor:
    """Resize a BCHW single-modality tensor with explicit interpolation mode."""

    if x.ndim != 4:
        raise ValueError(f"expected BCHW tensor, got {tuple(x.shape)}")
    size = tuple(int(v) for v in size)
    if mode == "nearest":
        return F.interpolate(x, size=size, mode=mode)
    return F.interpolate(x, size=size, mode=mode, align_corners=False)


def build_multimodal_condition_image(
    batch: BGSampleBatch,
    input_hw: tuple[int, int] = (1000, 70),
) -> torch.Tensor:
    """Build the common five-channel adapted-baseline condition tensor.

    Channel order is PSTM/migrated image, RMS velocity, horizon,
    masked well-log value, and well mask.
    """

    input_hw = tuple(int(v) for v in input_hw)
    migrated = resize_channel(batch.migrated_image, input_hw, mode="bilinear")
    rms = resize_channel(batch.rms_vel, input_hw, mode="bilinear")
    horizon = resize_channel(batch.horizon, input_hw, mode="nearest")
    well_mask = resize_channel(batch.well_mask, input_hw, mode="nearest")
    well_log = resize_channel(batch.well_log * batch.well_mask, input_hw, mode="bilinear")
    return torch.cat([migrated, rms, horizon, well_log, well_mask], dim=1).contiguous()
