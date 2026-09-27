"""Typed containers passed between BG-PDR-FM model components."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class DecoupledFeatures:
    numerical: torch.Tensor
    structural: torch.Tensor
    unique: torch.Tensor
    reliability: torch.Tensor
    all_heads: dict[str, dict[str, torch.Tensor]] | None = None
    mixed: torch.Tensor | None = None


@dataclass
class Conditions:
    numerical: torch.Tensor
    structural: torch.Tensor
    unique: torch.Tensor | None = None
    structural_modulated: torch.Tensor | None = None


@dataclass
class BackgroundEstimate:
    bg_hat: torch.Tensor
    epsilon_h_b: torch.Tensor
    bg_ll: torch.Tensor | None = None
    bg_lowres: torch.Tensor | None = None
    fm_loss: torch.Tensor | None = None
    fm_velocity_pred: torch.Tensor | None = None
    fm_velocity_target: torch.Tensor | None = None


@dataclass
class PredictionBatch:
    bg_hat: torch.Tensor
    velocity_hat: torch.Tensor
    residual_hat: torch.Tensor
    rho_hat_b: torch.Tensor
    alpha_hat: torch.Tensor
    epsilon_h_b: torch.Tensor
    metadata: dict[str, Any]
