"""Condition adapters that map decoupled encoder features to latent conditions."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .types import Conditions, DecoupledFeatures


class _ConditionAdapter(nn.Module):
    def __init__(self, in_channels: int = 32, hidden_channels: int = 64, out_channels: int = 64,
                 out_hw: tuple[int, int] = (16, 16)) -> None:
        super().__init__()
        self.out_hw = tuple(out_hw)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=min(8, hidden_channels), num_channels=hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=self.out_hw, mode="bilinear", align_corners=False)
        return self.net(x)


class NumericalConditionAdapter(_ConditionAdapter):
    pass


class StructuralConditionAdapter(_ConditionAdapter):
    pass


class UniqueConditionAdapter(nn.Module):
    """Gated/modulated U injection. Plain concat is intentionally not the default."""

    def __init__(self, in_channels: int = 32, cond_channels: int = 64, out_hw: tuple[int, int] = (16, 16),
                 eta: float = 0.1) -> None:
        super().__init__()
        self.eta = float(eta)
        self.out_hw = tuple(out_hw)
        self.to_residual = nn.Sequential(
            nn.Conv2d(in_channels, cond_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(cond_channels, cond_channels, kernel_size=1),
        )
        self.to_gate = nn.Sequential(
            nn.Conv2d(in_channels, cond_channels, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, structural_cond: torch.Tensor, unique: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        unique = F.interpolate(unique, size=self.out_hw, mode="bilinear", align_corners=False)
        residual = self.to_residual(unique)
        gate = self.to_gate(unique)
        modulated = structural_cond + self.eta * gate * residual
        return modulated, gate


class ConditionAdapters(nn.Module):
    def __init__(
        self,
        feature_channels: int = 32,
        cond_channels: int = 64,
        out_hw: tuple[int, int] = (16, 16),
        use_unique: bool = True,
        unique_eta: float = 0.1,
    ) -> None:
        super().__init__()
        self.numerical = NumericalConditionAdapter(feature_channels, cond_channels, cond_channels, out_hw)
        self.structural = StructuralConditionAdapter(feature_channels, cond_channels, cond_channels, out_hw)
        self.use_unique = bool(use_unique)
        self.unique = UniqueConditionAdapter(feature_channels, cond_channels, out_hw, eta=unique_eta)

    def forward(self, features: DecoupledFeatures) -> Conditions:
        cond_n = self.numerical(features.numerical)
        cond_s = self.structural(features.structural)
        cond_u = None
        cond_s_mod = cond_s
        if self.use_unique:
            cond_s_mod, cond_u = self.unique(cond_s, features.unique)
        return Conditions(numerical=cond_n, structural=cond_s, unique=cond_u, structural_modulated=cond_s_mod)
