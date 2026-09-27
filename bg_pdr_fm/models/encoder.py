"""Physics-decoupled multimodal encoder for BG-PDR-FM."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from bg_pdr_fm.data.batch import BGInferenceBatch, BGSampleBatch, STANDARD_MODALITIES
from .contrastive import MODALITY_PRIORS
from .modality_encoders import HorizonEncoderA, RMSVelocityEncoderA, SeismicImageEncoderA, WellLogEncoderA
from .types import DecoupledFeatures


class _TriSpaceHead(nn.Module):
    def __init__(self, in_channels: int, feature_channels: int) -> None:
        super().__init__()
        self.structural = nn.Conv2d(in_channels, feature_channels, kernel_size=1)
        self.numerical = nn.Conv2d(in_channels, feature_channels, kernel_size=1)
        self.unique = nn.Conv2d(in_channels, feature_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.numerical(x), self.structural(x), self.unique(x)


class PhysicsDecoupledEncoder(nn.Module):
    """Encoder that exposes only N/S/U/reliability, not mixed base features.

    Dataset loaders preserve the physical shape of each modality. Time-domain
    inputs such as migrated images or RMS velocity can therefore arrive as
    `(B, 1, 1000, 70)`, while depth-domain inputs are usually `(B, 1, 70, 70)`.
    Fusion happens only after each modality has been encoded and then projected
    to the observed depth-domain output grid.
    """

    def __init__(
        self,
        in_channels: int = 1,
        hidden_channels: int = 32,
        base_channels: int = 64,
        feature_channels: int = 32,
        fusion_mode: str = "reliability",
        modalities: tuple[str, ...] = STANDARD_MODALITIES,
    ) -> None:
        super().__init__()
        _ = in_channels, hidden_channels
        self.modalities = tuple(modalities)
        self.fusion_mode = str(fusion_mode).lower()
        if self.fusion_mode not in {"reliability", "uniform"}:
            raise ValueError(
                f"Unknown fusion_mode={fusion_mode!r}. Expected 'reliability' or 'uniform'."
            )
        unsupported = sorted(set(self.modalities) - set(STANDARD_MODALITIES))
        if unsupported:
            raise ValueError(f"Unsupported BG-PDR-FM modalities: {unsupported}.")
        self.modality_indices = [STANDARD_MODALITIES.index(name) for name in self.modalities]

        backbone_channels = {
            "migrated_image": base_channels,
            "horizon": feature_channels,
            "rms_vel": base_channels,
            "well_log": feature_channels,
        }
        self.backbones = nn.ModuleDict(
            {
                name: backbone
                for name, backbone in {
                    "migrated_image": SeismicImageEncoderA(C_out=base_channels, out_hw=(70, 70)),
                    "horizon": HorizonEncoderA(C_out=feature_channels, out_hw=(70, 70)),
                    "rms_vel": RMSVelocityEncoderA(c_out=base_channels, out_hw=(70, 70)),
                    "well_log": WellLogEncoderA(C_out=feature_channels, out_hw=(70, 70)),
                }.items()
                if name in self.modalities
            }
        )
        self.heads = nn.ModuleDict(
            {name: _TriSpaceHead(backbone_channels[name], feature_channels) for name in self.modalities}
        )
        self.base_heads = nn.ModuleDict(
            {name: nn.Conv2d(backbone_channels[name], feature_channels, kernel_size=1) for name in self.modalities}
        )
        self.reliability_heads = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(backbone_channels[name], 2),
                )
                for name in self.modalities
            }
        )
        self._init_reliability_priors()

    def _init_reliability_priors(self) -> None:
        for name in self.modalities:
            prior = MODALITY_PRIORS[name]
            head = self.reliability_heads[name][-1]
            nn.init.zeros_(head.weight)
            logits = torch.logit(torch.tensor(prior, dtype=head.bias.dtype), eps=1e-4)
            with torch.no_grad():
                head.bias.copy_(logits)

    @staticmethod
    def _weighted_sum(features: list[torch.Tensor], weights: torch.Tensor) -> torch.Tensor:
        stacked = torch.stack(features, dim=1)
        w = weights.view(weights.shape[0], weights.shape[1], 1, 1, 1)
        return (stacked * w).sum(dim=1)

    def routing_weights(
        self,
        reliability: torch.Tensor,
        available: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the normalized structural and numerical fusion weights.

        This is the exact weighting rule used by ``forward`` and is exposed so
        evaluation diagnostics cannot accidentally visualize the static prior
        instead of the learned, availability-masked routing.
        """
        if reliability.ndim != 3 or reliability.shape[-1] != 2:
            raise ValueError("reliability must have shape (B, M, 2).")
        if available.shape != reliability.shape[:2]:
            raise ValueError("available must have shape (B, M) matching reliability.")
        available = available.to(device=reliability.device, dtype=reliability.dtype)
        if self.fusion_mode == "uniform":
            structural_w = available
            numerical_w = available
        else:
            structural_w = reliability[..., 0] * available
            numerical_w = reliability[..., 1] * available
        structural_w = structural_w / structural_w.sum(dim=1, keepdim=True).clamp_min(1e-6)
        numerical_w = numerical_w / numerical_w.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return structural_w, numerical_w

    def forward(self, batch: BGSampleBatch | BGInferenceBatch, return_all_heads: bool = False) -> DecoupledFeatures:
        # Paper Eqs. (1)-(3): observed modalities form distinct C_bg/C_str interfaces.
        inputs = batch.as_model_inputs()
        device = next(self.parameters()).device
        mask = batch.modality_mask.to(device)[:, self.modality_indices]
        quality = batch.modality_quality.to(device)[:, self.modality_indices]
        target_hw = batch.output_hw
        well_mask = batch.well_mask.to(device)

        numerical_maps: list[torch.Tensor] = []
        structural_maps: list[torch.Tensor] = []
        unique_maps: list[torch.Tensor] = []
        mixed_maps: list[torch.Tensor] = []
        reliability_logits: list[torch.Tensor] = []
        all_heads: dict[str, dict[str, torch.Tensor]] | None = {} if return_all_heads else None

        for modality in self.modalities:
            x = inputs[modality].to(device)
            if modality == "well_log":
                base = self.backbones[modality](x, valid_mask=well_mask)
            else:
                base = self.backbones[modality](x)
            numerical, structural, unique = self.heads[modality](base)
            mixed = self.base_heads[modality](base)
            if tuple(numerical.shape[-2:]) != target_hw:
                numerical = F.interpolate(numerical, size=target_hw, mode="bilinear", align_corners=False)
                structural = F.interpolate(structural, size=target_hw, mode="bilinear", align_corners=False)
                unique = F.interpolate(unique, size=target_hw, mode="bilinear", align_corners=False)
                mixed = F.interpolate(mixed, size=target_hw, mode="bilinear", align_corners=False)
            if all_heads is not None:
                all_heads[modality] = {
                    "numerical": numerical,
                    "structural": structural,
                    "unique": unique,
                }
            numerical_maps.append(numerical)
            structural_maps.append(structural)
            unique_maps.append(unique)
            mixed_maps.append(mixed)
            reliability_logits.append(self.reliability_heads[modality](base))

        reliability = torch.stack(reliability_logits, dim=1).sigmoid()
        available = mask * quality
        structural_w, numerical_w = self.routing_weights(reliability, available)
        unique_w = (mask * quality) / (mask * quality).sum(dim=1, keepdim=True).clamp_min(1e-6)
        mixed_w = unique_w

        return DecoupledFeatures(
            numerical=self._weighted_sum(numerical_maps, numerical_w),
            structural=self._weighted_sum(structural_maps, structural_w),
            unique=self._weighted_sum(unique_maps, unique_w),
            reliability=reliability,
            all_heads=all_heads,
            mixed=self._weighted_sum(mixed_maps, mixed_w),
        )
