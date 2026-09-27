"""Adapted Auto-Linear baseline for the AAAI27 multimodal benchmark.

This is a same-input adaptation of the Auto-Linear idea for the local
PSTM/horizon/RMS/well-log protocol. It uses two masked autoencoders, freezes
them for the inverse stage, and trains a low-rank latent converter.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from bg_pdr_fm.data.batch import BGSampleBatch
from bg_pdr_fm.external_baselines.common import build_multimodal_condition_image


def build_auto_linear_measurement_input(
    batch: BGSampleBatch,
    input_hw: tuple[int, int] = (1000, 70),
) -> torch.Tensor:
    """Convert multimodal conditions to the adapted Auto-Linear measurement domain."""

    return build_multimodal_condition_image(batch, input_hw)


class PatchAutoEncoder(nn.Module):
    """Small masked transformer autoencoder for dense 2D fields."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        input_hw: tuple[int, int],
        patch_size: tuple[int, int],
        embed_dim: int,
        depth: int,
        num_heads: int,
        latent_tokens: int,
    ) -> None:
        super().__init__()
        self.input_hw = tuple(int(v) for v in input_hw)
        self.patch_size = tuple(int(v) for v in patch_size)
        if self.input_hw[0] % self.patch_size[0] != 0 or self.input_hw[1] % self.patch_size[1] != 0:
            raise ValueError(f"input_hw {self.input_hw} must be divisible by patch_size {self.patch_size}")
        self.grid_hw = (self.input_hw[0] // self.patch_size[0], self.input_hw[1] // self.patch_size[1])
        self.num_patches = self.grid_hw[0] * self.grid_hw[1]
        self.latent_tokens = int(latent_tokens)
        self.embed_dim = int(embed_dim)
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=self.patch_size, stride=self.patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=int(num_heads),
            dim_feedforward=embed_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(depth))
        self.to_latent = nn.Linear(self.num_patches, self.latent_tokens)
        self.from_latent = nn.Linear(self.latent_tokens, self.num_patches)
        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(embed_dim, out_channels, kernel_size=self.patch_size, stride=self.patch_size),
            nn.Tanh(),
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def _tokenize(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != self.input_hw:
            x = F.interpolate(x, size=self.input_hw, mode="bilinear", align_corners=False)
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)
        return tokens + self.pos_embed

    def _apply_random_mask(self, tokens: torch.Tensor, mask_ratio: float) -> torch.Tensor:
        if mask_ratio <= 0:
            return tokens
        keep = torch.rand(tokens.shape[:2], device=tokens.device) >= float(mask_ratio)
        mask_token = self.mask_token.expand(tokens.shape[0], tokens.shape[1], -1)
        return torch.where(keep.unsqueeze(-1), tokens, mask_token)

    def encode(self, x: torch.Tensor, mask_ratio: float = 0.0) -> torch.Tensor:
        tokens = self._apply_random_mask(self._tokenize(x), mask_ratio)
        encoded = self.encoder(tokens)
        return self.to_latent(encoded.transpose(1, 2)).transpose(1, 2).contiguous()

    def decode(self, latent: torch.Tensor, out_hw: tuple[int, int]) -> torch.Tensor:
        tokens = self.from_latent(latent.transpose(1, 2)).transpose(1, 2)
        feat = tokens.transpose(1, 2).reshape(latent.shape[0], self.embed_dim, *self.grid_hw)
        recon = self.decoder(feat)
        if recon.shape[-2:] != tuple(out_hw):
            recon = F.interpolate(recon, size=tuple(out_hw), mode="bilinear", align_corners=False)
        return recon

    def reconstruct(self, x: torch.Tensor, mask_ratio: float = 0.0, out_hw: tuple[int, int] | None = None) -> torch.Tensor:
        out_hw = tuple(out_hw or x.shape[-2:])
        return self.decode(self.encode(x, mask_ratio=mask_ratio), out_hw=out_hw)


class LowRankLatentConverter(nn.Module):
    """Two-layer low-rank converter between Auto-Linear latent manifolds."""

    def __init__(self, embed_dim: int, rank: int) -> None:
        super().__init__()
        self.down = nn.Linear(int(embed_dim), int(rank), bias=False)
        self.act = nn.GELU()
        self.up = nn.Linear(int(rank), int(embed_dim), bias=False)
        self.norm = nn.LayerNorm(int(embed_dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.norm(z + self.up(self.act(self.down(z))))


class AdaptedAutoLinearMultimodal(nn.Module):
    """Two-domain masked AE plus frozen latent converter for multimodal inversion."""

    def __init__(
        self,
        embed_dim: int = 256,
        depth: int = 4,
        num_heads: int = 8,
        latent_tokens: int = 16,
        converter_rank: int = 64,
        input_hw: tuple[int, int] = (1000, 70),
        measurement_patch_size: tuple[int, int] = (20, 10),
        velocity_patch_size: tuple[int, int] = (10, 10),
        mask_ratio: float = 0.5,
        base_channels: int | None = None,
        latent_dim: int | None = None,
    ) -> None:
        super().__init__()
        if latent_dim is not None:
            embed_dim = int(latent_dim)
        if base_channels is not None:
            embed_dim = max(int(embed_dim), int(base_channels) * 16)
        self.input_hw = tuple(int(v) for v in input_hw)
        self.mask_ratio = float(mask_ratio)
        self.measurement_mae = PatchAutoEncoder(
            in_channels=5,
            out_channels=5,
            input_hw=self.input_hw,
            patch_size=tuple(measurement_patch_size),
            embed_dim=int(embed_dim),
            depth=int(depth),
            num_heads=int(num_heads),
            latent_tokens=int(latent_tokens),
        )
        self.velocity_mae = PatchAutoEncoder(
            in_channels=1,
            out_channels=1,
            input_hw=(70, 70),
            patch_size=tuple(velocity_patch_size),
            embed_dim=int(embed_dim),
            depth=int(depth),
            num_heads=int(num_heads),
            latent_tokens=int(latent_tokens),
        )
        self.latent_converter = LowRankLatentConverter(embed_dim=int(embed_dim), rank=int(converter_rank))

    def measurement_input(self, batch: BGSampleBatch) -> torch.Tensor:
        return build_auto_linear_measurement_input(batch, self.input_hw)

    def encode_measurement(self, batch: BGSampleBatch) -> torch.Tensor:
        return self.measurement_mae.encode(self.measurement_input(batch), mask_ratio=0.0)

    def encode_velocity(self, batch: BGSampleBatch) -> torch.Tensor:
        return self.velocity_mae.encode(batch.depth_vel, mask_ratio=0.0)

    def reconstruct_measurement(self, batch: BGSampleBatch) -> torch.Tensor:
        measurement = self.measurement_input(batch)
        return self.measurement_mae.reconstruct(measurement, mask_ratio=self.mask_ratio, out_hw=measurement.shape[-2:])

    def reconstruct_velocity(self, batch: BGSampleBatch) -> torch.Tensor:
        return self.velocity_mae.reconstruct(batch.depth_vel, mask_ratio=self.mask_ratio, out_hw=batch.depth_vel.shape[-2:])

    def ae_pretrain_loss(self, batch: BGSampleBatch) -> dict[str, torch.Tensor]:
        measurement = self.measurement_input(batch)
        measurement_hat = self.measurement_mae.reconstruct(measurement, mask_ratio=self.mask_ratio, out_hw=measurement.shape[-2:])
        velocity_hat = self.velocity_mae.reconstruct(
            batch.depth_vel,
            mask_ratio=self.mask_ratio,
            out_hw=batch.depth_vel.shape[-2:],
        )
        measurement_l1 = F.l1_loss(measurement_hat, measurement)
        velocity_l1 = F.l1_loss(velocity_hat, batch.depth_vel)
        return {
            "loss": measurement_l1 + velocity_l1,
            "measurement_l1": measurement_l1,
            "velocity_l1": velocity_l1,
            "velocity_hat": velocity_hat,
        }

    def inverse_linear_loss(self, batch: BGSampleBatch) -> dict[str, torch.Tensor]:
        z_c = self.measurement_mae.encode(self.measurement_input(batch), mask_ratio=0.0)
        with torch.no_grad():
            z_v = self.velocity_mae.encode(batch.depth_vel, mask_ratio=0.0)
        z_v_hat = self.latent_converter(z_c)
        velocity_hat = self.velocity_mae.decode(z_v_hat, out_hw=batch.depth_vel.shape[-2:])
        latent_mse = F.mse_loss(z_v_hat, z_v)
        velocity_l1 = F.l1_loss(velocity_hat, batch.depth_vel)
        return {
            "loss": latent_mse + velocity_l1,
            "latent_mse": latent_mse,
            "velocity_l1": velocity_l1,
            "velocity_hat": velocity_hat,
        }

    def forward(self, batch: BGSampleBatch) -> torch.Tensor:
        z_c = self.measurement_mae.encode(self.measurement_input(batch), mask_ratio=0.0)
        z_v_hat = self.latent_converter(z_c)
        return self.velocity_mae.decode(z_v_hat, out_hw=batch.depth_vel.shape[-2:])
