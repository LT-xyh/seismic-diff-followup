"""Original-architecture Auto-Linear adaptation for the AAAI27 benchmark.

This module follows the Auto-Linear ICML 2024 architecture choices for
OpenFWI-sized data while adapting the measurement domain to the local
five-channel PSTM/RMS/horizon/well protocol.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from bg_pdr_fm.data.batch import BGSampleBatch
from bg_pdr_fm.external_baselines.common import build_multimodal_condition_image


def build_auto_linear_original_measurement_input(
    batch: BGSampleBatch,
    input_hw: tuple[int, int] = (1000, 70),
) -> torch.Tensor:
    """Build the five-channel adapted measurement image for Auto-Linear."""

    return build_multimodal_condition_image(batch, input_hw)


class _TransformerStack(nn.Module):
    def __init__(self, dim: int, depth: int, heads: int, mlp_dim: int) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=int(dim),
            nhead=int(heads),
            dim_feedforward=int(mlp_dim),
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.net = nn.TransformerEncoder(layer, num_layers=int(depth))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class OriginalPatchMAE(nn.Module):
    """Asymmetric transformer MAE matching the Auto-Linear paper dimensions."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        input_hw: tuple[int, int],
        patch_size: tuple[int, int],
        encoder_dim: int,
        encoder_depth: int,
        encoder_heads: int,
        decoder_dim: int,
        decoder_depth: int,
        decoder_heads: int,
    ) -> None:
        super().__init__()
        self.input_hw = tuple(int(v) for v in input_hw)
        self.patch_size = tuple(int(v) for v in patch_size)
        if self.input_hw[0] % self.patch_size[0] != 0 or self.input_hw[1] % self.patch_size[1] != 0:
            raise ValueError(f"input_hw {self.input_hw} must be divisible by patch_size {self.patch_size}")
        self.grid_hw = (self.input_hw[0] // self.patch_size[0], self.input_hw[1] // self.patch_size[1])
        self.num_patches = self.grid_hw[0] * self.grid_hw[1]
        self.patch_dim = int(in_channels) * self.patch_size[0] * self.patch_size[1]
        self.out_channels = int(out_channels)
        self.encoder_dim = int(encoder_dim)
        self.encoder_depth = int(encoder_depth)
        self.encoder_heads = int(encoder_heads)
        self.decoder_dim = int(decoder_dim)
        self.decoder_depth = int(decoder_depth)
        self.decoder_heads = int(decoder_heads)
        self.encoder_mlp_dim = self.encoder_dim * 4
        self.decoder_mlp_dim = 144 if self.encoder_dim == 132 else self.encoder_dim * 4
        if self.decoder_dim == 512 and self.encoder_dim == 516:
            self.decoder_mlp_dim = 2064

        self.patch_embed = nn.Conv2d(
            int(in_channels),
            self.encoder_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
        )
        self.encoder_pos = nn.Parameter(torch.zeros(1, self.num_patches, self.encoder_dim))
        self.encoder_mask_token = nn.Parameter(torch.zeros(1, 1, self.encoder_dim))
        self.encoder = _TransformerStack(
            dim=self.encoder_dim,
            depth=self.encoder_depth,
            heads=self.encoder_heads,
            mlp_dim=self.encoder_mlp_dim,
        )

        self.to_decoder = nn.Linear(self.encoder_dim, self.decoder_dim)
        self.decoder_pos = nn.Parameter(torch.zeros(1, self.num_patches, self.decoder_dim))
        self.decoder = _TransformerStack(
            dim=self.decoder_dim,
            depth=self.decoder_depth,
            heads=self.decoder_heads,
            mlp_dim=self.decoder_mlp_dim,
        )
        self.patch_reconstruct = nn.Linear(
            self.decoder_dim,
            self.out_channels * self.patch_size[0] * self.patch_size[1],
        )
        nn.init.trunc_normal_(self.encoder_pos, std=0.02)
        nn.init.trunc_normal_(self.encoder_mask_token, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos, std=0.02)

    def _tokenize(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2:] != self.input_hw:
            x = F.interpolate(x, size=self.input_hw, mode="bilinear", align_corners=False)
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)
        return tokens + self.encoder_pos

    def _apply_random_mask(self, tokens: torch.Tensor, mask_ratio: float) -> torch.Tensor:
        if mask_ratio <= 0:
            return tokens
        keep = torch.rand(tokens.shape[:2], device=tokens.device) >= float(mask_ratio)
        mask_token = self.encoder_mask_token.expand(tokens.shape[0], tokens.shape[1], -1)
        return torch.where(keep.unsqueeze(-1), tokens, mask_token)

    def encode(self, x: torch.Tensor, mask_ratio: float = 0.0) -> torch.Tensor:
        tokens = self._apply_random_mask(self._tokenize(x), float(mask_ratio))
        return self.encoder(tokens)

    def decode(self, latent: torch.Tensor, out_hw: tuple[int, int]) -> torch.Tensor:
        tokens = self.to_decoder(latent) + self.decoder_pos
        decoded = self.decoder(tokens)
        patches = self.patch_reconstruct(decoded)
        batch = patches.shape[0]
        patch_h, patch_w = self.patch_size
        grid_h, grid_w = self.grid_hw
        out = patches.view(batch, grid_h, grid_w, self.out_channels, patch_h, patch_w)
        out = out.permute(0, 3, 1, 4, 2, 5).contiguous()
        out = out.view(batch, self.out_channels, grid_h * patch_h, grid_w * patch_w)
        out = torch.tanh(out)
        if out.shape[-2:] != tuple(out_hw):
            out = F.interpolate(out, size=tuple(out_hw), mode="bilinear", align_corners=False)
        return out

    def reconstruct(self, x: torch.Tensor, *, mask_ratio: float, out_hw: tuple[int, int]) -> torch.Tensor:
        return self.decode(self.encode(x, mask_ratio=mask_ratio), out_hw=out_hw)


class OriginalLowRankLinearConverter(nn.Module):
    """Low-rank linear map from flattened seismic latent to velocity latent."""

    def __init__(
        self,
        *,
        input_shape: tuple[int, int],
        output_shape: tuple[int, int],
        rank: int,
    ) -> None:
        super().__init__()
        self.input_shape = tuple(int(v) for v in input_shape)
        self.output_shape = tuple(int(v) for v in output_shape)
        self.input_dim = self.input_shape[0] * self.input_shape[1]
        self.output_dim = self.output_shape[0] * self.output_shape[1]
        self.rank = int(rank)
        self.down = nn.Linear(self.input_dim, self.rank, bias=False)
        self.up = nn.Linear(self.rank, self.output_dim, bias=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if tuple(z.shape[1:]) != self.input_shape:
            raise ValueError(f"expected latent shape [B, {self.input_shape}], got {tuple(z.shape)}")
        flat = z.flatten(1)
        mapped = self.up(self.down(flat))
        return mapped.view(z.shape[0], *self.output_shape)


class AdaptedAutoLinearOriginalMultimodal(nn.Module):
    """Auto-Linear original architecture adapted to multimodal conditions."""

    def __init__(
        self,
        *,
        input_hw: tuple[int, int] = (1000, 70),
        mask_ratio: float = 0.75,
        converter_rank: int = 128,
    ) -> None:
        super().__init__()
        self.input_hw = tuple(int(v) for v in input_hw)
        self.mask_ratio = float(mask_ratio)
        self.measurement_mae = OriginalPatchMAE(
            in_channels=5,
            out_channels=5,
            input_hw=self.input_hw,
            patch_size=(100, 10),
            encoder_dim=132,
            encoder_depth=2,
            encoder_heads=12,
            decoder_dim=512,
            decoder_depth=2,
            decoder_heads=16,
        )
        self.velocity_mae = OriginalPatchMAE(
            in_channels=1,
            out_channels=1,
            input_hw=(70, 70),
            patch_size=(10, 10),
            encoder_dim=516,
            encoder_depth=3,
            encoder_heads=12,
            decoder_dim=512,
            decoder_depth=2,
            decoder_heads=16,
        )
        self.latent_converter = OriginalLowRankLinearConverter(
            input_shape=(self.measurement_mae.num_patches, self.measurement_mae.encoder_dim),
            output_shape=(self.velocity_mae.num_patches, self.velocity_mae.encoder_dim),
            rank=int(converter_rank),
        )

    def measurement_input(self, batch: BGSampleBatch) -> torch.Tensor:
        return build_auto_linear_original_measurement_input(batch, self.input_hw)

    def encode_measurement(self, batch: BGSampleBatch, *, mask_ratio: float = 0.0) -> torch.Tensor:
        return self.measurement_mae.encode(self.measurement_input(batch), mask_ratio=mask_ratio)

    def encode_velocity(self, batch: BGSampleBatch, *, mask_ratio: float = 0.0) -> torch.Tensor:
        return self.velocity_mae.encode(batch.depth_vel, mask_ratio=mask_ratio)

    def ae_pretrain_loss(self, batch: BGSampleBatch) -> dict[str, torch.Tensor]:
        measurement = self.measurement_input(batch)
        measurement_hat = self.measurement_mae.reconstruct(
            measurement,
            mask_ratio=self.mask_ratio,
            out_hw=measurement.shape[-2:],
        )
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

    def inverse_linear_loss(
        self,
        batch: BGSampleBatch,
        *,
        supervised_loss: str = "latent_mse_l1",
    ) -> dict[str, torch.Tensor]:
        z_c = self.encode_measurement(batch, mask_ratio=0.0)
        with torch.no_grad():
            z_v = self.encode_velocity(batch, mask_ratio=0.0)
        z_v_hat = self.latent_converter(z_c)
        velocity_hat = self.velocity_mae.decode(z_v_hat, out_hw=batch.depth_vel.shape[-2:])
        latent_mse = F.mse_loss(z_v_hat, z_v)
        velocity_l1 = F.l1_loss(velocity_hat, batch.depth_vel)
        supervised_loss = str(supervised_loss).lower()
        if supervised_loss == "l1":
            loss = velocity_l1
        elif supervised_loss == "latent_mse":
            loss = latent_mse
        elif supervised_loss in {"latent_mse_l1", "mse_l1"}:
            loss = latent_mse + velocity_l1
        else:
            raise ValueError(
                "supervised_loss must be one of 'l1', 'latent_mse', or 'latent_mse_l1'; "
                f"got {supervised_loss!r}."
            )
        return {
            "loss": loss,
            "latent_mse": latent_mse,
            "velocity_l1": velocity_l1,
            "velocity_hat": velocity_hat,
        }

    def forward(self, batch: BGSampleBatch) -> torch.Tensor:
        z_c = self.encode_measurement(batch, mask_ratio=0.0)
        z_v_hat = self.latent_converter(z_c)
        return self.velocity_mae.decode(z_v_hat, out_hw=batch.depth_vel.shape[-2:])
