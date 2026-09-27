"""Seismic autoencoder components used by the checkpoint-backed latent codec."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL


class SeismicAutoencoderKL(nn.Module):
    """KL autoencoder for 70x70 velocity patches.

    Inputs are reflect-padded from 70x70 to 72x72 so the diffusers
    AutoencoderKL can downsample cleanly to an efficient 4x18x18 latent. The
    decoded output is center-cropped back to the original 70x70 grid.
    """

    def __init__(
        self,
        latent_channels: int = 4,
        depth_vel_shape: tuple[int, int, int] = (1, 70, 70),
        down_block_types: tuple[str, ...] = ("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types: tuple[str, ...] = ("UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"),
        block_out_channels: tuple[int, ...] = (64, 128, 256),
        sample_size: int = 72,
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        **_: object,
    ) -> None:
        super().__init__()
        self.input_shape = tuple(int(v) for v in depth_vel_shape)
        self.sample_size = int(sample_size)
        self.latent_channels = int(latent_channels)
        if len(self.input_shape) != 3:
            raise ValueError(f"depth_vel_shape must be CHW, got {self.input_shape}.")
        if self.input_shape[1:] != (70, 70):
            raise ValueError("SeismicAutoencoderKL currently expects 70x70 velocity patches.")
        if self.sample_size < 70:
            raise ValueError(f"sample_size must be >= 70, got {self.sample_size}.")

        self.pad = self._symmetric_pad(self.input_shape[1:], (self.sample_size, self.sample_size))
        self.autoencoder_kl = AutoencoderKL(
            sample_size=self.sample_size,
            in_channels=self.input_shape[0],
            out_channels=self.input_shape[0],
            down_block_types=tuple(down_block_types),
            up_block_types=tuple(up_block_types),
            block_out_channels=tuple(int(v) for v in block_out_channels),
            latent_channels=self.latent_channels,
            layers_per_block=int(layers_per_block),
            norm_num_groups=int(norm_num_groups),
            force_upcast=False,
        )

    @staticmethod
    def _symmetric_pad(src_hw: tuple[int, int], dst_hw: tuple[int, int]) -> tuple[int, int, int, int]:
        src_h, src_w = src_hw
        dst_h, dst_w = dst_hw
        pad_h = dst_h - src_h
        pad_w = dst_w - src_w
        if pad_h < 0 or pad_w < 0:
            raise ValueError(f"Cannot pad from {src_hw} to smaller shape {dst_hw}.")
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left
        return left, right, top, bottom

    def _pad_input(self, x: torch.Tensor) -> torch.Tensor:
        if tuple(x.shape[-2:]) != self.input_shape[1:]:
            raise ValueError(f"Expected input spatial shape {self.input_shape[1:]}, got {tuple(x.shape[-2:])}.")
        if self.pad == (0, 0, 0, 0):
            return x
        return F.pad(x, self.pad, mode="reflect")

    def _crop_output(self, x: torch.Tensor) -> torch.Tensor:
        left, right, top, bottom = self.pad
        h_end = x.shape[-2] - bottom if bottom > 0 else x.shape[-2]
        w_end = x.shape[-1] - right if right > 0 else x.shape[-1]
        return x[..., top:h_end, left:w_end]

    def forward(self, x: torch.Tensor):
        posterior = self.encode(x)
        latents = posterior.sample()
        reconstructions = self.decode(latents)
        return reconstructions, posterior

    def encode(self, x: torch.Tensor):
        return self.autoencoder_kl.encode(self._pad_input(x)).latent_dist

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        reconstructions = self.autoencoder_kl.decode(latents, return_dict=True).sample
        return self._crop_output(reconstructions)
