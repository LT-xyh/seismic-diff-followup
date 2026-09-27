"""Adapted GFI-style multimodal inverse baseline.

This module is a disclosed same-input adaptation of GFI/InversionNet-style
inverse CNNs for the BG-PDR-FM multimodal benchmark. It does not claim to be
the official waveform-to-velocity GFI protocol: it converts the current
PSTM/horizon/RMS/well-log batch schema into a five-channel tensor with the
official GFI spatial contract ``[B, 5, 1000, 70]`` and predicts velocity maps
``[B, 1, 70, 70]``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from bg_pdr_fm.data.batch import BGSampleBatch
from bg_pdr_fm.external_baselines.common import build_multimodal_condition_image


def build_adapted_gfi_input(batch: BGSampleBatch, input_hw: tuple[int, int] = (1000, 70)) -> torch.Tensor:
    """Convert BG-PDR-FM multimodal conditions into a GFI-compatible tensor."""
    return build_multimodal_condition_image(batch, input_hw)


class _ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, kernel_size=3, stride=1, padding=1) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class _DeconvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, kernel_size=4, stride=2, padding=1) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.GroupNorm(num_groups=min(8, out_channels), num_channels=out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class AdaptedGFIMultimodal(nn.Module):
    """InversionNet/GFI-style direct inverse baseline for multimodal inputs."""

    def __init__(self, base_channels: int = 32, input_hw: tuple[int, int] = (1000, 70)) -> None:
        super().__init__()
        c1 = int(base_channels)
        if c1 <= 0:
            raise ValueError(f"base_channels must be positive, got {base_channels}")
        c2, c3, c4, c5 = c1 * 2, c1 * 4, c1 * 8, c1 * 16
        self.input_hw = tuple(int(v) for v in input_hw)

        self.convblock1 = _ConvBlock(5, c1, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0))
        self.convblock2_1 = _ConvBlock(c1, c2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock2_2 = _ConvBlock(c2, c2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock3_1 = _ConvBlock(c2, c2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock3_2 = _ConvBlock(c2, c2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock4_1 = _ConvBlock(c2, c3, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock4_2 = _ConvBlock(c3, c3, kernel_size=(3, 1), padding=(1, 0))
        self.convblock5_1 = _ConvBlock(c3, c3, stride=2)
        self.convblock5_2 = _ConvBlock(c3, c3)
        self.convblock6_1 = _ConvBlock(c3, c4, stride=2)
        self.convblock6_2 = _ConvBlock(c4, c4)
        self.convblock7_1 = _ConvBlock(c4, c4, stride=2)
        self.convblock7_2 = _ConvBlock(c4, c4)
        self.convblock8 = _ConvBlock(c4, c5, kernel_size=(8, 9), padding=0)

        self.deconv1_1 = nn.Sequential(
            nn.ConvTranspose2d(c5, c5, kernel_size=5),
            nn.GroupNorm(num_groups=min(8, c5), num_channels=c5),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.deconv1_2 = _ConvBlock(c5, c5)
        self.deconv2_1 = _DeconvBlock(c5, c4)
        self.deconv2_2 = _ConvBlock(c4, c4)
        self.deconv3_1 = _DeconvBlock(c4, c3)
        self.deconv3_2 = _ConvBlock(c3, c3)
        self.deconv4_1 = _DeconvBlock(c3, c2)
        self.deconv4_2 = _ConvBlock(c2, c2)
        self.deconv5_1 = _DeconvBlock(c2, c1)
        self.deconv5_2 = _ConvBlock(c1, c1)
        self.head = nn.Sequential(nn.Conv2d(c1, 1, kernel_size=3, padding=1), nn.Tanh())

    def forward(self, batch: BGSampleBatch) -> torch.Tensor:
        x = build_adapted_gfi_input(batch, self.input_hw)
        x = self.convblock1(x)
        x = self.convblock2_1(x)
        x = self.convblock2_2(x)
        x = self.convblock3_1(x)
        x = self.convblock3_2(x)
        x = self.convblock4_1(x)
        x = self.convblock4_2(x)
        x = self.convblock5_1(x)
        x = self.convblock5_2(x)
        x = self.convblock6_1(x)
        x = self.convblock6_2(x)
        x = self.convblock7_1(x)
        x = self.convblock7_2(x)
        x = self.convblock8(x)
        x = self.deconv1_1(x)
        x = self.deconv1_2(x)
        x = self.deconv2_1(x)
        x = self.deconv2_2(x)
        x = self.deconv3_1(x)
        x = self.deconv3_2(x)
        x = self.deconv4_1(x)
        x = self.deconv4_2(x)
        x = self.deconv5_1(x)
        x = self.deconv5_2(x)
        x = F.interpolate(x, size=batch.depth_vel.shape[-2:], mode="bilinear", align_corners=False)
        return self.head(x)
