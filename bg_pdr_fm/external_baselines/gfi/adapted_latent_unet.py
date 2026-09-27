"""GFI Latent-UNet architecture adapted to the multimodal benchmark.

This is a same-input adaptation of the GFI ICLR'25 ``UNetInverseModel``:
measurement-domain encoder, latent U-Net translator, and velocity-domain
decoder. The official native GFI consumes waveform amplitudes; here the
measurement tensor is the shared five-channel PSTM/RMS/horizon/well adapter.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from bg_pdr_fm.data.batch import BGSampleBatch
from bg_pdr_fm.external_baselines.common import build_multimodal_condition_image


class _GFIDoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, out_channels, 1e-3),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, out_channels, 1e-3),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class _GFIDown(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.maxpool_conv = nn.Sequential(_GFIDoubleConv(in_channels, out_channels), nn.MaxPool2d(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class _GFIUp(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = _GFIDoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, *, skip: bool = True) -> torch.Tensor:
        x1 = self.up(x1)
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1) if skip else x1
        return self.conv(x)


class _GFIRepeatDownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, repeat: int = 1) -> None:
        super().__init__()
        blocks: list[nn.Module] = [_GFIDown(in_channels, out_channels)]
        blocks.extend(_GFIDoubleConv(out_channels, out_channels) for _ in range(max(int(repeat) - 1, 0)))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class _GFIRepeatBlock(nn.Module):
    def __init__(self, channels: int, repeat: int = 1) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(_GFIDoubleConv(channels, channels) for _ in range(int(repeat)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class _GFIRepeatUpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, repeat: int = 1) -> None:
        super().__init__()
        blocks: list[nn.Module] = [_GFIUp(in_channels, out_channels)]
        blocks.extend(_GFIDoubleConv(out_channels, out_channels) for _ in range(max(int(repeat) - 1, 0)))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, *, skip: bool = True) -> torch.Tensor:
        x = self.blocks[0](x1, x2, skip=skip)
        for block in self.blocks[1:]:
            x = block(x)
        return x


class GFILatentUNet(nn.Module):
    """Faithful local copy of the official GFI latent U-Net topology."""

    def __init__(self, in_channels: int = 128, depth: int = 2, repeat: int = 2, skip: bool = True) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.depth = int(depth)
        self.skip = bool(skip)
        self.encoder = nn.ModuleList()
        for level in range(self.depth):
            in_dim = (2**level) * self.in_channels
            self.encoder.append(_GFIRepeatDownBlock(in_dim, 2 * in_dim, repeat=repeat))
        bottleneck_dim = (2**self.depth) * self.in_channels
        self.bottleneck = _GFIRepeatBlock(bottleneck_dim, repeat=2 * int(repeat))
        self.decoder = nn.ModuleList()
        for level in range(self.depth - 1, -1, -1):
            in_dim = (2 ** (level + 1)) * self.in_channels
            in_ch = 2 * in_dim if self.skip else in_dim
            self.decoder.append(_GFIRepeatUpBlock(in_ch, in_dim // 2, repeat=repeat))
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.decoder.append(_GFIDoubleConv(self.in_channels, self.in_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc_outputs = []
        for layer in self.encoder:
            x = layer(x)
            enc_outputs.append(x)
        x = self.bottleneck(x)
        for idx, layer in enumerate(self.decoder[:-1]):
            x = layer(x, enc_outputs[-(idx + 1)], skip=self.skip)
        x = self.upsample(x)
        return self.decoder[-1](x)


class _MeasurementEncoder(nn.Module):
    def __init__(self, latent_channels: int, latent_hw: tuple[int, int]) -> None:
        super().__init__()
        self.latent_hw = tuple(int(v) for v in latent_hw)
        self.layers = nn.Sequential(
            nn.Conv2d(5, 8, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0)),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(8, 16, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0)),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(16, 32, kernel_size=(5, 1), stride=(2, 1), padding=(2, 0)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(32, 64, kernel_size=(5, 1), stride=(2, 1), padding=(2, 0)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(64, latent_channels, kernel_size=(5, 1), stride=(1, 1), padding=(2, 0)),
            nn.BatchNorm2d(latent_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        return F.interpolate(x, size=self.latent_hw, mode="bilinear", align_corners=False)


class _VelocityDecoder(nn.Module):
    def __init__(self, latent_channels: int, latent_hw: tuple[int, int]) -> None:
        super().__init__()
        self.latent_hw = tuple(int(v) for v in latent_hw)
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.Tanh(),
            nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.Tanh(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.Tanh(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.Tanh(),
            nn.ConvTranspose2d(32, 16, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.ConvTranspose2d(16, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.Tanh(),
            nn.Conv2d(1, 1, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x: torch.Tensor, out_hw: tuple[int, int]) -> torch.Tensor:
        if x.shape[-2:] != self.latent_hw:
            x = F.interpolate(x, size=self.latent_hw, mode="bilinear", align_corners=False)
        x = self.layers(x)
        return F.interpolate(x, size=tuple(int(v) for v in out_hw), mode="bilinear", align_corners=False)


class AdaptedGFILatentUNetMultimodal(nn.Module):
    """GFI UNetInverseModel-style multimodal adapter."""

    def __init__(
        self,
        input_hw: tuple[int, int] = (1000, 70),
        latent_hw: tuple[int, int] = (70, 70),
        latent_channels: int = 128,
        unet_depth: int = 2,
        unet_repeat_blocks: int = 2,
        skip: bool = True,
    ) -> None:
        super().__init__()
        self.input_hw = tuple(int(v) for v in input_hw)
        self.latent_hw = tuple(int(v) for v in latent_hw)
        self.measurement_encoder = _MeasurementEncoder(int(latent_channels), self.latent_hw)
        self.unet_model = GFILatentUNet(
            in_channels=int(latent_channels),
            depth=int(unet_depth),
            repeat=int(unet_repeat_blocks),
            skip=bool(skip),
        )
        self.velocity_decoder = _VelocityDecoder(int(latent_channels), self.latent_hw)

    def forward(self, batch: BGSampleBatch) -> torch.Tensor:
        x = build_multimodal_condition_image(batch, self.input_hw)
        z = self.measurement_encoder(x)
        z = self.unet_model(z)
        return self.velocity_decoder(z, batch.depth_vel.shape[-2:])
