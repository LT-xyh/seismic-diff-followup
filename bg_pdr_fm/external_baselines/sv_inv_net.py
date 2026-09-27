"""Adapted SVInvNet-style baseline for multimodal velocity prediction."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from bg_pdr_fm.data.batch import BGSampleBatch
from bg_pdr_fm.external_baselines.common import build_multimodal_condition_image


class DenseLayer(nn.Module):
    def __init__(self, in_channels: int, growth_rate: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, self.net(x)], dim=1)


class DenseBlock(nn.Module):
    def __init__(self, in_channels: int, growth_rate: int, layers: int) -> None:
        super().__init__()
        modules = []
        channels = int(in_channels)
        for _ in range(int(layers)):
            modules.append(DenseLayer(channels, int(growth_rate)))
            channels += int(growth_rate)
        self.net = nn.Sequential(*modules)
        self.out_channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, growth_rate: int, layers: int) -> None:
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.dense = DenseBlock(out_channels, growth_rate, layers)
        self.project = nn.Sequential(
            nn.Conv2d(self.dense.out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(self.dense(self.down(x)))


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class SVInvNetMultimodal(nn.Module):
    """Dense-block encoder-decoder adapted from SVInvNet to five-channel inputs."""

    def __init__(
        self,
        base_channels: int = 32,
        growth_rate: int = 16,
        dense_layers: int = 4,
        input_hw: tuple[int, int] = (1000, 70),
    ) -> None:
        super().__init__()
        c = int(base_channels)
        if c <= 0:
            raise ValueError(f"base_channels must be positive, got {base_channels}")
        self.input_hw = tuple(int(v) for v in input_hw)
        self.stem = nn.Sequential(
            nn.Conv2d(5, c, kernel_size=7, stride=(4, 1), padding=3, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        self.down1 = DownBlock(c, c * 2, growth_rate, dense_layers)
        self.down2 = DownBlock(c * 2, c * 4, growth_rate, dense_layers)
        self.down3 = DownBlock(c * 4, c * 8, growth_rate, dense_layers)
        self.down4 = DownBlock(c * 8, c * 8, growth_rate, dense_layers)
        self.bottleneck = DenseBlock(c * 8, growth_rate, dense_layers)
        self.bottleneck_project = nn.Sequential(
            nn.Conv2d(self.bottleneck.out_channels, c * 8, kernel_size=1, bias=False),
            nn.BatchNorm2d(c * 8),
            nn.ReLU(inplace=True),
        )
        self.up4 = UpBlock(c * 8, c * 8, c * 4)
        self.up3 = UpBlock(c * 4, c * 4, c * 2)
        self.up2 = UpBlock(c * 2, c * 2, c)
        self.up1 = UpBlock(c, c, c)
        self.head = nn.Sequential(nn.Conv2d(c, 1, kernel_size=3, padding=1), nn.Tanh())

    def forward(self, batch: BGSampleBatch) -> torch.Tensor:
        x = build_multimodal_condition_image(batch, self.input_hw)
        s0 = self.stem(x)
        s1 = self.down1(s0)
        s2 = self.down2(s1)
        s3 = self.down3(s2)
        x = self.down4(s3)
        x = self.bottleneck_project(self.bottleneck(x))
        x = self.up4(x, s3)
        x = self.up3(x, s2)
        x = self.up2(x, s1)
        x = self.up1(x, s0)
        x = F.interpolate(x, size=batch.depth_vel.shape[-2:], mode="bilinear", align_corners=False)
        return self.head(x)
