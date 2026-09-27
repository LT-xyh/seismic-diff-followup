"""Source-derived OpenFWI networks used by the adapted AAAI27 baselines.

Architecture source: LANL OpenFWI commit
48754806b7b4c5877259c6b958a87f4513fcdc0b, ``network.py`` and ``utils.py``.

Copyright 2022 Triad National Security, LLC. All rights reserved. This source
was produced under U.S. Government contract 89233218CNA000001 for Los Alamos
National Laboratory. The U.S. Government is granted a nonexclusive, paid-up,
irrevocable worldwide license to reproduce, prepare derivative works,
distribute, perform, and display the work, and to permit others to do so.
"""

from __future__ import annotations

from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        padding=1,
        norm: str = "bn",
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        ]
        if norm == "bn":
            layers.append(nn.BatchNorm2d(out_channels))
        elif norm == "in":
            layers.append(nn.InstanceNorm2d(out_channels))
        elif norm == "ln":
            layers.append(nn.LayerNorm(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ConvBlockTanh(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size=3, stride=1, padding=1) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class DeconvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=2,
        stride=2,
        padding=0,
        output_padding=0,
    ) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                output_padding=output_padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class ResizeBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, scale_factor=2, mode: str = "nearest") -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode=mode),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class OpenFWIInversionNet(nn.Module):
    """Exact OpenFWI InversionNet topology for a 1000x70 input."""

    def __init__(
        self,
        dim1: int = 32,
        dim2: int = 64,
        dim3: int = 128,
        dim4: int = 256,
        dim5: int = 512,
        sample_spatial: float = 1.0,
    ) -> None:
        super().__init__()
        self.convblock1 = ConvBlock(5, dim1, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0))
        self.convblock2_1 = ConvBlock(dim1, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock2_2 = ConvBlock(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock3_1 = ConvBlock(dim2, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock3_2 = ConvBlock(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock4_1 = ConvBlock(dim2, dim3, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock4_2 = ConvBlock(dim3, dim3, kernel_size=(3, 1), padding=(1, 0))
        self.convblock5_1 = ConvBlock(dim3, dim3, stride=2)
        self.convblock5_2 = ConvBlock(dim3, dim3)
        self.convblock6_1 = ConvBlock(dim3, dim4, stride=2)
        self.convblock6_2 = ConvBlock(dim4, dim4)
        self.convblock7_1 = ConvBlock(dim4, dim4, stride=2)
        self.convblock7_2 = ConvBlock(dim4, dim4)
        self.convblock8 = ConvBlock(dim4, dim5, kernel_size=(8, ceil(70 * sample_spatial / 8)), padding=0)

        self.deconv1_1 = DeconvBlock(dim5, dim5, kernel_size=5)
        self.deconv1_2 = ConvBlock(dim5, dim5)
        self.deconv2_1 = DeconvBlock(dim5, dim4, kernel_size=4, stride=2, padding=1)
        self.deconv2_2 = ConvBlock(dim4, dim4)
        self.deconv3_1 = DeconvBlock(dim4, dim3, kernel_size=4, stride=2, padding=1)
        self.deconv3_2 = ConvBlock(dim3, dim3)
        self.deconv4_1 = DeconvBlock(dim3, dim2, kernel_size=4, stride=2, padding=1)
        self.deconv4_2 = ConvBlock(dim2, dim2)
        self.deconv5_1 = DeconvBlock(dim2, dim1, kernel_size=4, stride=2, padding=1)
        self.deconv5_2 = ConvBlock(dim1, dim1)
        self.deconv6 = ConvBlockTanh(dim1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        x = F.pad(x, [-5, -5, -5, -5], mode="constant", value=0)
        return self.deconv6(x)


class OpenFWIUPFWI(nn.Module):
    """Exact OpenFWI FCN4_Deep_Resize_2 topology used for UPFWI."""

    def __init__(
        self,
        dim1: int = 32,
        dim2: int = 64,
        dim3: int = 128,
        dim4: int = 256,
        dim5: int = 512,
        ratio: float = 1.0,
        upsample_mode: str = "nearest",
    ) -> None:
        super().__init__()
        self.convblock1 = ConvBlock(5, dim1, kernel_size=(7, 1), stride=(2, 1), padding=(3, 0))
        self.convblock2_1 = ConvBlock(dim1, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock2_2 = ConvBlock(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock3_1 = ConvBlock(dim2, dim2, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock3_2 = ConvBlock(dim2, dim2, kernel_size=(3, 1), padding=(1, 0))
        self.convblock4_1 = ConvBlock(dim2, dim3, kernel_size=(3, 1), stride=(2, 1), padding=(1, 0))
        self.convblock4_2 = ConvBlock(dim3, dim3, kernel_size=(3, 1), padding=(1, 0))
        self.convblock5_1 = ConvBlock(dim3, dim3, stride=2)
        self.convblock5_2 = ConvBlock(dim3, dim3)
        self.convblock6_1 = ConvBlock(dim3, dim4, stride=2)
        self.convblock6_2 = ConvBlock(dim4, dim4)
        self.convblock7_1 = ConvBlock(dim4, dim4, stride=2)
        self.convblock7_2 = ConvBlock(dim4, dim4)
        self.convblock8 = ConvBlock(dim4, dim5, kernel_size=(8, ceil(70 * ratio / 8)), padding=0)
        self.deconv1_1 = ResizeBlock(dim5, dim5, scale_factor=5, mode=upsample_mode)
        self.deconv1_2 = ConvBlock(dim5, dim5)
        self.deconv2_1 = ResizeBlock(dim5, dim4, scale_factor=2, mode=upsample_mode)
        self.deconv2_2 = ConvBlock(dim4, dim4)
        self.deconv3_1 = ResizeBlock(dim4, dim3, scale_factor=2, mode=upsample_mode)
        self.deconv3_2 = ConvBlock(dim3, dim3)
        self.deconv4_1 = ResizeBlock(dim3, dim2, scale_factor=2, mode=upsample_mode)
        self.deconv4_2 = ConvBlock(dim2, dim2)
        self.deconv5_1 = ResizeBlock(dim2, dim1, scale_factor=2, mode=upsample_mode)
        self.deconv5_2 = ConvBlock(dim1, dim1)
        self.deconv6 = ConvBlockTanh(dim1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        x = F.pad(x, [-5, -5, -5, -5], mode="constant", value=0)
        return self.deconv6(x)


class OpenFWIDiscriminator(nn.Module):
    def __init__(self, dim1: int = 32, dim2: int = 64, dim3: int = 128, dim4: int = 256) -> None:
        super().__init__()
        self.convblock1_1 = ConvBlock(1, dim1, stride=2)
        self.convblock1_2 = ConvBlock(dim1, dim1)
        self.convblock2_1 = ConvBlock(dim1, dim2, stride=2)
        self.convblock2_2 = ConvBlock(dim2, dim2)
        self.convblock3_1 = ConvBlock(dim2, dim3, stride=2)
        self.convblock3_2 = ConvBlock(dim3, dim3)
        self.convblock4_1 = ConvBlock(dim3, dim4, stride=2)
        self.convblock4_2 = ConvBlock(dim4, dim4)
        self.convblock5 = ConvBlock(dim4, 1, kernel_size=5, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convblock1_1(x)
        x = self.convblock1_2(x)
        x = self.convblock2_1(x)
        x = self.convblock2_2(x)
        x = self.convblock3_1(x)
        x = self.convblock3_2(x)
        x = self.convblock4_1(x)
        x = self.convblock4_2(x)
        x = self.convblock5(x)
        return x.view(x.shape[0], -1)


class WassersteinGradientPenalty(nn.Module):
    def __init__(self, weight: float = 10.0) -> None:
        super().__init__()
        self.weight = float(weight)

    def forward(
        self,
        real: torch.Tensor,
        fake: torch.Tensor,
        discriminator: nn.Module,
    ) -> dict[str, torch.Tensor]:
        penalty = self.compute_gradient_penalty(discriminator, real, fake)
        real_score = discriminator(real).mean()
        fake_score = discriminator(fake).mean()
        critic_gap = real_score - fake_score
        loss = -real_score + fake_score + self.weight * penalty
        return {"loss": loss, "critic_gap": critic_gap, "gradient_penalty": penalty}

    @staticmethod
    def compute_gradient_penalty(
        discriminator: nn.Module,
        real: torch.Tensor,
        fake: torch.Tensor,
    ) -> torch.Tensor:
        alpha = torch.rand(real.shape[0], 1, 1, 1, device=real.device, dtype=real.dtype)
        interpolated = (alpha * real + (1.0 - alpha) * fake).requires_grad_(True)
        predictions = discriminator(interpolated)
        gradients = autograd.grad(
            outputs=predictions,
            inputs=interpolated,
            grad_outputs=torch.ones_like(predictions),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        gradients = gradients.reshape(gradients.shape[0], -1)
        return ((gradients.norm(2, dim=1) - 1.0) ** 2).mean()
