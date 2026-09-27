"""Low/high-pass filtering utilities for background and residual objectives."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LowHighPassFilter(nn.Module):
    """Shared low/high-pass projector used by anchors and diagnostics."""

    def __init__(self, kernel_size: int = 5) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd.")
        self.kernel_size = int(kernel_size)

    def lowpass(self, x: torch.Tensor) -> torch.Tensor:
        pad = self.kernel_size // 2
        return F.avg_pool2d(x, kernel_size=self.kernel_size, stride=1, padding=pad)

    def highpass(self, x: torch.Tensor) -> torch.Tensor:
        return x - self.lowpass(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        low = self.lowpass(x)
        return low, x - low
