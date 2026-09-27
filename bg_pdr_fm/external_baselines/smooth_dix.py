"""Traditional smooth RMS/Dix-style baseline for AAAI27 evaluation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from bg_pdr_fm.data.batch import BGSampleBatch


class SmoothDixBaseline(nn.Module):
    """Parameter-free RMS-smoothing proxy for a Dix-style velocity baseline."""

    def __init__(self, kernel_size: int = 9) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        self.kernel_size = int(kernel_size)

    def forward(self, batch: BGSampleBatch) -> torch.Tensor:
        x = F.interpolate(batch.rms_vel, size=batch.depth_vel.shape[-2:], mode="bilinear", align_corners=False)
        pad = self.kernel_size // 2
        x = F.avg_pool2d(F.pad(x, (pad, pad, pad, pad), mode="reflect"), self.kernel_size, stride=1)
        return torch.clamp(x, -1.0, 1.0)
