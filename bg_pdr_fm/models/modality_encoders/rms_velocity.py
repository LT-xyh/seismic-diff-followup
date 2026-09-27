from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- Shared Blocks ----------
class ConvBNGELU1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int | None = None) -> None:
        super().__init__()
        if p is None:
            p = (k - 1) // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class ConvBNGELU2d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int | None = None) -> None:
        super().__init__()
        if p is None:
            p = (k - 1) // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class ResBlock2d(nn.Module):
    def __init__(self, ch: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv1 = ConvBNGELU2d(ch, ch, k=3, s=1)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = ConvBNGELU2d(ch, ch, k=3, s=1)
        self.out = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.drop(y)
        y = self.conv2(y)
        return self.out(x + y)


class TraceResBlock1d(nn.Module):
    """Residual 1D block shared by every RMS trace."""

    def __init__(self, ch: int, kernel_size: int = 5, expansion: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        hidden_ch = ch * expansion
        padding = (kernel_size - 1) // 2

        self.norm = nn.BatchNorm1d(ch)
        self.depthwise = nn.Conv1d(ch, ch, kernel_size=kernel_size, padding=padding, groups=ch, bias=False, )
        self.pointwise1 = nn.Conv1d(ch, hidden_ch, kernel_size=1, bias=False)
        self.pointwise2 = nn.Conv1d(hidden_ch, ch, kernel_size=1, bias=False)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.norm(x)
        y = self.depthwise(y)
        y = self.act(y)
        y = self.pointwise1(y)
        y = self.act(y)
        y = self.drop(y)
        y = self.pointwise2(y)
        return residual + y


class LateralMixerBlock(nn.Module):
    """Residual mixer that exchanges information across neighboring traces."""

    def __init__(self, ch: int, kernel_size: int = 5, expansion: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        hidden_ch = ch * expansion
        padding = (kernel_size - 1) // 2

        self.norm = nn.BatchNorm1d(ch)
        self.depthwise = nn.Conv1d(ch, ch, kernel_size=kernel_size, padding=padding, groups=ch, bias=False, )
        self.pointwise1 = nn.Conv1d(ch, hidden_ch, kernel_size=1, bias=False)
        self.pointwise2 = nn.Conv1d(hidden_ch, ch, kernel_size=1, bias=False)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.norm(x)
        y = self.depthwise(y)
        y = self.act(y)
        y = self.pointwise1(y)
        y = self.act(y)
        y = self.drop(y)
        y = self.pointwise2(y)
        return residual + y


class DilatedResBlock2d(nn.Module):
    def __init__(self, ch: int, dilation: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.BatchNorm2d(ch)
        self.conv1 = nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=False, )
        self.norm2 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1, bias=False)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.conv1(self.norm1(x)))
        y = self.drop(y)
        y = self.conv2(self.norm2(y))
        return self.act(x + y)


def add_coord_channels(x: torch.Tensor) -> torch.Tensor:
    """Append normalized y/x coordinates to a feature tensor."""

    b, _, h, w = x.shape
    device = x.device
    dtype = x.dtype
    yy = torch.linspace(-1.0, 1.0, steps=h, device=device, dtype=dtype).view(1, 1, h, 1).expand(b, 1, h, w)
    xx = torch.linspace(-1.0, 1.0, steps=w, device=device, dtype=dtype).view(1, 1, 1, w).expand(b, 1, h, w)
    return torch.cat([x, yy, xx], dim=1)


# ---------- Encoder ----------
class RMSVelocityEncoderA(nn.Module):
    """
    Trace-set RMS encoder.

    The encoder respects RMS velocity structure:
    1. Encode each lateral trace independently with a shared 1D stem.
    2. Compress each trace to a latent temporal length.
    3. Mix information across traces along the lateral axis.
    4. Project the trace set into a unified 2D feature map for fusion.
    """

    def __init__(self, trace_channels: int = 32, latent_time_steps: int = 48, lateral_kernel_size: int = 5,
                 out_hw: Tuple[int, int] = (70, 70), c_mid: int = 96, c_out: int = 64, n_trace_blocks: int = 3,
                 n_lateral_blocks: int = 2, n_refine2d: int = 2, dropout: float = 0.0, ) -> None:
        super().__init__()
        self.trace_channels = trace_channels
        self.latent_time_steps = latent_time_steps
        self.out_hw = out_hw
        self.c_out = c_out

        trace_blocks = [ConvBNGELU1d(1, trace_channels, k=7, s=1)]
        trace_blocks.extend(
            TraceResBlock1d(trace_channels, kernel_size=5, expansion=2, dropout=dropout) for _ in range(n_trace_blocks))
        self.trace_stem = nn.Sequential(*trace_blocks)
        self.temporal_pool = nn.AdaptiveAvgPool1d(latent_time_steps)

        self.lateral_mixer = nn.Sequential(
            *[LateralMixerBlock(trace_channels, kernel_size=lateral_kernel_size, expansion=2, dropout=dropout, ) for _
              in range(n_lateral_blocks)])

        refine_blocks = [ConvBNGELU2d(trace_channels, c_mid, k=3, s=1)]
        refine_blocks.extend(ResBlock2d(c_mid, dropout=dropout) for _ in range(n_refine2d))
        self.projection_head = nn.Sequential(*refine_blocks)
        self.out_proj = nn.Conv2d(c_mid, c_out, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected x with shape [B, 1, T, W], got {tuple(x.shape)}.")

        b, c, t, w = x.shape
        if c != 1:
            raise ValueError(f"Expected a single RMS channel, got {c}.")

        # Step 1: treat every lateral position as an independent 1D trace.
        trace_batch = x.permute(0, 3, 1, 2).reshape(b * w, 1, t)  # [B*W, 1, T]
        trace_features = self.trace_stem(trace_batch)  # [B*W, C_trace, T]

        # Step 2: compress each trace to a fixed latent temporal length Lt.
        trace_features = self.temporal_pool(trace_features)  # [B*W, C_trace, Lt]
        trace_features = trace_features.view(b, w, self.trace_channels, self.latent_time_steps)

        # Step 3: summarize each trace and mix along the lateral axis only.
        trace_tokens = trace_features.mean(dim=-1).permute(0, 2, 1).contiguous()  # [B, C_trace, W]
        trace_tokens = self.lateral_mixer(trace_tokens)  # [B, C_trace, W]
        trace_features = trace_features + trace_tokens.permute(0, 2, 1).unsqueeze(-1)

        # Step 4: convert the trace set into a canonical 2D feature map for fusion.
        feature_map = trace_features.permute(0, 2, 3, 1).contiguous()  # [B, C_trace, Lt, W]
        feature_map = F.interpolate(feature_map, size=self.out_hw, mode="bilinear", align_corners=False)
        feature_map = self.projection_head(feature_map)  # [B, C_mid, H_out, W_out]
        return self.out_proj(feature_map)  # [B, C_out, H_out, W_out]

