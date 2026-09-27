from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


Size2D = Tuple[int, int]
IntOrPair = Union[int, Sequence[int]]


def _to_2tuple(value: IntOrPair) -> Size2D:
    if isinstance(value, int):
        return (value, value)
    if len(value) != 2:
        raise ValueError(f"Expected a 2-element size, got {value}.")
    return int(value[0]), int(value[1])


def _same_padding(kernel_size: IntOrPair, dilation: IntOrPair = 1) -> Size2D:
    k_h, k_w = _to_2tuple(kernel_size)
    d_h, d_w = _to_2tuple(dilation)
    return ((k_h - 1) // 2) * d_h, ((k_w - 1) // 2) * d_w


def normalize_velocity_map(raw: torch.Tensor, vmin: float = 1500.0, vmax: float = 4500.0) -> torch.Tensor:
    """Apply the fixed min-max normalization used by the dataset."""

    if vmax <= vmin:
        raise ValueError(f"Expected vmax > vmin, got vmin={vmin}, vmax={vmax}.")
    return (raw - vmin) / (vmax - vmin)


def normalized_background_value(vmin: float = 1500.0, vmax: float = 4500.0, raw_background: float = 0.0) -> float:
    """Return the normalized constant produced by a raw background value."""

    return float((raw_background - vmin) / (vmax - vmin))


def build_valid_well_mask(
    x: torch.Tensor,
    *,
    explicit_mask: torch.Tensor | None = None,
    vmin: float = 1500.0,
    vmax: float = 4500.0,
    raw_background: float = 0.0,
    tol: float = 1e-4,
) -> torch.Tensor:
    """
    Infer valid well samples from the known normalized raw-zero background.

    The input ``x`` is already normalized. Background pixels are not zero after
    normalization, so validity must be defined by deviation from the normalized
    background constant instead of by ``x != 0``. Newer datasets should pass an
    explicit mask and keep unobserved well pixels at zero.
    """

    if explicit_mask is not None:
        if explicit_mask.shape != x.shape:
            raise ValueError(f"Expected explicit_mask with shape {tuple(x.shape)}, got {tuple(explicit_mask.shape)}.")
        return (explicit_mask > 0).to(dtype=x.dtype)

    background = normalized_background_value(vmin=vmin, vmax=vmax, raw_background=raw_background)
    background_tensor = x.new_full((1, 1, 1, 1), background)
    return (x - background_tensor).abs().gt(tol).to(dtype=x.dtype)


def add_coord_channels(x: torch.Tensor) -> torch.Tensor:
    """Append normalized y/x coordinates to a 2D feature tensor."""

    b, _, h, w = x.shape
    yy = torch.linspace(-1.0, 1.0, steps=h, device=x.device, dtype=x.dtype).view(1, 1, h, 1).expand(b, 1, h, w)
    xx = torch.linspace(-1.0, 1.0, steps=w, device=x.device, dtype=x.dtype).view(1, 1, 1, w).expand(b, 1, h, w)
    return torch.cat([x, yy, xx], dim=1)


def gaussian1d_kernel(k: int = 7, sigma: float = 1.5, device: torch.device | str = "cpu") -> torch.Tensor:
    """Create a deterministic 1D Gaussian kernel."""

    if k % 2 == 0:
        raise ValueError(f"Expected an odd kernel size, got {k}.")
    ax = torch.arange(k, device=device, dtype=torch.float32) - (k - 1) / 2.0
    kernel = torch.exp(-(ax.square()) / (2.0 * sigma * sigma))
    return kernel / kernel.sum().clamp_min(1e-12)


class FixedGaussianBlurWidth(nn.Module):
    """Spread sparse well support laterally while preserving vertical structure."""

    def __init__(self, k: int = 7, sigma: float = 1.5) -> None:
        super().__init__()
        self.k = k
        self.sigma = sigma
        self.register_buffer("weight", torch.zeros(1, 1, 1, k), persistent=False)
        self._initialized = False

    def _maybe_init(self, device: torch.device, dtype: torch.dtype) -> None:
        if (not self._initialized) or (self.weight.device != device) or (self.weight.dtype != dtype):
            kernel = gaussian1d_kernel(self.k, self.sigma, device=device).to(dtype=dtype)
            self.weight.data = kernel.view(1, 1, 1, self.k)
            self._initialized = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._maybe_init(x.device, x.dtype)
        return F.conv2d(x, self.weight, stride=1, padding=(0, self.k // 2))


class ConvBNAct2d(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k: IntOrPair = 3,
        s: IntOrPair = 1,
        p: Size2D | None = None,
        d: IntOrPair = 1,
        groups: int = 1,
        act: bool = True,
    ) -> None:
        super().__init__()
        if p is None:
            p = _same_padding(k, d)
        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=_to_2tuple(k),
            stride=_to_2tuple(s),
            padding=p,
            dilation=_to_2tuple(d),
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU() if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class ResBlock2d(nn.Module):
    def __init__(self, ch: int, dilation: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        self.conv1 = ConvBNAct2d(ch, ch, k=3, d=dilation)
        self.drop = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = ConvBNAct2d(ch, ch, k=3, d=1, act=False)
        self.out = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.drop(y)
        y = self.conv2(y)
        return self.out(x + y)


class ConvBNAct1d(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int | None = None) -> None:
        super().__init__()
        if p is None:
            p = (k - 1) // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class TraceResBlock1d(nn.Module):
    """Shared 1D residual block for every lateral trace."""

    def __init__(self, ch: int, kernel_size: int = 5, expansion: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = ch * expansion
        padding = (kernel_size - 1) // 2
        self.norm = nn.BatchNorm1d(ch)
        self.depthwise = nn.Conv1d(ch, ch, kernel_size=kernel_size, padding=padding, groups=ch, bias=False)
        self.pointwise1 = nn.Conv1d(ch, hidden, kernel_size=1, bias=False)
        self.pointwise2 = nn.Conv1d(hidden, ch, kernel_size=1, bias=False)
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
    """ConvNeXt-style width-only mixer for trace tokens."""

    def __init__(self, ch: int, kernel_size: int = 5, expansion: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        hidden = ch * expansion
        padding = (kernel_size - 1) // 2
        self.norm = nn.BatchNorm1d(ch)
        self.depthwise = nn.Conv1d(ch, ch, kernel_size=kernel_size, padding=padding, groups=ch, bias=False)
        self.pointwise1 = nn.Conv1d(ch, hidden, kernel_size=1, bias=False)
        self.pointwise2 = nn.Conv1d(hidden, ch, kernel_size=1, bias=False)
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


class TraceEncoder1d(nn.Module):
    """Encode each well trace as a vertical 1D signal with shared weights."""

    def __init__(self, in_ch: int, trace_ch: int, latent_steps: int, num_blocks: int, dropout: float = 0.0) -> None:
        super().__init__()
        blocks: list[nn.Module] = [ConvBNAct1d(in_ch, trace_ch, k=7, s=1)]
        blocks.extend(TraceResBlock1d(trace_ch, kernel_size=5, expansion=2, dropout=dropout) for _ in range(num_blocks))
        self.net = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool1d(latent_steps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.net(x))


class LateralTraceMixer(nn.Module):
    """Mix trace tokens across width while exposing well support priors."""

    def __init__(self, trace_ch: int, num_blocks: int, kernel_size: int = 5, dropout: float = 0.0) -> None:
        super().__init__()
        self.prepare = ConvBNAct1d(trace_ch + 2, trace_ch, k=1, s=1, p=0)
        self.blocks = nn.Sequential(
            *[LateralMixerBlock(trace_ch, kernel_size=kernel_size, expansion=2, dropout=dropout) for _ in range(num_blocks)]
        )
        self.gate = nn.Conv1d(trace_ch + 2, trace_ch, kernel_size=1, bias=True)

    def forward(self, trace_tokens: torch.Tensor, column_mask: torch.Tensor, soft_support: torch.Tensor) -> torch.Tensor:
        cond = torch.cat([column_mask, soft_support], dim=1)
        prepared = self.prepare(torch.cat([trace_tokens, cond], dim=1))
        mixed = self.blocks(prepared)
        gate = torch.sigmoid(self.gate(torch.cat([mixed, cond], dim=1)))
        return mixed * gate


class SparseTraceProjector(nn.Module):
    """Project sparse trace features into a dense 2D fusion-ready feature map."""

    def __init__(
        self,
        trace_ch: int,
        c_mid: int,
        c_out: int,
        num_blocks: int,
        dropout: float = 0.0,
        use_coords: bool = True,
    ) -> None:
        super().__init__()
        self.use_coords = use_coords
        in_ch = trace_ch + 4 + (2 if use_coords else 0)
        self.stem = ConvBNAct2d(in_ch, c_mid, k=3)
        self.blocks = nn.Sequential(
            *[ResBlock2d(c_mid, dilation=1 if idx % 2 == 0 else 2, dropout=dropout) for idx in range(num_blocks)]
        )
        self.fuse = ConvBNAct2d(c_mid + trace_ch, c_out, k=3)

    def forward(
        self,
        trace_map: torch.Tensor,
        sparse_value: torch.Tensor,
        point_mask: torch.Tensor,
        column_support: torch.Tensor,
        soft_support: torch.Tensor,
    ) -> torch.Tensor:
        dense_input = torch.cat([trace_map, sparse_value, point_mask, column_support, soft_support], dim=1)
        if self.use_coords:
            dense_input = add_coord_channels(dense_input)
        h = self.stem(dense_input)
        h = self.blocks(h)
        return self.fuse(torch.cat([h, trace_map], dim=1))


@dataclass
class PreparedWellInput:
    centered_value: torch.Tensor
    point_mask: torch.Tensor
    column_support: torch.Tensor
    soft_support: torch.Tensor
    background_value: float


class WellLogEncoderA(nn.Module):
    """
    Trace-first sparse well encoder.

    The encoder treats the well-log condition as a sparse set of reliable
    vertical traces instead of as a generic sparse image:
    1. Infer a validity mask from the normalized raw-zero background constant.
    2. Encode each lateral trace with a shared 1D trace encoder.
    3. Mix trace tokens along width to model cross-well interactions.
    4. Project sparse trace features and explicit support priors into a dense map.
    5. Resize the dense map to ``out_hw`` for downstream fusion.
    """

    def __init__(
        self,
        C_t: int = 32,
        C2d_1: int = 64,
        C2d_2: int = 96,
        C_out: int = 16,
        k_soft: int = 7,
        sigma_soft: float = 1.5,
        dropout: float = 0.0,
        use_coords: bool = True,
        out_hw: IntOrPair = (70, 70),
        trace_latent_steps: int = 48,
        n_trace_blocks: int = 3,
        n_lateral_blocks: int = 2,
        n_dense_blocks: int = 2,
        lateral_kernel_size: int = 5,
        vmin: float = 1500.0,
        vmax: float = 4500.0,
        raw_background: float = 0.0,
        mask_tol: float = 1e-4,
    ) -> None:
        super().__init__()
        self.trace_channels = C_t
        self.latent_steps = trace_latent_steps
        self.out_hw = _to_2tuple(out_hw)
        self.vmin = float(vmin)
        self.vmax = float(vmax)
        self.raw_background = float(raw_background)
        self.mask_tol = float(mask_tol)

        self.soft_support = FixedGaussianBlurWidth(k=k_soft, sigma=sigma_soft)
        self.trace_encoder = TraceEncoder1d(
            in_ch=2,
            trace_ch=C_t,
            latent_steps=trace_latent_steps,
            num_blocks=n_trace_blocks,
            dropout=dropout,
        )
        self.lateral_mixer = LateralTraceMixer(
            trace_ch=C_t,
            num_blocks=n_lateral_blocks,
            kernel_size=lateral_kernel_size,
            dropout=dropout,
        )
        self.projector = SparseTraceProjector(
            trace_ch=C_t,
            c_mid=C2d_1,
            c_out=C2d_2,
            num_blocks=n_dense_blocks,
            dropout=dropout,
            use_coords=use_coords,
        )
        self.out_proj = nn.Conv2d(C2d_2, C_out, kernel_size=1, bias=True)

    def normalized_background(self) -> float:
        return normalized_background_value(
            vmin=self.vmin,
            vmax=self.vmax,
            raw_background=self.raw_background,
        )

    def infer_valid_mask(self, x: torch.Tensor, explicit_mask: torch.Tensor | None = None) -> torch.Tensor:
        return build_valid_well_mask(
            x,
            explicit_mask=explicit_mask,
            vmin=self.vmin,
            vmax=self.vmax,
            raw_background=self.raw_background,
            tol=self.mask_tol,
        )

    def prepare_input(self, x: torch.Tensor, explicit_mask: torch.Tensor | None = None) -> PreparedWellInput:
        if x.ndim != 4:
            raise ValueError(f"Expected x with shape [B, 1, H, W], got {tuple(x.shape)}.")
        if x.size(1) != 1:
            raise ValueError(f"Expected a single well-log channel, got {x.size(1)}.")

        point_mask = self.infer_valid_mask(x, explicit_mask=explicit_mask)
        background_value = 0.0 if explicit_mask is not None else self.normalized_background()
        background = x.new_full((1, 1, 1, 1), background_value)

        # With an explicit mask, unobserved well pixels are already a zero baseline.
        # Without one, keep the legacy normalized raw-zero baseline behavior.
        centered_value = (x - background) * point_mask

        # Collapse point-wise validity into column-wise trace support along width.
        column_support = point_mask.amax(dim=2, keepdim=True).expand(-1, -1, x.size(2), -1)
        soft_support = self.soft_support(column_support)

        return PreparedWellInput(
            centered_value=centered_value,
            point_mask=point_mask,
            column_support=column_support,
            soft_support=soft_support,
            background_value=background_value,
        )

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
        prepared = self.prepare_input(x, explicit_mask=valid_mask)
        b, _, h, w = x.shape

        # [B, 2, H, W] -> [B*W, 2, H]: shared trace encoder over every lateral position.
        trace_input = torch.cat([prepared.centered_value, prepared.point_mask], dim=1)
        trace_batch = trace_input.permute(0, 3, 1, 2).reshape(b * w, 2, h)
        trace_features = self.trace_encoder(trace_batch)  # [B*W, C_t, Lt]

        # Restore the trace set layout as a latent trace map [B, C_t, Lt, W].
        trace_features = trace_features.view(b, w, self.trace_channels, self.latent_steps)
        trace_map = trace_features.permute(0, 2, 3, 1).contiguous()

        # Summarize each latent trace into a width token and mix across lateral positions.
        trace_tokens = trace_map.mean(dim=2)  # [B, C_t, W]
        column_mask_1d = prepared.column_support[:, :, 0, :]  # [B, 1, W]
        soft_support_1d = prepared.soft_support.mean(dim=2)  # [B, 1, W]
        mixed_tokens = self.lateral_mixer(trace_tokens, column_mask_1d, soft_support_1d)
        trace_map = trace_map + mixed_tokens.unsqueeze(2)

        # Build dense support priors at the latent vertical resolution.
        sparse_value = F.adaptive_avg_pool2d(prepared.centered_value, output_size=(self.latent_steps, w))
        point_mask = F.adaptive_max_pool2d(prepared.point_mask, output_size=(self.latent_steps, w))
        column_support = F.adaptive_max_pool2d(prepared.column_support, output_size=(self.latent_steps, w))
        soft_support = F.interpolate(prepared.soft_support, size=(self.latent_steps, w), mode="bilinear", align_corners=False)

        dense = self.projector(
            trace_map=trace_map,
            sparse_value=sparse_value,
            point_mask=point_mask,
            column_support=column_support,
            soft_support=soft_support,
        )
        dense = F.interpolate(dense, size=self.out_hw, mode="bilinear", align_corners=False)
        return self.out_proj(dense)


def _make_raw_well_map(shape: Tuple[int, int, int, int], columns: Sequence[int]) -> torch.Tensor:
    b, c, h, w = shape
    if c != 1:
        raise ValueError("This helper expects a single-channel raw well map.")

    raw = torch.zeros(shape, dtype=torch.float32)
    depth = torch.linspace(0.0, 1.0, steps=h, dtype=torch.float32)

    for batch_idx in range(b):
        for col_idx, col in enumerate(columns):
            if not (0 <= col < w):
                raise ValueError(f"Column index {col} is out of range for width {w}.")
            top_velocity = 1800.0 + 80.0 * batch_idx + 120.0 * col_idx
            bottom_velocity = 3200.0 + 120.0 * batch_idx + 160.0 * col_idx
            raw[batch_idx, 0, :, col] = top_velocity + (bottom_velocity - top_velocity) * depth

    return raw


def _run_test_case(
    *,
    name: str,
    shape: Tuple[int, int, int, int],
    columns: Sequence[int],
    out_hw: Size2D,
) -> None:
    raw = _make_raw_well_map(shape, columns)
    normalized = normalize_velocity_map(raw, vmin=1500.0, vmax=4500.0)

    encoder = WellLogEncoderA(
        C_t=24,
        C2d_1=48,
        C2d_2=64,
        C_out=16,
        out_hw=out_hw,
        trace_latent_steps=40,
        n_trace_blocks=3,
        n_lateral_blocks=2,
        n_dense_blocks=2,
        mask_tol=1e-5,
    )
    encoder.eval()

    inferred_mask = encoder.infer_valid_mask(normalized)
    expected_mask = raw.ne(0).to(dtype=normalized.dtype)

    assert torch.allclose(inferred_mask, expected_mask), f"{name}: inferred mask does not match raw well support."
    assert encoder.normalized_background() != 0.0, f"{name}: background should not stay zero after normalization."

    background_values = normalized[raw == 0]
    expected_background = torch.full_like(background_values, encoder.normalized_background())
    assert torch.allclose(background_values, expected_background), f"{name}: normalized background constant is incorrect."

    with torch.no_grad():
        output = encoder(normalized)

    expected_shape = (shape[0], 16, out_hw[0], out_hw[1])
    assert tuple(output.shape) == expected_shape, f"{name}: got {tuple(output.shape)}, expected {expected_shape}."

