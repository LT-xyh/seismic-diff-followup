from typing import Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


Size2D = Tuple[int, int]
IntOrPair = Union[int, Tuple[int, int]]


def _to_2tuple(value: Union[int, Sequence[int]]) -> Size2D:
    if isinstance(value, int):
        return (value, value)
    if len(value) != 2:
        raise ValueError(f"Expected a 2-element size, got {value}.")
    return int(value[0]), int(value[1])


def _same_padding(kernel_size: IntOrPair, dilation: IntOrPair = 1) -> Tuple[int, int]:
    k_t, k_w = _to_2tuple(kernel_size)
    d_t, d_w = _to_2tuple(dilation)
    return ((k_t - 1) // 2) * d_t, ((k_w - 1) // 2) * d_w


class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k: IntOrPair = 3,
        s: IntOrPair = 1,
        p: Tuple[int, int] | None = None,
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


class SEBlock(nn.Module):
    """Lightweight channel recalibration for fused structural features."""

    def __init__(self, ch: int, r: int = 8) -> None:
        super().__init__()
        hidden = max(1, ch // r)
        self.fc1 = nn.Conv2d(ch, hidden, kernel_size=1)
        self.fc2 = nn.Conv2d(hidden, ch, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = F.adaptive_avg_pool2d(x, output_size=1)
        w = F.gelu(self.fc1(w))
        w = torch.sigmoid(self.fc2(w))
        return x * w


class SeismicPatchStem(nn.Module):
    """
    PSTM front-end with anisotropic branches.

    Each branch emphasizes a different local cue before any aggressive resampling:
    - temporal wavelet texture
    - local reflector geometry
    - lateral continuity
    """

    def __init__(self, out_ch: int) -> None:
        super().__init__()
        branch_ch = max(out_ch // 2, 8)
        self.temporal_branch = ConvBNAct(1, branch_ch, k=(11, 1), s=1)
        self.event_branch = ConvBNAct(1, branch_ch, k=(7, 5), s=1)
        self.lateral_branch = ConvBNAct(1, branch_ch, k=(3, 9), s=1)
        self.fuse = ConvBNAct(branch_ch * 3, out_ch, k=1, s=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.cat(
            [
                self.temporal_branch(x),
                self.event_branch(x),
                self.lateral_branch(x),
            ],
            dim=1,
        )
        return self.fuse(h)


class StripPatchEmbedding(nn.Module):
    """
    Convert dense PSTM pixels into a lower-resolution feature grid.

    The stride is anisotropic so the time axis is reduced more aggressively than
    the lateral axis, while both dimensions still remain explicit 2D coordinates.
    """

    def __init__(self, channels: int, stride: Size2D = (2, 2)) -> None:
        super().__init__()
        self.pre = ConvBNAct(channels, channels, k=(7, 3), s=1)
        self.embed = ConvBNAct(channels, channels, k=(7, 5), s=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embed(self.pre(x))


class SeismicTextureBlock(nn.Module):
    """
    Residual block that separates temporal texture, lateral continuity,
    and local structural mixing.
    """

    def __init__(
        self,
        ch: int,
        dilation_time: int = 1,
        dilation_lateral: int = 1,
        dropout: float = 0.0,
        use_se: bool = True,
        se_ratio: int = 8,
    ) -> None:
        super().__init__()
        self.temporal = ConvBNAct(
            ch,
            ch,
            k=(7, 1),
            d=(dilation_time, 1),
            groups=ch,
        )
        self.lateral = ConvBNAct(
            ch,
            ch,
            k=(1, 9),
            d=(1, dilation_lateral),
            groups=ch,
        )
        self.structure = ConvBNAct(
            ch,
            ch,
            k=(3, 3),
            d=(dilation_time, dilation_lateral),
            groups=ch,
        )
        self.mix = nn.Sequential(
            nn.Conv2d(ch * 3, ch * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(ch * 2),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(ch * 2, ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.se = SEBlock(ch, r=se_ratio) if use_se else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.cat(
            [
                self.temporal(x),
                self.lateral(x),
                self.structure(x),
            ],
            dim=1,
        )
        y = self.se(self.mix(y))
        return self.act(x + y)


class DownsampleBlock(nn.Module):
    """Strided anisotropic transition between stages."""

    def __init__(self, in_ch: int, out_ch: int, stride: Size2D) -> None:
        super().__init__()
        self.pre = ConvBNAct(in_ch, in_ch, k=(5, 3), s=1)
        self.down = ConvBNAct(in_ch, out_ch, k=(5, 3), s=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.pre(x))


class LateralContextBlock(nn.Module):
    """
    Build one token per lateral position from a pooled strip summary, then apply
    lightweight self-attention along the lateral axis to model reflector continuity.
    """

    def __init__(
        self,
        ch: int,
        token_dim: int = 64,
        context_bins: int = 8,
        num_heads: int = 4,
        dropout: float = 0.0,
        se_ratio: int = 8,
    ) -> None:
        super().__init__()
        if token_dim % num_heads != 0:
            raise ValueError(f"token_dim={token_dim} must be divisible by num_heads={num_heads}.")
        self.context_bins = context_bins
        self.token_dim = token_dim

        self.context_reduce = ConvBNAct(ch, token_dim, k=1, s=1)
        self.token_proj = nn.Linear(token_dim * context_bins, token_dim)
        self.norm1 = nn.LayerNorm(token_dim)
        self.attn = nn.MultiheadAttention(token_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(token_dim)
        self.mlp = nn.Sequential(
            nn.Linear(token_dim, token_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(token_dim * 2, token_dim),
        )
        self.out_proj = nn.Sequential(
            nn.Conv2d(token_dim, ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.gate = SEBlock(ch, r=se_ratio)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape

        # Pool the time axis into a small number of bins but keep every lateral position.
        strip = F.adaptive_avg_pool2d(x, output_size=(self.context_bins, w))
        strip = self.context_reduce(strip)  # [B, token_dim, bins, W]

        # One token per lateral position, with a compact summary of the vertical strip.
        tokens = strip.permute(0, 3, 1, 2).reshape(b, w, self.token_dim * self.context_bins)
        tokens = self.token_proj(tokens)

        attn_in = self.norm1(tokens)
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        tokens = tokens + attn_out
        tokens = tokens + self.mlp(self.norm2(tokens))

        # Broadcast the lateral context back across time for fusion with local features.
        context = tokens.transpose(1, 2).unsqueeze(2)  # [B, token_dim, 1, W]
        context = F.interpolate(context, size=(h, w), mode="bilinear", align_corners=False)
        context = self.gate(self.out_proj(context))
        return self.act(x + context)


class ProjectionHead(nn.Module):
    """Project multi-level structural features into a fusion-friendly 2D map."""

    def __init__(self, deep_ch: int, mid_ch: int, hidden_ch: int, out_ch: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.mid_proj = ConvBNAct(mid_ch, hidden_ch // 2, k=1, s=1)
        self.deep_proj = ConvBNAct(deep_ch, hidden_ch, k=1, s=1)
        self.fuse = nn.Sequential(
            ConvBNAct(hidden_ch + hidden_ch // 2, hidden_ch, k=3, s=1),
            SeismicTextureBlock(hidden_ch, dilation_time=1, dilation_lateral=1, dropout=dropout, use_se=False),
            nn.Conv2d(hidden_ch, out_ch, kernel_size=1, bias=True),
        )

    def forward(self, mid: torch.Tensor, deep: torch.Tensor, out_hw: Size2D) -> torch.Tensor:
        mid = F.adaptive_avg_pool2d(self.mid_proj(mid), output_size=out_hw)
        deep = F.adaptive_avg_pool2d(self.deep_proj(deep), output_size=out_hw)
        return self.fuse(torch.cat([mid, deep], dim=1))


class SeismicImageEncoderA(nn.Module):
    """
    Shape-flexible PSTM encoder.

    Input:
        x: [B, 1, T, W]

    Output:
        y: [B, C_out, H_out, W_out]
    """

    def __init__(
        self,
        c1: int = 32,
        c2: int = 64,
        c3: int = 96,
        c4: int = 128,
        C_out: int = 64,
        use_se: bool = True,
        se_ratio: int = 8,
        dropout: float = 0.0,
        out_hw: Union[int, Sequence[int]] = (70, 70),
        context_bins: int = 8,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.out_hw = _to_2tuple(out_hw)

        self.stem = SeismicPatchStem(out_ch=c1)
        self.patch_embed = StripPatchEmbedding(channels=c1, stride=(2, 2))

        self.stage1 = nn.Sequential(
            SeismicTextureBlock(c1, dilation_time=1, dilation_lateral=1, dropout=dropout, use_se=use_se, se_ratio=se_ratio),
            SeismicTextureBlock(c1, dilation_time=2, dilation_lateral=1, dropout=dropout, use_se=use_se, se_ratio=se_ratio),
        )

        self.down1 = DownsampleBlock(c1, c2, stride=(2, 1))
        self.stage2 = nn.Sequential(
            SeismicTextureBlock(c2, dilation_time=1, dilation_lateral=1, dropout=dropout, use_se=use_se, se_ratio=se_ratio),
            SeismicTextureBlock(c2, dilation_time=2, dilation_lateral=2, dropout=dropout, use_se=use_se, se_ratio=se_ratio),
        )

        self.down2 = DownsampleBlock(c2, c3, stride=(2, 2))
        self.stage3 = nn.Sequential(
            SeismicTextureBlock(c3, dilation_time=1, dilation_lateral=1, dropout=dropout, use_se=use_se, se_ratio=se_ratio),
            SeismicTextureBlock(c3, dilation_time=2, dilation_lateral=2, dropout=dropout, use_se=use_se, se_ratio=se_ratio),
        )

        self.down3 = DownsampleBlock(c3, c4, stride=(2, 2))
        self.stage4 = nn.Sequential(
            SeismicTextureBlock(c4, dilation_time=1, dilation_lateral=1, dropout=dropout, use_se=use_se, se_ratio=se_ratio),
            SeismicTextureBlock(c4, dilation_time=2, dilation_lateral=2, dropout=dropout, use_se=use_se, se_ratio=se_ratio),
        )

        token_dim = min(max(c4 // 2, 32), 128)
        token_dim = max(num_heads, (token_dim // num_heads) * num_heads)
        self.lateral_context = LateralContextBlock(
            ch=c4,
            token_dim=token_dim,
            context_bins=context_bins,
            num_heads=num_heads,
            dropout=dropout,
            se_ratio=se_ratio,
        )
        self.projection = ProjectionHead(deep_ch=c4, mid_ch=c3, hidden_ch=c4, out_ch=C_out, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected a 4D tensor [B, 1, T, W], got shape {tuple(x.shape)}.")
        if x.size(1) != 1:
            raise ValueError(f"Expected a single PSTM input channel, got {x.size(1)} channels.")

        # [B, 1, T, W] -> [B, c1, T, W]
        h = self.stem(x)

        # Patchify while keeping a 2D time/lateral grid.
        h = self.patch_embed(h)  # [B, c1, T/2, W/2]
        h1 = self.stage1(h)

        h2 = self.stage2(self.down1(h1))  # mild extra time reduction, lateral axis preserved
        h3 = self.stage3(self.down2(h2))  # joint time/lateral compression
        h4 = self.stage4(self.down3(h3))

        h4 = self.lateral_context(h4)
        y = self.projection(mid=h3, deep=h4, out_hw=self.out_hw)
        return y

