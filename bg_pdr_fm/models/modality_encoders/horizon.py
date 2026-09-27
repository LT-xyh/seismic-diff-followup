from __future__ import annotations

from typing import Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


Size2D = Tuple[int, int]
IntOrPair = Union[int, Sequence[int]]


def _to_2tuple(value: Union[int, Sequence[int]]) -> Size2D:
    if isinstance(value, int):
        return (value, value)
    if len(value) != 2:
        raise ValueError(f"Expected a 2-element size, got {value}.")
    return int(value[0]), int(value[1])


def _same_padding(kernel_size: IntOrPair, dilation: IntOrPair = 1) -> Size2D:
    k_h, k_w = _to_2tuple(kernel_size)
    d_h, d_w = _to_2tuple(dilation)
    return ((k_h - 1) // 2) * d_h, ((k_w - 1) // 2) * d_w


class ConvBNGELU2d(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k: IntOrPair = 3,
        s: IntOrPair = 1,
        p: Size2D | None = None,
        d: IntOrPair = 1,
        groups: int = 1,
        bias: bool = False,
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
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.GELU() if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class SEBlock(nn.Module):
    """Channel recalibration used after local/global fusion."""

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


class SpatialAttn(nn.Module):
    """Optional final spatial gate for the projected dense feature map."""

    def __init__(self, k: int = 7) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=k, padding=k // 2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = torch.cat([x.max(dim=1, keepdim=True).values, x.mean(dim=1, keepdim=True)], dim=1)
        return x * torch.sigmoid(self.conv(pooled))


def make_gaussian_kernel(size: int = 7, sigma: float = 1.5, device: str | torch.device = "cpu") -> torch.Tensor:
    """Create a deterministic 2D Gaussian kernel."""

    ax = torch.arange(size, device=device, dtype=torch.float32) - (size - 1) / 2.0
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    ker = torch.exp(-(xx.square() + yy.square()) / (2.0 * sigma * sigma))
    ker = ker / ker.sum().clamp_min(1e-12)
    return ker


class FixedGaussianBlur2d(nn.Module):
    """Single-channel Gaussian blur used to widen thin horizon lines into soft bands."""

    def __init__(self, k: int = 7, sigma: float = 1.5) -> None:
        super().__init__()
        self.k = k
        self.sigma = sigma
        self.register_buffer("weight", torch.zeros(1, 1, k, k), persistent=False)
        self._initialized = False

    def _maybe_init(self, device: torch.device, dtype: torch.dtype) -> None:
        if (not self._initialized) or (self.weight.device != device) or (self.weight.dtype != dtype):
            ker = make_gaussian_kernel(self.k, self.sigma, device=device).to(dtype=dtype)
            self.weight.data = ker.view(1, 1, self.k, self.k)
            self._initialized = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._maybe_init(x.device, x.dtype)
        return F.conv2d(x, self.weight, stride=1, padding=self.k // 2)


class FixedSobelMagnitude2d(nn.Module):
    """Deterministic local gradient magnitude on a single-channel band map."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("weight_x", torch.zeros(1, 1, 3, 3), persistent=False)
        self.register_buffer("weight_y", torch.zeros(1, 1, 3, 3), persistent=False)
        self._initialized = False

    def _maybe_init(self, device: torch.device, dtype: torch.dtype) -> None:
        if (not self._initialized) or (self.weight_x.device != device) or (self.weight_x.dtype != dtype):
            sobel_x = torch.tensor(
                [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
                device=device,
                dtype=dtype,
            )
            sobel_y = torch.tensor(
                [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
                device=device,
                dtype=dtype,
            )
            self.weight_x.data = sobel_x.view(1, 1, 3, 3)
            self.weight_y.data = sobel_y.view(1, 1, 3, 3)
            self._initialized = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._maybe_init(x.device, x.dtype)
        gx = F.conv2d(x, self.weight_x, stride=1, padding=1)
        gy = F.conv2d(x, self.weight_y, stride=1, padding=1)
        return torch.sqrt(gx.square() + gy.square() + 1e-8)


def add_coord_channels(x: torch.Tensor) -> torch.Tensor:
    """Append normalized y/x coordinate channels to a 4D feature tensor."""

    b, _, h, w = x.shape
    device = x.device
    dtype = x.dtype
    yy = torch.linspace(-1.0, 1.0, steps=h, device=device, dtype=dtype).view(1, 1, h, 1).expand(b, 1, h, w)
    xx = torch.linspace(-1.0, 1.0, steps=w, device=device, dtype=dtype).view(1, 1, 1, w).expand(b, 1, h, w)
    return torch.cat([x, yy, xx], dim=1)


class InterfaceEnhancer(nn.Module):
    """
    Deterministic sparse-interface front-end.

    Input:
        x: [B, 1, H, W]

    Output:
        features: [B, 7, H, W] = [mask, soft_band, wide_band, local_density, grad_mag, y, x]
        guide: [B, 1, H, W]
    """

    def __init__(self, k_gauss: int = 7, sigma_gauss: float = 1.5, band_kernel: int = 5, density_kernel: int = 9) -> None:
        super().__init__()
        self.blur = FixedGaussianBlur2d(k=k_gauss, sigma=sigma_gauss)
        self.grad = FixedSobelMagnitude2d()
        self.band_kernel = band_kernel
        self.density_kernel = density_kernel

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        soft_band = self.blur(x)
        wide_band = F.max_pool2d(soft_band, kernel_size=self.band_kernel, stride=1, padding=self.band_kernel // 2)
        local_density = F.avg_pool2d(x, kernel_size=self.density_kernel, stride=1, padding=self.density_kernel // 2)
        grad_mag = self.grad(soft_band)

        dense_features = add_coord_channels(torch.cat([x, soft_band, wide_band, local_density, grad_mag], dim=1))
        guide = wide_band + 0.5 * grad_mag + 0.25 * local_density
        guide = guide / guide.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        return dense_features, guide


class InterfacePatchStem(nn.Module):
    """
    Multi-branch local stem specialized for thin interfaces.

    The branches emphasize:
    - pixel-scale interface presence
    - lateral continuity along horizons
    - local band fragments with small vertical support
    """

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        branch_ch = max(out_ch // 2, 8)
        self.thin_branch = ConvBNGELU2d(in_ch, branch_ch, k=3, s=1)
        self.lateral_branch = ConvBNGELU2d(in_ch, branch_ch, k=(1, 7), s=1)
        self.fragment_branch = ConvBNGELU2d(in_ch, branch_ch, k=(5, 3), s=1)
        self.fuse = ConvBNGELU2d(branch_ch * 3, out_ch, k=1, s=1)
        self.guide_gate = nn.Conv2d(1, out_ch, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        h = torch.cat(
            [
                self.thin_branch(x),
                self.lateral_branch(x),
                self.fragment_branch(x),
            ],
            dim=1,
        )
        h = self.fuse(h)
        gate = torch.sigmoid(self.guide_gate(guide))
        return h * (1.0 + gate)


class LocalInterfaceBlock(nn.Module):
    """
    Lightweight fragment encoder operating on dense or token grids.

    It is intentionally anisotropic: one branch follows lateral continuity,
    one branch focuses on thin vertical support, and one mixes local fragments.
    """

    def __init__(self, ch: int, dropout: float = 0.0, use_se: bool = True, se_ratio: int = 8) -> None:
        super().__init__()
        self.norm = nn.BatchNorm2d(ch)
        self.band_branch = ConvBNGELU2d(ch, ch, k=(5, 1), groups=ch)
        self.lateral_branch = ConvBNGELU2d(ch, ch, k=(1, 7), groups=ch)
        self.fragment_branch = ConvBNGELU2d(ch, ch, k=(3, 5), groups=ch)
        self.mix = nn.Sequential(
            nn.Conv2d(ch * 3, ch * 2, kernel_size=1, bias=False),
            nn.BatchNorm2d(ch * 2),
            nn.GELU(),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(ch * 2, ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.guide_gate = nn.Conv2d(1, ch, kernel_size=1, bias=True)
        self.se = SEBlock(ch, r=se_ratio) if use_se else nn.Identity()
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        base = self.norm(x)
        y = torch.cat(
            [
                self.band_branch(base),
                self.lateral_branch(base),
                self.fragment_branch(base),
            ],
            dim=1,
        )
        y = self.mix(y)
        y = y * (1.0 + torch.sigmoid(self.guide_gate(guide)))
        y = self.se(y)
        return self.act(x + y)


class SparseBandTokenizer(nn.Module):
    """
    Convert a dense local feature map into a compact token grid using band-weighted pooling.

    The tokens summarize only sparse active regions strongly, while still remaining robust
    when a patch is empty by adding a small uniform fallback weight.
    """

    def __init__(self, local_ch: int, token_ch: int, token_hw: IntOrPair = (14, 14)) -> None:
        super().__init__()
        self.token_hw = _to_2tuple(token_hw)
        self.proj = ConvBNGELU2d(local_ch + 4, token_ch, k=1, s=1)

    def forward(
        self,
        local_map: torch.Tensor,
        guide: torch.Tensor,
        mask: torch.Tensor,
        coords: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        weight = guide + 0.05
        pooled_weight = F.adaptive_avg_pool2d(weight, output_size=self.token_hw).clamp_min(1e-6)
        pooled_feat = F.adaptive_avg_pool2d(local_map * weight, output_size=self.token_hw) / pooled_weight
        pooled_mask = F.adaptive_avg_pool2d(mask * weight, output_size=self.token_hw) / pooled_weight
        pooled_guide = F.adaptive_avg_pool2d(guide, output_size=self.token_hw)
        pooled_coords = F.adaptive_avg_pool2d(coords * weight, output_size=self.token_hw) / pooled_weight

        token_grid = self.proj(torch.cat([pooled_feat, pooled_mask, pooled_guide, pooled_coords], dim=1))
        return token_grid, pooled_guide, pooled_coords


class GlobalTokenEncoder(nn.Module):
    """
    Global context encoder over the sparse token grid.

    Each token receives:
    - its locally encoded sparse-band content
    - pooled horizon occupancy
    - normalized 2D token position
    """

    def __init__(self, token_ch: int, num_heads: int = 4, num_layers: int = 2, dropout: float = 0.0) -> None:
        super().__init__()
        if token_ch % num_heads != 0:
            raise ValueError(f"token_ch={token_ch} must be divisible by num_heads={num_heads}.")

        self.pos_mlp = nn.Sequential(
            nn.Linear(3, token_ch),
            nn.GELU(),
            nn.Linear(token_ch, token_ch),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=token_ch,
            nhead=num_heads,
            dim_feedforward=token_ch * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)

    def forward(self, token_grid: torch.Tensor, guide_grid: torch.Tensor, coord_grid: torch.Tensor) -> torch.Tensor:
        b, c, gh, gw = token_grid.shape
        tokens = token_grid.flatten(2).transpose(1, 2).contiguous()  # [B, N, C]
        aux = torch.cat([coord_grid, guide_grid], dim=1).permute(0, 2, 3, 1).reshape(b, gh * gw, 3)
        tokens = self.encoder(tokens + self.pos_mlp(aux))
        return tokens.transpose(1, 2).reshape(b, c, gh, gw).contiguous()


class ProjectionHead(nn.Module):
    """
    Reproject local dense features and global token context into a unified 2D map.
    """

    def __init__(
        self,
        local_ch: int,
        token_ch: int,
        fusion_ch: int,
        out_ch: int,
        dropout: float = 0.0,
        use_se: bool = True,
        use_spatial_attn: bool = True,
    ) -> None:
        super().__init__()
        local_out = max(fusion_ch // 2, 8)
        token_out = max(fusion_ch // 2, 8)
        guide_out = max(fusion_ch // 4, 8)

        self.local_proj = ConvBNGELU2d(local_ch, local_out, k=1, s=1)
        self.token_proj = ConvBNGELU2d(token_ch, token_out, k=1, s=1)
        self.guide_proj = ConvBNGELU2d(3, guide_out, k=3, s=1)

        fused_ch = local_out + token_out + guide_out
        self.fuse = ConvBNGELU2d(fused_ch, fusion_ch, k=3, s=1)
        self.refine = LocalInterfaceBlock(fusion_ch, dropout=dropout, use_se=use_se)
        self.spatial = SpatialAttn(k=7) if use_spatial_attn else nn.Identity()
        self.out_proj = nn.Conv2d(fusion_ch, out_ch, kernel_size=1, bias=True)

    def forward(
        self,
        local_map: torch.Tensor,
        token_map: torch.Tensor,
        guide_pack: torch.Tensor,
        guide: torch.Tensor,
        out_hw: Size2D,
    ) -> torch.Tensor:
        local_dense = F.interpolate(self.local_proj(local_map), size=out_hw, mode="bilinear", align_corners=False)
        token_dense = F.interpolate(self.token_proj(token_map), size=out_hw, mode="bilinear", align_corners=False)
        guide_dense = F.interpolate(self.guide_proj(guide_pack), size=out_hw, mode="bilinear", align_corners=False)
        guide_dense_scalar = F.interpolate(guide, size=out_hw, mode="bilinear", align_corners=False)

        fused = self.fuse(torch.cat([local_dense, token_dense, guide_dense], dim=1))
        fused = self.refine(fused, guide_dense_scalar)
        fused = self.spatial(fused)
        return self.out_proj(fused)


class HorizonEncoderA(nn.Module):
    """
    Sparse interface / contour-aware horizon encoder.

    Pipeline:
    1. Deterministic interface enhancement expands thin binary lines into stable sparse bands.
    2. A local interface stem preserves thin structure localization on the native input grid.
    3. Band-weighted tokenization extracts local horizon fragments onto a compact token grid.
    4. Lightweight token-grid encoding plus Transformer context capture broader structural continuity.
    5. Local and global features are reprojected into a dense 2D feature map for fusion.

    Input:
        x: [B, 1, H, W]

    Output:
        y: [B, C_out, H_out, W_out]
    """

    def __init__(
        self,
        C_out: int = 16,
        c1: int = 32,
        c2: int = 48,
        c3: int = 64,
        k_gauss: int = 7,
        sigma_gauss: float = 1.5,
        use_se: bool = True,
        use_spatial_attn: bool = True,
        dropout: float = 0.0,
        out_hw: IntOrPair = (70, 70),
        token_hw: IntOrPair = (14, 14),
        num_global_layers: int = 2,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        self.out_hw = _to_2tuple(out_hw)

        self.interface_enhancer = InterfaceEnhancer(k_gauss=k_gauss, sigma_gauss=sigma_gauss)
        self.local_stem = InterfacePatchStem(in_ch=7, out_ch=c1)
        self.local_encoder = nn.ModuleList(
            [
                LocalInterfaceBlock(c1, dropout=dropout, use_se=use_se),
                LocalInterfaceBlock(c1, dropout=dropout, use_se=use_se),
            ]
        )

        self.tokenizer = SparseBandTokenizer(local_ch=c1, token_ch=c2, token_hw=token_hw)
        self.token_local_encoder = nn.ModuleList(
            [
                LocalInterfaceBlock(c2, dropout=dropout, use_se=use_se),
                LocalInterfaceBlock(c2, dropout=dropout, use_se=use_se),
            ]
        )
        self.global_encoder = GlobalTokenEncoder(
            token_ch=c2,
            num_heads=num_heads,
            num_layers=num_global_layers,
            dropout=dropout,
        )
        self.projection = ProjectionHead(
            local_ch=c1,
            token_ch=c2,
            fusion_ch=c3,
            out_ch=C_out,
            dropout=dropout,
            use_se=use_se,
            use_spatial_attn=use_spatial_attn,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected x with shape [B, 1, H, W], got {tuple(x.shape)}.")
        if x.size(1) != 1:
            raise ValueError(f"Expected a single horizon channel, got {x.size(1)}.")

        x = x.float()

        # Step 1: deterministic sparse-band enhancement on the native grid.
        enhanced, guide = self.interface_enhancer(x)  # [B, 7, H, W], [B, 1, H, W]

        # Step 2: dense local fragment encoding that preserves thin horizon localization.
        local_map = self.local_stem(enhanced, guide)  # [B, c1, H, W]
        for block in self.local_encoder:
            local_map = block(local_map, guide)

        # Step 3: band-weighted pooling extracts sparse local fragments into a token grid.
        token_grid, guide_grid, coord_grid = self.tokenizer(
            local_map=local_map,
            guide=guide,
            mask=enhanced[:, :1],
            coords=enhanced[:, -2:],
        )  # [B, c2, Gh, Gw], [B, 1, Gh, Gw], [B, 2, Gh, Gw]

        # Step 4: encode fragment-scale structure locally on the token grid, then globally across all tokens.
        for block in self.token_local_encoder:
            token_grid = block(token_grid, guide_grid)
        token_grid = self.global_encoder(token_grid, guide_grid=guide_grid, coord_grid=coord_grid)

        # Step 5: project back to a dense map for downstream multimodal fusion.
        guide_pack = torch.cat([enhanced[:, :1], enhanced[:, 1:2], guide], dim=1)
        return self.projection(local_map, token_grid, guide_pack, guide, out_hw=self.out_hw)


def _build_toy_horizon_batch(shape: tuple[int, int, int, int]) -> torch.Tensor:
    b, c, h, w = shape
    if c != 1:
        raise ValueError("Toy horizon generator expects a single-channel mask.")

    x = torch.zeros(shape, dtype=torch.float32)
    cols = torch.arange(w)

    for idx in range(b):
        # Flat but offset horizon.
        row1 = min(h - 1, max(0, int(0.22 * h) + idx))
        x[idx, 0, row1, :] = 1.0

        # Slightly curved horizon.
        curve = 0.55 * h + 0.08 * h * torch.sin(2.0 * torch.pi * cols.float() / max(w, 8))
        curve = curve.round().long().clamp(0, h - 1)
        x[idx, 0, curve, cols] = 1.0

        # Fragmented horizon segment.
        row2 = min(h - 1, max(0, int(0.76 * h) - idx))
        seg_start = w // 8
        seg_end = min(w, seg_start + max(w // 3, 6))
        x[idx, 0, row2, seg_start:seg_end] = 1.0

        # Small dipping fragment.
        frag_len = max(min(h, w) // 4, 6)
        start_x = min(w - 1, max(0, w // 3))
        frag_x = torch.arange(start_x, min(w, start_x + frag_len))
        frag_y = (0.35 * h + 0.3 * (frag_x - start_x)).round().long().clamp(0, h - 1)
        x[idx, 0, frag_y, frag_x] = 1.0

    return x

