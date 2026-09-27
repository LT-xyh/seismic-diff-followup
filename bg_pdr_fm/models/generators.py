"""Background and residual generation modules for BG-PDR-FM."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .codecs import LatentCodec
from .filters import LowHighPassFilter
from .residual_backends import build_residual_backend
from .types import BackgroundEstimate


def _check_even_hw(hw: tuple[int, int], label: str) -> None:
    if hw[0] % 2 != 0 or hw[1] % 2 != 0:
        raise ValueError(f"{label} must be even for one-level Haar projection, got {hw}.")


def _check_wavelet_level(level: int) -> int:
    level = int(level)
    if level < 1:
        raise ValueError(f"background_wavelet_level must be >= 1, got {level}.")
    return level


def _padded_hw_for_level(hw: tuple[int, int], level: int) -> tuple[int, int]:
    multiple = 2 ** _check_wavelet_level(level)
    return tuple(int((size + multiple - 1) // multiple * multiple) for size in hw)


def _right_bottom_pad_to_hw(x: torch.Tensor, padded_hw: tuple[int, int]) -> torch.Tensor:
    pad_h = int(padded_hw[0]) - int(x.shape[-2])
    pad_w = int(padded_hw[1]) - int(x.shape[-1])
    if pad_h < 0 or pad_w < 0:
        raise ValueError(f"Cannot pad tensor from {tuple(x.shape[-2:])} to smaller size {padded_hw}.")
    if pad_h == 0 and pad_w == 0:
        return x
    return F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")


def _haar_filter_bank(x: torch.Tensor) -> torch.Tensor:
    return x.new_tensor(
        [
            [[[0.5, 0.5], [0.5, 0.5]]],
            [[[-0.5, 0.5], [-0.5, 0.5]]],
            [[[-0.5, -0.5], [0.5, 0.5]]],
            [[[0.5, -0.5], [-0.5, 0.5]]],
        ]
    )


def haar_analyze(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """One-level orthonormal Haar analysis without padding."""

    if x.ndim != 4:
        raise ValueError(f"Expected BCHW tensor for Haar analysis, got {tuple(x.shape)}.")
    _check_even_hw(tuple(x.shape[-2:]), "input spatial size")
    b, c, _, _ = x.shape
    weight = _haar_filter_bank(x).repeat(c, 1, 1, 1)
    coeffs = F.conv2d(x, weight, stride=2, padding=0, groups=c)
    coeffs = coeffs.view(b, c, 4, coeffs.shape[-2], coeffs.shape[-1])
    return coeffs[:, :, 0], coeffs[:, :, 1], coeffs[:, :, 2], coeffs[:, :, 3]


def haar_reconstruct_from_ll(ll: torch.Tensor, out_hw: tuple[int, int] | None = None) -> torch.Tensor:
    """Reconstruct a full-resolution field from LL coefficients with zero details."""

    if ll.ndim != 4:
        raise ValueError(f"Expected BCHW LL tensor, got {tuple(ll.shape)}.")
    _, c, _, _ = ll.shape
    weight = _haar_filter_bank(ll)[:1].repeat(c, 1, 1, 1)
    recon = F.conv_transpose2d(ll, weight, stride=2, padding=0, groups=c)
    if out_hw is not None:
        if tuple(recon.shape[-2:]) != tuple(out_hw):
            raise ValueError(f"LL reconstruction produced {tuple(recon.shape[-2:])}, expected {tuple(out_hw)}.")
    return recon


def haar_lowpass_reconstruct(x: torch.Tensor, level: int = 1) -> torch.Tensor:
    level = _check_wavelet_level(level)
    out_hw = tuple(x.shape[-2:])
    padded_hw = _padded_hw_for_level(out_hw, level)
    ll = _right_bottom_pad_to_hw(x, padded_hw)
    for _ in range(level):
        ll, _, _, _ = haar_analyze(ll)
    recon = haar_reconstruct_from_ll_level(ll, level=level, out_hw=padded_hw)
    return recon[..., : out_hw[0], : out_hw[1]]


def haar_reconstruct_from_ll_level(
    ll: torch.Tensor,
    level: int = 1,
    out_hw: tuple[int, int] | None = None,
) -> torch.Tensor:
    level = _check_wavelet_level(level)
    recon = ll
    for _ in range(level):
        recon = haar_reconstruct_from_ll(recon)
    if out_hw is not None:
        if tuple(recon.shape[-2:]) != tuple(out_hw):
            raise ValueError(f"LL reconstruction produced {tuple(recon.shape[-2:])}, expected {tuple(out_hw)}.")
    return recon


def haar_detail_leak(x: torch.Tensor) -> torch.Tensor:
    _, lh, hl, hh = haar_analyze(x)
    return torch.stack([lh.abs().mean(), hl.abs().mean(), hh.abs().mean()]).mean()


class _BackgroundConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = min(8, int(out_channels))
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=groups, num_channels=out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=groups, num_channels=out_channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _BackgroundUNet(nn.Module):
    """One-level U-Net for direct full-resolution background regression."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int = 1) -> None:
        super().__init__()
        hidden = int(hidden_channels)
        mid = hidden * 2
        self.enc = _BackgroundConvBlock(in_channels, hidden)
        self.down = nn.Conv2d(hidden, mid, kernel_size=3, stride=2, padding=1)
        self.mid = _BackgroundConvBlock(mid, mid)
        self.dec = _BackgroundConvBlock(mid + hidden, hidden)
        self.out = nn.Conv2d(hidden, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.enc(x)
        h = self.down(skip)
        h = self.mid(h)
        h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        h = self.dec(torch.cat([h, skip], dim=1))
        return self.out(h)


class BackgroundEstimator(nn.Module):
    def __init__(self, cond_channels: int = 64, latent_channels: int = 16, codec: LatentCodec | None = None,
                 highpass_filter: LowHighPassFilter | None = None, output_activation: str = "tanh",
                 hidden_channels: int | None = None, output_mode: str = "direct",
                 wavelet_level: int = 1, bottleneck_hw: tuple[int, int] = (18, 18),
                 upsample_mode: str = "bilinear", backend: str = "conv",
                 fm_backend: str = "unet_fm_film", fm_latent_hw: tuple[int, int] = (70, 70),
                 fm_num_train_timesteps: int = 1000, fm_num_inference_steps: int = 20,
                 fm_loss_type: str = "mse",
                 fm_block_out_channels: Sequence[int] | int | str | None = None,
                 fm_layers_per_block: int = 2,
                 fm_cross_attention_dim: int | None = None,
                 fm_attention_head_dim: int = 8) -> None:
        super().__init__()
        del latent_channels, codec
        hidden_channels = int(hidden_channels or cond_channels)
        self.highpass_filter = highpass_filter if highpass_filter is not None else LowHighPassFilter()
        self.output_activation = str(output_activation).lower()
        if self.output_activation not in {"tanh", "identity"}:
            raise ValueError(
                f"Unknown background output_activation: {output_activation!r}. "
                "Expected 'tanh' or 'identity'."
            )
        self.output_mode = str(output_mode).lower()
        if self.output_mode not in {"direct", "wavelet_ll", "smooth_bottleneck"}:
            raise ValueError(
                f"Unknown background output_mode: {output_mode!r}. "
                "Expected 'direct', 'wavelet_ll', or 'smooth_bottleneck'."
            )
        self.wavelet_level = _check_wavelet_level(wavelet_level)
        self.bottleneck_hw = tuple(int(v) for v in bottleneck_hw)
        if len(self.bottleneck_hw) != 2 or min(self.bottleneck_hw) <= 0:
            raise ValueError(f"background_bottleneck_hw must be a positive HW pair, got {bottleneck_hw!r}.")
        self.upsample_mode = str(upsample_mode).lower()
        if self.upsample_mode not in {"bilinear", "bicubic"}:
            raise ValueError(
                f"Unknown background_upsample_mode: {upsample_mode!r}. "
                "Expected 'bilinear' or 'bicubic'."
            )
        self.backend = str(backend).lower()
        if self.backend not in {"conv", "unet_direct", "flow_matching"}:
            raise ValueError(
                f"Unknown background backend: {backend!r}. Expected 'conv', 'unet_direct', or 'flow_matching'."
            )
        if self.backend in {"unet_direct", "flow_matching"} and self.output_mode != "direct":
            raise ValueError(f"background_backend={self.backend!r} requires background_output_mode='direct'.")
        if self.backend == "unet_direct":
            self.network = _BackgroundUNet(cond_channels, hidden_channels, out_channels=1)
        elif self.backend == "flow_matching":
            self.fm_loss_type = str(fm_loss_type)
            self.fm_num_inference_steps = int(fm_num_inference_steps)
            self.network = build_residual_backend(
                backend=str(fm_backend),
                latent_channels=1,
                cond_channels=cond_channels,
                latent_hw=tuple(int(v) for v in fm_latent_hw),
                num_train_timesteps=int(fm_num_train_timesteps),
                hidden_channels=hidden_channels,
                block_out_channels=fm_block_out_channels,
                layers_per_block=int(fm_layers_per_block),
                cross_attention_dim=fm_cross_attention_dim,
                attention_head_dim=int(fm_attention_head_dim),
            )
        else:
            self.network = nn.Sequential(
                nn.Conv2d(cond_channels, hidden_channels, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv2d(hidden_channels, 1, kernel_size=1),
            )

    def _activate(self, bg: torch.Tensor) -> torch.Tensor:
        if self.output_activation == "tanh":
            return torch.tanh(bg)
        return bg

    def forward(
        self,
        cond_num: torch.Tensor,
        out_hw: tuple[int, int],
        target: torch.Tensor | None = None,
    ) -> BackgroundEstimate:
        fm_loss = None
        fm_velocity_pred = None
        fm_velocity_target = None
        if self.backend == "unet_direct":
            if tuple(cond_num.shape[-2:]) != tuple(out_hw):
                cond_num = F.interpolate(cond_num, size=out_hw, mode="bilinear", align_corners=False)
            bg = self.network(cond_num)
            bg = self._activate(bg)
            bg_ll = None
            bg_lowres = None
        elif self.backend == "flow_matching":
            if tuple(cond_num.shape[-2:]) != tuple(out_hw):
                cond_num = F.interpolate(cond_num, size=out_hw, mode="bilinear", align_corners=False)
            if target is not None:
                if tuple(target.shape[-2:]) != tuple(out_hw):
                    target = F.interpolate(target, size=out_hw, mode="bilinear", align_corners=False)
                fm = self.network.training_loss(target.detach(), cond_num, loss_type=self.fm_loss_type)
                bg = fm["z_res_pred"]
                fm_loss = fm["loss"]
                fm_velocity_pred = fm.get("velocity_pred")
                fm_velocity_target = fm.get("velocity_target")
            else:
                bg = self.network.sample(
                    cond_num,
                    x_size=(cond_num.shape[0], 1, int(out_hw[0]), int(out_hw[1])),
                    steps=self.fm_num_inference_steps,
                )
            bg = self._activate(bg)
            bg_ll = None
            bg_lowres = None
        elif self.output_mode == "wavelet_ll":
            padded_hw = _padded_hw_for_level(tuple(out_hw), self.wavelet_level)
            coeff_hw = (
                int(padded_hw[0]) // (2 ** self.wavelet_level),
                int(padded_hw[1]) // (2 ** self.wavelet_level),
            )
            cond_num = F.interpolate(cond_num, size=coeff_hw, mode="bilinear", align_corners=False)
            bg_ll = self.network(cond_num)
            if self.output_activation == "tanh":
                # A constant normalized field in [-1, 1] has LL coefficients in [-2**level, 2**level].
                bg_ll = float(2 ** self.wavelet_level) * torch.tanh(bg_ll)
            bg = haar_reconstruct_from_ll_level(bg_ll, level=self.wavelet_level, out_hw=padded_hw)
            bg = bg[..., : int(out_hw[0]), : int(out_hw[1])]
            bg_lowres = None
        elif self.output_mode == "smooth_bottleneck":
            cond_num = F.interpolate(cond_num, size=self.bottleneck_hw, mode="bilinear", align_corners=False)
            bg_lowres = self.network(cond_num)
            bg_lowres = self._activate(bg_lowres)
            bg = F.interpolate(bg_lowres, size=out_hw, mode=self.upsample_mode, align_corners=False)
            bg_ll = None
        else:
            cond_num = F.interpolate(cond_num, size=out_hw, mode="bilinear", align_corners=False)
            bg = self.network(cond_num)
            bg = self._activate(bg)
            bg_ll = None
            bg_lowres = None
        epsilon_h = self.highpass_filter.highpass(bg).abs().flatten(1).mean(dim=1)
        return BackgroundEstimate(
            bg_hat=bg,
            epsilon_h_b=epsilon_h,
            bg_ll=bg_ll,
            bg_lowres=bg_lowres,
            fm_loss=fm_loss,
            fm_velocity_pred=fm_velocity_pred,
            fm_velocity_target=fm_velocity_target,
        )


class ResidualFlowGenerator(nn.Module):
    """Residual generator wrapper with low-bandwidth background conditioning."""

    def __init__(self, cond_channels: int = 64, latent_channels: int = 16, bg_cond_channels: int = 8,
                 latent_hw: tuple[int, int] = (16, 16), backend: str = "simple_fm",
                 num_train_timesteps: int = 1000, num_inference_steps: int = 20, loss_type: str = "mse",
                 lowpass_filter: LowHighPassFilter | None = None, backend_hidden_channels: int = 96,
                 condition_merge: str = "add", background_context_source: str = "lowpass_embed",
                 backend_block_out_channels: Sequence[int] | int | str | None = None,
                 backend_layers_per_block: int = 2,
                 backend_cross_attention_dim: int | None = None,
                 backend_attention_head_dim: int = 8) -> None:
        super().__init__()
        self.bg_cond_channels = int(bg_cond_channels)
        self.backend_name = str(backend).lower()
        self.num_inference_steps = int(num_inference_steps)
        self.loss_type = str(loss_type)
        self.condition_merge = str(condition_merge).lower()
        if self.condition_merge not in {"add", "concat"}:
            raise ValueError(
                f"Unknown residual condition_merge: {condition_merge!r}. Expected 'add' or 'concat'."
            )
        self.background_context_source = str(background_context_source).lower()
        if self.background_context_source in {"lowpass", "lowpass_conv"}:
            self.background_context_source = "lowpass_embed"
        if self.background_context_source not in {"none", "lowpass_embed", "codec_latent", "codec_latent_raw"}:
            raise ValueError(
                "Unknown residual background_context_source: "
                f"{background_context_source!r}. Expected 'none', 'lowpass_embed', 'codec_latent', or 'codec_latent_raw'."
            )
        if self.background_context_source == "codec_latent_raw" and self.condition_merge != "concat":
            raise ValueError("residual_background_context_source='codec_latent_raw' requires residual_condition_merge='concat'.")
        bg_context_channels = latent_channels if self.background_context_source == "codec_latent_raw" else cond_channels
        if self.background_context_source == "none":
            backend_cond_channels = cond_channels
        else:
            backend_cond_channels = cond_channels if self.condition_merge == "add" else cond_channels + bg_context_channels
        self.lowpass_filter = lowpass_filter if lowpass_filter is not None else LowHighPassFilter()
        if self.background_context_source == "none":
            self.bg_embed = nn.Identity()
        elif self.background_context_source == "codec_latent":
            self.bg_embed = nn.Sequential(
                nn.Conv2d(latent_channels, cond_channels, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv2d(cond_channels, cond_channels, kernel_size=1),
            )
        elif self.background_context_source == "codec_latent_raw":
            self.bg_embed = nn.Identity()
        else:
            self.bg_embed = nn.Sequential(
                nn.Conv2d(1, self.bg_cond_channels, kernel_size=3, padding=1),
                nn.SiLU(),
                nn.Conv2d(self.bg_cond_channels, cond_channels, kernel_size=1),
            )
        self.backend = build_residual_backend(
            backend=self.backend_name,
            latent_channels=latent_channels,
            cond_channels=backend_cond_channels,
            latent_hw=latent_hw,
            num_train_timesteps=num_train_timesteps,
            hidden_channels=backend_hidden_channels,
            block_out_channels=backend_block_out_channels,
            layers_per_block=backend_layers_per_block,
            cross_attention_dim=backend_cross_attention_dim,
            attention_head_dim=backend_attention_head_dim,
        )

    def background_context(
        self,
        bg_hat: torch.Tensor,
        out_hw: tuple[int, int],
        z_bg: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.background_context_source == "none":
            return bg_hat.new_zeros((bg_hat.shape[0], self.bg_cond_channels, *out_hw))
        if self.background_context_source in {"codec_latent", "codec_latent_raw"}:
            if z_bg is None:
                raise ValueError(
                    "z_bg is required when residual_background_context_source is 'codec_latent' or 'codec_latent_raw'."
                )
            if tuple(z_bg.shape[-2:]) != tuple(out_hw):
                z_bg = F.interpolate(z_bg, size=out_hw, mode="bilinear", align_corners=False)
            return self.bg_embed(z_bg)
        bg_low = self.lowpass_filter.lowpass(bg_hat)
        bg_low = F.interpolate(bg_low, size=out_hw, mode="bilinear", align_corners=False)
        return self.bg_embed(bg_low)

    def merged_condition(
        self,
        cond_s: torch.Tensor,
        bg_hat: torch.Tensor,
        z_bg: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.background_context_source == "none":
            return cond_s
        bg_context = self.background_context(bg_hat, out_hw=cond_s.shape[-2:], z_bg=z_bg)
        if self.condition_merge == "add":
            return cond_s + bg_context
        return torch.cat([cond_s, bg_context], dim=1)

    def training_loss(
        self,
        z_res: torch.Tensor,
        cond_s: torch.Tensor,
        bg_hat: torch.Tensor,
        alpha_hat: torch.Tensor | None = None,
        z_bg: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        cond = self.merged_condition(cond_s, bg_hat, z_bg=z_bg)
        return self.backend.training_loss(z_res, cond, alpha_hat=alpha_hat, loss_type=self.loss_type)

    @torch.no_grad()
    def sample(
        self,
        cond_s: torch.Tensor,
        bg_hat: torch.Tensor,
        x_size: tuple[int, ...],
        steps: int | None = None,
        z_bg: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Paper Eq. (6): sample from C_str plus psi_bg(B_hat), never a target reference.
        cond = self.merged_condition(cond_s, bg_hat, z_bg=z_bg)
        sampler_kwargs = {
            "cond": cond,
            "x_size": x_size,
            "steps": int(steps or self.num_inference_steps),
        }
        if noise is not None:
            sampler_kwargs["noise"] = noise
        return self.backend.sample(**sampler_kwargs)

    @torch.no_grad()
    def sample_with_trace(
        self,
        cond_s: torch.Tensor,
        bg_hat: torch.Tensor,
        x_size: tuple[int, ...],
        steps: int | None = None,
        z_bg: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run a backend sampler and retain its exact Euler trajectory.

        The returned ``condition`` is the merged condition consumed by the
        residual field. For the formal pixel-space FM model this is the
        concatenation ``[C_str; psi_bg(B_hat)]``.
        """
        cond = self.merged_condition(cond_s, bg_hat, z_bg=z_bg)
        tracer = getattr(self.backend, "sample_with_trace", None)
        if tracer is None:
            raise NotImplementedError(
                f"Residual backend {self.backend_name!r} does not expose a traceable Euler sampler."
            )
        return tracer(
            cond,
            x_size=x_size,
            steps=int(steps or self.num_inference_steps),
            noise=noise,
        )
