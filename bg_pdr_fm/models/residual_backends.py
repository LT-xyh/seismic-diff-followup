"""Residual flow-matching backend abstractions."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualGeneratorBackend(Protocol):
    backend_name: str

    def training_loss(
        self,
        z_res: torch.Tensor,
        cond: torch.Tensor,
        alpha_hat: torch.Tensor | None = None,
        loss_type: str = "mse",
    ) -> dict[str, torch.Tensor]:
        ...

    def sample(self, cond: torch.Tensor, x_size: tuple[int, ...], steps: int) -> torch.Tensor:
        ...

    def sample_with_trace(
        self,
        cond: torch.Tensor,
        x_size: tuple[int, ...],
        steps: int,
        noise: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        ...


class _VelocityField(nn.Module):
    def __init__(self, latent_channels: int = 16, cond_channels: int = 64, hidden_channels: int = 96) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(latent_channels + cond_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=min(8, hidden_channels), num_channels=hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, latent_channels, kernel_size=1),
        )

    def forward(self, z_t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z_t, cond], dim=1))


class _SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = max(self.dim // 2, 1)
        freqs = torch.exp(
            torch.linspace(
                0.0,
                -torch.log(t.new_tensor(10000.0)),
                half,
                device=t.device,
                dtype=t.dtype,
            )
        )
        args = t.float().to(dtype=t.dtype).view(-1, 1) * freqs.view(1, -1)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if emb.shape[1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[1]))
        return emb[:, : self.dim]


class _FiLMResBlock(nn.Module):
    def __init__(self, channels: int, cond_channels: int, time_dim: int) -> None:
        super().__init__()
        groups = min(8, int(channels))
        self.norm1 = nn.GroupNorm(num_groups=groups, num_channels=channels)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=groups, num_channels=channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, channels * 2)
        self.cond_proj = nn.Conv2d(cond_channels, channels * 2, kernel_size=1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        if tuple(cond.shape[-2:]) != tuple(x.shape[-2:]):
            cond = F.interpolate(cond, size=x.shape[-2:], mode="bilinear", align_corners=False)
        gamma_beta = self.cond_proj(cond) + self.time_proj(time_emb).view(x.shape[0], -1, 1, 1)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        h = self.conv1(F.silu(self.norm1(x)))
        h = h * (1.0 + gamma) + beta
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h


class _LatentUNetFiLMField(nn.Module):
    def __init__(self, latent_channels: int, cond_channels: int, hidden_channels: int) -> None:
        super().__init__()
        hidden = int(hidden_channels)
        mid = max(hidden * 2, hidden)
        self.time_embed = nn.Sequential(
            _SinusoidalTimeEmbedding(hidden),
            nn.Linear(hidden, hidden * 4),
            nn.SiLU(),
            nn.Linear(hidden * 4, hidden),
        )
        self.in_proj = nn.Conv2d(latent_channels, hidden, kernel_size=3, padding=1)
        self.down_block = _FiLMResBlock(hidden, cond_channels, hidden)
        self.downsample = nn.Conv2d(hidden, mid, kernel_size=3, stride=2, padding=1)
        self.mid_block = _FiLMResBlock(mid, cond_channels, hidden)
        self.upsample = nn.ConvTranspose2d(mid, hidden, kernel_size=4, stride=2, padding=1)
        self.up_block = _FiLMResBlock(hidden, cond_channels, hidden)
        self.out_norm = nn.GroupNorm(num_groups=min(8, hidden), num_channels=hidden)
        self.out_proj = nn.Conv2d(hidden, latent_channels, kernel_size=3, padding=1)

    def forward(self, z_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        time_emb = self.time_embed(t)
        skip = self.in_proj(z_t)
        h = self.down_block(skip, cond, time_emb)
        h = self.downsample(h)
        h = self.mid_block(h, cond, time_emb)
        h = self.upsample(h)
        if tuple(h.shape[-2:]) != tuple(skip.shape[-2:]):
            h = F.interpolate(h, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        h = self.up_block(h + skip, cond, time_emb)
        return self.out_proj(F.silu(self.out_norm(h)))


def _as_int_tuple(value: Sequence[int] | int | str, *, label: str) -> tuple[int, ...]:
    if isinstance(value, str):
        items = [item.strip() for item in value.strip("[]()").split(",") if item.strip()]
    elif isinstance(value, int):
        items = [value]
    else:
        items = list(value)
    parsed = tuple(int(item) for item in items)
    if not parsed or min(parsed) <= 0:
        raise ValueError(f"{label} must contain positive integers, got {value!r}.")
    return parsed


def _diffusers_unet_condition_blocks(num_blocks: int) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if num_blocks < 1:
        raise ValueError(f"Diffusers UNet requires at least one block, got {num_blocks}.")
    return (
        tuple("CrossAttnDownBlock2D" for _ in range(num_blocks)),
        tuple("CrossAttnUpBlock2D" for _ in range(num_blocks)),
    )


def _weighted_mse_loss_map(
    prediction: torch.Tensor,
    target: torch.Tensor,
    alpha_hat: torch.Tensor | None = None,
) -> torch.Tensor:
    loss_map = (prediction - target) ** 2
    if alpha_hat is not None:
        loss_map = loss_map * (1.0 + alpha_hat.view(-1, 1, 1, 1))
    return loss_map.mean()


class SimpleResidualFMBackend(nn.Module):
    """Small linear Flow Matching backend for smoke runs."""

    backend_name = "simple_fm"

    def __init__(self, latent_channels: int = 16, cond_channels: int = 64, hidden_channels: int = 96) -> None:
        super().__init__()
        self.field = _VelocityField(
            latent_channels=latent_channels,
            cond_channels=cond_channels,
            hidden_channels=hidden_channels,
        )

    def training_loss(
        self,
        z_res: torch.Tensor,
        cond: torch.Tensor,
        alpha_hat: torch.Tensor | None = None,
        loss_type: str = "mse",
    ) -> dict[str, torch.Tensor]:
        if loss_type != "mse":
            raise ValueError(f"simple_fm residual backend only supports loss_type='mse', got {loss_type!r}.")
        batch_size = z_res.shape[0]
        z0 = torch.randn_like(z_res)
        t = torch.rand(batch_size, device=z_res.device, dtype=z_res.dtype)
        t_view = t.view(-1, *([1] * (z_res.ndim - 1)))
        z_t = (1.0 - t_view) * z0 + t_view * z_res
        target = z_res - z0
        pred = self.field(z_t, cond)
        loss = _weighted_mse_loss_map(pred, target, alpha_hat=alpha_hat)
        z_res_pred = z_t + (1.0 - t_view) * pred
        return {
            "loss": loss,
            "velocity_pred": pred,
            "velocity_target": target,
            "z_res_pred": z_res_pred,
            "t": t,
            "x_t": z_t,
            "x0": z0,
        }

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, x_size: tuple[int, ...], steps: int) -> torch.Tensor:
        z_t = torch.randn(x_size, device=cond.device, dtype=cond.dtype)
        dt = 1.0 / float(max(steps, 1))
        for _ in range(max(steps, 1)):
            z_t = z_t + dt * self.field(z_t, cond)
        return z_t


class UNetFiLMResidualFMBackend(nn.Module):
    """Conditional latent Flow Matching backend with U-Net and FiLM conditioning."""

    backend_name = "unet_fm_film"

    def __init__(self, latent_channels: int = 16, cond_channels: int = 64, hidden_channels: int = 128) -> None:
        super().__init__()
        self.field = _LatentUNetFiLMField(
            latent_channels=latent_channels,
            cond_channels=cond_channels,
            hidden_channels=hidden_channels,
        )

    def training_loss(
        self,
        z_res: torch.Tensor,
        cond: torch.Tensor,
        alpha_hat: torch.Tensor | None = None,
        loss_type: str = "mse",
    ) -> dict[str, torch.Tensor]:
        if loss_type != "mse":
            raise ValueError(f"unet_fm_film residual backend only supports loss_type='mse', got {loss_type!r}.")
        batch_size = z_res.shape[0]
        z0 = torch.randn_like(z_res)
        t = torch.rand(batch_size, device=z_res.device, dtype=z_res.dtype)
        t_view = t.view(-1, *([1] * (z_res.ndim - 1)))
        z_t = (1.0 - t_view) * z0 + t_view * z_res
        target = z_res - z0
        pred = self.field(z_t, t, cond)
        loss = _weighted_mse_loss_map(pred, target, alpha_hat=alpha_hat)
        z_res_pred = z_t + (1.0 - t_view) * pred
        return {
            "loss": loss,
            "velocity_pred": pred,
            "velocity_target": target,
            "z_res_pred": z_res_pred,
            "t": t,
            "x_t": z_t,
            "x0": z0,
        }

    @staticmethod
    def _initial_state(
        cond: torch.Tensor,
        x_size: tuple[int, ...],
        noise: torch.Tensor | None,
    ) -> torch.Tensor:
        if noise is None:
            return torch.randn(x_size, device=cond.device, dtype=cond.dtype)
        if tuple(noise.shape) != tuple(x_size):
            raise ValueError(f"noise shape {tuple(noise.shape)} does not match x_size {tuple(x_size)}.")
        if noise.device != cond.device or noise.dtype != cond.dtype:
            raise ValueError(
                "noise must use the same device and dtype as cond; "
                f"got noise=({noise.device}, {noise.dtype}), cond=({cond.device}, {cond.dtype})."
            )
        return noise

    @torch.no_grad()
    def sample(
        self,
        cond: torch.Tensor,
        x_size: tuple[int, ...],
        steps: int,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        z_t = self._initial_state(cond, x_size, noise)
        steps = int(max(steps, 1))
        dt = 1.0 / float(steps)
        for idx in range(steps):
            t_value = z_t.new_full((z_t.shape[0],), float(idx) / float(steps))
            z_t = z_t + dt * self.field(z_t, t_value, cond)
        return z_t

    @torch.no_grad()
    def sample_with_trace(
        self,
        cond: torch.Tensor,
        x_size: tuple[int, ...],
        steps: int,
        noise: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run the production Euler sampler while retaining every FM state."""
        z_t = self._initial_state(cond, x_size, noise)
        steps = int(max(steps, 1))
        dt = 1.0 / float(steps)
        states = [z_t]
        velocities: list[torch.Tensor] = []
        deltas: list[torch.Tensor] = []
        times: list[torch.Tensor] = []
        for idx in range(steps):
            t_value = z_t.new_full((z_t.shape[0],), float(idx) / float(steps))
            velocity = self.field(z_t, t_value, cond)
            delta = dt * velocity
            velocities.append(velocity)
            deltas.append(delta)
            times.append(t_value)
            z_t = z_t + delta
            states.append(z_t)
        return {
            "states": torch.stack(states, dim=0),
            "velocities": torch.stack(velocities, dim=0),
            "deltas": torch.stack(deltas, dim=0),
            "times": torch.stack(times, dim=0),
            "condition": cond,
        }


class DiffusersCrossAttnResidualFMBackend(nn.Module):
    """Diffusers UNet2DConditionModel latent Flow Matching backend.

    The condition map stays spatial: BCHW condition features are projected to
    cross-attention tokens with shape B x (H*W) x D. This keeps the existing
    residual-stage target and sampling contract intact while replacing the
    shallow concatenation/FiLM field with a standard conditional UNet.
    """

    backend_name = "diffusers_unet_crossattn_fm"

    def __init__(
        self,
        latent_channels: int = 4,
        cond_channels: int = 36,
        latent_hw: tuple[int, int] = (18, 18),
        num_train_timesteps: int = 1000,
        hidden_channels: int = 128,
        block_out_channels: Sequence[int] | int | str | None = None,
        layers_per_block: int = 2,
        cross_attention_dim: int | None = None,
        attention_head_dim: int = 8,
    ) -> None:
        super().__init__()
        try:
            from diffusers import UNet2DConditionModel
        except ImportError as exc:  # pragma: no cover - exercised only without optional dependency
            raise ImportError(
                "model.residual_backend='diffusers_unet_crossattn_fm' requires the diffusers package."
            ) from exc

        self.latent_channels = int(latent_channels)
        self.cond_channels = int(cond_channels)
        self.latent_hw = tuple(int(v) for v in latent_hw)
        self.num_train_timesteps = int(num_train_timesteps)
        self.cross_attention_dim = int(cross_attention_dim or hidden_channels)
        blocks = _as_int_tuple(
            block_out_channels if block_out_channels is not None else (hidden_channels, hidden_channels * 2, hidden_channels * 2),
            label="block_out_channels",
        )
        down_blocks, up_blocks = _diffusers_unet_condition_blocks(len(blocks))
        self.cond_token_proj = nn.Conv2d(self.cond_channels, self.cross_attention_dim, kernel_size=1)
        self.unet = UNet2DConditionModel(
            sample_size=self.latent_hw,
            in_channels=self.latent_channels,
            out_channels=self.latent_channels,
            down_block_types=down_blocks,
            up_block_types=up_blocks,
            block_out_channels=blocks,
            layers_per_block=int(layers_per_block),
            cross_attention_dim=self.cross_attention_dim,
            attention_head_dim=int(attention_head_dim),
            norm_num_groups=min(32, blocks[0]),
        )

    def _condition_tokens(self, cond: torch.Tensor, spatial_hw: tuple[int, int]) -> torch.Tensor:
        if cond.ndim != 4:
            raise ValueError(f"diffusers_unet_crossattn_fm expected BCHW cond, got {tuple(cond.shape)}.")
        if cond.shape[1] != self.cond_channels:
            raise ValueError(
                f"diffusers_unet_crossattn_fm expected {self.cond_channels} condition channels, "
                f"got {cond.shape[1]}."
            )
        if tuple(cond.shape[-2:]) != tuple(spatial_hw):
            cond = F.interpolate(cond, size=spatial_hw, mode="bilinear", align_corners=False)
        tokens = self.cond_token_proj(cond)
        return tokens.flatten(2).transpose(1, 2).contiguous()

    def _model_timestep(self, t: torch.Tensor) -> torch.Tensor:
        return t.to(dtype=torch.float32) * float(max(self.num_train_timesteps - 1, 1))

    def _field(self, z_t: torch.Tensor, t: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        tokens = self._condition_tokens(cond, spatial_hw=tuple(z_t.shape[-2:]))
        return self.unet(z_t, self._model_timestep(t), encoder_hidden_states=tokens).sample

    def training_loss(
        self,
        z_res: torch.Tensor,
        cond: torch.Tensor,
        alpha_hat: torch.Tensor | None = None,
        loss_type: str = "mse",
    ) -> dict[str, torch.Tensor]:
        if loss_type != "mse":
            raise ValueError(
                "diffusers_unet_crossattn_fm residual backend only supports "
                f"loss_type='mse', got {loss_type!r}."
            )
        batch_size = z_res.shape[0]
        z0 = torch.randn_like(z_res)
        t = torch.rand(batch_size, device=z_res.device, dtype=z_res.dtype)
        t_view = t.view(-1, *([1] * (z_res.ndim - 1)))
        z_t = (1.0 - t_view) * z0 + t_view * z_res
        target = z_res - z0
        pred = self._field(z_t, t, cond)
        loss = _weighted_mse_loss_map(pred, target, alpha_hat=alpha_hat)
        z_res_pred = z_t + (1.0 - t_view) * pred
        return {
            "loss": loss,
            "velocity_pred": pred,
            "velocity_target": target,
            "z_res_pred": z_res_pred,
            "t": t,
            "x_t": z_t,
            "x0": z0,
        }

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, x_size: tuple[int, ...], steps: int) -> torch.Tensor:
        z_t = torch.randn(x_size, device=cond.device, dtype=cond.dtype)
        steps = int(max(steps, 1))
        dt = 1.0 / float(steps)
        for idx in range(steps):
            t_value = z_t.new_full((z_t.shape[0],), float(idx) / float(steps))
            z_t = z_t + dt * self._field(z_t, t_value, cond)
        return z_t


class SimpleLatentDDPMBackend(nn.Module):
    """Small DDPM-style backend for same-framework residual comparisons."""

    backend_name = "simple_ddpm"

    def __init__(
        self,
        latent_channels: int = 16,
        cond_channels: int = 64,
        hidden_channels: int = 96,
        num_train_timesteps: int = 200,
    ) -> None:
        super().__init__()
        self.noise_model = _VelocityField(
            latent_channels=latent_channels,
            cond_channels=cond_channels,
            hidden_channels=hidden_channels,
        )
        betas = torch.linspace(1e-4, 2e-2, int(num_train_timesteps), dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_bar", alphas_bar)

    def _timestep_cond(self, cond: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # The tiny smoke backend keeps the network interface compatible with FM.
        del t
        return cond

    def training_loss(
        self,
        z_res: torch.Tensor,
        cond: torch.Tensor,
        alpha_hat: torch.Tensor | None = None,
        loss_type: str = "mse",
    ) -> dict[str, torch.Tensor]:
        if loss_type != "mse":
            raise ValueError(f"simple_ddpm backend only supports loss_type='mse', got {loss_type!r}.")
        batch_size = z_res.shape[0]
        t = torch.randint(0, self.alphas_bar.numel(), (batch_size,), device=z_res.device)
        alpha_bar = self.alphas_bar.to(z_res.device, z_res.dtype)[t].view(-1, 1, 1, 1)
        noise = torch.randn_like(z_res)
        x_t = alpha_bar.sqrt() * z_res + (1.0 - alpha_bar).sqrt() * noise
        pred_noise = self.noise_model(x_t, self._timestep_cond(cond, t))
        loss = _weighted_mse_loss_map(pred_noise, noise, alpha_hat=alpha_hat)
        z_res_pred = (x_t - (1.0 - alpha_bar).sqrt() * pred_noise) / alpha_bar.sqrt().clamp_min(1e-6)
        return {
            "loss": loss,
            "velocity_pred": pred_noise,
            "velocity_target": noise,
            "z_res_pred": z_res_pred,
            "t": t,
            "x_t": x_t,
            "x0": noise,
        }

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, x_size: tuple[int, ...], steps: int) -> torch.Tensor:
        steps = int(max(steps, 1))
        device = cond.device
        dtype = cond.dtype
        z_t = torch.randn(x_size, device=device, dtype=dtype)
        total = self.alphas_bar.numel()
        indices = torch.linspace(total - 1, 0, steps, device=device).long()
        betas = self.betas.to(device=device, dtype=dtype)
        alphas = self.alphas.to(device=device, dtype=dtype)
        alphas_bar = self.alphas_bar.to(device=device, dtype=dtype)
        for idx in indices:
            beta_t = betas[idx].view(1, 1, 1, 1)
            alpha_t = alphas[idx].view(1, 1, 1, 1)
            alpha_bar_t = alphas_bar[idx].view(1, 1, 1, 1)
            pred_noise = self.noise_model(z_t, cond)
            mean = (z_t - beta_t * pred_noise / (1.0 - alpha_bar_t).sqrt().clamp_min(1e-6))
            mean = mean / alpha_t.sqrt().clamp_min(1e-6)
            if int(idx.item()) > 0:
                z_t = mean + beta_t.sqrt() * torch.randn_like(z_t)
            else:
                z_t = mean
        return z_t


class DirectResidualPredictorBackend(nn.Module):
    """Deterministic latent residual predictor for teacher-forced diagnostics."""

    backend_name = "direct_predictor"

    def __init__(self, latent_channels: int = 16, cond_channels: int = 64, hidden_channels: int = 96) -> None:
        super().__init__()
        self.field = _VelocityField(
            latent_channels=latent_channels,
            cond_channels=cond_channels,
            hidden_channels=hidden_channels,
        )
        self.latent_channels = int(latent_channels)

    def _zero_latent(self, cond: torch.Tensor, x_size: tuple[int, ...]) -> torch.Tensor:
        return torch.zeros(x_size, device=cond.device, dtype=cond.dtype)

    def training_loss(
        self,
        z_res: torch.Tensor,
        cond: torch.Tensor,
        alpha_hat: torch.Tensor | None = None,
        loss_type: str = "mse",
    ) -> dict[str, torch.Tensor]:
        if loss_type != "mse":
            raise ValueError(f"direct_predictor residual backend only supports loss_type='mse', got {loss_type!r}.")
        z0 = self._zero_latent(cond, tuple(z_res.shape))
        pred = self.field(z0, cond)
        loss = _weighted_mse_loss_map(pred, z_res, alpha_hat=alpha_hat)
        return {
            "loss": loss,
            "velocity_pred": pred,
            "velocity_target": z_res,
            "z_res_pred": pred,
            "t": torch.zeros((z_res.shape[0],), device=z_res.device, dtype=z_res.dtype),
            "x_t": z0,
            "x0": z0,
        }

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, x_size: tuple[int, ...], steps: int) -> torch.Tensor:
        del steps
        return self.field(self._zero_latent(cond, tuple(x_size)), cond)


class LegacyLatentFlowMatchingBackend(nn.Module):
    """Compatibility adapter for externally supplied legacy FM implementations.

    Standalone BG-PDR-FM does not import the old diffusion code automatically.
    Use `simple_fm` for the default package path, or pass `flow_cls` explicitly
    from an external experiment if legacy behavior is required.
    """

    backend_name = "legacy_fm"

    def __init__(
        self,
        latent_channels: int = 16,
        cond_channels: int = 64,
        latent_hw: tuple[int, int] = (16, 16),
        num_train_timesteps: int = 1000,
        path: str = "linear",
        flow_cls: type[Any] | None = None,
    ) -> None:
        super().__init__()
        self._validate_supported_shape(latent_channels, cond_channels, latent_hw)
        if flow_cls is None:
            raise ImportError(
                "legacy_fm is not bundled in standalone bg_pdr_fm. "
                "Use model.residual_backend='simple_fm' or pass `flow_cls` explicitly."
            )
        try:
            self.flow = flow_cls(path=path, num_train_timesteps=int(num_train_timesteps))
        except Exception as exc:
            raise RuntimeError("Failed to construct injected legacy residual FM backend.") from exc

    @staticmethod
    def _validate_supported_shape(
        latent_channels: int,
        cond_channels: int,
        latent_hw: tuple[int, int],
    ) -> None:
        if int(latent_channels) != 16 or int(cond_channels) != 64 or tuple(latent_hw) != (16, 16):
            raise ValueError(
                "legacy_fm residual backend currently requires "
                "model.latent_channels=16, model.cond_channels=64, and model.latent_hw=[16, 16]. "
                f"Got latent_channels={latent_channels}, cond_channels={cond_channels}, latent_hw={tuple(latent_hw)}."
            )

    def _validate_runtime_shapes(self, z_res: torch.Tensor, cond: torch.Tensor) -> None:
        if z_res.ndim != 4 or cond.ndim != 4:
            raise ValueError(
                "legacy_fm expected BCHW tensors for z_res and cond, "
                f"got {tuple(z_res.shape)} and {tuple(cond.shape)}."
            )
        if tuple(z_res.shape[1:]) != (16, 16, 16):
            raise ValueError(f"legacy_fm expected z_res shape Bx16x16x16, got {tuple(z_res.shape)}.")
        if tuple(cond.shape[1:]) != (64, 16, 16):
            raise ValueError(f"legacy_fm expected cond shape Bx64x16x16, got {tuple(cond.shape)}.")

    def training_loss(
        self,
        z_res: torch.Tensor,
        cond: torch.Tensor,
        alpha_hat: torch.Tensor | None = None,
        loss_type: str = "mse",
    ) -> dict[str, torch.Tensor]:
        if loss_type != "mse":
            raise ValueError(f"legacy_fm residual backend only supports loss_type='mse', got {loss_type!r}.")
        self._validate_runtime_shapes(z_res, cond)
        batch_size = z_res.shape[0]
        z0 = torch.randn_like(z_res)
        t = torch.rand(batch_size, device=z_res.device, dtype=z_res.dtype)
        t_view = t.view(-1, *([1] * (z_res.ndim - 1)))
        z_t = (1.0 - t_view) * z0 + t_view * z_res
        target = z_res - z0
        model_t = self.flow._time_to_unet_timestep(t)
        pred = self.flow.velocity_model(z_t, model_t, cond)
        loss = _weighted_mse_loss_map(pred, target, alpha_hat=alpha_hat)
        z_res_pred = z_t + (1.0 - t_view) * pred
        return {
            "loss": loss,
            "velocity_pred": pred,
            "velocity_target": target,
            "z_res_pred": z_res_pred,
            "t": t,
            "x_t": z_t,
            "x0": z0,
        }

    @torch.no_grad()
    def sample(self, cond: torch.Tensor, x_size: tuple[int, ...], steps: int) -> torch.Tensor:
        if tuple(x_size[1:]) != (16, 16, 16):
            raise ValueError(f"legacy_fm expected sample x_size Bx16x16x16, got {tuple(x_size)}.")
        if tuple(cond.shape[1:]) != (64, 16, 16):
            raise ValueError(f"legacy_fm expected cond shape Bx64x16x16, got {tuple(cond.shape)}.")
        return self.flow.sample(cond=cond, x_size=x_size, num_inference_steps=int(steps))


def build_residual_backend(
    backend: str = "simple_fm",
    latent_channels: int = 16,
    cond_channels: int = 64,
    latent_hw: tuple[int, int] = (16, 16),
    num_train_timesteps: int = 1000,
    hidden_channels: int = 96,
    block_out_channels: Sequence[int] | int | str | None = None,
    layers_per_block: int = 2,
    cross_attention_dim: int | None = None,
    attention_head_dim: int = 8,
) -> nn.Module:
    backend = str(backend).lower()
    if backend == "simple_fm":
        return SimpleResidualFMBackend(
            latent_channels=latent_channels,
            cond_channels=cond_channels,
            hidden_channels=hidden_channels,
        )
    if backend == "simple_ddpm":
        return SimpleLatentDDPMBackend(
            latent_channels=latent_channels,
            cond_channels=cond_channels,
            hidden_channels=hidden_channels,
            num_train_timesteps=num_train_timesteps,
        )
    if backend == "unet_fm_film":
        return UNetFiLMResidualFMBackend(
            latent_channels=latent_channels,
            cond_channels=cond_channels,
            hidden_channels=hidden_channels,
        )
    if backend == "direct_predictor":
        return DirectResidualPredictorBackend(
            latent_channels=latent_channels,
            cond_channels=cond_channels,
            hidden_channels=hidden_channels,
        )
    if backend in {"diffusers_unet_crossattn_fm", "diffusers_crossattn_fm"}:
        return DiffusersCrossAttnResidualFMBackend(
            latent_channels=latent_channels,
            cond_channels=cond_channels,
            latent_hw=latent_hw,
            num_train_timesteps=num_train_timesteps,
            hidden_channels=hidden_channels,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            cross_attention_dim=cross_attention_dim,
            attention_head_dim=attention_head_dim,
        )
    if backend == "legacy_fm":
        return LegacyLatentFlowMatchingBackend(
            latent_channels=latent_channels,
            cond_channels=cond_channels,
            latent_hw=latent_hw,
            num_train_timesteps=num_train_timesteps,
        )
    raise ValueError(
        f"Unknown model.residual_backend: {backend!r}. "
        "Expected 'simple_fm', 'simple_ddpm', 'unet_fm_film', 'direct_predictor', "
        "'diffusers_unet_crossattn_fm', or 'legacy_fm'."
    )
