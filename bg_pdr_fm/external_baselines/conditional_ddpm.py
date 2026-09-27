"""Diffusers conditional DDPM baseline for multimodal velocity prediction."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDPMScheduler, UNet2DConditionModel

from bg_pdr_fm.data.batch import BGSampleBatch
from bg_pdr_fm.external_baselines.common import build_multimodal_condition_image


class ConditionTokenEncoder(nn.Module):
    def __init__(
        self,
        condition_dim: int = 256,
        input_hw: tuple[int, int] = (1000, 70),
        num_tokens: int = 16,
    ) -> None:
        super().__init__()
        self.input_hw = tuple(int(v) for v in input_hw)
        self.num_tokens = int(num_tokens)
        self.net = nn.Sequential(
            nn.Conv2d(5, 32, kernel_size=7, stride=(4, 2), padding=3),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=(4, 2), padding=1),
            nn.SiLU(),
            nn.Conv2d(64, int(condition_dim), kernel_size=3, stride=(4, 2), padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((self.num_tokens, 1)),
        )

    def forward(self, batch: BGSampleBatch) -> torch.Tensor:
        x = build_multimodal_condition_image(batch, self.input_hw)
        return self.net(x).squeeze(-1).transpose(1, 2).contiguous()


class ConditionalDDPMBaseline(nn.Module):
    """Conditional DDPM with multimodal condition tokens as cross-attention context."""

    def __init__(
        self,
        sample_size: int = 70,
        condition_input_hw: tuple[int, int] = (1000, 70),
        condition_dim: int = 256,
        num_condition_tokens: int = 16,
        train_timesteps: int = 1000,
        inference_steps: int = 50,
        block_out_channels: tuple[int, ...] = (64, 128, 256),
    ) -> None:
        super().__init__()
        self.condition_encoder = ConditionTokenEncoder(condition_dim, condition_input_hw, num_condition_tokens)
        self.scheduler = DDPMScheduler(num_train_timesteps=int(train_timesteps), prediction_type="epsilon")
        self.inference_steps = int(inference_steps)
        blocks = tuple(int(v) for v in block_out_channels)
        self.unet = UNet2DConditionModel(
            sample_size=int(sample_size),
            in_channels=1,
            out_channels=1,
            layers_per_block=2,
            block_out_channels=blocks,
            down_block_types=("CrossAttnDownBlock2D",) * len(blocks),
            up_block_types=("CrossAttnUpBlock2D",) * len(blocks),
            cross_attention_dim=int(condition_dim),
            norm_num_groups=8,
        )

    def training_loss(self, batch: BGSampleBatch) -> dict[str, torch.Tensor]:
        clean = batch.depth_vel
        noise = torch.randn_like(clean)
        timesteps = torch.randint(
            0,
            int(self.scheduler.config.num_train_timesteps),
            (clean.shape[0],),
            device=clean.device,
            dtype=torch.long,
        )
        noisy = self.scheduler.add_noise(clean, noise, timesteps)
        pred = self.unet(noisy, timesteps, encoder_hidden_states=self.condition_encoder(batch)).sample
        loss = F.mse_loss(pred, noise)
        return {"loss": loss, "noise_mse": loss}

    @torch.no_grad()
    def sample(self, batch: BGSampleBatch) -> torch.Tensor:
        sample = torch.randn_like(batch.depth_vel)
        condition = self.condition_encoder(batch)
        self.scheduler.set_timesteps(self.inference_steps, device=sample.device)
        for timestep in self.scheduler.timesteps:
            model_out = self.unet(sample, timestep, encoder_hidden_states=condition).sample
            sample = self.scheduler.step(model_out, timestep, sample).prev_sample
        return torch.clamp(sample, -1.0, 1.0)
