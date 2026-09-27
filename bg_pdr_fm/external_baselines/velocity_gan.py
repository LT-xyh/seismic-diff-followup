"""Official OpenFWI VelocityGAN adapted to the multimodal input protocol."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from bg_pdr_fm.data.batch import BGSampleBatch
from bg_pdr_fm.external_baselines.common import build_multimodal_condition_image
from bg_pdr_fm.external_baselines.openfwi_official import (
    OpenFWIDiscriminator,
    OpenFWIInversionNet,
    WassersteinGradientPenalty,
)


class VelocityGANBaseline(nn.Module):
    def __init__(
        self,
        input_hw: tuple[int, int] = (1000, 70),
        l1_weight: float = 100.0,
        mse_weight: float = 0.0,
        adversarial_weight: float = 1.0,
        gradient_penalty_weight: float = 10.0,
        n_critic: int = 5,
    ) -> None:
        super().__init__()
        self.input_hw = tuple(int(value) for value in input_hw)
        if self.input_hw != (1000, 70):
            raise ValueError(f"Official VelocityGAN requires input_hw=(1000, 70), got {self.input_hw}.")
        if int(n_critic) <= 0:
            raise ValueError(f"n_critic must be positive, got {n_critic}.")
        self.generator = OpenFWIInversionNet()
        self.discriminator = OpenFWIDiscriminator()
        self.gradient_penalty = WassersteinGradientPenalty(gradient_penalty_weight)
        self.l1_weight = float(l1_weight)
        self.mse_weight = float(mse_weight)
        self.adversarial_weight = float(adversarial_weight)
        self.n_critic = int(n_critic)

    def forward(self, batch: BGSampleBatch) -> torch.Tensor:
        conditions = build_multimodal_condition_image(batch, self.input_hw)
        return self.generator(conditions)

    def generator_loss(self, batch: BGSampleBatch) -> dict[str, torch.Tensor]:
        fake = self(batch)
        l1 = F.l1_loss(fake, batch.depth_vel)
        mse = F.mse_loss(fake, batch.depth_vel)
        adversarial = -self.discriminator(fake).mean()
        loss = self.l1_weight * l1 + self.mse_weight * mse + self.adversarial_weight * adversarial
        return {"loss": loss, "l1": l1, "mse": mse, "adversarial": adversarial, "velocity_hat": fake}

    def discriminator_loss(self, batch: BGSampleBatch) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            fake = self(batch)
        return self.gradient_penalty(batch.depth_vel, fake.detach(), self.discriminator)

    def validation_loss(self, batch: BGSampleBatch) -> dict[str, torch.Tensor]:
        fake = self(batch)
        l1 = F.l1_loss(fake, batch.depth_vel)
        mse = F.mse_loss(fake, batch.depth_vel)
        loss = self.l1_weight * l1 + self.mse_weight * mse
        return {"loss": loss, "l1": l1, "mse": mse, "velocity_hat": fake}

    def should_update_generator(self, batch_idx: int, num_batches: int | None = None) -> bool:
        periodic = (int(batch_idx) + 1) % self.n_critic == 0
        final_batch = num_batches is not None and int(batch_idx) + 1 == int(num_batches)
        return periodic or final_batch
