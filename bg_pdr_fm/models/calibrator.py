"""Residual low-frequency ratio calibration and gating helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from bg_pdr_fm.data.batch import BGInferenceBatch, BGSampleBatch


@dataclass
class ResidualGate:
    rho_hat_b: torch.Tensor
    alpha_hat: torch.Tensor


class RhoCalibrator(nn.Module):
    """Predict inference-time residual low-frequency leakage from visible quality signals."""

    def __init__(self, num_modalities: int = 4, hidden_dim: int = 32, tau: float = 1.0) -> None:
        super().__init__()
        self.tau = float(tau)
        in_dim = num_modalities * 3 + 1
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        reliability: torch.Tensor,
        batch: BGSampleBatch | BGInferenceBatch,
        epsilon_h_b: torch.Tensor,
    ) -> ResidualGate:
        b = reliability.shape[0]
        rel_num = reliability[..., 1]
        visible = torch.cat(
            [
                rel_num.reshape(b, -1),
                batch.modality_mask.reshape(b, -1).to(rel_num.device),
                batch.modality_quality.reshape(b, -1).to(rel_num.device),
                epsilon_h_b.reshape(b, 1).to(rel_num.device),
            ],
            dim=1,
        )
        rho_hat = F.softplus(self.net(visible)).squeeze(1)
        alpha = torch.clamp(1.0 - rho_hat / max(self.tau, 1e-6), min=0.0, max=1.0)
        return ResidualGate(rho_hat_b=rho_hat, alpha_hat=alpha)

    @staticmethod
    def rho_loss(gate: ResidualGate, rho_true: torch.Tensor) -> torch.Tensor:
        return F.l1_loss(gate.rho_hat_b, rho_true.detach())
