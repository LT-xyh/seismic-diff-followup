"""AAAI27 benchmark variants built on the BG-PDR-FM data/model stack."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import lightning
import torch
import torch.nn as nn
import torch.nn.functional as F

from bg_pdr_fm.data.batch import BGSampleBatch, collate_bg_samples
from bg_pdr_fm.external_baselines.auto_linear.adapted_multimodal import AdaptedAutoLinearMultimodal
from bg_pdr_fm.external_baselines.auto_linear.original_architecture import AdaptedAutoLinearOriginalMultimodal
from bg_pdr_fm.external_baselines.conditional_ddpm import ConditionalDDPMBaseline
from bg_pdr_fm.external_baselines.gfi.adapted_latent_unet import AdaptedGFILatentUNetMultimodal
from bg_pdr_fm.external_baselines.gfi.adapted_multimodal import AdaptedGFIMultimodal
from bg_pdr_fm.external_baselines.inversion_net import AdaptedInversionNetMultimodal
from bg_pdr_fm.external_baselines.smooth_dix import SmoothDixBaseline
from bg_pdr_fm.external_baselines.sv_inv_net import SVInvNetMultimodal
from bg_pdr_fm.external_baselines.upfwi import AdaptedUPFWIMultimodal
from bg_pdr_fm.external_baselines.velocity_gan import VelocityGANBaseline
from bg_pdr_fm.lightning.stage_losses import conf_get, true_residual_low_frequency_ratio
from bg_pdr_fm.models import (
    BackgroundEstimator,
    ConditionAdapters,
    LowHighPassFilter,
    PhysicsDecoupledEncoder,
    PredictionBatch,
    ResidualFlowGenerator,
    RhoCalibrator,
    build_latent_codec,
)


AAAI27_VARIANTS = (
    "smooth_dix",
    "adapted_inversion_net",
    "adapted_upfwi",
    "sv_inv_net",
    "velocity_gan",
    "conditional_ddpm",
    "mm_invnet",
    "concat_fm",
    "cncs_fm",
    "two_stage_ddpm",
    "bg_pdr_fm",
    "adapted_gfi",
    "adapted_gfi_latent_unet",
    "pdr_gfi_aligned_direct",
    "adapted_auto_linear",
    "adapted_auto_linear_original",
)


def _count_parameters(module: torch.nn.Module, trainable_only: bool = False) -> int:
    parameters = module.parameters()
    if trainable_only:
        return int(sum(parameter.numel() for parameter in parameters if parameter.requires_grad))
    return int(sum(parameter.numel() for parameter in parameters))


class _ImageDecoder(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 64, out_channels: int = 1) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=min(8, hidden_channels), num_channels=hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, out_hw: tuple[int, int]) -> torch.Tensor:
        x = F.interpolate(x, size=out_hw, mode="bilinear", align_corners=False)
        return torch.tanh(self.net(x))


class _ConditionProjector(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, out_hw: tuple[int, int]) -> None:
        super().__init__()
        self.out_hw = tuple(out_hw)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=min(8, hidden_channels), num_channels=hidden_channels),
            nn.SiLU(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=self.out_hw, mode="bilinear", align_corners=False)
        return self.net(x)


class AAAI27BenchmarkLightning(lightning.LightningModule):
    """Controlled benchmark layer for AAAI27 comparisons.

    This module intentionally shares BG-PDR-FM's batch schema, modality encoder,
    codec, low/high-pass filter, and prediction container across all variants.
    """

    def __init__(self, conf: Any | None = None) -> None:
        super().__init__()
        self.conf = conf or {}
        self.variant = str(conf_get(self.conf, "benchmark.variant", "bg_pdr_fm")).lower()
        if self.variant not in AAAI27_VARIANTS:
            raise ValueError(f"benchmark.variant must be one of {AAAI27_VARIANTS}, got {self.variant!r}.")
        self.capacity_tier = str(conf_get(self.conf, "benchmark.capacity_tier", "matched")).lower()
        self.automatic_optimization = self.variant != "velocity_gan"
        self.lr = float(conf_get(self.conf, "training.lr", 1e-4))
        self.weight_decay = float(conf_get(self.conf, "training.weight_decay", 0.0))
        self.scheduler_name = str(conf_get(self.conf, "training.scheduler", "none")).lower()
        self.max_epochs = int(conf_get(self.conf, "training.max_epochs", 1))
        self.freeze_encoder = bool(conf_get(self.conf, "training.freeze_encoder", False))
        self.freeze_background_branch = bool(conf_get(self.conf, "benchmark.freeze_background_branch", False))
        self.load_stage_checkpoint = conf_get(self.conf, "training.load_stage_checkpoint", None)
        self.stage_checkpoint_used = ""
        self.filter = LowHighPassFilter(kernel_size=int(conf_get(self.conf, "model.lowpass_kernel", 5)))
        feature_channels = int(conf_get(self.conf, "model.feature_channels", 16))
        cond_channels = int(conf_get(self.conf, "model.cond_channels", 32))
        latent_channels = int(conf_get(self.conf, "model.latent_channels", 4))
        latent_hw = tuple(conf_get(self.conf, "model.latent_hw", (18, 18)))
        generator_hidden = int(conf_get(self.conf, "benchmark.generator_hidden_channels", 96))
        direct_hidden = int(conf_get(self.conf, "benchmark.direct_hidden_channels", generator_hidden))
        adapted_inversion_net_input_hw = tuple(conf_get(self.conf, "benchmark.adapted_inversion_net_input_hw", (1000, 70)))
        adapted_upfwi_input_hw = tuple(conf_get(self.conf, "benchmark.adapted_upfwi_input_hw", (1000, 70)))
        sv_inv_net_input_hw = tuple(conf_get(self.conf, "benchmark.sv_inv_net_input_hw", (1000, 70)))
        adapted_gfi_base = int(conf_get(self.conf, "benchmark.adapted_gfi_base_channels", 32))
        adapted_gfi_input_hw = tuple(conf_get(self.conf, "benchmark.adapted_gfi_input_hw", (1000, 70)))
        adapted_gfi_latent_input_hw = tuple(conf_get(self.conf, "benchmark.adapted_gfi_latent_input_hw", (1000, 70)))
        adapted_gfi_latent_hw = tuple(conf_get(self.conf, "benchmark.adapted_gfi_latent_hw", (70, 70)))
        self.auto_linear_phase = str(conf_get(self.conf, "benchmark.auto_linear_phase", "inverse_linear")).lower()
        if self.auto_linear_phase not in {"ae_pretrain", "inverse_linear"}:
            raise ValueError("benchmark.auto_linear_phase must be 'ae_pretrain' or 'inverse_linear'.")
        self.auto_linear_supervised_loss = str(
            conf_get(self.conf, "benchmark.auto_linear_supervised_loss", "latent_mse_l1")
        ).lower()
        auto_linear_input_hw = tuple(conf_get(self.conf, "benchmark.auto_linear_input_hw", (1000, 70)))
        self.residual_num_inference_steps = int(conf_get(self.conf, "model.residual_num_inference_steps", 20))
        full_field_context_source = str(
            conf_get(self.conf, "benchmark.full_field_background_context_source", "none")
        )
        full_field_condition_merge = str(
            conf_get(self.conf, "benchmark.full_field_condition_merge", "add")
        )

        self.encoder = PhysicsDecoupledEncoder(
            hidden_channels=int(conf_get(self.conf, "model.hidden_channels", 16)),
            base_channels=int(conf_get(self.conf, "model.base_channels", 32)),
            feature_channels=feature_channels,
        )
        self.codec = build_latent_codec(
            codec_type=str(conf_get(self.conf, "model.codec_type", "simple")),
            latent_channels=latent_channels,
            latent_hw=latent_hw,
            checkpoint_path=conf_get(self.conf, "model.codec_checkpoint", None),
            module_path=conf_get(self.conf, "model.codec_module_path", None),
        )
        self.mixed_projector = _ConditionProjector(feature_channels, direct_hidden, cond_channels, latent_hw)
        self.adapters = ConditionAdapters(
            feature_channels=feature_channels,
            cond_channels=cond_channels,
            out_hw=latent_hw,
            use_unique=bool(conf_get(self.conf, "model.use_unique", False)),
            unique_eta=float(conf_get(self.conf, "model.unique_eta", 0.1)),
        )
        self.mm_decoder = _ImageDecoder(feature_channels, hidden_channels=direct_hidden)
        self.smooth_dix = SmoothDixBaseline(
            kernel_size=int(conf_get(self.conf, "benchmark.smooth_dix_kernel_size", 9)),
        )
        self.adapted_inversion_net = AdaptedInversionNetMultimodal(input_hw=adapted_inversion_net_input_hw)
        self.adapted_upfwi = AdaptedUPFWIMultimodal(input_hw=adapted_upfwi_input_hw)
        self.sv_inv_net = SVInvNetMultimodal(
            base_channels=int(conf_get(self.conf, "benchmark.sv_inv_net_base_channels", 32)),
            growth_rate=int(conf_get(self.conf, "benchmark.sv_inv_net_growth_rate", 16)),
            dense_layers=int(conf_get(self.conf, "benchmark.sv_inv_net_dense_layers", 4)),
            input_hw=sv_inv_net_input_hw,
        )
        ddpm_blocks = tuple(int(v) for v in conf_get(self.conf, "benchmark.ddpm_block_out_channels", (64, 128, 256)))
        self.conditional_ddpm = ConditionalDDPMBaseline(
            sample_size=int(conf_get(self.conf, "benchmark.ddpm_sample_size", 70)),
            condition_input_hw=tuple(conf_get(self.conf, "benchmark.ddpm_condition_input_hw", (1000, 70))),
            condition_dim=int(conf_get(self.conf, "benchmark.ddpm_condition_dim", 256)),
            num_condition_tokens=int(conf_get(self.conf, "benchmark.ddpm_num_condition_tokens", 16)),
            train_timesteps=int(conf_get(self.conf, "model.residual_num_train_timesteps", 1000)),
            inference_steps=int(conf_get(self.conf, "benchmark.ddpm_inference_steps", 50)),
            block_out_channels=ddpm_blocks,
        )
        self.velocity_gan = VelocityGANBaseline(
            input_hw=tuple(conf_get(self.conf, "benchmark.velocity_gan_input_hw", (1000, 70))),
            l1_weight=float(conf_get(self.conf, "benchmark.velocity_gan_l1_weight", 100.0)),
            mse_weight=float(conf_get(self.conf, "benchmark.velocity_gan_mse_weight", 0.0)),
            adversarial_weight=float(conf_get(self.conf, "benchmark.velocity_gan_adversarial_weight", 1.0)),
            gradient_penalty_weight=float(conf_get(self.conf, "benchmark.velocity_gan_gradient_penalty_weight", 10.0)),
            n_critic=int(conf_get(self.conf, "benchmark.velocity_gan_n_critic", 5)),
        )
        self.velocity_gan_lr_g = float(conf_get(self.conf, "benchmark.velocity_gan_lr_g", 1e-4))
        self.velocity_gan_lr_d = float(conf_get(self.conf, "benchmark.velocity_gan_lr_d", 1e-4))
        self.adapted_gfi = AdaptedGFIMultimodal(base_channels=adapted_gfi_base, input_hw=adapted_gfi_input_hw)
        self.adapted_gfi_latent_unet = AdaptedGFILatentUNetMultimodal(
            input_hw=adapted_gfi_latent_input_hw,
            latent_hw=adapted_gfi_latent_hw,
            latent_channels=int(conf_get(self.conf, "benchmark.adapted_gfi_latent_channels", 128)),
            unet_depth=int(conf_get(self.conf, "benchmark.adapted_gfi_unet_depth", 2)),
            unet_repeat_blocks=int(conf_get(self.conf, "benchmark.adapted_gfi_unet_repeat_blocks", 2)),
            skip=bool(conf_get(self.conf, "benchmark.adapted_gfi_unet_skip", True)),
        )
        self.adapted_auto_linear = AdaptedAutoLinearMultimodal(
            embed_dim=int(conf_get(self.conf, "benchmark.auto_linear_embed_dim", 256)),
            depth=int(conf_get(self.conf, "benchmark.auto_linear_depth", 4)),
            num_heads=int(conf_get(self.conf, "benchmark.auto_linear_num_heads", 8)),
            latent_tokens=int(conf_get(self.conf, "benchmark.auto_linear_latent_tokens", 16)),
            converter_rank=int(conf_get(self.conf, "benchmark.auto_linear_converter_rank", 64)),
            input_hw=auto_linear_input_hw,
            measurement_patch_size=tuple(conf_get(self.conf, "benchmark.auto_linear_measurement_patch_size", (20, 10))),
            velocity_patch_size=tuple(conf_get(self.conf, "benchmark.auto_linear_velocity_patch_size", (10, 10))),
            mask_ratio=float(conf_get(self.conf, "benchmark.auto_linear_mask_ratio", 0.5)),
        )
        self.adapted_auto_linear_original = AdaptedAutoLinearOriginalMultimodal(
            input_hw=auto_linear_input_hw,
            mask_ratio=float(conf_get(self.conf, "benchmark.auto_linear_mask_ratio", 0.75)),
            converter_rank=int(conf_get(self.conf, "benchmark.auto_linear_converter_rank", 128)),
        )
        self.background = BackgroundEstimator(
            cond_channels=cond_channels,
            latent_channels=latent_channels,
            highpass_filter=self.filter,
            output_activation=str(conf_get(self.conf, "model.background_output_activation", "tanh")),
            hidden_channels=int(conf_get(self.conf, "model.background_hidden_channels", cond_channels)),
            output_mode=str(conf_get(self.conf, "model.background_output_mode", "direct")),
        )
        residual_backend = str(conf_get(self.conf, "model.residual_backend", "simple_fm"))
        if self.variant == "two_stage_ddpm":
            residual_backend = "simple_ddpm"
        self.full_field = ResidualFlowGenerator(
            cond_channels=cond_channels,
            latent_channels=latent_channels,
            latent_hw=latent_hw,
            backend=str(conf_get(self.conf, "benchmark.full_field_backend", "simple_fm")),
            num_train_timesteps=int(conf_get(self.conf, "model.residual_num_train_timesteps", 1000)),
            num_inference_steps=self.residual_num_inference_steps,
            loss_type=str(conf_get(self.conf, "model.residual_loss_type", "mse")),
            lowpass_filter=self.filter,
            backend_hidden_channels=generator_hidden,
            condition_merge=full_field_condition_merge,
            background_context_source=full_field_context_source,
        )
        self.residual = ResidualFlowGenerator(
            cond_channels=cond_channels,
            latent_channels=latent_channels,
            latent_hw=latent_hw,
            backend=residual_backend,
            num_train_timesteps=int(conf_get(self.conf, "model.residual_num_train_timesteps", 1000)),
            num_inference_steps=self.residual_num_inference_steps,
            loss_type=str(conf_get(self.conf, "model.residual_loss_type", "mse")),
            lowpass_filter=self.filter,
            backend_hidden_channels=generator_hidden,
            background_context_source=str(conf_get(self.conf, "model.residual_background_context_source", "lowpass_embed")),
        )
        self.rho_calibrator = RhoCalibrator(tau=float(conf_get(self.conf, "model.rho_tau", 1.0)))
        for parameter in self.codec.parameters():
            parameter.requires_grad = False
        self._load_stage_checkpoint_if_requested()
        self._apply_variant_trainability()

    @staticmethod
    def _set_requires_grad(module: torch.nn.Module, requires_grad: bool) -> None:
        for parameter in module.parameters():
            parameter.requires_grad = requires_grad

    def _apply_variant_trainability(self) -> None:
        self._set_requires_grad(self.encoder, not self.freeze_encoder)
        self._set_requires_grad(self.codec, False)
        self._set_requires_grad(self.mixed_projector, False)
        self._set_requires_grad(self.adapters, False)
        self._set_requires_grad(self.mm_decoder, False)
        self._set_requires_grad(self.smooth_dix, False)
        self._set_requires_grad(self.adapted_inversion_net, False)
        self._set_requires_grad(self.adapted_upfwi, False)
        self._set_requires_grad(self.sv_inv_net, False)
        self._set_requires_grad(self.conditional_ddpm, False)
        self._set_requires_grad(self.velocity_gan, False)
        self._set_requires_grad(self.adapted_gfi, False)
        self._set_requires_grad(self.adapted_gfi_latent_unet, False)
        self._set_requires_grad(self.adapted_auto_linear, False)
        self._set_requires_grad(self.adapted_auto_linear_original, False)
        self._set_requires_grad(self.background, False)
        self._set_requires_grad(self.full_field, False)
        self._set_requires_grad(self.residual, False)
        self._set_requires_grad(self.rho_calibrator, False)

        if self.variant == "smooth_dix":
            pass
        elif self.variant == "adapted_inversion_net":
            self._set_requires_grad(self.adapted_inversion_net, True)
        elif self.variant == "adapted_upfwi":
            self._set_requires_grad(self.adapted_upfwi, True)
        elif self.variant == "sv_inv_net":
            self._set_requires_grad(self.sv_inv_net, True)
        elif self.variant == "conditional_ddpm":
            self._set_requires_grad(self.conditional_ddpm, True)
        elif self.variant == "velocity_gan":
            self._set_requires_grad(self.velocity_gan, True)
        elif self.variant == "mm_invnet":
            self._set_requires_grad(self.mm_decoder, True)
        elif self.variant in {"adapted_gfi", "pdr_gfi_aligned_direct"}:
            self._set_requires_grad(self.adapted_gfi, True)
        elif self.variant == "adapted_gfi_latent_unet":
            self._set_requires_grad(self.adapted_gfi_latent_unet, True)
        elif self.variant == "adapted_auto_linear":
            if self.auto_linear_phase == "ae_pretrain":
                self._set_requires_grad(self.adapted_auto_linear.measurement_mae, True)
                self._set_requires_grad(self.adapted_auto_linear.velocity_mae, True)
            else:
                self._set_requires_grad(self.adapted_auto_linear.latent_converter, True)
        elif self.variant == "adapted_auto_linear_original":
            if self.auto_linear_phase == "ae_pretrain":
                self._set_requires_grad(self.adapted_auto_linear_original.measurement_mae, True)
                self._set_requires_grad(self.adapted_auto_linear_original.velocity_mae, True)
            else:
                self._set_requires_grad(self.adapted_auto_linear_original.latent_converter, True)
        elif self.variant == "concat_fm":
            self._set_requires_grad(self.mixed_projector, True)
            self._set_requires_grad(self.full_field, True)
        elif self.variant == "cncs_fm":
            self._set_requires_grad(self.adapters, True)
            self._set_requires_grad(self.full_field, True)
        else:
            self._set_requires_grad(self.adapters, True)
            self._set_requires_grad(self.background, True)
            self._set_requires_grad(self.residual, True)
            train_gate = str(conf_get(self.conf, "quality_gate.mode", "none")).lower() == "learned"
            self._set_requires_grad(self.rho_calibrator, train_gate)
            if self.freeze_background_branch:
                self._set_requires_grad(self.adapters.numerical, False)
                self._set_requires_grad(self.adapters.unique, False)
                self._set_requires_grad(self.background, False)

    def _load_stage_checkpoint_if_requested(self) -> None:
        if self.load_stage_checkpoint is None or str(self.load_stage_checkpoint).strip() == "":
            return
        path = Path(str(self.load_stage_checkpoint))
        if not path.is_file():
            raise FileNotFoundError(f"AAAI27 benchmark stage checkpoint not found: {path}")
        state = torch.load(path, map_location="cpu")
        state_dict = state.get("state_dict", state) if isinstance(state, dict) else state
        if not isinstance(state_dict, dict):
            raise KeyError(f"AAAI27 benchmark checkpoint {path} does not contain a state dict.")
        self.load_state_dict(state_dict, strict=False)
        self.stage_checkpoint_used = str(path)

    def _prepare_frozen_modules_for_step(self) -> None:
        if self.freeze_encoder:
            self.encoder.eval()
        if self.variant == "adapted_auto_linear" and self.auto_linear_phase == "inverse_linear":
            self.adapted_auto_linear.measurement_mae.eval()
            self.adapted_auto_linear.velocity_mae.eval()
        if self.variant == "adapted_auto_linear_original" and self.auto_linear_phase == "inverse_linear":
            self.adapted_auto_linear_original.measurement_mae.eval()
            self.adapted_auto_linear_original.velocity_mae.eval()
        if self.freeze_background_branch:
            self.adapters.numerical.eval()
            self.adapters.unique.eval()
            self.background.eval()
        self.codec.eval()

    def on_train_epoch_start(self) -> None:
        self._prepare_frozen_modules_for_step()

    @staticmethod
    def _as_batch(batch: BGSampleBatch | dict[str, torch.Tensor]) -> BGSampleBatch:
        if isinstance(batch, BGSampleBatch):
            return batch
        return collate_bg_samples([batch]) if batch["depth_vel"].ndim == 3 else BGSampleBatch(
            depth_vel=batch["depth_vel"],
            migrated_image=batch["migrated_image"],
            horizon=batch["horizon"],
            rms_vel=batch["rms_vel"],
            well_log=batch["well_log"],
            well_mask=batch["well_mask"],
            modality_mask=batch["modality_mask"],
            modality_quality=batch["modality_quality"],
            metadata={k: v for k, v in batch.items() if k not in {
                "depth_vel", "migrated_image", "horizon", "rms_vel", "well_log", "well_mask",
                "modality_mask", "modality_quality"
            }},
        )

    def _conditions(self, batch: BGSampleBatch):
        self._prepare_frozen_modules_for_step()
        features = self.encoder(batch)
        conditions = self.adapters(features)
        return features, conditions

    def _direct_condition(self, features, conditions) -> torch.Tensor:
        if self.variant == "concat_fm":
            if features.mixed is None:
                raise RuntimeError("Concat-FM requires encoder mixed/F_base features.")
            return self.mixed_projector(features.mixed)
        if self.variant == "cncs_fm":
            return conditions.numerical + conditions.structural
        raise RuntimeError(f"Variant {self.variant!r} does not use direct full-field FM.")

    def _zero_bg(self, target: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(target)

    def _log_epoch_metric(self, name: str, value: torch.Tensor, *, prog_bar: bool = False) -> None:
        try:
            self.trainer
        except RuntimeError:
            return
        self.log(name, value, on_step=False, on_epoch=True, prog_bar=prog_bar)

    def _direct_full_field_loss(self, batch: BGSampleBatch, cond: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z_full = self.codec.encode(batch.depth_vel, deterministic=True).detach()
        zero_bg = torch.zeros_like(batch.depth_vel)
        fm = self.full_field.training_loss(z_full, cond, zero_bg, alpha_hat=None)
        recon = self.codec.decode(fm["z_res_pred"], out_hw=batch.depth_vel.shape[-2:])
        loss = fm["loss"] + float(conf_get(self.conf, "loss.final_l1_weight", 1.0)) * F.l1_loss(recon, batch.depth_vel)
        return loss, recon

    def _two_stage_loss(self, batch: BGSampleBatch, conditions, features) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bg = self.background(conditions.numerical, out_hw=batch.depth_vel.shape[-2:])
        gate = self.rho_calibrator(features.reliability, batch, bg.epsilon_h_b)
        z_full = self.codec.encode(batch.depth_vel, deterministic=True).detach()
        z_bg = self.codec.encode(bg.bg_hat.detach(), deterministic=True).detach()
        z_res = (z_full - z_bg).detach()
        alpha = gate.alpha_hat if str(conf_get(self.conf, "quality_gate.mode", "none")).lower() == "learned" else None
        fm = self.residual.training_loss(z_res, conditions.structural, bg.bg_hat.detach(), alpha_hat=alpha, z_bg=z_bg)
        recon = self.codec.decode(z_bg + fm["z_res_pred"], out_hw=batch.depth_vel.shape[-2:])
        low, _ = self.filter(batch.depth_vel)
        bg_l1 = F.l1_loss(bg.bg_hat, low)
        rho_true = true_residual_low_frequency_ratio(self.filter, batch.depth_vel, bg.bg_hat.detach())
        res_pred = recon - bg.bg_hat.detach()
        res_target = batch.depth_vel - bg.bg_hat.detach()
        target_lf_loss = F.l1_loss(self.filter.lowpass(res_pred), self.filter.lowpass(res_target))
        loss = (
            fm["loss"]
            + float(conf_get(self.conf, "loss.final_l1_weight", 1.0)) * F.l1_loss(recon, batch.depth_vel)
            + float(conf_get(self.conf, "loss.bg_weight", 1.0)) * bg_l1
            + float(conf_get(self.conf, "loss.bg_high_weight", 0.05)) * bg.epsilon_h_b.mean()
            + float(conf_get(self.conf, "loss.target_lf_weight", 0.1)) * target_lf_loss
            + float(conf_get(self.conf, "loss.rho_weight", 0.0)) * self.rho_calibrator.rho_loss(gate, rho_true)
        )
        return loss, recon, bg.bg_hat

    def _step(self, batch: BGSampleBatch | dict[str, torch.Tensor], prefix: str) -> torch.Tensor:
        batch = self._as_batch(batch)
        if self.variant == "smooth_dix":
            recon = self.smooth_dix(batch)
            loss = F.l1_loss(recon, batch.depth_vel)
            self._log_epoch_metric(f"{prefix}/loss", loss, prog_bar=True)
            self._log_epoch_metric(f"{prefix}/mae", loss)
            return loss
        if self.variant == "adapted_inversion_net":
            recon = self.adapted_inversion_net(batch)
            l1 = F.l1_loss(recon, batch.depth_vel)
            self._log_epoch_metric(f"{prefix}/loss", l1, prog_bar=True)
            self._log_epoch_metric(f"{prefix}/mae", l1)
            return l1
        if self.variant == "adapted_upfwi":
            recon = self.adapted_upfwi(batch)
            l1 = F.l1_loss(recon, batch.depth_vel)
            self._log_epoch_metric(f"{prefix}/loss", l1, prog_bar=True)
            self._log_epoch_metric(f"{prefix}/mae", l1)
            return l1
        if self.variant == "sv_inv_net":
            recon = self.sv_inv_net(batch)
            l1 = F.l1_loss(recon, batch.depth_vel)
            low_l1 = F.l1_loss(self.filter.lowpass(recon), self.filter.lowpass(batch.depth_vel))
            loss = l1 + float(conf_get(self.conf, "loss.ssim_weight", 0.1)) * low_l1
            self._log_epoch_metric(f"{prefix}/loss", loss, prog_bar=True)
            self._log_epoch_metric(f"{prefix}/mae", l1)
            self._log_epoch_metric(f"{prefix}/low_l1_proxy", low_l1)
            return loss
        if self.variant == "conditional_ddpm":
            losses = self.conditional_ddpm.training_loss(batch)
            loss = losses["loss"]
            self._log_epoch_metric(f"{prefix}/loss", loss, prog_bar=True)
            self._log_epoch_metric(f"{prefix}/noise_mse", losses["noise_mse"])
            return loss
        if self.variant == "velocity_gan":
            losses = self.velocity_gan.validation_loss(batch)
            self._log_epoch_metric(f"{prefix}/loss", losses["loss"], prog_bar=True)
            self._log_epoch_metric(f"{prefix}/mae", losses["l1"])
            self._log_epoch_metric(f"{prefix}/mse", losses["mse"])
            return losses["loss"]
        if self.variant == "adapted_gfi":
            recon = self.adapted_gfi(batch)
            loss = F.l1_loss(recon, batch.depth_vel)
            self._log_epoch_metric(f"{prefix}/loss", loss, prog_bar=True)
            self._log_epoch_metric(f"{prefix}/mae", loss)
            return loss
        if self.variant == "adapted_gfi_latent_unet":
            recon = self.adapted_gfi_latent_unet(batch)
            loss = F.l1_loss(recon, batch.depth_vel)
            self._log_epoch_metric(f"{prefix}/loss", loss, prog_bar=True)
            self._log_epoch_metric(f"{prefix}/mae", loss)
            return loss
        if self.variant == "pdr_gfi_aligned_direct":
            recon = self.adapted_gfi(batch)
            l1 = F.l1_loss(recon, batch.depth_vel)
            l2 = F.mse_loss(recon, batch.depth_vel)
            low_l1 = F.l1_loss(self.filter.lowpass(recon), self.filter.lowpass(batch.depth_vel))
            loss = (
                float(conf_get(self.conf, "loss.final_l1_weight", 1.0)) * l1
                + float(conf_get(self.conf, "loss.final_l2_weight", 0.5)) * l2
                + float(conf_get(self.conf, "loss.bg_weight", 0.1)) * low_l1
            )
            self._log_epoch_metric(f"{prefix}/loss", loss, prog_bar=True)
            self._log_epoch_metric(f"{prefix}/mae", l1)
            self._log_epoch_metric(f"{prefix}/mse", l2)
            self._log_epoch_metric(f"{prefix}/low_l1_proxy", low_l1)
            return loss
        if self.variant in {"adapted_auto_linear", "adapted_auto_linear_original"}:
            auto_linear = (
                self.adapted_auto_linear_original
                if self.variant == "adapted_auto_linear_original"
                else self.adapted_auto_linear
            )
            if self.auto_linear_phase == "ae_pretrain":
                losses = auto_linear.ae_pretrain_loss(batch)
                self._log_epoch_metric(f"{prefix}/measurement_l1", losses["measurement_l1"])
            else:
                losses = auto_linear.inverse_linear_loss(batch, supervised_loss=self.auto_linear_supervised_loss)
                self._log_epoch_metric(f"{prefix}/latent_mse", losses["latent_mse"])
            loss = losses["loss"]
            self._log_epoch_metric(f"{prefix}/loss", loss, prog_bar=True)
            self._log_epoch_metric(f"{prefix}/mae", F.l1_loss(losses["velocity_hat"], batch.depth_vel))
            self._log_epoch_metric(f"{prefix}/velocity_l1", losses["velocity_l1"])
            return loss
        features, conditions = self._conditions(batch)
        if self.variant == "mm_invnet":
            if features.mixed is None:
                raise RuntimeError("MM-InvNet requires encoder mixed/F_base features.")
            recon = self.mm_decoder(features.mixed, out_hw=batch.depth_vel.shape[-2:])
            loss = F.l1_loss(recon, batch.depth_vel)
        elif self.variant in {"concat_fm", "cncs_fm"}:
            loss, recon = self._direct_full_field_loss(batch, self._direct_condition(features, conditions))
        else:
            loss, recon, _ = self._two_stage_loss(batch, conditions, features)
        self._log_epoch_metric(f"{prefix}/loss", loss, prog_bar=True)
        self._log_epoch_metric(f"{prefix}/mae", F.l1_loss(recon, batch.depth_vel))
        return loss

    def training_step(self, batch, batch_idx):
        if self.variant == "velocity_gan":
            return self._velocity_gan_training_step(self._as_batch(batch), batch_idx)
        return self._step(batch, "train")

    def _velocity_gan_training_step(self, batch: BGSampleBatch, batch_idx: int) -> torch.Tensor:
        optimizer_g, optimizer_d = self.optimizers()
        optimizer_d.zero_grad()
        discriminator = self.velocity_gan.discriminator_loss(batch)
        self.manual_backward(discriminator["loss"])
        optimizer_d.step()

        num_batches_value = getattr(self.trainer, "num_training_batches", None)
        num_batches = int(num_batches_value) if isinstance(num_batches_value, int) else None
        generator_loss = None
        if self.velocity_gan.should_update_generator(batch_idx, num_batches):
            optimizer_g.zero_grad()
            generator = self.velocity_gan.generator_loss(batch)
            self.manual_backward(generator["loss"])
            optimizer_g.step()
            generator_loss = generator["loss"].detach()
            self._log_epoch_metric("train/gan_g", generator_loss)
            self._log_epoch_metric("train/mae", generator["l1"].detach())
            self._log_epoch_metric("train/adversarial", generator["adversarial"].detach())

        self._log_epoch_metric("train/gan_d", discriminator["loss"].detach())
        self._log_epoch_metric("train/critic_gap", discriminator["critic_gap"].detach())
        self._log_epoch_metric("train/gradient_penalty", discriminator["gradient_penalty"].detach())
        step_loss = generator_loss if generator_loss is not None else discriminator["loss"].detach()
        self._log_epoch_metric("train/loss", step_loss, prog_bar=True)
        return step_loss

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    @torch.no_grad()
    def predict_batch(self, batch: BGSampleBatch | dict[str, torch.Tensor]) -> PredictionBatch:
        batch = self._as_batch(batch)
        if self.variant in {
            "smooth_dix",
            "adapted_inversion_net",
            "adapted_upfwi",
            "sv_inv_net",
            "conditional_ddpm",
            "velocity_gan",
        }:
            if self.variant == "smooth_dix":
                velocity_hat = self.smooth_dix(batch)
            elif self.variant == "adapted_inversion_net":
                velocity_hat = self.adapted_inversion_net(batch)
            elif self.variant == "adapted_upfwi":
                velocity_hat = self.adapted_upfwi(batch)
            elif self.variant == "sv_inv_net":
                velocity_hat = self.sv_inv_net(batch)
            elif self.variant == "conditional_ddpm":
                velocity_hat = self.conditional_ddpm.sample(batch)
            else:
                velocity_hat = self.velocity_gan(batch)
            bg_hat = self._zero_bg(batch.depth_vel)
            residual_hat = velocity_hat
            rho_hat = torch.zeros(batch.depth_vel.shape[0], device=batch.depth_vel.device)
            alpha_hat = torch.zeros_like(rho_hat)
            epsilon_h = torch.zeros_like(rho_hat)
            return PredictionBatch(
                bg_hat=bg_hat,
                velocity_hat=velocity_hat,
                residual_hat=residual_hat,
                rho_hat_b=rho_hat,
                alpha_hat=alpha_hat,
                epsilon_h_b=epsilon_h,
                metadata=batch.metadata,
            )
        if self.variant in {"adapted_gfi", "adapted_gfi_latent_unet"}:
            if self.variant == "adapted_gfi":
                velocity_hat = self.adapted_gfi(batch)
            else:
                velocity_hat = self.adapted_gfi_latent_unet(batch)
            bg_hat = self._zero_bg(batch.depth_vel)
            residual_hat = velocity_hat
            rho_hat = torch.zeros(batch.depth_vel.shape[0], device=batch.depth_vel.device)
            alpha_hat = torch.zeros_like(rho_hat)
            epsilon_h = torch.zeros_like(rho_hat)
            return PredictionBatch(
                bg_hat=bg_hat,
                velocity_hat=velocity_hat,
                residual_hat=residual_hat,
                rho_hat_b=rho_hat,
                alpha_hat=alpha_hat,
                epsilon_h_b=epsilon_h,
                metadata=batch.metadata,
            )
        if self.variant == "pdr_gfi_aligned_direct":
            velocity_hat = self.adapted_gfi(batch)
            bg_hat = self.filter.lowpass(velocity_hat)
            residual_hat = velocity_hat - bg_hat
            rho_hat = torch.zeros(batch.depth_vel.shape[0], device=batch.depth_vel.device)
            alpha_hat = torch.zeros_like(rho_hat)
            epsilon_h = self.filter.highpass(bg_hat).abs().flatten(1).mean(dim=1)
            return PredictionBatch(
                bg_hat=bg_hat,
                velocity_hat=velocity_hat,
                residual_hat=residual_hat,
                rho_hat_b=rho_hat,
                alpha_hat=alpha_hat,
                epsilon_h_b=epsilon_h,
                metadata=batch.metadata,
            )
        if self.variant in {"adapted_auto_linear", "adapted_auto_linear_original"}:
            auto_linear = (
                self.adapted_auto_linear_original
                if self.variant == "adapted_auto_linear_original"
                else self.adapted_auto_linear
            )
            velocity_hat = auto_linear(batch)
            bg_hat = self._zero_bg(batch.depth_vel)
            residual_hat = velocity_hat
            rho_hat = torch.zeros(batch.depth_vel.shape[0], device=batch.depth_vel.device)
            alpha_hat = torch.zeros_like(rho_hat)
            epsilon_h = torch.zeros_like(rho_hat)
            return PredictionBatch(
                bg_hat=bg_hat,
                velocity_hat=velocity_hat,
                residual_hat=residual_hat,
                rho_hat_b=rho_hat,
                alpha_hat=alpha_hat,
                epsilon_h_b=epsilon_h,
                metadata=batch.metadata,
            )
        features, conditions = self._conditions(batch)
        if self.variant == "mm_invnet":
            assert features.mixed is not None
            velocity_hat = self.mm_decoder(features.mixed, out_hw=batch.depth_vel.shape[-2:])
            bg_hat = self._zero_bg(batch.depth_vel)
            residual_hat = velocity_hat
            rho_hat = torch.zeros(batch.depth_vel.shape[0], device=batch.depth_vel.device)
            alpha_hat = torch.zeros_like(rho_hat)
            epsilon_h = torch.zeros_like(rho_hat)
        elif self.variant in {"concat_fm", "cncs_fm"}:
            cond = self._direct_condition(features, conditions)
            z_hat = self.full_field.sample(
                cond,
                torch.zeros_like(batch.depth_vel),
                x_size=tuple(self.codec.encode(batch.depth_vel, deterministic=True).shape),
                steps=self.residual_num_inference_steps,
            )
            velocity_hat = self.codec.decode(z_hat, out_hw=batch.depth_vel.shape[-2:])
            bg_hat = self._zero_bg(batch.depth_vel)
            residual_hat = velocity_hat
            rho_hat = torch.zeros(batch.depth_vel.shape[0], device=batch.depth_vel.device)
            alpha_hat = torch.zeros_like(rho_hat)
            epsilon_h = torch.zeros_like(rho_hat)
        else:
            bg = self.background(conditions.numerical, out_hw=batch.depth_vel.shape[-2:])
            gate = self.rho_calibrator(features.reliability, batch, bg.epsilon_h_b)
            z_bg = self.codec.encode(bg.bg_hat, deterministic=True)
            z_res = self.residual.sample(
                conditions.structural,
                bg.bg_hat.detach(),
                x_size=tuple(z_bg.shape),
                steps=self.residual_num_inference_steps,
                z_bg=z_bg,
            )
            velocity_hat = self.codec.decode(z_bg + z_res, out_hw=batch.depth_vel.shape[-2:])
            bg_hat = bg.bg_hat
            residual_hat = velocity_hat - bg_hat
            rho_hat = gate.rho_hat_b
            alpha_hat = gate.alpha_hat
            epsilon_h = bg.epsilon_h_b
        return PredictionBatch(
            bg_hat=bg_hat,
            velocity_hat=velocity_hat,
            residual_hat=residual_hat,
            rho_hat_b=rho_hat,
            alpha_hat=alpha_hat,
            epsilon_h_b=epsilon_h,
            metadata=batch.metadata,
        )

    def benchmark_metadata(self) -> dict[str, int | str]:
        discriminator_params = 0
        if self.variant == "smooth_dix":
            generator_modules = [self.smooth_dix]
        elif self.variant == "adapted_inversion_net":
            generator_modules = [self.adapted_inversion_net]
        elif self.variant == "adapted_upfwi":
            generator_modules = [self.adapted_upfwi]
        elif self.variant == "sv_inv_net":
            generator_modules = [self.sv_inv_net]
        elif self.variant == "conditional_ddpm":
            generator_modules = [self.conditional_ddpm]
        elif self.variant == "velocity_gan":
            generator_modules = [self.velocity_gan.generator]
            discriminator_params = _count_parameters(self.velocity_gan.discriminator)
        elif self.variant == "mm_invnet":
            generator_modules = [self.mm_decoder]
        elif self.variant in {"adapted_gfi", "pdr_gfi_aligned_direct"}:
            generator_modules = [self.adapted_gfi]
        elif self.variant == "adapted_gfi_latent_unet":
            generator_modules = [self.adapted_gfi_latent_unet]
        elif self.variant == "adapted_auto_linear":
            generator_modules = [self.adapted_auto_linear]
        elif self.variant == "adapted_auto_linear_original":
            generator_modules = [self.adapted_auto_linear_original]
        elif self.variant == "concat_fm":
            generator_modules = [self.mixed_projector, self.full_field]
        elif self.variant == "cncs_fm":
            generator_modules = [self.adapters, self.full_field]
        else:
            generator_modules = [self.background, self.residual]
        generator_params = sum(_count_parameters(module) for module in generator_modules)
        trainable_generator_params = sum(_count_parameters(module, trainable_only=True) for module in generator_modules)
        reported_params = generator_params + discriminator_params if self.variant == "velocity_gan" else generator_params
        return {
            "variant": self.variant,
            "capacity_tier": self.capacity_tier,
            "params": _count_parameters(self),
            "trainable_params": _count_parameters(self, trainable_only=True),
            "generator_params": int(generator_params),
            "trainable_generator_params": int(trainable_generator_params),
            "discriminator_params": int(discriminator_params),
            "reported_params": int(reported_params),
            "residual_num_inference_steps": self.residual_num_inference_steps,
            "stage_checkpoint_used": self.stage_checkpoint_used,
        }

    def configure_optimizers(self):
        if self.variant == "velocity_gan":
            optimizer_g = torch.optim.AdamW(
                self.velocity_gan.generator.parameters(),
                lr=self.velocity_gan_lr_g,
                betas=(0.0, 0.9),
                weight_decay=self.weight_decay,
            )
            optimizer_d = torch.optim.AdamW(
                self.velocity_gan.discriminator.parameters(),
                lr=self.velocity_gan_lr_d,
                betas=(0.0, 0.9),
                weight_decay=self.weight_decay,
            )
            return [optimizer_g, optimizer_d]
        trainable = [parameter for parameter in self.parameters() if parameter.requires_grad]
        if not trainable:
            if self.variant == "smooth_dix":
                return None
            raise RuntimeError(f"No trainable parameters for benchmark variant {self.variant!r}.")
        optimizer = torch.optim.AdamW(trainable, lr=self.lr, weight_decay=self.weight_decay)
        if self.scheduler_name in {"none", "", "null"}:
            return optimizer
        if self.scheduler_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, self.max_epochs))
            return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
        raise ValueError(f"Unsupported training.scheduler {self.scheduler_name!r}.")
