"""Stage-specific loss helpers shared by the BG-PDR-FM Lightning module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from bg_pdr_fm.data.batch import BGSampleBatch
from bg_pdr_fm.models.filters import LowHighPassFilter
from bg_pdr_fm.models.generators import haar_detail_leak, haar_lowpass_reconstruct


def conf_get(conf: Any, path: str, default: Any) -> Any:
    current = conf
    for part in path.split("."):
        if isinstance(current, dict):
            if part not in current:
                return default
            current = current[part]
        else:
            if not hasattr(current, part):
                return default
            current = getattr(current, part)
    return current


@dataclass
class StageLossOutput:
    loss: torch.Tensor
    rho_true: torch.Tensor | None
    recon: torch.Tensor | None = None
    metrics: dict[str, torch.Tensor] | None = None


def true_residual_low_frequency_ratio(filter_module, depth_vel: torch.Tensor, bg_hat: torch.Tensor) -> torch.Tensor:
    _, high = filter_module(depth_vel)
    residual_low = filter_module.lowpass(depth_vel - bg_hat)
    num = residual_low.abs().flatten(1).mean(dim=1)
    den = high.abs().flatten(1).mean(dim=1) + 1e-6
    return num / den


def _background_scales(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        values = text.strip("[]").split(",")
    else:
        try:
            values = list(value)
        except TypeError:
            values = [value]
    scales: list[int] = []
    for item in values:
        scale = int(item)
        if scale <= 1:
            raise ValueError("loss.bg_scales must contain kernel sizes greater than 1.")
        if scale % 2 == 0:
            raise ValueError("loss.bg_scales must contain odd kernel sizes.")
        scales.append(scale)
    return scales


def multi_scale_background_loss_terms(
    depth_vel: torch.Tensor,
    bg_hat: torch.Tensor,
    scales: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    if not scales:
        zero = bg_hat.new_zeros(())
        return zero, zero
    low_terms: list[torch.Tensor] = []
    high_terms: list[torch.Tensor] = []
    for scale in scales:
        filter_module = LowHighPassFilter(kernel_size=scale)
        bg_low, bg_high = filter_module(bg_hat)
        vel_low = filter_module.lowpass(depth_vel)
        low_terms.append(F.l1_loss(bg_low, vel_low))
        high_terms.append(bg_high.abs().mean())
    return torch.stack(low_terms).mean(), torch.stack(high_terms).mean()


def background_low_residual_loss_term(
    depth_vel: torch.Tensor,
    bg_hat: torch.Tensor,
    kernel_size: int,
) -> torch.Tensor:
    filter_module = LowHighPassFilter(kernel_size=kernel_size)
    return filter_module.lowpass(depth_vel - bg_hat).abs().mean()


def _hw_pair(value: Any, default: tuple[int, int]) -> tuple[int, int]:
    if value is None:
        return default
    values = list(value) if not isinstance(value, str) else value.strip("[]").split(",")
    if len(values) != 2:
        raise ValueError(f"Expected an HW pair, got {value!r}.")
    hw = (int(values[0]), int(values[1]))
    if min(hw) <= 0:
        raise ValueError(f"Expected positive HW values, got {hw}.")
    return hw


def _resize_low_frequency(x: torch.Tensor, hw: tuple[int, int], out_hw: tuple[int, int] | None = None) -> torch.Tensor:
    low = F.interpolate(x, size=hw, mode="area")
    if out_hw is not None:
        low = F.interpolate(low, size=out_hw, mode="bilinear", align_corners=False)
    return low


def smooth_background_loss_terms(
    depth_vel: torch.Tensor,
    bg_hat: torch.Tensor,
    bottleneck_hw: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    out_hw = tuple(bg_hat.shape[-2:])
    smooth_target = _resize_low_frequency(depth_vel, bottleneck_hw, out_hw=out_hw)
    bg_l1 = F.l1_loss(bg_hat, smooth_target)
    bg_35 = _resize_low_frequency(bg_hat, (35, 35))
    vel_35 = _resize_low_frequency(depth_vel, (35, 35))
    bg_low = _resize_low_frequency(bg_hat, bottleneck_hw)
    vel_low = _resize_low_frequency(depth_vel, bottleneck_hw)
    return bg_l1, F.l1_loss(bg_35, vel_35), F.l1_loss(bg_low, vel_low)


def contrastive_stage_loss(module, batch: BGSampleBatch, features) -> StageLossOutput:
    anchors = module.wavelet_anchor(batch.depth_vel)
    loss_dict = module.contrastive_loss(
        features,
        batch,
        anchor_struct=anchors.anchor_struct.detach(),
        anchor_num=anchors.anchor_num.detach(),
    )
    module._last_contrastive_metrics = loss_dict
    return StageLossOutput(loss=loss_dict["loss"], rho_true=None)


def background_stage_loss(module, batch: BGSampleBatch, bg, gate) -> StageLossOutput:
    background_output_mode = str(conf_get(module.conf, "model.background_output_mode", "direct")).lower()
    background_backend = str(conf_get(module.conf, "model.background_backend", "conv")).lower()
    smooth_metrics: dict[str, torch.Tensor] = {}
    if background_output_mode == "smooth_bottleneck":
        bottleneck_hw = _hw_pair(conf_get(module.conf, "model.background_bottleneck_hw", (18, 18)), (18, 18))
        bg_l1, bg_smooth_l1_35, bg_smooth_l1_18 = smooth_background_loss_terms(
            batch.depth_vel,
            bg.bg_hat,
            bottleneck_hw,
        )
        low = _resize_low_frequency(batch.depth_vel, bottleneck_hw, out_hw=tuple(bg.bg_hat.shape[-2:]))
        smooth_metrics = {
            "bg_smooth_l1_35": bg_smooth_l1_35.detach(),
            "bg_smooth_l1_18": bg_smooth_l1_18.detach(),
        }
    elif background_output_mode == "wavelet_ll":
        wavelet_level = int(conf_get(module.conf, "model.background_wavelet_level", 1))
        low = haar_lowpass_reconstruct(batch.depth_vel, level=wavelet_level)
        bg_l1 = F.l1_loss(bg.bg_hat, low)
    else:
        bg_target = str(conf_get(module.conf, "loss.bg_target", "lowpass")).lower()
        if bg_target != "lowpass":
            raise ValueError(f"loss.bg_target must be 'lowpass' for direct background loss, got {bg_target!r}.")
        low = module.filter.lowpass(batch.depth_vel)
        bg_l1 = F.l1_loss(bg.bg_hat, low)
    bg_l2 = F.mse_loss(bg.bg_hat, low)
    rho_true = true_residual_low_frequency_ratio(module.filter, batch.depth_vel, bg.bg_hat)
    if bg.bg_hat.shape[-2] % 2 == 0 and bg.bg_hat.shape[-1] % 2 == 0:
        bg_wavelet_detail_leak = haar_detail_leak(bg.bg_hat)
    else:
        bg_wavelet_detail_leak = bg.bg_hat.new_zeros(())
    bg_scales = _background_scales(conf_get(module.conf, "loss.bg_scales", []))
    bg_ms_l1, bg_ms_high = multi_scale_background_loss_terms(batch.depth_vel, bg.bg_hat, bg_scales)
    bg_high_weight = float(conf_get(module.conf, "loss.bg_high_weight", 0.05))
    bg_multiscale_weight = float(conf_get(module.conf, "loss.bg_multiscale_weight", 0.0))
    bg_multiscale_high_weight = float(conf_get(module.conf, "loss.bg_multiscale_high_weight", 0.0))
    bg_low_residual_weight = float(conf_get(module.conf, "loss.bg_low_residual_weight", 0.0))
    bg_low_residual_kernel = int(conf_get(module.conf, "loss.bg_low_residual_kernel", 9))
    bg_low_residual = background_low_residual_loss_term(batch.depth_vel, bg.bg_hat, bg_low_residual_kernel)
    direct_l1_l2 = (
        background_output_mode == "direct"
        and (
            background_backend in {"unet_direct", "flow_matching"}
            or conf_get(module.conf, "loss.bg_l1_weight", None) is not None
            or conf_get(module.conf, "loss.bg_l2_weight", None) is not None
        )
    )
    if background_output_mode == "smooth_bottleneck":
        bg_smooth_l1_35_weight = float(conf_get(module.conf, "loss.bg_smooth_l1_35_weight", 0.5))
        bg_smooth_l1_18_weight = float(conf_get(module.conf, "loss.bg_smooth_l1_18_weight", 0.3))
        loss = bg_l1 + bg_smooth_l1_35_weight * bg_smooth_l1_35 + bg_smooth_l1_18_weight * bg_smooth_l1_18
    elif background_backend == "flow_matching":
        if bg.fm_loss is None:
            raise RuntimeError("background_backend='flow_matching' requires BackgroundEstimate.fm_loss during training.")
        bg_fm_weight = float(conf_get(module.conf, "loss.bg_fm_weight", 1.0))
        bg_l1_weight = float(conf_get(module.conf, "loss.bg_l1_weight", 1.0))
        bg_l2_weight = float(conf_get(module.conf, "loss.bg_l2_weight", 0.5))
        loss = bg_fm_weight * bg.fm_loss + bg_l1_weight * bg_l1 + bg_l2_weight * bg_l2
    elif direct_l1_l2:
        bg_l1_weight = float(conf_get(module.conf, "loss.bg_l1_weight", 1.0))
        bg_l2_weight = float(conf_get(module.conf, "loss.bg_l2_weight", 0.0))
        loss = bg_l1_weight * bg_l1 + bg_l2_weight * bg_l2
    else:
        loss = (
            bg_l1
            + bg_high_weight * bg.epsilon_h_b.mean()
            + bg_multiscale_weight * bg_ms_l1
            + bg_multiscale_high_weight * bg_ms_high
            + bg_low_residual_weight * bg_low_residual
            + float(conf_get(module.conf, "loss.rho_weight", 0.0)) * module.rho_calibrator.rho_loss(gate, rho_true)
        )
    metrics = {
        "bg_l1": bg_l1.detach(),
        "bg_l2": bg_l2.detach(),
        "bg_loss": loss.detach(),
        "bg_high_leak": bg.epsilon_h_b.mean().detach(),
        "bg_ms_l1": bg_ms_l1.detach(),
        "bg_ms_high_leak": bg_ms_high.detach(),
        "bg_low_residual": bg_low_residual.detach(),
        "bg_wavelet_detail_leak": bg_wavelet_detail_leak.detach(),
    }
    if bg.fm_loss is not None:
        metrics["bg_fm_loss"] = bg.fm_loss.detach()
    metrics.update(smooth_metrics)
    return StageLossOutput(
        loss=loss,
        rho_true=rho_true,
        metrics=metrics,
    )


def select_residual_background(
    conf: Any,
    filter_module: LowHighPassFilter,
    depth_vel: torch.Tensor,
    predicted_bg: torch.Tensor,
) -> torch.Tensor:
    source = str(conf_get(conf, "model.residual_background_source", "predicted")).lower()
    if source in {"predicted", ""}:
        return predicted_bg
    if source == "oracle_lowpass":
        return filter_module.lowpass(depth_vel).detach()
    raise ValueError(
        "model.residual_background_source must be one of 'predicted' or 'oracle_lowpass', "
        f"got {source!r}."
    )


def select_residual_target_background(
    module,
    depth_vel: torch.Tensor,
    predicted_background: torch.Tensor,
    default_background: torch.Tensor,
) -> torch.Tensor:
    """Select the residual target background without changing inference inputs.

    The prediction-consistent main path uses the same predicted background for
    the target, context, and reconstruction. The ablation may use the fixed
    low-pass target while retaining the predicted background at inference,
    exposing the train/inference mismatch explicitly.
    """
    source = str(conf_get(module.conf, "training.ablation.residual_target_source", "predicted")).lower()
    if source in {"predicted", ""}:
        return default_background
    if source in {"lowpass", "oracle_lowpass"}:
        return module.filter.lowpass(depth_vel).detach()
    raise ValueError(
        "training.ablation.residual_target_source must be 'predicted' or 'lowpass', "
        f"got {source!r}."
    )


def residual_stage_loss(module, batch: BGSampleBatch, conditions, bg, gate) -> StageLossOutput:
    # Paper Eqs. (4)-(6): default training reuses detached B_hat for target, context, and composition.
    z_full = module.codec.encode(batch.depth_vel, deterministic=True).detach()
    residual_bg = select_residual_background(module.conf, module.filter, batch.depth_vel, bg.bg_hat.detach())
    context_z_bg = module.codec.encode(residual_bg.detach(), deterministic=True).detach()
    target_bg = select_residual_target_background(module, batch.depth_vel, bg.bg_hat.detach(), residual_bg)
    target_z_bg = module.codec.encode(target_bg.detach(), deterministic=True).detach()
    z_res = (z_full - target_z_bg).detach()
    cond_s = conditions.structural
    alpha_hat = gate.alpha_hat if str(conf_get(module.conf, "quality_gate.mode", "none")).lower() == "learned" else None
    fm = module.residual.training_loss(
        z_res, cond_s, residual_bg.detach(), alpha_hat=alpha_hat, z_bg=context_z_bg
    )
    recon = module.codec.decode(context_z_bg + fm["z_res_pred"], out_hw=batch.depth_vel.shape[-2:])
    rho_true = true_residual_low_frequency_ratio(module.filter, batch.depth_vel, residual_bg.detach())
    target_lf_weight = float(conf_get(module.conf, "loss.target_lf_weight", 0.1))
    res_pred = recon - residual_bg.detach()
    res_target = batch.depth_vel - residual_bg.detach()
    target_lf_loss = F.l1_loss(module.filter.lowpass(res_pred), module.filter.lowpass(res_target))
    noise = torch.randn_like(z_full)
    e_v = ((z_full - noise) ** 2).mean()
    e_r = ((z_res - noise) ** 2).mean()
    e_ratio = e_r / e_v.clamp_min(1e-8)
    direct_recon = module.codec.decode(z_full, out_hw=batch.depth_vel.shape[-2:])
    bg_recon = module.codec.decode(context_z_bg, out_hw=batch.depth_vel.shape[-2:])
    composed_recon = module.codec.decode(context_z_bg + z_res, out_hw=batch.depth_vel.shape[-2:])
    vae_direct_l1 = F.l1_loss(direct_recon, batch.depth_vel)
    vae_bg_l1 = F.l1_loss(bg_recon, residual_bg.detach())
    vae_composed_l1 = F.l1_loss(composed_recon, batch.depth_vel)
    vae_identity_l1 = F.l1_loss(composed_recon, direct_recon)
    vae_identity_ratio = vae_identity_l1 / vae_direct_l1.clamp_min(1e-8)
    decoded_delta = composed_recon - bg_recon
    physical_delta = batch.depth_vel - residual_bg.detach()
    vae_gap_l1 = F.l1_loss(decoded_delta, physical_delta)
    vae_gap_ratio = vae_gap_l1 / (vae_direct_l1 + vae_bg_l1).clamp_min(1e-8)
    loss = (
        fm["loss"]
        + float(conf_get(module.conf, "loss.final_l1_weight", 1.0)) * F.l1_loss(recon, batch.depth_vel)
        + target_lf_weight * target_lf_loss
        + float(conf_get(module.conf, "loss.rho_weight", 0.0)) * module.rho_calibrator.rho_loss(gate, rho_true)
    )
    return StageLossOutput(
        loss=loss,
        rho_true=rho_true,
        recon=recon,
        metrics={
            "target_lf_loss": target_lf_loss.detach(),
            "E_V": e_v.detach(),
            "E_R": e_r.detach(),
            "E_R_over_E_V": e_ratio.detach(),
            "vae_direct_l1": vae_direct_l1.detach(),
            "vae_bg_l1": vae_bg_l1.detach(),
            "vae_composed_l1": vae_composed_l1.detach(),
            "vae_identity_l1": vae_identity_l1.detach(),
            "vae_identity_ratio": vae_identity_ratio.detach(),
            "vae_gap_l1": vae_gap_l1.detach(),
            "vae_gap_ratio": vae_gap_ratio.detach(),
        },
    )


def joint_full_stage_loss(module, batch: BGSampleBatch, features, conditions, bg, gate) -> StageLossOutput:
    """Train contrastive encoder, background, and residual FM together."""
    contrastive = contrastive_stage_loss(module, batch, features)
    background = background_stage_loss(module, batch, bg, gate)
    residual = residual_stage_loss(module, batch, conditions, bg, gate)
    lambda_contrastive = float(conf_get(module.conf, "training.joint.lambda_contrastive", 0.01))
    loss = residual.loss + background.loss + lambda_contrastive * contrastive.loss

    metrics: dict[str, torch.Tensor] = {
        "joint_contrastive_loss": contrastive.loss.detach(),
        "joint_background_loss": background.loss.detach(),
        "joint_residual_loss": residual.loss.detach(),
        "joint_lambda_contrastive": loss.new_tensor(lambda_contrastive),
    }
    if background.metrics:
        metrics.update({f"joint_bg_{key}": value for key, value in background.metrics.items()})
    if residual.metrics:
        metrics.update({f"joint_residual_{key}": value for key, value in residual.metrics.items()})

    return StageLossOutput(
        loss=loss,
        rho_true=residual.rho_true,
        recon=residual.recon,
        metrics=metrics,
    )
