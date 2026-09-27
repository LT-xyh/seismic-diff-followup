"""Engineering-realistic input perturbations for benchmark evaluation."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

from bg_pdr_fm.data.batch import BGSampleBatch


@dataclass(frozen=True)
class EngineeringStressScenario:
    """Deterministic perturbation protocol applied to model inputs only."""

    name: str
    pstm_snr_db: float | None
    rms_scale_max: float
    rms_smooth_noise_std_frac: float
    rms_lowfreq_bias_std_frac: float
    horizon_missing_prob: float
    well_missing_prob: float
    seed: int = 2027


ENGINEERING_STRESS_SCENARIOS: dict[str, EngineeringStressScenario] = {
    "pstm20_rms_mild_h025_w050": EngineeringStressScenario(
        name="pstm20_rms_mild_h025_w050",
        pstm_snr_db=20.0,
        rms_scale_max=0.02,
        rms_smooth_noise_std_frac=0.01,
        rms_lowfreq_bias_std_frac=0.01,
        horizon_missing_prob=0.25,
        well_missing_prob=0.50,
        seed=2027,
    ),
    "pstm15_rms_moderate_h050_w075": EngineeringStressScenario(
        name="pstm15_rms_moderate_h050_w075",
        pstm_snr_db=15.0,
        rms_scale_max=0.05,
        rms_smooth_noise_std_frac=0.025,
        rms_lowfreq_bias_std_frac=0.025,
        horizon_missing_prob=0.50,
        well_missing_prob=0.75,
        seed=2027,
    ),
}


def get_engineering_stress_scenario(name: str, *, seed: int | None = None) -> EngineeringStressScenario:
    if name not in ENGINEERING_STRESS_SCENARIOS:
        raise ValueError(f"Unknown engineering stress scenario {name!r}; expected {sorted(ENGINEERING_STRESS_SCENARIOS)}.")
    scenario = ENGINEERING_STRESS_SCENARIOS[name]
    if seed is None:
        return scenario
    return replace(scenario, seed=int(seed))


def engineering_stress_from_config(conf: Any) -> EngineeringStressScenario | None:
    enabled = bool(OmegaConf.select(conf, "evaluation.engineering_stress.enabled", default=False))
    if not enabled:
        return None
    name = str(OmegaConf.select(conf, "evaluation.engineering_stress.scenario", default="pstm20_rms_mild_h025_w050"))
    seed = OmegaConf.select(conf, "evaluation.engineering_stress.seed", default=None)
    return get_engineering_stress_scenario(name, seed=None if seed is None else int(seed))


def _randn_like(x: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    return torch.randn(x.shape, generator=generator, device=x.device, dtype=x.dtype)


def _rand(
    shape: tuple[int, ...],
    *,
    generator: torch.Generator,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.rand(shape, generator=generator, device=device, dtype=dtype)


def _sample_generator(batch: BGSampleBatch, scenario: EngineeringStressScenario, batch_index: int) -> torch.Generator:
    generator = torch.Generator(device=batch.depth_vel.device)
    generator.manual_seed(int(scenario.seed) + int(batch_index) * 1000003)
    return generator


def _add_pstm_snr_noise(x: torch.Tensor, snr_db: float, generator: torch.Generator) -> torch.Tensor:
    dims = tuple(range(1, x.ndim))
    signal_rms = x.square().mean(dim=dims, keepdim=True).sqrt().clamp_min(1e-8)
    noise_std = signal_rms / (10.0 ** (float(snr_db) / 20.0))
    return x + _randn_like(x, generator) * noise_std


def _smooth_field(noise: torch.Tensor, kernel_size: int) -> torch.Tensor:
    if kernel_size <= 1:
        return noise
    pad = kernel_size // 2
    flat = noise.view(-1, 1, noise.shape[-2], noise.shape[-1])
    smooth = F.avg_pool2d(F.pad(flat, (pad, pad, pad, pad), mode="replicate"), kernel_size=kernel_size, stride=1)
    return smooth.view_as(noise)


def _perturb_rms(x: torch.Tensor, scenario: EngineeringStressScenario, generator: torch.Generator) -> torch.Tensor:
    out = x
    batch_size = x.shape[0]
    dtype = x.dtype
    device = x.device
    if scenario.rms_scale_max > 0:
        scale = 1.0 + (_rand((batch_size, 1, 1, 1), generator=generator, device=device, dtype=dtype) * 2.0 - 1.0) * float(
            scenario.rms_scale_max
        )
        out = out * scale
    dims = tuple(range(1, x.ndim))
    sample_std = x.std(dim=dims, keepdim=True, unbiased=False).clamp_min(1e-8)
    if scenario.rms_smooth_noise_std_frac > 0:
        noise = _smooth_field(_randn_like(x, generator), kernel_size=17)
        noise_std = noise.std(dim=dims, keepdim=True, unbiased=False).clamp_min(1e-8)
        out = out + noise / noise_std * sample_std * float(scenario.rms_smooth_noise_std_frac)
    if scenario.rms_lowfreq_bias_std_frac > 0:
        bias = _smooth_field(_randn_like(x, generator), kernel_size=65)
        bias_std = bias.std(dim=dims, keepdim=True, unbiased=False).clamp_min(1e-8)
        out = out + bias / bias_std * sample_std * float(scenario.rms_lowfreq_bias_std_frac)
    return out


def _drop_per_sample(
    tensor: torch.Tensor,
    keep: torch.Tensor,
) -> torch.Tensor:
    shape = (keep.shape[0],) + (1,) * (tensor.ndim - 1)
    return tensor * keep.view(shape).to(dtype=tensor.dtype, device=tensor.device)


def apply_engineering_stress(
    batch: BGSampleBatch,
    scenario: EngineeringStressScenario,
    *,
    batch_index: int = 0,
) -> BGSampleBatch:
    """Apply a deterministic engineering stress scenario to inputs, never targets."""

    generator = _sample_generator(batch, scenario, batch_index)
    migrated_image = batch.migrated_image
    if scenario.pstm_snr_db is not None:
        migrated_image = _add_pstm_snr_noise(migrated_image, float(scenario.pstm_snr_db), generator)

    rms_vel = _perturb_rms(batch.rms_vel, scenario, generator)

    batch_size = int(batch.depth_vel.shape[0])
    random_values = _rand(
        (batch_size, 2),
        generator=generator,
        device=batch.depth_vel.device,
        dtype=batch.depth_vel.dtype,
    )
    horizon_keep = (random_values[:, 0] >= float(scenario.horizon_missing_prob)).to(batch.depth_vel.dtype)
    well_keep = (random_values[:, 1] >= float(scenario.well_missing_prob)).to(batch.depth_vel.dtype)

    modality_mask = batch.modality_mask.clone()
    modality_quality = batch.modality_quality.clone()
    modality_mask[:, 1] = modality_mask[:, 1] * horizon_keep.to(modality_mask.device, modality_mask.dtype)
    modality_quality[:, 1] = modality_quality[:, 1] * horizon_keep.to(modality_quality.device, modality_quality.dtype)
    modality_mask[:, 3] = modality_mask[:, 3] * well_keep.to(modality_mask.device, modality_mask.dtype)
    modality_quality[:, 3] = modality_quality[:, 3] * well_keep.to(modality_quality.device, modality_quality.dtype)

    return replace(
        batch,
        migrated_image=migrated_image,
        rms_vel=rms_vel,
        horizon=_drop_per_sample(batch.horizon, horizon_keep),
        well_log=_drop_per_sample(batch.well_log, well_keep),
        well_mask=_drop_per_sample(batch.well_mask, well_keep),
        modality_mask=modality_mask,
        modality_quality=modality_quality,
    )
