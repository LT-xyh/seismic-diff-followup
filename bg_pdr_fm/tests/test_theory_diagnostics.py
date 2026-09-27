from __future__ import annotations

import torch
from omegaconf import OmegaConf

from bg_pdr_fm.evaluation import benchmark_metrics
from bg_pdr_fm.lightning.bg_pdr_fm_module import _feature_low_high_ratios
from bg_pdr_fm.lightning.stage_losses import true_residual_low_frequency_ratio
from bg_pdr_fm.models.filters import LowHighPassFilter


def test_prediction_consistent_residual_has_no_background_composition_term():
    target = torch.tensor([[[[0.1, 0.4], [0.7, 1.0]]]])
    background_hat = torch.tensor([[[[0.2, 0.3], [0.5, 0.8]]]])
    residual_hat = torch.tensor([[[[-0.1, 0.2], [0.3, 0.4]]]])
    residual_target = target - background_hat

    reconstruction_error = background_hat + residual_hat - target

    torch.testing.assert_close(reconstruction_error, residual_hat - residual_target)


def test_oracle_residual_exposes_background_composition_error():
    target = torch.tensor([[[[0.1, 0.4], [0.7, 1.0]]]])
    oracle_background = torch.tensor([[[[0.2, 0.3], [0.6, 0.9]]]])
    background_hat = torch.tensor([[[[0.1, 0.4], [0.5, 0.8]]]])
    residual_hat = torch.tensor([[[[-0.1, 0.2], [0.3, 0.4]]]])
    oracle_residual = target - oracle_background

    reconstruction_error = background_hat + residual_hat - target
    decomposed_error = (background_hat - oracle_background) + (residual_hat - oracle_residual)

    torch.testing.assert_close(reconstruction_error, decomposed_error)


def test_low_frequency_burden_identity_holds_for_non_idempotent_average_pool():
    filter_module = LowHighPassFilter(kernel_size=3)
    target = torch.arange(25, dtype=torch.float32).reshape(1, 1, 5, 5) / 25
    background_hat = torch.zeros_like(target)
    background_hat[..., 2, 2] = 1.0
    background = filter_module.lowpass(target)

    lhs = filter_module.lowpass(target - background_hat)
    rhs = (background - background_hat) + filter_module.highpass(background_hat)

    torch.testing.assert_close(lhs, rhs)
    assert not torch.allclose(
        filter_module.lowpass(filter_module.lowpass(background_hat)),
        filter_module.lowpass(background_hat),
    )


def test_l2_residual_energy_ratio_matches_definition_per_sample():
    target = torch.tensor([[[[3.0, 4.0]]], [[[2.0, 0.0]]]])
    background_hat = torch.tensor([[[[1.0, 2.0]]], [[[1.0, 1.0]]]])

    ratio = benchmark_metrics.residual_energy_ratio_l2(target, background_hat)

    torch.testing.assert_close(ratio, torch.tensor([4.0 / 12.5, 2.0 / 4.0]))


def test_l2_residual_energy_ratio_adds_epsilon_to_denominator():
    target = torch.ones(1, 1, 1, 1)
    background_hat = torch.zeros_like(target)

    ratio = benchmark_metrics.residual_energy_ratio_l2(target, background_hat, eps=0.5)

    torch.testing.assert_close(ratio, torch.tensor([1.0 / 1.5]))


def test_transport_target_ratio_includes_unit_gaussian_energy_per_dimension():
    target = torch.tensor([[[[3.0, 4.0]]], [[[2.0, 0.0]]]])
    background_hat = torch.tensor([[[[1.0, 2.0]]], [[[1.0, 1.0]]]])

    ratio = benchmark_metrics.transport_target_ratio(target, background_hat)

    torch.testing.assert_close(ratio, torch.tensor([(4.0 + 1.0) / (12.5 + 1.0), 2.0 / 3.0]))


def test_residual_low_frequency_ratio_adds_epsilon_to_structural_denominator():
    class _ControlledFilter:
        def __call__(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return torch.zeros_like(value), torch.full_like(value, 1e-6)

        def lowpass(self, value: torch.Tensor) -> torch.Tensor:
            return torch.ones_like(value)

    target = torch.zeros(1, 1, 1, 1)
    background_hat = torch.zeros_like(target)

    ratio = true_residual_low_frequency_ratio(_ControlledFilter(), target, background_hat)

    torch.testing.assert_close(ratio, torch.tensor([1.0 / 2e-6]))


def test_fused_condition_frequency_ratios_use_mean_absolute_magnitudes():
    filter_module = LowHighPassFilter(kernel_size=3)
    feature = torch.arange(25, dtype=torch.float32).reshape(1, 1, 5, 5) / 25
    low, high = filter_module(feature)
    low_magnitude = low.abs().flatten(1).mean(dim=1)
    high_magnitude = high.abs().flatten(1).mean(dim=1)

    low_ratio, high_ratio = _feature_low_high_ratios(filter_module, feature)

    denominator = (low_magnitude + high_magnitude).clamp_min(1e-6)
    torch.testing.assert_close(low_ratio, low_magnitude / denominator)
    torch.testing.assert_close(high_ratio, high_magnitude / denominator)


def test_reported_pixelfm_config_activates_only_documented_condition_terms():
    conf = OmegaConf.load(
        "bg_pdr_fm/configs/openfwi_lmdb_joint_full_contrastive_bgfm_pixelfm_predbg_e100.yaml"
    )

    assert conf.contrastive.symile_structural_weight == 1.0
    assert conf.contrastive.symile_numerical_weight == 1.0
    assert conf.contrastive.anchor_weight == 0.5
    assert conf.contrastive.reliability_prior_weight == 0.05
    assert conf.contrastive.sn_ortho_weight == 0.05
    assert conf.contrastive.unique_ortho_weight == 0.05
    for key in (
        "pairwise_structural_weight",
        "pairwise_numerical_weight",
        "structural_highpass_weight",
        "numerical_lowpass_weight",
    ):
        assert OmegaConf.select(conf, f"contrastive.{key}", default=0.0) == 0.0
