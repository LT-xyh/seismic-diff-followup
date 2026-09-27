from __future__ import annotations

from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from bg_pdr_fm.data.batch import BGSampleBatch
from bg_pdr_fm.evaluation.missing_modalities import apply_missing_modality_mode
from bg_pdr_fm.external_baselines.common import build_multimodal_condition_image


def _batch(batch_size: int = 1) -> BGSampleBatch:
    depth = torch.linspace(-1.0, 1.0, 70 * 70).reshape(1, 1, 70, 70).repeat(batch_size, 1, 1, 1)
    migrated = torch.ones(batch_size, 1, 1000, 70)
    horizon = torch.full((batch_size, 1, 70, 70), 2.0)
    rms = torch.full((batch_size, 1, 1000, 70), 3.0)
    well = torch.full((batch_size, 1, 70, 70), 4.0)
    well_mask = torch.ones(batch_size, 1, 70, 70)
    return BGSampleBatch(
        depth_vel=depth,
        migrated_image=migrated,
        horizon=horizon,
        rms_vel=rms,
        well_log=well,
        well_mask=well_mask,
        modality_mask=torch.ones(batch_size, 4),
        modality_quality=torch.ones(batch_size, 4),
    )


def _params(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def _benchmark_conf(variant: str):
    return OmegaConf.create(
        {
            "benchmark": {"variant": variant, "capacity_tier": "adapted_external"},
            "model": {
                "feature_channels": 8,
                "cond_channels": 16,
                "latent_channels": 4,
                "latent_hw": [18, 18],
                "codec_type": "simple",
                "lowpass_kernel": 5,
            },
            "quality_gate": {"mode": "none"},
            "training": {
                "stage": "residual",
                "freeze_encoder": True,
                "load_stage_checkpoint": None,
                "lr": 1e-4,
                "weight_decay": 1e-4,
                "max_epochs": 100,
            },
        }
    )


def test_openfwi_official_architecture_parameter_fingerprints():
    from bg_pdr_fm.external_baselines.openfwi_official import (
        OpenFWIDiscriminator,
        OpenFWIInversionNet,
        OpenFWIUPFWI,
    )

    assert _params(OpenFWIInversionNet()) == 24_409_123
    assert _params(OpenFWIDiscriminator()) == 1_180_003
    assert _params(OpenFWIUPFWI()) == 18_996_259


def test_official_adapted_wrappers_preserve_shapes_and_capacity():
    from bg_pdr_fm.external_baselines.inversion_net import AdaptedInversionNetMultimodal
    from bg_pdr_fm.external_baselines.upfwi import AdaptedUPFWIMultimodal
    from bg_pdr_fm.external_baselines.velocity_gan import VelocityGANBaseline

    batch = _batch(2)
    inversion_net = AdaptedInversionNetMultimodal()
    upfwi = AdaptedUPFWIMultimodal()
    velocity_gan = VelocityGANBaseline()

    assert inversion_net(batch).shape == batch.depth_vel.shape
    assert upfwi(batch).shape == batch.depth_vel.shape
    assert velocity_gan(batch).shape == batch.depth_vel.shape
    assert _params(inversion_net) == 24_409_123
    assert _params(upfwi) == 18_996_259
    assert _params(velocity_gan.generator) == 24_409_123
    assert _params(velocity_gan.discriminator) == 1_180_003
    assert _params(velocity_gan) == 25_589_126


def test_multimodal_input_builder_zeros_missing_channels():
    batch = _batch()
    pstm_only = apply_missing_modality_mode(batch, "PSTM only")
    conditions = build_multimodal_condition_image(pstm_only)

    assert conditions.shape == (1, 5, 1000, 70)
    assert torch.count_nonzero(conditions[:, 0]).item() > 0
    assert torch.count_nonzero(conditions[:, 1:]).item() == 0


def test_velocity_gan_official_losses_are_finite():
    from bg_pdr_fm.external_baselines.velocity_gan import VelocityGANBaseline

    batch = _batch(2)
    model = VelocityGANBaseline(
        l1_weight=100.0,
        mse_weight=0.0,
        adversarial_weight=1.0,
        gradient_penalty_weight=10.0,
        n_critic=5,
    )
    generator = model.generator_loss(batch)
    discriminator = model.discriminator_loss(batch)

    for name in ("loss", "l1", "mse", "adversarial"):
        assert torch.isfinite(generator[name]), name
    for name in ("loss", "critic_gap", "gradient_penalty"):
        assert torch.isfinite(discriminator[name]), name
    assert model.should_update_generator(batch_idx=4, num_batches=100)
    assert not model.should_update_generator(batch_idx=3, num_batches=100)
    assert model.should_update_generator(batch_idx=99, num_batches=100)


def test_official_variants_report_paper_parameter_counts():
    from bg_pdr_fm.lightning.benchmark_module import AAAI27BenchmarkLightning

    inversion = AAAI27BenchmarkLightning(_benchmark_conf("adapted_inversion_net"))
    velocity_gan = AAAI27BenchmarkLightning(_benchmark_conf("velocity_gan"))
    upfwi = AAAI27BenchmarkLightning(_benchmark_conf("adapted_upfwi"))

    assert inversion.benchmark_metadata()["reported_params"] == 24_409_123
    assert upfwi.benchmark_metadata()["reported_params"] == 18_996_259
    gan_metadata = velocity_gan.benchmark_metadata()
    assert gan_metadata["generator_params"] == 24_409_123
    assert gan_metadata["discriminator_params"] == 1_180_003
    assert gan_metadata["reported_params"] == 25_589_126


def test_velocity_gan_uses_manual_official_optimizers():
    from bg_pdr_fm.lightning.benchmark_module import AAAI27BenchmarkLightning

    model = AAAI27BenchmarkLightning(_benchmark_conf("velocity_gan"))
    configured = model.configure_optimizers()
    optimizers = configured if isinstance(configured, (list, tuple)) else [configured]

    assert model.automatic_optimization is False
    assert len(optimizers) == 2
    assert all(isinstance(optimizer, torch.optim.AdamW) for optimizer in optimizers)
    assert optimizers[0].param_groups[0]["betas"] == (0.0, 0.9)
    assert optimizers[1].param_groups[0]["betas"] == (0.0, 0.9)


def test_official_e100_configs_use_isolated_artifact_roots():
    from bg_pdr_fm.training.benchmark_config import load_benchmark_config
    from bg_pdr_fm.training.train_aaai27_benchmark import validate_benchmark_config

    cases = {
        "adapted_inversion_net.yaml": "adapted_inversion_net_official_e100",
        "velocity_gan.yaml": "velocity_gan_official_e100",
        "adapted_upfwi.yaml": "adapted_upfwi_official_e100",
    }
    for filename, expected_root in cases.items():
        conf = load_benchmark_config(f"bg_pdr_fm/configs/experiments/aaai27/{filename}")
        validate_benchmark_config(conf)
        assert conf.training.max_epochs == 100
        assert conf.training.batch_size == 64
        assert conf.training.precision == "32-true"
        assert expected_root in str(conf.training.checkpoint.dirpath)


def test_official_pipeline_builds_only_missing_steps(tmp_path):
    from bg_pdr_fm.training.run_aaai27_official_pipeline import build_pipeline_commands

    train_config = tmp_path / "train.yaml"
    eval_config = tmp_path / "eval.yaml"
    checkpoint = tmp_path / "last.ckpt"
    summary = tmp_path / "summary.json"

    commands = build_pipeline_commands(
        train_config=train_config,
        eval_config=eval_config,
        checkpoint=checkpoint,
        summary=summary,
        python="python",
    )
    assert [command[2] for command in commands] == [
        "bg_pdr_fm.training.train_aaai27_benchmark",
        "bg_pdr_fm.evaluation.compare_experiments",
    ]

    checkpoint.touch()
    commands = build_pipeline_commands(
        train_config=train_config,
        eval_config=eval_config,
        checkpoint=checkpoint,
        summary=summary,
        python="python",
    )
    assert [command[2] for command in commands] == ["bg_pdr_fm.evaluation.compare_experiments"]

    summary.touch()
    assert build_pipeline_commands(
        train_config=train_config,
        eval_config=eval_config,
        checkpoint=checkpoint,
        summary=summary,
        python="python",
    ) == []


def test_official_checkpoint_validation_rejects_partial_and_legacy_shapes():
    from bg_pdr_fm.evaluation.compare_experiments import _validate_official_checkpoint_state

    class TinyOfficialWrapper(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.adapted_inversion_net = torch.nn.Linear(3, 2)

    model = TinyOfficialWrapper()
    path = Path("legacy.ckpt")
    state = model.state_dict()
    partial = {key: value for key, value in state.items() if not key.endswith("bias")}
    with pytest.raises(RuntimeError, match="missing .* critical adapted_inversion_net keys"):
        _validate_official_checkpoint_state(
            model,
            partial,
            variant="adapted_inversion_net",
            path=path,
        )

    legacy_shape = dict(state)
    legacy_shape["adapted_inversion_net.weight"] = torch.zeros(1, 1)
    with pytest.raises(RuntimeError, match="shape mismatches"):
        _validate_official_checkpoint_state(
            model,
            legacy_shape,
            variant="adapted_inversion_net",
            path=path,
        )
