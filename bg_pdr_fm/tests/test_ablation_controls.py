from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from bg_pdr_fm.lightning.stage_losses import select_residual_target_background
from bg_pdr_fm.models.filters import LowHighPassFilter
from bg_pdr_fm.models.generators import ResidualFlowGenerator
from bg_pdr_fm.training.dispatch_pd_bgrfm_ablations import DEFAULT_BASE, build_main_config


def test_residual_generator_none_context_keeps_structural_condition_shape():
    generator = ResidualFlowGenerator(
        cond_channels=4,
        latent_channels=1,
        latent_hw=(8, 8),
        backend="unet_fm_film",
        backend_hidden_channels=8,
        condition_merge="concat",
        background_context_source="none",
    )
    condition = torch.randn(2, 4, 8, 8)
    background = torch.randn(2, 1, 8, 8)

    merged = generator.merged_condition(condition, background)

    assert merged is condition
    assert generator.backend.field.down_block.cond_proj.in_channels == 4


def test_lowpass_residual_target_does_not_replace_predicted_context():
    filter_module = LowHighPassFilter(kernel_size=5)
    module = SimpleNamespace(
        conf=OmegaConf.create({"training": {"ablation": {"residual_target_source": "lowpass"}}}),
        filter=filter_module,
    )
    velocity = torch.randn(2, 1, 16, 16)
    predicted = torch.zeros_like(velocity)

    target = select_residual_target_background(module, velocity, predicted, predicted)

    assert torch.allclose(target, filter_module.lowpass(velocity))
    assert torch.count_nonzero(predicted) == 0


def test_main_ablation_config_preserves_original_stage_warm_starts(tmp_path):
    base = OmegaConf.load(DEFAULT_BASE)
    original_stage = str(base.training.load_stage_checkpoint)
    original_background = str(base.training.joint.warm_start_checkpoints.background)

    conf = build_main_config(
        base,
        "test_ablation",
        tmp_path / "test_ablation",
        epochs=3,
        batch_size=64,
        accumulate_grad_batches=4,
    )

    assert str(conf.training.load_stage_checkpoint) == original_stage
    assert str(conf.training.joint.warm_start_checkpoints.background) == original_background
    assert conf.training.batch_size == 64
    assert conf.training.accumulate_grad_batches == 4
