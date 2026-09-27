from __future__ import annotations

import torch

from bg_pdr_fm.models.generators import ResidualFlowGenerator
from bg_pdr_fm.models.residual_backends import build_residual_backend


def test_diffusers_crossattn_backend_trains_and_samples_with_spatial_tokens():
    backend = build_residual_backend(
        backend="diffusers_unet_crossattn_fm",
        latent_channels=4,
        cond_channels=36,
        latent_hw=(18, 18),
        num_train_timesteps=1000,
        hidden_channels=64,
        block_out_channels=(32, 64),
        layers_per_block=1,
        cross_attention_dim=64,
        attention_head_dim=8,
    )
    z_res = torch.randn(2, 4, 18, 18)
    cond = torch.randn(2, 36, 18, 18)

    output = backend.training_loss(z_res, cond)
    sample = backend.sample(cond, x_size=tuple(z_res.shape), steps=2)

    assert torch.isfinite(output["loss"])
    assert output["z_res_pred"].shape == z_res.shape
    assert output["velocity_pred"].shape == z_res.shape
    assert sample.shape == z_res.shape
    assert torch.isfinite(sample).all()


def test_residual_generator_can_merge_structural_condition_with_raw_codec_latent():
    generator = ResidualFlowGenerator(
        cond_channels=32,
        latent_channels=4,
        latent_hw=(18, 18),
        backend="simple_fm",
        backend_hidden_channels=8,
        condition_merge="concat",
        background_context_source="codec_latent_raw",
    )
    cond_s = torch.randn(2, 32, 18, 18)
    bg_hat = torch.randn(2, 1, 70, 70)
    z_bg = torch.randn(2, 4, 18, 18)

    merged = generator.merged_condition(cond_s, bg_hat, z_bg=z_bg)

    assert merged.shape == (2, 36, 18, 18)
    assert torch.allclose(merged[:, :32], cond_s)
    assert torch.allclose(merged[:, 32:], z_bg)
