from __future__ import annotations

import pytest
import torch
from omegaconf import OmegaConf

from bg_pdr_fm.lightning import BGPDRFMLightning


def _conf(stage: str = "background"):
    return OmegaConf.create(
        {
            "model": {
                "hidden_channels": 8,
                "base_channels": 16,
                "feature_channels": 8,
                "cond_channels": 16,
                "background_hidden_channels": 16,
                "latent_channels": 4,
                "latent_hw": [18, 18],
                "codec_type": "simple",
                "codec_checkpoint": None,
                "lowpass_kernel": 5,
                "use_unique": False,
                "unique_eta": 0.1,
                "rho_tau": 1.0,
                "residual_backend": "simple_fm",
                "residual_num_train_timesteps": 16,
                "residual_num_inference_steps": 1,
                "residual_loss_type": "mse",
                "residual_background_source": "predicted",
            },
            "loss": {
                "bg_high_weight": 0.05,
                "rho_weight": 0.0,
                "final_l1_weight": 1.0,
                "target_lf_weight": 0.1,
            },
            "quality_gate": {"mode": "none"},
            "contrastive": {
                "embed_dim": 16,
                "temperature": 0.1,
                "symile_structural_weight": 1.0,
                "symile_numerical_weight": 1.0,
                "anchor_weight": 0.5,
                "reliability_prior_weight": 0.05,
                "sn_ortho_weight": 0.05,
                "unique_ortho_weight": 0.05,
                "min_symile_group_size": 2,
                "num_negative_shuffles": 4,
                "wavelet": {"level": 1, "boundary": "edge", "resize_mode": "nearest", "detail_fusion": "l2"},
            },
            "training": {
                "stage": stage,
                "lr": 1e-4,
                "freeze_encoder": False,
                "save_stage_outputs": False,
            },
            "diagnostics": {"enabled": False},
        }
    )


def _observed_mapping(batch_size: int = 1) -> dict[str, torch.Tensor]:
    torch.manual_seed(7)
    shape = (batch_size, 1, 70, 70)
    return {
        "migrated_image": torch.randn(shape),
        "horizon": torch.randn(shape),
        "rms_vel": torch.randn(shape),
        "well_log": torch.randn(shape),
        "well_mask": torch.ones(shape),
        "modality_mask": torch.ones(batch_size, 4),
        "modality_quality": torch.ones(batch_size, 4),
    }


@pytest.fixture(scope="module")
def background_model() -> BGPDRFMLightning:
    torch.manual_seed(1)
    return BGPDRFMLightning(_conf()).eval()


def test_predict_batch_accepts_observed_mapping_without_target(background_model, monkeypatch):
    def fail_if_called(*args, **kwargs):
        raise AssertionError("training-only wavelet anchors must not be used during prediction")

    monkeypatch.setattr(background_model.wavelet_anchor, "forward", fail_if_called)

    prediction = background_model.predict_batch(_observed_mapping())

    assert prediction.velocity_hat.shape == (1, 1, 70, 70)
    assert torch.isfinite(prediction.velocity_hat).all()


def test_observed_inference_batch_has_no_target_attribute():
    observed_batch = BGPDRFMLightning._as_inference_batch(_observed_mapping())

    assert not hasattr(observed_batch, "depth_vel")
    assert observed_batch.output_hw == (70, 70)


@pytest.mark.parametrize("forbidden_key", ("depth_vel", "target", "target_quality", "A_bg", "A_str"))
def test_predict_batch_rejects_target_derived_mapping_keys(background_model, forbidden_key):
    observed = _observed_mapping()
    observed[forbidden_key] = torch.zeros_like(observed["horizon"])

    with pytest.raises(ValueError, match="target-derived|observed-only|forbidden"):
        background_model.predict_batch(observed)


@pytest.mark.parametrize(
    ("residual_background_source", "residual_override"),
    (("oracle_lowpass", "none"), ("predicted", "oracle"), ("predicted", "oracle_latent")),
)
def test_predict_batch_rejects_oracle_residual_configuration(
    residual_background_source: str,
    residual_override: str,
):
    conf = _conf(stage="residual")
    conf.model.residual_background_source = residual_background_source
    conf.evaluation = {"residual_override": residual_override}
    model = BGPDRFMLightning(conf).eval()

    with pytest.raises(ValueError, match="predicted|oracle"):
        model.predict_batch(_observed_mapping())
