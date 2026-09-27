from __future__ import annotations

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from bg_pdr_fm.data import SyntheticBGDataset, collate_bg_samples
from bg_pdr_fm.evaluation.background_feature_probe import BackgroundProbe
from bg_pdr_fm.evaluation.evaluate_bg_pdr_fm import load_evaluation_checkpoints
from bg_pdr_fm.lightning import BGPDRFMLightning


def _batch(batch_size: int = 2):
    dataset = SyntheticBGDataset(length=batch_size)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_bg_samples)
    return next(iter(loader))


def _conf() -> OmegaConf:
    return OmegaConf.create(
        {
            "model": {
                "hidden_channels": 8,
                "base_channels": 16,
                "feature_channels": 8,
                "cond_channels": 16,
                "background_hidden_channels": 8,
                "background_backend": "unet_direct",
                "background_condition_source": "encoder_numerical_raw_bypass",
                "background_raw_bypass_modalities": ["rms_vel", "horizon"],
                "background_raw_bypass_channels": 4,
                "background_output_mode": "direct",
                "latent_channels": 4,
                "latent_hw": [18, 18],
                "codec_type": "simple",
                "codec_checkpoint": None,
                "lowpass_kernel": 5,
                "use_unique": False,
                "unique_eta": 0.1,
                "rho_tau": 1.0,
                "residual_backend": "simple_fm",
                "residual_num_train_timesteps": 1000,
                "residual_num_inference_steps": 2,
                "residual_loss_type": "mse",
            },
            "loss": {
                "bg_l1_weight": 1.0,
                "bg_l2_weight": 0.5,
                "bg_high_weight": 0.0,
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
                "stage": "background",
                "lr": 1e-4,
                "freeze_encoder": False,
                "load_stage_checkpoint": None,
                "save_stage_outputs": False,
            },
        }
    )


def test_background_probe_outputs_full_resolution_prediction():
    probe = BackgroundProbe(in_channels=8, kind="mlp_1x1", hidden_channels=12)
    out = probe(torch.randn(2, 8, 70, 70))

    assert out.shape == (2, 1, 70, 70)
    assert torch.isfinite(out).all()


def test_raw_bypass_background_condition_matches_unet_input_channels():
    batch = _batch()
    conf = _conf()
    model = BGPDRFMLightning(conf)

    features = model.encoder(batch)
    condition = model._background_condition(batch, features, None)

    assert condition.shape == (2, 12, 70, 70)
    assert model.background.network.enc.net[0].in_channels == condition.shape[1]


def test_background_numhead_tune_only_unfreezes_numerical_heads_and_background():
    conf = _conf()
    conf.model.background_condition_source = "encoder_numerical"
    conf.training.tune_encoder_numerical_heads_for_background = True
    model = BGPDRFMLightning(conf)

    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}

    assert any(name.startswith("background.") for name in trainable)
    assert any(".numerical." in name and name.startswith("encoder.heads.") for name in trainable)
    assert not any(name.startswith("encoder.backbones.") for name in trainable)
    assert not any(".structural." in name and name.startswith("encoder.heads.") for name in trainable)
    assert not any(".unique." in name and name.startswith("encoder.heads.") for name in trainable)


def _checkpoint(path, model: BGPDRFMLightning) -> None:
    torch.save({"state_dict": model.state_dict()}, path)


def test_evaluation_loads_raw_background_bypass_weights(tmp_path):
    conf = _conf()
    trained = BGPDRFMLightning(conf)
    with torch.no_grad():
        trained.background_raw_bypass[0].weight.fill_(0.25)
        trained.background_raw_bypass[0].bias.fill_(0.5)
    checkpoint = tmp_path / "background.ckpt"
    _checkpoint(checkpoint, trained)

    eval_model = BGPDRFMLightning(conf)
    with torch.no_grad():
        eval_model.background_raw_bypass[0].weight.zero_()
        eval_model.background_raw_bypass[0].bias.zero_()
    conf.evaluation = {"checkpoints": {"background": str(checkpoint)}}

    loaded = load_evaluation_checkpoints(eval_model, conf)

    assert any(item["mode"] == "background" for item in loaded)
    assert torch.allclose(eval_model.background_raw_bypass[0].weight, trained.background_raw_bypass[0].weight)
    assert torch.allclose(eval_model.background_raw_bypass[0].bias, trained.background_raw_bypass[0].bias)


def test_evaluation_loads_background_tuned_numerical_heads(tmp_path):
    conf = _conf()
    conf.model.background_condition_source = "encoder_numerical"
    conf.training.tune_encoder_numerical_heads_for_background = True
    trained = BGPDRFMLightning(conf)
    with torch.no_grad():
        trained.encoder.heads["rms_vel"].numerical.weight.fill_(0.125)
        trained.encoder.heads["rms_vel"].numerical.bias.fill_(0.375)
    checkpoint = tmp_path / "background.ckpt"
    _checkpoint(checkpoint, trained)

    eval_model = BGPDRFMLightning(conf)
    with torch.no_grad():
        eval_model.encoder.heads["rms_vel"].numerical.weight.zero_()
        eval_model.encoder.heads["rms_vel"].numerical.bias.zero_()
    conf.evaluation = {"checkpoints": {"background": str(checkpoint)}}

    loaded = load_evaluation_checkpoints(eval_model, conf)

    assert any(item["mode"] == "background" for item in loaded)
    assert torch.allclose(
        eval_model.encoder.heads["rms_vel"].numerical.weight,
        trained.encoder.heads["rms_vel"].numerical.weight,
    )
    assert torch.allclose(
        eval_model.encoder.heads["rms_vel"].numerical.bias,
        trained.encoder.heads["rms_vel"].numerical.bias,
    )
