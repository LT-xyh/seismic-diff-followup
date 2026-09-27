from omegaconf import OmegaConf

from bg_pdr_fm.training.train_bg_pdr_fm import VALID_STAGES


def test_joint_full_is_valid_training_stage():
    assert "joint_full" in VALID_STAGES


def test_joint_full_config_defaults_are_addressable():
    conf = OmegaConf.create(
        {
            "training": {
                "stage": "joint_full",
                "joint": {
                    "lambda_contrastive": 0.01,
                    "encoder_backbone_lr": 1e-6,
                    "encoder_head_lr": 3e-6,
                    "background_lr": 1e-4,
                    "residual_lr": 5e-5,
                },
            }
        }
    )
    assert conf.training.stage == "joint_full"
    assert conf.training.joint.lambda_contrastive == 0.01


def test_joint_full_expected_loss_keys_documented():
    expected = {
        "loss",
        "joint_contrastive_loss",
        "joint_background_loss",
        "joint_residual_loss",
        "joint_lambda_contrastive",
    }
    assert "joint_contrastive_loss" in expected
    assert "joint_background_loss" in expected
    assert "joint_residual_loss" in expected


def test_joint_full_lr_values_match_confirmed_design():
    conf = OmegaConf.create(
        {
            "training": {
                "stage": "joint_full",
                "joint": {
                    "encoder_backbone_lr": 1e-6,
                    "encoder_head_lr": 3e-6,
                    "contrastive_lr": 3e-6,
                    "background_lr": 1e-4,
                    "residual_lr": 5e-5,
                },
            }
        }
    )
    assert conf.training.joint.encoder_backbone_lr == 1e-6
    assert conf.training.joint.encoder_head_lr == 3e-6
    assert conf.training.joint.background_lr == 1e-4
    assert conf.training.joint.residual_lr == 5e-5
