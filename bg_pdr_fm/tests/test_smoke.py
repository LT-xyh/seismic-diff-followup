from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from bg_pdr_fm.data import (
    MarmousiBGDataset,
    OpenFWIBGDataset,
    SyntheticBGDataset,
    batch_to_device,
    collate_bg_samples,
    resolve_marmousi_root,
    resolve_openfwi_root,
    validate_bg_batch,
)
from bg_pdr_fm.evaluation import load_evaluation_checkpoints, run_contrastive_visualization, run_evaluation
from bg_pdr_fm.evaluation.benchmark_metrics import compute_freq_metrics
from bg_pdr_fm.evaluation.compare_experiments import run_benchmark_evaluation
from bg_pdr_fm.evaluation.missing_modalities import AAAI27_MISSING_MODES, apply_missing_modality_mode
from bg_pdr_fm.external_baselines.auto_linear import build_auto_linear_measurement_input
from bg_pdr_fm.external_baselines.auto_linear.original_architecture import AdaptedAutoLinearOriginalMultimodal
from bg_pdr_fm.external_baselines.common import build_multimodal_condition_image
from bg_pdr_fm.external_baselines.gfi import build_adapted_gfi_input
from bg_pdr_fm.external_baselines.gfi.adapted_latent_unet import AdaptedGFILatentUNetMultimodal
from bg_pdr_fm.lightning import BGPDRFMLightning, SeismicAutoencoderKLLightning
from bg_pdr_fm.lightning.benchmark_module import AAAI27BenchmarkLightning
from bg_pdr_fm.models import (
    AutoencoderLatentCodec,
    BackgroundEstimator,
    ConditionAdapters,
    DepthVelocityWaveletTransform,
    HorizonEncoderA,
    IdentityLatentCodec,
    LegacyLatentFlowMatchingBackend,
    PhysicsAnchoredSubsetSymileLoss,
    PhysicsDecoupledEncoder,
    RMSVelocityEncoderA,
    ResidualFlowGenerator,
    SeismicImageEncoderA,
    SimpleLatentCodec,
    WellLogEncoderA,
)
from bg_pdr_fm.models.types import DecoupledFeatures
from bg_pdr_fm.models.generators import haar_detail_leak
from bg_pdr_fm.lightning.stage_losses import background_stage_loss, residual_stage_loss
from bg_pdr_fm.training.dispatch_background_night_queue import choose_next_branch
import bg_pdr_fm.training.train_bg_pdr_fm as train_bg_pdr_fm
from bg_pdr_fm.training.benchmark_config import load_benchmark_config
from bg_pdr_fm.training.train_bg_pdr_fm import apply_fast_run_overrides, run_one_stage
from bg_pdr_fm.training.run_background import load_background_config
from bg_pdr_fm.training.run_contrastive import load_contrastive_config
from bg_pdr_fm.training.run_residual import load_residual_config


AUTOENCODER_CHECKPOINT = Path("checkpoints/bg_pdr_fm/seismic_autoencoder_kl.ckpt")


def _batch(batch_size: int = 2):
    dataset = SyntheticBGDataset(length=batch_size)
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_bg_samples)
    return next(iter(loader))


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
                "residual_num_train_timesteps": 1000,
                "residual_num_inference_steps": 2,
                "residual_loss_type": "mse",
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
                "load_stage_checkpoint": None,
                "save_stage_outputs": False,
            },
        }
    )


def _write_openfwi_fixture(root: Path, count: int = 4, time_domain: bool = False, include_all: bool = True) -> None:
    dataset_dir = root / "FlatVelA"
    modalities = ("depth_vel", "migrated_image", "horizon", "rms_vel") if include_all else ("depth_vel",)
    for modality in modalities:
        (dataset_dir / modality).mkdir(parents=True, exist_ok=True)
    for idx in range(count):
        depth = np.full((1, 70, 70), 2500.0 + idx, dtype=np.float32)
        np.save(dataset_dir / "depth_vel" / f"{idx}.npy", depth)
        if not include_all:
            continue
        image_shape = (1, 1000, 70) if time_domain else (1, 70, 70)
        migrated = np.full(image_shape, float(idx), dtype=np.float32)
        rms = np.full(image_shape, 2300.0 + idx, dtype=np.float32)
        horizon = np.zeros((1, 70, 70), dtype=np.float32)
        np.save(dataset_dir / "migrated_image" / f"{idx}.npy", migrated)
        np.save(dataset_dir / "rms_vel" / f"{idx}.npy", rms)
        np.save(dataset_dir / "horizon" / f"{idx}.npy", horizon)


def _record_ids(dataset: OpenFWIBGDataset) -> set[tuple[str, int]]:
    return {(record["dataset_name"], int(record["sample_index"])) for record in dataset.records}


def _grads_are_none_or_zero(parameters) -> bool:
    return all(parameter.grad is None or torch.count_nonzero(parameter.grad).item() == 0 for parameter in parameters)


def test_dataset_adapter_smoke():
    batch = _batch()
    assert batch.depth_vel.shape == (2, 1, 70, 70)
    assert batch.well_mask.shape == batch.depth_vel.shape
    assert batch.modality_mask.shape == (2, 4)
    assert batch.modality_quality.dtype == torch.float32
    assert torch.all(batch.modality_quality[:, 3] == 1)


def test_batch_contract_allows_time_domain_modalities():
    items = []
    for _ in range(2):
        depth = torch.randn(1, 70, 70)
        well_mask = torch.zeros_like(depth)
        items.append(
            {
                "depth_vel": depth,
                "migrated_image": torch.randn(1, 1000, 70),
                "horizon": torch.zeros_like(depth),
                "rms_vel": torch.randn(1, 1000, 70),
                "well_log": torch.zeros_like(depth),
                "well_mask": well_mask,
                "modality_mask": torch.ones(4),
                "modality_quality": torch.ones(4),
            }
        )

    batch = collate_bg_samples(items)
    validate_bg_batch(batch, context="mixed-domain batch")
    assert batch.depth_vel.shape == (2, 1, 70, 70)
    assert batch.migrated_image.shape == (2, 1, 1000, 70)
    assert batch.rms_vel.shape == (2, 1, 1000, 70)


def test_physical_modality_encoder_shapes():
    seismic = SeismicImageEncoderA(c1=8, c2=16, c3=24, c4=32, C_out=8, out_hw=(70, 70), num_heads=4)
    rms = RMSVelocityEncoderA(
        trace_channels=8,
        latent_time_steps=12,
        c_mid=16,
        c_out=8,
        n_trace_blocks=1,
        n_lateral_blocks=1,
        n_refine2d=1,
        out_hw=(70, 70),
    )
    horizon = HorizonEncoderA(
        C_out=8,
        c1=8,
        c2=16,
        c3=16,
        token_hw=(7, 7),
        num_global_layers=1,
        num_heads=4,
        out_hw=(70, 70),
    )
    well = WellLogEncoderA(
        C_t=8,
        C2d_1=16,
        C2d_2=16,
        C_out=8,
        trace_latent_steps=12,
        n_trace_blocks=1,
        n_lateral_blocks=1,
        n_dense_blocks=1,
        out_hw=(70, 70),
    )

    seismic.eval()
    rms.eval()
    horizon.eval()
    well.eval()
    with torch.no_grad():
        assert seismic(torch.randn(1, 1, 1000, 70)).shape == (1, 8, 70, 70)
        assert rms(torch.randn(1, 1, 1000, 70)).shape == (1, 8, 70, 70)
        assert horizon(torch.randn(1, 1, 70, 70)).shape == (1, 8, 70, 70)
        well_log = torch.randn(1, 1, 70, 70)
        well_mask = torch.zeros_like(well_log)
        well_mask[:, :, :, [10, 35, 60]] = 1.0
        assert well(well_log, valid_mask=well_mask).shape == (1, 8, 70, 70)


def test_encoder_aligns_time_domain_features_to_depth_grid():
    items = []
    for _ in range(2):
        depth = torch.randn(1, 70, 70)
        items.append(
            {
                "depth_vel": depth,
                "migrated_image": torch.randn(1, 1000, 70),
                "horizon": torch.randn(1, 70, 70),
                "rms_vel": torch.randn(1, 1000, 70),
                "well_log": torch.zeros_like(depth),
                "well_mask": torch.zeros_like(depth),
                "modality_mask": torch.ones(4),
                "modality_quality": torch.ones(4),
            }
        )

    batch = collate_bg_samples(items)
    encoder = PhysicsDecoupledEncoder(hidden_channels=8, base_channels=16, feature_channels=8)
    features = encoder(batch)
    assert features.numerical.shape == (2, 8, 70, 70)
    assert features.structural.shape == (2, 8, 70, 70)
    assert features.unique.shape == (2, 8, 70, 70)


def test_config_templates_load_and_preserve_shape_contract():
    config_dir = Path("bg_pdr_fm/configs")
    names = [
        "bg_pdr_fm.yaml",
        "fast_run.yaml",
        "openfwi_lmdb_train.yaml",
        "openfwi_lmdb_contrastive.yaml",
        "openfwi_lmdb_background.yaml",
        "openfwi_lmdb_residual.yaml",
        "marmousi_train.yaml",
        "eval.yaml",
        "autoencoder_train.yaml",
        "autoencoder_fast_run.yaml",
        "autoencoder_eval.yaml",
    ]
    configs = {name: OmegaConf.load(config_dir / name) for name in names}

    for conf in configs.values():
        if "data" in conf:
            assert conf.data.target_shape is None
            assert list(conf.data.split_fractions) == [0.7, 0.2, 0.1]
        if "model" in conf:
            assert conf.model.use_unique is False
        if "quality_gate" in conf:
            assert conf.quality_gate.mode == "none"
        if "loss" in conf:
            if "rho_weight" in conf.loss:
                assert conf.loss.rho_weight == 0.0
            assert conf.loss.target_lf_weight == 0.1
    assert configs["bg_pdr_fm.yaml"].training.fast_run is False
    assert configs["fast_run.yaml"].model.codec_type == "autoencoder"
    assert configs["openfwi_lmdb_background.yaml"].model.codec_type == "autoencoder"
    assert configs["openfwi_lmdb_residual.yaml"].model.codec_type == "autoencoder"
    pixel_conf = OmegaConf.load(
        config_dir / "openfwi_lmdb_residual_smooth_hc64_nogate_pixelfm_film_predbg_e100.yaml"
    )
    assert pixel_conf.model.codec_type == "identity"
    assert pixel_conf.model.latent_channels == 1
    assert list(pixel_conf.model.latent_hw) == [70, 70]
    assert pixel_conf.model.residual_backend == "unet_fm_film"
    assert pixel_conf.model.residual_condition_merge == "concat"
    assert pixel_conf.model.codec_checkpoint is None
    assert pixel_conf.training.load_stage_checkpoint == (
        "logs/bg_pdr_fm/residual_train_smooth_hc64_nogate_unetfm_film_predbg_e100/stage_checkpoints/"
        "residual_last.ckpt"
    )
    h192_pixel_conf = OmegaConf.load(
        config_dir / "openfwi_lmdb_residual_unet_direct_h192_pixelfm_film_predbg_e100.yaml"
    )
    assert h192_pixel_conf.model.codec_type == "identity"
    assert h192_pixel_conf.model.latent_channels == 1
    assert list(h192_pixel_conf.model.latent_hw) == [70, 70]
    assert h192_pixel_conf.model.codec_checkpoint is None
    assert h192_pixel_conf.model.background_backend == "unet_direct"
    assert h192_pixel_conf.model.background_condition_source == "encoder_numerical"
    assert h192_pixel_conf.model.background_hidden_channels == 192
    assert h192_pixel_conf.model.residual_backend == "unet_fm_film"
    assert h192_pixel_conf.model.residual_condition_merge == "concat"
    assert h192_pixel_conf.training.load_stage_checkpoint == (
        "logs/bg_pdr_fm/background_train_unet_direct_h192_l1l2_e100/stage_checkpoints/background_last.ckpt"
    )
    joint_pixel_conf = OmegaConf.load(
        config_dir / "openfwi_lmdb_joint_full_contrastive_bgfm_pixelfm_predbg_e100.yaml"
    )
    assert joint_pixel_conf.training.stage == "joint_full"
    assert joint_pixel_conf.model.codec_type == "identity"
    assert joint_pixel_conf.model.latent_channels == 1
    assert list(joint_pixel_conf.model.latent_hw) == [70, 70]
    assert joint_pixel_conf.model.codec_checkpoint is None
    assert joint_pixel_conf.model.background_backend == "unet_direct"
    assert joint_pixel_conf.model.background_condition_source == "encoder_numerical"
    assert joint_pixel_conf.model.background_hidden_channels == 192
    assert joint_pixel_conf.model.residual_backend == "unet_fm_film"
    assert joint_pixel_conf.model.residual_background_context_source == "codec_latent"
    assert joint_pixel_conf.training.joint.warm_start_checkpoints.background == (
        "logs/bg_pdr_fm/background_train_unet_direct_h192_l1l2_e100/stage_checkpoints/background_last.ckpt"
    )
    assert joint_pixel_conf.training.joint.warm_start_checkpoints.residual == (
        "logs/bg_pdr_fm/residual_train_unet_direct_h192_pixelfm_film_predbg_e100/stage_checkpoints/residual_last.ckpt"
    )
    bg_fm_conf = OmegaConf.load(config_dir / "openfwi_lmdb_background_fm_h128_l1l2_e100_bs200.yaml")
    assert bg_fm_conf.training.stage == "background"
    assert bg_fm_conf.training.devices == 2
    assert bg_fm_conf.training.batch_size == 200
    assert bg_fm_conf.model.background_backend == "flow_matching"
    assert bg_fm_conf.model.background_condition_source == "encoder_numerical"
    assert bg_fm_conf.model.background_fm_backend == "unet_fm_film"
    assert bg_fm_conf.model.background_fm_hidden_channels == 128
    assert list(bg_fm_conf.model.background_fm_latent_hw) == [70, 70]
    assert bg_fm_conf.loss.bg_fm_weight == 1.0
    assert bg_fm_conf.loss.bg_l1_weight == 1.0
    assert bg_fm_conf.loss.bg_l2_weight == 0.5
    joint_bg_fm_conf = OmegaConf.load(
        config_dir / "openfwi_lmdb_joint_full_contrastive_bgfm_fmbackground_pixelfm_predbg_e100_bs100.yaml"
    )
    assert joint_bg_fm_conf.training.stage == "joint_full"
    assert joint_bg_fm_conf.training.devices == 2
    assert joint_bg_fm_conf.training.batch_size == 100
    assert joint_bg_fm_conf.model.codec_type == "identity"
    assert joint_bg_fm_conf.model.background_backend == "flow_matching"
    assert joint_bg_fm_conf.model.background_fm_backend == "unet_fm_film"
    assert joint_bg_fm_conf.model.residual_backend == "unet_fm_film"
    assert joint_bg_fm_conf.model.background_condition_source == "encoder_numerical"
    assert joint_bg_fm_conf.model.residual_background_context_source == "codec_latent"
    assert joint_bg_fm_conf.training.joint.background_lr == 1.0e-4
    assert joint_bg_fm_conf.training.joint.residual_lr == 5.0e-5
    assert configs["openfwi_lmdb_train.yaml"].training.stage == "background"
    assert configs["openfwi_lmdb_contrastive.yaml"].training.stage == "contrastive"
    assert configs["openfwi_lmdb_background.yaml"].training.stage == "background"
    assert configs["openfwi_lmdb_residual.yaml"].training.stage == "residual"
    assert configs["openfwi_lmdb_residual.yaml"].training.load_stage_checkpoint == (
        "logs/bg_pdr_fm/background_train/stage_checkpoints/background_last.ckpt"
    )
    assert configs["openfwi_lmdb_contrastive.yaml"].training.fast_run is False
    assert configs["openfwi_lmdb_contrastive.yaml"].training.fast_run_stages is None
    assert list(configs["openfwi_lmdb_contrastive.yaml"].training.devices) == [4, 5, 6, 7]
    assert configs["openfwi_lmdb_contrastive.yaml"].training.strategy == "ddp"
    assert configs["openfwi_lmdb_contrastive.yaml"].training.num_nodes == 1
    assert configs["openfwi_lmdb_contrastive.yaml"].training.sync_batchnorm is False
    assert configs["openfwi_lmdb_contrastive.yaml"].training.use_distributed_sampler is True
    for name in (
        "bg_pdr_fm.yaml",
        "fast_run.yaml",
        "openfwi_lmdb_train.yaml",
        "openfwi_lmdb_contrastive.yaml",
        "openfwi_lmdb_background.yaml",
        "openfwi_lmdb_residual.yaml",
        "marmousi_train.yaml",
        "eval.yaml",
    ):
        assert configs[name].training.precision == "bf16-mixed"
        assert configs[name].training.matmul_precision == "medium"
        assert "logging" in configs[name].training
        assert "checkpoint" in configs[name].training
    assert configs["openfwi_lmdb_contrastive.yaml"].contrastive.strict_derangement is True
    assert configs["openfwi_lmdb_contrastive.yaml"].contrastive.temperature == 0.08
    assert configs["openfwi_lmdb_contrastive.yaml"].contrastive.pairwise_temperature == 0.1
    assert configs["openfwi_lmdb_contrastive.yaml"].contrastive.symile_structural_weight == 0.25
    assert configs["openfwi_lmdb_contrastive.yaml"].contrastive.symile_numerical_weight == 0.25
    assert configs["openfwi_lmdb_contrastive.yaml"].contrastive.pairwise_structural_weight == 0.2
    assert configs["openfwi_lmdb_contrastive.yaml"].contrastive.pairwise_numerical_weight == 0.3
    assert configs["openfwi_lmdb_contrastive.yaml"].contrastive.pairwise_reliability_floor == 0.35
    assert configs["openfwi_lmdb_contrastive.yaml"].contrastive.anchor_weight == 0.10
    assert configs["openfwi_lmdb_contrastive.yaml"].contrastive.reliability_prior_weight == 0.05
    assert configs["openfwi_lmdb_contrastive.yaml"].contrastive.sn_ortho_weight == 0.02
    assert configs["openfwi_lmdb_contrastive.yaml"].contrastive.unique_ortho_weight == 0.02
    assert configs["openfwi_lmdb_contrastive.yaml"].contrastive.structural_highpass_weight == 0.005
    assert configs["openfwi_lmdb_contrastive.yaml"].contrastive.numerical_lowpass_weight == 0.005
    assert configs["openfwi_lmdb_contrastive.yaml"].contrastive.frequency_kernel_size == 5
    assert configs["openfwi_lmdb_contrastive.yaml"].contrastive.num_negative_shuffles == 96
    assert configs["openfwi_lmdb_contrastive.yaml"].contrastive.modality_dropout.enabled is False
    assert configs["openfwi_lmdb_contrastive.yaml"].training.logging.log_version == (
        "contrastive_pairwise_recover_{time}"
    )
    assert configs["openfwi_lmdb_contrastive.yaml"].training.checkpoint.filename == (
        "pairwise-epoch{epoch}-top5{val/pairwise_retrieval_top5:.4f}"
    )
    assert configs["openfwi_lmdb_contrastive.yaml"].training.checkpoint.monitor == "val/pairwise_retrieval_top5"
    assert configs["openfwi_lmdb_contrastive.yaml"].training.checkpoint.mode == "max"
    assert configs["openfwi_lmdb_contrastive.yaml"].training.checkpoint.save_top_k == 5
    assert list(configs["openfwi_lmdb_contrastive.yaml"].data.required_modalities) == [
        "depth_vel",
        "migrated_image",
        "horizon",
        "rms_vel",
    ]
    assert configs["fast_run.yaml"].training.fast_run is True
    assert list(configs["fast_run.yaml"].training.fast_run_stages) == ["contrastive", "background", "residual"]
    assert configs["eval.yaml"].evaluation.allow_untrained is False
    assert configs["autoencoder_train.yaml"].training.fast_run is False
    assert configs["autoencoder_fast_run.yaml"].training.fast_run is True


def test_programmatic_contrastive_launcher_uses_yaml_without_overrides():
    conf = load_contrastive_config()
    yaml_conf = OmegaConf.load("bg_pdr_fm/configs/openfwi_lmdb_contrastive.yaml")
    assert OmegaConf.to_container(conf, resolve=True) == OmegaConf.to_container(yaml_conf, resolve=True)
    assert conf.training.stage == "contrastive"
    assert conf.training.fast_run == yaml_conf.training.fast_run
    assert conf.training.fast_run_stages is None
    assert conf.training.batch_size == yaml_conf.training.batch_size
    assert conf.training.max_epochs == yaml_conf.training.max_epochs
    assert list(conf.training.devices) == list(yaml_conf.training.devices)
    assert conf.training.strategy == yaml_conf.training.strategy
    assert conf.training.use_distributed_sampler == yaml_conf.training.use_distributed_sampler
    assert conf.training.precision == yaml_conf.training.precision
    assert conf.training.matmul_precision == yaml_conf.training.matmul_precision


def test_programmatic_stage_launchers_use_yaml_without_overrides():
    stage_loaders = {
        "openfwi_lmdb_contrastive.yaml": load_contrastive_config,
        "openfwi_lmdb_background.yaml": load_background_config,
        "openfwi_lmdb_residual.yaml": load_residual_config,
    }
    for filename, loader in stage_loaders.items():
        conf = loader()
        yaml_conf = OmegaConf.load(Path("bg_pdr_fm/configs") / filename)
        assert OmegaConf.to_container(conf, resolve=True) == OmegaConf.to_container(yaml_conf, resolve=True)
        assert conf.training.batch_size == yaml_conf.training.batch_size
        assert conf.training.max_epochs == yaml_conf.training.max_epochs
        assert conf.training.fast_run == yaml_conf.training.fast_run
        assert conf.training.precision == yaml_conf.training.precision
        assert conf.training.matmul_precision == yaml_conf.training.matmul_precision


def test_bg_pdr_fm_log_version_includes_timestamp():
    conf = _conf("contrastive")
    conf.training.logging = {"log_version": "unit_{stage}", "run_timestamp": "20260525_123456"}

    assert train_bg_pdr_fm._log_version(conf, "contrastive") == "unit_contrastive_20260525_123456"

    conf.training.logging.log_version = "unit_{stage}_{time}"
    assert train_bg_pdr_fm._log_version(conf, "background") == "unit_background_20260525_123456"

    conf.training.logging.log_version = ""
    assert train_bg_pdr_fm._log_version(conf, "residual") == "residual_20260525_123456"

    conf.training.logging.add_timestamp = False
    assert train_bg_pdr_fm._log_version(conf, "residual") == "residual"


def test_run_one_stage_builds_lightning_logging_checkpoint_and_precision():
    conf = _conf("contrastive")
    conf.training.precision = "bf16-mixed"
    conf.training.matmul_precision = "medium"
    conf.training.max_epochs = 1
    conf.training.limit_train_batches = 1
    conf.training.limit_val_batches = 1
    conf.training.log_every_n_steps = 3
    conf.training.strategy = "ddp"
    conf.training.num_nodes = 1
    conf.training.sync_batchnorm = False
    conf.training.use_distributed_sampler = True
    conf.training.logging = {"log_dir": "logs/test_lightning", "log_version": "unit", "run_timestamp": "20260525_123456"}
    conf.training.checkpoint = {
        "dirpath": "logs/test_lightning/checkpoints",
        "filename": "{stage}-unit-{epoch}",
        "monitor": "val/loss",
        "mode": "min",
        "save_top_k": 1,
        "save_last": True,
        "every_n_epochs": 1,
    }
    captured = {}

    class FakeTrainer:
        def __init__(self, **kwargs):
            captured["trainer_kwargs"] = kwargs

        def fit(self, model, train_loader, val_loader):
            captured["fit_called"] = True

    class FakeModel:
        def __init__(self, model_conf):
            captured["model_conf"] = model_conf

    original_build_loaders = train_bg_pdr_fm.build_loaders
    original_model = train_bg_pdr_fm.BGPDRFMLightning
    original_trainer = train_bg_pdr_fm.lightning.Trainer
    original_set_matmul = train_bg_pdr_fm.torch.set_float32_matmul_precision
    original_csv_logger = train_bg_pdr_fm.CSVLogger
    original_tb_logger = train_bg_pdr_fm.TensorBoardLogger
    original_checkpoint = train_bg_pdr_fm.ModelCheckpoint

    class FakeLogger:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class FakeCheckpoint:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            captured["checkpoint_kwargs"] = kwargs

    try:
        train_bg_pdr_fm.build_loaders = lambda stage_conf: ("train", "val")
        train_bg_pdr_fm.BGPDRFMLightning = FakeModel
        train_bg_pdr_fm.lightning.Trainer = FakeTrainer
        train_bg_pdr_fm.CSVLogger = FakeLogger
        train_bg_pdr_fm.TensorBoardLogger = FakeLogger
        train_bg_pdr_fm.ModelCheckpoint = FakeCheckpoint
        train_bg_pdr_fm.torch.set_float32_matmul_precision = lambda value: captured.setdefault(
            "matmul_precision",
            value,
        )
        run_one_stage(conf)
    finally:
        train_bg_pdr_fm.build_loaders = original_build_loaders
        train_bg_pdr_fm.BGPDRFMLightning = original_model
        train_bg_pdr_fm.lightning.Trainer = original_trainer
        train_bg_pdr_fm.torch.set_float32_matmul_precision = original_set_matmul
        train_bg_pdr_fm.CSVLogger = original_csv_logger
        train_bg_pdr_fm.TensorBoardLogger = original_tb_logger
        train_bg_pdr_fm.ModelCheckpoint = original_checkpoint

    trainer_kwargs = captured["trainer_kwargs"]
    assert trainer_kwargs["precision"] == "bf16-mixed"
    assert captured["matmul_precision"] == "medium"
    assert trainer_kwargs["enable_checkpointing"] is True
    assert len(trainer_kwargs["logger"]) == 2
    assert [logger.kwargs["version"] for logger in trainer_kwargs["logger"]] == [
        "unit_20260525_123456",
        "unit_20260525_123456",
    ]
    assert len(trainer_kwargs["callbacks"]) == 1
    assert trainer_kwargs["limit_train_batches"] == 1
    assert trainer_kwargs["limit_val_batches"] == 1
    assert trainer_kwargs["log_every_n_steps"] == 3
    assert trainer_kwargs["strategy"] == "ddp"
    assert trainer_kwargs["num_nodes"] == 1
    assert trainer_kwargs["sync_batchnorm"] is False
    assert trainer_kwargs["use_distributed_sampler"] is True
    assert captured["checkpoint_kwargs"]["filename"] == "contrastive-unit-{epoch}"
    assert captured["fit_called"] is True


def test_contrastive_fast_run_keeps_single_stage():
    conf = OmegaConf.load("bg_pdr_fm/configs/openfwi_lmdb_contrastive.yaml")
    conf = apply_fast_run_overrides(conf)
    assert conf.training.stage == "contrastive"
    assert conf.training.fast_run_stages is None
    assert conf.training.max_epochs == 1
    assert conf.training.limit_train_batches == 1
    assert conf.training.limit_val_batches == 1
    assert conf.training.num_workers == 0
    assert conf.training.persistent_workers is False
    assert conf.training.prefetch_factor is None


def test_build_loaders_passes_multiprocessing_context():
    conf = _conf("contrastive")
    conf.data = {"name": "synthetic"}
    conf.training.batch_size = 2
    conf.training.num_workers = 2
    conf.training.persistent_workers = True
    conf.training.prefetch_factor = 4
    conf.training.dataloader_multiprocessing_context = "spawn"
    captured = []

    class FakeDataset:
        def __init__(self, split):
            self.split = split

        def __len__(self):
            return 2

        def __getitem__(self, index):
            return SyntheticBGDataset(length=1)[0]

    class FakeLoader:
        def __init__(self, dataset, **kwargs):
            captured.append(kwargs)
            self.dataset = dataset

        def __iter__(self):
            raise AssertionError("build_loaders should not start DataLoader workers during schema validation")

    original_build_dataset = train_bg_pdr_fm.build_dataset
    original_loader = train_bg_pdr_fm.DataLoader
    try:
        train_bg_pdr_fm.build_dataset = lambda _conf, split: FakeDataset(split)
        train_bg_pdr_fm.DataLoader = FakeLoader
        train_bg_pdr_fm.build_loaders(conf)
    finally:
        train_bg_pdr_fm.build_dataset = original_build_dataset
        train_bg_pdr_fm.DataLoader = original_loader

    assert captured[0]["multiprocessing_context"] == "spawn"
    assert captured[1]["multiprocessing_context"] == "spawn"
    assert captured[0]["persistent_workers"] is True
    assert captured[0]["prefetch_factor"] == 4


def test_encoder_interface_hides_base():
    batch = _batch()
    encoder = PhysicsDecoupledEncoder(hidden_channels=8, base_channels=16, feature_channels=8)
    features = encoder(batch)
    assert features.numerical.shape == (2, 8, 70, 70)
    assert features.structural.shape == (2, 8, 70, 70)
    assert features.unique.shape == (2, 8, 70, 70)
    assert features.reliability.shape == (2, 4, 2)
    assert not hasattr(features, "base")


def test_wavelet_anchor_shapes():
    wavelet = DepthVelocityWaveletTransform(level=1, boundary="edge", resize_mode="nearest", detail_fusion="l2")
    anchors = wavelet(torch.randn(2, 1, 70, 70))
    assert anchors.anchor_struct.shape == (2, 1, 70, 70)
    assert anchors.anchor_num.shape == (2, 1, 70, 70)
    assert torch.isfinite(anchors.anchor_struct).all()
    assert torch.isfinite(anchors.anchor_num).all()


def test_encoder_can_return_per_modality_heads():
    batch = _batch()
    encoder = PhysicsDecoupledEncoder(hidden_channels=8, base_channels=16, feature_channels=8)
    features = encoder(batch, return_all_heads=True)
    assert features.all_heads is not None
    assert set(features.all_heads) == {"migrated_image", "horizon", "rms_vel", "well_log"}
    for heads in features.all_heads.values():
        assert set(heads) == {"numerical", "structural", "unique"}
        assert heads["numerical"].shape == (2, 8, 70, 70)
        assert heads["structural"].shape == (2, 8, 70, 70)
        assert heads["unique"].shape == (2, 8, 70, 70)


def test_symile_shuffle_derangement_has_no_self_index():
    loss_fn = PhysicsAnchoredSubsetSymileLoss(feature_channels=8, embed_dim=16)
    for batch_size in (2, 3, 5):
        torch.manual_seed(batch_size)
        cyclic = {
            tuple(loss_fn._strict_derangement_indices(batch_size, torch.device("cpu"), offset).tolist())
            for offset in range(1, batch_size)
        }
        sampled = set()
        for _ in range(64):
            indices = loss_fn._shuffled_derangement_indices(batch_size, torch.device("cpu"))
            assert torch.equal(torch.sort(indices).values, torch.arange(batch_size))
            assert not torch.any(indices == torch.arange(batch_size))
            sampled.add(tuple(indices.tolist()))
        if batch_size > 3:
            assert sampled - cyclic


def test_symile_negative_shuffle_count_can_be_limited():
    default_loss = PhysicsAnchoredSubsetSymileLoss(feature_channels=8, embed_dim=16)
    limited_loss = PhysicsAnchoredSubsetSymileLoss(feature_channels=8, embed_dim=16, num_negative_shuffles=4)

    assert default_loss._negative_shuffle_count(2) == 1
    assert default_loss._negative_shuffle_count(150) == 149
    assert limited_loss._negative_shuffle_count(2) == 1
    assert limited_loss._negative_shuffle_count(150) == 4


def test_subset_symile_handles_missing_modalities():
    batch = _batch()
    batch.modality_mask = torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    batch.modality_quality = batch.modality_mask.clone()
    encoder = PhysicsDecoupledEncoder(hidden_channels=8, base_channels=16, feature_channels=8)
    features = encoder(batch, return_all_heads=True)
    anchors = DepthVelocityWaveletTransform()(batch.depth_vel)
    loss_fn = PhysicsAnchoredSubsetSymileLoss(feature_channels=8, embed_dim=16, min_symile_group_size=2)
    torch.manual_seed(123)
    out = loss_fn(features, batch, anchor_struct=anchors.anchor_struct, anchor_num=anchors.anchor_num)
    assert torch.isfinite(out["loss"])
    assert torch.isfinite(out["anchor"])
    assert torch.isfinite(out["reliability_prior"])
    assert out["observed_subset_count"] == 2
    assert torch.isfinite(out["observed_modality_mean"])


def test_frequency_decoupling_regularizer_is_reported_and_weighted():
    if not torch.cuda.is_available():
        pytest.skip("Frequency decoupling smoke test is GPU-only.")
    batch = batch_to_device(_batch(), "cuda:0")
    encoder = PhysicsDecoupledEncoder(hidden_channels=8, base_channels=16, feature_channels=8).to("cuda:0")
    features = encoder(batch, return_all_heads=True)
    anchors = DepthVelocityWaveletTransform()(batch.depth_vel)
    loss_fn = PhysicsAnchoredSubsetSymileLoss(
        feature_channels=8,
        embed_dim=16,
        symile_structural_weight=0.0,
        symile_numerical_weight=0.0,
        anchor_weight=0.0,
        reliability_prior_weight=0.0,
        sn_ortho_weight=0.0,
        unique_ortho_weight=0.0,
        structural_highpass_weight=0.2,
        numerical_lowpass_weight=0.3,
    ).to("cuda:0")
    out = loss_fn(features, batch, anchor_struct=anchors.anchor_struct, anchor_num=anchors.anchor_num)
    expected = (
        0.2 * out["frequency_structural_low_penalty"]
        + 0.3 * out["frequency_numerical_high_penalty"]
    )
    assert torch.allclose(out["loss"], expected)
    for key in (
        "frequency_structural_low_penalty",
        "frequency_numerical_high_penalty",
        "frequency_structural_high_ratio",
        "frequency_numerical_low_ratio",
    ):
        assert torch.isfinite(out[key])
        assert 0.0 <= float(out[key].detach()) <= 1.0


def test_pairwise_contrastive_loss_is_reported_and_weighted():
    if not torch.cuda.is_available():
        pytest.skip("Pairwise contrastive smoke test is GPU-only.")
    batch = batch_to_device(_batch(batch_size=4), "cuda:0")
    encoder = PhysicsDecoupledEncoder(hidden_channels=8, base_channels=16, feature_channels=8).to("cuda:0")
    features = encoder(batch, return_all_heads=True)
    anchors = DepthVelocityWaveletTransform()(batch.depth_vel)
    loss_fn = PhysicsAnchoredSubsetSymileLoss(
        feature_channels=8,
        embed_dim=16,
        symile_structural_weight=0.0,
        symile_numerical_weight=0.0,
        pairwise_structural_weight=0.2,
        pairwise_numerical_weight=0.3,
        pairwise_temperature=0.1,
        pairwise_reliability_floor=0.35,
        anchor_weight=0.0,
        reliability_prior_weight=0.0,
        sn_ortho_weight=0.0,
        unique_ortho_weight=0.0,
        structural_highpass_weight=0.0,
        numerical_lowpass_weight=0.0,
    ).to("cuda:0")
    out = loss_fn(features, batch, anchor_struct=anchors.anchor_struct, anchor_num=anchors.anchor_num)
    expected = 0.2 * out["pairwise_structural"] + 0.3 * out["pairwise_numerical"]
    assert torch.allclose(out["loss"], expected)
    assert out["pairwise_structural_pairs"] > 0
    assert out["pairwise_numerical_pairs"] > 0
    for key in (
        "pairwise_structural",
        "pairwise_numerical",
        "pairwise_retrieval_top1",
        "pairwise_retrieval_top5",
        "pairwise_retrieval_top1_structural",
        "pairwise_retrieval_top5_structural",
        "pairwise_retrieval_top1_numerical",
        "pairwise_retrieval_top5_numerical",
        "pairwise_retrieval_top5_numerical_well_log_rms_vel",
        "pairwise_alignment_numerical_well_log_rms_vel",
    ):
        assert torch.isfinite(out[key])
    for key in (
        "pairwise_retrieval_top1",
        "pairwise_retrieval_top5",
        "pairwise_retrieval_top1_structural",
        "pairwise_retrieval_top5_structural",
        "pairwise_retrieval_top1_numerical",
        "pairwise_retrieval_top5_numerical",
        "pairwise_retrieval_top5_numerical_well_log_rms_vel",
    ):
        assert 0.0 <= float(out[key].detach()) <= 1.0
    alignment = float(out["pairwise_alignment_numerical_well_log_rms_vel"].detach())
    assert -1.0 <= alignment <= 1.0


def test_subset_symile_ignores_missing_modality_heads():
    batch = _batch()
    batch.modality_mask = torch.tensor(
        [
            [1.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    batch.modality_quality = batch.modality_mask.clone()
    encoder = PhysicsDecoupledEncoder(hidden_channels=8, base_channels=16, feature_channels=8)
    features = encoder(batch, return_all_heads=True)
    assert features.all_heads is not None
    anchors = DepthVelocityWaveletTransform()(batch.depth_vel)
    loss_fn = PhysicsAnchoredSubsetSymileLoss(feature_channels=8, embed_dim=16, min_symile_group_size=2)
    torch.manual_seed(123)
    out = loss_fn(features, batch, anchor_struct=anchors.anchor_struct, anchor_num=anchors.anchor_num)

    perturbed_heads = {
        modality: {name: tensor.clone() for name, tensor in heads.items()}
        for modality, heads in features.all_heads.items()
    }
    torch.manual_seed(456)
    for name in perturbed_heads["well_log"]:
        perturbed_heads["well_log"][name] = torch.randn_like(perturbed_heads["well_log"][name]) * 1000.0
    perturbed = DecoupledFeatures(
        numerical=features.numerical,
        structural=features.structural,
        unique=features.unique,
        reliability=features.reliability,
        all_heads=perturbed_heads,
    )
    torch.manual_seed(123)
    perturbed_out = loss_fn(perturbed, batch, anchor_struct=anchors.anchor_struct, anchor_num=anchors.anchor_num)

    for key in (
        "loss",
        "symile_structural",
        "symile_numerical",
        "anchor",
        "reliability_prior",
        "sn_ortho",
        "unique_ortho",
        "frequency_structural_low_penalty",
        "frequency_numerical_high_penalty",
        "frequency_structural_high_ratio",
        "frequency_numerical_low_ratio",
    ):
        assert torch.allclose(out[key], perturbed_out[key], atol=1e-6)


def test_subset_symile_single_observed_modality_falls_back_to_anchor():
    batch = _batch()
    batch.modality_mask = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    batch.modality_quality = batch.modality_mask.clone()
    encoder = PhysicsDecoupledEncoder(hidden_channels=8, base_channels=16, feature_channels=8)
    features = encoder(batch, return_all_heads=True)
    anchors = DepthVelocityWaveletTransform()(batch.depth_vel)
    loss_fn = PhysicsAnchoredSubsetSymileLoss(feature_channels=8, embed_dim=16, min_symile_group_size=2)
    out = loss_fn(features, batch, anchor_struct=anchors.anchor_struct, anchor_num=anchors.anchor_num)
    assert torch.isfinite(out["loss"])
    assert out["symile_structural_groups"] == 0
    assert out["symile_numerical_groups"] == 0
    assert torch.isfinite(out["anchor"])


def test_subset_symile_batch_size_one_falls_back_to_anchor():
    batch = _batch(batch_size=1)
    batch.modality_mask = torch.ones(1, 4)
    batch.modality_quality = torch.ones(1, 4)
    encoder = PhysicsDecoupledEncoder(hidden_channels=8, base_channels=16, feature_channels=8)
    features = encoder(batch, return_all_heads=True)
    anchors = DepthVelocityWaveletTransform()(batch.depth_vel)
    loss_fn = PhysicsAnchoredSubsetSymileLoss(feature_channels=8, embed_dim=16, min_symile_group_size=2)
    out = loss_fn(features, batch, anchor_struct=anchors.anchor_struct, anchor_num=anchors.anchor_num)
    assert torch.isfinite(out["loss"])
    assert out["symile_structural_groups"] == 0
    assert out["symile_numerical_groups"] == 0
    assert torch.isfinite(out["anchor"])


def test_reliability_prior_only_uses_observed_modalities():
    loss_fn = PhysicsAnchoredSubsetSymileLoss(feature_channels=8, embed_dim=16)
    reliability = torch.tensor(
        [
            [[0.85, 0.45], [0.95, 0.10], [0.55, 0.90], [0.00, 0.00]],
            [[0.85, 0.45], [0.95, 0.10], [0.55, 0.90], [1.00, 1.00]],
        ],
        dtype=torch.float32,
    )
    availability = torch.tensor(
        [
            [1.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    assert torch.equal(loss_fn._reliability_prior_loss(reliability, availability), torch.zeros(()))


def test_condition_adapters():
    batch = _batch()
    encoder = PhysicsDecoupledEncoder(hidden_channels=8, base_channels=16, feature_channels=8)
    adapters = ConditionAdapters(feature_channels=8, cond_channels=16, out_hw=(18, 18), use_unique=True)
    conditions = adapters(encoder(batch))
    assert conditions.numerical.shape == (2, 16, 18, 18)
    assert conditions.structural_modulated.shape == (2, 16, 18, 18)


def test_simple_latent_codec_shape_and_backward():
    codec = SimpleLatentCodec(latent_channels=4, latent_hw=(18, 18))
    x = torch.randn(2, 1, 70, 70, requires_grad=True)
    z_first = codec.encode(x, deterministic=True)
    z_second = codec.encode(x, deterministic=True)
    z = codec.encode(x)
    recon = codec.decode(z, out_hw=(70, 70))
    loss = recon.abs().mean()
    loss.backward()
    assert torch.allclose(z_first, z_second)
    assert z.shape == (2, 4, 18, 18)
    assert recon.shape == (2, 1, 70, 70)
    assert x.grad is not None


def test_identity_latent_codec_passthrough_for_pixel_space_fm():
    codec = IdentityLatentCodec(image_channels=1, latent_channels=1, latent_hw=(70, 70))
    x = torch.randn(2, 1, 70, 70)
    z = codec.encode(x, deterministic=True)
    recon = codec.decode(z, out_hw=(70, 70))

    assert z.shape == x.shape
    assert recon.shape == x.shape
    assert torch.allclose(z, x)
    assert torch.allclose(recon, x)
    assert sum(parameter.numel() for parameter in codec.parameters()) == 0

    with pytest.raises(ValueError, match="IdentityLatentCodec"):
        codec.encode(torch.randn(2, 4, 70, 70))
    with pytest.raises(ValueError, match="out_hw"):
        codec.decode(x, out_hw=(18, 18))


def test_bg_pdr_fm_autoencoder_lightning_shape_and_backward():
    conf = OmegaConf.load("bg_pdr_fm/configs/autoencoder_fast_run.yaml")
    model = SeismicAutoencoderKLLightning(conf)
    batch = {"depth_vel": torch.randn(2, 1, 70, 70)}
    loss = model.training_step(batch, 0)
    loss.backward()
    posterior = model.vae.encode(batch["depth_vel"])
    latents = posterior.sample()
    recon = model.vae.decode(latents)
    assert torch.isfinite(loss)
    assert latents.shape == (2, 4, 18, 18)
    assert recon.shape == (2, 1, 70, 70)


def test_autoencoder_codec_missing_checkpoint_error():
    with tempfile.TemporaryDirectory() as tmp:
        missing = Path(tmp) / "missing.ckpt"
        try:
            AutoencoderLatentCodec(checkpoint_path=missing)
        except FileNotFoundError as exc:
            msg = str(exc)
            assert "checkpoint not found" in msg
            assert str(missing) in msg
        else:
            raise AssertionError("AutoencoderLatentCodec should fail clearly when checkpoint is missing.")


def test_autoencoder_codec_fake_loader_shape_and_freeze():
    class FakePosterior:
        def __init__(self, tensor):
            self.tensor = tensor

        def mode(self):
            return self.tensor

        def sample(self):
            return self.tensor + torch.randn_like(self.tensor) * 0.01

    class FakeVAE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Conv2d(1, 4, kernel_size=1)

        def encode(self, image):
            z = torch.nn.functional.interpolate(image, size=(16, 16), mode="bilinear", align_corners=False)
            return FakePosterior(self.proj(z))

        def decode(self, latent):
            x = latent[:, :1]
            return torch.nn.functional.interpolate(x, size=(64, 64), mode="bilinear", align_corners=False)

    class FakeLightning:
        @classmethod
        def load_from_checkpoint(cls, checkpoint_path, map_location="cpu"):
            module = cls()
            module.vae = FakeVAE()
            return module

    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = Path(tmp) / "fake.ckpt"
        checkpoint.write_bytes(b"fake")
        codec = AutoencoderLatentCodec(checkpoint_path=checkpoint, loader_cls=FakeLightning)
        x = torch.randn(2, 1, 70, 70)
        z = codec.encode(x, deterministic=True)
        z_again = codec.encode(x, deterministic=True)
        z_sample = codec.encode(x, deterministic=False)
        recon = codec.decode(z, out_hw=(70, 70))
        assert torch.allclose(z, z_again)
        assert not torch.allclose(z, z_sample)
        assert z.shape == (2, 4, 16, 16)
        assert recon.shape == (2, 1, 70, 70)
        assert not any(parameter.requires_grad for parameter in codec.parameters())


def test_autoencoder_codec_new_checkpoint_if_available():
    checkpoint = AUTOENCODER_CHECKPOINT
    if not checkpoint.is_file():
        return
    codec = AutoencoderLatentCodec(checkpoint_path=checkpoint)
    x = torch.randn(1, 1, 70, 70)
    z = codec.encode(x, deterministic=True)
    z_again = codec.encode(x, deterministic=True)
    recon = codec.decode(z, out_hw=(70, 70))
    gap_recon = codec.decode(z_again, out_hw=(70, 70))
    direct_l1 = torch.nn.functional.l1_loss(recon, x)
    gap_l1 = torch.nn.functional.l1_loss(gap_recon, x)
    gap_ratio = gap_l1 / direct_l1.clamp_min(1e-8)
    assert torch.allclose(z, z_again)
    assert z.ndim == 4
    assert recon.shape == (1, 1, 70, 70)
    assert torch.isfinite(gap_ratio)
    assert not any(parameter.requires_grad for parameter in codec.parameters())


def test_background_stage_backward():
    batch = _batch()
    model = BGPDRFMLightning(_conf("background"))
    loss = model.training_step(batch, 0)
    loss.backward()
    assert torch.isfinite(loss)


def test_background_estimator_direct_output_shape_unchanged():
    estimator = BackgroundEstimator(cond_channels=16, hidden_channels=8, output_mode="direct")
    estimate = estimator(torch.randn(2, 16, 18, 18), out_hw=(70, 70))
    assert estimate.bg_hat.shape == (2, 1, 70, 70)
    assert estimate.bg_ll is None
    assert torch.isfinite(estimate.bg_hat).all()
    assert torch.isfinite(estimate.epsilon_h_b).all()


def test_background_unet_direct_uses_encoder_resolution_condition():
    estimator = BackgroundEstimator(
        cond_channels=8,
        hidden_channels=8,
        output_mode="direct",
        backend="unet_direct",
    )
    estimate = estimator(torch.randn(2, 8, 70, 70), out_hw=(70, 70))
    assert estimate.bg_hat.shape == (2, 1, 70, 70)
    assert estimate.bg_ll is None
    assert estimate.bg_lowres is None
    assert torch.isfinite(estimate.bg_hat).all()
    assert torch.isfinite(estimate.epsilon_h_b).all()


def test_background_flow_matching_backend_trains_and_samples_full_resolution_background():
    estimator = BackgroundEstimator(
        cond_channels=8,
        hidden_channels=8,
        output_mode="direct",
        backend="flow_matching",
        fm_backend="unet_fm_film",
        fm_latent_hw=(70, 70),
        fm_num_inference_steps=2,
    )
    cond = torch.randn(2, 8, 70, 70)
    target = torch.randn(2, 1, 70, 70).clamp(-1, 1)

    train_estimate = estimator(cond, out_hw=(70, 70), target=target)
    assert train_estimate.bg_hat.shape == target.shape
    assert train_estimate.fm_loss is not None
    assert torch.isfinite(train_estimate.fm_loss)
    train_estimate.fm_loss.backward()
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in estimator.parameters()
    )

    estimator.eval()
    with torch.no_grad():
        sample = estimator(cond, out_hw=(70, 70))
    assert sample.bg_hat.shape == target.shape
    assert sample.fm_loss is None
    assert torch.isfinite(sample.bg_hat).all()


def test_background_stage_flow_matching_is_separate_from_residual_fm():
    batch = _batch()
    conf = _conf("background")
    conf.model.background_backend = "flow_matching"
    conf.model.background_condition_source = "encoder_numerical"
    conf.model.background_output_mode = "direct"
    conf.model.background_fm_backend = "unet_fm_film"
    conf.model.background_fm_hidden_channels = 8
    conf.model.background_fm_num_inference_steps = 2
    conf.model.background_fm_latent_hw = [70, 70]
    conf.loss.bg_fm_weight = 1.0
    conf.loss.bg_l1_weight = 1.0
    conf.loss.bg_l2_weight = 0.5
    model = BGPDRFMLightning(conf)
    model.on_train_epoch_start()
    assert model.background is not None
    assert model.residual is not None
    assert model.background.network.field.in_proj.out_channels == 8
    assert id(model.background) != id(model.residual)
    assert not any(parameter.requires_grad for parameter in model.residual.parameters())

    loss = model.training_step(batch, 0)
    loss.backward()
    assert torch.isfinite(loss)
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in model.background.parameters()
    )
    assert _grads_are_none_or_zero(model.residual.parameters())


def test_joint_full_background_flow_matching_trains_all_active_model_parts():
    conf = _conf("joint_full")
    conf.model.background_backend = "flow_matching"
    conf.model.background_condition_source = "encoder_numerical"
    conf.model.background_output_mode = "direct"
    conf.model.background_fm_backend = "unet_fm_film"
    conf.model.background_fm_hidden_channels = 8
    conf.model.background_fm_num_inference_steps = 2
    conf.model.background_fm_latent_hw = [70, 70]
    conf.model.codec_type = "identity"
    conf.model.latent_channels = 1
    conf.model.latent_hw = [70, 70]
    conf.model.residual_backend = "unet_fm_film"
    conf.model.residual_backend_hidden_channels = 8
    conf.model.residual_condition_merge = "concat"
    conf.model.residual_background_context_source = "codec_latent"
    conf.loss.bg_fm_weight = 1.0
    conf.loss.bg_l1_weight = 1.0
    conf.loss.bg_l2_weight = 0.5
    conf.training.joint = {
        "lambda_contrastive": 0.01,
        "encoder_backbone_lr": 1e-6,
        "encoder_head_lr": 3e-6,
        "contrastive_lr": 3e-6,
        "adapter_lr": 3e-6,
        "background_lr": 1e-4,
        "residual_lr": 5e-5,
    }
    model = BGPDRFMLightning(conf)

    assert any(parameter.requires_grad for parameter in model.encoder.parameters())
    assert any(parameter.requires_grad for parameter in model.contrastive_loss.parameters())
    assert model.background is not None
    assert any(parameter.requires_grad for parameter in model.background.parameters())
    assert model.residual is not None
    assert any(parameter.requires_grad for parameter in model.residual.parameters())
    assert model.adapters is not None
    assert any(parameter.requires_grad for parameter in model.adapters.structural.parameters())

    optimizer = model.configure_optimizers()
    assert len(optimizer.param_groups) >= 5


def test_background_encoder_numerical_source_freezes_condition_adapters():
    batch = _batch()
    conf = _conf("background")
    conf.model.background_backend = "unet_direct"
    conf.model.background_condition_source = "encoder_numerical"
    conf.model.background_output_mode = "direct"
    conf.model.background_hidden_channels = 8
    conf.loss.bg_l1_weight = 1.0
    conf.loss.bg_l2_weight = 0.5
    model = BGPDRFMLightning(conf)
    model.on_train_epoch_start()
    assert model.adapters is not None
    assert not any(parameter.requires_grad for parameter in model.adapters.parameters())

    loss = model.training_step(batch, 0)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.background is not None
    assert any(parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
               for parameter in model.background.parameters())
    assert _grads_are_none_or_zero(model.adapters.parameters())


def test_background_wavelet_ll_output_shape_and_loss_metrics_are_finite():
    batch = _batch()
    conf = _conf("background")
    conf.model.background_output_mode = "wavelet_ll"
    conf.model.background_hidden_channels = 16
    conf.loss.bg_multiscale_weight = 0.5
    conf.loss.bg_multiscale_high_weight = 0.1
    conf.loss.bg_scales = [5, 9, 17]
    model = BGPDRFMLightning(conf)
    _, _, bg, gate = model._common_forward(batch)
    assert bg.bg_hat.shape == (2, 1, 70, 70)
    assert bg.bg_ll is not None
    assert bg.bg_ll.shape == (2, 1, 35, 35)
    assert haar_detail_leak(bg.bg_hat) < 1e-6
    stage_output = background_stage_loss(model, batch, bg, gate)
    assert stage_output.metrics is not None
    assert torch.isfinite(stage_output.loss)
    assert "bg_wavelet_detail_leak" in stage_output.metrics
    assert stage_output.metrics["bg_wavelet_detail_leak"] < 1e-6


def test_background_multiscale_loss_metrics_are_finite():
    batch = _batch()
    conf = _conf("background")
    conf.loss.bg_multiscale_weight = 0.5
    conf.loss.bg_multiscale_high_weight = 0.1
    conf.loss.bg_scales = [5, 9, 17]
    model = BGPDRFMLightning(conf)
    _, _, bg, gate = model._common_forward(batch)
    assert bg.bg_hat.shape == (2, 1, 70, 70)
    stage_output = background_stage_loss(model, batch, bg, gate)
    assert stage_output.metrics is not None
    assert torch.isfinite(stage_output.loss)
    for key in ("bg_l1", "bg_high_leak", "bg_ms_l1", "bg_ms_high_leak"):
        assert key in stage_output.metrics
        assert torch.isfinite(stage_output.metrics[key]).all()


def test_background_low_residual_loss_metric_is_finite():
    batch = _batch()
    conf = _conf("background")
    conf.loss.bg_low_residual_weight = 0.3
    conf.loss.bg_low_residual_kernel = 9
    model = BGPDRFMLightning(conf)
    _, _, bg, gate = model._common_forward(batch)
    stage_output = background_stage_loss(model, batch, bg, gate)
    assert stage_output.metrics is not None
    assert torch.isfinite(stage_output.loss)
    assert "bg_low_residual" in stage_output.metrics
    assert torch.isfinite(stage_output.metrics["bg_low_residual"]).all()


def test_background_direct_l1_l2_lowpass_loss_ignores_auxiliary_terms():
    batch = _batch()
    conf = _conf("background")
    conf.model.background_output_mode = "direct"
    conf.loss.bg_target = "lowpass"
    conf.loss.bg_l1_weight = 1.0
    conf.loss.bg_l2_weight = 0.5
    conf.loss.bg_high_weight = 10.0
    conf.loss.bg_multiscale_weight = 10.0
    conf.loss.bg_low_residual_weight = 10.0
    model = BGPDRFMLightning(conf)
    _, _, bg, gate = model._common_forward(batch)
    stage_output = background_stage_loss(model, batch, bg, gate)
    target = model.filter.lowpass(batch.depth_vel)
    expected = torch.nn.functional.l1_loss(bg.bg_hat, target) + 0.5 * torch.nn.functional.mse_loss(bg.bg_hat, target)

    assert stage_output.metrics is not None
    assert torch.allclose(stage_output.loss, expected)
    assert "bg_l1" in stage_output.metrics
    assert "bg_l2" in stage_output.metrics
    assert "bg_loss" in stage_output.metrics


def test_residual_stage_accepts_wavelet_ll_background_interface():
    batch = _batch()
    conf = _conf("residual")
    conf.model.background_output_mode = "wavelet_ll"
    model = BGPDRFMLightning(conf)
    _, conditions, bg, gate = model._common_forward(batch)
    assert conditions is not None and bg is not None and gate is not None
    assert bg.bg_hat.shape == (2, 1, 70, 70)
    stage_output = residual_stage_loss(model, batch, conditions, bg, gate)
    assert torch.isfinite(stage_output.loss)
    assert stage_output.recon is not None
    assert stage_output.recon.shape == batch.depth_vel.shape


def test_background_night_queue_branch_selection():
    assert choose_next_branch({"val/rho_true": 1.9, "val/bg_high_leak": 0.015}, []) == "to100"
    assert choose_next_branch({"val/rho_true": 2.0, "val/bg_high_leak": 0.015}, []) == "lr03"
    leak_rows = [
        {"val/bg_high_leak": "0.0150"},
        {"val/bg_high_leak": "0.0153"},
        {"val/bg_high_leak": "0.0157"},
        {"val/bg_high_leak": "0.0161"},
        {"val/bg_high_leak": "0.0164"},
    ]
    assert choose_next_branch({"val/rho_true": 1.8, "val/bg_high_leak": 0.0164}, leak_rows) == "highguard"


def test_background_stage_autoencoder_codec_if_available():
    checkpoint = AUTOENCODER_CHECKPOINT
    if not checkpoint.is_file():
        return
    batch = _batch(batch_size=1)
    conf = _conf("background")
    conf.model.codec_type = "autoencoder"
    conf.model.codec_checkpoint = str(checkpoint)
    conf.model.latent_channels = 4
    conf.model.latent_hw = [18, 18]
    model = BGPDRFMLightning(conf)
    loss = model.training_step(batch, 0)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.codec_type == "autoencoder"
    assert not any(parameter.requires_grad for parameter in model.codec.parameters())


def test_contrastive_stage_backward():
    batch = _batch()
    model = BGPDRFMLightning(_conf("contrastive"))
    loss = model.training_step(batch, 0)
    loss.backward()
    assert torch.isfinite(loss)
    assert model._last_contrastive_metrics["symile_structural_groups"] > 0
    assert model._last_contrastive_metrics["symile_numerical_groups"] > 0
    assert any(parameter.grad is not None for parameter in model.encoder.parameters() if parameter.requires_grad)
    assert any(
        parameter.grad is not None
        for parameter in model.contrastive_loss.parameters()
        if parameter.requires_grad
    )


def test_contrastive_modality_dropout_only_changes_training_masks():
    batch = _batch()
    conf = _conf("contrastive")
    conf.contrastive.modality_dropout = {
        "enabled": True,
        "probs": {
            "migrated_image": 1.0,
            "horizon": 1.0,
            "rms_vel": 1.0,
            "well_log": 1.0,
        },
    }
    model = BGPDRFMLightning(conf)
    model.train()
    dropped = model._maybe_apply_modality_dropout(batch, "train")
    assert dropped is not batch
    assert dropped.modality_mask.sum(dim=1).min() >= 1
    assert torch.equal(dropped.depth_vel, batch.depth_vel)
    assert torch.equal(dropped.migrated_image, batch.migrated_image)
    model.eval()
    unchanged = model._maybe_apply_modality_dropout(batch, "val")
    assert unchanged is batch


def test_predict_batch_does_not_use_wavelet_anchor():
    batch = _batch()
    model = BGPDRFMLightning(_conf("residual"))

    def fail_if_called(*args, **kwargs):
        raise AssertionError("wavelet anchor must not be used during inference.")

    model.wavelet_anchor.forward = fail_if_called
    prediction = model.predict_batch(batch)
    assert prediction.bg_hat.shape == batch.depth_vel.shape


def test_residual_stage_backward():
    batch = _batch(batch_size=1)
    model = BGPDRFMLightning(_conf("residual"))
    model.on_train_epoch_start()
    features, conditions, bg, gate = model._common_forward(batch)
    assert bg.bg_hat.shape == (1, 1, 70, 70)
    stage_output = residual_stage_loss(model, batch, conditions, bg, gate)
    assert stage_output.rho_true is not None
    assert stage_output.metrics is not None
    assert torch.isfinite(stage_output.rho_true).all()
    assert torch.all(stage_output.rho_true >= 0)
    for key in (
        "E_V",
        "E_R",
        "E_R_over_E_V",
        "target_lf_loss",
        "vae_direct_l1",
        "vae_bg_l1",
        "vae_composed_l1",
        "vae_identity_l1",
        "vae_identity_ratio",
        "vae_gap_l1",
        "vae_gap_ratio",
    ):
        assert key in stage_output.metrics
        assert torch.isfinite(stage_output.metrics[key]).all()
    assert stage_output.metrics["vae_identity_l1"].item() < 1e-6
    loss = stage_output.loss
    loss.backward()
    assert torch.isfinite(loss)
    assert model.background is not None
    assert model.codec is not None
    assert model.adapters is not None
    assert model.rho_calibrator is not None
    assert _grads_are_none_or_zero(model.background.parameters())
    assert _grads_are_none_or_zero(model.codec.parameters())
    assert _grads_are_none_or_zero(model.adapters.numerical.parameters())
    assert _grads_are_none_or_zero(model.rho_calibrator.parameters())
    assert _grads_are_none_or_zero(model.num_to_anchor.parameters())
    assert _grads_are_none_or_zero(model.struct_to_anchor.parameters())


def test_residual_stage_supports_identity_pixel_space_fm():
    batch = _batch(batch_size=1)
    conf = _conf("residual")
    conf.model.codec_type = "identity"
    conf.model.latent_channels = 1
    conf.model.latent_hw = [70, 70]
    conf.model.cond_channels = 8
    conf.model.residual_backend = "unet_fm_film"
    conf.model.residual_backend_hidden_channels = 8
    conf.model.residual_condition_merge = "concat"
    conf.model.residual_num_inference_steps = 1
    model = BGPDRFMLightning(conf)
    model.on_train_epoch_start()

    features, conditions, bg, gate = model._common_forward(batch)
    assert conditions is not None and bg is not None and gate is not None
    assert conditions.structural.shape[-2:] == (70, 70)
    stage_output = residual_stage_loss(model, batch, conditions, bg, gate)

    assert stage_output.recon is not None
    assert stage_output.recon.shape == batch.depth_vel.shape
    assert stage_output.metrics is not None
    assert torch.isfinite(stage_output.loss)
    stage_output.loss.backward()
    assert model.codec is not None
    assert sum(parameter.numel() for parameter in model.codec.parameters()) == 0

    model.eval()
    prediction = model.predict_batch(batch)
    assert prediction.velocity_hat.shape == batch.depth_vel.shape
    assert prediction.residual_hat.shape == batch.depth_vel.shape


def test_predict_batch_outputs_shapes():
    batch = _batch()
    model = BGPDRFMLightning(_conf("residual"))
    prediction = model.predict_batch(batch)
    assert prediction.bg_hat.shape == (2, 1, 70, 70)
    assert prediction.bg_hat.shape == batch.depth_vel.shape
    assert prediction.velocity_hat.shape == batch.depth_vel.shape
    assert prediction.residual_hat.shape == batch.depth_vel.shape
    assert prediction.rho_hat_b.shape == (batch.depth_vel.shape[0],)
    assert prediction.alpha_hat.shape == (batch.depth_vel.shape[0],)


def test_predict_batch_structural_ablation_zero_is_evaluation_only():
    batch = _batch()
    conf = _conf("residual")
    conf.evaluation = {"structural_ablation": "zero"}
    model = BGPDRFMLightning(conf)
    prediction = model.predict_batch(batch)
    assert prediction.velocity_hat.shape == batch.depth_vel.shape
    assert torch.isfinite(prediction.velocity_hat).all()

    loss = model.training_step(batch, 0)
    assert torch.isfinite(loss)


def test_predict_batch_residual_override_zero_is_evaluation_only():
    batch = _batch()
    conf = _conf("residual")
    conf.evaluation = {"residual_override": "zero"}
    model = BGPDRFMLightning(conf)
    prediction = model.predict_batch(batch)
    assert prediction.velocity_hat.shape == batch.depth_vel.shape
    assert torch.allclose(prediction.velocity_hat, prediction.bg_hat)
    assert torch.count_nonzero(prediction.residual_hat).item() == 0

    loss = model.training_step(batch, 0)
    assert torch.isfinite(loss)


def test_residual_stage_accepts_codec_latent_background_context():
    batch = _batch()
    conf = _conf("residual")
    conf.model.residual_background_context_source = "codec_latent"
    model = BGPDRFMLightning(conf)

    loss = model.training_step(batch, 0)
    prediction = model.predict_batch(batch)

    assert torch.isfinite(loss)
    assert prediction.velocity_hat.shape == batch.depth_vel.shape
    assert torch.isfinite(prediction.velocity_hat).all()


def test_simple_residual_backend_sample_and_backward():
    generator = ResidualFlowGenerator(cond_channels=16, latent_channels=4, backend="simple_fm",
                                      num_inference_steps=2)
    assert generator.bg_cond_channels <= 8
    assert generator.bg_embed[0].out_channels <= 8
    z_res = torch.randn(2, 4, 18, 18)
    cond_s = torch.randn(2, 16, 18, 18)
    bg_hat = torch.randn(2, 1, 70, 70)
    out = generator.training_loss(z_res, cond_s, bg_hat, alpha_hat=torch.ones(2))
    out["loss"].backward()
    sample = generator.sample(cond_s, bg_hat, x_size=z_res.shape)
    assert torch.isfinite(out["loss"])
    assert out["velocity_pred"].shape == z_res.shape
    assert out["z_res_pred"].shape == z_res.shape
    assert sample.shape == z_res.shape


def test_residual_generator_can_condition_on_background_codec_latent():
    generator = ResidualFlowGenerator(
        cond_channels=16,
        latent_channels=4,
        backend="simple_fm",
        num_inference_steps=2,
        background_context_source="codec_latent",
    )
    z_res = torch.randn(2, 4, 18, 18)
    cond_s = torch.randn(2, 16, 18, 18)
    bg_hat = torch.randn(2, 1, 70, 70)
    z_bg = torch.randn_like(z_res)

    with pytest.raises(ValueError, match="z_bg"):
        generator.training_loss(z_res, cond_s, bg_hat)

    out = generator.training_loss(z_res, cond_s, bg_hat, z_bg=z_bg)
    sample = generator.sample(cond_s, bg_hat, x_size=z_res.shape, z_bg=z_bg)

    assert torch.isfinite(out["loss"])
    assert out["z_res_pred"].shape == z_res.shape
    assert sample.shape == z_res.shape


def test_legacy_residual_backend_requires_injected_flow():
    try:
        LegacyLatentFlowMatchingBackend()
    except ImportError as exc:
        assert "legacy_fm is not bundled" in str(exc)
        return
    raise AssertionError("legacy_fm should require an injected flow_cls in standalone bg_pdr_fm.")


def test_legacy_residual_backend_rejects_unsupported_shape():
    try:
        LegacyLatentFlowMatchingBackend(latent_channels=4, cond_channels=16, latent_hw=(16, 16))
    except ValueError as exc:
        assert "legacy_fm residual backend currently requires" in str(exc)
    else:
        raise AssertionError("legacy_fm should reject non-16/64/16x16 shapes.")


def test_legacy_residual_backend_fake_flow_shape_and_alpha_weight():
    class FakeVelocity(torch.nn.Module):
        def forward(self, z_t, t, cond):
            return 0.1 * z_t + cond[:, :16]

    class FakeFlow:
        def __init__(self, path="linear", num_train_timesteps=1000):
            self.path = path
            self.num_train_timesteps = num_train_timesteps
            self.velocity_model = FakeVelocity()

        def _time_to_unet_timestep(self, t):
            return torch.clamp((t * (self.num_train_timesteps - 1)).round().long(), min=0)

        def sample(self, cond, x_size, num_inference_steps=20):
            return torch.zeros(x_size, device=cond.device, dtype=cond.dtype) + cond[:, :16]

    backend = LegacyLatentFlowMatchingBackend(flow_cls=FakeFlow)
    z_res = torch.randn(2, 16, 16, 16)
    cond = torch.randn(2, 64, 16, 16)
    torch.manual_seed(123)
    plain = backend.training_loss(z_res, cond, alpha_hat=torch.zeros(2))
    torch.manual_seed(123)
    weighted = backend.training_loss(z_res, cond, alpha_hat=torch.ones(2))
    sample = backend.sample(cond, x_size=z_res.shape, steps=2)
    assert plain["z_res_pred"].shape == z_res.shape
    assert weighted["loss"] > plain["loss"]
    assert sample.shape == z_res.shape


def test_residual_stage_legacy_fm_requires_injected_flow():
    batch = _batch(batch_size=1)
    del batch
    conf = _conf("residual")
    conf.model.cond_channels = 64
    conf.model.latent_channels = 16
    conf.model.latent_hw = [16, 16]
    conf.model.residual_backend = "legacy_fm"
    try:
        BGPDRFMLightning(conf)
    except ImportError as exc:
        assert "legacy_fm is not bundled" in str(exc)
        return
    raise AssertionError("BGPDRFMLightning should reject legacy_fm without an injected flow_cls.")


def test_rho_calibrator_outputs_gate():
    batch = _batch()
    model = BGPDRFMLightning(_conf("background"))
    features, _, bg, gate = model._common_forward(batch)
    assert gate.rho_hat_b.shape == (2,)
    assert gate.alpha_hat.min() >= 0
    assert gate.alpha_hat.max() <= 1


def test_stage_trainability_contracts():
    contrastive = BGPDRFMLightning(_conf("contrastive"))
    assert contrastive.adapters is None
    assert contrastive.background is None
    assert contrastive.residual is None
    assert contrastive.codec is None
    assert contrastive.rho_calibrator is None
    assert any(p.requires_grad for p in contrastive.encoder.parameters())

    conf = _conf("background")
    conf.training.freeze_encoder = True
    background = BGPDRFMLightning(conf)
    assert not any(p.requires_grad for p in background.encoder.parameters())
    assert any(p.requires_grad for p in background.background.parameters())

    residual = BGPDRFMLightning(_conf("residual"))
    assert not any(p.requires_grad for p in residual.background.parameters())
    assert not any(p.requires_grad for p in residual.rho_calibrator.parameters())
    assert not any(p.requires_grad for p in residual.num_to_anchor.parameters())
    assert not any(p.requires_grad for p in residual.struct_to_anchor.parameters())
    assert not any(p.requires_grad for p in residual.adapters.numerical.parameters())
    assert any(p.requires_grad for p in residual.residual.parameters())


def test_background_estimator_hidden_channel_bottleneck():
    model = BGPDRFMLightning(_conf("background"))
    assert model.background.network[0].out_channels == 16
    cond_num = torch.randn(2, 16, 18, 18)
    bg = model.background(cond_num, out_hw=(70, 70))
    assert bg.bg_hat.shape == (2, 1, 70, 70)


def test_residual_condition_uses_low_bandwidth_background_context():
    batch = _batch()
    model = BGPDRFMLightning(_conf("residual"))
    _, conditions, bg, _ = model._common_forward(batch)
    cond_s = conditions.structural
    merged = model.residual.merged_condition(cond_s, bg.bg_hat)
    context = model.residual.background_context(bg.bg_hat, out_hw=cond_s.shape[-2:])
    assert merged.shape == cond_s.shape
    assert context.shape[-2:] == cond_s.shape[-2:]
    assert bg.bg_hat.shape[-2:] == batch.depth_vel.shape[-2:]


def test_marmousi_path_probe_error_mentions_dataset_meta():
    with tempfile.TemporaryDirectory() as tmp:
        try:
            resolve_marmousi_root(root_dir=tmp, root_candidates=[], include_defaults=False)
        except FileNotFoundError as exc:
            msg = str(exc)
            assert "dataset_meta.json" in msg
            assert str(Path(tmp).resolve()) in msg
        else:
            raise AssertionError("resolve_marmousi_root should fail when dataset_meta.json is absent.")


def test_openfwi_path_probe_error_mentions_missing_folders():
    with tempfile.TemporaryDirectory() as tmp:
        try:
            resolve_openfwi_root(
                root_dir=tmp,
                root_candidates=[],
                datasets=["FlatVelA"],
                required_modalities=["depth_vel"],
                include_defaults=False,
            )
        except FileNotFoundError as exc:
            msg = str(exc)
            assert "OpenFWI" in msg
            assert "FlatVelA" in msg
            assert "depth_vel" in msg
            assert str(Path(tmp).resolve()) in msg
        else:
            raise AssertionError("resolve_openfwi_root should fail when required folders are absent.")


def test_openfwi_missing_modality_uses_mask_zero():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        depth_dir = root / "FlatVelA" / "depth_vel"
        horizon_dir = root / "FlatVelA" / "horizon"
        depth_dir.mkdir(parents=True)
        horizon_dir.mkdir(parents=True)
        depth = np.full((1, 70, 70), 3000.0, dtype=np.float32)
        horizon = np.zeros((1, 70, 70), dtype=np.float32)
        np.save(depth_dir / "0.npy", depth)
        np.save(horizon_dir / "0.npy", horizon)
        dataset = OpenFWIBGDataset(
            root_dir=str(root),
            datasets=["FlatVelA"],
            split="all",
            required_modalities=["depth_vel"],
            target_shape=[70, 70],
        )
        batch = next(iter(DataLoader(dataset, batch_size=1, collate_fn=collate_bg_samples)))
        assert batch.modality_mask[:, :3].tolist() == [[0.0, 1.0, 0.0]]
        assert batch.modality_mask[0, 3].item() in {0.0, 1.0}
        assert batch.modality_quality[:, :3].tolist() == [[0.0, 1.0, 0.0]]
        assert batch.modality_quality[0, 3].item() in {0.0, 1.0}
        assert torch.all(batch.migrated_image == 0)
        assert torch.all(batch.rms_vel == 0)
        assert batch.well_log.shape == batch.depth_vel.shape
        assert batch.well_mask.shape == batch.depth_vel.shape
        assert int(batch.well_mask.sum().item()) in {0, 70, 140, 210}


def test_openfwi_profile_clamps_and_deterministic_wells():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        dataset_dir = root / "FlatVelA"
        for modality in ("depth_vel", "migrated_image", "horizon", "rms_vel"):
            (dataset_dir / modality).mkdir(parents=True)
        depth = np.linspace(1000.0, 5000.0, 70 * 70, dtype=np.float32).reshape(1, 70, 70)
        migrated = np.linspace(-500.0, 500.0, 70 * 70, dtype=np.float32).reshape(1, 70, 70)
        rms = np.linspace(1400.0, 4700.0, 70 * 70, dtype=np.float32).reshape(1, 70, 70)
        horizon = np.zeros((1, 70, 70), dtype=np.float32)
        for idx in range(2):
            np.save(dataset_dir / "depth_vel" / f"{idx}.npy", depth)
            np.save(dataset_dir / "migrated_image" / f"{idx}.npy", migrated)
            np.save(dataset_dir / "horizon" / f"{idx}.npy", horizon)
            np.save(dataset_dir / "rms_vel" / f"{idx}.npy", rms)

        dataset = OpenFWIBGDataset(
            root_dir=str(root),
            datasets=["FlatVelA"],
            split="all",
            target_shape=[70, 70],
            well_count_range=[3, 3],
            well_random=False,
            storage_backend="npy",
        )
        first = dataset[0]
        again = dataset[0]
        for key in ("depth_vel", "migrated_image", "rms_vel", "well_log"):
            assert first[key].min() >= -1.0
            assert first[key].max() <= 1.0
        assert torch.all(first["horizon"] == 0)
        assert int(first["well_mask"].sum().item()) == 210
        assert torch.all(first["well_log"][first["well_mask"] == 0] == 0)
        assert torch.equal(first["well_mask"], again["well_mask"])


def test_openfwi_default_preserves_native_time_domain_shapes():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_openfwi_fixture(root, count=2, time_domain=True, include_all=True)
        dataset = OpenFWIBGDataset(
            root_dir=str(root),
            datasets=["FlatVelA"],
            split="all",
            storage_backend="npy",
            target_shape=None,
            well_random=False,
        )
        item = dataset[0]
        assert item["depth_vel"].shape == (1, 70, 70)
        assert item["horizon"].shape == (1, 70, 70)
        assert item["well_log"].shape == (1, 70, 70)
        assert item["well_mask"].shape == (1, 70, 70)
        assert item["migrated_image"].shape == (1, 1000, 70)
        assert item["rms_vel"].shape == (1, 1000, 70)

        batch = next(iter(DataLoader(dataset, batch_size=2, collate_fn=collate_bg_samples)))
        validate_bg_batch(batch, context="native time-domain OpenFWI batch")
        assert batch.migrated_image.shape == (2, 1, 1000, 70)
        assert batch.rms_vel.shape == (2, 1, 1000, 70)


def test_openfwi_seeded_split_membership_is_reproducible_and_disjoint():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_openfwi_fixture(root, count=10, include_all=False)
        kwargs = dict(
            root_dir=str(root),
            datasets=["FlatVelA"],
            storage_backend="npy",
            split_fractions=[0.7, 0.2, 0.1],
            split_seed=99,
        )
        train = OpenFWIBGDataset(split="train", **kwargs)
        val = OpenFWIBGDataset(split="val", **kwargs)
        test = OpenFWIBGDataset(split="test", **kwargs)
        train_again = OpenFWIBGDataset(split="train", **kwargs)
        changed_seed = OpenFWIBGDataset(split="train", **{**kwargs, "split_seed": 100})

        train_ids = _record_ids(train)
        val_ids = _record_ids(val)
        test_ids = _record_ids(test)
        assert len(train_ids) == 7
        assert len(val_ids) == 2
        assert len(test_ids) == 1
        assert train_ids.isdisjoint(val_ids)
        assert train_ids.isdisjoint(test_ids)
        assert val_ids.isdisjoint(test_ids)
        assert len(train_ids | val_ids | test_ids) == 10
        assert train_ids == _record_ids(train_again)
        assert train_ids != _record_ids(changed_seed)


def test_openfwi_dynamic_wells_random_for_train_and_fixed_for_eval():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        _write_openfwi_fixture(root, count=1, include_all=False)
        random_dataset = OpenFWIBGDataset(
            root_dir=str(root),
            datasets=["FlatVelA"],
            split="all",
            storage_backend="npy",
            well_count_range=[1, 1],
            well_random=True,
        )
        fixed_dataset = OpenFWIBGDataset(
            root_dir=str(root),
            datasets=["FlatVelA"],
            split="all",
            storage_backend="npy",
            well_count_range=[1, 1],
            well_random=False,
        )

        random_columns = set()
        for _ in range(8):
            mask = random_dataset[0]["well_mask"][0]
            random_columns.add(tuple(torch.nonzero(mask.sum(dim=0) > 0, as_tuple=False).flatten().tolist()))
        fixed_first = fixed_dataset[0]["well_mask"]
        fixed_second = fixed_dataset[0]["well_mask"]

        assert len(random_columns) > 1
        assert torch.equal(fixed_first, fixed_second)


def test_openfwi_lmdb_backend_matches_npy_if_available():
    try:
        from bg_pdr_fm.data.openfwi_lmdb import openfwi_lmdb_path, require_lmdb, write_openfwi_lmdb_dataset

        require_lmdb()
    except ModuleNotFoundError:
        return

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "openfwi"
        lmdb_root = Path(tmp) / "openfwi_lmdb"
        dataset_dir = root / "FlatVelA"
        for modality in ("depth_vel", "migrated_image", "horizon", "rms_vel"):
            (dataset_dir / modality).mkdir(parents=True)
        arrays = {
            "depth_vel": np.full((1, 70, 70), 3000.0, dtype=np.float32),
            "migrated_image": np.linspace(-10.0, 10.0, 70 * 70, dtype=np.float32).reshape(1, 70, 70),
            "horizon": np.zeros((1, 70, 70), dtype=np.float32),
            "rms_vel": np.full((1, 70, 70), 2500.0, dtype=np.float32),
        }
        for modality, array in arrays.items():
            np.save(dataset_dir / modality / "0.npy", array)
        write_openfwi_lmdb_dataset(
            source_dataset_dir=dataset_dir,
            lmdb_path=openfwi_lmdb_path(lmdb_root, "FlatVelA"),
            dataset_name="FlatVelA",
        )
        npy_dataset = OpenFWIBGDataset(root_dir=str(root), datasets=["FlatVelA"], split="all",
                                       storage_backend="npy", well_random=False)
        lmdb_dataset = OpenFWIBGDataset(root_dir=str(root), lmdb_root=lmdb_root, datasets=["FlatVelA"],
                                        split="all", storage_backend="lmdb", well_random=False)
        second_lmdb_dataset = OpenFWIBGDataset(root_dir=str(root), lmdb_root=lmdb_root, datasets=["FlatVelA"],
                                               split="all", storage_backend="lmdb", well_random=False)
        npy_item = npy_dataset[0]
        lmdb_item = lmdb_dataset[0]
        second_lmdb_item = second_lmdb_dataset[0]
        for key in ("depth_vel", "migrated_image", "horizon", "rms_vel", "well_log", "well_mask"):
            assert torch.equal(npy_item[key], lmdb_item[key])
            assert torch.equal(npy_item[key], second_lmdb_item[key])
        lmdb_dataset.close()
        second_lmdb_dataset.close()


def test_marmousi_batch_smoke_if_available():
    try:
        root = resolve_marmousi_root(root_dir="data/marmousi")
    except FileNotFoundError:
        return
    dataset = MarmousiBGDataset(root_dir=str(root), split="train", target_shape=(70, 70))
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_bg_samples)
    batch = next(iter(loader))
    assert batch.depth_vel.shape == (2, 1, 70, 70)
    assert batch.migrated_image.shape == (2, 1, 70, 70)
    assert batch.rms_vel.shape == (2, 1, 70, 70)
    assert batch.well_mask.shape == (2, 1, 70, 70)
    assert batch.modality_mask.shape == (2, 4)
    assert torch.all(batch.modality_mask[:, :3] == 1)
    assert set(batch.modality_mask[:, 3].tolist()).issubset({0.0, 1.0})
    assert torch.all(batch.modality_quality[:, :3] == 1)
    assert set(batch.modality_quality[:, 3].tolist()).issubset({0.0, 1.0})
    for tensor in (batch.depth_vel, batch.migrated_image, batch.horizon, batch.rms_vel, batch.well_log,
                   batch.well_mask):
        assert tensor.dtype == torch.float32
        assert torch.isfinite(tensor).all()


def test_openfwi_batch_smoke_if_available():
    try:
        root = resolve_openfwi_root(root_dir="data/openfwi")
    except FileNotFoundError:
        return
    dataset = OpenFWIBGDataset(root_dir=str(root), datasets=["FlatVelA", "CurveFaultA"], split="train",
                               target_shape=(70, 70))
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_bg_samples)
    batch = next(iter(loader))
    assert batch.depth_vel.shape == (2, 1, 70, 70)
    assert batch.migrated_image.shape == (2, 1, 70, 70)
    assert batch.rms_vel.shape == (2, 1, 70, 70)
    assert batch.well_mask.shape == (2, 1, 70, 70)
    assert batch.modality_mask.shape == (2, 4)
    assert torch.all(batch.modality_mask[:, :3] == 1)
    assert set(batch.modality_mask[:, 3].tolist()).issubset({0.0, 1.0})
    assert torch.all(batch.modality_quality[:, :3] == 1)
    assert set(batch.modality_quality[:, 3].tolist()).issubset({0.0, 1.0})
    for tensor in (batch.depth_vel, batch.migrated_image, batch.horizon, batch.rms_vel, batch.well_log,
                   batch.well_mask):
        assert tensor.dtype == torch.float32
        assert torch.isfinite(tensor).all()


def test_three_stage_fast_run_synthetic():
    with tempfile.TemporaryDirectory() as tmp:
        conf = _conf("background")
        conf.data = {"name": "synthetic", "train_length": 2, "val_length": 2}
        conf.training.save_stage_outputs = True
        conf.training.stage_checkpoint_path = str(Path(tmp) / "checkpoints")
        conf.training.logging = {"log_dir": str(Path(tmp) / "lightning"), "log_version": "three_stage_{stage}"}
        conf.training.checkpoint = {
            "dirpath": str(Path(tmp) / "lightning" / "{stage}" / "checkpoints"),
            "filename": "{stage}-epoch_{epoch}-loss{val/loss:.4f}",
            "monitor": "val/loss",
            "mode": "min",
            "save_top_k": 1,
            "save_last": True,
            "every_n_epochs": 1,
        }
        conf.training.update(
            {
                "batch_size": 2,
                "max_epochs": 1,
                "accelerator": "cpu",
                "devices": 1,
                "limit_train_batches": 1,
                "limit_val_batches": 1,
                "num_workers": 0,
            }
        )
        conf.diagnostics = {"enabled": True, "output_dir": tmp}
        for stage in ("contrastive", "background", "residual"):
            stage_conf = OmegaConf.create(OmegaConf.to_container(conf, resolve=False))
            stage_conf.training.stage = stage
            run_one_stage(stage_conf)
        csv_text = (Path(tmp) / "metrics.csv").read_text(encoding="utf-8")
        for stage in ("contrastive", "background", "residual"):
            assert stage in csv_text
        checkpoints = Path(tmp) / "checkpoints"
        for checkpoint_name, expected_stage in (
            ("contrastive_last.ckpt", "contrastive"),
            ("background_last.ckpt", "background"),
            ("residual_last.ckpt", "residual"),
        ):
            checkpoint = checkpoints / checkpoint_name
            assert checkpoint.is_file()
            assert torch.load(checkpoint, map_location="cpu")["stage"] == expected_stage
        for header in (
            "symile_structural",
            "symile_numerical",
            "pairwise_structural",
            "pairwise_numerical",
            "pairwise_retrieval_top1",
            "pairwise_retrieval_top5",
            "pairwise_retrieval_top5_numerical_well_log_rms_vel",
            "pairwise_alignment_numerical_well_log_rms_vel",
            "anchor",
            "reliability_prior",
            "sn_ortho",
            "unique_ortho",
            "observed_subset_count",
            "observed_modality_mean",
            "structural_anchor_alignment",
            "numerical_anchor_alignment",
            "frequency_structural_low_penalty",
            "frequency_numerical_high_penalty",
            "frequency_structural_high_ratio",
            "frequency_numerical_low_ratio",
            "numerical_low_energy_ratio",
            "structural_high_energy_ratio",
            "reliability_migrated_image_structural",
            "reliability_rms_vel_numerical",
        ):
            assert header in csv_text


def test_background_checkpoint_can_seed_residual_stage():
    batch = _batch()
    with tempfile.TemporaryDirectory() as tmp:
        conf = _conf("background")
        conf.training.save_stage_outputs = True
        conf.training.stage_checkpoint_path = tmp
        model = BGPDRFMLightning(conf)
        loss = model.training_step(batch, 0)
        loss.backward()
        model.on_train_epoch_end()
        checkpoint = Path(tmp) / "background_last.ckpt"
        assert checkpoint.is_file()

        residual_conf = _conf("residual")
        residual_conf.training.load_stage_checkpoint = str(checkpoint)
        residual = BGPDRFMLightning(residual_conf)
        assert residual.stage_checkpoint_used == str(checkpoint)
        residual_loss = residual.training_step(batch, 0)
        residual_loss.backward()
        assert torch.isfinite(residual_loss)


def test_evaluation_missing_checkpoint_requires_explicit_untrained():
    conf = _conf("residual")
    conf.evaluation = {"allow_untrained": False, "checkpoints": {}}
    model = BGPDRFMLightning(conf)
    try:
        load_evaluation_checkpoints(model, conf)
    except FileNotFoundError as exc:
        assert "evaluation requires a checkpoint" in str(exc)
    else:
        raise AssertionError("evaluation should require a checkpoint unless allow_untrained=true.")


def test_evaluation_bad_checkpoint_errors():
    with tempfile.TemporaryDirectory() as tmp:
        bad = Path(tmp) / "bad.ckpt"
        torch.save({"not_state_dict": {}}, bad)
        conf = _conf("residual")
        conf.evaluation = {"allow_untrained": False, "checkpoints": {"full": str(bad)}}
        model = BGPDRFMLightning(conf)
        try:
            load_evaluation_checkpoints(model, conf)
        except KeyError as exc:
            assert "does not contain `state_dict`" in str(exc)
        else:
            raise AssertionError("evaluation should reject checkpoints without state_dict.")


def test_evaluation_prefixed_checkpoint_no_match_errors():
    with tempfile.TemporaryDirectory() as tmp:
        bad = Path(tmp) / "bad_prefix.ckpt"
        torch.save({"state_dict": {"unrelated.weight": torch.ones(1)}}, bad)
        conf = _conf("residual")
        conf.evaluation = {"allow_untrained": False, "checkpoints": {"background": str(bad)}}
        model = BGPDRFMLightning(conf)
        try:
            load_evaluation_checkpoints(model, conf)
        except KeyError as exc:
            assert "has no keys for background" in str(exc)
        else:
            raise AssertionError("evaluation should reject stage checkpoints with no matching prefix.")


def test_evaluation_residual_checkpoint_loads_residual_stage_trainables():
    with tempfile.TemporaryDirectory() as tmp:
        conf = _conf("residual")
        model = BGPDRFMLightning(conf)
        state_dict = model.state_dict()
        filtered = {
            key: value
            for key, value in state_dict.items()
            if key.startswith(("adapters.structural.", "rho_calibrator.", "residual."))
        }
        checkpoint = Path(tmp) / "residual.ckpt"
        torch.save({"state_dict": filtered}, checkpoint)
        conf.evaluation = {
            "allow_untrained": False,
            "checkpoints": {"residual": str(checkpoint)},
        }

        fresh = BGPDRFMLightning(conf)
        loaded = load_evaluation_checkpoints(fresh, conf)
        assert loaded[0]["mode"] == "residual"
        assert loaded[0]["loaded_keys"] == len(filtered)
        assert any(key.startswith("adapters.structural.") for key in filtered)
        assert any(key.startswith("rho_calibrator.") for key in filtered)
        assert any(key.startswith("residual.") for key in filtered)


def test_evaluation_synthetic_outputs_files():
    batch = _batch(batch_size=1)
    with tempfile.TemporaryDirectory() as tmp:
        train_conf = _conf("residual")
        train_conf.training.save_stage_outputs = True
        train_conf.training.stage_checkpoint_path = str(Path(tmp) / "checkpoints")
        model = BGPDRFMLightning(train_conf)
        loss = model.training_step(batch, 0)
        loss.backward()
        model.on_train_epoch_end()
        checkpoint = Path(tmp) / "checkpoints" / "residual_last.ckpt"
        assert checkpoint.is_file()

        eval_conf = _conf("residual")
        eval_conf.data = {"name": "synthetic", "test_length": 1}
        eval_conf.training.batch_size = 1
        eval_conf.training.num_workers = 0
        eval_conf.training.accelerator = "cpu"
        eval_conf.evaluation = {
            "output_dir": str(Path(tmp) / "eval"),
            "split": "test",
            "max_batches": 1,
            "save_arrays": True,
            "save_panels": True,
            "allow_untrained": False,
            "checkpoints": {"full": str(checkpoint)},
        }
        summary = run_evaluation(eval_conf)
        output_dir = Path(tmp) / "eval"
        files = {path.name for path in output_dir.rglob("*") if path.is_file()}
        assert "metrics.csv" in files
        assert "summary.json" in files
        assert any(name.endswith("_velocity_hat.npy") for name in files)
        assert any(name.endswith(".png") for name in files)
        assert summary["num_samples"] == 1
        csv_text = (output_dir / "metrics.csv").read_text(encoding="utf-8")
        for header in (
            "mae", "rmse", "mse", "ssim", "mae_l", "mae_h",
            "bg_mae", "epsilon_H_B", "rho_B", "rho_hat_B",
            "residual_energy_ratio_l2", "transport_target_ratio", "alpha_hat",
        ):
            assert header in csv_text


def test_evaluation_synthetic_rejects_uninjected_legacy_fm():
    with tempfile.TemporaryDirectory() as tmp:
        conf = _conf("residual")
        conf.model.cond_channels = 64
        conf.model.latent_channels = 16
        conf.model.latent_hw = [16, 16]
        conf.model.residual_backend = "legacy_fm"
        conf.model.residual_num_inference_steps = 1
        conf.data = {"name": "synthetic", "test_length": 1}
        conf.training.batch_size = 1
        conf.training.num_workers = 0
        conf.evaluation = {
            "output_dir": str(Path(tmp) / "eval"),
            "split": "test",
            "max_batches": 1,
            "save_arrays": False,
            "save_panels": False,
            "allow_untrained": True,
            "checkpoints": {},
        }
        try:
            run_evaluation(conf)
        except ImportError as exc:
            assert "legacy_fm is not bundled" in str(exc)
            return
        raise AssertionError("evaluation should reject legacy_fm without an injected flow_cls.")


def _run_three_stage_fast(conf, tmp: str):
    conf.training.update(
        {
            "batch_size": 2,
            "max_epochs": 1,
            "accelerator": "cpu",
            "devices": 1,
            "limit_train_batches": 1,
            "limit_val_batches": 1,
            "num_workers": 0,
        }
    )
    conf.training.logging = {"log_dir": str(Path(tmp) / "lightning"), "log_version": "three_stage_{stage}"}
    conf.training.checkpoint = {
        "dirpath": str(Path(tmp) / "lightning" / "{stage}" / "checkpoints"),
        "filename": "{stage}-epoch_{epoch}-loss{val/loss:.4f}",
        "monitor": "val/loss",
        "mode": "min",
        "save_top_k": 1,
        "save_last": True,
        "every_n_epochs": 1,
    }
    conf.diagnostics = {"enabled": True, "output_dir": tmp}
    for stage in ("contrastive", "background", "residual"):
        stage_conf = OmegaConf.create(OmegaConf.to_container(conf, resolve=False))
        stage_conf.training.stage = stage
        run_one_stage(stage_conf)
    return (Path(tmp) / "metrics.csv").read_text(encoding="utf-8")


def test_three_stage_fast_run_marmousi_if_available():
    try:
        root = resolve_marmousi_root(root_dir="data/marmousi")
    except FileNotFoundError:
        return
    with tempfile.TemporaryDirectory() as tmp:
        conf = _conf("background")
        conf.data = {
            "name": "marmousi",
            "root_dir": str(root),
            "root_candidates": [],
            "target_shape": [70, 70],
            "val_split": "test",
            "use_normalize": "-1_1",
        }
        csv_text = _run_three_stage_fast(conf, tmp)
        for header in ("loss", "epsilon_H_B", "rho_B", "rho_hat_B", "alpha_hat"):
            assert header in csv_text
        for header in ("stage_checkpoint_used", "encoder_frozen", "codec_type", "residual_backend"):
            assert header in csv_text
        for stage in ("contrastive", "background", "residual"):
            assert stage in csv_text


def test_three_stage_fast_run_openfwi_if_available():
    try:
        root = resolve_openfwi_root(root_dir="data/openfwi", datasets=["FlatVelA"])
    except FileNotFoundError:
        return
    with tempfile.TemporaryDirectory() as tmp:
        conf = _conf("background")
        conf.data = {
            "name": "openfwi",
            "root_dir": str(root),
            "root_candidates": [],
            "openfwi_datasets": ["FlatVelA"],
            "required_modalities": ["depth_vel"],
            "target_shape": [70, 70],
            "val_split": "test",
            "val_fraction": 0.001,
            "use_normalize": "-1_1",
        }
        csv_text = _run_three_stage_fast(conf, tmp)
        for header in ("loss", "epsilon_H_B", "rho_B", "rho_hat_B", "alpha_hat"):
            assert header in csv_text
        for header in ("residual_backend", "residual_num_inference_steps"):
            assert header in csv_text
        for stage in ("contrastive", "background", "residual"):
            assert stage in csv_text


def test_diagnostics_outputs_png_and_csv():
    batch = _batch()
    with tempfile.TemporaryDirectory() as tmp:
        conf = _conf("background")
        conf.diagnostics = {"enabled": True, "output_dir": tmp}
        model = BGPDRFMLightning(conf)
        loss = model.training_step(batch, 0)
        assert torch.isfinite(loss)
        files = {path.name for path in Path(tmp).rglob("*") if path.is_file()}
        assert "background_panel.png" in files
        assert "feature_panel.png" in files
        assert "rho_calibration.png" in files
        assert "metrics.csv" in files
        csv_text = next(Path(tmp).rglob("metrics.csv")).read_text(encoding="utf-8")
        for header in ("loss", "epsilon_H_B", "rho_B", "rho_hat_B", "alpha_hat"):
            assert header in csv_text


def test_contrastive_diagnostics_outputs_core_metrics():
    batch = _batch()
    with tempfile.TemporaryDirectory() as tmp:
        conf = _conf("contrastive")
        conf.diagnostics = {"enabled": True, "output_dir": tmp}
        model = BGPDRFMLightning(conf)
        loss = model.training_step(batch, 0)
        assert torch.isfinite(loss)
        files = {path.name for path in Path(tmp).rglob("*") if path.is_file()}
        assert "feature_panel.png" in files
        assert "background_panel.png" not in files
        assert "rho_calibration.png" not in files
        csv_text = next(Path(tmp).rglob("metrics.csv")).read_text(encoding="utf-8")
        for header in (
            "symile_structural",
            "symile_numerical",
            "anchor",
            "reliability_prior",
            "sn_ortho",
            "unique_ortho",
            "observed_subset_count",
            "observed_modality_mean",
            "structural_anchor_alignment",
            "numerical_anchor_alignment",
            "numerical_low_energy_ratio",
            "structural_high_energy_ratio",
            "reliability_migrated_image_structural",
            "reliability_rms_vel_numerical",
        ):
            assert header in csv_text


def test_contrastive_visualization_synthetic_outputs_files():
    if not torch.cuda.is_available():
        pytest.skip("Contrastive visualization smoke test is GPU-only.")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        conf = _conf("contrastive")
        conf.data = {"name": "synthetic", "test_length": 4}
        conf.training.batch_size = 2
        conf.training.accelerator = "gpu"
        conf.training.devices = [0]

        checkpoint = tmp_path / "contrastive.ckpt"
        torch.save({"state_dict": BGPDRFMLightning(conf).to("cuda:0").state_dict()}, checkpoint)
        config_path = tmp_path / "contrastive.yaml"
        OmegaConf.save(conf, config_path)
        metrics_path = tmp_path / "metrics.csv"
        metrics_path.write_text(
            "\n".join(
                [
                    "epoch,step,train/loss,val/loss,train/structural_anchor_alignment,"
                    "val/structural_anchor_alignment,train/reliability_migrated_image_structural,"
                    "val/reliability_migrated_image_structural",
                    "0,1,9.0,,0.2,,0.3,",
                    "0,1,,8.8,,0.4,,0.5",
                    "1,2,8.5,,0.6,,0.7,",
                    "1,2,,8.4,,0.8,,0.9",
                ]
            ),
            encoding="utf-8",
        )

        output_dir = tmp_path / "visualization"
        summary = run_contrastive_visualization(
            config=config_path,
            checkpoint=checkpoint,
            metrics_csv=metrics_path,
            split="test",
            max_samples=4,
            batch_size=2,
            output_dir=output_dir,
        )
        files = {path.name for path in output_dir.rglob("*") if path.is_file()}
        assert summary["num_samples"] == 4
        assert "summary.json" in files
        assert "sample_metrics.csv" in files
        assert "loss.png" in files
        assert "mean_reliability.png" in files
        assert "structural_cosine_heatmap.png" in files
        assert "retrieval_metrics.csv" in files


def test_aaai27_missing_modality_protocol_masks_inputs():
    batch = _batch()
    assert list(AAAI27_MISSING_MODES) == [
        "full",
        "w/o well_log",
        "w/o horizon",
        "w/o rms_vel",
        "w/o well+rms",
        "PSTM only",
    ]
    pstm_only = apply_missing_modality_mode(batch, "PSTM only")
    assert torch.all(pstm_only.modality_mask[:, 0] == batch.modality_mask[:, 0])
    assert torch.all(pstm_only.modality_mask[:, 1:] == 0)
    assert torch.count_nonzero(pstm_only.horizon) == 0
    assert torch.count_nonzero(pstm_only.rms_vel) == 0
    assert torch.count_nonzero(pstm_only.well_log) == 0
    assert torch.count_nonzero(pstm_only.well_mask) == 0


def test_adapted_gfi_input_builder_respects_missing_modalities():
    batch = _batch(1)
    full = build_adapted_gfi_input(batch, input_hw=(1000, 70))
    assert full.shape == (1, 5, 1000, 70)

    no_well = build_adapted_gfi_input(apply_missing_modality_mode(batch, "w/o well_log"), input_hw=(1000, 70))
    assert torch.count_nonzero(no_well[:, 3]) == 0
    assert torch.count_nonzero(no_well[:, 4]) == 0

    pstm_only = build_adapted_gfi_input(apply_missing_modality_mode(batch, "PSTM only"), input_hw=(1000, 70))
    assert torch.count_nonzero(pstm_only[:, 1:]) == 0
    assert torch.count_nonzero(pstm_only[:, 0]) == torch.count_nonzero(full[:, 0])


def test_adapted_gfi_latent_unet_matches_official_capacity_scale():
    model = AdaptedGFILatentUNetMultimodal(
        input_hw=(128, 70),
        latent_hw=(16, 16),
        latent_channels=128,
        unet_depth=2,
        unet_repeat_blocks=2,
    )
    params = sum(parameter.numel() for parameter in model.parameters())

    assert 34_000_000 <= params <= 37_000_000


def test_adapted_gfi_latent_unet_forward_shape_is_finite():
    batch = _batch(1)
    model = AdaptedGFILatentUNetMultimodal(
        input_hw=(128, 70),
        latent_hw=(16, 16),
        latent_channels=32,
        unet_depth=1,
        unet_repeat_blocks=1,
    )
    velocity_hat = model(batch)

    assert velocity_hat.shape == batch.depth_vel.shape
    assert torch.isfinite(velocity_hat).all()


def test_multimodal_condition_image_builder_respects_missing_modalities():
    batch = _batch(1)
    full = build_multimodal_condition_image(batch, input_hw=(1000, 70))
    assert full.shape == (1, 5, 1000, 70)
    assert torch.isfinite(full).all()

    no_well = build_multimodal_condition_image(apply_missing_modality_mode(batch, "w/o well_log"), input_hw=(1000, 70))
    assert torch.count_nonzero(no_well[:, 3]) == 0
    assert torch.count_nonzero(no_well[:, 4]) == 0

    pstm_only = build_multimodal_condition_image(apply_missing_modality_mode(batch, "PSTM only"), input_hw=(1000, 70))
    assert torch.count_nonzero(pstm_only[:, 1:]) == 0
    assert torch.count_nonzero(pstm_only[:, 0]) == torch.count_nonzero(full[:, 0])


def test_adapted_auto_linear_input_builder_respects_missing_modalities():
    batch = _batch(1)
    full = build_auto_linear_measurement_input(batch, input_hw=(1000, 70))
    assert full.shape == (1, 5, 1000, 70)

    no_well = build_auto_linear_measurement_input(
        apply_missing_modality_mode(batch, "w/o well_log"),
        input_hw=(1000, 70),
    )
    assert torch.count_nonzero(no_well[:, 3]) == 0
    assert torch.count_nonzero(no_well[:, 4]) == 0

    pstm_only = build_auto_linear_measurement_input(
        apply_missing_modality_mode(batch, "PSTM only"),
        input_hw=(1000, 70),
    )
    assert torch.count_nonzero(pstm_only[:, 1:]) == 0
    assert torch.count_nonzero(pstm_only[:, 0]) == torch.count_nonzero(full[:, 0])


def test_smooth_dix_predicts_velocity_shape_and_zero_params():
    from bg_pdr_fm.external_baselines.smooth_dix import SmoothDixBaseline

    batch = _batch(2)
    model = SmoothDixBaseline(kernel_size=9)
    velocity_hat = model(batch)

    assert velocity_hat.shape == batch.depth_vel.shape
    assert torch.isfinite(velocity_hat).all()
    assert sum(parameter.numel() for parameter in model.parameters()) == 0


def test_sv_inv_net_forward_shape_and_capacity():
    from bg_pdr_fm.external_baselines.sv_inv_net import SVInvNetMultimodal

    batch = _batch(1)
    model = SVInvNetMultimodal(base_channels=16, growth_rate=8, dense_layers=2, input_hw=(128, 70))
    velocity_hat = model(batch)
    params = sum(parameter.numel() for parameter in model.parameters())

    assert velocity_hat.shape == batch.depth_vel.shape
    assert torch.isfinite(velocity_hat).all()
    assert params > 200_000


def test_adapted_inversion_net_forward_shape_and_capacity():
    from bg_pdr_fm.external_baselines.inversion_net import AdaptedInversionNetMultimodal

    batch = _batch(2)
    model = AdaptedInversionNetMultimodal()
    velocity_hat = model(batch)
    params = sum(parameter.numel() for parameter in model.parameters())

    assert velocity_hat.shape == batch.depth_vel.shape
    assert torch.isfinite(velocity_hat).all()
    assert params == 24_409_123


def test_conditional_ddpm_training_loss_and_prediction_shape():
    from bg_pdr_fm.external_baselines.conditional_ddpm import ConditionalDDPMBaseline

    batch = _batch(1)
    model = ConditionalDDPMBaseline(
        sample_size=70,
        condition_input_hw=(128, 70),
        condition_dim=32,
        num_condition_tokens=4,
        train_timesteps=10,
        inference_steps=1,
        block_out_channels=(16, 32),
    )
    losses = model.training_loss(batch)
    velocity_hat = model.sample(batch)

    assert losses["loss"].ndim == 0
    assert torch.isfinite(losses["loss"])
    assert velocity_hat.shape == batch.depth_vel.shape
    assert torch.isfinite(velocity_hat).all()


def test_velocity_gan_generator_and_losses_are_finite():
    from bg_pdr_fm.external_baselines.velocity_gan import VelocityGANBaseline

    batch = _batch(2)
    model = VelocityGANBaseline(l1_weight=100.0, mse_weight=0.0)
    losses = model.generator_loss(batch)
    disc_loss = model.discriminator_loss(batch)
    velocity_hat = model(batch)

    assert velocity_hat.shape == batch.depth_vel.shape
    assert torch.isfinite(losses["loss"])
    assert torch.isfinite(losses["adversarial"])
    assert torch.isfinite(losses["l1"])
    assert torch.isfinite(disc_loss["loss"])


def test_adapted_auto_linear_uses_masked_autoencoders_and_low_rank_converter():
    from bg_pdr_fm.external_baselines.auto_linear.adapted_multimodal import AdaptedAutoLinearMultimodal

    model = AdaptedAutoLinearMultimodal(
        embed_dim=64,
        depth=1,
        num_heads=4,
        latent_tokens=4,
        converter_rank=16,
        input_hw=(128, 70),
        measurement_patch_size=(16, 10),
    )

    assert hasattr(model, "measurement_mae")
    assert hasattr(model, "velocity_mae")
    assert hasattr(model, "latent_converter")
    assert hasattr(model.latent_converter, "down")
    assert hasattr(model.latent_converter, "up")


def test_adapted_auto_linear_masked_losses_are_finite():
    from bg_pdr_fm.external_baselines.auto_linear.adapted_multimodal import AdaptedAutoLinearMultimodal

    batch = _batch(1)
    model = AdaptedAutoLinearMultimodal(
        embed_dim=32,
        depth=1,
        num_heads=4,
        latent_tokens=4,
        converter_rank=8,
        input_hw=(128, 70),
        measurement_patch_size=(16, 10),
        mask_ratio=0.25,
    )
    ae = model.ae_pretrain_loss(batch)
    inv = model.inverse_linear_loss(batch)
    velocity_hat = model(batch)

    assert torch.isfinite(ae["loss"])
    assert torch.isfinite(inv["loss"])
    assert velocity_hat.shape == batch.depth_vel.shape


def test_adapted_auto_linear_original_matches_paper_hyperparameters():
    model = AdaptedAutoLinearOriginalMultimodal()
    assert model.measurement_mae.patch_size == (100, 10)
    assert model.velocity_mae.patch_size == (10, 10)
    assert model.mask_ratio == pytest.approx(0.75)
    assert model.latent_converter.rank == 128
    assert model.latent_converter.input_shape == (70, 132)
    assert model.latent_converter.output_shape == (49, 516)
    assert model.measurement_mae.encoder_dim == 132
    assert model.measurement_mae.encoder_depth == 2
    assert model.measurement_mae.encoder_heads == 12
    assert model.measurement_mae.decoder_dim == 512
    assert model.measurement_mae.decoder_depth == 2
    assert model.measurement_mae.decoder_heads == 16
    assert model.velocity_mae.encoder_dim == 516
    assert model.velocity_mae.encoder_depth == 3
    assert model.velocity_mae.encoder_heads == 12
    assert model.velocity_mae.decoder_dim == 512
    assert model.velocity_mae.decoder_depth == 2
    assert model.velocity_mae.decoder_heads == 16


def test_adapted_auto_linear_original_paper35m_matches_reported_capacity_and_l1_loss():
    batch = _batch(1)
    model = AdaptedAutoLinearOriginalMultimodal(converter_rank=375)
    params = sum(parameter.numel() for parameter in model.parameters())
    assert abs(params - 35_450_000) < 50_000
    assert model.latent_converter.rank == 375

    inverse_losses = model.inverse_linear_loss(batch, supervised_loss="l1")
    assert torch.allclose(inverse_losses["loss"], inverse_losses["velocity_l1"])


def test_adapted_auto_linear_original_forward_and_losses_are_finite():
    batch = _batch(1)
    model = AdaptedAutoLinearOriginalMultimodal()
    velocity_hat = model(batch)
    assert velocity_hat.shape == batch.depth_vel.shape
    assert torch.isfinite(velocity_hat).all()

    ae_losses = model.ae_pretrain_loss(batch)
    inverse_losses = model.inverse_linear_loss(batch)
    assert ae_losses["velocity_hat"].shape == batch.depth_vel.shape
    assert inverse_losses["velocity_hat"].shape == batch.depth_vel.shape
    assert torch.isfinite(ae_losses["loss"])
    assert torch.isfinite(inverse_losses["loss"])


@pytest.mark.parametrize("phase", ["ae_pretrain", "inverse_linear"])
def test_adapted_auto_linear_phase_loss_and_trainability(phase):
    conf = _conf("residual")
    conf.benchmark = {
        "variant": "adapted_auto_linear",
        "capacity_tier": "adapted_external",
        "auto_linear_phase": phase,
        "auto_linear_embed_dim": 32,
        "auto_linear_depth": 1,
        "auto_linear_num_heads": 4,
        "auto_linear_latent_tokens": 4,
        "auto_linear_converter_rank": 8,
        "auto_linear_input_hw": [128, 70],
        "auto_linear_measurement_patch_size": [16, 10],
        "auto_linear_mask_ratio": 0.25,
    }
    batch = _batch(1)
    model = AAAI27BenchmarkLightning(conf)
    loss = model.training_step(batch, 0)
    assert torch.isfinite(loss)
    assert model.benchmark_metadata()["generator_params"] > 0

    trainable_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if phase == "ae_pretrain":
        assert any(name.startswith("adapted_auto_linear.measurement_mae") for name in trainable_names)
        assert any(name.startswith("adapted_auto_linear.velocity_mae") for name in trainable_names)
        assert not any(name.startswith("adapted_auto_linear.latent_converter") for name in trainable_names)
    else:
        assert any(name.startswith("adapted_auto_linear.latent_converter") for name in trainable_names)
        assert not any(name.startswith("adapted_auto_linear.measurement_mae") for name in trainable_names)
        assert not any(name.startswith("adapted_auto_linear.velocity_mae") for name in trainable_names)


@pytest.mark.parametrize("phase", ["ae_pretrain", "inverse_linear"])
def test_adapted_auto_linear_original_phase_loss_and_trainability(phase):
    conf = _conf("residual")
    conf.benchmark = {
        "variant": "adapted_auto_linear_original",
        "capacity_tier": "adapted_external",
        "auto_linear_phase": phase,
        "auto_linear_input_hw": [1000, 70],
    }
    batch = _batch(1)
    model = AAAI27BenchmarkLightning(conf)
    loss = model.training_step(batch, 0)
    assert torch.isfinite(loss)
    assert model.benchmark_metadata()["generator_params"] > 12_000_000

    trainable_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if phase == "ae_pretrain":
        assert any(name.startswith("adapted_auto_linear_original.measurement_mae") for name in trainable_names)
        assert any(name.startswith("adapted_auto_linear_original.velocity_mae") for name in trainable_names)
        assert not any(name.startswith("adapted_auto_linear_original.latent_converter") for name in trainable_names)
    else:
        assert any(name.startswith("adapted_auto_linear_original.latent_converter") for name in trainable_names)
        assert not any(name.startswith("adapted_auto_linear_original.measurement_mae") for name in trainable_names)
        assert not any(name.startswith("adapted_auto_linear_original.velocity_mae") for name in trainable_names)


def test_adapted_auto_linear_original_optimizer_can_match_paper_schedule():
    conf = _conf("residual")
    conf.benchmark = {
        "variant": "adapted_auto_linear_original",
        "capacity_tier": "adapted_external",
        "auto_linear_phase": "inverse_linear",
        "auto_linear_input_hw": [1000, 70],
        "auto_linear_converter_rank": 375,
    }
    conf.training.lr = 1e-3
    conf.training.weight_decay = 0.05
    conf.training.scheduler = "cosine"
    conf.training.max_epochs = 100
    model = AAAI27BenchmarkLightning(conf)
    configured = model.configure_optimizers()
    optimizer = configured["optimizer"]
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.05)
    assert configured["lr_scheduler"]["interval"] == "epoch"


def test_benchmark_config_extends_and_strong_capacity_is_larger():
    matched = load_benchmark_config("bg_pdr_fm/configs/experiments/aaai27/concat_fm_matched.yaml")
    strong = load_benchmark_config("bg_pdr_fm/configs/experiments/aaai27/concat_fm_strong.yaml")
    assert matched.benchmark.variant == "concat_fm"
    assert matched.training.accelerator == "cpu"
    matched_model = AAAI27BenchmarkLightning(matched)
    strong_model = AAAI27BenchmarkLightning(strong)
    assert matched_model.benchmark_metadata()["capacity_tier"] == "matched"
    assert strong_model.benchmark_metadata()["capacity_tier"] == "strong"
    assert strong_model.benchmark_metadata()["params"] > matched_model.benchmark_metadata()["params"]


def test_pdr_gfi_aligned_direct_upper_bound_config_loads():
    from bg_pdr_fm.training.train_aaai27_benchmark import validate_benchmark_config

    conf = load_benchmark_config("bg_pdr_fm/configs/experiments/aaai27/pdr_gfi_aligned_direct_upper_bound.yaml")
    validate_benchmark_config(conf)
    model = AAAI27BenchmarkLightning(conf)
    metadata = model.benchmark_metadata()

    assert conf.benchmark.variant == "pdr_gfi_aligned_direct"
    assert metadata["variant"] == "pdr_gfi_aligned_direct"
    assert metadata["generator_params"] > 0
    assert metadata["trainable_generator_params"] == metadata["generator_params"]


@pytest.mark.parametrize(
    "variant",
    [
        "smooth_dix",
        "sv_inv_net",
        "conditional_ddpm",
        "concat_fm",
        "cncs_fm",
        "bg_pdr_fm",
        "adapted_gfi",
        "adapted_gfi_latent_unet",
        "pdr_gfi_aligned_direct",
        "adapted_auto_linear",
    ],
)
def test_aaai27_benchmark_variants_predict(variant):
    conf = _conf("residual")
    conf.benchmark = {
        "variant": variant,
        "capacity_tier": "matched",
        "generator_hidden_channels": 32,
        "direct_hidden_channels": 32,
        "full_field_backend": "simple_fm",
        "adapted_gfi_base_channels": 8,
        "adapted_gfi_input_hw": [1000, 70],
        "adapted_gfi_latent_input_hw": [128, 70],
        "adapted_gfi_latent_hw": [16, 16],
        "adapted_gfi_latent_channels": 32,
        "adapted_gfi_unet_depth": 1,
        "adapted_gfi_unet_repeat_blocks": 1,
        "auto_linear_phase": "inverse_linear",
        "auto_linear_embed_dim": 32,
        "auto_linear_depth": 1,
        "auto_linear_num_heads": 4,
        "auto_linear_latent_tokens": 4,
        "auto_linear_converter_rank": 8,
        "auto_linear_input_hw": [128, 70],
        "auto_linear_measurement_patch_size": [16, 10],
        "auto_linear_mask_ratio": 0.25,
        "sv_inv_net_base_channels": 8,
        "sv_inv_net_growth_rate": 4,
        "sv_inv_net_dense_layers": 1,
        "sv_inv_net_input_hw": [128, 70],
        "ddpm_condition_input_hw": [128, 70],
        "ddpm_condition_dim": 32,
        "ddpm_num_condition_tokens": 4,
        "ddpm_block_out_channels": [16, 32],
        "ddpm_inference_steps": 1,
    }
    conf.model.residual_num_train_timesteps = 20
    conf.model.residual_num_inference_steps = 1
    batch = _batch(1)
    model = AAAI27BenchmarkLightning(conf)
    loss = model.training_step(batch, 0)
    assert torch.isfinite(loss)
    prediction = model.predict_batch(batch)
    assert prediction.velocity_hat.shape == batch.depth_vel.shape
    assert prediction.bg_hat.shape == batch.depth_vel.shape
    assert prediction.rho_hat_b.shape == (batch.depth_vel.shape[0],)
    if variant == "pdr_gfi_aligned_direct":
        assert torch.allclose(prediction.bg_hat, model.filter.lowpass(prediction.velocity_hat))
        assert torch.allclose(prediction.residual_hat, prediction.velocity_hat - prediction.bg_hat)


def test_openfwi_main_table_method_list_uses_paper_faithful_baselines():
    from bg_pdr_fm.evaluation.run_aaai27_openfwi_eval import METHODS

    names = [item["name"] for item in METHODS]
    assert "mm_invnet" not in names
    assert "two_stage_ddpm" not in names
    assert names == [
        "smooth_dix",
        "sv_inv_net",
        "adapted_inversion_net",
        "adapted_upfwi",
        "velocity_gan",
        "conditional_ddpm",
        "adapted_gfi",
        "adapted_auto_linear",
        "bg_pdr_fm",
    ]


def test_compute_freq_metrics_uses_low_high_filter_fields():
    from bg_pdr_fm.models import LowHighPassFilter

    target = torch.randn(2, 1, 70, 70)
    pred = target + 0.1 * torch.randn_like(target)
    filter_module = LowHighPassFilter(kernel_size=5)
    metrics = compute_freq_metrics(pred, target, filter_module)
    assert set(metrics) == {"mae_l", "mae_h"}
    assert metrics["mae_l"].shape == (2,)
    assert metrics["mae_h"].shape == (2,)


def test_aaai27_benchmark_evaluation_writes_unified_schema():
    with tempfile.TemporaryDirectory() as tmp:
        conf = load_benchmark_config("bg_pdr_fm/configs/experiments/aaai27/cncs_fm_matched.yaml")
        conf.evaluation.output_dir = str(Path(tmp) / "eval")
        conf.evaluation.missing_modes = ["full", "PSTM only"]
        conf.evaluation.max_batches = 1
        conf.training.batch_size = 2
        summary = run_benchmark_evaluation(conf)
        metrics_csv = Path(tmp) / "eval" / "metrics.csv"
        assert summary["num_samples"] == 4
        assert metrics_csv.exists()
        csv_text = metrics_csv.read_text(encoding="utf-8")
        for header in (
            "method",
            "capacity_tier",
            "dataset_name",
            "dataset_id",
            "dataset_sample_index",
            "missing_mode",
            "mae",
            "rmse",
            "ssim",
            "mae_l",
            "mae_h",
            "rho_B",
            "rho_hat_B",
            "E_R_over_E_V",
            "residual_energy_ratio_l2",
            "transport_target_ratio",
            "params",
            "inference_time",
        ):
            assert header in csv_text
        assert "PSTM only" in csv_text
        missing_mode_csv = Path(tmp) / "eval" / "missing_mode_summary.csv"
        assert missing_mode_csv.exists()
        assert "missing_mode" in missing_mode_csv.read_text(encoding="utf-8")
        assert (Path(tmp) / "eval" / "dataset_summary.csv").exists()
        dataset_mode_csv = Path(tmp) / "eval" / "dataset_missing_mode_summary.csv"
        assert dataset_mode_csv.exists()
        dataset_mode_text = dataset_mode_csv.read_text(encoding="utf-8")
        assert "dataset_name" in dataset_mode_text
        assert "missing_mode" in dataset_mode_text
