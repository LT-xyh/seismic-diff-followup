r"""Training entry point for BG-PDR-FM.

Run examples:

    # Safe minimal default config.
    D:\Conda\envs\seg\python.exe -m bg_pdr_fm.training.train_bg_pdr_fm

    # Three-stage one-batch smoke run.
    D:\Conda\envs\seg\python.exe -m bg_pdr_fm.training.train_bg_pdr_fm \
        --config bg_pdr_fm/configs/fast_run.yaml

    # OpenFWI LMDB training template.
    D:\Conda\envs\seg\python.exe -m bg_pdr_fm.training.train_bg_pdr_fm \
        --config bg_pdr_fm/configs/openfwi_lmdb_train.yaml

What this script does:
    1. Read `bg_pdr_fm/configs/bg_pdr_fm.yaml`.
    2. Build the selected dataset: synthetic, Marmousi, or OpenFWI.
    3. Build train/validation DataLoaders with BG-PDR-FM's fixed batch schema.
    4. Instantiate `BGPDRFMLightning`.
    5. Train either one stage or every stage listed in `training.fast_run_stages`.

Important config fields:
    data.name: synthetic | marmousi | openfwi
    data.split_fractions: train/val/test ratio, default [0.7, 0.2, 0.1]
    data.split_seed: fixed seed for reproducible random split membership
    data.storage_backend: auto | lmdb | npy, used by OpenFWI
    training.stage: contrastive | background | residual | joint_full
    training.fast_run: true means smoke-test the whole flow
    training.fast_run_stages: optional list of stages to run one after another
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import lightning
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bg_pdr_fm.data import (
    MarmousiBGDataset,
    OpenFWIBGDataset,
    SyntheticBGDataset,
    collate_bg_samples,
    validate_bg_batch,
)
from bg_pdr_fm.lightning import BGPDRFMLightning
from bg_pdr_fm.reproducibility import seed_everything
from bg_pdr_fm.runtime import configure_checkpoint_temp_dir, configure_torch_runtime
from bg_pdr_fm.training.benchmark_config import load_benchmark_config


FAST_RUN_STAGES = ("contrastive", "background", "residual")
VALID_STAGES = set(FAST_RUN_STAGES) | {"joint_full"}


def apply_training_seed(seed: int | None) -> int | None:
    """Seed Python, NumPy, PyTorch, and DataLoader workers for one stage.

    The data split and deterministic well placement are controlled separately
    by ``data.split_seed`` and ``data.well_seed``. ``training.seed`` controls
    model initialization and stochastic training behavior only.
    """
    if seed is None:
        return None
    return seed_everything(int(seed))


def build_dataset(conf, split: str):
    """Create one dataset object for the requested split.

    The returned dataset already follows the BG-PDR-FM sample schema:
    `depth_vel`, `migrated_image`, `horizon`, `rms_vel`, `well_log`,
    `well_mask`, `modality_mask`, and `modality_quality`.

    Split behavior:
        - OpenFWI uses a seeded random 7:2:1 split by default.
        - Marmousi keeps its physical `test/` folder independent, and creates
          train/val from the physical `train/` folder.
        - Synthetic is only for smoke tests.
    """
    dataset_name = str(OmegaConf.select(conf, "data.name", default="synthetic"))

    # Dynamic well generation is shared by OpenFWI and Marmousi. `well_random`
    # can be left null in the config: train will sample new wells repeatedly,
    # while validation/test use deterministic wells.
    well_count_range = OmegaConf.select(conf, "data.well_count_range", default=[0, 3])
    well_seed = int(OmegaConf.select(conf, "data.well_seed", default=1234))
    well_random = OmegaConf.select(conf, "data.well_random", default=None)

    # `auto` normally picks the profile matching the dataset name. When the
    # new checkpoint-backed AE codec is enabled, use the same global velocity
    # anchor as AE training so latent statistics remain compatible.
    normalization_profile = str(OmegaConf.select(conf, "data.normalization_profile", default="auto"))
    if normalization_profile == "auto":
        codec_type = str(OmegaConf.select(conf, "model.codec_type", default="simple")).lower()
        normalization_profile = "seismic_global" if codec_type in {"autoencoder", "vae"} else dataset_name

    if dataset_name == "synthetic":
        return SyntheticBGDataset(length=int(OmegaConf.select(conf, f"data.{split}_length", default=8)))

    if dataset_name == "marmousi":
        root_candidates = OmegaConf.select(conf, "data.root_candidates", default=[])
        root_candidates = list(root_candidates) if root_candidates is not None else []
        return MarmousiBGDataset(
            root_dir=OmegaConf.select(conf, "data.root_dir", default=None),
            root_candidates=root_candidates,
            split=split,
            use_normalize=OmegaConf.select(conf, "data.use_normalize", default="-1_1"),
            target_shape=OmegaConf.select(conf, "data.target_shape", default=None),
            return_metadata=bool(OmegaConf.select(conf, "data.return_metadata", default=False)),
            split_fractions=OmegaConf.select(conf, "data.split_fractions", default=[0.7, 0.2, 0.1]),
            split_seed=int(OmegaConf.select(conf, "data.split_seed", default=42)),
            well_count_range=well_count_range,
            well_seed=well_seed,
            well_random=well_random,
            normalization_profile=normalization_profile,
            normalize_clamp=bool(OmegaConf.select(conf, "data.normalize_clamp", default=True)),
        )

    if dataset_name == "openfwi":
        root_candidates = OmegaConf.select(conf, "data.root_candidates", default=[])
        root_candidates = list(root_candidates) if root_candidates is not None else []
        openfwi_datasets = OmegaConf.select(conf, "data.openfwi_datasets", default=None)
        openfwi_datasets = list(openfwi_datasets) if openfwi_datasets is not None else None
        post_split_datasets = OmegaConf.select(conf, "data.post_split_datasets", default=None)
        post_split_datasets = list(post_split_datasets) if post_split_datasets is not None else None
        required_modalities = OmegaConf.select(conf, "data.required_modalities", default=None)
        required_modalities = list(required_modalities) if required_modalities is not None else None
        return OpenFWIBGDataset(
            root_dir=OmegaConf.select(conf, "data.root_dir", default=None),
            root_candidates=root_candidates,
            datasets=openfwi_datasets,
            post_split_datasets=post_split_datasets,
            split=split,
            use_normalize=OmegaConf.select(conf, "data.use_normalize", default="-1_1"),
            target_shape=OmegaConf.select(conf, "data.target_shape", default=None),
            required_modalities=required_modalities,
            split_fractions=OmegaConf.select(conf, "data.split_fractions", default=[0.7, 0.2, 0.1]),
            split_seed=int(OmegaConf.select(conf, "data.split_seed", default=42)),
            val_fraction=OmegaConf.select(conf, "data.val_fraction", default=None),
            well_count_range=well_count_range,
            well_seed=well_seed,
            well_random=well_random,
            storage_backend=OmegaConf.select(conf, "data.storage_backend", default="auto"),
            lmdb_root=OmegaConf.select(conf, "data.lmdb_root", default=None),
            normalization_profile=normalization_profile,
            normalize_clamp=bool(OmegaConf.select(conf, "data.normalize_clamp", default=True)),
        )
    raise ValueError(f"Unknown data.name: {dataset_name}")


def _validation_split(conf) -> str:
    """Return which split should be used for validation during training."""
    dataset_name = str(OmegaConf.select(conf, "data.name", default="synthetic"))
    if dataset_name == "synthetic":
        return "val"
    return str(OmegaConf.select(conf, "data.val_split", default="val"))


def build_loaders(conf):
    """Build and sanity-check the train and validation DataLoaders."""
    train_set = build_dataset(conf, "train")
    val_set = build_dataset(conf, _validation_split(conf))
    batch_size = int(OmegaConf.select(conf, "training.batch_size", default=2))
    num_workers = int(OmegaConf.select(conf, "training.num_workers", default=4))

    # These DataLoader defaults are chosen for SSD/LMDB-style reads. On Windows,
    # start with 4 workers; increase to 8 only if the machine stays stable.
    loader_kwargs = dict(
        num_workers=num_workers,
        pin_memory=bool(OmegaConf.select(conf, "training.pin_memory", default=True)),
    )
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(OmegaConf.select(conf, "training.persistent_workers", default=True))
        loader_kwargs["prefetch_factor"] = int(OmegaConf.select(conf, "training.prefetch_factor", default=3))
        multiprocessing_context = OmegaConf.select(conf, "training.dataloader_multiprocessing_context", default=None)
        if multiprocessing_context is not None and str(multiprocessing_context).strip():
            loader_kwargs["multiprocessing_context"] = str(multiprocessing_context)

    # Train is shuffled each epoch. The split membership itself is already
    # determined by `data.split_seed` inside the dataset adapter. Release
    # profiles provide a separate local generator through `training.seed`.
    train_loader_kwargs = dict(loader_kwargs)
    training_seed = OmegaConf.select(conf, "training.seed", default=None)
    if training_seed is not None:
        train_loader_kwargs["generator"] = torch.Generator().manual_seed(int(training_seed))
    # Training batches retain depth_vel only for targets and training-only anchors.
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_bg_samples,
                              **train_loader_kwargs)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_bg_samples,
                            **loader_kwargs)

    # Fail early without starting multiprocessing workers before Lightning owns
    # the training loop. This avoids ROCm crashes from pre-spawned DataLoader
    # workers during launcher-side schema validation.
    first_batch = collate_bg_samples([train_set[0]])
    validate_bg_batch(first_batch, context="train_set first sample")
    return train_loader, val_loader


def _stage(conf: Any) -> str:
    stage = str(OmegaConf.select(conf, "training.stage", default="background"))
    if stage not in VALID_STAGES:
        raise ValueError(f"training.stage must be one of {sorted(VALID_STAGES)}, got {stage!r}.")
    return stage


def validate_stage_config(conf: Any, expected_stage: str) -> Any:
    """Load-time guard for thin stage launchers."""
    stage = _stage(conf)
    if stage != str(expected_stage):
        raise ValueError(f"Expected training.stage: {expected_stage}, got {stage!r}.")
    return conf


def load_stage_config(config_path: str | Path, expected_stage: str) -> Any:
    conf = OmegaConf.load(config_path)
    return validate_stage_config(conf, expected_stage)


def run_stage_config(config_path: str | Path, expected_stage: str) -> None:
    load_stage_config(config_path, expected_stage)
    main(str(config_path), fast_run=None)


def _logging_timestamp(conf: Any) -> str:
    timestamp = OmegaConf.select(conf, "training.logging.run_timestamp", default=None)
    if timestamp is not None and str(timestamp).strip():
        return str(timestamp)
    time_format = str(OmegaConf.select(conf, "training.logging.time_format", default="%Y%m%d_%H%M%S"))
    timestamp = datetime.now().strftime(time_format)
    OmegaConf.update(conf, "training.logging.run_timestamp", timestamp, merge=True)
    return timestamp


def _log_version(conf: Any, stage: str) -> str:
    version_template = str(OmegaConf.select(conf, "training.logging.log_version", default="") or "")
    timestamp = _logging_timestamp(conf)
    if version_template.strip():
        version = _format_stage_template(version_template, stage).replace("{time}", timestamp)
    else:
        version = stage
    if "{time}" in version_template:
        return version
    add_timestamp = bool(OmegaConf.select(conf, "training.logging.add_timestamp", default=True))
    return f"{version}_{timestamp}" if add_timestamp else version


def _format_stage_template(value: Any, stage: str) -> Any:
    if value is None:
        return None
    return str(value).replace("{stage}", stage)


def _optional_training_value(conf: Any, key: str, default: Any = None) -> Any:
    value = OmegaConf.select(conf, f"training.{key}", default=default)
    if value is None:
        return None
    return value


def build_trainer(conf: Any) -> tuple[lightning.Trainer, ModelCheckpoint]:
    """Build the Lightning Trainer from YAML without stage-specific overrides."""
    stage = _stage(conf)
    log_dir = _format_stage_template(
        OmegaConf.select(conf, "training.logging.log_dir", default="logs/bg_pdr_fm/lightning"),
        stage,
    )
    log_version = _log_version(conf, stage)
    loggers = [
        CSVLogger(save_dir=log_dir, name="csv", version=log_version),
        TensorBoardLogger(save_dir=log_dir, name="tensorboard", version=log_version),
    ]
    filename_template = str(
        OmegaConf.select(
            conf,
            "training.checkpoint.filename",
            default=f"{stage}-epoch_{{epoch}}-loss{{val/loss:.4f}}",
        )
    ).replace("{stage}", stage)
    checkpoint = ModelCheckpoint(
        dirpath=_format_stage_template(OmegaConf.select(conf, "training.checkpoint.dirpath", default=None), stage),
        filename=filename_template,
        auto_insert_metric_name=False,
        save_top_k=int(OmegaConf.select(conf, "training.checkpoint.save_top_k", default=3)),
        monitor=str(OmegaConf.select(conf, "training.checkpoint.monitor", default="val/loss")),
        mode=str(OmegaConf.select(conf, "training.checkpoint.mode", default="min")),
        save_last=bool(OmegaConf.select(conf, "training.checkpoint.save_last", default=True)),
        every_n_epochs=int(OmegaConf.select(conf, "training.checkpoint.every_n_epochs", default=1)),
    )
    trainer_kwargs = {
        "max_epochs": int(OmegaConf.select(conf, "training.max_epochs", default=1)),
        "accelerator": str(OmegaConf.select(conf, "training.accelerator", default="auto")),
        "devices": OmegaConf.select(conf, "training.devices", default="auto"),
        "precision": OmegaConf.select(conf, "training.precision", default="32-true"),
        "logger": loggers,
        "callbacks": [checkpoint],
        "enable_checkpointing": True,
        "log_every_n_steps": int(OmegaConf.select(conf, "training.log_every_n_steps", default=1)),
        "limit_train_batches": OmegaConf.select(conf, "training.limit_train_batches", default=None),
        "limit_val_batches": OmegaConf.select(conf, "training.limit_val_batches", default=None),
    }
    gradient_clip_val = OmegaConf.select(conf, "training.gradient_clip_val", default=None)
    if gradient_clip_val is not None:
        trainer_kwargs["gradient_clip_val"] = float(gradient_clip_val)
    optional_trainer_keys = {
        "strategy": "strategy",
        "num_nodes": "num_nodes",
        "sync_batchnorm": "sync_batchnorm",
        "use_distributed_sampler": "use_distributed_sampler",
        "accumulate_grad_batches": "accumulate_grad_batches",
    }
    for config_key, trainer_key in optional_trainer_keys.items():
        value = _optional_training_value(conf, config_key)
        if value is not None:
            trainer_kwargs[trainer_key] = value
    return lightning.Trainer(**trainer_kwargs), checkpoint


def run_one_stage(conf) -> None:
    """Train the stage specified by `training.stage` in the config."""
    apply_training_seed(OmegaConf.select(conf, "training.seed", default=None))
    configure_checkpoint_temp_dir(OmegaConf.select(conf, "training.checkpoint_tmp_dir", default=None))
    configure_torch_runtime(OmegaConf.select(conf, "training.matmul_precision", default=None))

    train_loader, val_loader = build_loaders(conf)
    model = BGPDRFMLightning(conf)
    trainer, _ = build_trainer(conf)
    ckpt_path = OmegaConf.select(conf, "training.ckpt_path", default=None)
    trainer.fit(model, train_loader, val_loader, ckpt_path=str(ckpt_path) if ckpt_path else None)


def apply_fast_run_overrides(conf: Any) -> Any:
    """Apply temporary smoke-test settings without editing the yaml file.

    Fast-run mode is meant to answer one question: does the current full
    training flow execute end to end? It keeps the dataset/model choices from
    the config, but limits work to one epoch and one train/val batch per stage.
    """
    stages = OmegaConf.select(conf, "training.fast_run_stages", default=None)
    if stages is not None:
        conf.training.fast_run_stages = list(stages)
    conf.training.max_epochs = 1
    conf.training.limit_train_batches = 1
    conf.training.limit_val_batches = 1
    conf.training.num_workers = 0
    conf.training.persistent_workers = False
    conf.training.prefetch_factor = None
    return conf


def main(config_path: str = "bg_pdr_fm/configs/bg_pdr_fm.yaml", fast_run: bool | None = None) -> None:
    """Load a config and run one or more training stages."""
    conf = load_benchmark_config(config_path)
    if fast_run is None:
        fast_run = bool(OmegaConf.select(conf, "training.fast_run", default=False))
    if fast_run:
        conf = apply_fast_run_overrides(conf)
    stages = OmegaConf.select(conf, "training.fast_run_stages", default=None)

    # `fast_run_stages` is useful for smoke testing the whole three-stage path.
    # For real training, leave it null and set `training.stage` explicitly.
    if stages:
        for stage in list(stages):
            stage_conf = OmegaConf.create(OmegaConf.to_container(conf, resolve=False))
            stage_conf.training.stage = str(stage)
            _stage(stage_conf)
            run_one_stage(stage_conf)
    else:
        _stage(conf)
        run_one_stage(conf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or fast-run BG-PDR-FM.")
    parser.add_argument("--config", default="bg_pdr_fm/configs/bg_pdr_fm.yaml")
    parser.add_argument(
        "--fast-run",
        action="store_true",
        help="Smoke-test contrastive, background, and residual stages with one train/val batch each.",
    )
    args = parser.parse_args()
    main(args.config, fast_run=True if args.fast_run else None)
