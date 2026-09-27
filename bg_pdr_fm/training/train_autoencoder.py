r"""Train BG-PDR-FM's seismic autoencoder.

Run directly from the repo root or an IDE:

    D:\Conda\envs\seg\python.exe -m bg_pdr_fm.training.train_autoencoder

The script reads `bg_pdr_fm/configs/autoencoder_train.yaml` by default and
exports the best checkpoint to `checkpoints/bg_pdr_fm/seismic_autoencoder_kl.ckpt`.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import logging
from pathlib import Path
import shutil
import sys
from typing import Any

import lightning
import torch
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from omegaconf import OmegaConf
from torch.utils.data import ConcatDataset, DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bg_pdr_fm.data import OpenFWI
from bg_pdr_fm.lightning import SeismicAutoencoderKLLightning
from bg_pdr_fm.runtime import configure_checkpoint_temp_dir, configure_torch_runtime


DEFAULT_CONFIG_PATH = REPO_ROOT / "bg_pdr_fm" / "configs" / "autoencoder_train.yaml"
LOGGER = logging.getLogger(__name__)


def _conf_get(conf: Any, path: str, default: Any) -> Any:
    return OmegaConf.select(conf, path, default=default)


def _as_path(path_like: str | Path, base: Path = REPO_ROOT) -> Path:
    path = Path(str(path_like))
    return path if path.is_absolute() else base / path


def _dataset_names(conf: Any) -> list[str]:
    names = _conf_get(conf, "datasets.dataset_name", None)
    if names is None:
        names = _conf_get(conf, "datasets.openfwi_datasets", None)
    if names is None:
        names = ["FlatVelA", "FlatVelB", "CurveVelA", "CurveVelB", "CurveFaultA"]
    return [str(name) for name in names]


def _split_indices(total: int, fractions, seed: int) -> tuple[list[int], list[int], list[int]]:
    if len(fractions) != 3:
        raise ValueError("datasets.split_fractions must contain train, val, test fractions.")
    train_fraction, val_fraction, test_fraction = (float(v) for v in fractions)
    if min(train_fraction, val_fraction, test_fraction) < 0:
        raise ValueError(f"datasets.split_fractions cannot contain negative values: {fractions!r}.")
    total_fraction = train_fraction + val_fraction + test_fraction
    if total_fraction <= 0:
        raise ValueError(f"Invalid datasets.split_fractions: {fractions!r}.")
    train_fraction /= total_fraction
    val_fraction /= total_fraction

    generator = torch.Generator().manual_seed(int(seed))
    indices = torch.randperm(int(total), generator=generator).tolist()
    train_end = int(round(total * train_fraction))
    val_end = train_end + int(round(total * val_fraction))
    train_end = max(1, min(train_end, total))
    val_end = max(train_end, min(val_end, total))
    if val_end >= total and total >= 3:
        val_end = total - 1
    if train_end >= val_end and total >= 3:
        train_end = val_end - 1
    return indices[:train_end], indices[train_end:val_end], indices[val_end:]


def _openfwi_dataset(conf: Any, dataset_name: str, *, well_random: bool) -> OpenFWI:
    return OpenFWI(
        root_dir=_conf_get(conf, "datasets.root_dir", "data/openfwi"),
        use_data=("depth_vel",),
        datasets=(dataset_name,),
        use_normalize=_conf_get(conf, "datasets.use_normalize", "-1_1"),
        well_count_range=_conf_get(conf, "datasets.well_count_range", (0, 3)),
        well_seed=int(_conf_get(conf, "datasets.well_seed", 1234)),
        well_random=well_random,
        storage_backend=_conf_get(conf, "datasets.storage_backend", "auto"),
        lmdb_root=_conf_get(conf, "datasets.lmdb_root", None),
        normalization_profile=_conf_get(conf, "datasets.normalization_profile", "seismic_global"),
        normalize_clamp=bool(_conf_get(conf, "datasets.normalize_clamp", True)),
    )


def build_openfwi_splits(conf: Any):
    train_parts = []
    val_parts = []
    fractions = _conf_get(conf, "datasets.split_fractions", (0.7, 0.2, 0.1))
    split_seed = int(_conf_get(conf, "datasets.split_seed", 42))
    for dataset_id, dataset_name in enumerate(_dataset_names(conf)):
        train_dataset = _openfwi_dataset(conf, dataset_name, well_random=True)
        val_dataset = _openfwi_dataset(conf, dataset_name, well_random=False)
        train_idx, val_idx, _ = _split_indices(len(train_dataset), fractions, split_seed + dataset_id)
        train_parts.append(Subset(train_dataset, train_idx))
        val_parts.append(Subset(val_dataset, val_idx))
    return ConcatDataset(train_parts), ConcatDataset(val_parts)


def build_dataloaders(conf: Any) -> tuple[DataLoader, DataLoader]:
    train_set, val_set = build_openfwi_splits(conf)
    batch_size = int(_conf_get(conf, "training.dataloader.batch_size", 64))
    num_workers = int(_conf_get(conf, "training.dataloader.num_workers", 0))
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": bool(_conf_get(conf, "training.dataloader.pin_memory", True)),
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(_conf_get(conf, "training.dataloader.persistent_workers", True))
        loader_kwargs["prefetch_factor"] = int(_conf_get(conf, "training.dataloader.prefetch_factor", 3))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, **loader_kwargs)
    return train_loader, val_loader


def _resolved_devices(conf: Any):
    accelerator = str(_conf_get(conf, "training.accelerator", "auto"))
    devices = _conf_get(conf, "training.devices", _conf_get(conf, "training.device", "auto"))
    if accelerator == "gpu" and not torch.cuda.is_available():
        raise RuntimeError("training.accelerator is 'gpu', but torch.cuda.is_available() is false.")
    return accelerator, devices


def _log_version(conf: Any) -> str:
    version = str(_conf_get(conf, "training.logging.log_version", "") or "")
    if version.strip():
        return version
    return "seismic_kl_" + datetime.now().strftime("%m%d_%H%M")


def build_trainer(conf: Any) -> tuple[lightning.Trainer, ModelCheckpoint]:
    log_dir = str(_conf_get(conf, "training.logging.log_dir", "logs/autoencoder/seismic_kl"))
    log_version = _log_version(conf)
    tensorboard_logger = TensorBoardLogger(save_dir=log_dir, name="tensorboard", version=log_version)
    csv_logger = CSVLogger(save_dir=log_dir, name="csv", version=log_version)
    early_stop = EarlyStopping(
        monitor=str(_conf_get(conf, "training.callbacks.early_stopping.monitor", "val/loss")),
        min_delta=0.0,
        patience=int(_conf_get(conf, "training.callbacks.early_stopping.patience", 15)),
        mode=str(_conf_get(conf, "training.callbacks.early_stopping.mode", "min")),
        verbose=True,
    )
    checkpoint = ModelCheckpoint(
        filename=str(_conf_get(conf, "training.callbacks.checkpoint.filename", "epoch_{epoch}-loss{val/loss:.4f}")),
        auto_insert_metric_name=False,
        save_top_k=int(_conf_get(conf, "training.callbacks.checkpoint.save_top_k", 3)),
        monitor=str(_conf_get(conf, "training.callbacks.checkpoint.monitor", "val/loss")),
        mode=str(_conf_get(conf, "training.callbacks.checkpoint.mode", "min")),
        save_last=True,
        every_n_epochs=1,
    )
    accelerator, devices = _resolved_devices(conf)
    trainer_kwargs = {
        "precision": _conf_get(conf, "training.precision", "32-true"),
        "max_epochs": int(_conf_get(conf, "training.max_epochs", 1)),
        "min_epochs": int(_conf_get(conf, "training.min_epochs", 1)),
        "accelerator": accelerator,
        "devices": devices,
        "logger": [tensorboard_logger, csv_logger],
        "callbacks": [early_stop, checkpoint],
        "fast_dev_run": bool(_conf_get(conf, "training.fast_run", False)),
        "log_every_n_steps": int(_conf_get(conf, "training.log_every_n_steps", 1)),
        "limit_train_batches": _conf_get(conf, "training.limit_train_batches", None),
        "limit_val_batches": _conf_get(conf, "training.limit_val_batches", None),
    }
    gradient_clip_val = _conf_get(conf, "training.gradient_clip_val", None)
    if gradient_clip_val is not None:
        trainer_kwargs["gradient_clip_val"] = float(gradient_clip_val)
    return lightning.Trainer(**trainer_kwargs), checkpoint


def export_best_checkpoint(trainer: lightning.Trainer, checkpoint: ModelCheckpoint, conf: Any) -> Path | None:
    if bool(_conf_get(conf, "training.fast_run", False)):
        return None
    source = checkpoint.best_model_path or checkpoint.last_model_path
    if not source:
        return None
    source_path = Path(source)
    if not source_path.is_file():
        return None
    export_path = _as_path(_conf_get(
        conf,
        "training.export_checkpoint_path",
        "checkpoints/bg_pdr_fm/seismic_autoencoder_kl.ckpt",
    ))
    export_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, export_path)
    LOGGER.info("Exported autoencoder checkpoint: %s", export_path)
    return export_path


def main(config_path: str | Path | None = None) -> None:
    config_path = Path(config_path) if config_path is not None else DEFAULT_CONFIG_PATH
    conf = OmegaConf.load(config_path)
    configure_checkpoint_temp_dir(_conf_get(conf, "training.checkpoint_tmp_dir", None))
    configure_torch_runtime(_conf_get(conf, "training.matmul_precision", "medium"))
    model = SeismicAutoencoderKLLightning(conf)
    train_loader, val_loader = build_dataloaders(conf)
    trainer, checkpoint = build_trainer(conf)
    ckpt_path = _conf_get(conf, "training.ckpt_path", None)
    trainer.fit(model, train_loader, val_loader, ckpt_path=ckpt_path if ckpt_path else None)
    exported = export_best_checkpoint(trainer, checkpoint, conf)

    if exported is not None and bool(_conf_get(conf, "evaluation.run_after_train", False)):
        from bg_pdr_fm.evaluation.evaluate_autoencoder import run_reconstruction_evaluation

        run_reconstruction_evaluation(conf, checkpoint_path=exported)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the standalone BG-PDR-FM seismic autoencoder.")
    parser.add_argument(
        "config",
        nargs="?",
        default=str(DEFAULT_CONFIG_PATH),
        help="Path to an autoencoder config yaml.",
    )
    parser.add_argument(
        "--config",
        dest="config_override",
        default=None,
        help="Path to an autoencoder config yaml. Overrides the positional config.",
    )
    args = parser.parse_args()
    main(args.config_override or args.config)
