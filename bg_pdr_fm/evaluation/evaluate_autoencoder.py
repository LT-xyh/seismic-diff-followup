"""Autoencoder checkpoint reconstruction evaluation."""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from ignite import metrics
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bg_pdr_fm.data import Marmousi, OpenFWI
from bg_pdr_fm.lightning import SeismicAutoencoderKLLightning
from bg_pdr_fm.runtime import configure_torch_runtime, configured_torch_device


LOGGER = logging.getLogger(__name__)


def _conf_get(conf: Any, path: str, default: Any) -> Any:
    return OmegaConf.select(conf, path, default=default)


def _as_path(path_like: str | Path, base: Path = REPO_ROOT) -> Path:
    path = Path(str(path_like))
    return path if path.is_absolute() else base / path


def _finite_difference_mae(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    dy_pred = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    dy_target = target[:, :, 1:, :] - target[:, :, :-1, :]
    dx_pred = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    dx_target = target[:, :, :, 1:] - target[:, :, :, :-1]
    return F.l1_loss(dy_pred, dy_target) + 0.5 * F.l1_loss(dx_pred, dx_target)


def _high_frequency_mae(pred: torch.Tensor, target: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    padding = kernel_size // 2
    pred_low = F.avg_pool2d(pred, kernel_size=kernel_size, stride=1, padding=padding)
    target_low = F.avg_pool2d(target, kernel_size=kernel_size, stride=1, padding=padding)
    return F.l1_loss(pred - pred_low, target - target_low)


def _denormalize_velocity(x: torch.Tensor, min_value: float = 1000.0, max_value: float = 6000.0) -> torch.Tensor:
    return ((x.clamp(-1.0, 1.0) + 1.0) * 0.5) * (max_value - min_value) + min_value


def _save_panel(target: torch.Tensor, recon: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    target_np = target[0, 0].detach().cpu().float().numpy()
    recon_np = recon[0, 0].detach().cpu().float().numpy()
    err_np = abs(recon_np - target_np)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for ax, image, title in zip(axes, [target_np, recon_np, err_np], ["Target", "Reconstruction", "Abs error"]):
        im = ax.imshow(image, cmap="jet", aspect="auto")
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _subset(dataset, max_samples: int | None):
    if max_samples is None or max_samples >= len(dataset):
        return dataset
    return Subset(dataset, list(range(max_samples)))


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


def _select_split_indices(total: int, split: str, fractions, seed: int) -> list[int]:
    train_idx, val_idx, test_idx = _split_indices(total, fractions, seed)
    if split == "train":
        return train_idx
    if split in {"val", "valid"}:
        return val_idx
    if split == "test":
        return test_idx
    if split == "all":
        return train_idx + val_idx + test_idx
    raise ValueError(f"evaluation.openfwi_split must be train, val, valid, test, or all, got {split!r}.")


@torch.no_grad()
def evaluate_dataset(
    model,
    loader,
    device: torch.device,
    panel_dir: Path | None = None,
    max_batches: int | None = None,
):
    psnr = metrics.PSNR(data_range=2.0, device=str(device))
    ssim = metrics.SSIM(data_range=2.0, device=str(device))
    totals = {"mae": 0.0, "mse": 0.0, "grad_mae": 0.0, "hf_mae": 0.0, "physical_mae_mps": 0.0}
    count = 0

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        target = batch["depth_vel"].to(device)
        posterior = model.vae.encode(target)
        recon = model.vae.decode(posterior.sample()).clamp(-1.0, 1.0)
        psnr.update((recon, target))
        ssim.update((recon, target))
        batch_size = target.shape[0]
        totals["mae"] += F.l1_loss(recon, target, reduction="mean").item() * batch_size
        totals["mse"] += F.mse_loss(recon, target, reduction="mean").item() * batch_size
        totals["grad_mae"] += _finite_difference_mae(recon, target).item() * batch_size
        totals["hf_mae"] += _high_frequency_mae(recon, target).item() * batch_size
        totals["physical_mae_mps"] += F.l1_loss(
            _denormalize_velocity(recon),
            _denormalize_velocity(target),
            reduction="mean",
        ).item() * batch_size
        count += batch_size
        if panel_dir is not None and batch_idx < 2:
            _save_panel(target, recon, panel_dir / f"batch_{batch_idx}.png")

    if count <= 0:
        raise ValueError("No samples were evaluated.")
    return {key: value / count for key, value in totals.items()} | {
        "psnr": float(psnr.compute()),
        "ssim": float(ssim.compute()),
        "samples": count,
    }


def _dataset_names(conf: Any) -> list[str]:
    names = _conf_get(conf, "datasets.dataset_name", None)
    if names is None:
        names = _conf_get(conf, "datasets.openfwi_datasets", [])
    return [str(name) for name in names]


def run_reconstruction_evaluation(
    conf: Any,
    checkpoint_path: str | Path | None = None,
) -> list[dict[str, float | int | str]]:
    configure_torch_runtime(_conf_get(conf, "training.matmul_precision", "medium"))
    checkpoint = checkpoint_path or _conf_get(
        conf,
        "evaluation.checkpoint",
        "checkpoints/bg_pdr_fm/seismic_autoencoder_kl.ckpt",
    )
    checkpoint = _as_path(checkpoint)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Autoencoder checkpoint not found: {checkpoint}")

    device = configured_torch_device(
        _conf_get(conf, "training.accelerator", "auto"),
        _conf_get(conf, "training.devices", "auto"),
    )
    model = SeismicAutoencoderKLLightning.load_from_checkpoint(str(checkpoint), map_location=device)
    model.eval().to(device)

    output_dir = _as_path(_conf_get(conf, "evaluation.output_dir", "logs/autoencoder/seismic_kl/evaluation"))
    output_dir.mkdir(parents=True, exist_ok=True)
    max_batches = _conf_get(conf, "evaluation.max_batches", None)
    max_batches = None if max_batches in (None, "null") else int(max_batches)
    max_samples = _conf_get(conf, "evaluation.max_samples", None)
    max_samples = None if max_samples in (None, "null") else int(max_samples)
    batch_size = int(_conf_get(conf, "evaluation.batch_size", 64))
    num_workers = int(_conf_get(conf, "evaluation.num_workers", 0))

    rows = []
    openfwi_split = str(_conf_get(conf, "evaluation.openfwi_split", "test"))
    split_fractions = _conf_get(conf, "datasets.split_fractions", (0.7, 0.2, 0.1))
    split_seed = int(_conf_get(conf, "datasets.split_seed", 42))
    for dataset_id, dataset_name in enumerate(_dataset_names(conf)):
        full_dataset = OpenFWI(
            root_dir=_conf_get(conf, "datasets.root_dir", "data/openfwi"),
            use_data=("depth_vel",),
            datasets=(dataset_name,),
            use_normalize=_conf_get(conf, "datasets.use_normalize", "-1_1"),
            storage_backend=_conf_get(conf, "datasets.storage_backend", "auto"),
            lmdb_root=_conf_get(conf, "datasets.lmdb_root", None),
            normalization_profile=_conf_get(conf, "datasets.normalization_profile", "seismic_global"),
            normalize_clamp=bool(_conf_get(conf, "datasets.normalize_clamp", True)),
            well_random=False,
        )
        split_indices = _select_split_indices(
            len(full_dataset),
            split=openfwi_split,
            fractions=split_fractions,
            seed=split_seed + dataset_id,
        )
        dataset = Subset(full_dataset, split_indices)
        loader = DataLoader(
            _subset(dataset, max_samples),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        metrics_row = evaluate_dataset(model, loader, device, output_dir / "panels" / dataset_name, max_batches)
        rows.append({"dataset": dataset_name, "split": openfwi_split, **metrics_row})

    marmousi_root = _conf_get(conf, "evaluation.marmousi_root", "data/marmousi")
    try:
        marmousi = Marmousi(
            root_dir=marmousi_root,
            split="test",
            use_data=("depth_vel",),
            use_normalize=_conf_get(conf, "datasets.use_normalize", "-1_1"),
            normalization_profile=_conf_get(conf, "datasets.normalization_profile", "seismic_global"),
            normalize_clamp=bool(_conf_get(conf, "datasets.normalize_clamp", True)),
            well_random=False,
        )
        loader = DataLoader(
            _subset(marmousi, max_samples),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        metrics_row = evaluate_dataset(model, loader, device, output_dir / "panels" / "Marmousi", max_batches)
        rows.append({"dataset": "Marmousi", "split": "test", **metrics_row})
    except FileNotFoundError as exc:
        LOGGER.info("Marmousi autoencoder evaluation skipped: %s", exc)

    if not rows:
        raise ValueError("No autoencoder evaluation rows were produced.")
    csv_path = output_dir / "reconstruction_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    LOGGER.info("Saved autoencoder reconstruction metrics to %s", csv_path)
    return rows


def main(config_path: str | Path = REPO_ROOT / "bg_pdr_fm" / "configs" / "autoencoder_eval.yaml") -> None:
    conf = OmegaConf.load(config_path)
    run_reconstruction_evaluation(conf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the standalone BG-PDR-FM seismic autoencoder.")
    parser.add_argument(
        "config",
        nargs="?",
        default=str(REPO_ROOT / "bg_pdr_fm" / "configs" / "autoencoder_eval.yaml"),
        help="Path to an autoencoder evaluation config yaml.",
    )
    parser.add_argument(
        "--config",
        dest="config_override",
        default=None,
        help="Path to an autoencoder evaluation config yaml. Overrides the positional config.",
    )
    args = parser.parse_args()
    main(args.config_override or args.config)
