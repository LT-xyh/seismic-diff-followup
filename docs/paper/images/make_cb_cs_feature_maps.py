"""Export real C_B and C_S feature-map visualizations for Fig. 2.

The maps are extracted from a trained PD-BG-RFM checkpoint through the live
`PhysicsDecoupledEncoder`.  They are intended as paper drawing assets, not as a
training/evaluation entry point.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bg_pdr_fm.data import batch_to_device, collate_bg_samples
from bg_pdr_fm.models import PhysicsDecoupledEncoder
from bg_pdr_fm.runtime import configure_torch_runtime
from bg_pdr_fm.training.train_bg_pdr_fm import build_dataset


DEFAULT_CONFIG = REPO_ROOT / "bg_pdr_fm/configs/openfwi_lmdb_joint_full_contrastive_bgfm_pixelfm_predbg_e100.yaml"
DEFAULT_CHECKPOINT = (
    REPO_ROOT
    / "logs/bg_pdr_fm/joint_full_contrastive_bgfm_pixelfm_predbg_e100/stage_checkpoints/joint_full_last.ckpt"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "docs/paper/images/figure2_assets/feature_maps"
DEFAULT_DATASET_NAME = "FlatFaultB"
DEFAULT_SAMPLE_INDEX = 27006

GREEN = "#16A34A"
BLUE = "#2563EB"
TEXT = "#111827"
MUTED = "#64748B"


def _as_repo_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _prepare_conf(config_path: Path) -> Any:
    conf = OmegaConf.load(config_path)
    OmegaConf.update(conf, "training.batch_size", 1, merge=True)
    OmegaConf.update(conf, "training.num_workers", 0, merge=True)
    OmegaConf.update(conf, "training.persistent_workers", False, merge=True)
    OmegaConf.update(conf, "training.prefetch_factor", 2, merge=True)
    return conf


def _build_single_sample_batch(conf: Any, dataset_name: str, sample_index: int):
    original_datasets = OmegaConf.select(conf, "data.openfwi_datasets", default=None)
    OmegaConf.update(conf, "data.openfwi_datasets", [dataset_name], merge=True)
    dataset = build_dataset(conf, "all")
    if original_datasets is not None:
        OmegaConf.update(conf, "data.openfwi_datasets", original_datasets, merge=True)

    match_idx = None
    for idx, record in enumerate(getattr(dataset, "records", [])):
        if record.get("dataset_name") == dataset_name and int(record.get("sample_index", -1)) == sample_index:
            match_idx = idx
            break
    if match_idx is None:
        raise ValueError(f"Could not find {dataset_name} sample_index={sample_index} in OpenFWI dataset records.")

    item = dataset[match_idx]
    batch = collate_bg_samples([item])
    metadata = {
        "dataset_name": dataset_name,
        "sample_index": sample_index,
        "dataset_record_index": match_idx,
        "split": "all",
    }
    close = getattr(dataset, "close", None)
    if callable(close):
        close()
    return batch, metadata


def _load_encoder(conf: Any, checkpoint_path: Path, device: torch.device) -> tuple[PhysicsDecoupledEncoder, dict[str, Any]]:
    encoder = PhysicsDecoupledEncoder(
        hidden_channels=int(OmegaConf.select(conf, "model.hidden_channels", default=16)),
        base_channels=int(OmegaConf.select(conf, "model.base_channels", default=32)),
        feature_channels=int(OmegaConf.select(conf, "model.feature_channels", default=16)),
    )
    state = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(state, dict) or "state_dict" not in state:
        raise KeyError(f"Checkpoint {checkpoint_path} does not contain a `state_dict`.")
    source = state["state_dict"]
    encoder_state = {
        key.removeprefix("encoder."): value
        for key, value in source.items()
        if key.startswith("encoder.")
    }
    if not encoder_state:
        raise KeyError(f"Checkpoint {checkpoint_path} does not contain encoder.* weights.")
    current = encoder.state_dict()
    compatible = {
        key: value
        for key, value in encoder_state.items()
        if key in current and tuple(value.shape) == tuple(current[key].shape)
    }
    if not compatible:
        raise KeyError(f"Checkpoint {checkpoint_path} has no shape-compatible encoder weights.")
    incompatible = encoder.load_state_dict(compatible, strict=False)
    encoder.eval().to(device)
    loaded = {
        "mode": "encoder_only",
        "path": _as_repo_path(checkpoint_path),
        "loaded_keys": len(compatible),
        "skipped_shape_or_missing_keys": len(encoder_state) - len(compatible),
        "missing_keys": len(incompatible.missing_keys),
        "unexpected_keys": len(incompatible.unexpected_keys),
    }
    return encoder, loaded


def _low_high_ratios(x: torch.Tensor, kernel_size: int = 5) -> tuple[np.ndarray, np.ndarray]:
    pad = kernel_size // 2
    low = F.avg_pool2d(F.pad(x, (pad, pad, pad, pad), mode="replicate"), kernel_size=kernel_size, stride=1)
    high = x - low
    low_e = low.square().flatten(2).mean(dim=2)
    high_e = high.square().flatten(2).mean(dim=2)
    total = (low_e + high_e).clamp_min(1e-8)
    return (low_e / total)[0].detach().cpu().numpy(), (high_e / total)[0].detach().cpu().numpy()


def _feature_energy(x: torch.Tensor) -> np.ndarray:
    energy = x[0].detach().float().square().mean(dim=0).sqrt()
    arr = energy.cpu().numpy()
    lo, hi = np.percentile(arr, [2.0, 98.0])
    return np.clip((arr - lo) / max(hi - lo, 1e-8), 0.0, 1.0)


def _signed_channel(x: torch.Tensor, channel: int) -> np.ndarray:
    arr = x[0, channel].detach().float().cpu().numpy()
    vmax = float(np.percentile(np.abs(arr), 98.0))
    if vmax <= 1e-8:
        return arr
    return np.clip(arr / vmax, -1.0, 1.0)


def _save_image(path: Path, image: np.ndarray, cmap: str, *, vmin: float | None = None, vmax: float | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(3.0, 3.0), dpi=300)
    ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _save_channel_grid(path: Path, x: torch.Tensor, title: str, color: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    channels = min(int(x.shape[1]), 16)
    fig, axes = plt.subplots(4, 4, figsize=(7.2, 7.4), dpi=180)
    axes = np.asarray(axes).reshape(-1)
    for idx, ax in enumerate(axes):
        if idx < channels:
            image = _signed_channel(x, idx)
            ax.imshow(image, cmap="RdBu_r", vmin=-1.0, vmax=1.0, interpolation="nearest")
            ax.set_title(f"ch {idx}", fontsize=8)
        ax.axis("off")
    fig.suptitle(title, fontsize=13, color=color, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _imshow_plain(ax: plt.Axes, image: np.ndarray, title: str, cmap: str, color: str, vmin=None, vmax=None) -> None:
    im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=10, color=color, weight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color(color)
        spine.set_linewidth(1.2)
    return im


def _save_paper_panel(
    path: Path,
    cb_energy: np.ndarray,
    cs_energy: np.ndarray,
    cb_channel: np.ndarray,
    cs_channel: np.ndarray,
    cb_channel_idx: int,
    cs_channel_idx: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(6.3, 5.9), dpi=220)
    _imshow_plain(axes[0, 0], cb_energy, r"$C_B$ aggregate", "YlGn", GREEN, 0.0, 1.0)
    _imshow_plain(axes[0, 1], cb_channel, rf"$C_B$ low-freq ch. {cb_channel_idx}", "RdBu_r", GREEN, -1.0, 1.0)
    _imshow_plain(axes[1, 0], cs_energy, r"$C_S$ aggregate", "Blues", BLUE, 0.0, 1.0)
    _imshow_plain(axes[1, 1], cs_channel, rf"$C_S$ high-freq ch. {cs_channel_idx}", "RdBu_r", BLUE, -1.0, 1.0)
    fig.text(
        0.5,
        0.035,
        "Feature maps are extracted from the trained encoder; no target-derived anchor is used at inference.",
        ha="center",
        fontsize=8,
        color=MUTED,
    )
    fig.tight_layout(rect=(0.02, 0.06, 0.98, 0.98))
    fig.savefig(path, dpi=220)
    plt.close(fig)


def run(
    config_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    dataset_name: str,
    sample_index: int,
    device_text: str,
) -> dict[str, Any]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")

    conf = _prepare_conf(config_path)
    configure_torch_runtime(OmegaConf.select(conf, "training.matmul_precision", default=None))
    if device_text == "auto":
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_text)

    batch, sample_meta = _build_single_sample_batch(conf, dataset_name, sample_index)
    encoder, loaded = _load_encoder(conf, checkpoint_path, device)
    batch = batch_to_device(batch, device)

    with torch.no_grad():
        features = encoder(batch, return_all_heads=False)
        cb = features.numerical.detach()
        cs = features.structural.detach()

    cb_low, cb_high = _low_high_ratios(cb, kernel_size=int(OmegaConf.select(conf, "model.lowpass_kernel", default=5)))
    cs_low, cs_high = _low_high_ratios(cs, kernel_size=int(OmegaConf.select(conf, "model.lowpass_kernel", default=5)))
    cb_channel_idx = int(np.argmax(cb_low))
    cs_channel_idx = int(np.argmax(cs_high))

    cb_energy = _feature_energy(cb)
    cs_energy = _feature_energy(cs)
    cb_channel = _signed_channel(cb, cb_channel_idx)
    cs_channel = _signed_channel(cs, cs_channel_idx)

    output_dir.mkdir(parents=True, exist_ok=True)
    _save_image(output_dir / "cb_aggregate.png", cb_energy, "YlGn", vmin=0.0, vmax=1.0)
    _save_image(output_dir / "cs_aggregate.png", cs_energy, "Blues", vmin=0.0, vmax=1.0)
    _save_image(output_dir / "cb_lowfreq_channel.png", cb_channel, "RdBu_r", vmin=-1.0, vmax=1.0)
    _save_image(output_dir / "cs_highfreq_channel.png", cs_channel, "RdBu_r", vmin=-1.0, vmax=1.0)
    _save_channel_grid(output_dir / "cb_channels_grid.png", cb, r"$C_B$ channels", GREEN)
    _save_channel_grid(output_dir / "cs_channels_grid.png", cs, r"$C_S$ channels", BLUE)
    _save_paper_panel(
        output_dir / "cb_cs_feature_panel.png",
        cb_energy,
        cs_energy,
        cb_channel,
        cs_channel,
        cb_channel_idx,
        cs_channel_idx,
    )

    metadata = {
        "config": _as_repo_path(config_path),
        "checkpoint": _as_repo_path(checkpoint_path),
        "loaded_checkpoint": loaded,
        "sample": sample_meta,
        "feature_shapes": {
            "C_B": list(cb.shape),
            "C_S": list(cs.shape),
        },
        "device": str(device),
        "selected_channels": {
            "C_B_low_frequency_channel": cb_channel_idx,
            "C_S_high_frequency_channel": cs_channel_idx,
        },
        "channel_ratios": {
            "C_B_low_ratio": cb_low.tolist(),
            "C_B_high_ratio": cb_high.tolist(),
            "C_S_low_ratio": cs_low.tolist(),
            "C_S_high_ratio": cs_high.tolist(),
        },
        "outputs": sorted(path.name for path in output_dir.glob("*.png")),
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Export real C_B/C_S feature-map visualizations.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--sample-index", type=int, default=DEFAULT_SAMPLE_INDEX)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    metadata = run(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        sample_index=args.sample_index,
        device_text=args.device,
    )
    print(json.dumps({"output_dir": str(args.output_dir), "selected_channels": metadata["selected_channels"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
