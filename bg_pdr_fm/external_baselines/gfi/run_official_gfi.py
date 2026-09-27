"""Run the official GFI inverse pipeline and capture a local AAAI27 manifest.

This module is intentionally external-only:

- it does not train BG-PDR-FM;
- it does not patch the official GFI source tree;
- it only orchestrates the external GFI repository, exports predictions, and
  evaluates them with the local metric bridge.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from bg_pdr_fm.external_baselines.gfi.adapter import read_gfi_annotation
from bg_pdr_fm.external_baselines.gfi.evaluate_gfi_predictions import evaluate_arrays
from bg_pdr_fm.external_baselines.gfi.prepare_gfi_annotations import write_annotation


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_GFI_ROOT = Path("/public/home/xuyinghao/workspace/external_repos/gfi")
DEFAULT_GFI_PYTHON = Path("/public/home/xuyinghao/workspace/external_repos/gfi_venv/bin/python")
DEFAULT_DATA = Path("/public/home/xuyinghao/workspace/datasets/Datasets/data1.npy")
DEFAULT_LABEL = Path("/public/home/xuyinghao/workspace/datasets/Datasets/model1.npy")
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "logs" / "bg_pdr_fm" / "aaai27" / "external_gfi_native"


@dataclass(frozen=True)
class GFIPaths:
    gfi_root: Path
    gfi_python: Path
    annotation: Path
    train_log: Path
    checkpoint: Path
    predictions: Path
    targets: Path
    metrics_dir: Path
    manifest: Path


class _TransformArgs:
    def __init__(self, velocity_transform: str, amplitude_transform: str) -> None:
        self.velocity_transform = velocity_transform
        self.amplitude_transform = amplitude_transform
        self.k = 1


def _ensure_gfi_root(gfi_root: Path) -> None:
    rainbow_src = gfi_root / "utils" / "rainbow256.npy"
    rainbow_dst = gfi_root / "rainbow256.npy"
    if rainbow_dst.exists():
        return
    if not rainbow_src.is_file():
        raise FileNotFoundError(f"missing rainbow palette in official checkout: {rainbow_src}")
    rainbow_dst.symlink_to(Path("utils") / "rainbow256.npy")


def _run_command(command: list[str], *, cwd: Path, env: dict[str, str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("$ " + " ".join(command) + "\n")
        log_file.flush()
        completed = subprocess.run(
            command,
            cwd=str(cwd),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            check=True,
            text=True,
        )
        del completed


def _build_official_model(
    model_name: str,
    *,
    cfg_path: Path,
    latent_dim: int,
    unet_depth: int,
    unet_repeat_blocks: int,
    skip: bool,
):
    from networks import inverse_network

    if model_name == "InversionNet":
        params: dict[str, object] = {}
    elif model_name in {"UNetInverseModel", "UNetInverseModel_Legacy"}:
        params = {
            "cfg_path": str(cfg_path),
            "latent_dim": int(latent_dim),
            "unet_depth": int(unet_depth),
            "unet_repeat_blocks": int(unet_repeat_blocks),
            "skip": bool(skip),
        }
    elif model_name in {"IUnetInverseModel", "IUnetInverseModel_Legacy"}:
        params = {
            "cfg_path": str(cfg_path),
            "latent_dim": int(latent_dim),
        }
    else:
        raise ValueError(f"unsupported official GFI model: {model_name!r}")
    return inverse_network.model_dict[model_name](**params)


def _build_official_train_command(
    *,
    gfi_python: Path,
    gfi_root: Path,
    annotation: Path,
    output_root: Path,
    run_name: str,
    suffix: str,
    dataset_name: str,
    model_name: str,
    batch_size: int,
    epoch_block: int,
    num_blocks: int,
    num_workers: int,
    print_freq: int,
    plot_interval: int,
    num_images: int,
    lr_scheduler: str,
    cfg_path: Path,
    latent_dim: int,
    unet_depth: int,
    unet_repeat_blocks: int,
    skip: bool,
    lr: float,
    optimizer: str,
    rm_direct_arrival: int,
    sample_temporal: int,
    velocity_transform: str,
    amplitude_transform: str,
    mask_factor: float,
) -> tuple[list[str], Path]:
    train_output_root = output_root / run_name
    if suffix:
        train_output_root = train_output_root / suffix
    log_path = train_output_root / "train.log"
    command = [
        str(gfi_python),
        "-u",
        "src/train_inverse.py",
        "-d",
        "cuda",
        "-ds",
        dataset_name,
        "-ap",
        "",
        "-t",
        str(annotation),
        "-v",
        str(annotation),
        "-m",
        model_name,
        "-b",
        str(batch_size),
        "-j",
        str(num_workers),
        "-eb",
        str(epoch_block),
        "-nb",
        str(num_blocks),
        "--print-freq",
        str(print_freq),
        "--plot_interval",
        str(plot_interval),
        "--num_images",
        str(num_images),
        "--lr_scheduler",
        lr_scheduler,
        "-o",
        str(output_root),
        "-l",
        str(output_root),
        "-n",
        run_name,
        "-s",
        suffix,
        "--cfg-path",
        str(cfg_path),
        "--latent-dim",
        str(latent_dim),
        "--unet_depth",
        str(unet_depth),
        "--unet_repeat_blocks",
        str(unet_repeat_blocks),
        "--skip",
        "1" if skip else "0",
        "--lr",
        str(lr),
        "--optimizer",
        optimizer,
        "--rm_direct_arrival",
        str(rm_direct_arrival),
        "--sample-temporal",
        str(sample_temporal),
        "--velocity_transform",
        velocity_transform,
        "--amplitude_transform",
        amplitude_transform,
        "--mask_factor",
        str(mask_factor),
        "--lambda_vgg_vel",
        "0",
        "--lambda_recons",
        "0",
    ]
    return command, log_path


def _export_predictions(
    *,
    gfi_root: Path,
    checkpoint_path: Path,
    annotation: Path,
    dataset_name: str,
    model_name: str,
    output_dir: Path,
    cfg_path: Path,
    latent_dim: int,
    unet_depth: int,
    unet_repeat_blocks: int,
    skip: bool,
    batch_size: int,
    sample_temporal: int,
    velocity_transform: str,
    amplitude_transform: str,
    device: torch.device,
) -> tuple[Path, Path, int]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    if str(gfi_root) not in sys.path:
        sys.path.insert(0, str(gfi_root))
    src_path = gfi_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    from dataset import FWIDataset
    import utils.utilities as utils
    from networks import inverse_network

    with (gfi_root / "dataset_config.json").open("r", encoding="utf-8") as handle:
        ctx = json.load(handle)[dataset_name]

    transform_args = _TransformArgs(velocity_transform, amplitude_transform)
    transform_data, transform_label = utils.get_transforms(transform_args, ctx)
    dataset = FWIDataset(
        str(annotation),
        preload=True,
        sample_ratio=sample_temporal,
        file_size=ctx["file_size"],
        transform_data=transform_data,
        transform_label=transform_label,
        mask_factor=0.0,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=default_collate,
    )

    model = _build_official_model(
        model_name,
        cfg_path=cfg_path,
        latent_dim=latent_dim,
        unet_depth=unet_depth,
        unet_repeat_blocks=unet_repeat_blocks,
        skip=skip,
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(inverse_network.replace_legacy(checkpoint["model"]))
    model.eval()

    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for _, amp, vel in loader:
            amp = amp.to(device, non_blocking=True)
            pred = model(amp).detach().cpu().numpy().astype("float32")
            predictions.append(pred)
            targets.append(vel.numpy().astype("float32"))

    pred_array = np.concatenate(predictions, axis=0)
    target_array = np.concatenate(targets, axis=0)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "gfi_predictions.npy"
    target_path = output_dir / "gfi_targets.npy"
    np.save(prediction_path, pred_array)
    np.save(target_path, target_array)
    return prediction_path, target_path, int(pred_array.shape[0])


def _export_predictions_subprocess(
    *,
    gfi_python: Path,
    gfi_root: Path,
    shim_path: Path | None,
    checkpoint_path: Path,
    annotation: Path,
    dataset_name: str,
    model_name: str,
    output_dir: Path,
    cfg_path: Path,
    latent_dim: int,
    unet_depth: int,
    unet_repeat_blocks: int,
    skip: bool,
    batch_size: int,
    sample_temporal: int,
    velocity_transform: str,
    amplitude_transform: str,
    gpu: str,
    log_path: Path,
) -> tuple[Path, Path, int]:
    command = [
        str(gfi_python),
        "-m",
        "bg_pdr_fm.external_baselines.gfi.run_official_gfi",
        "export",
        "--gfi-root",
        str(gfi_root),
        "--checkpoint",
        str(checkpoint_path),
        "--annotation",
        str(annotation),
        "--dataset-name",
        dataset_name,
        "--model",
        model_name,
        "--output-dir",
        str(output_dir),
        "--cfg-path",
        str(cfg_path),
        "--latent-dim",
        str(latent_dim),
        "--unet-depth",
        str(unet_depth),
        "--unet-repeat-blocks",
        str(unet_repeat_blocks),
        "--batch-size",
        str(batch_size),
        "--sample-temporal",
        str(sample_temporal),
        "--velocity-transform",
        velocity_transform,
        "--amplitude-transform",
        amplitude_transform,
    ]
    command.append("--skip" if skip else "--no-skip")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    python_paths = [str(REPO_ROOT), str(gfi_root), str(gfi_root / "src")]
    if shim_path is not None:
        python_paths.insert(0, str(shim_path))
    env["PYTHONPATH"] = (
        ":".join(python_paths)
        + (f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else "")
    )
    env.setdefault("PYTHONUNBUFFERED", "1")
    _run_command(command, cwd=gfi_root, env=env, log_path=log_path)
    prediction_path = output_dir / "gfi_predictions.npy"
    target_path = output_dir / "gfi_targets.npy"
    if not prediction_path.is_file() or not target_path.is_file():
        raise FileNotFoundError(f"prediction export failed under {output_dir}")
    return prediction_path, target_path, int(np.load(prediction_path, mmap_mode="r").shape[0])


def run_official_gfi(
    *,
    data: Path,
    label: Path,
    annotation: Path,
    output_root: Path,
    run_name: str,
    suffix: str,
    gfi_root: Path,
    gfi_python: Path,
    shim_path: Path | None,
    dataset_name: str,
    model_name: str,
    batch_size: int,
    epoch_block: int,
    num_blocks: int,
    num_workers: int,
    print_freq: int,
    plot_interval: int,
    num_images: int,
    lr_scheduler: str,
    cfg_path: Path,
    latent_dim: int,
    unet_depth: int,
    unet_repeat_blocks: int,
    skip: bool,
    lr: float,
    optimizer: str,
    rm_direct_arrival: int,
    sample_temporal: int,
    velocity_transform: str,
    amplitude_transform: str,
    mask_factor: float,
    gpu: str,
    skip_train: bool = False,
) -> dict[str, object]:
    _ensure_gfi_root(gfi_root)
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / run_name
    if suffix:
        run_dir = run_dir / suffix
    run_dir.mkdir(parents=True, exist_ok=True)

    if not annotation.is_file():
        write_annotation(data, label, annotation)
    else:
        read_gfi_annotation(annotation)

    command, train_log = _build_official_train_command(
        gfi_python=gfi_python,
        gfi_root=gfi_root,
        annotation=annotation,
        output_root=output_root,
        run_name=run_name,
        suffix=suffix,
        dataset_name=dataset_name,
        model_name=model_name,
        batch_size=batch_size,
        epoch_block=epoch_block,
        num_blocks=num_blocks,
        num_workers=num_workers,
        print_freq=print_freq,
        plot_interval=plot_interval,
        num_images=num_images,
        lr_scheduler=lr_scheduler,
        cfg_path=cfg_path,
        latent_dim=latent_dim,
        unet_depth=unet_depth,
        unet_repeat_blocks=unet_repeat_blocks,
        skip=skip,
        lr=lr,
        optimizer=optimizer,
        rm_direct_arrival=rm_direct_arrival,
        sample_temporal=sample_temporal,
        velocity_transform=velocity_transform,
        amplitude_transform=amplitude_transform,
        mask_factor=mask_factor,
    )

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    python_paths = [str(REPO_ROOT), str(gfi_root), str(gfi_root / "src")]
    if shim_path is not None:
        python_paths.insert(0, str(shim_path))
    env["PYTHONPATH"] = (
        ":".join(python_paths)
        + (f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else "")
    )
    env.setdefault("PYTHONUNBUFFERED", "1")

    if not skip_train:
        _run_command(command, cwd=gfi_root, env=env, log_path=train_log)

    checkpoint_candidates = [
        run_dir / "checkpoint.pth",
        run_dir / "latest_checkpoint.pth",
        run_dir / f"model_{epoch_block * num_blocks}.pth",
    ]
    checkpoint_path = next((path for path in checkpoint_candidates if path.is_file()), None)
    if checkpoint_path is None:
        raise FileNotFoundError(f"no checkpoint found under {run_dir}")

    prediction_path, target_path, num_samples = _export_predictions_subprocess(
        gfi_python=gfi_python,
        gfi_root=gfi_root,
        shim_path=shim_path,
        checkpoint_path=checkpoint_path,
        annotation=annotation,
        dataset_name=dataset_name,
        model_name=model_name,
        output_dir=run_dir,
        cfg_path=cfg_path,
        latent_dim=latent_dim,
        unet_depth=unet_depth,
        unet_repeat_blocks=unet_repeat_blocks,
        skip=skip,
        batch_size=batch_size,
        sample_temporal=sample_temporal,
        velocity_transform=velocity_transform,
        amplitude_transform=amplitude_transform,
        gpu=gpu,
        log_path=run_dir / "export.log",
    )

    metrics_dir = run_dir / "gfi_metrics"
    metrics_summary = evaluate_arrays(prediction_path, target_path, metrics_dir)
    manifest = {
        "official_repo_root": str(gfi_root),
        "official_python": str(gfi_python),
        "official_commit": (gfi_root / ".source_commit").read_text(encoding="utf-8").strip()
        if (gfi_root / ".source_commit").is_file()
        else "",
        "gpu": gpu,
        "data": str(data),
        "label": str(label),
        "annotation": str(annotation),
        "dataset_name": dataset_name,
        "model_name": model_name,
        "train_command": command,
        "train_log": str(train_log),
        "checkpoint": str(checkpoint_path),
        "predictions": str(prediction_path),
        "targets": str(target_path),
        "metrics_dir": str(metrics_dir),
        "num_samples": num_samples,
        "metrics": metrics_summary["means"],
    }
    manifest_path = run_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def _export_main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description="Export predictions from an official GFI checkpoint.")
    parser.add_argument("--gfi-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--annotation", type=Path, required=True)
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cfg-path", type=Path, required=True)
    parser.add_argument("--latent-dim", type=int, default=70)
    parser.add_argument("--unet-depth", type=int, default=2)
    parser.add_argument("--unet-repeat-blocks", type=int, default=2)
    parser.add_argument("--skip", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--sample-temporal", type=int, default=1)
    parser.add_argument("--velocity-transform", type=str, default="normalize")
    parser.add_argument("--amplitude-transform", type=str, default="normalize")
    args = parser.parse_args(argv)
    if str(args.gfi_root) not in sys.path:
        sys.path.insert(0, str(args.gfi_root))
    src_path = args.gfi_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prediction_path, target_path, num_samples = _export_predictions(
        gfi_root=args.gfi_root,
        checkpoint_path=args.checkpoint,
        annotation=args.annotation,
        dataset_name=args.dataset_name,
        model_name=args.model,
        output_dir=args.output_dir,
        cfg_path=args.cfg_path,
        latent_dim=args.latent_dim,
        unet_depth=args.unet_depth,
        unet_repeat_blocks=args.unet_repeat_blocks,
        skip=bool(args.skip),
        batch_size=args.batch_size,
        sample_temporal=args.sample_temporal,
        velocity_transform=args.velocity_transform,
        amplitude_transform=args.amplitude_transform,
        device=device,
    )
    print(json.dumps({
        "predictions": str(prediction_path),
        "targets": str(target_path),
        "num_samples": num_samples,
    }, indent=2))


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "export":
        _export_main(sys.argv[2:])
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--label", type=Path, default=DEFAULT_LABEL)
    parser.add_argument("--annotation", type=Path, default=DEFAULT_OUTPUT_ROOT / "inputs" / "gfi_annotation.txt")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", type=str, default="smoke_probe")
    parser.add_argument("--suffix", type=str, default="run")
    parser.add_argument("--gfi-root", type=Path, default=DEFAULT_GFI_ROOT)
    parser.add_argument("--gfi-python", type=Path, default=DEFAULT_GFI_PYTHON)
    parser.add_argument("--shim-path", type=Path, default=Path("/tmp/gfi_smoke/pyshim"))
    parser.add_argument("--dataset-name", type=str, default="flatvel-a")
    parser.add_argument("--model", type=str, default="InversionNet")
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--epoch-block", type=int, default=1)
    parser.add_argument("--num-blocks", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--print-freq", type=int, default=1)
    parser.add_argument("--plot-interval", type=int, default=999)
    parser.add_argument("--num-images", type=int, default=2)
    parser.add_argument("--lr-scheduler", type=str, default="None")
    parser.add_argument("--cfg-path", type=Path, default=None)
    parser.add_argument("--latent-dim", type=int, default=70)
    parser.add_argument("--unet-depth", type=int, default=2)
    parser.add_argument("--unet-repeat-blocks", type=int, default=2)
    parser.add_argument("--skip", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--optimizer", type=str, default="Adam")
    parser.add_argument("--rm-direct-arrival", type=int, default=1)
    parser.add_argument("--sample-temporal", type=int, default=1)
    parser.add_argument("--velocity-transform", type=str, default="normalize")
    parser.add_argument("--amplitude-transform", type=str, default="normalize")
    parser.add_argument("--mask-factor", type=float, default=0.0)
    parser.add_argument("--gpu", type=str, default="7")
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    cfg_path = args.cfg_path or (args.gfi_root / "configs")
    manifest = run_official_gfi(
        data=args.data,
        label=args.label,
        annotation=args.annotation,
        output_root=args.output_root,
        run_name=args.run_name,
        suffix=args.suffix,
        gfi_root=args.gfi_root,
        gfi_python=args.gfi_python,
        shim_path=args.shim_path if args.shim_path else None,
        dataset_name=args.dataset_name,
        model_name=args.model,
        batch_size=args.batch_size,
        epoch_block=args.epoch_block,
        num_blocks=args.num_blocks,
        num_workers=args.num_workers,
        print_freq=args.print_freq,
        plot_interval=args.plot_interval,
        num_images=args.num_images,
        lr_scheduler=args.lr_scheduler,
        cfg_path=cfg_path,
        latent_dim=args.latent_dim,
        unet_depth=args.unet_depth,
        unet_repeat_blocks=args.unet_repeat_blocks,
        skip=bool(args.skip),
        lr=args.lr,
        optimizer=args.optimizer,
        rm_direct_arrival=args.rm_direct_arrival,
        sample_temporal=args.sample_temporal,
        velocity_transform=args.velocity_transform,
        amplitude_transform=args.amplitude_transform,
        mask_factor=args.mask_factor,
        gpu=args.gpu,
        skip_train=bool(args.skip_train),
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
