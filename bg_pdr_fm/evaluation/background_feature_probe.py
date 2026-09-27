"""Small probes for testing whether numerical features contain oracle background information."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from bg_pdr_fm.data import batch_to_device, collate_bg_samples
from bg_pdr_fm.diagnostics import append_metrics_csv
from bg_pdr_fm.evaluation.benchmark_metrics import compute_velocity_metrics
from bg_pdr_fm.lightning import BGPDRFMLightning
from bg_pdr_fm.runtime import configure_torch_runtime, configured_torch_device
from bg_pdr_fm.training.train_bg_pdr_fm import build_dataset


class BackgroundProbe(nn.Module):
    """A tiny full-resolution regressor from C_N to P_L(V)."""

    def __init__(self, in_channels: int, kind: str = "linear_1x1", hidden_channels: int = 32) -> None:
        super().__init__()
        kind = str(kind).lower()
        self.kind = kind
        if kind == "linear_1x1":
            self.net = nn.Conv2d(in_channels, 1, kernel_size=1)
        elif kind == "mlp_1x1":
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, hidden_channels, kernel_size=1),
                nn.SiLU(),
                nn.Conv2d(hidden_channels, 1, kernel_size=1),
            )
        else:
            raise ValueError(f"Unknown background probe kind {kind!r}; expected 'linear_1x1' or 'mlp_1x1'.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(x))


def _conf_get(conf: Any, path: str, default: Any) -> Any:
    return OmegaConf.select(conf, path, default=default)


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else float("nan")


def _dataset_lookup(loader: DataLoader, conf: Any) -> dict[int, str]:
    names = getattr(loader.dataset, "dataset_names", None)
    if names is not None:
        return {idx: str(name) for idx, name in enumerate(tuple(names))}
    return {0: str(_conf_get(conf, "data.name", "synthetic"))}


def _metadata_int(batch: Any, key: str, item_idx: int, default: int) -> int:
    value = batch.metadata.get(key)
    if value is None:
        return int(default)
    return int(round(float(value[item_idx].detach().view(-1)[0].cpu())))


def _build_loader(conf: Any, split: str, shuffle: bool) -> DataLoader:
    dataset = build_dataset(conf, split)
    batch_size = int(_conf_get(conf, "training.batch_size", 64))
    num_workers = int(_conf_get(conf, "training.num_workers", 4))
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "collate_fn": collate_bg_samples,
        "num_workers": num_workers,
        "pin_memory": bool(_conf_get(conf, "training.pin_memory", True)),
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = bool(_conf_get(conf, "training.persistent_workers", True))
        kwargs["prefetch_factor"] = int(_conf_get(conf, "training.prefetch_factor", 3))
    return DataLoader(dataset, **kwargs)


def _load_contrastive_encoder(model: BGPDRFMLightning, checkpoint: str | None) -> None:
    if not checkpoint:
        return
    path = Path(checkpoint)
    if not path.is_file():
        raise FileNotFoundError(f"Contrastive checkpoint not found: {path}")
    state = torch.load(path, map_location="cpu")
    state_dict = state.get("state_dict")
    if not isinstance(state_dict, dict):
        raise KeyError(f"Checkpoint {path} does not contain a state_dict.")
    filtered = {k: v for k, v in state_dict.items() if k.startswith("encoder.")}
    current = model.state_dict()
    compatible = {k: v for k, v in filtered.items() if k in current and tuple(current[k].shape) == tuple(v.shape)}
    model.load_state_dict(compatible, strict=False)


def _evaluate_probe(
    *,
    conf: Any,
    model: BGPDRFMLightning,
    probe: BackgroundProbe,
    loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    name: str,
    max_batches: int | None,
) -> dict[str, Any]:
    metrics_path = output_dir / name / "metrics.csv"
    if metrics_path.exists():
        metrics_path.unlink()
    values: dict[str, list[float]] = {key: [] for key in ("bg_mae", "bg_l1", "bg_l2", "mae_l", "ssim")}
    by_dataset: dict[str, dict[str, list[float]]] = {}
    sample_index = 0
    dataset_lookup = _dataset_lookup(loader, conf)
    probe.eval()
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            batch = batch_to_device(BGPDRFMLightning._as_batch(batch), device)
            target = model.filter.lowpass(batch.depth_vel)
            features = model.encoder(batch)
            pred = probe(features.numerical)
            err = pred - target
            vel_metrics = compute_velocity_metrics(pred, target, model.filter)
            bg_l1 = err.abs().flatten(1).mean(dim=1)
            bg_l2 = err.square().flatten(1).mean(dim=1)
            for item_idx in range(pred.shape[0]):
                dataset_id = _metadata_int(batch, "dataset_id", item_idx, 0)
                dataset = dataset_lookup.get(dataset_id, f"dataset_{dataset_id}")
                row = {
                    "sample_index": sample_index,
                    "batch_index": batch_idx,
                    "item_index": item_idx,
                    "dataset": dataset,
                    "bg_mae": float(bg_l1[item_idx].detach().cpu()),
                    "bg_l1": float(bg_l1[item_idx].detach().cpu()),
                    "bg_l2": float(bg_l2[item_idx].detach().cpu()),
                    "mae_l": float(vel_metrics["mae_l"][item_idx].detach().cpu()),
                    "ssim": float(vel_metrics["ssim"][item_idx].detach().cpu()),
                }
                append_metrics_csv(metrics_path, row)
                for key in values:
                    values[key].append(float(row[key]))
                if dataset:
                    bucket = by_dataset.setdefault(dataset, {key: [] for key in values})
                    for key in values:
                        bucket[key].append(float(row[key]))
                sample_index += 1
    summary = {
        "name": name,
        "num_samples": sample_index,
        "means": {key: _mean(items) for key, items in values.items()},
        "dataset_summary": {
            dataset: {"num_samples": len(next(iter(bucket.values()))), **{key: _mean(items) for key, items in bucket.items()}}
            for dataset, bucket in sorted(by_dataset.items())
        },
    }
    out = output_dir / name
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    dataset_csv = out / "dataset_summary.csv"
    with dataset_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "num_samples", *values.keys()])
        writer.writeheader()
        for dataset, row in summary["dataset_summary"].items():
            writer.writerow({"dataset": dataset, **row})
    return summary


def run_background_feature_probe(conf: Any) -> dict[str, Any]:
    configure_torch_runtime(_conf_get(conf, "training.matmul_precision", None))
    output_dir = Path(str(_conf_get(conf, "diagnostic.output_dir", "logs/bg_pdr_fm/bg_feature_diagnostic/linear_mlp_oracle_bg")))
    output_dir.mkdir(parents=True, exist_ok=True)
    device = configured_torch_device(_conf_get(conf, "training.accelerator", "auto"), _conf_get(conf, "training.devices", "auto"))
    train_loader = _build_loader(conf, "train", shuffle=True)
    eval_split = str(_conf_get(conf, "evaluation.split", "test"))
    eval_loader = _build_loader(conf, eval_split, shuffle=False)
    model = BGPDRFMLightning(conf)
    _load_contrastive_encoder(model, _conf_get(conf, "evaluation.checkpoints.contrastive", None))
    model.eval().to(device)
    for parameter in model.encoder.parameters():
        parameter.requires_grad = False

    in_channels = int(_conf_get(conf, "model.feature_channels", 16))
    hidden = int(_conf_get(conf, "diagnostic.probe_hidden_channels", 32))
    kinds = list(_conf_get(conf, "diagnostic.probes", ["linear_1x1", "mlp_1x1"]))
    max_epochs = int(_conf_get(conf, "training.max_epochs", 10))
    max_train_batches = _conf_get(conf, "training.limit_train_batches", None)
    max_train_batches = None if max_train_batches is None else int(max_train_batches)
    max_eval_batches = _conf_get(conf, "evaluation.max_batches", None)
    max_eval_batches = None if max_eval_batches is None else int(max_eval_batches)
    lr = float(_conf_get(conf, "training.lr", 1e-3))
    summaries = []
    for kind in kinds:
        probe = BackgroundProbe(in_channels=in_channels, kind=kind, hidden_channels=hidden).to(device)
        opt = torch.optim.AdamW(probe.parameters(), lr=lr)
        train_losses: list[float] = []
        for _epoch in range(max_epochs):
            probe.train()
            for batch_idx, batch in enumerate(train_loader):
                if max_train_batches is not None and batch_idx >= max_train_batches:
                    break
                batch = batch_to_device(BGPDRFMLightning._as_batch(batch), device)
                with torch.no_grad():
                    target = model.filter.lowpass(batch.depth_vel)
                    features = model.encoder(batch)
                pred = probe(features.numerical)
                l1 = F.l1_loss(pred, target)
                l2 = F.mse_loss(pred, target)
                loss = l1 + 0.5 * l2
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                train_losses.append(float(loss.detach().cpu()))
        summary = _evaluate_probe(
            conf=conf,
            model=model,
            probe=probe,
            loader=eval_loader,
            device=device,
            output_dir=output_dir,
            name=kind,
            max_batches=max_eval_batches,
        )
        summary["train_loss_last"] = train_losses[-1] if train_losses else float("nan")
        torch.save({"probe": kind, "state_dict": probe.state_dict(), "summary": summary}, output_dir / kind / "probe_last.ckpt")
        (output_dir / kind / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        summaries.append(summary)
    combined = {"summaries": summaries}
    (output_dir / "summary.json").write_text(json.dumps(combined, indent=2), encoding="utf-8")
    return combined


def main(config_path: str) -> dict[str, Any]:
    conf = OmegaConf.load(config_path)
    return run_background_feature_probe(conf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train/evaluate tiny probes from C_N to oracle lowpass background.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    main(args.config)
