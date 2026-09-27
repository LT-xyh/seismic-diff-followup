"""Small plotting and CSV helpers for BG-PDR-FM diagnostics."""

from __future__ import annotations

from pathlib import Path
import csv

import matplotlib.pyplot as plt
import torch


def _to_image(x: torch.Tensor):
    x = x.detach().float().cpu()
    if x.ndim == 4:
        x = x[0]
    if x.ndim == 3:
        x = x[0]
    return x.numpy()


def save_tensor_panel(tensors: dict[str, torch.Tensor], output_path: str | Path, cmap: str = "jet") -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    names = list(tensors.keys())
    fig, axes = plt.subplots(1, len(names), figsize=(4 * len(names), 4))
    if len(names) == 1:
        axes = [axes]
    for ax, name in zip(axes, names):
        ax.imshow(_to_image(tensors[name]), cmap=cmap)
        ax.set_title(name)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_metric_pair_plot(
    x: torch.Tensor,
    y: torch.Tensor,
    output_path: str | Path,
    xlabel: str = "rho_B",
    ylabel: str = "rho_hat_B",
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    x_np = x.detach().float().flatten().cpu().numpy()
    y_np = y.detach().float().flatten().cpu().numpy()
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    ax.scatter(x_np, y_np, s=18, alpha=0.8)
    if x_np.size and y_np.size:
        lo = min(float(x_np.min()), float(y_np.min()))
        hi = max(float(x_np.max()), float(y_np.max()))
        if lo == hi:
            lo -= 0.5
            hi += 0.5
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=1, linestyle="--")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def append_metrics_csv(output_path: str | Path, row: dict[str, float | int | str]) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = output_path.exists()
    fieldnames = list(row.keys())
    existing_rows: list[dict[str, str]] = []
    needs_rewrite = False
    if file_exists:
        with output_path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames:
                fieldnames = list(reader.fieldnames)
            existing_rows = list(reader)
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
                needs_rewrite = True

    mode = "w" if (not file_exists or needs_rewrite) else "a"
    with output_path.open(mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if mode == "w":
            writer.writeheader()
            writer.writerows(existing_rows)
        writer.writerow(row)
