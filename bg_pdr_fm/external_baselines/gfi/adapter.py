"""Dataloader adapter for the official GFI waveform-to-velocity contract."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


EXPECTED_DATA_SHAPE = (5, 1000, 70)
EXPECTED_LABEL_SHAPE = (1, 70, 70)


@dataclass(frozen=True)
class GFISmokePair:
    """One annotation entry in the official GFI text-file convention."""

    data_path: Path
    label_path: Path


def _resolve_pair_line(line: str) -> GFISmokePair:
    text = line.strip()
    if not text or text.startswith("#"):
        raise ValueError("annotation line is empty")
    parts = [part for part in text.split("\t") if part]
    if len(parts) != 2:
        raise ValueError(f"expected one tab-separated data/label pair, got: {line!r}")
    return GFISmokePair(data_path=Path(parts[0]).expanduser().resolve(), label_path=Path(parts[1]).expanduser().resolve())


def read_gfi_annotation(annotation_path: str | Path) -> list[GFISmokePair]:
    path = Path(annotation_path)
    if not path.is_file():
        raise FileNotFoundError(f"GFI annotation file not found: {path}")
    pairs: list[GFISmokePair] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        pairs.append(_resolve_pair_line(line))
    if not pairs:
        raise ValueError(f"GFI annotation file is empty: {path}")
    return pairs


def _load_array(path: Path, expected_tail: tuple[int, ...], role: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"{role} array not found: {path}")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    shape = tuple(int(dim) for dim in array.shape)
    if len(shape) != len(expected_tail) + 1 or tuple(shape[1:]) != expected_tail:
        raise ValueError(f"{role} array must have shape [N, {expected_tail}], got {shape} at {path}")
    return np.array(array, copy=True)


class GFIAnnotationDataset(Dataset):
    """Minimal dataset adapter for official GFI annotation files."""

    def __init__(self, annotation_path: str | Path):
        self.annotation_path = Path(annotation_path)
        self.pairs = read_gfi_annotation(self.annotation_path)
        self._loaded: list[tuple[np.ndarray, np.ndarray]] = []
        self._offsets: list[int] = []
        total = 0
        for pair in self.pairs:
            data = _load_array(pair.data_path, EXPECTED_DATA_SHAPE, "data")
            label = _load_array(pair.label_path, EXPECTED_LABEL_SHAPE, "label")
            if data.shape[0] != label.shape[0]:
                raise ValueError(
                    f"data/label sample counts differ for {pair.data_path} and {pair.label_path}: "
                    f"{data.shape[0]} vs {label.shape[0]}"
                )
            self._loaded.append((data, label))
            self._offsets.append(total)
            total += int(data.shape[0])
        self._length = total

    def __len__(self) -> int:
        return self._length

    def _locate(self, index: int) -> tuple[int, int]:
        if index < 0 or index >= self._length:
            raise IndexError(index)
        for pair_idx, start in enumerate(self._offsets):
            data = self._loaded[pair_idx][0]
            end = start + int(data.shape[0])
            if index < end:
                return pair_idx, index - start
        raise IndexError(index)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        pair_idx, local_idx = self._locate(index)
        data, label = self._loaded[pair_idx]
        amp = torch.from_numpy(np.array(data[local_idx], copy=True)).float()
        vel = torch.from_numpy(np.array(label[local_idx], copy=True)).float()
        return amp, vel


def build_gfi_dataloader(
    annotation_path: str | Path,
    *,
    batch_size: int = 2,
    shuffle: bool = False,
    num_workers: int = 0,
    pin_memory: bool = False,
) -> DataLoader:
    """Build a minimal GFI dataloader for smoke checks or post-hoc evaluation."""

    dataset = GFIAnnotationDataset(annotation_path)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def annotation_summary(annotation_path: str | Path) -> dict[str, object]:
    pairs = read_gfi_annotation(annotation_path)
    return {
        "annotation_path": str(Path(annotation_path).resolve()),
        "num_pairs": len(pairs),
        "pairs": [
            {
                "data_path": str(pair.data_path),
                "label_path": str(pair.label_path),
            }
            for pair in pairs
        ],
    }
