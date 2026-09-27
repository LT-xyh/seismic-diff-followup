"""Create official-GFI annotation files for waveform-to-velocity smoke checks."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


EXPECTED_DATA_SHAPE_TAIL = (5, 1000, 70)
EXPECTED_LABEL_SHAPE_TAIL = (1, 70, 70)


def _check_array(path: Path, expected_tail: tuple[int, ...], role: str) -> tuple[int, ...]:
    if not path.is_file():
        raise FileNotFoundError(f"{role} array not found: {path}")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    shape = tuple(int(dim) for dim in array.shape)
    if len(shape) != len(expected_tail) + 1 or shape[1:] != expected_tail:
        raise ValueError(f"{role} array must have shape [N, {expected_tail}], got {shape} at {path}")
    return shape


def write_annotation(data_path: Path, label_path: Path, output_path: Path) -> dict[str, object]:
    data_path = data_path.resolve()
    label_path = label_path.resolve()
    output_path = output_path.resolve()
    data_shape = _check_array(data_path, EXPECTED_DATA_SHAPE_TAIL, "data")
    label_shape = _check_array(label_path, EXPECTED_LABEL_SHAPE_TAIL, "label")
    if data_shape[0] != label_shape[0]:
        raise ValueError(f"data/label sample counts differ: {data_shape[0]} vs {label_shape[0]}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(f"{data_path}\t{label_path}\n", encoding="utf-8")
    return {
        "output": str(output_path),
        "data": str(data_path),
        "label": str(label_path),
        "num_samples": data_shape[0],
        "data_shape": data_shape,
        "label_shape": label_shape,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Path to GFI waveform array [N,5,1000,70].")
    parser.add_argument("--label", type=Path, required=True, help="Path to velocity array [N,1,70,70].")
    parser.add_argument("--output", type=Path, required=True, help="Output annotation text file.")
    args = parser.parse_args()
    info = write_annotation(args.data, args.label, args.output)
    for key, value in info.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

