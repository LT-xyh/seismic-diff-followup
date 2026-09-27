"""Audit OpenFWI raw waveform shards for the official GFI baseline.

The official GFI split files point to OpenFWI raw waveform arrays shaped
``[N, 5, 1000, 70]`` and velocity arrays shaped ``[N, 1, 70, 70]``. This helper
remaps those split files to a local raw-data root, verifies shapes, and writes
local annotation files that can be consumed by the official GFI code.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


DATASET_MAP = {
    "FlatVelA": "flatvel_a",
    "FlatVelB": "flatvel_b",
    "CurveVelA": "curvevel_a",
    "CurveVelB": "curvevel_b",
}

EXPECTED_DATA_TAIL = (5, 1000, 70)
EXPECTED_LABEL_TAIL = (1, 70, 70)


def _resolve_local_path(original: str, raw_root: Path) -> Path:
    parts = Path(original).parts
    try:
        openfwi_idx = parts.index("openfwi")
        rel_parts = parts[openfwi_idx + 1 :]
    except ValueError:
        rel_parts = parts[-4:]
    return raw_root.joinpath(*rel_parts)


def _array_shape(path: Path, expected_tail: tuple[int, ...], role: str) -> tuple[int, ...]:
    if not path.is_file():
        raise FileNotFoundError(f"{role} shard not found: {path}")
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    shape = tuple(int(dim) for dim in array.shape)
    if len(shape) != len(expected_tail) + 1 or shape[1:] != expected_tail:
        raise ValueError(f"{role} shard must have shape [N,{expected_tail}], got {shape}: {path}")
    return shape


def _read_split(split_path: Path, raw_root: Path) -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for raw_line in split_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = [item for item in line.split("\t") if item]
        if len(parts) != 2:
            raise ValueError(f"expected tab-separated data/label line in {split_path}: {raw_line!r}")
        pairs.append((_resolve_local_path(parts[0], raw_root), _resolve_local_path(parts[1], raw_root)))
    if not pairs:
        raise ValueError(f"split file is empty: {split_path}")
    return pairs


def audit_split(
    *,
    raw_root: Path,
    split_root: Path,
    output_dir: Path,
    datasets: list[str],
    strict: bool,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "raw_root": str(raw_root),
        "split_root": str(split_root),
        "output_dir": str(output_dir),
        "datasets": {},
        "status": "ok",
    }
    any_missing = False
    for dataset in datasets:
        split_prefix = DATASET_MAP[dataset]
        dataset_info: dict[str, object] = {}
        for split in ("train", "val"):
            split_path = split_root / f"{split_prefix}_{split}.txt"
            pairs = _read_split(split_path, raw_root)
            rows: list[str] = []
            sample_count = 0
            missing: list[str] = []
            shape_errors: list[str] = []
            for data_path, label_path in pairs:
                try:
                    data_shape = _array_shape(data_path, EXPECTED_DATA_TAIL, "data")
                    label_shape = _array_shape(label_path, EXPECTED_LABEL_TAIL, "label")
                    if data_shape[0] != label_shape[0]:
                        raise ValueError(f"sample count mismatch {data_shape[0]} vs {label_shape[0]}")
                    sample_count += int(data_shape[0])
                    rows.append(f"{data_path.resolve()}\t{label_path.resolve()}")
                except FileNotFoundError as exc:
                    missing.append(str(exc))
                except ValueError as exc:
                    shape_errors.append(str(exc))
            annotation_path = output_dir / f"{dataset}_{split}.txt"
            if rows:
                annotation_path.write_text("\n".join(rows) + "\n", encoding="utf-8")
            dataset_info[split] = {
                "official_split": str(split_path),
                "annotation": str(annotation_path),
                "num_shards": len(pairs),
                "num_valid_shards": len(rows),
                "num_samples": sample_count,
                "missing": missing,
                "shape_errors": shape_errors,
            }
            if missing or shape_errors or not rows:
                any_missing = True
        manifest["datasets"][dataset] = dataset_info
    if any_missing:
        manifest["status"] = "blocked_missing_or_invalid_raw_shards"
    if strict and manifest["status"] != "ok":
        raise RuntimeError(json.dumps(manifest, indent=2))
    (output_dir / "gfi_raw_split_audit.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=Path("/public/home/xuyinghao/workspace/datasets/openfwi_raw"))
    parser.add_argument(
        "--split-root",
        type=Path,
        default=Path("/public/home/xuyinghao/workspace/external_repos/gfi/train_test_splits"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logs/bg_pdr_fm/aaai27/external_gfi_native/inputs"),
    )
    parser.add_argument("--datasets", nargs="+", default=list(DATASET_MAP), choices=list(DATASET_MAP))
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    manifest = audit_split(
        raw_root=args.raw_root,
        split_root=args.split_root,
        output_dir=args.output_dir,
        datasets=list(args.datasets),
        strict=bool(args.strict),
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
