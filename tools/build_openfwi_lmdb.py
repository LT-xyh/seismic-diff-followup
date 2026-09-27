from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bg_pdr_fm.data.openfwi_lmdb import (
    DEFAULT_OPENFWI_LMDB_MODALITIES,
    openfwi_lmdb_path,
    write_openfwi_lmdb_dataset,
)


DEFAULT_DATASETS = ("FlatVelA", "FlatVelB", "CurveVelA", "CurveVelB", "CurveFaultA")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build map-style LMDB stores for OpenFWI.")
    parser.add_argument("--source-root", default="data/openfwi", help="Root containing OpenFWI sub-datasets.")
    parser.add_argument("--lmdb-root", default="data/openfwi_lmdb", help="Destination root for *.lmdb folders.")
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS), help="OpenFWI sub-datasets to pack.")
    parser.add_argument("--modalities", nargs="+", default=list(DEFAULT_OPENFWI_LMDB_MODALITIES))
    parser.add_argument("--normalization-profile", default="openfwi")
    parser.add_argument("--map-size-gb", type=float, default=None, help="Per-dataset LMDB map size. Defaults to auto estimate.")
    parser.add_argument("--commit-interval", type=int, default=512, help="Samples per write transaction.")
    parser.add_argument("--overwrite", action="store_true", help="Remove an existing destination LMDB before writing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = Path(args.source_root)
    lmdb_root = Path(args.lmdb_root)
    lmdb_root.mkdir(parents=True, exist_ok=True)
    map_size = None if args.map_size_gb is None else int(args.map_size_gb * (1024 ** 3))

    for dataset_name in args.datasets:
        source_dir = source_root / dataset_name
        output_path = openfwi_lmdb_path(lmdb_root, dataset_name)
        if args.overwrite and output_path.exists():
            shutil.rmtree(output_path)
        manifest = write_openfwi_lmdb_dataset(
            source_dataset_dir=source_dir,
            lmdb_path=output_path,
            dataset_name=dataset_name,
            modalities=args.modalities,
            normalization_profile=args.normalization_profile,
            map_size=map_size,
            overwrite=args.overwrite,
            commit_interval=args.commit_interval,
        )
        print(
            f"{dataset_name}: wrote {manifest['sample_count']} samples, "
            f"modalities={manifest['modalities']} -> {output_path}"
        )


if __name__ == "__main__":
    main()
