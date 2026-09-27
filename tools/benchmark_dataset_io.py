from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bg_pdr_fm.data.raw_datasets import DEFAULT_OPENFWI_SCHEMA, OpenFWI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark OpenFWI Dataset throughput.")
    parser.add_argument("--root-dir", default="data/openfwi")
    parser.add_argument("--lmdb-root", default="data/openfwi_lmdb")
    parser.add_argument("--datasets", nargs="+", default=["FlatVelA"])
    parser.add_argument("--backend", choices=["auto", "npy", "lmdb"], default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=3)
    parser.add_argument("--batches", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = OpenFWI(
        root_dir=args.root_dir,
        lmdb_root=args.lmdb_root,
        datasets=args.datasets,
        use_data=DEFAULT_OPENFWI_SCHEMA,
        storage_backend=args.backend,
        well_random=False,
    )
    loader_kwargs = dict(
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    loader = DataLoader(dataset, **loader_kwargs)

    start = time.perf_counter()
    samples = 0
    bytes_seen = 0
    for batch_idx, batch in enumerate(loader):
        current = next(iter(batch.values()))
        samples += int(current.shape[0])
        bytes_seen += sum(tensor.numel() * tensor.element_size() for tensor in batch.values() if torch.is_tensor(tensor))
        if batch_idx + 1 >= args.batches:
            break
    elapsed = max(time.perf_counter() - start, 1e-9)
    print(f"backend={dataset.storage_backend}")
    print(f"samples={samples} seconds={elapsed:.3f} samples_per_s={samples / elapsed:.2f}")
    print(f"tensor_mb={bytes_seen / (1024 ** 2):.2f} tensor_mb_per_s={bytes_seen / (1024 ** 2) / elapsed:.2f}")


if __name__ == "__main__":
    main()
