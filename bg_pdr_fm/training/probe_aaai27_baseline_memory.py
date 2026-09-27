"""Measure one official baseline train/validation batch peak GPU memory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from bg_pdr_fm.training.benchmark_config import load_benchmark_config
from bg_pdr_fm.training.dispatch_aaai27_baseline_gpu_queue import required_free_mib
from bg_pdr_fm.training.train_aaai27_benchmark import run_benchmark_config
from bg_pdr_fm.training.train_bg_pdr_fm import apply_fast_run_overrides


def run_probe(config_path: Path, output_path: Path) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("A visible ROCm/CUDA GPU is required for the memory probe.")
    conf = apply_fast_run_overrides(load_benchmark_config(config_path))
    artifact_dir = output_path.parent / f"{output_path.stem}_artifacts"
    conf.training.logging.log_dir = str(artifact_dir / "lightning")
    conf.training.logging.log_version = "probe"
    conf.training.logging.add_timestamp = False
    conf.training.checkpoint.dirpath = str(artifact_dir / "checkpoints")
    conf.training.checkpoint.save_top_k = 0
    conf.training.checkpoint.save_last = False
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    run_benchmark_config(conf)
    torch.cuda.synchronize()
    peak_mib = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
    payload: dict[str, object] = {
        "config": str(config_path),
        "variant": str(conf.benchmark.variant),
        "batch_size": int(conf.training.batch_size),
        "peak_allocated_mib": peak_mib,
        "required_free_mib": required_free_mib(peak_mib),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run_probe(args.config, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
