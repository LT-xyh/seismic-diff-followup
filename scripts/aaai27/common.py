"""Shared argument and configuration helpers for AAAI27 release scripts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from omegaconf import OmegaConf

from bg_pdr_fm.reproducibility import assert_release_paths, resolved_config_hash
from bg_pdr_fm.training.benchmark_config import load_benchmark_config


def add_release_arguments(parser: argparse.ArgumentParser, *, default_config: str) -> None:
    parser.add_argument("--config", default=default_config)
    parser.add_argument("--output", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")


def resolve_release_config(args: argparse.Namespace, *, apply_output: bool = True) -> tuple[Path, Any]:
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    conf = load_benchmark_config(config_path)
    assert_release_paths(conf, REPO_ROOT)
    if args.data_root:
        data_root = str(Path(args.data_root))
        OmegaConf.update(conf, "data.root_dir", data_root, merge=True)
        OmegaConf.update(conf, "data.root_candidates", [data_root], merge=True)
    if apply_output and args.output:
        _set_output_root(conf, Path(args.output))
    return config_path, conf


def emit_dry_run(config_path: Path, conf: Any) -> None:
    summary = {
        "config": str(config_path.relative_to(REPO_ROOT)),
        "config_sha256": resolved_config_hash(conf),
        "profile": str(OmegaConf.select(conf, "release.profile", default="")),
        "data_name": str(OmegaConf.select(conf, "data.name", default="")),
        "stage": str(OmegaConf.select(conf, "training.stage", default="")),
        "training_seed": OmegaConf.select(conf, "training.seed", default=None),
        "evaluation_split": str(OmegaConf.select(conf, "evaluation.split", default="")),
    }
    print(json.dumps(summary, sort_keys=True))


def _set_output_root(conf: Any, output_root: Path) -> None:
    root = str(output_root)
    OmegaConf.update(conf, "training.stage_checkpoint_path", f"{root}/stage_checkpoints", merge=True)
    OmegaConf.update(conf, "training.logging.log_dir", f"{root}/lightning", merge=True)
    OmegaConf.update(conf, "training.checkpoint.dirpath", f"{root}/checkpoints", merge=True)
    OmegaConf.update(conf, "diagnostics.output_dir", f"{root}/diagnostics", merge=True)
    OmegaConf.update(conf, "evaluation.output_dir", f"{root}/evaluation", merge=True)
