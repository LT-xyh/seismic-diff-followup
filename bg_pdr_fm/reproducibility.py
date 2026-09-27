"""Shared reproducibility helpers for paper-facing release profiles."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterator
from pathlib import Path, PureWindowsPath
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf


DEFAULT_TRAINING_SEED = 2027


def seed_everything(seed: int, *, deterministic: bool = True) -> int:
    """Seed local RNGs and Lightning worker initialization from one seed."""
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    import lightning

    lightning.seed_everything(seed, workers=True)
    return seed


def assert_release_paths(conf: Any, repo_root: str | Path) -> None:
    """Reject machine-absolute data, checkpoint, and evaluation paths."""
    del repo_root
    container = OmegaConf.to_container(conf, resolve=True)
    if not isinstance(container, dict):
        raise TypeError("Release config must resolve to a mapping.")

    for section in ("data", "evaluation"):
        for field, value in _iter_string_values(container.get(section), section):
            _raise_if_absolute(field, value)

    for field, value in _iter_string_values(container.get("training"), "training"):
        key_parts = field.split(".")
        if any("checkpoint" in part for part in key_parts) or key_parts[-1] == "ckpt_path":
            _raise_if_absolute(field, value)


def _iter_string_values(value: object, prefix: str) -> Iterator[tuple[str, str]]:
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}"
            yield from _iter_string_values(child, child_prefix)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _iter_string_values(child, f"{prefix}[{index}]")
    elif isinstance(value, str):
        yield prefix, value


def _raise_if_absolute(field: str, value: str) -> None:
    if Path(value).is_absolute() or PureWindowsPath(value).is_absolute():
        raise ValueError(f"Release config path {field} must be relative, got {value!r}.")


def resolved_config_hash(conf: Any) -> str:
    """Return a stable SHA-256 identity for an OmegaConf-resolved config."""
    resolved = OmegaConf.to_container(conf, resolve=True)
    serialized = json.dumps(resolved, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()
