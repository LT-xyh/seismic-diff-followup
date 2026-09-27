"""Config loading helpers for AAAI27 benchmark configs."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from bg_pdr_fm.reproducibility import resolved_config_hash


def load_benchmark_config(config_path: str | Path) -> Any:
    path = Path(config_path).resolve()
    conf = OmegaConf.load(path)
    extends = OmegaConf.select(conf, "extends", default=None)
    if extends is None:
        return conf
    base_path = (path.parent / str(extends)).resolve()
    base = load_benchmark_config(base_path)
    data = OmegaConf.to_container(conf, resolve=False)
    if isinstance(data, dict):
        data.pop("extends", None)
    return OmegaConf.merge(base, OmegaConf.create(data))


def benchmark_config_hash(config_path: str | Path) -> str:
    """Return the provenance hash of a fully resolved benchmark config."""
    return resolved_config_hash(load_benchmark_config(config_path))
