from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from bg_pdr_fm.reproducibility import assert_release_paths, resolved_config_hash, seed_everything
from bg_pdr_fm.training.train_bg_pdr_fm import build_loaders


def test_seed_everything_replays_python_numpy_and_torch_draws() -> None:
    seed_everything(2027)
    first = (random.random(), np.random.rand(), torch.rand(3))

    seed_everything(2027)
    repeated = (random.random(), np.random.rand(), torch.rand(3))

    assert first[0] == repeated[0]
    assert first[1] == repeated[1]
    torch.testing.assert_close(first[2], repeated[2])


def test_assert_release_paths_rejects_absolute_data_root() -> None:
    conf = OmegaConf.create({"data": {"root_dir": "/tmp/openfwi"}})

    with pytest.raises(ValueError, match="data.root_dir"):
        assert_release_paths(conf, repo_root=Path("."))


def test_assert_release_paths_rejects_absolute_training_checkpoint() -> None:
    conf = OmegaConf.create({"training": {"checkpoint": {"dirpath": "/tmp/checkpoints"}}})

    with pytest.raises(ValueError, match="training.checkpoint.dirpath"):
        assert_release_paths(conf, repo_root=Path("."))


def test_assert_release_paths_rejects_absolute_evaluation_output() -> None:
    conf = OmegaConf.create({"evaluation": {"output_dir": "/tmp/evaluation"}})

    with pytest.raises(ValueError, match="evaluation.output_dir"):
        assert_release_paths(conf, repo_root=Path("."))


@pytest.mark.parametrize(
    ("config", "field"),
    [
        ({"data": {"lmdb_root": "/tmp/openfwi_lmdb"}}, "data.lmdb_root"),
        ({"training": {"load_stage_checkpoint": "/tmp/warm_start.ckpt"}}, "training.load_stage_checkpoint"),
        ({"evaluation": {"checkpoints": {"full": "/tmp/final.ckpt"}}}, "evaluation.checkpoints.full"),
    ],
)
def test_assert_release_paths_rejects_nested_absolute_paths(config: dict[str, object], field: str) -> None:
    with pytest.raises(ValueError, match=field):
        assert_release_paths(OmegaConf.create(config), repo_root=Path("."))


def test_resolved_config_hash_is_stable_across_mapping_order() -> None:
    first = OmegaConf.create({"training": {"seed": 2027}, "data": {"split_seed": 42}})
    second = OmegaConf.create({"data": {"split_seed": 42}, "training": {"seed": 2027}})

    assert resolved_config_hash(first) == resolved_config_hash(second)


def test_training_loader_uses_its_configured_seed_for_shuffle() -> None:
    conf = OmegaConf.create(
        {
            "data": {"name": "synthetic", "train_length": 12, "val_length": 4},
            "training": {"seed": 2027, "batch_size": 3, "num_workers": 0, "pin_memory": False},
        }
    )

    first_loader, _ = build_loaders(conf)
    first_order = list(first_loader.sampler)
    torch.manual_seed(3407)
    second_loader, _ = build_loaders(conf)
    second_order = list(second_loader.sampler)

    assert first_order == second_order
