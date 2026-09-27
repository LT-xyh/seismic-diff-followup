from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import numpy as np

from bg_pdr_fm.data.openfwi_lmdb import DEFAULT_OPENFWI_LMDB_MODALITIES, prepare_openfwi_lmdb_root


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts" / "aaai27"


@pytest.mark.parametrize("name", ("prepare_openfwi.py", "train.py", "evaluate.py", "smoke.py", "validate.py"))
def test_release_entrypoints_expose_help(name: str) -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / name), "--help"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--config" in result.stdout


def test_train_dry_run_resolves_release_config_without_lightning() -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPTS_DIR / "train.py"),
            "--config",
            "bg_pdr_fm/configs/release/aaai27/train.yaml",
            "--dry-run",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert '"training_seed": 2027' in result.stdout
    assert '"stage": "joint_full"' in result.stdout


def test_release_cli_config_import_does_not_load_lightning() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; sys.path.insert(0, 'scripts/aaai27'); import common; "
            "print('lightning_loaded=' + str('lightning' in sys.modules))",
        ],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "lightning_loaded=False" in result.stdout


def test_smoke_dry_run_uses_synthetic_data_without_openfwi_root() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "smoke.py"), "--dry-run"],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert '"data_name": "synthetic"' in result.stdout


def test_prepare_openfwi_lmdb_root_records_source_provenance(tmp_path: Path) -> None:
    source_root = tmp_path / "openfwi"
    dataset_root = source_root / "FlatVelA"
    for modality in DEFAULT_OPENFWI_LMDB_MODALITIES:
        modality_root = dataset_root / modality
        modality_root.mkdir(parents=True)
        for index in range(2):
            np.save(modality_root / f"{index:04d}.npy", np.full((2, 2), index, dtype=np.float32))

    prepared = prepare_openfwi_lmdb_root(
        source_root=source_root,
        lmdb_root=tmp_path / "openfwi_lmdb",
        dataset_names=("FlatVelA",),
        split_seed=42,
        well_seed=1234,
    )

    assert prepared["split_seed"] == 42
    assert prepared["well_seed"] == 1234
    assert prepared["datasets"]["FlatVelA"]["sample_count"] == 2
    assert prepared["datasets"]["FlatVelA"]["input_sha256"]
