from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from bg_pdr_fm.data import SyntheticBGDataset, collate_bg_samples
from bg_pdr_fm.evaluation.missing_modalities import MissingModalityProtocol
from bg_pdr_fm.evaluation.run_missing_modality_benchmark import (
    JOBS,
    apply_openfwi_missing_overrides,
    validate_missing_outputs,
)


def test_missing_protocol_applies_official_masks_to_batch():
    protocol = MissingModalityProtocol.aaai27()
    batch = collate_bg_samples([SyntheticBGDataset(length=1)[0]])

    assert protocol.mode_names == [
        "full",
        "w/o well_log",
        "w/o horizon",
        "w/o rms_vel",
        "w/o well+rms",
        "PSTM only",
    ]
    assert protocol.expected_num_modes == 6

    pstm_only = protocol.apply(batch, "PSTM only")
    assert torch.all(pstm_only.horizon == 0)
    assert torch.all(pstm_only.rms_vel == 0)
    assert torch.all(pstm_only.well_log == 0)
    assert torch.all(pstm_only.well_mask == 0)
    assert pstm_only.modality_mask[0].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert pstm_only.modality_quality[0].tolist() == [1.0, 0.0, 0.0, 0.0]


def test_apply_openfwi_missing_overrides_forces_six_modes_without_changing_formal_config():
    conf = OmegaConf.create(
        {
            "data": {"name": "placeholder"},
            "evaluation": {"missing_modes": ["full"], "output_dir": "old", "checkpoint": "old.ckpt"},
            "training": {"devices": 8, "num_workers": 0},
        }
    )
    protocol = MissingModalityProtocol.aaai27()

    updated = apply_openfwi_missing_overrides(
        conf,
        checkpoint=Path("model.ckpt"),
        output_dir=Path("out/job"),
        openfwi_root=Path("/data/openfwi"),
        lmdb_root=Path("/data/openfwi_lmdb"),
        max_batches=3,
        protocol=protocol,
    )

    assert list(updated.evaluation.missing_modes) == protocol.mode_names
    assert updated.evaluation.output_dir == "out/job"
    assert updated.evaluation.checkpoint == "model.ckpt"
    assert updated.evaluation.max_batches == 3
    assert updated.data.name == "openfwi"
    assert updated.data.storage_backend == "lmdb"
    assert updated.training.devices == 1
    assert updated.training.num_workers >= 1


def test_apply_openfwi_missing_overrides_allows_stable_zero_worker_mode():
    conf = OmegaConf.create(
        {
            "data": {"name": "placeholder"},
            "evaluation": {"missing_modes": ["full"], "output_dir": "old", "checkpoint": "old.ckpt"},
            "training": {"devices": 8, "num_workers": 8, "persistent_workers": True, "prefetch_factor": 4},
        }
    )
    protocol = MissingModalityProtocol.aaai27()

    updated = apply_openfwi_missing_overrides(
        conf,
        checkpoint=Path("model.ckpt"),
        output_dir=Path("out/job"),
        openfwi_root=Path("/data/openfwi"),
        lmdb_root=Path("/data/openfwi_lmdb"),
        max_batches=None,
        protocol=protocol,
        num_workers=0,
    )

    assert updated.training.num_workers == 0
    assert updated.training.persistent_workers is False
    assert updated.training.prefetch_factor is None


def test_validate_missing_outputs_rejects_missing_modes(tmp_path: Path):
    out = tmp_path / "job"
    out.mkdir()
    (out / "summary.json").write_text(json.dumps({"num_samples": 33600}), encoding="utf-8")
    (out / "metrics.csv").write_text("mae\n0.1\n", encoding="utf-8")
    (out / "dataset_missing_mode_summary.csv").write_text("dataset_name,missing_mode,num_samples\n", encoding="utf-8")
    with (out / "missing_mode_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["missing_mode", "num_samples", "mae", "rmse", "ssim", "mae_l", "mae_h"])
        writer.writeheader()
        writer.writerow(
            {
                "missing_mode": "full",
                "num_samples": 33600,
                "mae": 0.1,
                "rmse": 0.2,
                "ssim": 0.8,
                "mae_l": 0.08,
                "mae_h": 0.02,
            }
        )

    with pytest.raises(RuntimeError, match="missing modes"):
        validate_missing_outputs(out, MissingModalityProtocol.aaai27(), max_batches=None)


def test_validate_missing_outputs_accepts_dry_run_counts(tmp_path: Path):
    out = tmp_path / "job"
    out.mkdir()
    protocol = MissingModalityProtocol.aaai27()
    (out / "summary.json").write_text(json.dumps({"num_samples": 12}), encoding="utf-8")
    with (out / "missing_mode_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["missing_mode", "num_samples", "mae", "rmse", "ssim", "mae_l", "mae_h"])
        writer.writeheader()
        for mode in protocol.mode_names:
            writer.writerow(
                {
                    "missing_mode": mode,
                    "num_samples": 2,
                    "mae": 0.1,
                    "rmse": 0.2,
                    "ssim": 0.8,
                    "mae_l": 0.08,
                    "mae_h": 0.02,
                }
            )
    (out / "dataset_missing_mode_summary.csv").write_text("dataset_name,missing_mode,num_samples\n", encoding="utf-8")
    (out / "metrics.csv").write_text("mae\n0.1\n", encoding="utf-8")

    result = validate_missing_outputs(out, protocol, max_batches=1)

    assert result["status"] == "complete"
    assert result["num_modes"] == 6
    assert result["num_samples"] == 12


def test_missing_runner_job_registry_contains_external_table_methods():
    names = {job.name for job in JOBS}
    assert {"smooth_dix", "adapted_gfi", "adapted_auto_linear", "sv_inv_net", "velocity_gan"}.issubset(names)


def test_missing_runner_sv_inv_net_uses_full_formal_checkpoint_not_short_highio():
    job = next(job for job in JOBS if job.name == "sv_inv_net")

    assert job.config == "bg_pdr_fm/configs/experiments/aaai27/formal_sv_inv_net.yaml"
    assert job.checkpoint_hint == "logs/bg_pdr_fm/aaai27/formal/sv_inv_net/checkpoints/last.ckpt"
    assert "highio" not in job.config
    assert "highio" not in job.checkpoint_hint
