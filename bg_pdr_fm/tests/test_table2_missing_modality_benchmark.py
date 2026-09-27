from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
import torch

from bg_pdr_fm.data import SyntheticBGDataset, collate_bg_samples
from bg_pdr_fm.evaluation.missing_modalities import MissingModalityProtocol
from bg_pdr_fm.external_baselines.common import build_multimodal_condition_image
from bg_pdr_fm.evaluation.run_table2_missing_modality_benchmark import (
    EXPECTED_DATASET_COUNTS,
    EXPECTED_METHOD_NAMES,
    load_reusable_common_mode_rows,
    protocol_from_name,
    recover_complete_mode_rows,
    validate_identity_rows,
)


def _manifest() -> dict[tuple[str, int], int]:
    rows: dict[tuple[str, int], int] = {}
    for dataset_name, count in EXPECTED_DATASET_COUNTS.items():
        for source_sample_index in range(count):
            rows[(dataset_name, source_sample_index)] = 0
    return rows


def test_table2_method_registry_has_exact_six_methods():
    assert EXPECTED_METHOD_NAMES == (
        "InversionNet",
        "VelocityGAN",
        "UPFWI",
        "Auto-Linear",
        "Latent U-Net (Large)",
        "PD-BG-RFM",
    )


def test_paper_rms_only_protocol_is_versioned_from_legacy_protocol():
    legacy = MissingModalityProtocol.aaai27()
    paper = MissingModalityProtocol.aaai27_rms_only()

    assert "PSTM only" in legacy.mode_names
    assert "PSTM only" not in paper.mode_names
    assert paper.mode_names[-1] == "RMS only"
    assert paper.modes["RMS only"] == ("rms_vel",)


def test_rms_only_zeroes_non_rms_inputs_and_keeps_quality_contract():
    batch = collate_bg_samples([SyntheticBGDataset(length=1)[0]])
    paper = MissingModalityProtocol.aaai27_rms_only()

    rms_only = paper.apply(batch, "RMS only")
    condition = build_multimodal_condition_image(rms_only, input_hw=(70, 70))

    assert torch.count_nonzero(rms_only.migrated_image) == 0
    assert torch.count_nonzero(rms_only.horizon) == 0
    assert torch.equal(rms_only.rms_vel, batch.rms_vel)
    assert torch.count_nonzero(rms_only.well_log) == 0
    assert torch.count_nonzero(rms_only.well_mask) == 0
    assert rms_only.modality_mask[0].tolist() == [0.0, 0.0, 1.0, 0.0]
    assert rms_only.modality_quality[0].tolist() == [0.0, 0.0, 1.0, 0.0]
    assert torch.count_nonzero(condition[:, 0]) == 0
    assert torch.count_nonzero(condition[:, 2:]) == 0
    assert torch.count_nonzero(condition[:, 1]) > 0


def test_single_modality_protocol_masks_exactly_one_input_at_a_time():
    batch = collate_bg_samples([SyntheticBGDataset(length=1)[0]])
    protocol = protocol_from_name("paper-single-only-v1")

    assert protocol.mode_names == ["RMS only", "Horizon only", "Well only", "PSTM only"]
    expected_masks = {
        "RMS only": [0.0, 0.0, 1.0, 0.0],
        "Horizon only": [0.0, 1.0, 0.0, 0.0],
        "Well only": [0.0, 0.0, 0.0, 1.0],
        "PSTM only": [1.0, 0.0, 0.0, 0.0],
    }
    for mode, expected_mask in expected_masks.items():
        selected = protocol.apply(batch, mode)
        assert selected.modality_mask[0].tolist() == expected_mask
        assert selected.modality_quality[0].tolist() == expected_mask


def test_radar_protocol_zeroes_only_pstm_and_preserves_other_modalities():
    batch = collate_bg_samples([SyntheticBGDataset(length=1)[0]])
    protocol = protocol_from_name("paper-radar-v1")

    without_pstm = protocol.apply(batch, "w/o PSTM")
    condition = build_multimodal_condition_image(without_pstm, input_hw=(70, 70))

    assert torch.count_nonzero(without_pstm.migrated_image) == 0
    assert torch.equal(without_pstm.horizon, batch.horizon)
    assert torch.equal(without_pstm.rms_vel, batch.rms_vel)
    assert torch.equal(without_pstm.well_log, batch.well_log)
    assert torch.equal(without_pstm.well_mask, batch.well_mask)
    assert without_pstm.modality_mask[0].tolist() == [0.0, 1.0, 1.0, 1.0]
    assert without_pstm.modality_quality[0].tolist() == [0.0, 1.0, 1.0, 1.0]
    assert torch.count_nonzero(condition[:, 0]) == 0
    assert torch.equal(condition[:, 1], build_multimodal_condition_image(batch, input_hw=(70, 70))[:, 1])


def test_reusable_common_modes_require_matching_provenance(tmp_path: Path):
    source_dir = tmp_path / "inversion_net"
    source_dir.mkdir()
    manifest = {("FlatVelA", 4): 0}
    modes = ("full", "w/o well_log", "w/o horizon", "w/o rms_vel", "w/o well+rms")
    fields = [
        "method",
        "dataset_id",
        "dataset_name",
        "source_sample_index",
        "missing_mode",
        "mae",
        "rmse",
        "ssim",
        "mae_l",
        "mae_h",
        "params",
    ]
    with (source_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for mode in modes:
            writer.writerow(
                {
                    "method": "InversionNet",
                    "dataset_id": 0,
                    "dataset_name": "FlatVelA",
                    "source_sample_index": 4,
                    "missing_mode": mode,
                    "mae": 0.1,
                    "rmse": 0.2,
                    "ssim": 0.8,
                    "mae_l": 0.08,
                    "mae_h": 0.02,
                    "params": 10,
                }
            )
    (source_dir / "run_manifest.json").write_text(
        json.dumps(
                {
                    "method": "InversionNet",
                    "canonical_manifest_sha256": "manifest-sha",
                    "checkpoint": {"sha256": "checkpoint-sha"},
                    "missing_modes": list(modes),
                }
        ),
        encoding="utf-8",
    )

    rows = load_reusable_common_mode_rows(
        source_dir,
        method_name="InversionNet",
        manifest=manifest,
        common_modes=modes,
        checkpoint_sha256="checkpoint-sha",
        canonical_manifest_sha256="manifest-sha",
    )

    assert len(rows) == len(modes)
    assert {row["missing_mode"] for row in rows} == set(modes)
    with pytest.raises(ValueError, match="checkpoint"):
        load_reusable_common_mode_rows(
            source_dir,
            method_name="InversionNet",
            manifest=manifest,
            common_modes=modes,
            checkpoint_sha256="wrong-checkpoint",
            canonical_manifest_sha256="manifest-sha",
        )
    with pytest.raises(ValueError, match="manifest"):
        load_reusable_common_mode_rows(
            source_dir,
            method_name="InversionNet",
            manifest=manifest,
            common_modes=modes,
            checkpoint_sha256="checkpoint-sha",
            canonical_manifest_sha256="wrong-manifest",
        )


def test_validate_identity_rows_accepts_one_complete_mode():
    manifest = _manifest()
    rows = [
        {
            "dataset_name": dataset_name,
            "source_sample_index": str(source_sample_index),
            "missing_mode": "full",
            "mae": "0.1",
            "rmse": "0.2",
            "ssim": "0.8",
            "mae_l": "0.08",
            "mae_h": "0.02",
        }
        for dataset_name, source_sample_index in manifest
    ]

    result = validate_identity_rows(rows, manifest, expected_modes=("full",))

    assert result["status"] == "passed"
    assert result["mode_counts"] == {"full": len(manifest)}
    assert result["dataset_counts"]["CurveVelB"] == EXPECTED_DATASET_COUNTS["CurveVelB"]


def test_validate_identity_rows_rejects_cross_split_or_duplicate_identity():
    manifest = {("FlatVelA", 4): 0}
    rows = [
        {
            "dataset_name": "FlatVelA",
            "source_sample_index": "4",
            "missing_mode": "full",
            "mae": "0.1",
            "rmse": "0.2",
            "ssim": "0.8",
            "mae_l": "0.08",
            "mae_h": "0.02",
        },
        {
            "dataset_name": "FlatVelA",
            "source_sample_index": "4",
            "missing_mode": "full",
            "mae": "0.1",
            "rmse": "0.2",
            "ssim": "0.8",
            "mae_l": "0.08",
            "mae_h": "0.02",
        },
    ]

    with pytest.raises(ValueError, match="duplicate"):
        validate_identity_rows(rows, manifest, expected_modes=("full",))


def test_validate_identity_rows_rejects_dataset_id_mismatch():
    manifest = {("FlatVelA", 4): 0}
    row = {
        "dataset_id": "1",
        "dataset_name": "FlatVelA",
        "source_sample_index": "4",
        "missing_mode": "full",
        "mae": "0.1",
        "rmse": "0.2",
        "ssim": "0.8",
        "mae_l": "0.08",
        "mae_h": "0.02",
    }

    with pytest.raises(ValueError, match="dataset_id mismatch"):
        validate_identity_rows([row], manifest, expected_modes=("full",))


def test_validate_identity_rows_rejects_non_finite_metrics():
    rows = [
        {
            "dataset_name": "FlatVelA",
            "source_sample_index": "4",
            "missing_mode": "full",
            "mae": "nan",
            "rmse": "0.2",
            "ssim": "0.8",
            "mae_l": "0.08",
            "mae_h": "0.02",
        }
    ]

    with pytest.raises(ValueError, match="non-finite"):
        validate_identity_rows(rows, {("FlatVelA", 4): 0}, expected_modes=("full",))


def test_recover_complete_mode_rows_discards_only_partial_tail():
    rows = [
        {"missing_mode": "full", "source_sample_index": str(index)} for index in range(3)
    ] + [
        {"missing_mode": "w/o well_log", "source_sample_index": str(index)} for index in range(2)
    ]

    kept, completed = recover_complete_mode_rows(
        rows,
        ("full", "w/o well_log"),
        samples_per_mode=3,
    )

    assert completed == ("full",)
    assert kept == rows[:3]
