from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from bg_pdr_fm.evaluation.visualize_residual_transport import (
    SUBSET_ORDER,
    compute_q_t,
    plot_figure4,
    select_best_cases,
    shared_color_limits,
    validate_split_identity,
    validate_heldout_identity,
    validate_replay_metrics,
    _denormalize_velocity,
    _canonicalize_qt_rows,
    _summarize_qt,
)


def test_compute_q_t_matches_theorem_ratio_and_is_positive():
    target = np.zeros((2, 2), dtype=np.float32)
    background = np.ones((2, 2), dtype=np.float32)

    value = compute_q_t(target, background)

    assert np.isclose(value, 2.0)
    assert value > 0.0


def test_validate_replay_metrics_checks_composition_qt_and_mae():
    target = np.zeros((2, 2), dtype=np.float32)
    background = np.ones((2, 2), dtype=np.float32)
    final = np.full((2, 2), 0.5, dtype=np.float32)
    expected_qt = compute_q_t(target, background)

    result = validate_replay_metrics(
        target,
        background,
        final,
        {"q_T": str(expected_qt), "mae": "0.5"},
    )

    assert np.isclose(result["q_t_replay"], expected_qt)
    assert np.isclose(result["mae_replay"], 0.5)


def test_openfwi_velocity_denormalization_uses_physical_range():
    values = _denormalize_velocity(np.asarray([[-1.0, 0.0, 1.0]], dtype=np.float32), "openfwi")

    assert np.allclose(values, [[1500.0, 3000.0, 4500.0]])


def test_select_best_cases_uses_mae_then_rmse_ssim_and_source_index():
    rows = [
        {"dataset_name": "FlatVelA", "mae": "0.1", "rmse": "0.2", "ssim": "0.9", "source_sample_index": "4"},
        {"dataset_name": "FlatVelA", "mae": "0.1", "rmse": "0.1", "ssim": "0.8", "source_sample_index": "8"},
        {"dataset_name": "FlatVelA", "mae": "0.1", "rmse": "0.1", "ssim": "0.9", "source_sample_index": "3"},
    ]
    rows.extend(
        {
            "dataset_name": name,
            "mae": "0.5",
            "rmse": "0.5",
            "ssim": "0.5",
            "source_sample_index": str(index),
        }
        for index, name in enumerate(SUBSET_ORDER[1:], start=10)
    )

    selected = select_best_cases(rows)

    assert tuple(selected) == SUBSET_ORDER
    assert selected["FlatVelA"]["source_sample_index"] == "3"
    assert len(selected) == 8


def test_validate_heldout_identity_rejects_missing_or_duplicate_records(tmp_path):
    manifest = tmp_path / "held_out_manifest.csv"
    metrics = tmp_path / "merged_metrics.csv"
    fields = ["dataset_id", "dataset_name", "source_sample_index"]
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(
            [
                {"dataset_id": "0", "dataset_name": "FlatVelA", "source_sample_index": "1"},
                {"dataset_id": "0", "dataset_name": "FlatVelA", "source_sample_index": "1"},
            ]
        )
    with metrics.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({"dataset_id": "0", "dataset_name": "FlatVelA", "source_sample_index": "1"})

    try:
        validate_heldout_identity(manifest, metrics, expected_count=2)
    except ValueError as exc:
        assert "duplicate" in str(exc).lower()
    else:
        raise AssertionError("duplicate held-out identities must be rejected")


def test_validate_split_identity_checks_disjointness_and_manifest_membership():
    train = [{"dataset_id": "0", "dataset_name": "FlatVelA", "source_sample_index": "0"}]
    valid = [{"dataset_id": "0", "dataset_name": "FlatVelA", "source_sample_index": "1"}]
    test = [{"dataset_id": "0", "dataset_name": "FlatVelA", "source_sample_index": "2"}]
    manifest = [{"dataset_id": "0", "dataset_name": "FlatVelA", "source_sample_index": "2"}]

    audit = validate_split_identity(train, valid, test, manifest)

    assert audit["train_count"] == 1
    assert audit["val_count"] == 1
    assert audit["test_count"] == 1
    assert audit["overlap_train"] == 0
    assert audit["overlap_val"] == 0

    overlapping_val = [{"dataset_id": "0", "dataset_name": "FlatVelA", "source_sample_index": "2"}]
    try:
        validate_split_identity(train, overlapping_val, test, manifest)
    except ValueError as exc:
        assert "overlap" in str(exc).lower()
    else:
        raise AssertionError("train/val/test identity overlap must be rejected")


def test_shared_color_limits_are_global_across_selected_cases():
    cases = [
        {
            "target": np.asarray([[1500.0, 4500.0]]),
            "background": np.asarray([[1800.0, 4200.0]]),
            "target_residual": np.asarray([[-100.0, 200.0]]),
            "generated_residual": np.asarray([[-80.0, 180.0]]),
            "final": np.asarray([[1600.0, 4400.0]]),
            "absolute_error": np.asarray([[0.0, 50.0]]),
        },
        {
            "target": np.asarray([[1700.0, 4300.0]]),
            "background": np.asarray([[1900.0, 4100.0]]),
            "target_residual": np.asarray([[-300.0, 400.0]]),
            "generated_residual": np.asarray([[-250.0, 350.0]]),
            "final": np.asarray([[1750.0, 4250.0]]),
            "absolute_error": np.asarray([[10.0, 100.0]]),
        },
    ]

    limits = shared_color_limits(cases)

    assert limits["velocity"][0] == 1500.0
    assert limits["velocity"][1] == 4500.0
    assert limits["residual"][0] == -400.0
    assert limits["residual"][1] == 400.0
    assert limits["error"][0] == 0.0
    assert limits["error"][1] == 100.0


def test_qt_artifact_uses_theorem_spelling_and_records_all_subsets():
    rows = [
        {"dataset_name": name, "dataset_id": index, "source_sample_index": index, "q_t": 0.8}
        for index, name in enumerate(SUBSET_ORDER)
    ]
    canonical = _canonicalize_qt_rows(rows)
    assert all("q_T" in row and "q_t" not in row for row in canonical)

    subset_summary, global_summary = _summarize_qt(
        canonical,
        bootstrap_seed=2027,
        bootstrap_repeats=20,
    )
    assert len(subset_summary) == 8
    assert global_summary["n"] == 8
    assert global_summary["fraction_q_t_lt_1"] == 1.0


def test_plot_figure4_writes_all_publication_assets(tmp_path):
    tile = np.arange(16, dtype=np.float32).reshape(4, 4)
    best_cases = {
        name: {
            "dataset_name": name,
            "target": tile,
            "background": tile * 0.8,
            "target_residual": tile - tile * 0.8,
            "generated_residual": (tile - tile * 0.8) * 0.9,
            "final": tile * 0.98,
            "absolute_error": np.abs(tile * 0.02),
            "mae": 0.01,
            "rmse": 0.02,
            "ssim": 0.99,
            "q_t": 0.8,
            "background_mae": 0.02,
            "dataset_id": index,
            "source_sample_index": index,
        }
        for index, name in enumerate(SUBSET_ORDER)
    }
    qt_rows = [
        {"dataset_name": name, "q_t": 0.8 + 0.01 * index, "dataset_id": index, "source_sample_index": index}
        for index, name in enumerate(SUBSET_ORDER)
    ]

    outputs = plot_figure4(
        output_dir=tmp_path,
        best_cases=best_cases,
        qt_rows=qt_rows,
        metadata={"source": "synthetic-test"},
    )

    assert {path.suffix for path in outputs} == {".png", ".pdf", ".svg"}
    assert (tmp_path / "best_case_selection.csv").is_file()
    assert (tmp_path / "qt_samples.csv").is_file()
    assert (tmp_path / "qt_subset_summary.csv").is_file()
    assert (tmp_path / "metadata.json").is_file()
    with (tmp_path / "qt_samples.csv").open(newline="", encoding="utf-8") as handle:
        artifact_rows = list(csv.DictReader(handle))
    assert artifact_rows and "q_T" in artifact_rows[0]
    assert "q_t" not in artifact_rows[0]
