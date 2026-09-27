from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

from bg_pdr_fm.evaluation.benchmark_metrics import METRIC_SCHEMA, validate_metric_row
from bg_pdr_fm.evaluation import evaluate_bg_pdr_fm, run_manifest
from bg_pdr_fm.evaluation.evaluate_bg_pdr_fm import evaluation_record_identity
from bg_pdr_fm.evaluation.run_manifest import write_run_manifest
from bg_pdr_fm.reproducibility import resolved_config_hash


def _conf() -> object:
    return OmegaConf.create(
        {
            "data": {"split_seed": 42, "well_seed": 1234},
            "training": {"seed": 2027},
            "evaluation": {"split": "test", "expected_records": 2},
        }
    )


def _record_ids() -> list[dict[str, int | str]]:
    return [
        {"dataset_id": 0, "dataset_name": "FlatVelA", "source_sample_index": 7},
        {"dataset_id": 1, "dataset_name": "FlatVelB", "source_sample_index": 11},
    ]


def test_write_run_manifest_atomically_records_evaluation_provenance(tmp_path: Path) -> None:
    checkpoint = tmp_path / "joint_full.ckpt"
    checkpoint.write_bytes(b"checkpoint provenance")
    conf = _conf()

    manifest = write_run_manifest(
        tmp_path,
        conf,
        [checkpoint],
        _record_ids(),
        trajectory_count=1,
        literature_transcribed=False,
    )

    manifest_path = tmp_path / "run_manifest.json"
    saved = json.loads(manifest_path.read_text(encoding="utf-8"))
    git_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=True
    ).stdout.strip()

    assert saved == manifest
    assert manifest["git_commit"] == git_commit
    assert manifest["config_sha256"] == resolved_config_hash(conf)
    assert manifest["training_seed"] == 2027
    assert manifest["split_seed"] == 42
    assert manifest["well_seed"] == 1234
    assert manifest["checkpoint_sha256"] == [
        {"path": str(checkpoint), "sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
    ]
    assert manifest["trajectory_count"] == 1
    assert manifest["literature_transcribed"] is False
    assert manifest["split"] == "test"
    assert manifest["shuffle"] is False
    assert manifest["metric_names"] == list(METRIC_SCHEMA)
    assert manifest["record_ids"] == _record_ids()
    assert manifest["evaluated_records"] == 2
    assert manifest["sample_counts"] == {
        "expected_records": 2,
        "evaluated_records": 2,
        "metric_rows": 2,
    }
    assert not list(tmp_path.glob(".run_manifest.json.*.tmp"))


def test_captured_provenance_retains_loaded_checkpoint_hash_after_mutation(tmp_path: Path) -> None:
    checkpoint = tmp_path / "joint_full.ckpt"
    checkpoint.write_bytes(b"loaded checkpoint")
    expected_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    provenance_snapshot = run_manifest.capture_evaluation_provenance(_conf(), [checkpoint])

    checkpoint.write_bytes(b"checkpoint changed after loading")
    manifest = write_run_manifest(
        tmp_path,
        _conf(),
        [checkpoint],
        _record_ids(),
        trajectory_count=1,
        literature_transcribed=False,
        provenance_snapshot=provenance_snapshot,
    )

    assert manifest["checkpoint_sha256"] == [{"path": str(checkpoint), "sha256": expected_hash}]


def test_write_run_manifest_rejects_duplicate_record_identities(tmp_path: Path) -> None:
    duplicated = [_record_ids()[0], _record_ids()[0]]

    with pytest.raises(ValueError, match="duplicate"):
        write_run_manifest(
            tmp_path,
            _conf(),
            [],
            duplicated,
            trajectory_count=1,
            literature_transcribed=False,
        )

    assert not (tmp_path / "run_manifest.json").exists()


@pytest.mark.parametrize(
    ("record_ids", "expected_records", "message"),
    [
        (_record_ids()[:1], 2, "missing"),
        (_record_ids() + [_record_ids()[0] | {"source_sample_index": 13}], 2, "extra"),
    ],
)
def test_write_run_manifest_rejects_incomplete_or_extra_evaluation_intent(
    tmp_path: Path,
    record_ids: list[dict[str, int | str]],
    expected_records: int,
    message: str,
) -> None:
    conf = _conf()
    conf.evaluation.expected_records = expected_records

    with pytest.raises(ValueError, match=message):
        write_run_manifest(
            tmp_path,
            conf,
            [],
            record_ids,
            trajectory_count=1,
            literature_transcribed=False,
        )


def test_write_run_manifest_marks_capped_evaluation_count_unavailable(tmp_path: Path) -> None:
    manifest = write_run_manifest(
        tmp_path,
        _conf(),
        [],
        _record_ids()[:1],
        trajectory_count=1,
        literature_transcribed=False,
        expected_records=None,
    )

    assert manifest["sample_counts"] == {
        "expected_records": None,
        "evaluated_records": 1,
        "metric_rows": 1,
    }


@pytest.mark.parametrize("metric_name", METRIC_SCHEMA)
def test_validate_metric_row_rejects_non_finite_core_metrics(metric_name: str) -> None:
    row = {name: 0.1 for name in METRIC_SCHEMA}
    row[metric_name] = float("nan")

    with pytest.raises(ValueError, match=metric_name):
        validate_metric_row(row)


def test_emit_validated_evaluation_record_rejects_invalid_metric_before_output() -> None:
    row = {name: 0.1 for name in METRIC_SCHEMA}
    row["rmse"] = float("nan")
    emitted: list[str] = []
    seen: set[tuple[int, str, int]] = set()

    with pytest.raises(ValueError, match="rmse"):
        evaluate_bg_pdr_fm.emit_validated_evaluation_record(
            row,
            _record_ids()[0],
            seen,
            lambda: emitted.append("metrics.csv"),
        )

    assert emitted == []
    assert seen == set()


def test_emit_validated_evaluation_record_rejects_duplicate_before_output() -> None:
    row = {name: 0.1 for name in METRIC_SCHEMA}
    identity = _record_ids()[0]
    emitted: list[str] = []
    seen: set[tuple[int, str, int]] = set()

    evaluate_bg_pdr_fm.emit_validated_evaluation_record(
        row,
        identity,
        seen,
        lambda: emitted.append("first"),
    )
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_bg_pdr_fm.emit_validated_evaluation_record(
            row,
            identity,
            seen,
            lambda: emitted.append("duplicate"),
        )

    assert emitted == ["first"]


def test_synthetic_record_identity_uses_global_fallback_index() -> None:
    batch = SimpleNamespace(metadata={})

    first = evaluation_record_identity(
        batch,
        item_idx=0,
        dataset_names=[],
        fallback_source_sample_index=17,
    )
    second = evaluation_record_identity(
        batch,
        item_idx=1,
        dataset_names=[],
        fallback_source_sample_index=18,
    )

    assert first == {"dataset_id": -1, "dataset_name": "synthetic", "source_sample_index": 17}
    assert second == {"dataset_id": -1, "dataset_name": "synthetic", "source_sample_index": 18}
    assert first != second
