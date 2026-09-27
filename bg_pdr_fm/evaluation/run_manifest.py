"""Atomic provenance manifests for checkpoint-based evaluation runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any

from omegaconf import OmegaConf

from bg_pdr_fm.evaluation.benchmark_metrics import METRIC_SAMPLE_COUNT_FIELDS, METRIC_SCHEMA
from bg_pdr_fm.reproducibility import resolved_config_hash


RECORD_ID_FIELDS = ("dataset_id", "dataset_name", "source_sample_index")
_UNSET = object()


def _conf_get(conf: Any, path: str, default: Any = None) -> Any:
    return OmegaConf.select(conf, path, default=default)


def _integer_identity_value(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"record identity {field} must be an integer")
    return int(value)


def record_identity_key(record_id: Mapping[str, Any]) -> tuple[int, str, int]:
    """Return the validated canonical key for one evaluated record."""
    if not isinstance(record_id, Mapping):
        raise ValueError("record identity must be a mapping")
    missing = [field for field in RECORD_ID_FIELDS if field not in record_id]
    if missing:
        raise ValueError(f"record identity is missing fields: {', '.join(missing)}")
    dataset_name = record_id["dataset_name"]
    if not isinstance(dataset_name, str):
        raise ValueError("record identity dataset_name must be a string")
    return (
        _integer_identity_value(record_id["dataset_id"], "dataset_id"),
        dataset_name,
        _integer_identity_value(record_id["source_sample_index"], "source_sample_index"),
    )


def _canonical_record_ids(record_ids: Sequence[Mapping[str, Any]]) -> list[dict[str, int | str]]:
    canonical: list[dict[str, int | str]] = []
    seen: set[tuple[int, str, int]] = set()
    for record_id in record_ids:
        identity = record_identity_key(record_id)
        if identity in seen:
            raise ValueError(f"duplicate record identity: {identity}")
        seen.add(identity)
        canonical.append(
            {
                "dataset_id": identity[0],
                "dataset_name": identity[1],
                "source_sample_index": identity[2],
            }
        )
    return canonical


def _expected_record_count(conf: Any, expected_records: object) -> int | None:
    configured_count = _conf_get(conf, "evaluation.expected_records", None)
    value = configured_count if expected_records is _UNSET else expected_records
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("expected_records must be a non-negative integer")
    try:
        count = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("expected_records must be a non-negative integer") from exc
    if count < 0:
        raise ValueError("expected_records must be a non-negative integer")
    return count


def _validate_record_count(record_ids: Sequence[Mapping[str, Any]], expected_records: int | None) -> None:
    if expected_records is None or len(record_ids) == expected_records:
        return
    if len(record_ids) < expected_records:
        raise ValueError(
            f"missing record identities for evaluation intent: expected {expected_records}, got {len(record_ids)}"
        )
    raise ValueError(
        f"extra record identities for evaluation intent: expected {expected_records}, got {len(record_ids)}"
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_hashes(checkpoint_paths: Sequence[str | Path]) -> list[dict[str, str]]:
    hashes: list[dict[str, str]] = []
    for checkpoint_path in checkpoint_paths:
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(f"evaluation checkpoint not found for manifest: {path}")
        hashes.append({"path": str(path), "sha256": _sha256(path)})
    return hashes


def capture_evaluation_provenance(
    conf: Any,
    checkpoint_paths: Sequence[str | Path],
) -> dict[str, Any]:
    """Capture code, config, and loaded-checkpoint identities before inference."""
    return {
        "git_commit": _git_commit(),
        "config_sha256": resolved_config_hash(conf),
        "checkpoint_sha256": _checkpoint_hashes(checkpoint_paths),
    }


def _provenance_fields(
    conf: Any,
    checkpoint_paths: Sequence[str | Path],
    provenance_snapshot: Mapping[str, Any] | None,
) -> dict[str, Any]:
    snapshot = capture_evaluation_provenance(conf, checkpoint_paths) if provenance_snapshot is None else provenance_snapshot
    try:
        checkpoint_hashes = [
            {"path": str(checkpoint["path"]), "sha256": str(checkpoint["sha256"])}
            for checkpoint in snapshot["checkpoint_sha256"]
        ]
        return {
            "git_commit": str(snapshot["git_commit"]),
            "config_sha256": str(snapshot["config_sha256"]),
            "checkpoint_sha256": checkpoint_hashes,
        }
    except (KeyError, TypeError) as exc:
        raise ValueError("provenance snapshot is missing required fields") from exc


def _repository_root() -> Path:
    source_root = Path(__file__).resolve().parents[2]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=source_root,
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError:
        return source_root
    if result.returncode == 0 and result.stdout.strip():
        return Path(result.stdout.strip())
    return source_root


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=_repository_root(),
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError:
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def write_run_manifest(
    output_dir: str | Path,
    conf: Any,
    checkpoint_paths: Sequence[str | Path],
    record_ids: Sequence[Mapping[str, Any]],
    *,
    trajectory_count: int,
    literature_transcribed: bool,
    expected_records: object = _UNSET,
    provenance_snapshot: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate and atomically write provenance for one evaluator invocation."""
    canonical_record_ids = _canonical_record_ids(record_ids)
    expected_count = _expected_record_count(conf, expected_records)
    _validate_record_count(canonical_record_ids, expected_count)
    if isinstance(trajectory_count, bool):
        raise ValueError("trajectory_count must be a positive integer")
    try:
        trajectory_count = int(trajectory_count)
    except (TypeError, ValueError) as exc:
        raise ValueError("trajectory_count must be a positive integer") from exc
    if trajectory_count < 1:
        raise ValueError("trajectory_count must be a positive integer")

    evaluated_records = len(canonical_record_ids)
    sample_counts = {
        "expected_records": expected_count,
        "evaluated_records": evaluated_records,
        "metric_rows": evaluated_records,
    }
    payload = {
        **_provenance_fields(conf, checkpoint_paths, provenance_snapshot),
        "training_seed": _conf_get(conf, "training.seed", None),
        "split_seed": _conf_get(conf, "data.split_seed", None),
        "well_seed": _conf_get(conf, "data.well_seed", None),
        "trajectory_count": trajectory_count,
        "literature_transcribed": bool(literature_transcribed),
        "split": str(_conf_get(conf, "evaluation.split", "test")),
        "shuffle": False,
        "metric_names": list(METRIC_SCHEMA),
        "record_ids": canonical_record_ids,
        "evaluated_records": evaluated_records,
        "sample_counts": {field: sample_counts[field] for field in METRIC_SAMPLE_COUNT_FIELDS},
    }
    _atomic_write_json(Path(output_dir) / "run_manifest.json", payload)
    return payload
