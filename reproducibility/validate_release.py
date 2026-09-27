"""Validate the source-only AAAI27 release surface and emit checklist evidence."""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CHECKLIST_ITEM_IDS = tuple(f"4.{index}" for index in range(2, 15))
CANONICAL_CONFIGS = (
    "base.yaml",
    "train.yaml",
    "smoke.yaml",
    "evaluate.yaml",
    "ablation.yaml",
)
REQUIRED_REQUIREMENTS = {
    "torch",
    "lightning",
    "tensorboard",
    "numpy",
    "scipy",
    "scikit-image",
    "omegaconf",
    "pyyaml",
    "lmdb",
    "pytest",
}
REQUIRED_RUN_MANIFEST_FIELDS = {
    "git_commit",
    "config_sha256",
    "checkpoint_sha256",
    "training_seed",
    "split_seed",
    "well_seed",
    "trajectory_count",
    "literature_transcribed",
    "split",
    "shuffle",
    "metric_names",
    "record_ids",
    "evaluated_records",
    "sample_counts",
}
FORBIDDEN_SOURCE_PARTS = {
    "_build_full",
    "_build_submission",
    "checkpoints",
    "logs",
    "release_outputs",
    "__pycache__",
}


class ReleaseValidationError(RuntimeError):
    """Raised when a source-only release check fails."""


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ReleaseValidationError(f"expected YAML mapping: {path}")
    return payload


def _relative_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _evidence_path_exists(value: object, root: Path) -> bool:
    if isinstance(value, str):
        return (root / value).is_file()
    if isinstance(value, list):
        return bool(value) and all(_evidence_path_exists(item, root) for item in value)
    return False


def _load_evidence(root: Path) -> dict[str, Any]:
    evidence = _load_yaml(root / "reproducibility" / "checklist_evidence.yaml")
    items = evidence.get("items")
    if not isinstance(items, list):
        raise ReleaseValidationError("checklist evidence items must be a list")
    by_id = {str(item.get("id")): item for item in items if isinstance(item, dict)}
    missing = [item_id for item_id in CHECKLIST_ITEM_IDS if item_id not in by_id]
    if missing:
        raise ReleaseValidationError(f"checklist evidence is missing items: {', '.join(missing)}")
    for item_id in CHECKLIST_ITEM_IDS:
        item = by_id[item_id]
        for field in ("status", "evidence_path", "command"):
            if not item.get(field):
                raise ReleaseValidationError(f"checklist evidence {item_id} lacks {field}")
        if item["status"] not in {"yes", "partial", "no", "conditional"}:
            raise ReleaseValidationError(f"unsupported checklist status for {item_id}: {item['status']!r}")
        if not _evidence_path_exists(item["evidence_path"], root):
            raise ReleaseValidationError(f"missing checklist evidence path for {item_id}: {item['evidence_path']!r}")
    publication_claims = evidence.get("publication_dependent")
    if not isinstance(publication_claims, list):
        raise ReleaseValidationError("publication_dependent evidence must be a list")
    required_claims = {"license", "public_availability", "literature_rerun"}
    actual_claims = {str(claim.get("claim")) for claim in publication_claims if isinstance(claim, dict)}
    if not required_claims.issubset(actual_claims):
        raise ReleaseValidationError("publication-dependent evidence must cover license, public_availability, and literature_rerun")
    for claim in publication_claims:
        if claim.get("status") != "conditional" or not _evidence_path_exists(claim.get("evidence_path"), root):
            raise ReleaseValidationError(f"publication claim is not explicitly conditional: {claim!r}")
    return evidence


def _validate_manifest(root: Path) -> dict[str, Any]:
    path = root / "reproducibility" / "release_manifest.yaml"
    manifest = _load_yaml(path)
    if manifest.get("version") != 1:
        raise ReleaseValidationError("release manifest version must be 1")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ReleaseValidationError("release manifest files must be a non-empty list")
    for value in files:
        if not isinstance(value, str):
            raise ReleaseValidationError(f"manifest path must be a string: {value!r}")
        path_parts = Path(value).parts
        if FORBIDDEN_SOURCE_PARTS.intersection(path_parts) or value.endswith(".local.yaml") or value.endswith(".pyc"):
            raise ReleaseValidationError(f"generated or local path in release manifest: {value}")
        if not (root / value).is_file():
            raise ReleaseValidationError(f"manifest source path is missing: {value}")
    entrypoints = manifest.get("entrypoints")
    if not isinstance(entrypoints, dict):
        raise ReleaseValidationError("release manifest entrypoints must be a mapping")
    for name, value in entrypoints.items():
        if not isinstance(value, str) or not (root / value).is_file():
            raise ReleaseValidationError(f"entrypoint {name!r} is not a file: {value!r}")
    external = manifest.get("external_sources", [])
    if not isinstance(external, list):
        raise ReleaseValidationError("external_sources must be a list")
    for source in external:
        if not isinstance(source, dict) or not (root / str(source.get("path", ""))).exists():
            raise ReleaseValidationError(f"external source path is missing: {source!r}")
    return manifest


def _validate_requirements(root: Path) -> None:
    path = root / "reproducibility" / "requirements.txt"
    if not path.is_file():
        raise ReleaseValidationError(f"missing dependency file: {path}")
    packages: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        name = line.split("==", 1)[0].split(">=", 1)[0].strip().lower()
        packages.add(name)
    missing = sorted(REQUIRED_REQUIREMENTS - packages)
    if missing:
        raise ReleaseValidationError(f"requirements.txt is missing pinned packages: {', '.join(missing)}")


def _load_config(path: Path) -> Any:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from bg_pdr_fm.training.benchmark_config import load_benchmark_config

    return load_benchmark_config(path)


def _validate_configs(root: Path) -> Any:
    config_dir = root / "bg_pdr_fm" / "configs" / "release" / "aaai27"
    for filename in CANONICAL_CONFIGS:
        if not (config_dir / filename).is_file():
            raise ReleaseValidationError(f"missing canonical config: {config_dir / filename}")
    from bg_pdr_fm.reproducibility import assert_release_paths
    from omegaconf import OmegaConf

    resolved: dict[str, Any] = {}
    for filename in CANONICAL_CONFIGS:
        conf = _load_config(config_dir / filename)
        assert_release_paths(conf, root)
        resolved[filename] = conf
    train = resolved["train.yaml"]
    datasets = list(OmegaConf.select(train, "data.openfwi_datasets", default=[]))
    if len(datasets) != 8:
        raise ReleaseValidationError(f"canonical train config must contain eight OpenFWI subsets, got {len(datasets)}")
    expected = {
        "data.split_fractions": [0.7, 0.2, 0.1],
        "data.split_seed": 42,
        "data.well_seed": 1234,
        "training.seed": 2027,
        "model.codec_type": "identity",
        "model.background_backend": "unet_direct",
        "model.background_hidden_channels": 192,
        "model.residual_backend": "unet_fm_film",
        "model.residual_backend_hidden_channels": 128,
        "model.residual_num_inference_steps": 50,
        "training.max_epochs": 100,
        "training.batch_size": 64,
        "training.precision": "bf16-mixed",
        "training.joint.lambda_contrastive": 0.01,
    }
    for key, expected_value in expected.items():
        actual = OmegaConf.select(train, key, default=None)
        if isinstance(expected_value, list):
            actual_value = list(actual) if actual is not None else None
        else:
            actual_value = actual
        if actual_value != expected_value:
            raise ReleaseValidationError(f"canonical train config mismatch for {key}: expected {expected_value!r}, got {actual!r}")
    return resolved


def _validate_entrypoint_help(root: Path, manifest: dict[str, Any]) -> None:
    entrypoints = manifest["entrypoints"]
    for name in ("prepare_openfwi", "train", "evaluate", "smoke", "validate"):
        script = entrypoints.get(name)
        if not isinstance(script, str):
            raise ReleaseValidationError(f"release manifest lacks {name} entrypoint")
        result = subprocess.run(
            [sys.executable, str(root / script), "--help"],
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
            timeout=60,
        )
        if result.returncode != 0:
            raise ReleaseValidationError(f"entrypoint {name} --help failed: {result.stderr.strip()}")


def _validate_imports(root: Path) -> None:
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    module_names = (
        "bg_pdr_fm.reproducibility",
        "bg_pdr_fm.training.benchmark_config",
        "bg_pdr_fm.evaluation.benchmark_metrics",
        "bg_pdr_fm.evaluation.run_manifest",
    )
    for module_name in module_names:
        try:
            importlib.import_module(module_name)
        except Exception as exc:  # pragma: no cover - error text is surfaced by the validator
            raise ReleaseValidationError(f"source import failed for {module_name}: {exc}") from exc


def _validate_metrics_and_run_manifest(root: Path, train_conf: Any) -> None:
    from bg_pdr_fm.evaluation.benchmark_metrics import METRIC_SCHEMA
    from bg_pdr_fm.evaluation.run_manifest import write_run_manifest

    if tuple(METRIC_SCHEMA) != ("mae", "rmse", "ssim", "mae_l", "mae_h"):
        raise ReleaseValidationError(f"metric schema drifted: {METRIC_SCHEMA!r}")
    with tempfile.TemporaryDirectory(prefix="aaai27-release-") as temp_dir:
        temp_root = Path(temp_dir)
        checkpoint = temp_root / "checkpoint.ckpt"
        checkpoint.write_bytes(b"synthetic release validator checkpoint")
        payload = write_run_manifest(
            temp_root,
            train_conf,
            [checkpoint],
            [{"dataset_id": 0, "dataset_name": "synthetic", "source_sample_index": 0}],
            trajectory_count=1,
            literature_transcribed=False,
        )
        missing = sorted(REQUIRED_RUN_MANIFEST_FIELDS - set(payload))
        if missing:
            raise ReleaseValidationError(f"run manifest is missing fields: {', '.join(missing)}")
        if tuple(payload["metric_names"]) != tuple(METRIC_SCHEMA) or payload["shuffle"] is not False:
            raise ReleaseValidationError("run manifest metric or shuffle contract is invalid")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _print_evidence_table(report: dict[str, Any]) -> None:
    print("checklist_id\tstatus\tevidence_path\tcommand")
    for item in report["items"]:
        evidence_path = item["evidence_path"]
        if isinstance(evidence_path, list):
            evidence_path = ",".join(str(path) for path in evidence_path)
        print(f"{item['id']}\t{item['status']}\t{evidence_path}\t{item['command']}")


def validate_release(
    *,
    repo_root: str | Path | None = None,
    config_path: str | Path | None = None,
    output: str | Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Run source-only checks and write a JSON checklist evidence report."""
    del force
    root = Path(repo_root).resolve() if repo_root is not None else REPO_ROOT
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    evidence = _load_evidence(root)
    manifest = _validate_manifest(root)
    _validate_requirements(root)
    configs = _validate_configs(root)
    if config_path is not None:
        requested_config = Path(config_path)
        if not requested_config.is_absolute():
            requested_config = root / requested_config
        if not requested_config.is_file():
            raise ReleaseValidationError(f"requested validation config is missing: {requested_config}")
        _load_config(requested_config)
    _validate_entrypoint_help(root, manifest)
    _validate_imports(root)
    _validate_metrics_and_run_manifest(root, configs["train.yaml"])

    checklist_path = root / "docs" / "paper" / "AAAI2027" / "ReproducibilityChecklist.tex"
    checklist_text = checklist_path.read_text(encoding="utf-8")
    answer_section = checklist_text.split("% The questions start here", 1)[-1]
    if "Type your response here" in answer_section:
        raise ReleaseValidationError("checklist answer section still contains instructional placeholder")

    report: dict[str, Any] = {
        "version": 1,
        "status": "pass",
        "manifest": _relative_path(root / "reproducibility" / "release_manifest.yaml", root),
        "items": evidence["items"],
        "repository_facts": evidence["repository_facts"],
        "publication_dependent": evidence["publication_dependent"],
        "checks": {
            "manifest": "pass",
            "canonical_configs": "pass",
            "requirements": "pass",
            "entrypoint_help": "pass",
            "source_imports": "pass",
            "metric_schema": "pass",
            "run_manifest_schema": "pass",
            "checklist_answers": "pass",
        },
    }
    output_path = Path(output) if output is not None else root / "reproducibility" / "generated" / "checklist_evidence.json"
    if not output_path.is_absolute():
        output_path = root / output_path
    _write_json(output_path, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="bg_pdr_fm/configs/release/aaai27/train.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = validate_release(config_path=args.config, output=args.output, force=args.force)
    except ReleaseValidationError as exc:
        print(f"release validation failed: {exc}", file=sys.stderr)
        return 1
    _print_evidence_table(report)
    print(json.dumps({"status": report["status"], "output": args.output or "reproducibility/generated/checklist_evidence.json"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
