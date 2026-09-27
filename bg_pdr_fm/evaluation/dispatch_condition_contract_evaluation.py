"""Prepare and run leakage-free evaluations for the condition-contract protocol.

Training and evaluation are deliberately separate.  This dispatcher accepts a
completed protocol manifest, loads only its joint checkpoint, and then delegates
the fixed global-test evaluation to ``dispatch_parallel_eval``.  It refuses
prepared or partially completed runs so that a missing stage cannot silently
be replaced by an older checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from omegaconf import DictConfig, OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONDITION_ROOT = REPO_ROOT / "logs/bg_pdr_fm/aaai27/condition_contract"
DEFAULT_EVALUATION_ROOT = DEFAULT_CONDITION_ROOT / "evaluations"
DEFAULT_REGIMES = ("full", "no_contract")
DEFAULT_SEEDS = (2027, 3407, 7777)


def _resolve(path: str | Path) -> Path:
    item = Path(path)
    return item if item.is_absolute() else REPO_ROOT / item


def _clone(conf: DictConfig) -> DictConfig:
    return OmegaConf.create(OmegaConf.to_container(conf, resolve=False))


def build_evaluation_config(
    conf: DictConfig,
    *,
    regime: str,
    seed: int,
    checkpoint: str | Path,
    output_dir: str | Path,
) -> DictConfig:
    """Build an evaluation-only config from a completed joint-stage config."""
    if regime not in DEFAULT_REGIMES:
        raise ValueError(f"Unknown condition-contract regime {regime!r}.")
    result = _clone(conf)
    checkpoint_path = _resolve(checkpoint)
    output_path = _resolve(output_dir)
    OmegaConf.update(result, "training.stage", "joint_full", merge=True)
    OmegaConf.update(result, "training.seed", int(seed), merge=True)
    OmegaConf.update(result, "training.load_stage_checkpoint", None, merge=True)
    OmegaConf.update(
        result,
        "training.joint.warm_start_checkpoints",
        {"contrastive": None, "background": None, "residual": None},
        merge=True,
    )
    OmegaConf.update(result, "training.devices", 1, merge=True)
    OmegaConf.update(result, "training.strategy", None, merge=True)
    OmegaConf.update(result, "data.post_split_datasets", None, merge=False)
    OmegaConf.update(result, "evaluation.split", "test", merge=True)
    OmegaConf.update(result, "evaluation.output_dir", str(output_path), merge=True)
    OmegaConf.update(result, "evaluation.checkpoint", None, merge=True)
    OmegaConf.update(
        result,
        "evaluation.checkpoints",
        {
            "full": str(checkpoint_path),
            "contrastive": None,
            "background": None,
            "residual": None,
        },
        merge=True,
    )
    OmegaConf.update(result, "evaluation.max_batches", None, merge=True)
    OmegaConf.update(result, "evaluation.save_arrays", False, merge=True)
    OmegaConf.update(result, "evaluation.save_panels", False, merge=True)
    return result


def _manifest_checkpoint(manifest: Mapping[str, Any], *, regime: str, seed: int) -> Path:
    if str(manifest.get("regime")) != regime or int(manifest.get("seed", -1)) != int(seed):
        raise ValueError(f"Manifest identity does not match {regime}/seed_{seed}.")
    if str(manifest.get("status")) != "completed":
        raise RuntimeError(
            f"Cannot evaluate {regime}/seed_{seed}: manifest status is {manifest.get('status')!r}, expected 'completed'."
        )
    info = manifest.get("stage_checkpoints", {}).get("joint_full")
    if not isinstance(info, Mapping):
        raise KeyError(f"Manifest for {regime}/seed_{seed} has no joint_full checkpoint metadata.")
    path = _resolve(str(info.get("path", "")))
    if not bool(info.get("exists")) or not path.is_file():
        raise FileNotFoundError(f"Completed manifest points to a missing joint checkpoint: {path}")
    return path


def _metric_identities(path: str | Path, *, expected_samples: int) -> set[tuple[int, str, int]]:
    metric_path = _resolve(path)
    if not metric_path.is_file():
        raise FileNotFoundError(f"Missing metrics file: {metric_path}")
    with metric_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"dataset_id", "dataset_name", "source_sample_index"}
        missing = sorted(required.difference(reader.fieldnames or ()))
        if missing:
            raise KeyError(f"{metric_path} is missing identity fields: {missing}")
        identity_rows = [
            (int(row["dataset_id"]), str(row["dataset_name"]), int(row["source_sample_index"]))
            for row in reader
        ]
    identities = set(identity_rows)
    if len(identity_rows) != len(identities):
        raise ValueError(f"{metric_path} contains duplicate metric identities.")
    if len(identity_rows) != int(expected_samples):
        raise ValueError(
            f"{metric_path} contains {len(identity_rows)} metric identities; expected {expected_samples}."
        )
    return identities


def validate_paired_metric_identities(
    jobs: Sequence[Mapping[str, str | int]],
    *,
    expected_samples: int = 33_600,
) -> dict[str, Any]:
    """Require identical complete test identities for Full/no-contract pairs."""
    by_seed: dict[int, dict[str, set[tuple[int, str, int]]]] = {}
    for job in jobs:
        regime = str(job["regime"])
        seed = int(job["seed"])
        if regime not in DEFAULT_REGIMES:
            continue
        identities = _metric_identities(str(job["metrics"]), expected_samples=expected_samples)
        if regime in by_seed.setdefault(seed, {}):
            raise ValueError(f"Duplicate metric result for {regime}/seed_{seed}.")
        by_seed[seed][regime] = identities
    expected_regimes = set(DEFAULT_REGIMES)
    audit: dict[str, Any] = {}
    for seed, regimes in sorted(by_seed.items()):
        if set(regimes) != expected_regimes:
            raise ValueError(
                f"Seed {seed} must have both Full and no_contract metrics; got {sorted(regimes)}."
            )
        if regimes["full"] != regimes["no_contract"]:
            missing = sorted(regimes["full"] - regimes["no_contract"])[:5]
            unexpected = sorted(regimes["no_contract"] - regimes["full"])[:5]
            raise ValueError(
                f"Full/no-contract identity mismatch for seed {seed}; "
                f"missing={missing}, unexpected={unexpected}."
            )
        audit[str(seed)] = {"count": len(regimes["full"]), "identity_match": True}
    if not audit:
        raise ValueError("No paired Full/no-contract metric results were supplied.")
    return audit


def prepare_evaluation_configs(
    *,
    condition_root: str | Path = DEFAULT_CONDITION_ROOT,
    output_root: str | Path = DEFAULT_EVALUATION_ROOT,
    regimes: Sequence[str] = DEFAULT_REGIMES,
    seeds: Sequence[int] = DEFAULT_SEEDS,
) -> list[dict[str, str | int]]:
    """Create one evaluation config per completed regime/seed."""
    condition_root = _resolve(condition_root)
    output_root = _resolve(output_root)
    jobs: list[dict[str, str | int]] = []
    for regime in regimes:
        if regime not in DEFAULT_REGIMES:
            raise ValueError(f"Unknown condition-contract regime {regime!r}.")
        for seed in seeds:
            run_dir = condition_root / regime / f"seed_{int(seed)}"
            manifest_path = run_dir / "manifest.json"
            if not manifest_path.is_file():
                raise FileNotFoundError(f"Missing protocol manifest: {manifest_path}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            checkpoint = _manifest_checkpoint(manifest, regime=regime, seed=int(seed))
            stage_config = manifest.get("stage_configs", {}).get("joint_full")
            if not stage_config:
                raise KeyError(f"Manifest for {regime}/seed_{seed} has no joint_full config path.")
            stage_config_path = _resolve(str(stage_config))
            if not stage_config_path.is_file():
                raise FileNotFoundError(f"Missing joint-stage config: {stage_config_path}")
            evaluation_dir = output_root / regime / f"seed_{int(seed)}"
            evaluation_config_path = evaluation_dir / "eval_config.yaml"
            evaluation_config = build_evaluation_config(
                OmegaConf.load(stage_config_path),
                regime=regime,
                seed=int(seed),
                checkpoint=checkpoint,
                output_dir=evaluation_dir,
            )
            evaluation_dir.mkdir(parents=True, exist_ok=True)
            OmegaConf.save(evaluation_config, evaluation_config_path)
            jobs.append(
                {
                    "regime": regime,
                    "seed": int(seed),
                    "manifest": str(manifest_path),
                    "checkpoint": str(checkpoint),
                    "config": str(evaluation_config_path),
                    "output_dir": str(evaluation_dir),
                }
            )
    (output_root / "evaluation_queue_manifest.json").parent.mkdir(parents=True, exist_ok=True)
    (output_root / "evaluation_queue_manifest.json").write_text(
        json.dumps({"jobs": jobs}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return jobs


def run_evaluations(
    jobs: Sequence[Mapping[str, str | int]],
    *,
    gpus: str,
    batch_size: int = 64,
    expected_samples: int = 33_600,
    auto_idle: bool = False,
    python: str = sys.executable,
) -> list[dict[str, Any]]:
    """Run each evaluation sequentially and require a complete merged output."""
    results: list[dict[str, Any]] = []
    for job in jobs:
        output_dir = Path(str(job["output_dir"]))
        command = [
            python,
            "-m",
            "bg_pdr_fm.evaluation.dispatch_parallel_eval",
            "--config",
            str(job["config"]),
            "--output-dir",
            str(output_dir),
            "--gpus",
            str(gpus),
            "--batch-size",
            str(int(batch_size)),
            "--expected-samples",
            str(int(expected_samples)),
        ]
        if auto_idle:
            command.append("--auto-idle")
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = output_dir / "dispatch.log"
        with log_path.open("w", encoding="utf-8") as stream:
            stream.write("COMMAND: " + " ".join(command) + "\n")
            stream.flush()
            completed = subprocess.run(command, cwd=REPO_ROOT, stdout=stream, stderr=subprocess.STDOUT, check=False)
        metrics_path = output_dir / "merged_metrics.csv"
        summary_path = output_dir / "summary.json"
        if completed.returncode != 0 or not metrics_path.is_file() or not summary_path.is_file():
            raise RuntimeError(
                f"Condition-contract evaluation failed for {job['regime']}/seed_{job['seed']}; "
                f"see {log_path}."
            )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if int(summary.get("num_samples", -1)) != int(expected_samples):
            raise RuntimeError(
                f"Evaluation for {job['regime']}/seed_{job['seed']} returned "
                f"{summary.get('num_samples')} samples; expected {expected_samples}."
            )
        results.append({**dict(job), "metrics": str(metrics_path), "summary": str(summary_path)})
    validate_paired_metric_identities(results, expected_samples=expected_samples)
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition-root", type=Path, default=DEFAULT_CONDITION_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_EVALUATION_ROOT)
    parser.add_argument("--regimes", default=",".join(DEFAULT_REGIMES))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--gpus", default="4,5,6,7")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--expected-samples", type=int, default=33_600)
    parser.add_argument("--auto-idle", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    regimes = tuple(item.strip() for item in str(args.regimes).split(",") if item.strip())
    seeds = tuple(int(item.strip()) for item in str(args.seeds).split(",") if item.strip())
    jobs = prepare_evaluation_configs(
        condition_root=args.condition_root,
        output_root=args.output_root,
        regimes=regimes,
        seeds=seeds,
    )
    if args.prepare_only:
        return 0
    results = run_evaluations(
        jobs,
        gpus=args.gpus,
        batch_size=args.batch_size,
        expected_samples=args.expected_samples,
        auto_idle=args.auto_idle,
        python=args.python,
    )
    output_root = _resolve(args.output_root)
    paired_audit = validate_paired_metric_identities(results, expected_samples=args.expected_samples)
    (output_root / "evaluation_results.json").write_text(
        json.dumps({"jobs": results, "paired_identity_audit": paired_audit}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
