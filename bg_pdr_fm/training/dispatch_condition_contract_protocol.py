"""Prepare and run leakage-free condition-contract training protocols.

The main comparison removes the entire condition-contract supervision regime,
not merely the joint-stage tether.  Every stage checkpoint used by a run is
therefore produced under the same regime and model seed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from omegaconf import DictConfig, OmegaConf

from bg_pdr_fm.runtime import configure_checkpoint_worker_environment

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTRASTIVE_CONFIG = REPO_ROOT / "bg_pdr_fm/configs/openfwi_lmdb_contrastive.yaml"
DEFAULT_BACKGROUND_CONFIG = REPO_ROOT / "bg_pdr_fm/configs/openfwi_lmdb_background_unet_direct_h192_l1l2_e100.yaml"
DEFAULT_RESIDUAL_CONFIG = REPO_ROOT / "bg_pdr_fm/configs/openfwi_lmdb_residual_unet_direct_h192_pixelfm_film_predbg_e100.yaml"
DEFAULT_JOINT_CONFIG = REPO_ROOT / "bg_pdr_fm/configs/openfwi_lmdb_joint_full_contrastive_bgfm_pixelfm_predbg_e100.yaml"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "logs/bg_pdr_fm/aaai27/condition_contract"
DEFAULT_SEEDS = (2027, 3407, 7777)
REGIMES = ("full", "no_contract")


def _clone(conf: DictConfig) -> DictConfig:
    return OmegaConf.create(OmegaConf.to_container(conf, resolve=False))


def _set(conf: DictConfig, path: str, value: Any) -> None:
    OmegaConf.update(conf, path, value, merge=True)


def _checkpoint_path(stage_dir: Path, stage: str) -> Path:
    return stage_dir / "stage_checkpoints" / f"{stage}_last.ckpt"


def _prepare_common(
    conf: DictConfig,
    *,
    stage: str,
    seed: int,
    stage_dir: Path,
    epochs: int,
    batch_size: int,
    accumulate_grad_batches: int,
    devices: int,
) -> DictConfig:
    _set(conf, "training.stage", stage)
    _set(conf, "training.seed", int(seed))
    _set(conf, "training.max_epochs", int(epochs))
    _set(conf, "training.batch_size", int(batch_size))
    _set(conf, "training.accumulate_grad_batches", int(accumulate_grad_batches))
    _set(conf, "training.devices", int(devices))
    _set(conf, "training.strategy", "ddp" if int(devices) > 1 else "auto")
    _set(conf, "training.accelerator", "gpu")
    _set(conf, "training.load_stage_checkpoint", None)
    _set(conf, "training.stage_checkpoint_path", str(stage_dir / "stage_checkpoints"))
    _set(conf, "training.save_stage_outputs", True)
    _set(conf, "training.logging.log_dir", str(stage_dir / "lightning"))
    _set(conf, "training.logging.log_version", f"{stage}_seed{seed}")
    _set(conf, "training.logging.add_timestamp", False)
    _set(conf, "training.checkpoint.dirpath", str(stage_dir / "checkpoints"))
    _set(conf, "training.checkpoint.filename", f"{stage}-seed{seed}-epoch_{{epoch}}-loss{{val/loss:.4f}}")
    _set(conf, "training.checkpoint.save_last", True)
    _set(conf, "training.checkpoint.save_top_k", 3)
    _set(conf, "training.checkpoint.every_n_epochs", 10)
    _set(conf, "diagnostics.enabled", True)
    _set(conf, "diagnostics.output_dir", str(stage_dir / "diagnostics"))
    _set(conf, "evaluation.output_dir", str(stage_dir / "evaluation"))
    _set(conf, "evaluation.checkpoint", None)
    _set(
        conf,
        "evaluation.checkpoints",
        {"full": None, "contrastive": None, "background": None, "residual": None},
    )
    _set(
        conf,
        "training.joint.warm_start_checkpoints",
        {"contrastive": None, "background": None, "residual": None},
    )
    return conf


def build_stage_configs(
    *,
    regime: str,
    seed: int,
    run_dir: Path,
    contrastive_config: DictConfig,
    background_config: DictConfig,
    residual_config: DictConfig,
    joint_config: DictConfig,
    epochs: int,
    batch_size: int,
    accumulate_grad_batches: int,
    devices: int = 1,
) -> dict[str, DictConfig]:
    """Build a same-seed stage chain without reusing checkpoints across regimes."""
    if regime not in REGIMES:
        raise ValueError(f"Unknown regime {regime!r}; expected one of {REGIMES}.")

    stage_dirs = {stage: run_dir / stage for stage in ("contrastive", "background", "residual", "joint_full")}
    configs: dict[str, DictConfig] = {}

    if regime == "full":
        configs["contrastive"] = _prepare_common(
            _clone(contrastive_config),
            stage="contrastive",
            seed=seed,
            stage_dir=stage_dirs["contrastive"],
            epochs=epochs,
            batch_size=batch_size,
            accumulate_grad_batches=accumulate_grad_batches,
            devices=devices,
        )

    configs["background"] = _prepare_common(
        _clone(background_config),
        stage="background",
        seed=seed,
        stage_dir=stage_dirs["background"],
        epochs=epochs,
        batch_size=batch_size,
        accumulate_grad_batches=accumulate_grad_batches,
        devices=devices,
    )
    if regime == "full":
        _set(configs["background"], "training.load_stage_checkpoint", str(_checkpoint_path(stage_dirs["contrastive"], "contrastive")))

    configs["residual"] = _prepare_common(
        _clone(residual_config),
        stage="residual",
        seed=seed,
        stage_dir=stage_dirs["residual"],
        epochs=epochs,
        batch_size=batch_size,
        accumulate_grad_batches=accumulate_grad_batches,
        devices=devices,
    )
    _set(configs["residual"], "training.load_stage_checkpoint", str(_checkpoint_path(stage_dirs["background"], "background")))

    configs["joint_full"] = _prepare_common(
        _clone(joint_config),
        stage="joint_full",
        seed=seed,
        stage_dir=stage_dirs["joint_full"],
        epochs=epochs,
        batch_size=batch_size,
        accumulate_grad_batches=accumulate_grad_batches,
        devices=devices,
    )
    _set(
        configs["joint_full"],
        "training.joint.warm_start_checkpoints",
        {
            "contrastive": str(_checkpoint_path(stage_dirs["contrastive"], "contrastive")) if regime == "full" else None,
            "background": str(_checkpoint_path(stage_dirs["background"], "background")),
            "residual": str(_checkpoint_path(stage_dirs["residual"], "residual")),
        },
    )
    _set(configs["joint_full"], "training.joint.lambda_contrastive", 0.01 if regime == "full" else 0.0)
    return configs


def _write_config(conf: DictConfig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(conf, path)
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_metadata(path: Path) -> dict[str, Any]:
    payload: dict[str, Any] = {"path": str(path), "exists": path.is_file(), "sha256": None, "epoch": None}
    if not path.is_file():
        return payload
    payload["sha256"] = _sha256(path)
    try:
        import torch

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict):
            epoch = checkpoint.get("epoch")
            payload["epoch"] = None if epoch is None else int(epoch)
    except Exception as exc:  # Metadata remains useful for stage-only checkpoints.
        payload["metadata_error"] = f"{type(exc).__name__}: {exc}"
    return payload


def write_manifest(
    *,
    run_dir: Path,
    regime: str,
    seed: int,
    stage_configs: Mapping[str, Path],
    stage_checkpoints: Mapping[str, Path],
    sampler_seed: int,
    test_record_count: int,
    status: str = "prepared",
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "regime": regime,
        "seed": int(seed),
        "status": status,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sampler_seed": int(sampler_seed),
        "test_record_count": int(test_record_count),
        "stage_configs": {name: str(path) for name, path in stage_configs.items()},
        "stage_checkpoints": {name: _checkpoint_metadata(path) for name, path in stage_checkpoints.items()},
    }
    output = run_dir / "manifest.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output


def prepare_protocol(
    *,
    output_root: Path,
    regimes: tuple[str, ...],
    seeds: tuple[int, ...],
    contrastive_config: Path,
    background_config: Path,
    residual_config: Path,
    joint_config: Path,
    epochs: int,
    batch_size: int,
    accumulate_grad_batches: int,
    devices: int,
    sampler_seed: int = 1234,
    test_record_count: int = 33_600,
) -> list[dict[str, Any]]:
    base_configs = {
        "contrastive": OmegaConf.load(contrastive_config),
        "background": OmegaConf.load(background_config),
        "residual": OmegaConf.load(residual_config),
        "joint_full": OmegaConf.load(joint_config),
    }
    jobs: list[dict[str, Any]] = []
    for regime in regimes:
        for seed in seeds:
            run_dir = output_root / regime / f"seed_{seed}"
            configs = build_stage_configs(
                regime=regime,
                seed=seed,
                run_dir=run_dir,
                contrastive_config=base_configs["contrastive"],
                background_config=base_configs["background"],
                residual_config=base_configs["residual"],
                joint_config=base_configs["joint_full"],
                epochs=epochs,
                batch_size=batch_size,
                accumulate_grad_batches=accumulate_grad_batches,
                devices=devices,
            )
            config_paths = {stage: _write_config(conf, run_dir / "configs" / f"{stage}.yaml") for stage, conf in configs.items()}
            checkpoint_paths = {
                stage: _checkpoint_path(run_dir / stage, stage)
                for stage in configs
            }
            manifest = write_manifest(
                run_dir=run_dir,
                regime=regime,
                seed=seed,
                stage_configs=config_paths,
                stage_checkpoints=checkpoint_paths,
                sampler_seed=sampler_seed,
                test_record_count=test_record_count,
            )
            jobs.append({"regime": regime, "seed": seed, "run_dir": run_dir, "configs": config_paths, "manifest": manifest})
    queue_manifest = output_root / "queue_manifest.json"
    queue_manifest.parent.mkdir(parents=True, exist_ok=True)
    queue_manifest.write_text(
        json.dumps(
            {
                "regimes": list(regimes),
                "seeds": list(seeds),
                "jobs": [
                    {
                        "regime": job["regime"],
                        "seed": job["seed"],
                        "run_dir": str(job["run_dir"]),
                        "manifest": str(job["manifest"]),
                        "configs": {stage: str(path) for stage, path in job["configs"].items()},
                    }
                    for job in jobs
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return jobs


def _run_job(job: Mapping[str, Any], gpu_spec: str) -> int:
    run_dir = Path(job["run_dir"])
    manifest_path = Path(job["manifest"])
    if (run_dir / "CANCELLED").exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["status"] = "cancelled"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        return 0
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_spec
    env["HIP_VISIBLE_DEVICES"] = gpu_spec
    env.setdefault("PYTHONPATH", str(REPO_ROOT))
    env["PYTHONUNBUFFERED"] = "1"
    env = configure_checkpoint_worker_environment(env, run_dir / "tmp")
    stage_configs: Mapping[str, Path] = job["configs"]
    status = "completed"
    returncode = 0
    for stage, config in stage_configs.items():
        log_path = run_dir / stage / "worker.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "-m", "bg_pdr_fm.training.train_bg_pdr_fm", "--config", str(config)]
        with log_path.open("w", encoding="utf-8") as stream:
            stream.write("COMMAND: " + " ".join(command) + "\n")
            stream.flush()
            result = subprocess.run(command, cwd=REPO_ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        returncode = int(result.returncode)
        if returncode != 0:
            status = f"failed_{stage}"
            break
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = status
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    manifest["returncode"] = returncode
    manifest["stage_checkpoints"] = {
        stage: _checkpoint_metadata(Path(info["path"]))
        for stage, info in manifest["stage_checkpoints"].items()
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return returncode


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare or run the Figure 3 condition-contract protocol.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--regimes", default=",".join(REGIMES))
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--contrastive-config", type=Path, default=DEFAULT_CONTRASTIVE_CONFIG)
    parser.add_argument("--background-config", type=Path, default=DEFAULT_BACKGROUND_CONFIG)
    parser.add_argument("--residual-config", type=Path, default=DEFAULT_RESIDUAL_CONFIG)
    parser.add_argument("--joint-config", type=Path, default=DEFAULT_JOINT_CONFIG)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--accumulate-grad-batches", type=int, default=1)
    parser.add_argument("--gpus", default="4,5,6,7", help="Visible GPU ids for one DDP job, comma-separated.")
    parser.add_argument("--sampler-seed", type=int, default=1234)
    parser.add_argument("--test-record-count", type=int, default=33_600)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    regimes = tuple(item.strip() for item in str(args.regimes).split(",") if item.strip())
    unknown = sorted(set(regimes).difference(REGIMES))
    if unknown:
        raise SystemExit(f"Unknown regime(s): {unknown}; expected {REGIMES}.")
    seeds = tuple(int(item.strip()) for item in str(args.seeds).split(",") if item.strip())
    gpu_ids = [item.strip() for item in str(args.gpus).split(",") if item.strip()]
    if not gpu_ids:
        raise SystemExit("At least one GPU id is required.")
    jobs = prepare_protocol(
        output_root=args.output_root,
        regimes=regimes,
        seeds=seeds,
        contrastive_config=args.contrastive_config,
        background_config=args.background_config,
        residual_config=args.residual_config,
        joint_config=args.joint_config,
        epochs=args.epochs,
        batch_size=args.batch_size,
        accumulate_grad_batches=args.accumulate_grad_batches,
        devices=len(gpu_ids),
        sampler_seed=args.sampler_seed,
        test_record_count=args.test_record_count,
    )
    if args.prepare_only:
        return
    gpu_spec = ",".join(gpu_ids)
    for job in jobs:
        result = _run_job(job, gpu_spec)
        if result != 0:
            raise SystemExit(result)
        time.sleep(1)


if __name__ == "__main__":
    main()
