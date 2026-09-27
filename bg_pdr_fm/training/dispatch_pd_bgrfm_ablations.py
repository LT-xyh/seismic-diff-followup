"""Generate and queue fair PD-BG-RFM ablation runs on independent GPUs.

Each worker owns one GPU. This keeps the independent ablations from sharing
DDP synchronization and makes the wall-clock queue easy to inspect.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from bg_pdr_fm.runtime import configure_checkpoint_worker_environment, short_checkpoint_temp_dir
from bg_pdr_fm.training.benchmark_config import load_benchmark_config


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE = REPO_ROOT / "bg_pdr_fm/configs/openfwi_lmdb_joint_full_contrastive_bgfm_pixelfm_predbg_e100.yaml"
DEFAULT_CONTRASTIVE_CKPT = REPO_ROOT / "logs/bg_pdr_fm/contrastive_train/stage_checkpoints/contrastive_last.ckpt"


VARIANTS: dict[str, dict[str, Any]] = {
    "concat_fm": {"kind": "benchmark", "benchmark.variant": "concat_fm"},
    "decoupled_fm": {"kind": "benchmark", "benchmark.variant": "cncs_fm"},
    "wocontract_tether": {
        "kind": "main",
        "training.joint.lambda_contrastive": 0.0,
    },
    "woprediction_consistency": {
        "kind": "main",
        "training.ablation.residual_target_source": "lowpass",
    },
    "wobackground_context": {
        "kind": "main",
        "model.residual_background_context_source": "none",
        "model.residual_condition_merge": "add",
    },
    "direct_residual_predictor": {
        "kind": "main",
        "model.residual_backend": "direct_predictor",
    },
    "uniform_fusion": {
        "kind": "main",
        "model.fusion_mode": "uniform",
    },
}


def _set(conf: DictConfig, dotted_key: str, value: Any) -> None:
    OmegaConf.update(conf, dotted_key, value, merge=True)


def _common_paths(conf: DictConfig, run_dir: Path, name: str, epochs: int, batch_size: int,
                  accumulate_grad_batches: int) -> None:
    _set(conf, "training.max_epochs", int(epochs))
    _set(conf, "training.batch_size", int(batch_size))
    _set(conf, "training.accumulate_grad_batches", int(accumulate_grad_batches))
    _set(conf, "training.devices", 1)
    _set(conf, "training.strategy", "auto")
    _set(conf, "training.accelerator", "gpu")
    _set(conf, "training.num_workers", 8)
    _set(conf, "training.logging.log_dir", str(run_dir / "lightning"))
    _set(conf, "training.logging.log_version", name)
    _set(conf, "training.logging.add_timestamp", False)
    _set(conf, "training.checkpoint.dirpath", str(run_dir / "checkpoints"))
    _set(conf, "training.checkpoint.filename", f"{name}-epoch_{{epoch}}-loss{{val/loss:.4f}}")
    _set(conf, "training.checkpoint.save_last", True)
    _set(conf, "training.checkpoint.save_top_k", 3)
    _set(conf, "training.checkpoint.every_n_epochs", 1)
    # Keep atomic checkpoint writes and ML caches beside this run on /public.
    # A per-run directory prevents concurrent workers from accumulating stale
    # temporary files in one shared scratch directory.
    _set(conf, "training.checkpoint_tmp_dir", str(short_checkpoint_temp_dir(run_dir / "tmp")))
    _set(conf, "diagnostics.enabled", True)
    _set(conf, "diagnostics.output_dir", str(run_dir / "diagnostics"))
    _set(conf, "evaluation.output_dir", str(run_dir / "evaluation"))


def build_main_config(base: DictConfig, name: str, run_dir: Path, epochs: int, batch_size: int,
                      accumulate_grad_batches: int) -> DictConfig:
    conf = OmegaConf.create(OmegaConf.to_container(base, resolve=False))
    _common_paths(conf, run_dir, name, epochs, batch_size, accumulate_grad_batches)
    _set(conf, "training.stage", "joint_full")
    _set(conf, "model.residual_background_source", "predicted")
    return conf


def build_benchmark_config(name: str, run_dir: Path, epochs: int, batch_size: int,
                           accumulate_grad_batches: int, contrastive_ckpt: Path) -> DictConfig:
    base_path = REPO_ROOT / "bg_pdr_fm/configs/experiments/aaai27/_base_formal.yaml"
    conf = load_benchmark_config(str(base_path))
    _common_paths(conf, run_dir, name, epochs, batch_size, accumulate_grad_batches)
    spec = VARIANTS[name]
    _set(conf, "benchmark.variant", spec["benchmark.variant"])
    _set(conf, "benchmark.capacity_tier", "strong")
    _set(conf, "benchmark.generator_hidden_channels", 128)
    _set(conf, "benchmark.direct_hidden_channels", 128)
    _set(conf, "benchmark.full_field_backend", "unet_fm_film")
    _set(conf, "benchmark.full_field_background_context_source", "none")
    _set(conf, "benchmark.full_field_condition_merge", "add")
    _set(conf, "training.freeze_encoder", False)
    _set(conf, "training.load_stage_checkpoint", str(contrastive_ckpt))
    _set(conf, "training.checkpoint.monitor", "val/loss")
    _set(conf, "evaluation.checkpoint", str(run_dir / "checkpoints" / "last.ckpt"))
    _set(conf, "evaluation.output_dir", str(run_dir / "evaluation"))
    _set(conf, "model.codec_type", "identity")
    _set(conf, "model.latent_channels", 1)
    _set(conf, "model.latent_hw", [70, 70])
    _set(conf, "model.residual_backend", "unet_fm_film")
    _set(conf, "model.residual_backend_hidden_channels", 128)
    _set(conf, "model.residual_background_context_source", "none")
    _set(conf, "model.residual_condition_merge", "add")
    _set(conf, "model.residual_num_inference_steps", 50)
    return conf


def _write_config(conf: DictConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(conf, path)


def _command(name: str, config_path: Path) -> list[str]:
    if VARIANTS[name]["kind"] == "benchmark":
        return [sys.executable, "-m", "bg_pdr_fm.training.train_aaai27_benchmark", "--config", str(config_path)]
    return [sys.executable, "-m", "bg_pdr_fm.training.train_bg_pdr_fm", "--config", str(config_path)]


def _worker_env(gpu: str, scratch_dir: Path | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["HIP_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("PYTHONPATH", str(REPO_ROOT))
    env["PYTHONUNBUFFERED"] = "1"
    return configure_checkpoint_worker_environment(env, scratch_dir)


def _run_worker(name: str, gpu: str, config_path: Path, run_dir: Path) -> int:
    env = _worker_env(gpu, run_dir / "tmp")
    log_path = run_dir / "worker.log"
    state_path = run_dir / "worker_state.json"
    state = {
        "name": name,
        "gpu": gpu,
        "status": "running",
        "started_at": time.time(),
        "config": str(config_path),
        "scratch_dir": env.get("BG_PDR_FM_CHECKPOINT_TMPDIR", ""),
    }
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    with log_path.open("w", encoding="utf-8") as stream:
        stream.write(f"COMMAND: {' '.join(_command(name, config_path))}\n")
        stream.flush()
        result = subprocess.run(_command(name, config_path), cwd=REPO_ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
    state.update({"status": "completed" if result.returncode == 0 else "failed", "returncode": result.returncode,
                  "finished_at": time.time()})
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
    return int(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="4,5,6,7")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--accumulate-grad-batches", type=int, default=1)
    parser.add_argument("--output-root", default="logs/bg_pdr_fm/ablations/pd_bgrfm_current")
    parser.add_argument("--base-config", default=str(DEFAULT_BASE))
    parser.add_argument("--contrastive-ckpt", default=str(DEFAULT_CONTRASTIVE_CKPT))
    parser.add_argument("--variants", default=",")
    parser.add_argument("--limit-batches", type=int, default=None)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--resume-existing", action="store_true",
                        help="Resume each variant from its existing checkpoints/last.ckpt when available.")
    parser.add_argument("--resume-source-root", default=None,
                        help="Root containing the prior variant directories used for resume checkpoints.")
    args = parser.parse_args()

    gpus = [item.strip() for item in str(args.gpus).split(",") if item.strip()]
    names = list(VARIANTS) if args.variants in {"", ","} else [item.strip() for item in args.variants.split(",")]
    unknown = [name for name in names if name not in VARIANTS]
    if unknown:
        raise SystemExit(f"Unknown variants: {unknown}; available: {list(VARIANTS)}")

    base = OmegaConf.load(args.base_config)
    output_root = REPO_ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    jobs: list[dict[str, Any]] = []
    for name in names:
        run_dir = output_root / name
        if VARIANTS[name]["kind"] == "benchmark":
            conf = build_benchmark_config(
                name, run_dir, args.epochs, args.batch_size, args.accumulate_grad_batches,
                Path(args.contrastive_ckpt),
            )
        else:
            conf = build_main_config(
                base, name, run_dir, args.epochs, args.batch_size, args.accumulate_grad_batches,
            )
            for key, value in VARIANTS[name].items():
                if key != "kind":
                    _set(conf, key, value)
        if args.limit_batches is not None:
            _set(conf, "training.limit_train_batches", int(args.limit_batches))
            _set(conf, "training.limit_val_batches", int(args.limit_batches))
        if args.resume_existing:
            source_root = Path(args.resume_source_root) if args.resume_source_root else output_root
            resume_path = source_root / name / "checkpoints" / "last.ckpt"
            if resume_path.is_file():
                _set(conf, "training.ckpt_path", str(resume_path))
                if VARIANTS[name]["kind"] == "benchmark":
                    _set(conf, "training.partial_checkpoint_restore", True)
        config_path = run_dir / "run_config.yaml"
        _write_config(conf, config_path)
        jobs.append({"name": name, "config": str(config_path), "run_dir": str(run_dir)})

    (output_root / "queue_manifest.json").write_text(json.dumps({"gpus": gpus, "jobs": jobs}, indent=2), encoding="utf-8")
    if args.prepare_only:
        return
    pending = list(jobs)
    running: dict[str, tuple[subprocess.Popen[Any], dict[str, Any]]] = {}
    while pending or running:
        for gpu in gpus:
            if not pending or str(gpu) in running:
                continue
            job = pending.pop(0)
            run_dir = Path(job["run_dir"])
            run_dir.mkdir(parents=True, exist_ok=True)
            if (run_dir / "CANCELLED").exists():
                (run_dir / "worker_state.json").write_text(
                    json.dumps({
                        "name": job["name"],
                        "gpu": None,
                        "status": "skipped_cancelled",
                        "finished_at": time.time(),
                    }, indent=2),
                    encoding="utf-8",
                )
                continue
            env = _worker_env(gpu, Path(job["run_dir"]) / "tmp")
            log_path = run_dir / "worker.log"
            state_path = run_dir / "worker_state.json"
            command = _command(job["name"], Path(job["config"]))
            state_path.write_text(json.dumps({
                "name": job["name"],
                "gpu": str(gpu),
                "status": "running",
                "started_at": time.time(),
                "command": command,
                "scratch_dir": env.get("BG_PDR_FM_CHECKPOINT_TMPDIR", ""),
            }, indent=2), encoding="utf-8")
            stream = log_path.open("w", encoding="utf-8")
            stream.write("COMMAND: " + " ".join(command) + "\n")
            stream.flush()
            proc = subprocess.Popen(command, cwd=REPO_ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
            running[str(gpu)] = (proc, {"job": job, "stream": stream, "state_path": state_path})
        for gpu, (proc, info) in list(running.items()):
            code = proc.poll()
            if code is None:
                continue
            info["stream"].close()
            state = json.loads(info["state_path"].read_text(encoding="utf-8"))
            state.update({"status": "completed" if code == 0 else "failed", "returncode": code,
                          "finished_at": time.time()})
            info["state_path"].write_text(json.dumps(state, indent=2), encoding="utf-8")
            del running[gpu]
        time.sleep(10)


if __name__ == "__main__":
    main()
