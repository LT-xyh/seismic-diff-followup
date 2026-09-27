"""Training entry point for AAAI27 benchmark variants."""

from __future__ import annotations

import argparse
from typing import Any

import lightning
from omegaconf import OmegaConf

from bg_pdr_fm.lightning import AAAI27BenchmarkLightning
from bg_pdr_fm.runtime import configure_checkpoint_temp_dir, configure_torch_runtime
from bg_pdr_fm.training.benchmark_config import load_benchmark_config
from bg_pdr_fm.training.train_bg_pdr_fm import apply_fast_run_overrides, build_loaders, build_trainer


def validate_benchmark_config(conf: Any) -> Any:
    variant = str(OmegaConf.select(conf, "benchmark.variant", default="")).lower()
    allowed = {
        "smooth_dix",
        "adapted_inversion_net",
        "adapted_upfwi",
        "sv_inv_net",
        "velocity_gan",
        "conditional_ddpm",
        "concat_fm",
        "cncs_fm",
        "bg_pdr_fm",
        "adapted_gfi",
        "adapted_gfi_latent_unet",
        "pdr_gfi_aligned_direct",
        "adapted_auto_linear",
        "adapted_auto_linear_original",
        # Legacy variants remain loadable for old checkpoints but are not part
        # of the paper-facing main-table runner.
        "mm_invnet",
        "two_stage_ddpm",
    }
    if variant not in allowed:
        raise ValueError(f"benchmark.variant must be one of {sorted(allowed)}, got {variant!r}.")
    tier = str(OmegaConf.select(conf, "benchmark.capacity_tier", default="matched")).lower()
    if tier not in {"matched", "strong", "reference", "adapted_external", "traditional"}:
        raise ValueError(
            "benchmark.capacity_tier must be matched, strong, reference, adapted_external, or traditional; "
            f"got {tier!r}."
        )
    return conf


def run_benchmark_config(conf: Any) -> lightning.Trainer:
    validate_benchmark_config(conf)
    configure_checkpoint_temp_dir(OmegaConf.select(conf, "training.checkpoint_tmp_dir", default=None))
    configure_torch_runtime(OmegaConf.select(conf, "training.matmul_precision", default=None))
    train_loader, val_loader = build_loaders(conf)
    model = AAAI27BenchmarkLightning(conf)
    if bool(OmegaConf.select(conf, "training.partial_checkpoint_restore", default=False)):
        # Older benchmark checkpoints contain frozen external-baseline modules
        # whose schemas may evolve independently of the active FM branch.
        # Lightning still restores every compatible active parameter and the
        # optimizer/epoch state from the checkpoint.
        model.strict_loading = False
    trainer, _ = build_trainer(conf)
    ckpt_path = OmegaConf.select(conf, "training.ckpt_path", default=None)
    trainer.fit(model, train_loader, val_loader, ckpt_path=str(ckpt_path) if ckpt_path else None)
    return trainer


def main(config_path: str, fast_run: bool | None = None) -> None:
    conf = load_benchmark_config(config_path)
    if fast_run is None:
        fast_run = bool(OmegaConf.select(conf, "training.fast_run", default=False))
    if fast_run:
        conf = apply_fast_run_overrides(conf)
    run_benchmark_config(conf)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an AAAI27 benchmark variant.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--fast-run", action="store_true")
    args = parser.parse_args()
    main(args.config, fast_run=True if args.fast_run else None)
