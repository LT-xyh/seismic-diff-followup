"""Run the one-batch synthetic AAAI27 training and evaluation smoke path."""

from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf

from common import add_release_arguments, emit_dry_run, resolve_release_config


DEFAULT_CONFIG = "bg_pdr_fm/configs/release/aaai27/smoke.yaml"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_release_arguments(parser, default_config=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    config_path, conf = resolve_release_config(args)
    _use_synthetic_when_openfwi_is_unavailable(conf)
    if args.dry_run:
        emit_dry_run(config_path, conf)
        return 0

    from bg_pdr_fm.evaluation.evaluate_bg_pdr_fm import run_evaluation
    from bg_pdr_fm.training.train_bg_pdr_fm import main as train_main
    from bg_pdr_fm.training.train_bg_pdr_fm import run_one_stage

    if args.data_root or args.output:
        run_one_stage(conf)
    else:
        train_main(str(config_path))
    run_evaluation(conf)
    return 0


def _use_synthetic_when_openfwi_is_unavailable(conf: object) -> None:
    if str(OmegaConf.select(conf, "data.name", default="")).lower() != "openfwi":
        return
    root_dir = Path(str(OmegaConf.select(conf, "data.root_dir", default="")))
    if root_dir.is_dir():
        return
    OmegaConf.update(conf, "data.name", "synthetic", merge=True)
    OmegaConf.update(conf, "data.train_length", 2, merge=True)
    OmegaConf.update(conf, "data.val_length", 2, merge=True)


if __name__ == "__main__":
    raise SystemExit(main())
