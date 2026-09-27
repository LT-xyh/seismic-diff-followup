"""Train the canonical AAAI27 PD-BG-RFM profile."""

from __future__ import annotations

import argparse

from common import add_release_arguments, emit_dry_run, resolve_release_config


DEFAULT_CONFIG = "bg_pdr_fm/configs/release/aaai27/train.yaml"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_release_arguments(parser, default_config=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    config_path, conf = resolve_release_config(args)
    if args.dry_run:
        emit_dry_run(config_path, conf)
        return 0

    from bg_pdr_fm.training.train_bg_pdr_fm import main as train_main
    from bg_pdr_fm.training.train_bg_pdr_fm import run_one_stage

    if args.data_root or args.output:
        run_one_stage(conf)
    else:
        train_main(str(config_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
