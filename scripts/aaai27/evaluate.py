"""Evaluate a canonical AAAI27 checkpoint on the held-out test split."""

from __future__ import annotations

import argparse

from common import add_release_arguments, emit_dry_run, resolve_release_config


DEFAULT_CONFIG = "bg_pdr_fm/configs/release/aaai27/evaluate.yaml"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_release_arguments(parser, default_config=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    config_path, conf = resolve_release_config(args)
    if args.dry_run:
        emit_dry_run(config_path, conf)
        return 0

    from bg_pdr_fm.evaluation.evaluate_bg_pdr_fm import main as evaluate_main
    from bg_pdr_fm.evaluation.evaluate_bg_pdr_fm import run_evaluation

    if args.data_root or args.output:
        run_evaluation(conf)
    else:
        evaluate_main(str(config_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
