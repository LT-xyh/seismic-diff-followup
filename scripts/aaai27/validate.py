"""Validate the AAAI27 release source surface and canonical configuration."""

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

    from reproducibility.validate_release import validate_release

    validate_release(config_path=config_path, output=args.output, force=bool(args.force))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
