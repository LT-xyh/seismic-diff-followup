"""Build or verify OpenFWI LMDB inputs for the AAAI27 release profile."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf

from common import add_release_arguments, emit_dry_run, resolve_release_config


DEFAULT_CONFIG = "bg_pdr_fm/configs/release/aaai27/train.yaml"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_release_arguments(parser, default_config=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    config_path, conf = resolve_release_config(args, apply_output=False)
    if args.dry_run:
        emit_dry_run(config_path, conf)
        return 0

    from bg_pdr_fm.data.openfwi_lmdb import prepare_openfwi_lmdb_root

    source_root = Path(args.data_root or OmegaConf.select(conf, "data.root_dir"))
    lmdb_root = Path(args.output or OmegaConf.select(conf, "data.lmdb_root"))
    prepared = prepare_openfwi_lmdb_root(
        source_root=source_root,
        lmdb_root=lmdb_root,
        dataset_names=list(OmegaConf.select(conf, "data.openfwi_datasets")),
        split_seed=int(OmegaConf.select(conf, "data.split_seed")),
        well_seed=int(OmegaConf.select(conf, "data.well_seed")),
        overwrite=bool(args.force),
    )
    output_path = lmdb_root / "aaai27_prepare_manifest.json"
    output_path.write_text(json.dumps(prepared, indent=2, sort_keys=True), encoding="utf-8")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
