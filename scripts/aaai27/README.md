# AAAI27 Release Commands

Run commands from the repository root with the pinned release environment.

```bash
python scripts/aaai27/prepare_openfwi.py --data-root data/openfwi --output data/openfwi_lmdb
python scripts/aaai27/train.py --config bg_pdr_fm/configs/release/aaai27/train.yaml
python scripts/aaai27/evaluate.py --config bg_pdr_fm/configs/release/aaai27/evaluate.yaml
python scripts/aaai27/smoke.py
python scripts/aaai27/validate.py --config bg_pdr_fm/configs/release/aaai27/train.yaml
```

`prepare_openfwi.py` expects each configured subset to contain `depth_vel`,
`migrated_image`, `horizon`, and `rms_vel` directories of aligned `.npy` files.
It writes LMDB manifests plus `aaai27_prepare_manifest.json`, recording source
hashes, counts, and the configured split/well seeds. Use `--force` only when an
existing LMDB store must be refreshed from the source files.

All commands accept `--dry-run` to resolve and print the selected configuration
without starting training, reading OpenFWI arrays, or loading a checkpoint.
`smoke.py` uses the synthetic one-batch profile, so it does not require OpenFWI
data or an accelerator.
