# AAAI Reproducibility Assets

The source inventory for this release is
`reproducibility/release_manifest.yaml`. It is deliberately small: it records
the project-owned implementation files needed for the PD-BG-RFM training and
evaluation path, rather than generated outputs from a particular machine.

## Scope

- `files` in the manifest are versioned project-owned source, configuration,
  documentation, and test inputs.
- `external_sources` records source-derived baseline components separately.
  They retain their upstream provenance and are not covered by a future project
  license merely because a bridge imports them.
- Data, LMDB stores, checkpoints, logs, rendered paper builds, and local
  override files are excluded from the source inventory.

The release entry points are thin scripts that resolve the canonical profile
without duplicating model or training logic:

```bash
python scripts/aaai27/prepare_openfwi.py --data-root data/openfwi --output data/openfwi_lmdb
python scripts/aaai27/train.py --config bg_pdr_fm/configs/release/aaai27/train.yaml
python scripts/aaai27/evaluate.py --config bg_pdr_fm/configs/release/aaai27/evaluate.yaml
python scripts/aaai27/smoke.py
```

Canonical relative-path AAAI profiles live under
`bg_pdr_fm/configs/release/aaai27/`:

- `train.yaml` is the 100-epoch, four-device paper profile and starts from
  random initialization.
- `smoke.yaml` keeps the same model/loss contract while using a one-batch,
  CPU synthetic check.
- `evaluate.yaml` requires a relative checkpoint path and traverses the test
  split without shuffling.
- `ablation.yaml` records the concat/full-field and prediction-consistency
  controls separately from the main profile.

Each wrapper accepts `--dry-run` to print the resolved profile and its hash
without launching Lightning or reading the OpenFWI data. See
`scripts/aaai27/README.md` for the complete command contract.

## Environment

Use Python 3.10.20 and install the pinned direct dependencies:

```bash
python -m pip install -r reproducibility/requirements.txt
```

The paper reports PyTorch 2.7.1 built with ROCm/HIP 6.3.25405. Select the
appropriate PyTorch wheel for the target CPU, CUDA, or ROCm platform before
installing the remaining requirements. The synthetic smoke path is intended to
run without the OpenFWI data or an accelerator.

## Data and Outputs

OpenFWI data and LMDB materializations are intentionally not committed. A
release profile must point to a local copy of the documented data layout and
must keep the split seed, well seed, and training seed explicit. Generated run
manifests belong under `reproducibility/generated/`; model outputs belong under
`release_outputs/`.

## License Gate

This repository does not yet declare a project license. A publication owner
must choose and add one before making a public archive. See
`reproducibility/THIRD_PARTY_NOTICES.md` before redistributing source-derived
baseline components.
