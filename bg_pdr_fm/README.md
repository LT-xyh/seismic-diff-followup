# BG-PDR-FM Engineering Notes

This package is the standalone training and evaluation surface for BG-PDR-FM.
It is intentionally separate from the older experimental workspace.

Code style and maintenance rules live in [`CODE_STYLE.md`](CODE_STYLE.md).

## AAAI27 Release Path

The paper-linked source release starts with the validator and synthetic smoke
check. These commands use only relative paths and do not require OpenFWI data
for the dry-run or smoke checks:

```bash
python reproducibility/validate_release.py
python scripts/aaai27/smoke.py
python scripts/aaai27/prepare_openfwi.py --data-root data/openfwi --output data/openfwi_lmdb
python scripts/aaai27/train.py --config bg_pdr_fm/configs/release/aaai27/train.yaml
python scripts/aaai27/evaluate.py --config bg_pdr_fm/configs/release/aaai27/evaluate.yaml
```

The validator writes `reproducibility/generated/checklist_evidence.json`.
Training and evaluation outputs are kept under `release_outputs/aaai27/`.
The full data layout, seed contract, and checkpoint requirements are listed in
[`reproducibility/README.md`](../reproducibility/README.md).

## Data Contract

Every dataset item and batch contains:

```text
depth_vel, migrated_image, horizon, rms_vel, well_log, well_mask,
modality_mask, modality_quality
```

Shape policy:

- `depth_vel`, `horizon`, `well_log`, and `well_mask` are depth-domain fields.
  The default training target is `(B, 1, 70, 70)`.
- `migrated_image` and `rms_vel` keep their dataset-native physical shape.
  For Marmousi and the current OpenFWI LMDB export this can be `(B, 1, 1000, 70)`.
- Dataset adapters must not silently resize time-domain inputs to `70 x 70`.
  If the model needs a shared grid, the projection must happen inside the model
  encoder or adapter layer.
- `well_log` is generated dynamically from raw `depth_vel`; it is not read from
  or written to disk.
- `well_mask` marks spatial well observations. It is not interchangeable with
  `modality_mask`, which only says whether a modality exists for the sample.

`target_shape` is a legacy/debug escape hatch. The default is `null`.

## Model Contract

The `bg_pdr_fm.models` modality-encoder stack is self-contained at runtime. The
encoder uses package-local physical modality encoders:

- `SeismicImageEncoderA` for PSTM migrated images, with anisotropic temporal and
  lateral texture processing.
- `HorizonEncoderA` for sparse horizon/interface masks.
- `RMSVelocityEncoderA` for RMS velocity trace sets.
- `WellLogEncoderA` for sparse trace-first well-log conditioning with explicit
  `well_mask` support.

Historical experiment files can remain in the repository, but they are not
imported by `bg_pdr_fm.models` at runtime. Native time/depth shapes are still
preserved by dataset adapters; each modality is encoded first and aligned to the
`depth_vel` grid inside `PhysicsDecoupledEncoder`.

## Configs

Available config templates:

```text
bg_pdr_fm/configs/bg_pdr_fm.yaml          # safe minimal default
bg_pdr_fm/configs/fast_run.yaml           # three-stage one-batch smoke run
bg_pdr_fm/configs/openfwi_lmdb_contrastive.yaml # OpenFWI contrastive stage
bg_pdr_fm/configs/openfwi_lmdb_background.yaml  # OpenFWI background stage
bg_pdr_fm/configs/openfwi_lmdb_residual.yaml    # OpenFWI residual stage
bg_pdr_fm/configs/openfwi_lmdb_train.yaml # legacy OpenFWI background alias
bg_pdr_fm/configs/marmousi_train.yaml     # Marmousi training template
bg_pdr_fm/configs/eval.yaml               # checkpoint-first evaluation template
bg_pdr_fm/configs/autoencoder_train.yaml  # formal seismic KL autoencoder training
bg_pdr_fm/configs/autoencoder_fast_run.yaml # one-batch autoencoder smoke test
bg_pdr_fm/configs/autoencoder_eval.yaml   # autoencoder reconstruction evaluation
```

The default split contract is controlled by:

```yaml
data:
  split_fractions: [0.7, 0.2, 0.1]
  split_seed: 42
```

OpenFWI uses a seeded random permutation for train/val/test membership.
Marmousi keeps its physical `test/` folder independent and creates train/val
from the physical `train/` folder.

## Internal Research and Legacy Commands

The commands below support historical experiments and exploratory components;
they are not the canonical AAAI27 release path.

Linux server migration and post-migration validation checklist:

```text
docs/linux_migration_validation.md
```

Autoencoder full training, no command-line config needed:

```powershell
D:\Conda\envs\seg\python.exe -m bg_pdr_fm.training.train_autoencoder
```

Autoencoder fast run:

```powershell
D:\Conda\envs\seg\python.exe -m bg_pdr_fm.training.train_autoencoder bg_pdr_fm/configs/autoencoder_fast_run.yaml
```

Autoencoder reconstruction evaluation:

```powershell
D:\Conda\envs\seg\python.exe -m bg_pdr_fm.evaluation.evaluate_autoencoder
```

The autoencoder trainer exports the best non-fast-run checkpoint to:

```text
checkpoints/bg_pdr_fm/seismic_autoencoder_kl.ckpt
```

That fixed checkpoint path is the handoff point consumed by downstream
`model.codec_type: autoencoder` runs.

Fast-run smoke test:

```powershell
D:\Conda\envs\seg\python.exe -m bg_pdr_fm.training.train_bg_pdr_fm --config bg_pdr_fm/configs/fast_run.yaml
```

OpenFWI LMDB staged training:

```powershell
D:\Conda\envs\seg\python.exe -m bg_pdr_fm.training.run_contrastive
D:\Conda\envs\seg\python.exe -m bg_pdr_fm.training.run_background
D:\Conda\envs\seg\python.exe -m bg_pdr_fm.training.run_residual
```

The stage launchers are thin wrappers around the shared trainer. They load
their YAML files and do not override batch size, epoch count, fast-run,
precision, or checkpoint paths.

Marmousi training:

```powershell
D:\Conda\envs\seg\python.exe -m bg_pdr_fm.training.train_bg_pdr_fm --config bg_pdr_fm/configs/marmousi_train.yaml
```

Evaluation:

```powershell
D:\Conda\envs\seg\python.exe -m bg_pdr_fm.evaluation.evaluate_bg_pdr_fm --config bg_pdr_fm/configs/eval.yaml
```

Evaluation is checkpoint-first by default. Set one of:

```yaml
evaluation:
  checkpoints:
    full: path/to/full.ckpt
```

or:

```yaml
evaluation:
  checkpoints:
    contrastive: logs/bg_pdr_fm/contrastive_train/stage_checkpoints/contrastive_last.ckpt
    background: logs/bg_pdr_fm/background_train/stage_checkpoints/background_last.ckpt
    residual: logs/bg_pdr_fm/residual_train/stage_checkpoints/residual_last.ckpt
```

Use `evaluation.allow_untrained: true` only for smoke tests.

## OpenFWI LMDB

OpenFWI source data should contain only the default physical modalities:

```text
depth_vel, migrated_image, horizon, rms_vel
```

`well_log` is never stored in LMDB. The recommended LMDB root is:

```text
I:/Datasets/openfwi_lmdb
```

Each OpenFWI subset has its own LMDB directory and `manifest.json`.
To add another subset later, build one new LMDB for that subset and append the
subset name under `data.openfwi_datasets`.

## Outputs

Lightning CSV/TensorBoard logs are written under each stage directory:

```text
logs/bg_pdr_fm/{contrastive_train,background_train,residual_train}/lightning
```

Diagnostics CSV/PNG are written under:

```text
logs/bg_pdr_fm/{contrastive_train,background_train,residual_train}/diagnostics
```

Stage handoff checkpoints are written under:

```text
logs/bg_pdr_fm/{contrastive_train,background_train,residual_train}/stage_checkpoints
```

Evaluation writes:

```text
summary.json
metrics.csv
predictions/*.npy
panels/*.png
```
