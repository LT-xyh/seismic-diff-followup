# AAAI Code Release Surface Design

**Status:** Design approved for implementation by the user (方案 1).

**Goal:** Prepare a paper-driven, checklist-backed code release for the AAAI submission while preserving the complete pre-cleanup workspace in commit `74fdd6146c6ae5a21c2a5bf6a134d3cb9ef3c793`.

## Scope

The release surface covers the code needed to reproduce the claims and reported protocols in `docs/paper/AAAI2027/paper.tex`:

- the physics-decoupled condition encoder and role-aware fusion;
- deterministic background prediction and prediction-consistent residual Flow Matching;
- OpenFWI preprocessing, split construction, well sampling, normalization, and missing-modality masks;
- the canonical PD-BG-RFM training and evaluation path;
- the matched-input baseline adapters and controlled ablation entry points named by the paper;
- the evaluator, metrics, held-out manifest checks, diagnostics used in the paper, and synthetic smoke tests.

Historical experiment queues, exploratory model variants, one-off plotting/export scripts, generated figures, build products, logs, checkpoints, and machine-local configuration files are not canonical release inputs. They remain recoverable through the checkpoint commit and are either excluded by the release manifest or explicitly labeled as internal research material.

The model equations and paper claims remain the source of truth. The cleanup may fix implementation/configuration drift required to make the canonical path execute the equations, but it does not silently change reported numbers or strengthen manuscript claims without a corresponding evidence artifact.

## Architecture

The repository keeps `bg_pdr_fm` as the importable package and adds a small, explicit release layer rather than duplicating model logic:

1. `bg_pdr_fm/data`, `bg_pdr_fm/models`, `bg_pdr_fm/lightning`, and `bg_pdr_fm/evaluation` remain the single implementation path.
2. A canonical AAAI configuration family under `bg_pdr_fm/configs/release/aaai27/` states every final parameter, split/seed, device, precision, checkpoint, and output setting. It has separate train, smoke, and evaluation profiles and contains no absolute machine-local paths.
3. Thin command-line wrappers under `scripts/aaai27/` call the existing package APIs for data preparation, training, evaluation, and release validation. They accept a config path and do not override hyperparameters.
4. `reproducibility/` contains the dependency/environment specification, a release manifest schema, and a validator that checks required files, config resolution, data-contract metadata, seed handling, metric names, and generated run manifests.
5. `bg_pdr_fm/tests/` gains focused release-contract tests. The existing broad smoke suite remains available, but the release validator provides a fast, deterministic gate for a clean checkout without the OpenFWI data.
6. The AAAI paper build is made self-contained again by restoring/verifying the submission wrapper referenced by `docs/paper/AAAI2027/build.sh`; build outputs remain ignored and are never used as source inputs.

The release manifest records the distinction between project-owned code and external baseline bridges. External implementations keep their source/provenance notes and are not relicensed by the project license.

## Data Flow

The canonical path follows the paper's contract:

```text
OpenFWI raw/LMDB records
  -> deterministic 7/2/1 split (seed 42)
  -> dynamic well observations (well seed 1234)
  -> native-shape modality adapters and availability/quality masks
  -> physics-decoupled encoder
  -> C_bg and C_str
  -> B_hat = f_bg(C_bg)
  -> R_hatB = V - stop_gradient(B_hat)
  -> residual Flow Matching conditioned on (C_str, psi_bg(B_hat))
  -> B_hat + R_hat at inference
  -> per-sample MAE/RMSE/SSIM/MAE_L/MAE_H and run manifest
```

Training-only Haar anchors and target-derived diagnostics are created inside the training/diagnostic path and are rejected by the inference contract. The validator and tests explicitly check that inference receives observed modalities only and that no target, anchor, or oracle residual is routed into prediction.

## Checklist Contract

The implementation will produce direct evidence for the applicable checklist items:

| Checklist item | Release evidence |
| --- | --- |
| Pre-processing code (4.2) | Versioned OpenFWI preparation/manifest command and data-contract documentation |
| All experiment source (4.3) | Canonical manifest covering train, evaluate, metrics, baselines, ablations, and diagnostics |
| Public source release (4.4) | Project license plus third-party notices; the final license choice is an explicit release gate |
| Method comments/references (4.5) | Equation/section references in canonical modules and a code-to-paper map |
| Randomness (4.6) | One seed entry point covering Python, NumPy, PyTorch, workers, split membership, and well generation; config records the seed |
| Infrastructure (4.7) | Pinned environment file, version probe, hardware/software capture, and documented CPU fallback/smoke path |
| Metrics (4.8) | One evaluator implementation with definitions, finite-value checks, and metric schema tests |
| Run counts (4.9) | Per-result run manifest recording trajectory count, checkpoint identity, evaluated-record count, and whether a row is literature-transcribed |
| Distributional evidence (4.10) | Existing diagnostic/CI artifacts are referenced by manifest without treating a mean as a significance claim |
| Final parameters (4.14) | Resolved canonical YAML plus a machine-readable parameter snapshot |

Items that depend on publication logistics or historical facts remain evidence-bounded. In particular, the code cannot by itself prove that an external repository will be public at publication time, that a specific legal license is authorized, or that literature rows were independently rerun. Those statements must remain accurately qualified in the checklist and manuscript.

## Error Handling

- Missing data roots, manifests, checkpoints, or required modalities fail early with the path, dataset, split, and expected contract in the error message.
- A training or evaluation command refuses to run a formal profile when its resolved config contains an absolute local path, an unstated seed, an oracle/target input, or an untracked checkpoint reference.
- Evaluation rejects duplicate, missing, extra, non-finite, or out-of-split record identities before aggregating metrics.
- Release validation exits nonzero for missing source files, unresolved imports, unsupported dependency versions, invalid metric schemas, or a stale generated manifest.
- Optional real-data tests skip only when the documented dataset root is absent; synthetic smoke tests remain mandatory and must pass.

## Verification

The success gate is reproducible from a clean checkout of the release branch:

1. `python -m compileall -q bg_pdr_fm scripts/aaai27` succeeds.
2. The release validator reports a complete manifest and no local-path or target-leakage violations.
3. The synthetic one-batch training/evaluation smoke path produces finite metrics and a run manifest with the expected schema.
4. Focused release-contract tests pass, including seed replay, split/well determinism, config resolution, checkpoint provenance, inference input restrictions, and metric definitions.
5. When the OpenFWI root is supplied, the held-out manifest validator confirms 33,600 unique test records with no split overlap and the evaluator traverses them with `shuffle=False`.
6. `docs/paper/AAAI2027/build.sh full` and `build.sh submission` both resolve their source wrapper and compile when the documented TeX toolchain is present.
7. Every changed method/config line is traceable to the paper or to one checklist requirement; generated assets and unrelated legacy code are not reformatted opportunistically.

## Explicit Boundaries

- The checkpoint commit is an immutable recovery point; cleanup commits on `codex/aaai-code-release` are separate and reviewable.
- No model result is re-labeled as independently replicated merely because a deterministic release seed is added.
- No third-party baseline is copied into the project license without its provenance and license notice.
- The final public-license choice and the exact publication archive scope are release gates; until confirmed, the checklist must not claim stronger availability than the repository can support.
