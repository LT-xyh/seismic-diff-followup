# Experiment protocol

## Fixed data contract

The formal OpenFWI path uses the eight documented subsets and constructs aligned records with `depth_vel`, `migrated_image`, `horizon`, `rms_vel`, optional `well_log`, `well_mask`, modality availability, and modality quality. Data preparation must record source paths, counts, split membership, well locations, and input digests in a manifest.

## Seeds and split

- `training.seed = 2027`
- `data.split_seed = 42`
- `data.well_seed = 1234`
- Do not silently change a seed between runs.
- Every run receives a new output directory and records its resolved configuration hash.

## Evaluation

Use the common evaluator and report per-sample MAE, RMSE, SSIM, low-pass MAE, and high-pass MAE. Record the checkpoint, commit, config hash, evaluated record count, split, shuffle setting, and metric definitions. Do not compare source-native literature values as if they were matched-input results.

## Required comparison layers

1. Source-native literature context, clearly labeled as cross-protocol.
2. Matched-input baselines using identical observations, split, normalization, evaluator, and record identifiers.
3. Design attribution: full-field predictor, direct residual predictor, oracle low-pass background, predicted background, and prediction-consistent residual path where feasible.
4. Robustness: modality dropout and observation perturbation before making deployment claims.
5. Repeated runs: at least three seeds for the proposed method and strongest baselines before claiming stable ranking.

## Artifact contract

Each completed run must preserve:

- `git_commit`
- resolved config and `config_sha256`
- training/split/well seeds
- data manifest or source digest
- checkpoint path and `checkpoint_sha256`
- evaluated record IDs/counts
- metrics and aggregation definitions
- explicit status: smoke, full run, replicated, or blocked
