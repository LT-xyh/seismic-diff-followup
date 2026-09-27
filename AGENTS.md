# Repository collaboration rules

## Read first

Before changing code or interpreting results, read `docs/research/START_HERE.md`, `PROJECT_STATE.md`, and `EXPERIMENT_PROTOCOL.md`.

## Evidence boundaries

- Treat committed source, configs, manifests, and test output as inspectable evidence.
- Do not claim that a training run, benchmark, seed repeat, or checkpoint exists unless a manifest or result artifact identifies it.
- Distinguish static code inspection, smoke validation, full training, and empirical acceptance.
- Preserve the AAAI submission snapshot under `docs/paper/aaaii/`; do not edit it during new work.
- Do not overwrite historical artifacts. Use a new run directory for every experiment.

## Canonical paths

Development uses `bg_pdr_fm/`, `scripts/aaai27/`, `reproducibility/`, `tests/`, and `docs/research/`. The supported release profile is `bg_pdr_fm/configs/release/aaai27/`.

## Safe changes

Run the release validator, CPU smoke path, and focused tests after changes. Keep data, checkpoints, logs, credentials, and machine-local paths out of commits. Update the evidence index and decision log when a paper claim, protocol, or acceptance status changes.
