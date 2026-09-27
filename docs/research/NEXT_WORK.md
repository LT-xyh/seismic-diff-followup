# Next work

## P0 — Remote Sensing empirical freeze

- Recover the retained external checkpoints, manifests, configs, and metric artifacts required by RS-A0.
- Freeze one held-out manifest and one evaluator for the journal result package.
- Re-evaluate available matched-input checkpoints under the common contract.
- Resolve or supersede the historical InversionNet `MAE=0.0149`, `MAE_L=0.0180` inconsistency.

## P1 — high-information evaluation before new training

- Directly evaluate the predicted background against the defined low-pass target.
- Evaluate existing full-field / direct-residual / oracle-background / predicted-background / prediction-consistent artifacts under one evaluator.
- Run bounded RMS and horizon perturbation tests without retraining.
- Run a synchronized inference-step/runtime sweep.

## P2 — gated training

Only after P0/P1:

- Train missing attribution controls if existing artifacts cannot answer the retained claim.
- Run three-seed repetition only if the final manuscript makes a stable-ranking claim and the clean margin is small enough for seed variation to matter.

## P3 — manuscript revision

- Use `docs/research/remote_sensing/PAPER_STORY.md` as the narrative source of truth.
- Make the common matched-input table primary.
- Treat source-native literature results as context.
- Use PCA rank and target-energy ratios as diagnostics, not causal proof.
- Use the predicted-background reuse contract as the central method memory point.
- Keep 3D, field transfer, and downstream FWI/migration outside the first-pass required scope.

## P4 — submission readiness

- Run `docs/research/remote_sensing/REVIEW_CHECKLIST.md`.
- Compile the Remote Sensing manuscript and supplementary material.
- Perform one blind-review pass in a fresh GPT session that does not receive the project modification history.
