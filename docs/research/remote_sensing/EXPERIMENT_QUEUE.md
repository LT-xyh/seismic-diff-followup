# Remote Sensing Experiment Queue

## Submission sprint override

**Effective 2026-09-27.**

The project is in Remote Sensing submission sprint mode.

Only one empirical task is mandatory before the first submission:

### RS-E01-Lite — minimal unified re-evaluation

- existing checkpoints only;
- no training;
- common 33,600 held-out records;
- common evaluator and normalization;
- required metrics: MAE, RMSE, SSIM;
- parameter count may be reported as supporting efficiency evidence;
- frequency metrics are optional and only retained if internally consistent.

Primary candidate methods:
- BG-RFM / PD-BG-RFM;
- InversionNet adaptation;
- VelocityGAN adaptation;
- Latent U-Net adaptation.

UPFWI-adapted is optional and subject to PI decision after the unified evaluation. Auto-Linear is not required for the first submission.

RS-A0 remains a provenance/asset-preparation activity needed only to identify the exact retained checkpoints and common record contract for RS-E01-Lite. It does **not** block manuscript writing.

RS-E02, RS-E03, RS-E04, RS-E05, RS-E06, and extended missing-modality work below are retained as historical plans but are now **deferred / reviewer-triggered** unless the manuscript cannot support a retained core claim without them.

No new training is authorized for the first submission sprint.

---

The queue is intentionally staged to minimize unnecessary GPU work.

## Global rules

- Never overwrite historical runs.
- Every new run gets a new output directory.
- Freeze split membership and evaluator before new comparisons.
- Prefer evaluation of existing checkpoints before training new models.
- Every experiment must map to a retained claim in `CLAIM_EVIDENCE_MATRIX.md`.
- Main GPT defines protocol/code; Codex executes only when real runtime access is required.

---

## RS-A0 — Recover and freeze empirical assets

**Priority:** BLOCKER  
**Owner:** Main GPT designs retrieval specification; Codex performs server-side lookup if artifacts are outside GitHub.  
**Training:** none.

### Goal

Identify the exact retained checkpoints, resolved configs, manifests, and per-record metrics needed for the proposed method and adapted baselines.

### Minimum outputs

- checkpoint path + SHA256 where feasible;
- resolved config + hash;
- original git commit;
- held-out record manifest;
- record count;
- evaluator/version information;
- status of each candidate baseline artifact.

### Acceptance

At least one PD-BG-RFM checkpoint and the checkpoints needed for the clean comparison can be linked to immutable run metadata.

---

## RS-E01 — Common-evaluator matched-input reconstruction

**Priority:** REPLACED BY RS-E01-Lite for first submission  
**Owner:** GPT prepares/fixes evaluation code and result schema; Codex executes on retained checkpoints.  
**Training:** none if usable checkpoints exist.

### Goal

Replace the historical manuscript table with one clean result package under:

- one held-out manifest;
- one normalization;
- one evaluator;
- one set of metric definitions;
- deterministic test traversal.

### Candidate methods

Evaluate every usable retained matched-input checkpoint first. The manuscript can later present only the scientifically useful subset.

At minimum include:
- BG-RFM / PD-BG-RFM;
- InversionNet adaptation, because its historical frequency metric is inconsistent;
- the strongest one or two additional adapted references available under the same contract.

### Metrics

Core:
- MAE
- RMSE
- SSIM

Optional frequency metrics:
- MAE_L
- MAE_H

Frequency metrics enter the final paper only if they pass consistency validation and materially strengthen the story.

### Acceptance

- every result row has per-record metrics;
- counts match the frozen manifest;
- no train/val overlap;
- all method rows use the same evaluator;
- the historical InversionNet inconsistency is either resolved or superseded by the new run.

---

## RS-E02 — Direct background evaluation

**Priority:** DEFERRED / REVIEWER-TRIGGERED  
**Owner:** GPT defines evaluator output; Codex evaluates the retained proposed-method checkpoint.  
**Training:** none.

### Goal

Establish what the deterministic background branch actually predicts.

### Compare

`hat_B` vs. the exact background target used by the model, `P_L(V)`.

### Metrics

- background MAE;
- background RMSE if useful;
- low-pass error;
- high-frequency leakage / `epsilon_H_B`;
- per-subset summary.

### Main question

Does the background predictor reliably recover the smooth component it is explicitly trained to estimate?

Do not require it to beat every full-field model's final low-frequency error; this experiment evaluates the branch's assigned role.

---

## RS-E03 — Attribution controls

**Priority:** DEFERRED / REVIEWER-TRIGGERED  
**Owner:** GPT defines the comparison; Codex evaluates/trains only as required.

### Desired comparison

1. full-field deterministic predictor;
2. deterministic direct residual predictor;
3. oracle-background residual path;
4. predicted-background residual path;
5. full prediction-consistent BG-RFM.

### Stage A — existing checkpoint evaluation

Search and evaluate retained artifacts first.

### Stage B — minimal training

Train only controls that are both:
- missing after Stage A; and
- necessary to support C3.

### Acceptance

One common evaluator and data contract. The table must make it possible to separate:
- decomposition effect;
- predicted-background consistency effect;
- generative residual effect.

Not every control must appear in the main paper; final placement depends on the result.

---

## RS-E04 — Observation perturbation stress test

**Priority:** OPTIONAL / REVIEWER-TRIGGERED  
**Owner:** GPT implements perturbation logic; Codex runs evaluation.  
**Training:** none.

### Goal

Test deployment-relevant sensitivity without converting the paper into a robustness benchmark.

### Predeclared perturbation families

Use a small bounded grid.

#### RMS velocity

At least one amplitude-bias family, for example:
- clean;
- -5%;
- +5%;
- -10%;
- +10%.

If Gaussian perturbation is added, define its magnitude in normalized units before running and keep the grid small.

#### Horizon

Use deterministic vertical shifts, for example:
- clean;
- ±1 pixel;
- ±2 pixels;
- ±3 pixels if the first two levels are not informative.

### Output

MAE/RMSE/SSIM degradation relative to clean input.

### Paper role

Demonstrate behavior under plausible errors in interpreted/kinematic constraints; do not claim field robustness from this test alone.

---

## RS-E05 — Inference step/runtime tradeoff

**Priority:** OPTIONAL / REVIEWER-TRIGGERED  
**Owner:** GPT defines timing protocol; Codex measures on fixed hardware.  
**Training:** none.

### Suggested steps

- 10
- 20
- 50

Add 100 only if it clarifies convergence.

### Timing protocol

- same GPU;
- same batch size;
- warm-up iterations;
- synchronized device timing;
- report vector-field evaluations;
- report per-sample latency;
- peak memory optional but useful.

### Goal

Determine whether 10–20 steps retain most of the 50-step quality and quantify the actual cost of residual Flow Matching.

---

## RS-E06 — Multi-seed training stability

**Priority:** DEFERRED / REVIEWER-TRIGGERED  
**Owner:** GPT decides after RS-E01/RS-E03; Codex trains.  
**Training:** expensive.

### Trigger

Run only if:
- the final manuscript intends to claim stable ranking; and
- clean matched-input margins are small enough that training variability could plausibly reverse the conclusion.

### Minimum if triggered

- proposed method: three training seeds;
- strongest scientifically relevant matched-input reference: three training seeds;
- identical split/data manifest;
- new output directory per seed.

### Reporting

Mean and spread; paired per-record intervals can supplement but do not replace training-seed variation.

---

## Existing missing-modality evaluation

**Priority:** RECONCILE BEFORE RE-RUNNING

Historical six-mode evaluation exists. First identify which artifact version is authoritative. Re-run only if necessary to align with the frozen checkpoint/evaluator.

---

# Do not prioritize in the first pass

- 3D model conversion;
- field-data transfer;
- downstream FWI;
- downstream migration;
- large-scale solver search;
- broad hyperparameter tuning;
- additional baselines without a direct claim role.

---

# Execution order

`RS-A0 provenance check -> RS-E01-Lite -> submit; RS-E02–RS-E06 only if manuscript evidence fails or reviewers request them`

This ordering maximizes information gained before expensive training.
