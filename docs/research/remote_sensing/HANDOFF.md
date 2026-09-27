# Remote Sensing Handoff

Updated: 2026-09-27

## Objective

Turn the PD-BG-RFM research line into a focused Remote Sensing paper as efficiently as possible, with **formal acceptance notification** as the practical target.

The revision is not a rebuttal document and should not mechanically answer every AAAI comment. Reviewer feedback is used as evidence about where the previous narrative was vulnerable.

## Repository state

- AAAI research handoff source: `4b91f626`
- Remote Sensing evidence audit: `0bb320b`
- RS-0A source-handoff merge: `4c046a4a68dd4fa2d1b26a8283a68febf5fd26a2`
- Evidence audit:
  - `docs/research/REMOTE_SENSING_EVIDENCE_AUDIT.md`
  - `docs/research/REMOTE_SENSING_RESULT_MAP.csv`

### RS-0A source-handoff closure

**RS-0A = PASS / CLOSED.**

Root cause: the follow-up handoff repository omitted the canonical tracked `bg_pdr_fm/data/` package.

Authoritative source:
- `/public/home/xuyinghao/workspace/seismic-diff`
- source commit `75dcf1c5bb539aef97d4aaeadf01b54584108b09`

The canonical `bg_pdr_fm/data/` package is restored.

A second release-contract issue was also closed: the training/smoke path unconditionally constructs `TensorBoardLogger`, so the authoritative runtime dependency `tensorboard==2.20.0` is now pinned and enforced by the release validator. The retained environment does not use `tensorboardX`.

Verified on `main`:
- release validator: PASS;
- release tests: PASS;
- CPU smoke: PASS;
- GitHub Actions `quality`: PASS.

This history is internal reproducibility/project evidence and is not manuscript content.

### Immutable historical material

- `docs/paper/aaaii/`

Do not edit historical submission artifacts.

### Development material

- `bg_pdr_fm/`
- `scripts/aaai27/`
- `reproducibility/`
- `tests/`
- future Remote Sensing manuscript source
- this directory

## Strongest defensible scientific story

The paper should center on:

1. **Role-aware use of heterogeneous geophysical constraints.**
2. **Deterministic prediction of a background velocity component.**
3. **Prediction-relative residual Flow Matching.**
4. **Prediction consistency**: the same predicted background is used to define the residual target, condition residual transport, and compose the final velocity estimate.
5. **Compact reconstruction with strong structural fidelity under a common multimodal input contract.**

The broad claim that the method proves a universal reduction in "conditional complexity" is not required for the journal paper.

The historical name **PD-BG-RFM** remains valid for code and provenance. The paper-facing name is not frozen. Until final title selection, use **Background-Guided Residual Flow Matching (BG-RFM)** as a neutral working label. Do not make "physics-decoupled" the headline unless the manuscript gives a precise operational meaning that does not imply wave-equation or forward-operator decoupling.

## Current evidence status

The repository implementation, configs, evaluator, and historical manuscript are sufficiently organized to begin manuscript restructuring.

However, the final Remote Sensing quantitative package must not simply copy the AAAI tables.

Highest-priority evidence issue:

- Historical matched-input InversionNet reports `MAE=0.0149` and `MAE_L=0.0180`.
- This pair is incompatible with the current positive averaging-kernel low-pass metric under the same normalized error field.
- Exact historical provenance is unresolved.
- Therefore, the final matched-input table must come from one frozen evaluator + held-out manifest + checkpoint mapping.

Other historical results are useful as research assets, but their status must be determined from run manifests rather than from manuscript tables alone.

## Execution philosophy

Use a staged evidence strategy.

### Stage 0 — freeze the empirical contract

Before new training:

- inventory retained checkpoints, resolved configs, manifests, record IDs, and existing metric artifacts;
- identify the authoritative 33,600-record global held-out manifest used by the formal Table 2 tooling;
- freeze one evaluator and normalization contract only after asset inventory review;
- then re-evaluate available checkpoints under that common contract;
- resolve or replace the historical inconsistent frequency metrics.

### Stage 1 — cheap, high-information evaluation

Prefer evaluation-only work using existing checkpoints:

- direct background quality;
- common-evaluator matched-input comparison;
- existing attribution checkpoints;
- bounded RMS/horizon perturbations;
- inference-step/runtime sweep.

### Stage 2 — train only when the Stage 1 result leaves a claim unsupported

New training is allowed only when it strengthens a retained headline claim.

Examples:
- train a deterministic residual control only if no valid checkpoint exists;
- run additional seeds only if the final paper wants a stable ranking claim and the clean margin is small enough that seed variability could change the conclusion.

## Scope controls

Do not make the following default requirements for the first Remote Sensing submission:

- full 3D implementation;
- field-data transfer;
- downstream FWI;
- downstream migration;
- broad architecture search;
- large hyperparameter sweeps;
- extra baselines that do not test the paper's core claim.

They can become future work or later revision tasks if an editor/reviewer specifically requires them.

## Work-session responsibilities

### Main GPT work session

Own:
- paper narrative;
- claim/evidence mapping;
- repository code changes possible through GitHub;
- experiment specifications;
- manuscript restructuring;
- interpretation of results;
- decision whether a result belongs in the main paper, supplement, or nowhere.

### Codex runtime executor

Use only for:
- locating external checkpoints/logs on the real machine;
- GPU training;
- large evaluation jobs;
- timing/memory measurement on real hardware;
- environment-specific debugging.

Codex should return facts and artifacts, not decide the scientific story.

## Submission sprint mode

The project is now in **Remote Sensing submission sprint mode**.

Work proceeds in parallel:

### Track A — manuscript reconstruction

Start and maintain the new manuscript under:

`docs/paper/RemoteSensing/`

Do not wait for RS-E01-Lite before rewriting the non-numerical scientific narrative. The title, abstract structure, introduction, related work, method, observation contract, experimental protocol, discussion, conclusion, and figure/table architecture should be developed immediately.

### Track B — minimal evidence cleaning

The only mandatory pre-submission empirical task is **RS-E01-Lite**:

- existing checkpoints only;
- common 33,600-record held-out contract;
- one evaluator;
- one normalization contract;
- core metrics: MAE, RMSE, SSIM;
- parameter count as supporting evidence;
- frequency metrics only if the clean unified evaluation validates them.

Candidate main-table methods:
- BG-RFM / PD-BG-RFM;
- adapted InversionNet;
- adapted VelocityGAN;
- adapted Latent U-Net;
- adapted UPFWI only if it improves the scientific comparison without complicating interpretation.

Auto-Linear is optional.

RS-E02 through RS-E06 remain documented but are **deferred / reviewer-triggered** unless a retained core claim cannot be supported without them.

No new training is required before the first Remote Sensing submission.

## Immediate next action

Advance Track A immediately while finishing only the provenance work needed to launch RS-E01-Lite. Do not delay manuscript reconstruction for optional experiments.
