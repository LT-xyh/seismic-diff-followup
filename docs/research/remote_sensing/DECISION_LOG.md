# Decision Log

## 2026-09-27 — Journal target

Decision: **Remote Sensing** is the first journal target for the revised method paper.

Practical success criterion: formal acceptance notification.

Reasoning: the intended contribution fits an applied geoscience / remote-sensing-adjacent methodology framing, and a bounded journal revision is preferred over expanding the work back toward a broad AI-conference claim set.

## 2026-09-27 — Publication strategy

Decision: use the **scientific launch / publication-stage** principle.

Consequences:
- build the paper around its strongest evidence;
- do not narrate trial-and-error history;
- do not mechanically answer every prior reviewer comment;
- narrow claims before adding unnecessary experiments;
- every main-paper experiment must serve a specific claim.

Integrity constraint: no selective omission of facts required to substantiate a claim.

## 2026-09-27 — Core narrative

Decision: center the paper on:
- role-aware multi-source conditioning;
- deterministic background prediction;
- prediction-relative residual Flow Matching;
- prediction consistency through shared use of the predicted background.

Do not require the paper to prove universal "conditional complexity" reduction.

## 2026-09-27 — Naming

Decision: keep `PD-BG-RFM` as historical/code terminology for provenance.

For the journal manuscript, use `BG-RFM` as a neutral working shorthand until the final title/name is selected.

The term "physics-decoupled" must not imply an explicit wave-equation or forward-operator constraint.

## 2026-09-27 — Evidence audit

Audit commit: `0bb320b`.

Decision:
- begin manuscript restructuring now;
- do not copy the historical quantitative tables unchanged into the final submission;
- first freeze one held-out manifest/evaluator and re-evaluate retained checkpoints.

Key blocker: historical InversionNet `MAE=0.0149`, `MAE_L=0.0180` provenance is unresolved.

## 2026-09-27 — Compute strategy

Decision: conserve Codex quota.

- GPT handles research reasoning, GitHub code edits, paper writing, experiment design, and static audits.
- Codex is used only for tasks requiring the actual runtime environment, GPUs, private checkpoints/data/logs, or environment-specific debugging.

## 2026-09-27 — Multi-seed strategy

Decision: multi-seed repetition is **gated**, not an unconditional first action.

First obtain clean common-evaluator results.

Trigger three-seed training when the final paper intends a stable-ranking claim and the clean margin is narrow enough that seed variability materially affects the conclusion.

## 2026-09-27 — Scope

Not first-pass requirements:
- 3D implementation;
- field-data validation;
- downstream FWI;
- downstream migration.

These remain optional follow-up work unless later evidence or reviewer feedback makes them necessary.

## 2026-09-27 — RS-0A source handoff gate closed

Decision: **RS-0A = PASS / CLOSED**.

Root cause: the follow-up handoff repository omitted the canonical tracked `bg_pdr_fm/data/` package.

Authoritative source:
- workspace: `/public/home/xuyinghao/workspace/seismic-diff`
- source commit: `75dcf1c5bb539aef97d4aaeadf01b54584108b09`

Restored canonical package:
- `bg_pdr_fm/data/`

A second independent release issue was found: `TensorBoardLogger` is unconditionally constructed by the training/smoke path, but the release dependency contract did not declare a TensorBoard backend.

Authoritative environment:
- Python: `/public/home/xuyinghao/miniconda3/envs/seg/bin/python`
- `tensorboard==2.20.0`
- `tensorboardX`: not installed

Resolution:
- restored the canonical data package in `e37ef863`;
- declared `tensorboard==2.20.0`, added it to the validator dependency contract, and added regression coverage in `e8fcd641`;
- merged PR #1 with merge commit `4c046a4a68dd4fa2d1b26a8283a68febf5fd26a2`.

Verification on `main`:
- release validator: PASS;
- release tests: PASS;
- CPU smoke: PASS;
- GitHub Actions `quality`: PASS.

This debugging history is internal reproducibility evidence only and must not be turned into manuscript content.


## 2026-09-27 — Submission sprint mode

Decision: enter **Remote Sensing submission sprint mode**.

The publication goal is now to produce a focused, scientifically defensible manuscript as quickly as possible using the strongest existing evidence plus one minimal unified re-evaluation package.

Two tracks proceed in parallel:

- **Track A — Manuscript reconstruction:** rewrite the journal manuscript immediately. Title, abstract structure, introduction, related work, method, observation contract, experimental protocol, discussion, conclusion, and figure architecture do not wait for new quantitative results.
- **Track B — Minimal evidence cleaning:** the only mandatory pre-submission empirical task is **RS-E01-Lite**, using existing checkpoints only under one common held-out/evaluator contract.

Pre-submission experiment policy:

- no new training;
- no multi-seed campaign;
- no new robustness suite;
- no 3D extension;
- no field-transfer study;
- no downstream FWI/migration study;
- no expanded baseline search unless later reviewer feedback makes it necessary.

RS-E02 through RS-E06 are retained in the historical plan but are now **optional / reviewer-triggered** unless the manuscript cannot support a retained core claim without one of them.

The main quantitative table may be limited to MAE, RMSE, SSIM, and parameter count. Frequency metrics are optional and must not become central unless RS-E01-Lite validates them cleanly.

The working journal title is:

> **Background-Guided Residual Flow Matching for Multi-Constraint Seismic Velocity Model Building**

The manuscript-facing working name is **BG-RFM**. Historical `PD-BG-RFM` terminology remains valid for code and provenance.


## 2026-09-28 — Final manuscript production stage

Decision: enter **final Remote Sensing manuscript production**.

The next milestone is a complete manuscript suitable for blind review, not another experiment.

Evidence policy:
- the AAAI manuscript remains the scientific source manuscript unless a specific result is known to be incorrect;
- RS-E01-Lite is the authoritative unified main comparison and an additional clean evaluation asset;
- historical ablations, qualitative results, missing-modality tests, PCA/effective-rank analyses, condition diagnostics, residual/transport diagnostics, architecture details, and supplementary implementation material may be reused when they strengthen the retained paper story;
- missing modern `run_manifest.json` provenance alone is not a reason to discard an otherwise valid historical experiment;
- the known suspicious historical InversionNet `MAE_L=0.0180` value remains excluded unless independently verified.

Submission-blocking experiments: **none**.

Do not request new training, multi-seed experiments, robustness experiments, additional baselines, 3D/field/downstream FWI experiments, checkpoint reconstruction, or open-ended historical artifact archaeology before submission.

Historical diagnostics are interpreted as empirical characterization, not formal causal proof:
- PCA/effective rank -> data-level characterization;
- transport-energy analysis -> transport diagnostic;
- missing-modality tests -> supporting stress evidence;
- historical ablations -> design evidence without overclaiming causal isolation.

Main manuscript emphasis:
1. role-aware use of heterogeneous constraints;
2. deterministic background estimation;
3. prediction-relative residual Flow Matching;
4. prediction consistency through reuse of the same predicted background;
5. compact parameterization and supporting empirical evidence.

Do not claim universal accuracy dominance or faster inference.
