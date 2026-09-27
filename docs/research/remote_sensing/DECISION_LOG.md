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
