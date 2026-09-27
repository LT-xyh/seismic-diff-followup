# AAAI-27 Abstract Registration Draft

> STATUS: DRAFT ONLY. Do not submit before author confirmation.
> Source: `docs/paper/AAAI2027/paper.tex`

## Paper Information

### Title

Predicting the Background, Transporting the Structure: Physics-Decoupled Multimodal Subsurface Velocity Inversion

### TL;DR

PD-BG-RFM predicts well-constrained background velocity directly and uses residual Flow Matching for the remaining structural variation.

### Abstract

Multimodal subsurface velocity inversion combines complementary observations: post-stack migrated images and horizons reveal structural geometry, whereas RMS velocity and well logs calibrate the smooth background velocity and absolute scale. Existing methods fuse these cues with one full-field predictor, treating predictable background velocity and remaining structural variation as one task. We identify this mismatch as an asymmetry in conditional complexity and propose Physics-Decoupled Background-Guided Residual Flow Matching (PD-BG-RFM). Our method uses physics-decoupled contrastive learning to construct separate condition interfaces for background-velocity prediction and structural guidance. Its background branch produces a deterministic background-velocity prediction that is reused throughout the pipeline. This prediction defines the residual target during training, conditions the residual transport vector field together with the structural condition, and composes the final velocity estimate at inference. We derive an exact composition-consistency identity and prove a necessary-and-sufficient condition for residual transport to have lower straight-path target energy than full-field transport. On the globally held-out OpenFWI test set, PD-BG-RFM achieves the best RMSE, SSIM, and high-frequency MAE among matched-input baselines while using only 10.6M parameters. Together, controlled ablations and branch-specific diagnostics support a broader principle for multimodal inverse problems: predict well-constrained components directly and reserve generative transport for the structure-bearing variation left unexplained by the predicted background.

## Topics

### Primary Topic

APP: AI for Science (Natural & Physical Sciences)

### Secondary Topics

- ML: Deep Generative Models & Autoencoders
- ML: Representation Learning
- CV: Low-Level & Physics-based Vision
- APP: Natural Sciences

## Items Requiring Confirmation

### Authors and Order

1. `[First author: confirm OpenReview profile]`
2. `[Additional author: confirm only if substantial contribution]`
3. `[Last/corresponding author: advisor profile]`

### Countries of Institutions

- `[Confirm institution country for every author]`

### Reciprocal Reviewer

- `[Nominate one qualified author, or confirm that no author qualifies]`
- `[Do not fill this field without the nominee's consent]`

### Conflicts of Interest

- `[Confirm self-declared conflicts, if any]`

## Submission State

- Main PDF: not uploaded.
- Reproducibility checklist: not uploaded.
- Technical supplement: not uploaded.
- Code/data supplement: not uploaded.
- Submission action: intentionally not performed.
