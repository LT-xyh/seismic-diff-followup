# Paper Story

Status: working narrative for Remote Sensing.

## Working title

**Background-Guided Residual Flow Matching for Multi-Constraint Seismic Velocity Model Building**

Alternative if role-aware conditioning remains a strong empirical contribution:

**Role-Aware Background-Guided Residual Flow Matching for Multi-Constraint Seismic Velocity Model Building**

Do not freeze the title until the clean evidence package is available.

## One-sentence thesis

Multi-source geophysical constraints can be used more effectively for initial velocity model building when the model explicitly predicts a background velocity component and uses that same prediction as the reference and context for residual Flow Matching, instead of assigning the entire velocity field to a single undifferentiated estimator.

## Problem

Initial velocity model building under post-stack workflows uses heterogeneous information with different geophysical roles:

- RMS velocity provides smooth kinematic and absolute-scale information;
- migrated images and horizons provide structural geometry;
- sparse wells provide local velocity anchors.

A learning system should exploit these complementary constraints without forcing one predictor to carry the entire reconstruction responsibility.

## Gap

Most learning-based formulations assemble observations and then predict or generate the full velocity field through a single estimator. This does not explicitly assign background recovery and remaining correction to different modeling responsibilities.

The journal paper should focus on this **responsibility-allocation gap**, not on a universal claim about conditional entropy or intrinsic dimensionality.

## Insight

Use an asymmetric reconstruction contract:

`multi-source conditions -> predicted background -> prediction-relative residual -> final composition`

The predicted background is not a disposable intermediate result. It is the common reference used throughout the residual path.

## Method

1. Encode heterogeneous constraints into role-aware condition representations.
2. Predict the velocity background deterministically.
3. Define the training residual relative to the predicted background.
4. Condition the residual vector field on structural information and the same predicted background.
5. Integrate the residual Flow Matching model.
6. Compose the final model as predicted background + predicted residual.

## Core memory point

**Predict the background; transport the prediction-relative correction.**

The same predicted background links supervision, conditioning, and composition.

## Primary contributions

### C1 — Role-aware multi-source conditioning

The framework organizes heterogeneous geophysical information around background and residual reconstruction roles rather than treating all modalities as interchangeable image channels.

### C2 — Prediction-consistent residual Flow Matching

The method defines the residual with respect to the actual predicted background used at inference and reuses that prediction for residual conditioning and final composition.

### C3 — Compact high-fidelity reconstruction

Under a matched multimodal input contract, the method is intended to deliver strong structural and overall velocity reconstruction with a compact parameter footprint. Final numerical wording must be based on the clean Remote Sensing evaluation package.

## Evidence architecture

### Main paper

1. **Observation/task definition**
   - four geophysical inputs;
   - target velocity;
   - frozen split/evaluator.

2. **Method figure**
   - make the three uses of predicted background visually obvious.

3. **Matched-input quantitative comparison**
   - one common evaluator;
   - one held-out manifest;
   - only adapted references that are scientifically interpretable.

4. **Qualitative structural comparison**
   - fixed-rule cases;
   - common scales;
   - complex structures emphasized while preserving complete quantitative coverage.

5. **Attribution experiment**
   - background prediction;
   - deterministic/full-field or residual controls;
   - prediction-consistent full method.

6. **Targeted robustness/efficiency**
   - bounded observation perturbations;
   - inference-step/runtime tradeoff if it strengthens the engineering value.

### Secondary / supplementary

- source-native literature context;
- PCA effective-rank analysis;
- straight-path target-energy diagnostics;
- extended missing-modality stress tests;
- additional condition-space diagnostics.

These can explain or contextualize the design, but should not carry the main novelty claim.

## Claims not to center

Do not make the paper depend on:

- proving that smoothing establishes lower conditional complexity;
- claiming universal easier learnability from target-energy reduction;
- implying explicit wave-equation physics coupling when none exists;
- claiming real-field deployment;
- claiming 3D readiness from 2D experiments;
- claiming stable superiority across random seeds before repeated-run evidence exists.

## Abstract blueprint

1. Important IVMB problem under heterogeneous post-stack constraints.
2. Gap: full-field prediction does not explicitly allocate background vs. remaining correction.
3. Solution: role-aware conditions + deterministic background + prediction-consistent residual Flow Matching.
4. Strongest clean quantitative results.
5. Significance: compact, structured use of complementary geophysical information for initial velocity model building.

## Introduction blueprint

- Paragraph 1: why initial velocity models matter.
- Paragraph 2: multi-source post-stack constraints are already available and complementary.
- Paragraph 3: learned full-field estimators do not explicitly assign modeling responsibilities.
- Paragraph 4: generative modeling is useful but full-field transport spends capacity on components that can be directly estimated.
- Paragraph 5: proposed background-guided residual strategy.
- Paragraph 6: contributions and strongest empirical takeaway.

Do not open with reviewer concerns, debugging history, or a long taxonomy.
