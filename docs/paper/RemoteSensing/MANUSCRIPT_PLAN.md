# Remote Sensing Manuscript Reconstruction Plan

Updated: 2026-09-27

## Working title

**Background-Guided Residual Flow Matching for Multi-Constraint Seismic Velocity Model Building**

Alternative if the condition-routing evidence remains strong enough to carry title-level emphasis:

**Role-Aware Background-Guided Residual Flow Matching for Multi-Constraint Seismic Velocity Model Building**

The default title is intentionally simpler and avoids making "physics-decoupled" the headline.

## One-paragraph thesis

Initial velocity model building can exploit several processed and interpreted geophysical products whose information is complementary rather than interchangeable. RMS velocity and sparse wells provide strong guidance on large-scale velocity magnitude, while migrated images and interpreted horizons provide dense structural cues. BG-RFM turns this observation into a reconstruction architecture that first predicts a deterministic background velocity and then applies Flow Matching only to the correction relative to that prediction. The same predicted background is used to define the residual target, condition the residual vector field, and compose the final velocity estimate. This prediction-consistent design gives the two estimators explicit reconstruction responsibilities while still allowing every available modality to contribute to both learned condition roles. The paper therefore argues for structured responsibility allocation in multimodal velocity model building, supported by one unified matched-input evaluation rather than by broad claims about conditional entropy or universal transport complexity.

## Final contribution structure

1. **Role-aware multi-source conditioning.**  
   A multimodal condition encoder learns background-oriented and structure-oriented interfaces from RMS velocity, post-stack migrated images, interpreted horizons, and sparse wells without imposing a hard one-modality/one-role partition.

2. **Prediction-consistent residual Flow Matching.**  
   The method deterministically predicts a background velocity and reuses that same prediction to define the residual target, condition residual transport, and compose the final model. The identity
   `hat V - V = hat R - (V - hat B)`
   makes the consistency objective explicit without requiring a universal complexity theorem.

3. **A parameter-compact prediction-consistent generative formulation for structured velocity reconstruction.**  
   Under the unified matched-input protocol, BG-RFM provides strong reconstruction quality with 10.6M parameters, while the paper explicitly does not claim overall metric dominance or computational-speed superiority.

## Proposed section structure

1. **Introduction**
   - importance of initial velocity model building;
   - availability of heterogeneous processed/interpreted constraints;
   - gap: full-field estimators leave reconstruction responsibility unassigned;
   - proposed background-guided residual formulation;
   - contributions.

2. **Related Work**
   - learned seismic velocity model building;
   - generative modeling for seismic velocity reconstruction;
   - multi-source geophysical conditioning.

3. **Materials and Methods**
   - problem formulation and observation contract;
   - synthetic multimodal construction;
   - role-aware multi-source condition encoding;
   - deterministic background prediction;
   - prediction-consistent residual Flow Matching;
   - joint training and observed-only inference;
   - experimental protocol and metrics.

4. **Results**
   - unified matched-input comparison (RS-E01-Lite);
   - fixed-rule qualitative comparison;
   - compact attribution evidence only if existing provenance is sufficient.

5. **Discussion**
   - why responsibility allocation is useful;
   - why prediction consistency matters;
   - interpretation of compactness and inference cost;
   - concise limitations.

6. **Conclusions**
   - reinforce the central memory point only.

## Main evidence map

### Table 1 — Task and observation contract

Include:
- 8 OpenFWI subsets;
- target grid: 70 x 70;
- RMS and migrated image: 1000 x 70;
- horizon and well map: 70 x 70;
- global split: 70/20/10 with split seed 42;
- held-out records: 33,600;
- deterministic test well seed: 1234.

### Table 2 — Main quantitative comparison

Rows:
- adapted InversionNet;
- adapted VelocityGAN;
- supervised multimodal UPFWI adaptation;
- adapted Latent U-Net;
- BG-RFM.

Auto-Linear remains omitted from the first main table.

Default columns:
- MAE;
- RMSE;
- SSIM;
- Params.

Optional columns:
- MAE_L;
- MAE_H only if RS-E01-Lite validates them cleanly and they materially improve the story.

### Figure 1 — Central method figure

Single visual memory point:

**Predict the background; transport the prediction-relative correction.**

The figure must show the same `hat B` entering:
1. residual target definition during training;
2. residual-vector-field conditioning;
3. final composition.

Training-only target-derived paths should be visually separated from inference paths.

### Figure 2 — Observation/task contract

Show one target and the four observations:
- RMS velocity;
- post-stack migrated image;
- horizon map;
- sparse well map.

The caption should state which products are time-domain or depth-domain and that the benchmark observations are constructed offline from the synthetic reference model.

### Figure 3 — Fixed-rule qualitative comparison

Use documented held-out selections, common color scales, target, two/three strongest references, and BG-RFM.

### Figure 4 / Table 3 — Attribution evidence

Use only if the retained historical evidence can be linked to a sufficiently clear checkpoint/config/evaluator contract.

Do not retrain solely to complete a large attribution matrix before first submission.

## AAAI material disposition

### Keep, but rewrite

- multimodal problem motivation;
- prediction-relative residual definition;
- shared use of `hat B`;
- background loss and Flow Matching objective;
- observed-only inference contract;
- synthetic observation construction;
- common held-out evaluation protocol;
- adapted-baseline disclosure;
- existing fixed-rule qualitative selection assets.

### Move to supplement / secondary material

- full condition-learning objective;
- training-only wavelet-anchor details;
- extended reliability diagnostics;
- PCA effective-rank analysis;
- transport-energy diagnostic;
- extended missing-modality tables;
- theorem derivations if retained at all;
- detailed baseline accounting.

### Delete from the journal main narrative

- "conditional complexity asymmetry" as a proved mechanism;
- PCA rank as proof of conditional complexity;
- transport-energy theorem as causal proof of easier learning;
- source-native literature table as a ranking table;
- "physics-decoupled" as a headline claim;
- rebuttal-style responses to AAAI reviewer concerns;
- debugging/reproducibility history.

### Rewrite substantially

- title;
- abstract;
- introduction;
- contributions;
- method opening and terminology;
- baseline-comparison language;
- results interpretation;
- discussion;
- conclusion.

## Abstract contract

The abstract should contain:
1. IVMB problem and multi-source setting;
2. estimator-responsibility gap;
3. BG-RFM solution;
4. prediction-consistency mechanism;
5. one clean RS-E01-Lite result sentence;
6. significance.

No limitations paragraph and no historical-review language.

## RS-E01-Lite integration

RS-E01-Lite is accepted as the authoritative common-protocol result package.

Authoritative rows:
- BG-RFM: MAE 0.01502705, RMSE 0.02694626, SSIM 0.99164512, 10.60M parameters;
- InversionNet adaptation: MAE 0.01193781, RMSE 0.02730467, SSIM 0.99118468, 24.41M parameters;
- VelocityGAN adaptation: MAE 0.02170463, RMSE 0.04340946, SSIM 0.97267124, 25.59M parameters;
- supervised multimodal UPFWI adaptation: MAE 0.01217775, RMSE 0.02986827, SSIM 0.99050921, approximately 19.00M parameters;
- Latent U-Net adaptation: MAE 0.00705923, RMSE 0.01308783, SSIM 0.99403170, 35.12M parameters.

The paper must not convert this table into an overall winner ranking. Latent U-Net is the strongest method on all three reconstruction metrics. BG-RFM's retained methodological story is structured responsibility allocation, prediction-consistent residual transport, and parameter compactness.

Frequency metrics passed the sanity check but remain optional for the first submission.
