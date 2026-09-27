# Remote Sensing Evidence Audit

Audit basis: handoff commit `4b91f626` on the `main` branch. This is a read-only audit; no model, manuscript number, historical artifact, or training job was changed.

## 1. Executive Summary

The repository has a coherent, inspectable implementation of PD-BG-RFM, a fixed data/split/evaluation protocol, and retained external run directories for the AAAI-era tables. The AAAI submission snapshot is `docs/paper/aaaii/`; the editable source is `docs/paper/AAAI2027/`; canonical code is `bg_pdr_fm/` plus `scripts/aaai27/` and `reproducibility/`.

The strongest defensible Remote Sensing story is narrower than the original AAAI framing: role-aware multimodal conditioning, deterministic background prediction, and prediction-relative residual transport that reuses the same predicted background for residual definition, conditioning, and composition. The source-native literature table should be context only. The matched-input and attribution tables should carry the main empirical argument.

The current paper results are sufficient to begin manuscript restructuring, but not sufficient to claim a clean empirical replication package. Retained manifests and metrics make several rows auditable in the original environment, while the GitHub handoff omits data, checkpoints, and logs. Multi-seed stability, realistic perturbation robustness, and real-data transfer are absent. The matched-input InversionNet row has a metric inconsistency: total MAE `0.0149` is smaller than low-pass MAE `0.0180`, which is incompatible with the current positive averaging-kernel evaluator under the same error field. Its exact historical cause is unresolved.

## 2. Current Auditable Assets

### Immutable historical evidence

- `docs/paper/aaaii/`: submitted AAAI manuscript and bundled historical paper/code snapshot. Do not edit.
- The numerical tables and captions in `docs/paper/aaaii/pd-bg-rfm.tex` are historical manuscript evidence.
- Retained run directories under the original workspace `logs/bg_pdr_fm/` and their checkpoints/manifests are external evidence, not part of the GitHub release.

### Development sources

- `bg_pdr_fm/`: model, data, metric, training, evaluation, baseline adapters, and manifest code.
- `scripts/aaai27/`: supported preparation, training, evaluation, smoke, and release entrypoints.
- `reproducibility/`: dependency, source-manifest, checklist, and release validation contracts.
- `docs/paper/AAAI2027/`: editable manuscript and paper-to-code map; it mirrors the historical numbers and is not a newly validated manuscript.
- `docs/research/`: collaboration and evidence notes.

### Protocol evidence

`docs/research/EXPERIMENT_PROTOCOL.md` fixes training seed 2027, split seed 42, well seed 1234, held-out traversal with `shuffle=False`, per-sample MAE/RMSE/SSIM/low-pass MAE/high-pass MAE, and a manifest contract containing commit, config hash, checkpoint hash, record identities, and counts. The release profile is `bg_pdr_fm/configs/release/aaai27/`.

## 3. Quantitative Result Provenance Matrix

The exact row-level inventory is also in `REMOTE_SENSING_RESULT_MAP.csv`.

| Paper result | Reported values / scope | Source | Retained artifact | Status |
|---|---|---|---|---|
| Source-native benchmark | 8 subsets × MAE/RMSE/SSIM; literature values plus PD-BG-RFM; 10.6M parameters | `docs/paper/aaaii/pd-bg-rfm.tex`, Table `main-comparison` | PD-BG-RFM global held-out manifest/checkpoint exists externally; literature rows are transcribed from cited papers | Partially auditable; cross-protocol by design |
| Matched-input comparison | InversionNet `0.0149/0.0273/0.9812/0.0180/0.0079`; VelocityGAN `0.0217/0.0434/0.9727/0.0149/0.0136`; UPFWI `0.0132/0.0299/0.9895/0.0079/0.0098`; Auto-Linear `0.0313/0.0593/0.9703/0.0182/0.0229`; Latent U-Net `0.0170/0.0311/0.9910/0.0071/0.0134`; PD-BG-RFM `0.0150/0.0270/0.9913/0.0119/0.0065` | `docs/paper/aaaii/pd-bg-rfm.tex`, Table `adapted-comparison` | `eval_table2_missing_modalities_global_heldout/*` has metrics, manifests, checkpoints, configs, commit `7c1adcd...`; artifacts are outside GitHub | Partially auditable; exact table aggregation needs re-run from retained files |
| Core ablation | Concat-FM `0.0225/0.0340/0.9865/0.0189/0.0068`; Decoupled-FM `0.0205/0.0320/0.9880/0.0175/0.0068`; Uniform `0.0163/0.0290/0.9898/0.0128/0.0072`; w/o consistency `0.0190/0.0340/0.9875/0.0156/0.0072`; w/o context `0.0230/0.0410/0.9820/0.0175/0.0100`; full `0.0150/0.0270/0.9913/0.0119/0.0065` | `docs/paper/aaaii/pd-bg-rfm.tex`, Table `ablation` | Formal training CSVs exist under `logs/bg_pdr_fm/aaai27/formal/{concat_fm_strong,cncs_fm_strong,mm_invnet,bg_pdr_fm}`; complete matching evaluation manifests were not found in the GitHub tree | Partially auditable |
| Condition diagnostics | Matched/deranged cosine margins `0.46/0.35`; fused medians `eta_bg=0.76`, `eta_str=0.66`, cross-role `chi=0.17`; complete-modality count 25,222 | Manuscript text and figures; diagnostic implementation under `bg_pdr_fm/diagnostics/` and evaluation scripts | Figure assets and some CSVs retained; a single manifest-backed aggregate artifact for every quoted statistic was not established | Partially auditable |
| Residual transport diagnostics | Effective ranks: median background `4.95`, structure `54.03`; `q_T` and residual-burden diagnostics reported in text/figures | Manuscript, `bg_pdr_fm/evaluation/benchmark_metrics.py`, diagnostic scripts | Training-analysis CSVs and evaluation metrics exist in original workspace; provenance is split across several directories | Partially auditable |
| Missing modalities | Full input `0.0150/0.0269/0.9915`; w/o well `0.0168/0.0295/0.9899`; w/o horizon `0.1011/0.1669/0.8202`; w/o RMS `0.3947/0.4705/0.0753`; w/o well+RMS `0.4684/0.5303/0.0192`; PoSTM only `0.4706/0.5356/0.0154` | Manuscript, Table `missing-modality`; `run_table2_missing_modality_benchmark.py` | Global held-out metrics/manifests exist externally for six modes and six methods; the paper values are not identical to the cross-method stress summary, so source selection must be resolved | Partially auditable; possible artifact-version mismatch |
| Parameter counts | PD-BG-RFM `10,630,216`; adapted InversionNet `24,409,123`; VelocityGAN `25,589,126`; UPFWI `18,996,259`; Auto-Linear `35,470,032`; Latent U-Net `35,123,429` | Manuscript Appendix `adapted-accounting`; `count_parameters`/baseline constructors | PD-BG-RFM and adapted baseline manifests record counts | Auditable as implementation counts; training-vs-inference convention differs for VelocityGAN |
| Runtime / inference | 50 Euler steps; no paper wall-clock table | Manuscript and release configs | Step count is source/config evidence; no complete hardware-normalized timing artifact | Source-auditable only |

The original paper also contains source-native values copied from external literature. They cannot be used as matched-input evidence, even when the table is numerically correct.

## 4. Metric Consistency Findings

### Current evaluator

`bg_pdr_fm/evaluation/benchmark_metrics.py` computes per-sample MAE and RMSE from the flattened error, a global population-variance SSIM, and frequency metrics from `LowHighPassFilter`. `bg_pdr_fm/models/filters.py` implements `P_L` as stride-one `5x5` `avg_pool2d` with padding and `P_H(x)=x-P_L(x)`. The evaluation path in `bg_pdr_fm/evaluation/evaluate_bg_pdr_fm.py` writes one row per record and records checkpoint/config/record provenance through `run_manifest.py`.

Under one common error field and this positive averaging kernel, pointwise Jensen contraction implies the low-pass absolute error should not exceed the corresponding average absolute error apart from boundary/aggregation convention effects. Therefore the matched-input InversionNet manuscript pair `MAE=0.0149`, `MAE_L=0.0180` is not consistent with the current evaluator when both are claimed to be computed from the same normalized prediction error.

### Historical trace

- The pair appears in the submitted snapshot, editable manuscript, and the older `paper0726.tex`/figure source.
- The current evaluator and filter implementation are different evidence from the table itself; no retained row-level artifact linked to that exact InversionNet table value was found in the release tree.
- Retained external evaluation directories do contain InversionNet metrics and manifests, but their summary values and mode layout do not establish that the exact manuscript aggregation used the current evaluator.
- The most likely explanations are an older low/high-pass implementation, a different boundary/normalization convention, or a table-generation/column mismatch. The repository does not establish which explanation is correct.

Do not silently correct `0.0180`. For Remote Sensing, either reconstruct the exact historical evaluator from an immutable artifact or remove/recompute the frequency columns under one common evaluator and label the result as a new run.

## 5. Baseline Adaptation Audit

| Baseline | Original method input/mechanism | Adapted input/objective in source | Classification | Safe claim |
|---|---|---|---|---|
| InversionNet | Official OpenFWI waveform-to-velocity supervised encoder-decoder | Five-channel multimodal condition adapter; direct L1; official-capacity network | Architecture-preserving adaptation | Controlled estimator comparison only; not the original waveform protocol |
| VelocityGAN | OpenFWI supervised generator plus adversarial discriminator | Multimodal input; GAN+L1 with generator/discriminator training | Architecture-preserving adaptation with changed input contract | Matched-input comparison; retain that adversarial mechanism is present |
| UPFWI | Physics/unsupervised inversion mechanism in the published method | Current adapter exposes a supervised multimodal velocity predictor in the experiment configs | Substantially altered or unclear | Do not call it a faithful reproduction of the original unsupervised method |
| Auto-Linear | Staged autoencoder/inverse latent translation | Multimodal measurement construction; staged AE/inverse schedule; rank-375 variant in paper | Architecture-inspired supervised adaptation | Compare the disclosed adapted estimator; avoid original-method equivalence |
| Latent U-Net / GFI | Large latent U-Net translation architecture | Multimodal input adapter, L1 training, validation-selected checkpoint | Architecture-inspired supervised adaptation | Architecture-level matched-input reference; not a full GFI protocol reproduction |

Source support: `bg_pdr_fm/external_baselines/`, experiment configs, and the optimization/checkpoint accounting in the manuscript Appendix. The table's “same inputs” claim is credible as an experiment contract, but the adaptations must be described as adaptations.

## 6. Existing Revision-Relevant Evidence

| Item searched | Finding |
|---|---|
| Direct `hat_B` vs low-pass target | Existing: evaluator writes `bg_mae`, `rho_hat_B`, `epsilon_H_B`; background training diagnostics contain `bg_l1`, `rho_hat_b`, and related fields. A single clean full-test background table is not committed. |
| Deterministic direct residual predictor | Existing code/config and retained runs: `residual_train_unet_direct_bg_*`, plus component-control definitions. Complete common-test summary is not established. |
| Full-field deterministic predictor | Existing adapters and formal runs (`mm_invnet`, `sv_inv_net`, baseline families). Use as a control only after common aggregation is verified. |
| Oracle-background residual | Existing retained artifacts under `oracle_upper_bound/` and `*_oraclebg*`; mostly bounded probes/partial runs, not a complete matched main-table result. |
| Predicted-background residual | Existing and strongest: PD-BG-RFM evaluation manifest records prediction-relative diagnostics and full six-mode stress evaluations. |
| Inference steps other than 50 | Existing retained probes for 20 and 100 steps under `residual_train_smooth_hc64_nogate_h256_e200`; no paper-level speed/accuracy sweep is committed. |
| Wall-clock inference | Timing support exists in `benchmark_metrics.time_prediction`; no complete hardware-normalized result table was found. |
| Missing modality | Existing and usable as a stress-test family, subject to selecting one artifact/version and reconciling the manuscript values. |
| RMS perturbation / noise robustness | Code/config names and stress directories exist in the original workspace, but no audited full-test protocol/result pair was found. |
| Horizon perturbation / shift robustness | No audited full-test result found. |
| Multi-seed training | Absent as accepted evidence; protocol explicitly says not established. |
| Repeated inference seeds | Sampling policy is recorded for paired modes; no independent stability report is accepted. |
| Low/high-frequency error | Existing in evaluator and paper tables; InversionNet row needs resolution of the inconsistency above. |
| Parameters/FLOPs/memory | Parameter counts are available; FLOP/memory comparison is absent. |
| Qualitative failure cases | Existing figure panels and selection manifest; they are selection artifacts, not a complete failure-rate analysis. |
| 1D profiles / well consistency | No audited quantitative result found. |
| Downstream FWI / migration | Absent. |

## 7. Evidence Gaps

1. The handoff repository does not contain the OpenFWI data, LMDB stores, checkpoints, or logs; external paths are not portable evidence for a reviewer.
2. The exact table-generation scripts/aggregation inputs for the matched-input table are not linked to one immutable manifest per row.
3. The InversionNet low-pass MAE inconsistency is unresolved.
4. The ablation table has training CSVs and manuscript values, but not one verified common-test artifact for every row.
5. There is no three-seed estimate or confidence interval for the proposed method or strongest baselines.
6. Noise/perturbation and horizon-shift robustness are not established under a frozen, manifest-backed protocol.
7. “Physics-decoupled” remains a naming/claim risk because the implementation has no explicit wave-equation or forward-operator constraint; PCA rank and transport-energy diagnostics are not proofs of lower conditional complexity.

## 8. Minimum Remote Sensing Experiment Package

### MUST RUN

1. **Common-evaluator re-audit and one clean matched-input rerun**: proposed method plus the two strongest practical adapted references, with one immutable test manifest, resolved config hashes, checkpoint hashes, per-record metrics, and the current evaluator. This must settle the InversionNet frequency-metric issue before final tables.
2. **Direct background evaluation**: report `hat_B` against `P_L(V)` on the same held-out records, including background MAE and low/high leakage. This directly supports deterministic background prediction.
3. **Attribution controls**: at minimum full PD-BG-RFM, deterministic direct residual, full-field deterministic predictor, oracle-background residual (upper bound), and predicted-background residual/prediction-consistent path. Use the same budget and evaluator; do not infer causal attribution from training loss alone.
4. **Three seeds for the proposed method and the strongest matched-input reference**: seeds must be recorded in separate run directories with the same split and data manifest. Report mean and spread; do not claim stable ranking without this.
5. **One bounded observation-perturbation test**: RMS amplitude/noise perturbation and a horizon perturbation with predeclared levels, evaluated without retraining. This is the minimum deployment-relevance test for synthetic multimodal observations.

### HIGH VALUE

- Inference-step sweep (20/50/100) reporting accuracy and synchronized wall-clock per sample on the same hardware.
- Full missing-modality table for the proposed method, reconciled to one checkpoint and one artifact version; retain RMS-missing as stress testing rather than a deployment claim.
- A compact error/failure analysis on fixed held-out subsets, including one-dimensional profiles where they clarify background/interface errors.
- Parameter count plus peak memory and throughput under the same batch/input contract.

### DO NOT PRIORITIZE

- 3D extension before the 2D evidence chain is closed.
- Field-data validation unless a reliable labeled or interpretable evaluation protocol is already available.
- Downstream FWI/migration unless the paper is explicitly repositioned around downstream utility; it is a separate study, not a minimum repair.
- More architectural variants, broad hyperparameter sweeps, or additional literature baselines that do not answer background attribution, prediction consistency, or robustness.

## 9. Proposed Main-Paper Evidence Layout

1. **Problem and observation contract**: define the four modalities, target, split, and what is observable at inference.
2. **Method**: role-aware conditions, deterministic `hat_B`, prediction-relative residual, and exact composition identity. Rename or narrow “physics-decoupled” unless a physical constraint is added.
3. **Main matched-input table**: one common evaluator, one held-out manifest, proposed method and two/three strongest adapted references; place source-native literature in a short context table or appendix.
4. **Attribution figure/table**: direct background, deterministic full-field, oracle background, predicted background, and full prediction-consistent path.
5. **Robustness panel**: predeclared RMS/horizon perturbations and a reconciled missing-modality result.
6. **Reproducibility/efficiency**: three-seed spread, parameter count, and step/runtime sweep.
7. **Limitations**: synthetic data, 2D scope, observation construction, no forward-operator constraint, and unresolved field transfer.

## 10. Open Questions / Blockers

- Which retained checkpoint/config/aggregation produced the exact manuscript matched-input rows? Until this is identified or recomputed, the old frequency columns should not be reused.
- Can the original evaluation environment and data manifest be reconstructed from the external run directories? If not, the Remote Sensing tables should be labeled as historical and new clean runs should replace them.
- What exact term should replace “physics-decoupled” if no wave-equation constraint is added?
- Which two baselines are affordable for three-seed repetition under the available hardware?
- Is the intended Remote Sensing contribution methodological reconstruction quality, robustness to missing/noisy constraints, or downstream utility? The minimum package above assumes the first two.

## Decision

The existing results are trustworthy enough to begin manuscript restructuring as historical evidence with explicit provenance labels. They are not trustworthy enough to serve unchanged as the final Remote Sensing quantitative package. Before new experiments, resolve the InversionNet metric provenance and freeze one common-test manifest/evaluator. The minimum scientific package is the five MUST-RUN items above; no 3D, field-data, or downstream FWI work is required for the first revision pass.
