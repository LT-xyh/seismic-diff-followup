# Claim–Evidence Matrix

Status values:

- **SOURCE-VERIFIED**: implementation/math can be verified from repository source.
- **HISTORICAL**: reported previously but not yet accepted as the final Remote Sensing empirical artifact.
- **NEEDS-EVAL**: existing checkpoint/code likely sufficient; requires a clean common evaluation.
- **NEEDS-TRAIN**: new training may be required.
- **GATED**: run only if a previous result shows it is necessary.
- **SUPPLEMENTARY**: useful context, not a headline claim.
- **EXCLUDE**: do not make this a paper claim.

| ID | Candidate claim | Evidence | Current status | Main-paper role | Action |
|---|---|---|---|---|---|
| C1 | Heterogeneous geophysical constraints are organized into background-oriented and residual/structural roles. | Encoder/fusion source; condition diagnostics; ablations. | SOURCE-VERIFIED + HISTORICAL | Primary method claim | Re-evaluate only the diagnostics needed to explain the retained method. |
| C2 | The method uses the same predicted background for residual target definition, residual conditioning, and final composition. | Method source, training path, stop-gradient contract, exact composition identity. | SOURCE-VERIFIED | Primary novelty claim | Keep central. No new training needed to establish the implementation fact. |
| C3 | The complete background-guided residual organization is supported relative to simpler full-field Flow-Matching formulations. | Retained three-formulation comparison under a common dataset/split/normalization/metric family. | HISTORICAL / SUPPORTING | Supporting empirical claim | Keep as formulation-level evidence; do not treat as component-isolated or causal attribution. |
| C4 | The deterministic background branch is explicitly supervised to recover the defined smooth component. | Source implementation and active background objective. | SOURCE-VERIFIED | Method specification | State as an implementation/design fact; direct quantitative branch evaluation is reviewer-triggered. |
| C5 | BG-RFM provides strong structural/overall reconstruction under the common multimodal input contract. | RS-E01-Lite: 33,600 common held-out records, common evaluator. | ACCEPTED-EVIDENCE | Primary result | Use MAE/RMSE/SSIM from RS-E01-Lite; exclude unresolved historical frequency metric. |
| C6 | The model is compact relative to adapted references. | Source parameter counts/manifests. | SOURCE-VERIFIED | Supporting advantage | Report inference-relevant parameter convention clearly. |
| C7 | The model remains useful under bounded RMS/horizon observation perturbations. | No accepted artifact. | NEEDS-EVAL | Targeted robustness evidence | Run fixed no-retraining stress tests. |
| C8 | Fewer ODE steps retain most reconstruction quality and reduce inference cost. | 20/100-step probes exist but no common benchmark. | NEEDS-EVAL | Efficiency evidence | Run a synchronized step/runtime sweep if inexpensive. |
| C9 | Ranking is stable across training seeds. | Not established. | GATED / NEEDS-TRAIN | Only if paper claims stable ranking | Run three seeds only if clean margins are narrow enough that seed variability matters. |
| C10 | PCA rank gap proves lower conditional complexity. | PCA effective-rank diagnostic. | SUPPLEMENTARY | Motivation only | Rephrase as decomposition/data diagnostic, not proof. |
| C11 | Lower straight-path target energy proves the proposed routing is intrinsically easier to learn. | `q_T` diagnostic and theorem. | SUPPLEMENTARY | Diagnostic only | Do not use as causal proof. |
| C12 | "Physics-decoupled" means wave-equation or forward-operator decoupling. | No such mechanism. | EXCLUDE | None | Do not make this claim. |
| C13 | Method is validated for real field data / 3D production use. | Not established. | EXCLUDE | None | Do not claim. Future work only if needed. |

## Main-paper claim discipline

The final Abstract and Conclusion should draw primarily from C1, C2, C5, and C6, with C3 used only as supporting formulation-level evidence.

C9 is conditional on repeated-run evidence.

C10–C11 can be retained only if they clarify the method without becoming headline proof.

C12–C13 are outside the evidence boundary.
