# Remote Sensing Submission Status

Updated: 2026-09-29

## Scientific status

- Core story: frozen for first submission.
- Mandatory experiments: complete.
- Additional experiments: not submission blockers.
- Main matched-input comparison: RS-E01-Lite authoritative.
- Historical AAAI evidence: reusable unless a specific result is known to be incorrect.
- Known excluded historical value: InversionNet MAE_L = 0.0180.

## Manuscript completeness

- Title: complete working title.
- Abstract: complete.
- Keywords: complete.
- Highlights: complete working draft.
- Introduction: complete first-submission draft.
- Related Work: complete first-submission draft.
- Materials and Methods: complete first-submission draft.
- Results 4.1: complete.
- Results 4.2: complete with retained qualitative evidence.
- Results 4.3: complete as a supporting three-formulation comparison (Concat-FM, role-aware full-field FM, BG-RFM), explicitly not treated as tightly matched because identical Euler-step settings cannot be verified for every archived run.
- Results 4.4: complete using diagnostic evidence.
- Discussion: complete.
- Conclusions: complete.
- Supplementary Material: structured and populated.
- Data Availability: drafted.
- Code Availability: drafted.
- Ethics statements: drafted as not applicable.
- Author Contributions: metadata placeholder.
- Funding: metadata placeholder.
- Acknowledgments/GenAI disclosure: metadata/policy placeholder.
- Conflicts of Interest: author-confirmation placeholder.

## Figure status

1. Figure 1 — BG-RFM conceptual overview: functional LaTeX schematic in manuscript; final graphic polish optional before submission.
2. Figure 2 — modality/decomposition characterization: reused historical source panels.
3. Figure 3 — fixed-rule qualitative comparison: reused historical source panel.
4. Figure 4 — condition/transport diagnostics: reused historical source panels.

## Table status

1. Table 1 — observation/task contract.
2. Table 2 — RS-E01-Lite unified matched-input comparison.
3. Table 3 — supporting comparison of three Flow-Matching formulations; not treated as a tightly matched benchmark.
4. Supplementary tables — effective rank and missing-modality stress tests.

## Remaining submission blockers

Scientific:
- none known.

Administrative/format:
- insert final author names, affiliations, and corresponding-author metadata;
- insert verified funding information;
- confirm conflict-of-interest statement;
- finalize acknowledgment and any journal-required generative-AI disclosure;
- archive/tag the exact code revision used for submission;
- prepare the mandatory Remote Sensing cover letter after all authors confirm the required submission statements;
- migrate the free-format LaTeX draft to the current MDPI Remote Sensing template at final packaging;
- verify final template pagination against the journal's current Article guidance;
- compile and visually inspect the final template package.

## Submission principle

Do not launch new experiments to resolve formatting, metadata, or presentation issues.


## Blind-review targeted revision

Independent blind review requested targeted clarification rather than new experiments.

Completed:
- novelty positioned as reconstruction organization rather than a new Flow-Matching mathematics;
- full-field background-centered path interpretation added;
- composition-error equality identified explicitly as an algebraic identity;
- task statement narrowed to reconstruction from already available processed/interpreted constraints;
- synthetic RMS information content distinguished from uncertain field-derived RMS products;
- common evaluation protocol expanded;
- baseline adaptations expanded;
- qualitative examples labeled as performance-selected high-margin cases;
- reviewer-facing method labels aligned;
- observation-removal language changed from generic robustness/complementarity claims to sensitivity/dependence language;
- internal project vocabulary removed from reviewer-facing prose.

Final Table 3 fact gate: CLOSED. Uniform Fusion, w/o Prediction Consistency, and w/o Background Context are excluded from reviewer-facing evidence because their implementation provenance could not be established sufficiently. No reconstruction, retraining, or artifact archaeology is required.


## Reviewer M1/M2 documentation closure

Verified implementation facts have been integrated:
- complete active condition objective and zero-weight exclusions;
- loader quality/availability and learned reliability definitions;
- four-stage training schedule and validation-based selection;
- split-provenance wording with prerequisite-manifest limitation;
- offline observation-product contract and current 0--3-column well implementation;
- tensor/interface shape table;
- continuous Flow-Matching time wording and 50-step Euler inference;
- parameter-count conventions, including exact BG-RFM inference count and omission of an unverified UPFWI-derived count;
- evaluation batch size, stochastic trajectory policy, and batch-keyed seed explanation;
- observation-removal tensor/mask implementation;
- operational definitions for CKA, retrieval, anchor similarity, role specialization, and bootstrap intervals;
- removal of the duplicate supplementary full-input score.

Table 3 is now supporting formulation-level evidence rather than a strictly controlled benchmark.

## Final scientific closure

Scientific source closure: **PASS**.

- Reviewer-2 M1: resolved through method/objective/training/split/observation/architecture/evaluation documentation.
- Reviewer-2 M2: resolved by positioning Table 3 as supporting formulation-level evidence and avoiding unsupported strict-control or component-isolation language.
- Publication-principle cleanup: completed; repeated internal provenance/audit language and unnecessary self-weakening comparisons were reduced while retaining the qualifications needed for scientific interpretation.
- Numerical/terminology ledger: synchronized on 2026-09-29.
- New experiments: none.
- Journal submission: not yet authorized; final PDF/template and author-metadata packaging remain separate steps.
