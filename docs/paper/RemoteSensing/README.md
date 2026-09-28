# Remote Sensing Manuscript Workspace

This directory is the active manuscript package for the journal revision of PD-BG-RFM.

Manuscript-facing name: **BG-RFM**  
Working title: **Background-Guided Residual Flow Matching for Multi-Constraint Seismic Velocity Model Building**

## Status

The project is in **final manuscript production**.

The manuscript is reconstructed from the strongest scientifically valid AAAI evidence plus the authoritative RS-E01-Lite unified evaluation. The historical AAAI snapshot under `docs/paper/aaaii/` remains immutable.

No additional experiment is a submission prerequisite.

## Main files

- `main.tex`: complete free-format journal draft.
- `supplement.tex`: populated supplementary material.
- `sections/`: main manuscript sections.
- `tables/`: task, quantitative, and design-analysis tables.
- `figures/`: manuscript figure assembly files.
- `FIGURE_PLAN.md`: final figure roles and reused source assets.
- `NUMERICAL_AUDIT.md`: final numerical consistency ledger.
- `SUBMISSION_STATUS.md`: manuscript and packaging readiness.
- `MANUSCRIPT_PLAN.md`: retained story/evidence design history.
- `references.bib`: bibliography source.

## Formatting strategy

Remote Sensing currently permits free-format initial submission, provided the required scientific and back-matter sections are present. The current manuscript therefore prioritizes a stable scientific draft. At final packaging, migrate the text into the current MDPI Remote Sensing LaTeX template without changing scientific claims or numerical values.

## Evidence rule

- RS-E01-Lite supplies the authoritative main matched-input table.
- Historical AAAI ablations, diagnostics, qualitative panels, and missing-modality tests may be reused as scientific source evidence unless a specific result is known to be incorrect.
- Do not use the unresolved historical InversionNet MAE_L = 0.0180 value.
