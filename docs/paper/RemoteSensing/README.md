# Remote Sensing Manuscript Workspace

This directory is the active manuscript package for the journal revision of PD-BG-RFM.

Manuscript-facing name: **BG-RFM**  
Working title: **Background-Guided Residual Flow Matching for Multi-Constraint Seismic Velocity Model Building**

## Status

The unified manuscript source is compiled as a single PDF; the scientific content remains frozen.

The manuscript is reconstructed from the strongest scientifically valid AAAI evidence plus the authoritative RS-E01-Lite unified evaluation. The historical AAAI snapshot under `docs/paper/aaaii/` remains immutable.

No additional experiment is a submission prerequisite.

## Main files

- `main.tex`: sole active LaTeX entry point; includes Appendices A--C in the same manuscript.
- `appendices/`: integrated mathematical, protocol, and additional-results details.
- `supplement.tex`: retained historical pre-unification source; it is not compiled for submission.
- `archive/pre-unification-2026-10-09/`: preserved original main/supplement and modified section sources.
- `sections/`: main manuscript sections.
- `tables/`: task, quantitative, and design-analysis tables.
- `figures/`: manuscript figure assembly files.
- `FIGURE_PLAN.md`: final figure roles and reused source assets.
- `NUMERICAL_AUDIT.md`: final numerical consistency ledger.
- `SUBMISSION_STATUS.md`: manuscript and packaging readiness.
- `MANUSCRIPT_PLAN.md`: retained story/evidence design history.
- `references.bib`: bibliography source.

## Formatting strategy

Remote Sensing currently permits free-format initial submission, provided the required scientific and back-matter sections are present. The current manuscript therefore prioritizes a stable scientific draft. For first submission the journal accepts free-format manuscripts. The official LaTeX template is recommended but not mandatory at this stage. This branch preserves the verified free-format manuscript and its frozen scientific content.

## Evidence rule

- RS-E01-Lite supplies the authoritative main matched-input table.
- Historical AAAI ablations, diagnostics, qualitative panels, and missing-modality tests may be reused as scientific source evidence unless a specific result is known to be incorrect.
- Do not use the unresolved historical InversionNet MAE_L = 0.0180 value.

## Unified manuscript (2026-10-09)

The manuscript contains one continuous main-plus-appendices PDF. Appendix A covers mathematical and implementation detail, Appendix B baseline adaptations and experimental/evaluation protocols, and Appendix C additional results, diagnostics and qualitative cases. The standalone supplementary PDF is discontinued. Compile from `main.tex` in this directory with `latexmk -pdf main.tex`. The workflow `.github/workflows/remote-sensing-paper-build.yml` collects the sole PDF and a self-contained LaTeX source ZIP including all referenced images.

## Non-anonymous submission preparation (2026-10-09)

Source of truth for the submission-preparation stage is submission/SUBMISSION_READINESS.md.

- main.tex now uses the structured Highlights required by Remote Sensing.
- Hyperlink anchors are made unique internally with hypertexnames=false; printed figure/table/equation numbering is unchanged.
- All author-sensitive front/back matter and GenAI language are held in submission/author_metadata.tex, submission/admin_statements.tex and submission/ai_methods_disclosure.tex.
- These files currently contain clearly marked author-confirmation placeholders; the preview is NOT authorized for SUSY upload.
- submission/package_submission.py builds a minimal source ZIP with recursively referenced LaTeX source, bibliography and figures only. Historical manuscript and supplement files are preserved in Git but excluded from this ZIP.
- Cover letter is a draft until originality, author-approval and related-manuscript declarations are signed by authors.
- DO NOT create the final submission tag until all author data and mandatory declarations are confirmed.
