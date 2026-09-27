# Remote Sensing Manuscript Workspace

This directory is the active manuscript workspace for the journal revision of PD-BG-RFM.

Working manuscript-facing name: **BG-RFM**  
Working title: **Background-Guided Residual Flow Matching for Multi-Constraint Seismic Velocity Model Building**

## Status

The project is in **submission sprint mode**.

The manuscript is being reconstructed from the scientific core rather than edited line-by-line from the AAAI submission. The historical AAAI snapshot under `docs/paper/aaaii/` remains immutable.

Only the exact quantitative values and numerical interpretation of the main matched-input comparison are intentionally left unresolved pending **RS-E01-Lite**.

## Files

- `main.tex`: content-first manuscript entry point.
- `sections/`: journal-oriented section drafts.
- `tables/`: task definition and RS-E01-Lite result table.
- `MANUSCRIPT_PLAN.md`: story, section architecture, figure/table plan, and AAAI material disposition.
- `references.bib`: copied from the editable AAAI source as a starting bibliography.

## Formatting note

The current source uses a lightweight `article` class so the scientific text remains easy to edit and review. After the scientific structure and RS-E01-Lite table are frozen, migrate the content into the current MDPI/Remote Sensing LaTeX template without changing the scientific claims.

## Submission rule

Do not copy historical AAAI quantitative tables into the final journal manuscript. The main comparison table must be populated only from the unified RS-E01-Lite package.
