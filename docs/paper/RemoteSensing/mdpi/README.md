# Remote Sensing Final Packaging

Scientific-content baseline:
`96fddfcab601dc61562b73998a74572dde4a160f`

This directory is the journal-compatible packaging layer. It does not reopen the scientific scope.

## Compile targets

- `main.tex`: MDPI Remote Sensing submission source with explicit administrative placeholders.
- `main_blind.tex`: anonymized wrapper used for blind-review PDF generation.
- `supplement.tex`: MDPI supplementary source.
- `supplement_blind.tex`: anonymized supplementary wrapper.

## Supporting files

- `Definitions/`: MDPI v6 class/style resources.
- `figures/`: copied publication figures needed to recompile the package.
- `references.bib`: pruned bibliography containing only cited/useful entries.
- `FINAL_NUMERICAL_AUDIT.md`: numerical verification ledger.
- `ADMIN_INFO_REQUIRED.md`: metadata still required from the authors.
- `COVER_LETTER.md`: Remote Sensing cover-letter draft.
- `TEMPLATE_PROVENANCE.md`: template/version record.

No additional experiments are required for this packaging layer.
