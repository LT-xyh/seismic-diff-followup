# BG-RFM manuscript revision ledger

This ledger records modifications to the **active MDPI working manuscript**. The archived AAAI submission and previous step snapshots are not retrospectively altered.

Author approval: 2026-10-12 (Step 2C routing panel + permanent revision marking rule).

| ID | Status | Prior location | Working-manuscript location | Edit type | Visibility | Experimental effect |
|---|---|---|---|---|---|---|
| TITLE-PRIOR | Previously approved | Original AAAI title in archived source | `main.tex`, `\\Title` | Metadata wording changed in earlier Step 1 work | `% BG-REV TITLE-PRIOR` and this ledger; avoid fragile color in MDPI title macro | None |
| M1 | Previously approved | No dedicated task statement | `Method` opening, before *The Physics-Decoupled Condition Contract* | One added paragraph; corrected opening “This study considers…” | Full paragraph in `\\BGRevAdd` (blue in marked PDF) | None |
| M3 | Previously approved | Short paragraph after `eq:condition-loss` | `Method → Instantiating the Condition Contract` | Replaced one explanation paragraph; no equations altered | Full paragraph in `\\BGRevAdd` | None |
| R1-2C | Newly approved | Interpretation only in Appendix B | `Experiments → Condition and Transport Diagnostics`, after existing condition-alignment results | Added one evidence-grounded paragraph and a new main-text reference | Full paragraph in `\\BGRevAdd` | None |
| FIG-2C | Newly approved | Appendix B full diagnostics, panel (d) | `Experiments → Condition and Transport Diagnostics` | Added a standalone **cropped view** of panel (d); retains full unchanged diagnostic PNG in appendix | `\\BGRevFigure` colored frame plus `\\BGRevAdd` caption | No new analysis/data |
| M2 | Not approved | Appendix C reliability normalization | Unchanged | No new formula | N/A | None |
| M4 | Not approved | Appendix D training settings | Unchanged | Keep detailed settings in appendix | N/A | None |
| M5 | Not approved | Appendix D metric formulas | Unchanged | Keep metrics in appendix | N/A | None |
| Step 2B | Explicitly declined | Original AAAI comparison | Unchanged | No replacement of original AAAI results/rankings | N/A | None |

## Figure data and presentation provenance

- Original unmodified asset: `figures/aaai/appendix_full_condition_diagnostics.png` (identical shared repo file; original 2561 × 1545 pixels, 320 dpi).
- Standalone view uses the same file via `\\includegraphics[trim=12bp 7bp 8bp 195bp,clip]`. This selects the lower role-specific reliability routing heatmaps; it does **not** regenerate weights or manipulate numeric values.
- The appendix continues to show the full four-panel (a)–(d) diagnostic.
- Main-text sample count: **25,222** records containing all four modalities. Averages are from the trained encoder and influenced by the role-specific reliability training priors; they are descriptive, not proof of autonomous geophysical role discovery.
- New main-text figure label: `fig:main-role-routing`. Existing appendix label `fig:full-condition-diagnostics` remains unchanged.

## Active working files and immutable sources

- Active entry: `main.tex` selects `CONTENT_STEP2C_MARKED.tex`.
- Active marked source: `CONTENT_STEP2C_MARKED.tex`.
- Previous M1/M3 snapshot: `CONTENT_M1_M3_REVISED.tex`, unchanged.
- Step 1 formatting snapshot: `CONTENT_MDPI_LAYOUT.tex`, unchanged.
- Historical source: `original/pd-bg-rfm.tex`, unchanged.
- `\\BGshowrevisionstrue` displays all marked changes; `\\BGshowrevisionsfalse` hides markings for a clean eventual journal submission.

## Remaining verification

- Stage-specific reversible text scope audit: verified in the editing operation. Relative to M1/M3 stage, the only semantic addition is the routing paragraph/figure and its caption; earlier M1/M3 paragraphs are only wrapped in revision-coloring macros.
- LaTeX compilation with actual MDPI `Definitions/` dependency has not been completed in the GitHub environment because the publisher download returned HTTP 403. A passing GitHub quality job is **not** an MDPI PDF compilation.
- A marked and clean PDF must both be visually inspected before journal submission, particularly the crop of the new routing panel. No merge or publication tag is authorized by this change.
