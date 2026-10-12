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
| 2D-H-METHOD | Approved Step 2D | `Method` section heading | `Materials and Methods` section heading | Rename only | Source marker `2D-H-METHOD` | None |
| 2D-MOVE-SETUP | Approved Step 2D | `Experiments` > `Experimental Setup` | `Materials and Methods` > `Experimental Setup` | Existing subsection and both paragraphs moved verbatim | Source marker `2D-MOVE-SETUP`; unchanged text is not newly blue | None |
| 2D-H-RESULTS | Approved Step 2D | `Experiments` section heading | `Results` section heading | Rename only; result tables and figures unchanged | Source marker `2D-H-RESULTS` | None |
| 2D-H-DISCUSSION | Approved Step 2D | No standalone Discussion | Between Results and Conclusions | New heading | Source marker `2D-H-DISCUSSION` | No new data or analyses |
| 2D-DISC-P1 | Approved Step 2D | No standalone Discussion paragraph | Discussion paragraph 1 | New scholarly synthesis of existing evidence | Whole paragraph in BGRevAdd (blue in marked PDF) | No new data or analyses |
| 2D-DISC-P2 | Approved Step 2D | No standalone Discussion paragraph | Discussion paragraph 2 | New scholarly synthesis of existing evidence | Whole paragraph in BGRevAdd (blue in marked PDF) | No new data or analyses |
| 2D-DISC-P3 | Approved Step 2D | No standalone Discussion paragraph | Discussion paragraph 3 | New scholarly synthesis of existing evidence | Whole paragraph in BGRevAdd (blue in marked PDF) | No new data or analyses |
| 2D-DISC-P4 | Approved Step 2D | No standalone Discussion paragraph | Discussion paragraph 4 | New scholarly synthesis of existing evidence | Whole paragraph in BGRevAdd (blue in marked PDF) | No new data or analyses |
| 2D-H-CONCLUSIONS | Approved Step 2D | `Conclusion` section heading | `Conclusions` section heading | Rename only; original conclusion text preserved | Source marker `2D-H-CONCLUSIONS` | None |
| 2D-ENTRY | Approved Step 2D | `main.tex` selects Step 2C source | `main.tex` selects Step 2D source | Active source toggle | Source marker `2D-ENTRY` | None |

## Figure data and presentation provenance

- Original unmodified asset: `figures/aaai/appendix_full_condition_diagnostics.png` (identical shared repo file; original 2561 × 1545 pixels, 320 dpi).
- Standalone view uses the same file via `\\includegraphics[trim=12bp 7bp 8bp 195bp,clip]`. This selects the lower role-specific reliability routing heatmaps; it does **not** regenerate weights or manipulate numeric values.
- The appendix continues to show the full four-panel (a)–(d) diagnostic.
- Main-text sample count: **25,222** records containing all four modalities. Averages are from the trained encoder and influenced by the role-specific reliability training priors; they are descriptive, not proof of autonomous geophysical role discovery.
- New main-text figure label: `fig:main-role-routing`. Existing appendix label `fig:full-condition-diagnostics` remains unchanged.

## Active working files and immutable sources

- Active entry (Step 2D): `main.tex` selects `CONTENT_STEP2D_MARKED.tex`.
- Previous Step 2C marked snapshot: `CONTENT_STEP2C_MARKED.tex`, unchanged.
- Active marked source: `CONTENT_STEP2C_MARKED.tex`.
- Previous M1/M3 snapshot: `CONTENT_M1_M3_REVISED.tex`, unchanged.
- Step 1 formatting snapshot: `CONTENT_MDPI_LAYOUT.tex`, unchanged.
- Historical source: `original/pd-bg-rfm.tex`, unchanged.
- `\\BGshowrevisionstrue` displays all marked changes; `\\BGshowrevisionsfalse` hides markings for a clean eventual journal submission.

## Remaining verification

- Stage-specific reversible text scope audit: verified in the editing operation. Relative to M1/M3 stage, the only semantic addition is the routing paragraph/figure and its caption; earlier M1/M3 paragraphs are only wrapped in revision-coloring macros.
- LaTeX compilation with actual MDPI `Definitions/` dependency has not been completed in the GitHub environment because the publisher download returned HTTP 403. A passing GitHub quality job is **not** an MDPI PDF compilation.
- A marked and clean PDF must both be visually inspected before journal submission, particularly the crop of the new routing panel. No merge or publication tag is authorized by this change.

## Step 2D — Section organization and Discussion (2026-10-12)

- Source: `remote-sensing/step2c-routing-marked-20261012` at `297d51f9d804a5839eee89895a7be7785e1f37cb`.
- New independent branch: `remote-sensing/step2d-sections-discussion-marked-20261012`.
- The four new Discussion paragraphs use the approved BGRevAdd source macro; earlier M1, M3, R1-2C and FIG-2C markings are retained.
- The Experimental Setup subsection moves verbatim, including all numeric values and citations; no mathematical, algorithmic or benchmark change.
- Appendices A–D, original figures, title, abstract, Introduction, Conclusion body, citations and references are preserved.
- Heading/input changes are recorded with BG-REV comments rather than unsafe inline coloration of structural commands.
- Implementation commit: 6d216e505c0001a79e3b8e7f1944190141b4eb61.
- MDPI PDF rendering still requires the verified ACS v6.5a class and binary figures; existing GitHub dependency-download failures do not authorize an infrastructure rewrite.
