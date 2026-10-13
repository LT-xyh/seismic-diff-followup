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

- Active entry (Step 2D snapshot): `CONTENT_STEP2D_TABLE1_MARKED.tex`; Step 2E active content is `CONTENT_STEP2E_MARKED.tex`.
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

## Step 2D — Table 1 page-width correction (2026-10-12)

- ID: `2D-TABLE1-FIT` — The table previously used `adjustwidth{-\\extralength}{0cm}` and a full expanded `\\linewidth`, extending into the wide journal-review margin. The new layout uses zero extra width and scales the unchanged 9-column tabular body within the native available `\\linewidth` via `\\resizebox{\\linewidth}{!}{...}`. The caption and label stay in place; dataset entries, numerical values, emphasis/rank, citations, rows, and columns remain byte-identical between tabular contents.
- Active input is now `CONTENT_STEP2D_TABLE1_MARKED.tex`; `CONTENT_STEP2D_MARKED.tex` and all earlier snapshots are immutable.
- Scope: pure table typography/layout; no scientific content or experimental changes, no new equations or tests.
- Marked review PDF: `% BG-REV 2D-TABLE1-FIT` source marker and this ledger; all M1/M3/Step 2C and Step 2D Discussion paragraph blue markings remain unchanged.
- Source: `a060db8652e892c507e736b9193fcfb3538f0304`; implementation commit recorded in Git history.

## Step 2E — Minimal Introduction revision (2026-10-12)

- Parent branch: `remote-sensing/step2d-table1-proof-20261012`; exact parent commit: `80f4ccaf270a1cf7443921c33d0f5a5cbeddcec9`.
- Working branch: `remote-sensing/step2e-introduction`; active entry: `main.tex` uses `CONTENT_STEP2E_MARKED.tex`.
- `2E-INTRO-P7`: Introduction method-summary paragraph beginning “We propose” (previous Step 2D manuscript, Introduction paragraph 7); rewrite is limited to clearer role-aware conditioning, deterministic background prediction, the one-way generator-facing dependency, and alignment with the approved “Decoupled Multimodal” title. Full revised paragraph wrapped in `\\BGRevAdd`.
- `2E-INTRO-C1`: First item of the Introduction contributions list; clarify the role-aware, nonexclusive condition assignment without modifying the block-triangular contract. Full revised item text wrapped in `\\BGRevAdd`.
- `2E-ENTRY`: `main.tex` input change to independent Step 2E snapshot; no text or metadata in `main.tex` changed.
- All other Introduction prose and bullets, citations, references, figures, equations, theorem/definition/corollary, original AAAI tables, Methods, Results, Discussion, Conclusions and Appendices A–D are identical to source. The Table 1 layout fix and previous blue revision marks are preserved. No new data, citations, numerical results or experiments.
- The previously approved headline/title and abstract are unchanged; no benchmark replacement or new dataset contribution.
- Commit: recorded in the Step 2E Git commit history. The PDF build may use a separate proof branch solely for CI and artifact packaging.

## Step 2F — Related Work development (2026-10-12)

- Branch: `remote-sensing/step2f-related-work`; baseline: `f0669b883e82379baaa4caac21eb9876e21d25a5`. Active entry: `main.tex` -> `CONTENT_STEP2F_MARKED.tex`; prior source snapshot `CONTENT_STEP2E_MARKED.tex` remains immutable.
- `2F-H-RELATED-WORK`: inserted `Related Work` with label `sec:related-work` after Introduction and before `Materials and Methods`; three new subsections: Learning-Based Seismic Velocity Inversion, Generative Models for Subsurface Reconstruction, and Multimodal Geophysical Conditioning.
- `2F-RW-S1-P1` through `2F-RW-S3-P3`: nine new scholarly paragraphs (1008 whitespace-separated words before citation command removal), each fully wrapped in `\BGRevAdd` and individually marked in source. No re-used passage is duplicated verbatim from Introduction.
- `2F-INTRO-P2`, `2F-INTRO-P3`, `2F-INTRO-P4`: three original literature-heavy Introduction paragraphs condensed to shorter conceptual transitions so that the Introduction retains geophysical motivation, the heterogeneity problem, reconstruction responsibility, prediction-relative reference, and author-approved Step 2E contribution statement. Entire revised paragraphs marked blue. Earlier Introduction material, citations, and methodological claims are otherwise maintained.
- `2F-BIB`: four independently verified additions to `references.bib`, with titles, author names, source types, years, DOI/stable URLs: Kim et al. (2024) GJI DOI 10.1093/gji/ggae169; Zhang, Li & Huang (2024) DiffusionVel arXiv 2410.21776 (preprint label, not upgraded to a journal claim); Zhang et al. (2025) IEEE TGRS DOI 10.1109/TGRS.2025.3584250; Wu et al. (2026) Journal of Applied Geophysics DOI 10.1016/j.jappgeo.2026.106137.
- EAAI/SeisMIC-Diff boundary: no SeisMIC-Init dataset contribution or processing novelty asserted for BG-RFM; related EAAI manuscript publication status and DOI not inferred, and any overlap disclosure remains an editorial matter outside this scientific content revision.
- Source diff audit: Methods, Results, Discussion, Conclusions, figures/algorithm/table/appendices/theorem/equation text and old bibliography entries are unchanged; all earlier blue author markings persist. No new experiments or numeric claims. Revision markers for all changed material as above; title and abstract immutable.
- PDF proof to be generated with the repository-bundled MDPI class on an isolated build branch; this does not establish identity with the earlier mentioned 2026-09-11 v6.5a class.
- Git change: see Step 2F branch commit history.

## Step 2G / A -- authorized narrative enhancement (2026-10-13)

- Baseline: `cd2dd8da5df49594669ec207e57b0572e8ec27c2`; branch: `remote-sensing/step2g-narrative-enhancement`.
- Active source: `CONTENT_STEP2G_METHODS_MARKED.tex`. Earlier snapshots and original AAAI sources are immutable.
- 2G-A-EQ and 2G-A-P1: insert one equivalent full-field training-path equation and one paragraph after eq:fm-objective and its existing interpretation sentence. Equation color is scoped outside the equation environment; the new paragraph uses BGRevAdd. Existing definitions, losses, and Theorem 1 are untouched.
- Visibility: entire new/revised paragraphs blue in marked mode; scientific content identical in clean mode.
- Numerical/experimental effect: none. No training or new evaluation.
- Commit: this section's Git history records the corresponding stage commit.
