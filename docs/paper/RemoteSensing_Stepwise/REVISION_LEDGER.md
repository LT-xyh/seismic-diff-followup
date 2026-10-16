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

## Step 2G / B -- authorized narrative enhancement (2026-10-13)

- Baseline: `cd2dd8da5df49594669ec207e57b0572e8ec27c2`; branch: `remote-sensing/step2g-narrative-enhancement`.
- Active source: `CONTENT_STEP2G_METHODS_ABLATION_MARKED.tex`. Earlier snapshots and original AAAI sources are immutable.
- 2G-B-P1 and 2G-B-P2: replace the original percentage-summary interpretation paragraph in Results / Ablation Study with two evidence-grounded paragraphs. Variant-definition paragraph and tab:ablation are byte-identical. All cited numbers are existing table entries, not new analyses. Background-context removal explicitly retains the predictor, residual reference, and final composition.
- Visibility: entire new/revised paragraphs blue in marked mode; scientific content identical in clean mode.
- Numerical/experimental effect: none. No training or new evaluation.
- Commit: this section's Git history records the corresponding stage commit.
- Previous semantic stage A commit: `4e17ee3f3f71baa497508437589d11a242d00c8e`.

## Step 2G / C -- authorized narrative enhancement (2026-10-13)

- Baseline: `cd2dd8da5df49594669ec207e57b0572e8ec27c2`; branch: `remote-sensing/step2g-narrative-enhancement`.
- Active source: `CONTENT_STEP2G_MARKED.tex`. Earlier snapshots and original AAAI sources are immutable.
- 2G-C-P2, 2G-C-P3, and 2G-C-P4: revise only Discussion paragraphs 2-4; paragraph 1 is byte-identical. Preserve the established role allocation and shared-reference arguments, synthesize existing matched-input/routing/energy evidence, retain synthetic observation scope and target-derived horizon/well provenance, and remove repeated unsupported-property disclaimers. Prior 2D revision comments and all other earlier markings remain.
- Visibility: entire new/revised paragraphs blue in marked mode; scientific content identical in clean mode.
- Numerical/experimental effect: none. No training or new evaluation.
- Commit: this section's Git history records the corresponding stage commit.
- Previous semantic stage B commit: `853ee0fca71bf13220565b562494ccdf27ec22ab`.


## Step 2H — Final Abstract, Conclusions, Keywords and Highlights (2026-10-13)

- Source branch: `remote-sensing/step2g-narrative-enhancement` at frozen commit `054b2f3a001f36242c6280b5da43fd2ee4627d15`. New independent branch: `remote-sensing/step2h-final-narrative`. The active entry `main.tex` now inputs `CONTENT_STEP2H_MARKED.tex`; Step 2G and all historical snapshots remain unchanged.
- `2H-ABSTRACT`: replace the prior single-paragraph Abstract with the author-reviewed full manuscript Abstract, marking the entire revised paragraph using `\BGRevAdd` inside MDPI's `\abstract` front-matter macro. Scientific results and matched-input protocol derive from existing Step 2G evidence; no experimental or numerical data changed.
- `2H-CONCLUSIONS`: replace the sole Conclusions paragraph with the author-reviewed version, fully wrapped in `\BGRevAdd`. All text outside that single conclusion paragraph is unchanged in the active scientific content file.
- `2H-KEYWORDS`: replace the unapproved placeholder with six topic-specific keywords, fully marked using `\BGRevAdd` inside MDPI's `\keyword` argument.
- `2H-HIGHLIGHTS-MF1`, `2H-HIGHLIGHTS-MF2`, `2H-HIGHLIGHTS-IM1`, `2H-HIGHLIGHTS-IM2`: replace the four placeholder Highlight bullets with author-approved, evidence-grounded text in the existing two-category format. Each new bullet is fully marked by `\BGRevAdd`. The Highlights list is locally typeset with `\small` to prevent overlap with the MDPI copyright notice while the Abstract retains its normal font size. The question headings remain unmodified.
- `2H-ENTRY`: administrative input-path change in `main.tex`, marked by source comment. Prior M1/M3/Step 2C–2G blue markings and figure borders are preserved.
- Complete preservation: Introduction, Related Work, Materials and Methods, Results, Discussion, original figures, equations, algorithms, comparisons, experimental values, references and Appendices A–D unchanged. No new experiments, dataset-construction contributions or newly invented claims. Title and unconfirmed author metadata unchanged.
- Compilation uses the repository-bundled MDPI class dated 2026-04-14; this is not an independent verification against the previously reported 2026-09-11 ACS v6.5a class.
- Commit: recorded in Step 2H branch Git history.

## Step 2H — Findings-Oriented Highlights refinement (2026-10-13)

- Source: Step 2H commit `1116e47051f054dc3d07b936d5703e6f6809924f`; independent review branch `remote-sensing/step2h-highlights-findings-20261013`.
- ID `2H-HIGHLIGHTS-R1-MF1`: replace the first Main Findings bullet with a high-level finding about background-guided residual Flow Matching's matched-input accuracy and model size; the claim remains scoped to evaluated baselines.
- ID `2H-HIGHLIGHTS-R1-MF2`: replace the second Main Findings bullet's metric listing with the evidence-supported contribution of prediction-consistent supervision, explicit background conditioning and role-aware fusion.
- ID `2H-HIGHLIGHTS-R1-IM1`: clarify that the predicted background is an observation-dependent, shared reference for supervision, conditional transport and reconstruction.
- ID `2H-HIGHLIGHTS-R1-IM2`: express the broader methodological implication as complementary background-estimation and structure-correction roles.
- All four changes are confined to the text arguments of the existing `\\BGRevAdd` bullet wrappers inside `main.tex`; full blue markings, the two MDPI question headings, layout macros, title, Abstract, Keywords, Conclusions, full active body, references, tables, all results, equations, figures, and appendices are unchanged.
- Historical Step 2H source and snapshots remain immutable. This ledger-only change records the four highlights edits in accordance with the permanent revision rule; no scientific data or model changes.

## Step 2I — MDPI Back Matter and appendix formatting (2026-10-15)

- **Scope/parent:** Format-only revision based on `86eafe3c8d2e897271302d2ced8e4c0712b397ff` from `remote-sensing/step2h-highlights-findings-20261013`. New independent branch `remote-sensing/step2i-backmatter-layout`. `main.tex` now selects the new immutable snapshot `CONTENT_STEP2I_MARKED.tex`; all Step 2H and earlier sources remain unchanged.
- **2I-BACKMATTER:** Move the seven existing administrative declarations as complete, byte-identical LaTeX commands from after `\\input` in `main.tex` to the end of Conclusions in the active content before the appendices. Keep every administrative placeholder and each of the existing `Not applicable.` texts unchanged. The resulting section order is Conclusions → Author Contributions → Funding → Institutional Review Board Statement → Informed Consent Statement → Data Availability Statement → Acknowledgments → Conflicts of Interest → Appendices A–D → References. Do not invent Supplementary Materials or glossary text.
- **2I-APPENDIX-NUMBERING:** Remove the three explicit `\\numberwithin{equation}{section}`, `\\numberwithin{figure}{section}`, and `\\numberwithin{table}{section}` overrides, and use the existing bundled MDPI class `\\appendixstart`/ `\\appendix` continuous prefix-`A` counters. Section and subsection headings retain the letter-based appendix titles. Scientific appendix text, every `\\label`, equation and figure/table body are unchanged.
- **2I-HYPERREF:** Assign unique appendix-only hyperref destination names to equation/figure/table counters (using `\\theH...`) to avoid duplicate PDF targets when counters reset at the start of Appendix A. This is link formatting only; displayed numbers are defined by the MDPI class.
- **2I-REFERENCES:** Move the sole `\\bibliography{references}` to after the `\\ifaaaiwithappendix ... \\fi` block, with one `\\clearpage` to flush all appendix floats before the reference list. Citations use the unchanged `references.bib` and publisher bibliography style. The bibliography is unconditional in both appendix modes; the existing appendix-on mode is the publication preview. If appendices are excluded, references to their intentionally omitted content cannot resolve; no artificial labels or fictitious supplementary material were introduced.
- **2I-ENTRY:** Change only the `main.tex` input path and remove the relocated declarations; author order, title, Abstract, Keywords, Highlights, and their macros remain unchanged.
- **Revision markings:** All existing `\\BGRevAdd` and `\\BGRevFigure` macros are retained untouched. Changes are administrative location/counter/bibliography placement and therefore use stable `BG-REV` source comments and this ledger; no scientific text has been rewritten.
- **Formatting authority:** MDPI Style Guide at `https://www.mdpi.com/authors/layout`, §§ 8, 8.10, 8.11, and Remote Sensing instructions `https://www.mdpi.com/journal/remotesensing/instructions`. Local verified `Definitions/mdpi.cls` declares 2026-04-14; no claim of applying the previously mentioned 2026-09-11 v6.5a template.
- **Preservation test:** The edit is mechanically reversible to the parent main/body; 53 original labels, original citation keys, original data and all Appendix A–D scientific text are retained. Full marked/clean PDF builds and hyperlink/numbering checks are separate verification artifacts.

- **2I-APPENDIX-SUBHEADINGS:** MDPI §8.10 requires numbered appendix subheadings. Replace the existing appendix-only `\\setcounter{secnumdepth}{1}` with level 3 and a `BG-REV` marker so the bundled MDPI class displays `Appendix A.1`, `Appendix B.4.1`, etc. This changes heading numbering and autogenerated section-reference displays only; all heading text, section content, equations and labels remain unchanged.

## Step 2I — Author-supplied generative-AI use declaration (2026-10-15)

- **Source and branch:** Start from Step 2I scientific-layout commit `18e1aee3016e5ac55641015b67d176ae3047cd5e`; independent branch `remote-sensing/step2i-ai-disclosure-20261015`.
- **2I-AI-DISCLOSURE:** Replace the pending Acknowledgments/AI placeholder with the author's exact supplied ChatGPT-use statement, without paraphrasing, in the existing `\\acknowledgments{...}` Back Matter location between Data Availability and Conflicts of Interest. The complete new statement uses `\\BGRevAdd{...}`, appearing blue in marked PDFs and ordinary text in clean PDFs.
- **2I-AI-ENTRY:** Switch `main.tex` to an independent snapshot `CONTENT_STEP2I_AI_DISCLOSURE_MARKED.tex`; keep `CONTENT_STEP2I_MARKED.tex` and every earlier snapshot immutable.
- **Outstanding editorial matter:** Any non-AI acknowledgments remain pending author confirmation, recorded as an adjacent source comment rather than inventing an acknowledgment. The substantive truth and completeness of the user-supplied declaration must be confirmed by all authors before submission.
- **Preservation:** Every scientific passage, figure/table, numerical result, equation, citation, appendix, other Back Matter declaration, authorship metadata, title, Abstract, Highlights and previous change marking remains unchanged. There is exactly one added complete blue-text paragraph and one main-input path update. No new research data or experiments.
- **Compilation:** Marked and clean mode each verified against repository-bundled MDPI class dated 2026-04-14; separate GitHub Actions checks may also be performed. No template migration or submission initiated.

## Step 2J — Scientific main-text expansion (2026-10-16)

- **Frozen source:** `remote-sensing/step2i-ai-disclosure-20261015` at `7c7fb72d9d3678d0fce6c8c06c9e6f128090a6c7`. **Independent branch:** `remote-sensing/step2j-maintext-18pages`. Active `main.tex` now inputs `CONTENT_STEP2J_MARKED.tex`, leaving all earlier source snapshots untouched.
- **2J-E1:** A new observation-contract paragraph within `Materials and Methods > Experimental Setup` before the original Implementation Details paragraph, explaining the scientific significance of time-domain PoSTM/RMS versus depth-domain interpreted horizons/sparse wells, and why matched-input comparison controls input information. Dataset construction remains credited to prior SeisMIC-Diff/EAAI work; no new dataset contribution or pipeline description.
- **2J-A1, 2J-A2:** Two new Results paragraphs immediately after the `Paradigm-level benchmark context` paragraph. Interpret the existing PD-BG-RFM OpenFWI subset results by geological family and A/B split while explicitly keeping the source-native literature comparison separate from a matched-input comparison. No numerical values are recomputed or changed.
- **2J-B1, 2J-B2:** Two new Results paragraphs after `Matched-input estimator comparison`; interpret the joint RMSE, SSIM and high-frequency MAE evidence and the 10.6M model capacity relative to adapted baselines, without claiming reduced inference latency or universal superiority.
- **2J-C1, 2J-C2:** Two Results paragraphs before the `Additional condition, background-residual...` appendix-pointer sentence. Integrate distinct diagnostic questions: tuple alignment, role specialization, reliability routing regularized by fixed priors, and conditional residual target energy. No claims of autonomous physical factor discovery or guaranteed ODE efficiency.
- **2J-D1, 2J-D2:** Two new Discussion paragraphs immediately before the existing final paragraph beginning `These findings concern depth-domain velocity reconstruction...`. Link training-only effective-rank analysis to estimator responsibility allocation and interpret how the processed numerical and structural constraints jointly serve a depth-domain target. Paragraph D2 was condensed for a nonredundant, coherent transition.
- **Blue review marking:** All nine complete new paragraphs are wrapped in `\\BGRevAdd` with stable identifiers. All prior 27 `\\BGRevAdd` uses and original figure frames remain exactly as in Step 2I. No original scientific sentence, mathematical statement, loss, theorem/proof, figure/caption, table/value, citation/bibliography entry, appendix, author declaration, Abstract, Keywords, Highlights or Conclusions was edited.
- **Reversible change audit:** All new paragraphs can be removed using their nine identifiers, and the main input selector can be restored to reproduce the frozen Step 2I source exactly. The compiled baseline has Conclusions on physical page 16, Appendix A on 17, references on 31, and 33 total PDF pages. The locally compiled Step 2J (identical paper class, fonts, figure assets, and layout settings) has Conclusions completely on physical page 18, Appendix A starting page 19, references starting page 33, and 35 total pages. No font, margin, spacing, float or page-break manipulation was performed.
- **Verification:** LaTeX mark/clean builds, citation/hyperlink/layout and PDF page measurements are captured separately. Bundled MDPI `Definitions/mdpi.cls` is dated 2026-04-14, not independently verified as the 2026-09 v6.5a template. No branch merge or submission is authorized.
- **Commits:** See Step 2J branch history.

## Step 2K — Eight-subset qualitative figure replacement (2026-10-16)

- **Frozen source:** `remote-sensing/step2j-maintext-18pages` at commit `0326ea5d3d0794d70b5038707bea7d980276146a`. Independent branch `remote-sensing/step2k-full-qualitative`. `main.tex` now selects `CONTENT_STEP2K_MARKED.tex`; the Step 2J snapshot remains immutable.
- **2K-ASSET:** Use author-provided original two-page PDF `vis(1).pdf` without rasterization or alterations as `figures/aaai/qualitative_eight_subsets.pdf`. Exact asset SHA-256 `81425f52b25e45402b794382f0342602332d653d1bd446bc7ca633bf04a1e216` (367744 bytes). Page 1 FlatVel-A/B and CurveVel-A/B; page 2 FlatFault-A/B and CurveFault-A/B. Original representative maps visually and numerically agree with historical appendix files despite differing panel order/layout; formal record IDs are not embedded in the author PDF.
- **2K-REMOVE-OLD-F3:** Remove prior Methods-stage `figure*` using `figures/aaai/qualitative.pdf` and label `fig:adapted-qualitative`; retain the historical PDF asset unchanged.
- **2K-FIG3:** New independent full-width qualitative `figure*` for page 1 of the author-supplied PDF, label `fig:qualitative-nonfault`, blue author-review border and complete blue caption.
- **2K-FIG4:** New independent `figure*` for PDF page 2, label `fig:qualitative-fault`, blue border and caption. Both figures are placed in Results after matched-input quantitative and model-capacity analysis, using `keepaspectratio` without image data regeneration.
- **2K-RESULTS-COMPARE:** Fully blue revised matched-input paragraph references both new figures and all eight formally selected subset examples; source-native Table 1, matched-input Table 2, and all metrics remain untouched.
- **2K-APPENDIX-REMOVE-A / 2K-APPENDIX-REMOVE-B:** Remove the two duplicate appendix representative `figure*` environments with labels `fig:table2-representative-a` and `fig:table2-representative-b`. Historical `table2_representative_cases_1.pdf` and `table2_representative_cases_2.pdf` remain present in figure assets.
- **2K-APPENDIX-INTRO:** Blue revised Appendix B qualitative subsection opening refers to the two promoted main figures and the retained fixed-rule diagnostic figure `fig:table2-failures`.
- **2K-APPENDIX-SELECTION:** Blue revised Appendix D qualitative selection description replaces the legacy “three main-text cases” statement with eight subset-level representatives selected using the unchanged predefined per-subset maximum formal SSIM-margin rule, keeping all five adapted references and the remaining negative-margin diagnostic cases.
- **Preservation:** All previously approved 36 blue prose markings and one pre-existing blue figure border retained; no original comparison/ablation data, equations, algorithms, proofs, methods, Highlights, Abstract, author declaration, bibliography, appendix scientific diagnostics or model parameters are changed. No original PDF asset files deleted.
- **Rendering:** Marked/Clean PDF compilation results, figure pages, exact labels and any figure legibility limitations are recorded in Step 2K verification artifacts. The existing bundled MDPI class (dated 2026-04-14) is retained; this is not a new official-template migration.
- **2K-FLOAT-ORDER:** Keep the two large qualitative figures associated with the matched-input Results subsection by using the standard `placeins` package and a single `\FloatBarrier` immediately before the following Condition and Transport Diagnostics subsection. The figures remain independent MDPI `figure*` float environments, now visibly preceding the subsequent subsection. This is reading-order/layout control, not a scientific or page-count modification; the total PDF count is unchanged.
