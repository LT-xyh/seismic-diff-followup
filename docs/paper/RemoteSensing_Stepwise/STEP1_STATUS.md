# Step 1 — Official MDPI Remote Sensing format-only migration

**STATUS: LOCAL PDF COMPILE PASS; FULL GITHUB ZIP INSTALL PENDING; PI REVIEW REQUIRED.**

The user supplied the genuine **MDPI ACS LaTeX ZIP**, latest from the publisher's
https://www.mdpi.com/authors/latex download page. The uploaded archive was opened,
authenticated by its contents, and compiled in the local deliverable.

## Verified publisher template

- Package: official MDPI ACS LaTeX source (not a third-party mirror).
- ZIP SHA-256: `62744425fbcec9cd3e58147cbee65bdec2e2ff0440f29792c26edc97a11b6c70`
- `Definitions/mdpi.cls`: **v6.5a**, date **2026-09-11**.
- Class SHA-256: `658dbb5b2db2f6560bf3de3ecff7efac310a5817721eebd7539c1990b5345f01`
- Class invocation: `\documentclass[remotesensing,article,submit,moreauthors]{Definitions/mdpi}`.
- BibTeX: official numeric ACS `Definitions/mdpi.bst`; original source bibliography entries unchanged.

## Files in this branch

- `original/pd-bg-rfm.tex`: exact AAAI primary source.
- `original/paper0726.tex`: exact earlier AAAI reference.
- `ABSTRACT_EXACT_FROM_AAAI.tex`: exact original abstract.
- `CONTENT_EXACT_FROM_AAAI.tex`: exact original body, including scientific appendices.
- `CONTENT_MDPI_LAYOUT.tex`: mechanically adjusted body ONLY:
  - official `\appendixstart` numbering added to the historical `\appendix`;
  - two wide tables wrapped in MDPI `adjustwidth`, changing only one
    `tabular*` width specification from `\textwidth` to `\linewidth`;
  - no added/removed scientific paragraphs, equations, captions, table data, citations or claims.
- `main.tex`: actual MDPI v6.5a macros with confirmed three-author order and
  clearly marked **unconfirmed** affiliations, correspondence details, CRediT,
  funding, conflicts and acknowledgment placeholders.
- `figures/aaai/`: nine original figure exports (eight user-supplied PDFs plus appendix PNG).
- `references.bib`: byte-identical historical BibTeX.
- `tools/install_uploaded_mdpi_acs.py`: install the exact verified uploaded
  official ZIP into this workspace, without altering any scientific source.

## Local compile and preservation audit

- Successfully built from an independent clean source directory using the actual
  downloaded MDPI class and graphics: **28 pages, A4**.
- Official Highlights headings present; their scientific bullet contents remain
  placeholders, deliberately unauthored during format-only Step 1.
- Original main source SHA-256: `51d57b5954516e0040114aee9a3863a4ea5869b78500cde7d4112d6edf4d7e83`.
- Scientific body: **75,230 characters**, identical after reversing the three
  types of purely structural formatting changes and normalizing wrapper whitespace.
- Equations: 26 `equation` plus 11 `align` blocks unchanged;
  tabular content unchanged (5 `tabular` plus 3 `tabular*`).
- Citations: 66 occurrences, 39 unique keys; all BibTeX citations resolved.
- Cross-references: 50 labels and 69 `ref/eqref` references unchanged.
- No LaTeX undefined citations/references, overfull boxes, duplicate PDF destinations,
  or text outside page bounds.
- Original figure bytes and scientific numerical results remain identical to Step 0.
- The `https://doi.org/10.3390/rs1010000` printed in the MDPI preview footer is
  **publisher template dummy metadata**, not a registered or assigned DOI.

## Working-tree dependency vs downloadable deliverable

The exact official ZIP was uploaded **into the ChatGPT conversation**, not into GitHub.
The branch tracks `main.tex`, the exact science and formatting changes,
metadata provenance and an installer, but cannot yet run the MDPI class build
directly on GitHub without the binary `Definitions/` assets.

The downloadable **BG-RFM_AAAI_to_MDPI_STEP1_SOURCE.zip** prepared in this conversation
DOES include the official 2026-09-11 class/style assets and has already passed
independent recompilation. For an equivalent GitHub CI run, **do not commit the official MDPI
ACS ZIP to this public repository**. The Step 1.1 workflow now attempts
publisher-hosted acquisition first and, if HTTP 403 persists, offers a
private read-only GitHub Release dependency using an author-controlled secret.

See `STEP1_1_BUILD_GATE.md` for exact release/tag, Actions variables,
fine-grained read-only token, immutable SHA-256 checks and steps to trigger
the GitHub build. Local reproducibility remains available using the
already verified private author-supplied ZIP.

**DO NOT** merge into `remote-sensing/submission-sprint`, create a final tag,
submit to SUSY or begin scientific Step 2 without explicit PI approval.

## Step 1.1 CI update — 2026-10-10

- Official MDPI ACS v6.5a (2026-09-11) ZIP and class SHA-256 verified.
- Original MDPI Highlights macro hook and template redefinition pattern
  verified; scientific Highlights content remains unapproved.
- `step1-content-preservation`: **PASS** on the Step 1.1 branch.
- `step1-official-mdpi-build`: **BLOCKED** because the publisher's official
  direct URLs return HTTP **403 Forbidden** from GitHub Actions. Private
  dependency fallback is implemented but requires an author-configured
  read-only private release and corresponding GitHub Actions secret.
- No GitHub-side MDPI PDF was generated on the blocked run. The previously
  independently compiled **28-page local MDPI PDF** remains valid but must
  not be described as a CI-produced PDF.
- Workflow results and sanitized blocker report:
  https://github.com/LT-xyh/seismic-diff-followup/actions/runs/38035656453

No science, historical manuscript, original figures, citations or experiment
results were modified by Step 1.1.
