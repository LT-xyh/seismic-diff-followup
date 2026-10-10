# Step 1.1 — reproducible official MDPI template build gate

Status: **BLOCKED pending a private, authorized template dependency**.
The compiled 28-page local MDPI PDF remains scientifically preserved and valid as a
local preview; this file does **not** claim a passing GitHub-side MDPI compile.

## Exact publisher provenance (already verified from the author's ZIP)

| Identifier | Value |
|---|---|
| Official URL | https://www.mdpi.com/authors/latex |
| Publisher direct ZIP | https://www.mdpi.com/data/MDPI_template.zip |
| MDPI class | v6.5a, 2026-09-11 |
| Official ZIP SHA-256 | `62744425fbcec9cd3e58147cbee65bdec2e2ff0440f29792c26edc97a11b6c70` |
| Definitions/mdpi.cls SHA-256 | `658dbb5b2db2f6560bf3de3ecff7efac310a5817721eebd7539c1990b5345f01` |

The supplied author's official ZIP has both exact digests. Older Overleaf
snapshots or public third-party copies are unacceptable substitutions.
The publisher download failed under GitHub Actions with HTTP 403 Forbidden,
including both the author page and the direct file endpoint (run
`38033108307`). The new workflow tries publisher endpoints again and,
only if needed, an authenticated author-controlled private dependency.

## Compliant private dependency — owner setup

The actual ZIP was uploaded to the ChatGPT conversation, **not** to GitHub.
For a reproducible GitHub-side build without distributing the publisher's
templates in a **public Git repository**:

1. Create/use a **private** GitHub repository for author-approved manuscript
   build dependencies, subject to the publisher's license and permitted use.
2. In that private repository, create a release tagged
   `mdpi-acs-2026-09-11`, with an asset named exactly
   `MDPI_template_ACS.zip`. Upload the unchanged archive whose ZIP
   SHA-256 equals the specified expected hash. Do not use a public release.
3. Create a fine-grained GitHub personal access token limited to the
   **private dependency repository** with **Contents: Read-only**.
4. In `LT-xyh/seismic-diff-followup` → Settings → Secrets and variables
   → Actions, set:
   - Repository variable `MDPI_ACS_PRIVATE_REPO` =
     `OWNER/PRIVATE-DEPENDENCY-REPO`
   - Repository variable `MDPI_ACS_PRIVATE_TAG` =
     `mdpi-acs-2026-09-11`
   - Secret `MDPI_ACS_PRIVATE_READ_TOKEN` =
     the restricted read-only token.
5. Trigger **step1-official-mdpi-build** with `workflow_dispatch`
   from branch `remote-sensing/step1-mdpi-format` after private storage
   and authorization are confirmed.

The workflow first attempts the publisher. Its private fallback runs
`gh release download` with the read-only token. It verifies both hashes
*before* installing the official `Definitions/` class and styles into the
temporary checkout. It does not persist any public Git blob or Action
artifact containing the official ZIP or class files. No secret values are
logged. Avoid uploading the token/ZIP into issues, PR comments or public
Actions variables.

## Build and verification

- `.github/workflows/step1-obtain-official-mdpi-template.yml`
  (workflow name `step1-official-mdpi-build`) does source-preservation
  checks, verified acquisition, official class install, pdfLaTeX/BibTeX
  via `xu-cheng/latex-action@v4`, then compiled-PDF checks.
- `.github/scripts/acquire_official_mdpi_acs.py` records exact download
  attempts and sanitized statuses in
  `build/step1-mdpi/template-acquisition.json`.
- `.github/scripts/check_step1_mdpi_pdf.py` verifies official class,
  original figure bytes, all four appendix titles, citations, bibliography,
  visible Highlights placeholders, and absence of fatal LaTeX warnings.
- Successful runs upload compilation logs and a PDF-verification JSON for
  **seven days**, excluding the official template ZIP/Definitions.
- Because GitHub Actions artifacts on a public repository may be visible
  to other GitHub users, **the PDF itself is uploaded only with the PI's
  explicit approval**: repository Actions variable
  `MDPI_PUBLIC_PREVIEW_ARTIFACT_APPROVED=true`. Without this authorization
  the compile and checks still run, but only the reports/logs are uploaded.
- Failed/blocked runs upload only sanitized acquisition diagnostics.
- `.github/workflows/step1-content-preservation.yml` independently
  verifies scientific content and artwork without needing the template.

## Highlights implementation

The official MDPI ACS `template.tex` (2026-09-11) explicitly documents
`\\addhighlights{yes}` together with
`\\renewcommand{\\addhighlights}{...}`. The MDPI class actually defines
this hook and renders it after the article front matter. Current
`main.tex` follows this documented **native template pattern**; the
unconfirmed scientific Highlights bullets remain placeholders.
No new scientific Highlights were drafted or approved.

## Immutable boundaries

The format-only main.tex still includes the exact historical AAAI abstract
and body (apart from reversible appendix/wide-table layout commands).
Original AAAI numerical values, equations, figure files and citations are
unchanged. No changes to the RS submission branch, Step-0 originals, author
records, final publication tag, or Step 2 are authorized.

**GitHub MDPI compilation must remain reported as BLOCKED until a run
actually succeeds using the exact official ZIP.**
