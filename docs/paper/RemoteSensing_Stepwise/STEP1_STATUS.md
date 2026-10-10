# Step 1 — MDPI format migration (scientific content frozen)

**STATUS: PARTIAL — OFFICIAL MDPI CLASS BLOCKER. NOT A COMPILED MDPI MANUSCRIPT.**

## Independently completed

- Created a separate branch from remote-sensing/aaai-stepwise-rebuild without merging into the submission or editorial branches.
- Restored exact AAAI main source `original/pd-bg-rfm.tex` (SHA-256: `51d57b5954516e0040114aee9a3863a4ea5869b78500cde7d4112d6edf4d7e83`).
- Preserved earlier reference `original/paper0726.tex` (SHA-256: `5c933cb974ba83b47b3394e11a3b518818832aa8f2315e2210a70d9f721710e1`).
- Preserved the original `references.bib` as a byte-exact copy and all nine original graphical references, including eight user-uploaded original figure PDFs.
- Extracted the exact original abstract and all original body text, including the appendices and scientific sections, without rewriting or changing any numerical values. See `ABSTRACT_EXACT_FROM_AAAI.tex` and `CONTENT_EXACT_FROM_AAAI.tex`.
- Added a strict scientific-preservation auditor.

## Official 2026 MDPI template blocker

As of 2026-10-10, MDPI's official author page lists **11 September 2026** as the last-update date for the ACS, APA and Chicago LaTeX packages: https://www.mdpi.com/authors/latex.

Official-page and official ZIP download attempts in the GitHub runner both returned HTTP **403 Forbidden**:
- `https://www.mdpi.com/authors/latex`
- `https://www.mdpi.com/data/MDPI_template.zip`
- `https://www.mdpi.com/data/MDPI_template.zip?v=20260911`

Because the user explicitly forbids outdated or unofficial class files, **no third-party MDPI class has been substituted**. The main MDPI `main.tex` is deliberately NOT generated until the authentic current archive is available.

**Required input:** the ACS citation-style ZIP directly downloaded from `https://www.mdpi.com/authors/latex`, last updated 2026-09-11. Please upload this ZIP, including `template.tex` and `Definitions/` directory.

## Mandatory format operations once official ZIP is supplied

1. Record exact official archive SHA-256 and class metadata/version; retain original ZIP.
2. Use the original template's genuine `remotesensing, article, submit` class options and citation syntax, with confirmed author order Chunlei Wu, Yinghao Xu, Jing Lu (corresponding).
3. Add unconfirmed affiliation, correspondence, ORCID, CRediT, Funding and acknowledgment placeholders without fabricating details.
4. Provide two **empty content placeholders** headed 'What are the main findings?' and 'What are the implications of the main findings?' (Highlights are not to be authored in this format-only step).
5. Convert only incompatible AAAI structural syntax (title, abstract, class commands, bibliography invocation, theorem/algorithm/float environments and appendix counter displays). Preserve the scientific substrings verbatim whenever structurally possible.
6. Include all nine unchanged original figure objects and `references.bib` without altering entries or citation keys.
7. Compile the true MDPI class with pdfLaTeX/BibTeX and perform a semantic diff with the original source; report any token-level changes in scientific content as an error.
8. Check rendered math, figures, appendices, captions, tables and overfull boxes; package a self-contained MDPI ZIP.
9. Return for PI review; DO NOT merge, tag or submit. Do not begin Step 2.

## Prior historical publication PDF

The actual submitted AAAI PDF was later provided, and the Step 0 original-source reconstruction was compared in an earlier turn with no substantive visual differences. Step 1 deliberately uses the restored **AAAI LaTeX source**, not the previously rewritten Remote Sensing prose.
