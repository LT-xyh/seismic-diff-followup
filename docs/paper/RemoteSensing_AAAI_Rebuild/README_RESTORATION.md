# Original AAAI-27 manuscript restoration — Step 0

**STATUS: RECONSTRUCTED FROM ARCHIVED SOURCE, NOW USING USER-SUPPLIED PDF FIGURE EXPORTS.**

The exact AAAI **submitted manuscript PDF** remains unavailable in the bounded repository and uploaded project material. Accordingly, the rebuilt PDF is **not verified equivalent to the submitted manuscript PDF**, even though the figure exports are now available.

## Byte-exact historical sources

The following files were copied without modification from `docs/paper/aaaii/`:
- `pd-bg-rfm.tex`
- `paper0726.tex`
- `references.bib`
- `aaai2027.sty`
- `aaai2027.bst`
- `ReproducibilityChecklist.tex`

Source SHA-256:
- `pd-bg-rfm.tex`: `51d57b5954516e0040114aee9a3863a4ea5869b78500cde7d4112d6edf4d7e83`
- `paper0726.tex`: `5c933cb974ba83b47b3394e11a3b518818832aa8f2315e2210a70d9f721710e1`

The original AAAI title, abstract, claims, equations, tables, numbers, references, and section structure are preserved exactly. **No Remote Sensing scientific corrections are imported.**

## Original AAAI PDF figures — uploaded 2026-10-10

The author provided eight historical PDF graphic exports via the branch
`xyh/upload-aaai-pdf-figures`, commit
`dc1e70dab42ad01f043961a01711ae8f4e950a48`.

All eight PDF exports now exist **as original Git blobs** at each of:
- `docs/paper/aaaii/figures/aaai/`
- `docs/paper/AAAI2027/figures/aaai/`
- `docs/paper/RemoteSensing_AAAI_Rebuild/figures/aaai/`

The eight files are:
- `motivation.pdf`
- `method_overview.pdf`
- `qualitative.pdf`
- `condition_learning.pdf`
- `residual_transport.pdf`
- `table2_representative_cases_1.pdf`
- `table2_representative_cases_2.pdf`
- `table2_failure_cases.pdf`

An additional `appendix_full_condition_diagnostics.png` is copied byte-for-byte from the original AAAI artwork. Editable SVG, PPTX, and PNG source materials are preserved under `original_artwork/`.

**Integrity:** The eight supplied PDFs all parse successfully (one page each); for every filename all three current directories hold byte-identical content. `PROVENANCE.json` records each Git blob SHA, SHA-256, file size, and the original `dc1e70d` upload commit. Prior generated SVG/PNG-to-PDF export provenance is retained inside `previous_provisional_export` for audit purposes, but those provisional PDF exports are no longer the compiled figures.

**Important limitation:** These are verified files supplied on the stated upload branch; their inclusion does not prove which specific PDF rendering was included in the *submitted manuscript PDF*, which remains unavailable.

## Compiling the historical manuscript

From this directory:
```bash
latexmk -pdf pd-bg-rfm.tex
```

The rebuilt output uses the **unaltered original AAAI manuscript** and the supplied eight PDF figures. No MDPI formatting or Step 1 scientific editing is performed. The rest of the Remote Sensing work and existing author-confirmation records remain unchanged.

## Workflow

- `aaai-step0-original-restoration.yml` performs read-only byte/hashes checks, validates PDF files, compiles the manuscript and uploads artifacts.
- `audit_aaai_uploaded_pdfs.py` verifies all three copies of each uploaded file and produces an updated immutable-source SHA-256 inventory.
- `SHA256SUMS.txt` lists the immutable files inside this restored workspace (not mutable metadata or LaTeX build outputs).

**STOP after Step 0.** Do not begin scientific revision without authorization.
