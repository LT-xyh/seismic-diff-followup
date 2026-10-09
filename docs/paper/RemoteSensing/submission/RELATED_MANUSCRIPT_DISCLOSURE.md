# Related SeisMIC-Diff / EAAI manuscript: editorial disclosure assessment

## Inspected documents

BG-RFM is frozen at scientific commit b6bdebb94f4bf2676ffbcadddd60668cfa514b50, with a 24-page single PDF and Appendices A-C.

The private SeisMIC-Diff EAAI repository (LT-xyh/SeisMIC-Diff-EAAI) includes a manuscript at paper/eaai/manuscript/seismic_diff_eaai.tex, a dataset description at docs/DATA.md, and an archived submission dated 2026-07-06. That archive date is not evidence of its current review status.

The user's supplied SeisMIC-Diff PDF is another related draft headed for Computers & Geosciences and is not evidence of the current journal/status.

## Evidence-grounded comparison

| Dimension | BG-RFM | SeisMIC-Diff / EAAI |
|---|---|---|
| Domain | Processed-constraint initial velocity reconstruction | Heterogeneous geophysical evidence for initial velocity reconstruction |
| Inputs | RMS velocity, PSTM, horizons, sparse wells | Same four modality families |
| Method | Deterministic background + prediction-relative residual Flow Matching | Low-compression VAE + modality-specific reliability allocation + conditional latent DDPM |
| Source datasets | Eight OpenFWI-derived subsets | FlatVel-A/B and CurveVel-A/B: four subsets |
| Evaluated split | Global 70/20/10, split seed 42 | 67.5/22.5/10 under index-holdout plus random train/validation division |
| Evidence | 33,600-record matched-input evaluation | Four-subset latent-diffusion comparisons and modality/encoder ablations |
| Publication | Remote Sensing manuscript under preparation | Dated EAAI submission archive; current publication/review state unverified |

Both works share the scientific task and four interpreted-input families, and at least four underlying OpenFWI structural subsets. Whether they reuse exactly the same processed arrays, plots, diagrams, paragraphs or experiment records has not been independently established.

## Editorially relevant confirmations

- Confirm the CURRENT journal, date, title, manuscript ID, DOI/preprint status, and accepted/under-review/withdrawn/rejected status of the EAAI paper.
- Identify any exact overlapping datasets, figures, numerical tables, or text. Provide copies of related work to the editor if required.
- Explain the distinct method and empirical contribution without making a blanket no-overlap or duplicate-publication claim.
- Check related SeisMIC-Init dataset publications and any prior SEG 2025 workshop abstract for disclosure/citation as appropriate.

Recommendation pending confirmation: include a concise related-work disclosure in the editor's cover letter (or separately in SUSY), stating the shared task/data context and differences in model formulation, with the current related manuscript status. This is a precaution, not a determination of redundant publication.
