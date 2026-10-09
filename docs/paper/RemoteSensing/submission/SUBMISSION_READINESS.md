# Remote Sensing submission readiness (2026-10-09)

**PREVIEW ONLY — NOT READY FOR SUSY; final commit and tag intentionally withheld.**

## Official requirements and status

Remote Sensing author guidance: https://www.mdpi.com/journal/remotesensing/instructions
MDPI ethics / AI guidance: https://www.mdpi.com/ethics
Highlights requirement announcement: https://www.mdpi.com/about/announcements/13077

| Requirement | First submission | Current status |
|---|---|---|
| Journal scope, title, Abstract, keywords, methods, results, captions, references | Mandatory | Scientifically frozen and present |
| Highlights: main findings / implications, max two bullets each | Mandatory for new Articles | Updated |
| Author names/order, affiliation, correspondence, author approval | Mandatory | **PENDING AUTHOR CONFIRMATION** |
| CRediT roles for multi-author work | Mandatory | **PENDING** |
| Funding, COI and required ethics statements | Mandatory | **PENDING for author-dependent declarations** |
| Data Availability Statement | Mandatory | OpenFWI source verified; exact eight-subset processed data access **PENDING** |
| Code Availability | Required/recommended for developed code | Public source and frozen scientific commit identified; final release tag **PENDING** |
| Cover letter including author approval and non-simultaneous submission declarations | Mandatory | Draft only; declarations not signed |
| GenAI disclosure for substantive drafting or research uses | Mandatory when applicable | Draft evidence of ChatGPT use; complete tool/version and author approval **PENDING** |
| LaTeX ZIP with all active source and graphics, re-compilable | Mandatory when submitting LaTeX | Minimal packaging script and isolated CI build configured |
| MDPI journal LaTeX template | Recommended; not obligatory for free-format initial submission | Free-format retained; template migration optional at initial submission |
| Article biographies / SciProfiles | Optional | Not included |
| Related SeisMIC-Diff/EAAI publication / same-manuscript prior MDPI submission disclosures | Required where applicable | Evidence comparison drafted; current status and overlap **PENDING** |

## Frozen scientific content

Baseline: b6bdebb94f4bf2676ffbcadddd60668cfa514b50.
No model training, model evaluation, scientific numerical change or new baseline is authorized.

Main manuscript remains one PDF with Appendices A-C; no separate supplemental PDF.
Duplicate PDF hyperlink destination warnings were eliminated by the hypertexnames=false option with unchanged visible numbering (to be verified in CI output).

## Finalization gate

Complete AUTHOR_CONFIRMATION_REQUIRED.md, replace preview placeholders in the three LaTeX submission modules, approve the related-manuscript/editor declaration and cover letter, then rebuild and check. Only after these confirmations should the final submission commit/tag be created. DO NOT submit automatically.
