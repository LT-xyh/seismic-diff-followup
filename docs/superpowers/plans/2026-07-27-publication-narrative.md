# AAAI 2027 Publication Narrative Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize the manuscript around the strongest publishable claim: multimodal velocity inversion benefits from assigning deterministic prediction to the well-constrained background and Flow Matching to the remaining structure-bearing variation.

**Architecture:** Keep `paper.tex` as the single manuscript source. The main text will present the problem, routed solution, exact identities, matched-input advantage, and component attribution in that order. Reproducibility details and unavoidable mathematical scope statements remain in the appendix rather than interrupting the central narrative.

**Tech Stack:** LaTeX, AAAI 2027 style, pdfLaTeX/BibTeX, `git diff`, and `pdftotext`.

---

### Task 1: Establish the publication narrative

**Files:**
- Modify: `docs/paper/AAAI2027/paper.tex:42-75`

- [ ] Rewrite the abstract and introduction transitions so the asymmetry, routed estimator assignment, prediction consistency, theorem-backed energy criterion, and strongest matched-input results appear as one forward argument.
- [ ] Replace defensive or process-oriented phrases with direct evidence statements; retain factual comparison scope and quantitative results.

### Task 2: Focus Method on the contribution mechanism

**Files:**
- Modify: `docs/paper/AAAI2027/paper.tex:84-309`

- [ ] Present the condition contract, prediction-relative residual, vector-field objective, and inference composition as the core mechanism.
- [ ] Rename low-pass `burden` language to neutral residual-content language, remove duplicated prose, and move nonessential caveats out of the main derivation while preserving theorem assumptions and equations.

### Task 3: Make experiments read as evidence

**Files:**
- Modify: `docs/paper/AAAI2027/paper.tex:318-407`

- [ ] Frame Table 1 as paradigm context, Table 2 as the matched-input advantage, diagnostics as contract/energy verification, and Table 3 as component attribution.
- [ ] Keep all reported values and fair-comparison facts, but make the prose foreground the strongest supported findings and avoid a failure-log tone.
- [ ] End the conclusion with the method and general design principle, without adding new limitations.

### Task 4: Preserve technical accountability in the appendix

**Files:**
- Modify only where needed: `docs/paper/AAAI2027/paper.tex:411-741`

- [ ] Retain proof assumptions, protocol provenance, checkpoint details, and diagnostic definitions in the appendix.
- [ ] Rephrase only statements that unnecessarily undermine the main claim; do not remove reproducibility or validity conditions.

### Task 5: Verify the manuscript

**Files:**
- Verify: `docs/paper/AAAI2027/paper.tex`
- Verify: `docs/paper/AAAI2027/_build_full/paper.pdf`

- [ ] Run `git diff --check` and targeted scans for stale terminology, duplicated paragraphs, and accidental numerical changes.
- [ ] Run `bash build.sh full`, inspect the exit status and log, and confirm the generated PDF and cross-references.
