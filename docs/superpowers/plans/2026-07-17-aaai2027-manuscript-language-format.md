# AAAI 2027 Manuscript Language and Format Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve the fluency and AAAI-style presentation of the current manuscript without changing formulas, reported values, experimental provenance, claim boundaries, or pending-result status.

**Architecture:** Treat `paper.tex` as the authoritative full manuscript. Apply a conservative language pass section by section, then correct template-facing table, algorithm, and float formatting. Preserve the existing appendix and the intentional deletion of `paper_submission.tex`.

**Tech Stack:** LaTeX, AAAI 2027 author kit, pdfLaTeX/BibTeX, shell-based compliance checks.

---

### Task 1: Normalize the AAAI Preamble and Float Conventions

**Files:**
- Modify: `docs/paper/AAAI2027/paper.tex`
- Reference: `docs/paper/AuthorKit27/AnonymousSubmission2027.tex`

- [ ] **Step 1: Verify the required anonymous-submission declarations**

Confirm that the manuscript retains `letterpaper`, `\usepackage[submission]{aaai2027}`, the required URL, graphics, citation, and caption packages, `/TemplateVersion (2027.1)`, `Anonymous Submission`, and empty affiliations.

- [ ] **Step 2: Align algorithm and figure placement with the template**

Use `[tb]` for Algorithm 1. Keep figures at the top of a column or page. Present the dense qualitative comparison as a centered double-column figure at `0.76\textwidth`, without trimming or clipping in LaTeX.

- [ ] **Step 3: Remove nonessential float barriers**

Retain a barrier only where it prevents the appendix from overtaking main-text floats. Allow Table 2 and Figure 2 to follow normal AAAI top-float placement within their subsection.

### Task 2: Rewrite Main-Text Prose for a Concrete AAAI Style

**Files:**
- Modify: `docs/paper/AAAI2027/paper.tex`

- [ ] **Step 1: Tighten the Abstract and Introduction**

Preserve the asymmetric-responsibility insight, method definition, theoretical scope, and matched-input result claims. Remove repeated formulations of the condition contract and replace awkward constructions with direct subject-verb statements.

- [ ] **Step 2: Improve Method transitions**

Keep the four existing Method subsections and all equations. Make each subsection follow the order motivation, computation, objective, and interpretation. Retain explicit caveats on identifiability, entropy, and transport energy without repeating them in adjacent paragraphs.

- [ ] **Step 3: Improve Experiment and Conclusion transitions**

Separate cross-protocol context, matched-input architectural comparison, controlled attribution, robustness, and diagnostics. Preserve every reported number and all `TBD` cells. Keep the conclusion limited to supported structural-fidelity and efficiency claims.

### Task 3: Make Tables AAAI-Compliant

**Files:**
- Modify: `docs/paper/AAAI2027/paper.tex`

- [ ] **Step 1: Remove sub-9-point table text**

Replace every table-level `\scriptsize` declaration with `\small`. Do not use `\resizebox`, `\tiny`, or negative spacing.

- [ ] **Step 2: Give wide tables sufficient width**

Convert the seven-column matched-input table to `table*` when necessary and use `\textwidth`. Retain `\tabcolsep` compression permitted by the AAAI template. Keep captions below all tables.

- [ ] **Step 3: Check ranking language against table styling**

Ensure bold and underline descriptions match the actual cells, and distinguish parameter-count commentary from metric ranking.

### Task 4: Verify the Full Manuscript

**Files:**
- Verify: `docs/paper/AAAI2027/paper.tex`
- Verify: `docs/paper/AAAI2027/_build_full/paper.pdf`

- [ ] **Step 1: Run source checks**

Run `git diff --check` and scan for forbidden commands, forbidden packages, `\scriptsize`, `\tiny`, negative spacing, `\resizebox`, undefined citations, and unintentional changes to formulas or numerical table entries.

- [ ] **Step 2: Build the full manuscript**

Run `bash build.sh full` from `docs/paper/AAAI2027`. Expected result: exit code 0 and `_build_full/paper.pdf` produced.

- [ ] **Step 3: Inspect the build log and rendered pages**

Require no undefined references, undefined citations, overfull boxes, fatal errors, or emergency stops. Visually inspect pages containing the main tables, algorithm, and Figures 2--4 for overlap, clipping, and illegible layout.

