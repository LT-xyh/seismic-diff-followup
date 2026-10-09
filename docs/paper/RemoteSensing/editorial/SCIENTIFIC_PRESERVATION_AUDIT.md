# Scientific preservation audit — AAAI-first editorial branch

## Authority and scope

- **Scientific baseline:** `b6bdebb94f4bf2676ffbcadddd60668cfa514b50`.
- **Branch source:** `remote-sensing/submission-sprint` at `cc71d310e06622b826ea5bd821e96f3a4e7e9ce6` (confirmed three-author preview).
- **Original literary/visual reference:** `docs/paper/aaaii/pd-bg-rfm.tex`, Git blob `25c48aaa6d43961fd529bd694cdc6c5b51a0a571`.
- **Experimental work:** no training, inference, checkpoint re-evaluation, or metric recomputation authorized or performed.

## Mechanism invariants

1. Role-aware condition interfaces and the active condition objective remain as in frozen RS source.
2. Background estimation uses the same 5×5 average-pool target and active loss.
3. Residual Flow Matching, stop-gradient path, conditional vector field and composed reconstruction loss are unchanged.
4. Sampling remains continuous in training with 50 explicit Euler inference steps.
5. Four-stage training sequence, optimization coefficients and checkpoint selection are unchanged.
6. Corpus and test membership remain 336,000 aligned OpenFWI-derived records and 33,600 held-out examples using the 70/20/10 contract and split seed 42.
7. The evaluated well protocol remains 0–3 columns, test well seed 1234.
8. The active common evaluator retains normalized target coordinates and unchanged per-record MAE/RMSE/global SSIM definitions.
9. BG-RFM trainable inference count remains 10,630,216; enclosing wrapper 10,648,427.
10. Latent U-Net retains best aggregate reconstruction metrics; no overall-superiority claim is added.

## Empirical invariants

- **Table 2:** all values and rows unchanged, including missing UPFWI-derived parameter count.
- **Table 3:** all three rows and values unchanged, with the same solver-step parity qualification.
- **Appendix tables and diagnostics:** effective rank, observation-removal sensitivity, and bootstrap values remain unchanged.
- **Qualitative panel:** same selected source indices and identical checkpoint-generated image files (no substitution).
- **Mathematics:** 26 numbered equation blocks, one align block, and six tabular blocks across the active method/results/appendices remain unchanged from source baseline.
- **Author/admin metadata:** the confirmed Chunlei Wu / Yinghao Xu / Jing Lu order and pending administrative approvals are not edited.

## Literature references

The revised bibliography adds eight entries retained from the original AAAI manuscript's bibliography for legitimate contextual citations. No historical benchmark numbers or obsolete AAAI mathematical claims were imported.

Static reference scan: 30 bibliography keys available; 26 keys cited in the edited manuscript source and all resolved. Cross-reference integrity is additionally checked by the LaTeX build.

## Legacy claims specifically excluded

- Physics-decoupled proof headlines and block-triangular posterior theorem claims;
- Asymmetry of conditional entropy inferred from participation-ratio effective rank;
- A universal residual transport learnability or optimization guarantee;
- Historical matched-input benchmark leadership and suspicious frequency metrics;
- Component-isolated effects inferred from the non-strictly-matched three-formulation study.

## Figure integration

The restored Figure 1/2/4 sources are introduced from the separately verified figure-restoration proposal and do not change numerical values. The proof of correctness is conceptual/caption consistency, not a new empirical measurement. Source reproducibility and PDF rendering are to be validated in the editorial branch's manuscript build.

**Merge gate:** PI approval required. No final submission tag or SUSY upload may be initiated by this editorial branch.
