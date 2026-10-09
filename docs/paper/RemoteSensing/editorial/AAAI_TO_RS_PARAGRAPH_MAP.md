# AAAI-first to Remote Sensing paragraph map

Reference: `docs/paper/aaaii/pd-bg-rfm.tex` (843-line original submission, Git blob `25c48aaa6d43961fd529bd694cdc6c5b51a0a571`).
Scientific authority: `docs/paper/RemoteSensing/` at frozen source baseline `cc71d310e06622b826ea5bd821e96f3a4e7e9ce6`, descended from scientific commit `b6bdebb94f4bf2676ffbcadddd60668cfa514b50`.

| Original AAAI narrative | Revised RS location | Editorial decision | Scientific boundary |
|---|---|---|---|
| Introduction, opening: FWI motivation, learned inversion, processed cues | Introduction paragraphs 1–2 | **Adapted/developed** | Task is explicitly processed-constraint reconstruction, not raw-shot inversion |
| Introduction, learned inversion strategies and full-field estimator | Introduction paragraphs 3–4; Related Work sections 1–2 | **Retained and expanded** | Descriptive formulation differences rather than invalid native-protocol rankings |
| Introduction, generative and multimodal conditioning perspectives | Introduction paragraphs 4–5; Related Work sections 2–3 | **Retained/rewritten** | Standard diffusion/Flow Matching; no new mathematical FM family |
| Introduction, claimed conditional-complexity asymmetry and PCA proof | Introduction paragraph 5 and Results diagnostics | **Scientific claim corrected** | Effective rank describes chosen data decomposition, not conditional entropy or easier optimization |
| Introduction, oracle background vs predicted residual reference | Introduction paragraph 6; Methods residual/consistency | **Retained and clarified** | Shared inference-available background; exact composition identity only |
| Introduction, physics-decoupled contract and contribution list | Introduction final paragraphs; Methods condition interfaces | **Reframed and expanded** | Role-aware conditioning + deterministic background + residual FM; removed Physics-Decoupled headline |
| Method, modality-specific encoders, fusion and anchors | Methods: role-aware conditioning; Appendix A | **Implementation corrected/retained** | Current evaluated channels, active coefficients and training-only targets prevail |
| Method, background and residual equations | Methods: deterministic background and residual FM | **Equations frozen; prose rebuilt** | Current 5×5 low pass, stop-gradient, conditional vector field and residual loss |
| Method, theorem and conditional-energy characterization | Methods full-field path; Results diagnostics; Appendix C | **Omitted invalid theorem claims; retained valid descriptive identity/ratio** | No universal lower-energy or learnability proof |
| Experiments, source-native and historical matched-input tables | Results common-protocol benchmark | **Replaced historical numbers with frozen RS Table 2** | Latent U-Net has best aggregate accuracy; UPFWI parameters remain unverified |
| Experiments, multi-variant ablation attribution | Results three-formulation study | **Restricted to RS Table 3** | Solver-step parity not proven; no component-isolated causality |
| Experiments, condition and target-energy analyses | Results diagnostics; Appendix C | **Retained as diagnostics** | Exact current medians/margins/CI retained with bounded interpretation |
| AAAI qualitative cases and visualization | RS figures and Appendix C | **Scientifically matched/restored only** | Figure 3 qualitative sample identities and panel values are unchanged |
| AAAI conclusion's theoretical and global-best claims | New Conclusions | **Replaced with accurate scientific synthesis** | No unsupported theoretical result or benchmark supremacy |
| Original AAAI artwork | RS Figure 1 (reconstructed), Figure 2/4 (compatible higher-quality assets) | **Design retained, legacy text removed** | Figure semantics follow RS method and observed-only inference contract |

This map concerns paragraph-level scientific and literary inheritance, not verbatim transplantation. It is intended for PI review before the editorial branch is considered for merging.
