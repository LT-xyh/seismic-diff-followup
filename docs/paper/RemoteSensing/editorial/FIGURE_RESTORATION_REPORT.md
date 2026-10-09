# Figure-design provenance and editorial decisions

The figure-only source was evaluated separately in the Draft PR on `remote-sensing/aaai-figure-restoration-20261009`; this editorial proposal copies the validated rendering instructions without merging that draft branch.

| RS figure | AAAI figure/artwork | Action in proposed editorial branch | Provenance and evidence |
|---|---|---|---|
| Figure 1 | `AAAI2027/figures/method_overview_editable.pptx` → `method_overview.svg` (nine embedded image thumbnails) | **Regenerated** as `figure1_bg_rfm_restored.svg/pdf` via `render_restored_artwork.py`; uses original imagery, improved typesetting and solid/dashed inference/training paths | Original AAAI visuals; new BG-RFM labels and correct shared predicted-background reference; no physics-decoupled or theorem headline |
| Figure 2 | `figure1_motivation/motivation_statistics_combined_v2.png` | **Reused high-quality composite** with RS scientific caption | Descriptive modality alignment and effective rank only |
| Figure 3 | Original selected CurveFault-A source 26193 and FlatFault-B source 21686 standalone panels | **Unchanged scientifically** | Same selected panels and 22 source images; no re-evaluation or sample substitution |
| Figure 4 | `figure3_condition_learning/figure3_condition_learning.svg` and `figure4_residual_transport/figure4_residual_transport.svg` | **Re-exported** to vector PDF masters | Existing diagnostic values preserved; higher legibility at journal magnification |
| Appendix C figures | Additional held-out, condition-routing and qualitative assets | **Kept unchanged** | Avoid replacing empirical evidence without verified checkpoint identity and protocol |

## Reproducibility

The GitHub build renders Figure 1 from the original embedded SVG tiles, generates vector PDFs for Figure 4, builds the unified paper, and checks a minimal isolated LaTeX source ZIP. Neither training nor evaluation of a machine-learning model is needed.

## Checks for PI

Inspect small-font image labels on the printed one-column representation, verify that dashed orange lines are recognizable as training-only information, and review the changed Figure 1 manuscript caption against the graphic. The original editable PPTX remains in the AAAI historical tree; the restored SVG, PDF and renderer comprise the derivative figure source.
