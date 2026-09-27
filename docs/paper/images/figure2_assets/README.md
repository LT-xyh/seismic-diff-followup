# Figure 2 Asset Package

This directory contains editable assets for the PD-BG-RFM method overview figure.
The goal is to support a top-conference style method-as-argument figure rather
than a literal layer-by-layer architecture diagram.

## Recommended Figure 2 Story

Use the final figure to show this chain:

```text
Multimodal evidence
-> Physics-decoupled condition contract
-> Background responsibility assignment
-> Residual-space Flow Matching
-> Velocity reconstruction
```

The figure should distinguish inference computation from training-only
supervision:

- Solid arrows: inference path.
- Dashed orange arrows/boxes: training-only anchors, losses, and residual target
  construction.
- Never put `V`, `A_B`, `A_S`, or `R_{hat B}=V-hat B` on the inference path.

## Asset Map

- `concept_panels/condition_space_cluster.svg`: actual-data contract panel
  showing how RMS/well evidence supports `C_B` and PoSTM/horizon evidence
  supports `C_S`. Despite the filename, this is not a t-SNE/UMAP plot.
- `concept_panels/anchor_calibration.svg`: training-only frequency-anchor
  calibration (`A_B=W_L(V)`, `A_S=W_H(V)`).
- `concept_panels/residual_transport.svg`: residual Flow Matching principle
  (`R_t=(1-t)xi+tR_{hat B}`).
- `concept_panels/background_residual_composition.svg`: background/residual
  composition (`hat V=hat B+hat R`) using reusable tiles.
- `concept_panels/training_only_box.svg`: compact dashed side box for
  training-only information.
- `concept_panels/diagnostic_hooks.svg`: small footer for `eta_B`, `eta_S`,
  and `rho_B`.
- `transport_burden/residual_transport_3stage.svg`: compact three-stage sketch
  for the Residual FM module (`xi -> R_t -> hat R`).
- `transport_burden/full_vs_residual_burden.svg`: theoretical inset comparing
  full-field FM and residual FM target-vector norms under the stated condition.
- `data_tiles/`: copied Fig. 1 tiles for PoSTM, horizon, RMS, well, velocity,
  background, and structure.
- `feature_maps/`: real `C_B` and `C_S` feature-map visualizations exported
  from the trained encoder checkpoint by `../make_cb_cs_feature_maps.py`.
  Use `cb_cs_feature_panel.png` for a compact paper panel, or use the
  individual aggregate/channel tiles for manual Visio layout.
- `colorbars/`: copied velocity and structure colorbars.
- `figure2_draft.svg`: rough full-layout starting point; use this as a
  composition guide, not as the final figure.
- `previews/`: PNG renderings for quick inspection.

## Visio Layout Prompt

Create a wide two-column AAAI method overview figure titled:
`PD-BG-RFM: Physics-Decoupled Background-Guided Residual Flow Matching`.

Use four left-to-right stages:

1. Multimodal evidence: PoSTM, horizon, RMS velocity, well log.
2. Physics-decoupled condition contract: show `C_B` in green and `C_S` in blue,
   using the actual-data contract panel. `C_B` should be visually tied to
   RMS/well/background, while `C_S` should be tied to PoSTM/horizon/structure.
   Add dashed orange arrows from training-only anchors `A_B=W_L(V)` and
   `A_S=W_H(V)`.
   If space permits, replace or augment this stage with
   `feature_maps/cb_cs_feature_panel.png` to show the actual learned
   `C_B/C_S` feature maps.
3. Background responsibility: show `C_B -> f_B -> hat B` as a deterministic
   green branch.
4. Residual transport and reconstruction: show `C_S + psi_B(hat B)` guiding
   residual Flow Matching, then `hat V = hat B + hat R`.

Add a small diagnostic footer:
`eta_B, eta_S -> condition-role checks; rho_B -> residual leakage check`.

Keep U-Net internals, channel counts, optimizer details, and full loss
expansions out of the main figure.
