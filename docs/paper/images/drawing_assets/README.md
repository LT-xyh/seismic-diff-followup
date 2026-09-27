# Drawing Assets for AAAI 2027 Figures

This folder collects reusable visual assets for redrawing the paper figures.
The assets are meant for figure composition and layout drafting. They are not
standalone quantitative results.

## Recommended Use

### Figure 1: Motivation and Design Principle

Use the modality tiles in `inputs/` to show heterogeneous evidence:

- `pstm_dense_reflector_texture`: dense seismic reflector texture; use as the
  structural seismic cue.
- `horizon_sparse_interfaces`: sparse horizon/interface cue; use as the
  structural prior.
- `rms_smooth_numerical_trend`: smooth RMS velocity cue; use as the numerical
  background cue.
- `well_log_sparse_absolute_calibration`: sparse well-log cue; use as absolute
  local calibration.

Use the decomposition tiles in `decomposition/` to show physical decomposition:

- `velocity_target_v`: target velocity field.
- `low_frequency_background_b`: low-frequency background component.
- `high_frequency_structure_s`: high-frequency residual/structure component.
- `figure1a_multimodal_asymmetry_panel`: pre-composed modality-asymmetry panel.
- `figure1b_physical_decomposition_panel`: pre-composed physical-decomposition
  panel.

Current selected illustrative sample:

- Dataset: `FlatFaultB`
- Sample index: `27006`
- Low-pass kernel: `5`
- Well columns: `[20, 43]`

The exact extraction metadata is in
`metadata/figure1_visuals_metadata.json`.

### Figure 2: Method Overview

Use `draft_figures/figure2_method_overview_v2_gpt_image2.png` as a layout
reference only. Redraw the final figure as vector graphics or a clean PDF.

The final method overview should keep the pipeline readable:

1. multimodal inputs
2. modality-specific encoders
3. physics-anchored condition contract `C_N/C_S`
4. background estimation from `C_N`
5. residual flow matching conditioned on `C_S` and background context
6. final composition

### Residual Flow Matching Submodule

Use `draft_figures/residual_flow_matching_gpt_image2.png` as a reference for
the residual-FM submodule. In the final figure, this module should emphasize:

- residual target rather than full-field target
- flow path from noise/intermediate state to residual
- conditioning by structural condition and background context
- final velocity composition from background plus residual

### Candidate Sample Selection

Use `candidate_overviews/` when choosing a more visually representative OpenFWI
sample. The overview sheets compare candidate modality panels and velocity
fields from `CurveFaultA` and `FlatFaultA`.

These overview sheets are for sample selection and visual consistency checks.
They should not be inserted directly into the paper unless redrawn or cropped.

## File Groups

- `inputs/`: individual modality tiles in PNG and PDF.
- `decomposition/`: target/background/structure tiles and pre-composed panels.
- `draft_figures/`: GPT-image draft figures for layout reference.
- `candidate_overviews/`: contact sheets for selecting representative samples.
- `metadata/`: extraction metadata for the selected illustrative sample.

## Paper-Safety Notes

- Treat GPT-image outputs as drafting aids, not final scientific evidence.
- Prefer PDF/vector redraws for the final AAAI submission.
- Use real OpenFWI tiles for modality and decomposition visuals.
- Do not imply that illustrative tiles are quantitative comparisons.
- Keep figure captions explicit: these panels illustrate motivation and method
  design, while quantitative results are reported in tables and diagnostics.
