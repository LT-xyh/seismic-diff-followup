# Post-hoc Spatial Registration Diagnostic

Figure mode: `posthoc_spatial_registration_diagnostic`.

This artifact is explicitly a post-hoc image-registration diagnostic. PD-BG-RFM implements Residual Flow Matching, not a spatial-warping layer; the displacement field is not a model output.

## Provenance

- Source kind: `independent_rendered_raster_proxy`.
- Raw scalar model arrays found: `False`.
- Crop used: `False`.
- Composite attachment inspected: `True`.
- Composite attachment: `/public/home/xuyinghao/.codex/attachments/3553c7ea-8bd3-4cf9-a723-bc6a0aeeb055/codex-clipboard-552750d5-8764-4fa7-ab5e-b64b4210c61e.png`; it was not used for field extraction.
- Scalar decoding: signed red-minus-blue proxy from the common RdBu_r rendered PNGs.
- Generator note: The repository generator constructs noise and intermediate as a signed visual proxy from the structure tile; no checkpoint R_t tensor is stored.
- The three PNGs are independent rendered proxy assets; they are not checkpoint trajectory states.

## Registration

The TV-L1 estimator computes D for `G(x)=x+D(x)`, and the warped result is sampled as `R_warp(x)=S(R_t,G(x))` with bilinear interpolation.
- Before: MAE `0.118732`, SSIM `0.036119`.
- After: MAE `0.104103`, SSIM `0.049331`.
- Mean displacement magnitude: `1.9864` pixels.
- Maximum displacement magnitude: `10.7357` pixels.
- Jacobian determinant: min `-2.076996`, median `0.991004`, max `4.707213`.
- Folding fraction (det(J) <= 0): `0.484444%`.

## Decision

- Error improvement: `True`.
- Structural similarity improvement: `True`.
- No severe folding: `True` with threshold `0.0100`.
- Post-hoc registration supported for this pair: `True`.
- Model-internal RFW supported: `False`.
- Interpretation: Post-hoc registration is numerically supported for this image pair, but the source is not a raw model state.

When the registration gate fails, the primary figure is replaced by a Residual Flow Matching Trajectory view. Its reference linear-path panels are visualization aids, not recorded model states.

No `paper.tex` file was modified by this diagnostic.
