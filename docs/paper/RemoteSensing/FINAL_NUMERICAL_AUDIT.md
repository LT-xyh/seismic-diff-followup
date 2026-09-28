# Final Numerical Audit — Remote Sensing Submission Package

Scientific-content baseline: `96fddfcab601dc61562b73998a74572dde4a160f`

| Item | Manuscript value | Authoritative source | Status |
|---|---:|---|---|
| BG-RFM MAE | 0.01503 | RS-E01-Lite 0.01502705 | ROUNDING_ONLY |
| BG-RFM RMSE | 0.02695 | RS-E01-Lite 0.02694626 | ROUNDING_ONLY |
| BG-RFM SSIM | 0.99165 | RS-E01-Lite 0.99164512 | ROUNDING_ONLY |
| BG-RFM parameters | 10.60M / 10.6M | RS-E01-Lite 10.60M | VERIFIED |
| InversionNet MAE/RMSE/SSIM | 0.01194 / 0.02730 / 0.99118 | RS-E01-Lite | ROUNDING_ONLY |
| InversionNet parameters | 24.41M | RS-E01-Lite | VERIFIED |
| VelocityGAN MAE/RMSE/SSIM | 0.02170 / 0.04341 / 0.97267 | RS-E01-Lite | ROUNDING_ONLY |
| VelocityGAN parameters | 25.59M | RS-E01-Lite | VERIFIED |
| supervised multimodal UPFWI MAE/RMSE/SSIM | 0.01218 / 0.02987 / 0.99051 | RS-E01-Lite | ROUNDING_ONLY |
| supervised multimodal UPFWI parameters | approx. 19.00M | RS-E01-Lite | VERIFIED |
| Latent U-Net MAE/RMSE/SSIM | 0.00706 / 0.01309 / 0.99403 | RS-E01-Lite | ROUNDING_ONLY |
| Latent U-Net parameters | 35.12M | RS-E01-Lite | VERIFIED |
| total multimodal records | 336,000 | retained protocol/source manuscript | VERIFIED |
| held-out records | 33,600 | RS-E01-Lite + retained protocol | VERIFIED |
| split fractions | 0.7/0.2/0.1 | release config | VERIFIED |
| split seed | 42 | release config | VERIFIED |
| well seed | 1234 | release config | VERIFIED |
| Euler inference steps | 50 | frozen BG-RFM config | VERIFIED |
| Concat-FM | 0.0225 / 0.0340 / 0.9865 | AAAI controlled ablation | VERIFIED |
| role-aware full-field FM | 0.0205 / 0.0320 / 0.9880 | AAAI Decoupled-FM row | VERIFIED (terminology rewritten only) |
| historical BG-RFM ablation | 0.0150 / 0.0270 / 0.9913 | AAAI controlled ablation | VERIFIED |
| effective-rank medians | 4.95 / 54.03 | AAAI training-set analysis | VERIFIED |
| condition margins | 0.46 / 0.35 | AAAI condition diagnostic | VERIFIED |
| q_T mean and 95% CI | 0.772; [0.771, 0.773] | AAAI transport diagnostic | VERIFIED |

Known suspicious historical `InversionNet MAE_L = 0.0180` remains excluded. The historical Uniform Fusion, w/o Prediction Consistency, and w/o Background Context rows are also excluded from reviewer-facing empirical evidence because their implementation provenance is not sufficiently recoverable.

**Audit result: PASS.** No manuscript-visible number is SOURCE_MISMATCH or UNRESOLVED.
