# Final Numerical Audit — Scientific Freeze 96fddfc

Audit date: 2026-09-28  
Scientific-content baseline: `96fddfcab601dc61562b73998a74572dde4a160f`

Status vocabulary: `VERIFIED`, `ROUNDING_ONLY`, `SOURCE_MISMATCH`, `UNRESOLVED`.

## A. RS-E01-Lite authoritative common-protocol comparison

All rows use the same 33,600 held-out records, normalization, and evaluator.

| Manuscript item | Authoritative value | Display value | Status | Source |
| --- | --- | --- | --- | --- |
| BG-RFM MAE | 0.01502705 | 0.01503 | ROUNDING_ONLY | PI-approved RS-E01-Lite |
| BG-RFM RMSE | 0.02694626 | 0.02695 / 0.0269 in abstract | ROUNDING_ONLY | PI-approved RS-E01-Lite |
| BG-RFM SSIM | 0.99164512 | 0.99165 / 0.9916 in abstract | ROUNDING_ONLY | PI-approved RS-E01-Lite |
| BG-RFM Params | 10.60M | 10.60M / 10.6M prose | ROUNDING_ONLY | PI-approved RS-E01-Lite |
| InversionNet MAE | 0.01193781 | 0.01194 | ROUNDING_ONLY | PI-approved RS-E01-Lite |
| InversionNet RMSE | 0.02730467 | 0.02730 | ROUNDING_ONLY | PI-approved RS-E01-Lite |
| InversionNet SSIM | 0.99118468 | 0.99118 | ROUNDING_ONLY | PI-approved RS-E01-Lite |
| InversionNet Params | 24.41M | 24.41M | VERIFIED | PI-approved RS-E01-Lite |
| VelocityGAN MAE | 0.02170463 | 0.02170 | ROUNDING_ONLY | PI-approved RS-E01-Lite |
| VelocityGAN RMSE | 0.04340946 | 0.04341 | ROUNDING_ONLY | PI-approved RS-E01-Lite |
| VelocityGAN SSIM | 0.97267124 | 0.97267 | ROUNDING_ONLY | PI-approved RS-E01-Lite |
| VelocityGAN Params | 25.59M | 25.59M | VERIFIED | PI-approved RS-E01-Lite |
| UPFWI-adapted MAE | 0.01217775 | 0.01218 | ROUNDING_ONLY | PI-approved RS-E01-Lite |
| UPFWI-adapted RMSE | 0.02986827 | 0.02987 | ROUNDING_ONLY | PI-approved RS-E01-Lite |
| UPFWI-adapted SSIM | 0.99050921 | 0.99051 | ROUNDING_ONLY | PI-approved RS-E01-Lite |
| UPFWI-adapted Params | approx. 19.00M | approx. 19.00M | VERIFIED | PI-approved RS-E01-Lite |
| Latent U-Net MAE | 0.00705923 | 0.00706 | ROUNDING_ONLY | PI-approved RS-E01-Lite |
| Latent U-Net RMSE | 0.01308783 | 0.01309 | ROUNDING_ONLY | PI-approved RS-E01-Lite |
| Latent U-Net SSIM | 0.99403170 | 0.99403 | ROUNDING_ONLY | PI-approved RS-E01-Lite |
| Latent U-Net Params | 35.12M | 35.12M | VERIFIED | PI-approved RS-E01-Lite |
| held-out record count | 33,600 | 33,600 | VERIFIED | RS-E01-Lite + historical fixed manifest |

## B. Historical controlled design analysis

Verified against the frozen AAAI scientific source `docs/paper/AAAI2027/pd-bg-rfm.tex`.

| Variant | MAE | RMSE | SSIM | Status |
| --- | ---: | ---: | ---: | --- |
| Concat-FM | 0.0225 | 0.0340 | 0.9865 | VERIFIED |
| Role-aware full-field FM (historical Decoupled-FM) | 0.0205 | 0.0320 | 0.9880 | VERIFIED |
| Uniform fusion | 0.0163 | 0.0290 | 0.9898 | VERIFIED |
| w/o prediction consistency | 0.0190 | 0.0340 | 0.9875 | VERIFIED |
| w/o background context | 0.0230 | 0.0410 | 0.9820 | VERIFIED |
| BG-RFM (historical PD-BG-RFM row) | 0.0150 | 0.0270 | 0.9913 | VERIFIED |

## C. Protocol and architecture numbers

| Item | Value | Status | Evidence |
| --- | --- | --- | --- |
| total multimodal records | 336,000 | VERIFIED | frozen AAAI source |
| held-out records | 33,600 | VERIFIED | frozen AAAI source / RS-E01-Lite |
| split fractions | 0.7 / 0.2 / 0.1 | VERIFIED | final model config |
| split seed | 42 | VERIFIED | final model config + AAAI source |
| well seed | 1234 | VERIFIED | final model config + AAAI source |
| inference integration | 50 Euler steps | VERIFIED | final model config + AAAI source |
| BG-RFM params | 10.6M | VERIFIED | RS-E01-Lite + AAAI source |
| background hidden channels | 192 | VERIFIED | final model config |
| residual backend hidden channels | 128 | VERIFIED | final model config |
| target / depth-grid modalities | 70 x 70 | VERIFIED | source manuscript/data contract |
| RMS / migrated image | 1000 x 70 | VERIFIED | source manuscript/data contract |

## D. Historical diagnostic/supporting numbers

| Item | Value | Status |
| --- | --- | --- |
| records per subset for effective-rank analysis | 1,000 | VERIFIED |
| median effective rank, background | 4.95 | VERIFIED |
| median effective rank, structural complement | 54.03 | VERIFIED |
| background matched--deranged margin | 0.46 | VERIFIED |
| structural matched--deranged margin | 0.35 | VERIFIED |
| q_T held-out condition | all 33,600 records < 1 | VERIFIED |
| q_T mean | 0.772 | VERIFIED |
| q_T bootstrap 95% CI | [0.771, 0.773] | VERIFIED |
| qualitative velocity display | 1500--4500 m/s | VERIFIED |
| qualitative absolute-error display | 0--750 m/s | VERIFIED |

Historical missing-modality rows in the Supplement are VERIFIED against the frozen AAAI source:
- full: 0.0150 / 0.0269 / 0.9915
- w/o well log: 0.0168 / 0.0295 / 0.9899
- w/o horizon: 0.1011 / 0.1669 / 0.8202
- w/o RMS velocity: 0.3947 / 0.4705 / 0.0753
- w/o well log + RMS velocity: 0.4684 / 0.5303 / 0.0192
- PoSTM only: 0.4706 / 0.5356 / 0.0154

## Explicit exclusion

The historical matched-input value `InversionNet MAE_L = 0.0180` remains excluded. It is not present in the MDPI main manuscript or supplement.

## Audit conclusion

- SOURCE_MISMATCH: none.
- UNRESOLVED manuscript-visible scientific numbers: none.
- Remaining differences are intentional display rounding only.
