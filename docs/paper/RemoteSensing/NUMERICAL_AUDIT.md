# Remote Sensing Numerical Consistency Ledger

Updated: 2026-09-29

This file is the final pre-submission numerical-check list. It is not manuscript prose.

## Authoritative main comparison — RS-E01-Lite

Common contract: 33,600 held-out records, same evaluator, same normalization/evaluation contract.

| Method | MAE | RMSE | SSIM | Params |
| --- | ---: | ---: | ---: | ---: |
| BG-RFM | 0.01502705 | 0.02694626 | 0.99164512 | 10.63M |
| InversionNet adaptation | 0.01193781 | 0.02730467 | 0.99118468 | 24.41M |
| VelocityGAN adaptation | 0.02170463 | 0.04340946 | 0.97267124 | 25.59M |
| UPFWI-derived supervised multimodal backbone | 0.01217775 | 0.02986827 | 0.99050921 | — |
| Latent U-Net adaptation | 0.00705923 | 0.01308783 | 0.99403170 | 35.12M |

BG-RFM exact accounting:
- trainable inference parameters: 10,630,216;
- enclosing wrapper total: 10,648,427.

Rounded main-table values are intentionally limited to five decimal places.

## Supporting Flow-Matching formulation comparison

| Formulation | MAE | RMSE | SSIM |
| --- | ---: | ---: | ---: |
| Concat-FM | 0.0225 | 0.0340 | 0.9865 |
| role-aware full-field FM | 0.0205 | 0.0320 | 0.9880 |
| BG-RFM | 0.0150 | 0.0270 | 0.9913 |

These values belong to a separate formulation-level study. The retained runs share the dataset family, split contract, normalization, target, metric definitions, optimizer family, and validation-based model selection; exact solver-step equality is not asserted across every formulation. Do not describe this table as a strictly matched, tightly controlled, component-isolated, or causal ablation.

## Historical diagnostics reused

- effective-rank median background: 4.95
- effective-rank median structural complement: 54.03
- condition matched--deranged margin, background: 0.46
- condition matched--deranged margin, structural: 0.35
- q_T: every 33,600 held-out sample below 1
- q_T mean: 0.772
- q_T bootstrap 95% CI: [0.771, 0.773]

## Observation-removal sensitivity reused in Supplement

The authoritative full-input score is reported only in the main common-evaluation table and is not duplicated in the Supplement.

- w/o well log: 0.0168 / 0.0295 / 0.9899
- w/o horizon: 0.1011 / 0.1669 / 0.8202
- w/o RMS: 0.3947 / 0.4705 / 0.0753
- w/o well + RMS: 0.4684 / 0.5303 / 0.0192
- PSTM only: 0.4706 / 0.5356 / 0.0154

## Explicit exclusion

Do not use the historical matched-input value:

InversionNet MAE_L = 0.0180

unless its original source is independently verified.

## Final consistency pass

Before submission, verify:
- all rounded main-table values against the accepted RS-E01-Lite artifact;
- every manuscript occurrence of 10.63M / 10,630,216 / 10,648,427;
- 336,000 total records and 33,600 held-out records;
- 70/20/10 split, split seed 42, and well seed 1234;
- 0--3 well columns in the evaluated observation contract;
- continuous Flow-Matching training time and 50 Euler inference steps;
- Figure/Table captions do not imply stronger evidence than the corresponding result;
- Table 3 remains supporting formulation-level evidence;
- the supplementary full-input observation-removal value is not duplicated;
- historical design-study values are not mixed with the RS-E01-Lite baseline table.
