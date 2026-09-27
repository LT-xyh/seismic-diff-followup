# BG-PDR-FM Internal Result Archive

This archive separates current proposed-method candidates and controlled
diagnostics from external published baselines. These rows must not be described
as external SOTA methods.

## Current Final-Method Candidate

| candidate | formal eval | samples | SSIM | MAE | MAE_L | MAE_H | status |
|---|---|---:|---:|---:|---:|---:|---|
| Smooth h256 no-gate e200 | `logs/bg_pdr_fm/aaai27/eval_interim_best_smooth_nogate_h256/bg_pdr_fm_smooth_nogate_h256_e200_full/summary.json` | 33600 | 0.874148 | 0.070727 | 0.059067 | 0.021287 | Strong completed candidate. |
| Smooth h256 no-gate lf02 e100 | `logs/bg_pdr_fm/aaai27/eval_interim_best_smooth_nogate_h256_lf02/bg_pdr_fm_smooth_nogate_h256_lf02_e100_full/summary.json` | 33600 | 0.878189 | 0.069610 | 0.058121 | 0.020932 | Best completed full-modality signal; candidate final Ours pending provenance cleanup. |

## Capacity Ablation

| variant | samples | SSIM | MAE | MAE_L | MAE_H | interpretation |
|---|---:|---:|---:|---:|---:|---|
| h192 e200 | 33600 | 0.864961 | 0.074201 | 0.062223 | 0.022000 | Capacity is insufficient for SOTA. |
| h224 e200 | 33600 | 0.868850 | 0.072754 | 0.060919 | 0.021649 | Capacity helps and crosses the Concat-FM-Strong range. |
| h256 e200 | 33600 | 0.874148 | 0.070727 | 0.059067 | 0.021287 | Capacity still helps through h256. |

## Inference-Step Ablation

10-batch h256 e200 fixed-load sweep:

| setting | samples | SSIM | MAE | MAE_L | MAE_H | decision |
|---|---:|---:|---:|---:|---:|---|
| 20 steps | 1000 | 0.883459 | 0.071071 | 0.059224 | 0.021619 | Keep default. |
| 50 steps | 1000 | 0.879204 | 0.072023 | 0.060091 | 0.021821 | Do not formal-evaluate. |
| 100 steps | 1000 | 0.878725 | 0.072884 | 0.060818 | 0.022066 | Do not formal-evaluate. |

## Contrastive Checkpoint Diagnostic

10-batch h256 downstream check:

| checkpoint | SSIM | MAE | MAE_L | MAE_H | decision |
|---|---:|---:|---:|---:|---|
| current last | 0.887972 | 0.070534 | 0.058865 | 0.021465 | Keep current encoder checkpoint. |
| pairwise epoch77 | 0.870032 | 0.073875 | 0.061979 | 0.022241 | Do not swap. |
| anchor epoch22 | 0.093444 | 0.336547 | 0.313577 | 0.049445 | Discard for downstream generation. |
| recover epoch4 | 0.025557 | 0.395426 | 0.372324 | 0.047938 | Discard for downstream generation. |

## Reporting Rules

- Use only one selected final Ours row in the main table.
- Put h192/h224/h256 and lf02 in an internal ablation/capacity table.
- Treat step sweeps and contrastive checkpoint swaps as diagnostics unless
  promoted by a formal full evaluation.
- Keep GFI and Auto-Linear separate as external-baseline work; current GFI is
  still blocked pending a full raw waveform split.
