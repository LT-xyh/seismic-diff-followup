# BG-PDR-FM h256 Next-Step Evidence

## Current Decision

`smooth_hc64 + No-Gate residual h256 e200` is the current full-modality mainline. The controlled joint route is stopped: `background_coadapt` degraded the 10-batch fixed-load result from the h256 baseline, and the contrastive checkpoint-swap diagnostic did not identify a better encoder checkpoint for downstream generation.

## Formal Full Capacity Curve

| variant | samples | generator params | SSIM | MAE | MAE_L | MAE_H | rho_B |
|---|---:|---:|---:|---:|---:|---:|---:|
| smooth_nogate_h192_e200 | 33600 | 396164 | 0.864961 | 0.074201 | 0.062223 | 0.022000 | 2.297013 |
| smooth_nogate_h224_e200 | 33600 | 526324 | 0.868850 | 0.072754 | 0.060919 | 0.021649 | 2.297013 |
| smooth_nogate_h256_e200 | 33600 | 675172 | 0.874148 | 0.070727 | 0.059067 | 0.021287 | 2.297013 |

Capacity is still helping through h256, but the largest remaining full-modality loss is concentrated in low-frequency error on velocity-only B subsets.

## Weak Subsets

| variant | subset | SSIM | MAE | MAE_L | MAE_H | rho_B |
|---|---|---:|---:|---:|---:|---:|
| h192 | CurveVelB | 0.582429 | 0.160819 | 0.132108 | 0.049624 | 2.878590 |
| h224 | CurveVelB | 0.599185 | 0.156281 | 0.128088 | 0.048524 | 2.878590 |
| h256 | CurveVelB | 0.607420 | 0.152550 | 0.124604 | 0.047438 | 2.878590 |
| h192 | FlatVelB | 0.669429 | 0.132622 | 0.116380 | 0.027651 | 3.702445 |
| h224 | FlatVelB | 0.681725 | 0.129133 | 0.113408 | 0.026631 | 3.702445 |
| h256 | FlatVelB | 0.691430 | 0.126895 | 0.111244 | 0.026269 | 3.702445 |

## Immediate Actions

1. Same-window 10-batch inference-step sweep for h256 e200 is complete: 50 and 100 ODE steps both underperform 20 steps.
2. Do not run a formal full evaluation for increased inference steps.
3. Start `h256_lf02_e100`, warm-started from h256 e200 with only `target_lf_weight` increased from 0.1 to 0.2.
4. Use `h288_e200` only if the low-frequency-weight branch fails on formal full.

## Inference-Step Sweep

| setting | samples | SSIM | MAE | MAE_L | MAE_H | rho_B |
|---|---:|---:|---:|---:|---:|---:|
| 20 steps | 1000 | 0.883459 | 0.071071 | 0.059224 | 0.021619 | 2.271614 |
| 50 steps | 1000 | 0.879204 | 0.072023 | 0.060091 | 0.021821 | 2.271614 |
| 100 steps | 1000 | 0.878725 | 0.072884 | 0.060818 | 0.022066 | 2.271614 |

The sweep uses the same 10 batches and the same h256 checkpoint. More sampling steps do not improve reconstruction quality in this setup, so training changes are the next useful lever.

## Provenance Note

`Concat-FM-Strong`, `CNCS-FM-Strong`, and `BG-PDR-FM No-Gate` are controlled internal variants under the shared AAAI27 protocol, not external published baselines. They should be reported with provenance labels in paper-facing tables.
