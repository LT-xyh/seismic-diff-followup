# Result Ledger

This file records accepted Remote Sensing evidence.

Do not copy a number into the manuscript unless it can be linked to an entry here or to an explicitly identified historical artifact.

## Status vocabulary

- PLANNED
- RUNNING
- BLOCKED
- COMPLETE-UNREVIEWED
- ACCEPTED-EVIDENCE
- REJECTED-EVIDENCE
- SUPERSEDED

## Entry template

### RS-EXXX — experiment name

- Status:
- Purpose / claim ID:
- Git commit:
- Resolved config:
- Config SHA256:
- Dataset manifest:
- Evaluated split:
- Evaluated record count:
- Training seed:
- Split seed:
- Well seed:
- Checkpoint:
- Checkpoint SHA256:
- Runtime hardware:
- Exact command:
- Output directory:
- Metrics artifact:
- Per-record artifact:
- Timing artifact:
- Deviations from protocol:
- Review decision:
- Main paper / supplement / internal:
- Notes:

## Acceptance rule

A result becomes **ACCEPTED-EVIDENCE** only after:

1. protocol matches the experiment specification;
2. record count and split are verified;
3. evaluator is the frozen Remote Sensing evaluator;
4. result artifact exists;
5. relevant checkpoint/config provenance is recorded;
6. no unresolved metric inconsistency affects the reported claim.

## Historical results

Historical AAAI values remain research context until explicitly promoted through a reviewed ledger entry. Manuscript presence alone is not sufficient.

## Accepted Remote Sensing evidence

### RS-E01-Lite — unified matched-input reconstruction

- Status: **ACCEPTED-EVIDENCE**
- Purpose / claim ID: C5, with C6 parameter accounting support
- Evaluated split: fixed global test split
- Evaluated record count: 33,600
- Split seed: 42
- Well seed: 1234
- Evaluator: common Remote Sensing evaluator; MAE/RMSE/SSIM computed per record then averaged
- Test traversal: batch size 64; shuffle disabled
- BG-RFM stochastic policy: one trajectory per record; no ensemble averaging; base seed 2027; 50 explicit Euler steps
- Main paper / supplement / internal: main paper Table 2
- Accepted values:

| Method | MAE | RMSE | SSIM | Trainable params |
| --- | ---: | ---: | ---: | ---: |
| InversionNet adaptation | 0.01194 | 0.02730 | 0.99118 | 24.41M |
| VelocityGAN adaptation | 0.02170 | 0.04341 | 0.97267 | 25.59M* |
| UPFWI-derived supervised multimodal backbone | 0.01218 | 0.02987 | 0.99051 | — |
| Latent U-Net adaptation | 0.00706 | 0.01309 | 0.99403 | 35.12M |
| BG-RFM | 0.01503 | 0.02695 | 0.99165 | 10.63M |

\*VelocityGAN accounting includes generator and discriminator; the discriminator is training-only.

Review decision: authoritative common-evaluation comparison for the Remote Sensing manuscript. Do not substitute historical frequency-metric values or an approximate UPFWI parameter count.

### RS-FM-SUPPORT — retained three-formulation comparison

- Status: **ACCEPTED-EVIDENCE / SUPPORTING ONLY**
- Purpose / claim ID: C3
- Main paper / supplement / internal: main paper Table 3
- Shared contract: same dataset family, 70/20/10 split, seed 42, target, normalization, metric definitions, AdamW family, validation-based checkpoint selection, nominal 100-epoch training
- Qualification: exact Euler-step equality is not asserted across every retained formulation
- Accepted values:

| Formulation | MAE | RMSE | SSIM |
| --- | ---: | ---: | ---: |
| Concat-FM | 0.0225 | 0.0340 | 0.9865 |
| Role-aware full-field FM | 0.0205 | 0.0320 | 0.9880 |
| BG-RFM | 0.0150 | 0.0270 | 0.9913 |

Review decision: use only as formulation-level support. Do not describe it as strictly matched, tightly controlled, component-isolated, or causal ablation evidence.
