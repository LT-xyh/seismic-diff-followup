# OpenFWI Official-Capacity Adapted Baselines Design

## Purpose

Replace the current lightweight `adapted_inversion_net` and `velocity_gan`
implementations with official-capacity OpenFWI architectures, and add an
`adapted_upfwi` baseline. All three methods use the same multimodal input and
evaluation protocol as the existing AAAI27 benchmark while preserving the
source network topology and parameter count.

The architecture source is the LANL OpenFWI repository at commit
`48754806b7b4c5877259c6b958a87f4513fcdc0b`. Source-derived code must retain
the upstream copyright notice and identify this commit in module documentation.

## Scope And Naming

- Replace the existing `adapted_inversion_net` variant in place.
- Replace the existing `velocity_gan` variant in place.
- Add a new `adapted_upfwi` variant.
- Preserve existing lightweight checkpoints and evaluation artifacts as legacy
  data, but do not load them into the replacement architectures.
- Write new training and evaluation artifacts to directories ending in
  `_official_e100`.
- Paper-facing names are `Adapted InversionNet`, `Adapted VelocityGAN`, and
  `Adapted UPFWI`. They are same-input architecture adaptations, not native
  waveform-protocol reproductions.

## Shared Input Contract

The existing multimodal input builder produces a tensor with shape
`[B, 5, 1000, 70]` in this order:

1. post-stack migrated image (PoSTM), bilinearly resized;
2. RMS velocity, bilinearly resized;
3. horizon map, nearest-neighbor resized;
4. masked well log (`well_log * well_mask`), resized;
5. well mask, nearest-neighbor resized.

Only channel semantics change relative to native OpenFWI waveform input. The
official network spatial contract and output shape `[B, 1, 70, 70]` remain
unchanged. Missing-modality and engineering-stress transforms continue to run
before this input builder, so all three methods remain compatible with the
unified evaluator.

## Shared Official Network Module

Add a local source-derived module containing the exact OpenFWI definitions
needed by these baselines:

- `ConvBlock` and `ConvBlock_Tanh`;
- `DeconvBlock` and `ResizeBlock`;
- `OpenFWIInversionNet` from official `InversionNet`;
- `OpenFWIDiscriminator` from official `Discriminator`;
- `OpenFWIUPFWI` from official `FCN4_Deep_Resize_2`;
- a local WGAN-GP loss matching OpenFWI's discriminator objective.

This module is vendored into `bg_pdr_fm/external_baselines/` rather than
imported from an external clone. Runtime behavior therefore does not depend on
`/tmp`, another workspace, or network access.

## Adapted InversionNet

`AdaptedInversionNetMultimodal` becomes a thin wrapper around
`OpenFWIInversionNet`. Its forward path builds the five-channel condition
tensor and passes it directly to the official model.

Training uses supervised L1 reconstruction, matching the OpenFWI L1 baseline
used for the paper comparison. No low-frequency proxy, gate, contrastive
encoder, background decomposition, or flow-matching component is used.

Exact parameter acceptance target:

- predictor parameters: `24,409,123`.

## Adapted VelocityGAN

`VelocityGANBaseline` uses:

- `OpenFWIInversionNet` as its generator;
- `OpenFWIDiscriminator` as its discriminator;
- separate AdamW optimizers with `betas=(0.0, 0.9)`;
- discriminator updates every batch;
- generator updates every `n_critic=5` batches and on the final batch;
- WGAN-GP with `lambda_gp=10`;
- generator loss `100 * L1 + 1 * adversarial`; the velocity-MSE term is
  retained as a diagnostic with zero training weight (`lambda_g2v=0`).

The Lightning integration uses manual optimization only for `velocity_gan`.
Other benchmark variants retain automatic optimization. Validation evaluates
the generator reconstruction loss without updating the discriminator.

Exact parameter acceptance targets:

- generator: `24,409,123`;
- discriminator: `1,180,003`;
- training-time total: `25,589,126`.

Metadata records all three counts. Paper tables must state whether the Params
column uses inference generator parameters or training-time total; the default
for comparison with the OpenFWI VelocityGAN count is the training-time total.

## Adapted UPFWI

Add `AdaptedUPFWIMultimodal` as a thin wrapper around the official
`FCN4_Deep_Resize_2` network. The architecture is preserved, but its native
unsupervised wave-equation training objective is not claimed because the
current multimodal protocol does not provide the same raw waveform and forward
operator contract.

The adapted baseline uses supervised L1 reconstruction under the same split
and normalization as the other adapted methods. This isolates architecture
under a same-input benchmark and must be labeled `Adapted UPFWI`.

Exact parameter acceptance target:

- predictor parameters: `18,996,259`.

## Configuration And Artifact Isolation

Update the current InversionNet and VelocityGAN configs to the replacement
architectures and add train/eval configs for UPFWI. New formal output roots are:

- `logs/bg_pdr_fm/aaai27/formal/adapted_inversion_net_official_e100/`;
- `logs/bg_pdr_fm/aaai27/formal/velocity_gan_official_e100/`;
- `logs/bg_pdr_fm/aaai27/formal/adapted_upfwi_official_e100/`.

Corresponding evaluation roots use `eval_*_official_e100`. Training defaults to
100 epochs. Existing lightweight directories remain untouched and are excluded
from new formal configs.

## Error Handling And Compatibility

- A forward pass must reject inputs that cannot be converted to
  `[B, 5, 1000, 70]`.
- Checkpoint loading must fail on old lightweight checkpoint shape mismatches;
  no partial-load fallback is permitted for formal evaluation.
- Parameter-count tests act as architecture fingerprints and fail on any
  accidental width, normalization, bias, or layer change.
- The unified prediction contract remains `PredictionBatch`, with zero-valued
  background-only diagnostics for direct inverse baselines.
- Unified OpenFWI full, missing-modality, and engineering-stress evaluators
  remain unchanged apart from recognizing `adapted_upfwi`.

## Test And Acceptance Plan

### Architecture tests

- Assert exact parameter counts for all three official architectures.
- Assert `[B, 5, 1000, 70] -> [B, 1, 70, 70]` forward shapes and finite values.
- Verify that missing input modalities zero the expected channels.
- Verify source-derived default dimensions and layer ordering against commit
  `48754806b7b4c5877259c6b958a87f4513fcdc0b`.

### Training tests

- InversionNet and UPFWI produce finite supervised L1 losses.
- VelocityGAN produces finite L1, MSE, adversarial, discriminator, and gradient
  penalty terms.
- VelocityGAN exposes two optimizers with the expected AdamW hyperparameters.
- A bounded training test proves discriminator updates every batch and generator
  updates at `n_critic=5` cadence.

### Benchmark tests

- All variants produce evaluator-compatible `PredictionBatch` objects.
- Metadata reports exact method-specific counts rather than shared wrapper
  totals.
- A smoke checkpoint round trip has zero target-branch missing, unexpected, or
  shape-skipped keys.
- Formal evaluation requires `33,600` full-modality OpenFWI test samples with
  finite MAE, RMSE, SSIM, MAE_L, and MAE_H.

## Completion Criteria

Implementation is complete when:

1. all exact parameter assertions pass;
2. smoke forward and training tests pass in the `seg` environment;
3. old lightweight checkpoints cannot be mistaken for replacement checkpoints;
4. formal training configs point only to `_official_e100` artifact roots;
5. documentation labels all three rows as same-input adapted architectures;
6. the reported parameter table is exactly:

| Method | Inference Params | Additional Training Params | Training Total |
|---|---:|---:|---:|
| Adapted InversionNet | 24,409,123 | 0 | 24,409,123 |
| Adapted VelocityGAN | 24,409,123 | 1,180,003 | 25,589,126 |
| Adapted UPFWI | 18,996,259 | 0 | 18,996,259 |
