# AAAI27 Paper-Faithful Adapted Baselines Design

## Purpose

Build defensible same-input external/adapted baselines for the AAAI27 comparison table. The current lightweight `MM-InvNet` and repository `Two-stage DDPM` rows are not suitable for the main table because their model capacity and architecture do not match published baselines. This design replaces them with paper-faithful adapted methods while preserving the unified OpenFWI/Marmousi evaluation protocol.

## Main-Table Method Set

The main table will use:

- `Smooth-Dix`
- `SV_Inv_Net`
- `VelocityGAN`
- `Conditional DDPM`
- `Adapted GFI`
- `Adapted Auto-Linear`
- `BG-PDR-FM (Ours)`

The following rows are removed from the main table:

- `MM-InvNet`
- `Two-stage DDPM`

Internal ablation tables remain separate:

- `Concat-FM`
- `CNCS-FM`
- `Ours w/o gate`
- `BG-PDR-FM (Ours)`

Capacity/version ablations stay in the appendix:

- `h192 / h224 / h256 / h288`
- `gate / no-gate`
- `lf02`
- sampling-step sweeps

## Baseline Definitions

### Smooth-Dix

`Smooth-Dix` is a traditional/physics baseline with no training phase. It will produce a velocity estimate from available RMS/Dix-style numerical inputs under the current batch schema. It does not use PSTM, horizon, or well logs except where a documented physics rule is implemented. If a strict Dix transform is not possible from the normalized local tensors, the implementation must be labeled `RMS-smooth/Dix-style` in docs and table captions.

### SV_Inv_Net

`SV_Inv_Net` replaces `MM-InvNet`. The source is the user-confirmed paper at `https://arxiv.org/abs/2312.08194`. The implementation will be a full-capacity adapted SVInvNet-style dense-block encoder-decoder for `[B, 5, 1000, 70]` multimodal inputs and `[B, 1, 70, 70]` velocity outputs.

The adaptation changes only the input protocol:

- channel 1: migrated/PSTM image
- channel 2: RMS velocity
- channel 3: horizon
- channel 4: well log multiplied by well mask
- channel 5: well mask

Training loss uses L1 plus SSIM loss, matching the paper-level objective more closely than the current pure L1 lightweight decoder.

### Conditional DDPM

`Conditional DDPM` replaces `Two-stage DDPM`. It uses `diffusers.UNet2DConditionModel` and `diffusers.DDPMScheduler`.

Training flow:

1. Target velocity `V` is the denoising sample.
2. Random timestep `t` is sampled.
3. Noise is added with `DDPMScheduler.add_noise`.
4. The U-Net predicts noise.
5. Condition tokens are produced from the same five multimodal channels and passed as `encoder_hidden_states`.

Inference flow:

1. Start from Gaussian noise shaped `[B, 1, 70, 70]`.
2. Run DDPM denoising steps with fixed multimodal condition tokens.
3. Return the denoised velocity map.

This method is a supervised same-input adapted conditional DDPM baseline, not the old repository two-stage scaffold.

### VelocityGAN

`VelocityGAN` is an adapted full-capacity conditional GAN baseline. The generator maps the five-channel multimodal tensor to velocity. The discriminator receives condition plus velocity and predicts real/fake. Training uses adversarial loss plus L1 reconstruction. The generator is the method used for prediction and parameter reporting; discriminator parameters are reported in metadata but not in the main table `Params` column unless a supplemental note explicitly asks for total train-time parameters.

### Adapted GFI

`Adapted GFI` remains an adapted external backbone. It is not the official native waveform-to-velocity GFI reproduction. It keeps the same five-channel multimodal input protocol and unified benchmark metrics.

### Adapted Auto-Linear

The current CNN autoencoder plus single linear layer is not paper-faithful enough. It will be replaced with a closer adapted Auto-Linear design:

1. A measurement-domain masked autoencoder for five-channel condition tensors.
2. A velocity-domain masked autoencoder for one-channel velocity tensors.
3. Frozen encoders/decoders after AE pretraining.
4. A latent converter trained between measurement and velocity latents.

The converter will be low-rank/two-layer rather than a single `nn.Linear`. The method remains explicitly labeled as adapted to the multimodal PSTM/horizon/RMS/well-log input protocol and not an official native reproduction.

## Unified Benchmark Contract

All learned baselines must support:

- `training_step(batch) -> finite loss`
- `predict_batch(batch) -> PredictionBatch`
- output `velocity_hat` shape `[B, 1, 70, 70]`
- full-modality OpenFWI evaluation grouped by OpenFWI subset
- Marmousi test evaluation
- missing-mode evaluation as a separate table

The shared five-channel input builder must be reused by `SV_Inv_Net`, `VelocityGAN`, `Conditional DDPM`, `Adapted GFI`, and `Adapted Auto-Linear` unless a method has a documented reason to use a different representation.

## Parameter Reporting

The existing unified Lightning wrapper instantiates multiple modules, so wrapper-level `params` can be misleading. Table 1 must report method-specific effective parameters:

- `Smooth-Dix`: `0`
- direct learned baselines: predictor/generator parameters
- `VelocityGAN`: generator parameters in the main table, discriminator parameters in metadata
- `Conditional DDPM`: `UNet2DConditionModel` plus condition encoder parameters
- `Adapted Auto-Linear`: inference path parameters or full AE-plus-converter parameters must be clearly specified; default table value is full method parameters because the frozen AE is required for inference
- `BG-PDR-FM`: final selected method-specific model parameters

The `Trainable Params` column is removed from Table 1.

## Metrics

The main OpenFWI table is grouped by OpenFWI subset and reports:

- `MAE`
- `RMSE`
- `SSIM`
- `MAE_L`
- `MAE_H`
- `Params`

`MAE_L` and `MAE_H` are computed uniformly by the evaluation filter for every model from `velocity_hat` and target velocity. They are not model-specific outputs.

Missing-mode results are reported in a separate table.

## Documentation Rules

Every adapted baseline must be labeled honestly:

- `SV_Inv_Net`: source architecture adapted to the current multimodal input.
- `Conditional DDPM`: diffusers conditional DDPM adapted to the current multimodal input.
- `VelocityGAN`: VelocityGAN-style adapted conditional GAN baseline.
- `Adapted Auto-Linear`: Auto-Linear architecture adapted to the multimodal input protocol; not official Auto-Linear reproduction.
- `Adapted GFI`: GFI architecture adapted to multimodal input; not official waveform-to-velocity GFI.

No internal variant should be framed as an external published baseline.

## Implementation Boundaries

This work must not modify BG-PDR-FM model behavior except for shared evaluator compatibility and table generation. Existing dirty worktree changes in paper images, residual diagnostics, and internal experiment files are not part of this scope.

## Verification

Required checks before long training:

- `py_compile` for new baseline modules, `benchmark_module.py`, training entry, and evaluator.
- Unit tests for input builder, forward shapes, finite losses, and method-specific parameter counts.
- Fast-run training for `SV_Inv_Net`, `VelocityGAN`, `Conditional DDPM`, and Auto-Linear AE/inverse phases.
- Full OpenFWI evaluation only after checkpoint existence and one-batch evaluation pass.
- Marmousi evaluation after OpenFWI full-modality evaluation succeeds.
