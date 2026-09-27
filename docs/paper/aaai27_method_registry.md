# AAAI27 Method Registry

This registry fixes the provenance of methods already present in the
`bg_pdr_fm` AAAI27 experiment tree. Its main purpose is to prevent internal
controlled variants from being described as external published baselines.

Source paths inspected:

- `bg_pdr_fm/configs/experiments/aaai27/`
- `bg_pdr_fm/configs/`
- `logs/bg_pdr_fm/aaai27/eval_interim_best/`
- `logs/bg_pdr_fm/aaai27/eval_interim_best_smooth_*`

## Table Policy

- **Main SOTA table:** external published baselines and carefully labeled
  adapted published baselines only, plus the final proposed method.
- **Ablation table:** internal controlled variants, architecture variants,
  capacity variants, historical gate variants, and diagnostic branches.
- **Related-work-only methods:** methods whose physics, dimensionality, input
  contract, or output target differs too much for a fair OpenFWI full-modality
  table.

## External Published Baselines

No completed AAAI27 config in the current repository is a direct reproduction
of a recent top-conference external method such as GFI or Auto-Linear. These
methods must be added through the external reproduction plan before they can be
used as external baselines in the paper.

| Method | Provenance | Current repo config | Current eval/checkpoint evidence | Main table? | Ablation only? | Fairness/input notes |
|---|---|---|---|---|---|---|
| GFI / Latent U-Net / Invertible X-Net | External published method, ICLR 2025. Bridge skeleton added for official-repo reproduction. | `bg_pdr_fm/configs/experiments/aaai27/formal_gfi.yaml`; `bg_pdr_fm/external_baselines/gfi/` | Smoke annotation can be generated from local `data1.npy/model1.npy`; no formal trained checkpoint yet. | Yes, for GFI-native waveform reproduction. Adapted GFI is tracked separately below. | No. | Highest-priority external baseline. Official inverse input is waveform amplitude `[B,5,1000,70]`, not the PDR-FM multimodal condition stack. |
| Auto-Linear | External published method, ICML 2024. Native protocol not reproduced here. | `bg_pdr_fm/configs/experiments/aaai27/adapted_auto_linear_ae_pretrain.yaml`; `bg_pdr_fm/configs/experiments/aaai27/adapted_auto_linear_multimodal.yaml`; `bg_pdr_fm/external_baselines/auto_linear/adapted_multimodal.py` | Pending training/evaluation. | Only as Adapted Auto-Linear once trained and evaluated under the same AAAI27 split. | No. | Native assumptions differ from the current task; same-input row must be labeled as adapted. |

## Adapted Published Baselines / References

These methods can appear in comparison tables only with explicit adaptation
language. They should not be framed as exact reproductions unless their
original paper protocol and code are matched.

| Method | Provenance | Current repo config | Current eval/checkpoint evidence | Main table? | Ablation only? | Fairness/input notes |
|---|---|---|---|---|---|---|
| Smooth-Dix | Traditional/physics RMS-smooth/Dix-style reference. | `bg_pdr_fm/configs/experiments/aaai27/formal_smooth_dix.yaml` | Pending updated evaluation. | Yes, as traditional reference. | No. | No learned parameters. If strict Dix inputs are unavailable, caption must say RMS-smooth/Dix-style. |
| SV_Inv_Net | External SVInvNet architecture adapted to the current PSTM/horizon/RMS/well-log input protocol. | `bg_pdr_fm/configs/experiments/aaai27/sv_inv_net.yaml`; `bg_pdr_fm/configs/experiments/aaai27/formal_sv_inv_net.yaml`; `bg_pdr_fm/external_baselines/sv_inv_net.py` | Pending training/evaluation. | Yes, once trained and evaluated under the same AAAI27 split. | No. | Source: arXiv:2312.08194. This is an adapted same-input baseline, not an official native reproduction. |
| VelocityGAN | VelocityGAN-style conditional GAN adapted to the current multimodal input protocol. | `bg_pdr_fm/configs/experiments/aaai27/velocity_gan.yaml`; `bg_pdr_fm/configs/experiments/aaai27/formal_velocity_gan.yaml`; `bg_pdr_fm/external_baselines/velocity_gan.py` | Pending training/evaluation. | Yes, once trained and evaluated under the same AAAI27 split. | No. | Main table reports generator parameters; discriminator parameters are metadata. |
| Conditional DDPM | Conditional DDPM using diffusers `UNet2DConditionModel` and `DDPMScheduler`, adapted to current multimodal input. | `bg_pdr_fm/configs/experiments/aaai27/conditional_ddpm.yaml`; `bg_pdr_fm/configs/experiments/aaai27/formal_conditional_ddpm.yaml`; `bg_pdr_fm/external_baselines/conditional_ddpm.py` | Pending training/evaluation. | Yes, once trained and evaluated under the same AAAI27 split. | No. | Replaces the previous repository two-stage DDPM scaffold. Noise prediction is conditioned through `encoder_hidden_states`. |
| Adapted GFI / GFI-Backbone multimodal | GFI/InversionNet-style ICLR 2025 inverse backbone adapted to the current PSTM/horizon/RMS/well-log input protocol. | `bg_pdr_fm/configs/experiments/aaai27/adapted_gfi_multimodal.yaml`; `bg_pdr_fm/configs/experiments/aaai27/formal_adapted_gfi_multimodal.yaml`; `bg_pdr_fm/external_baselines/gfi/adapted_multimodal.py` | Pending training/evaluation. | Yes, once trained and evaluated under the same AAAI27 split. | No. | Must be labeled as adapted. This is not official waveform-to-velocity GFI. |
| Adapted Auto-Linear / Auto-Linear multimodal | Auto-Linear ICML 2024 two-domain masked-autoencoder latent translation adapted to the current PSTM/horizon/RMS/well-log input protocol. | `bg_pdr_fm/configs/experiments/aaai27/adapted_auto_linear_ae_pretrain.yaml`; `bg_pdr_fm/configs/experiments/aaai27/adapted_auto_linear_multimodal.yaml`; `bg_pdr_fm/external_baselines/auto_linear/adapted_multimodal.py` | Pending training/evaluation under the rewritten masked-AE implementation. | Yes, once trained and evaluated under the same AAAI27 split. | No. | Caption must state: Auto-Linear architecture adapted to the multimodal PSTM/horizon/RMS/well-log input protocol; not official Auto-Linear reproduction. |
| MM-InvNet | Legacy lightweight repository regression baseline. | `bg_pdr_fm/configs/experiments/aaai27/formal_mm_invnet.yaml` | Existing old eval artifacts only. | No. | Legacy/reference only. | Removed from main table because the current implementation is not a paper-faithful full-capacity external baseline. |
| Two-stage DDPM | Legacy repository two-stage DDPM/FM-compatible scaffold. | `bg_pdr_fm/configs/experiments/aaai27/formal_two_stage_ddpm.yaml` | Existing old eval artifacts only. | No. | Legacy/reference only. | Removed from main table and replaced by diffusers Conditional DDPM. |

## Internal Controlled Variants / Ablations

These are repository-designed controls. They are useful, but they must be
reported as controlled variants or ablations, not external baselines.

| Method | Provenance | Current repo config | Current eval/checkpoint evidence | Main table? | Ablation only? | Fairness/input notes |
|---|---|---|---|---|---|---|
| Concat-FM / Concat-FM-Strong | Internal single-stage flow control using concatenated conditions. | `bg_pdr_fm/configs/experiments/aaai27/concat_fm_strong.yaml`; `bg_pdr_fm/configs/experiments/aaai27/formal_concat_fm_strong.yaml` | `logs/bg_pdr_fm/aaai27/eval_interim_best/concat_fm_strong/summary.json` | No, unless table is explicitly labeled "controlled variants". | Yes. | Shows the behavior of naive condition concatenation under the same evaluator. Not a published external method. |
| CNCS-FM / CNCS-FM-Strong | Internal decoupled-condition flow control. | `bg_pdr_fm/configs/experiments/aaai27/cncs_fm_strong.yaml`; `bg_pdr_fm/configs/experiments/aaai27/formal_cncs_fm_strong.yaml` | `logs/bg_pdr_fm/aaai27/eval_interim_best/cncs_fm_strong/summary.json` | No, unless table is explicitly labeled "controlled variants". | Yes. | Useful to isolate condition decoupling from the full proposed physical decomposition. Not a published external method. |
| PDR-FM legacy/gate reference | Proposed-method family, legacy background and learned-gate branch. | `bg_pdr_fm/configs/experiments/aaai27/bg_pdr_fm.yaml`; `bg_pdr_fm/configs/experiments/aaai27/formal_bg_pdr_fm.yaml`; `bg_pdr_fm/configs/experiments/aaai27/formal_bg_pdr_fm_smooth_gate_e50_eval.yaml` | `logs/bg_pdr_fm/aaai27/eval_interim_best/bg_pdr_fm/summary.json`; `logs/bg_pdr_fm/aaai27/eval_interim_best_smooth_fixed/bg_pdr_fm_smooth_gate_e50/summary.json` | No, unless explicitly selected after relabeling. | Yes for historical gate variants. | Gate rows are diagnostic extensions, not the core paper narrative. |
| PDR-FM direct residual e50 | Proposed-method early direct residual branch. | `bg_pdr_fm/configs/experiments/aaai27/formal_bg_pdr_fm_smooth_nogate_e50_eval.yaml`; `bg_pdr_fm/configs/openfwi_lmdb_residual_smooth_hc64_nogate_e50.yaml` | `logs/bg_pdr_fm/aaai27/eval_interim_best_smooth_nogate/bg_pdr_fm_smooth_nogate_e50/summary.json` | No, unless it becomes the final proposed method. | Yes. | Early checkpoint reference for the physics-decomposed residual route. |
| PDR-FM h192 e200 | Proposed-method residual-capacity variant. | `bg_pdr_fm/configs/experiments/aaai27/formal_bg_pdr_fm_smooth_nogate_h192_e200_full_eval.yaml`; `bg_pdr_fm/configs/openfwi_lmdb_residual_smooth_hc64_nogate_h192_e200.yaml` | `logs/bg_pdr_fm/aaai27/eval_interim_best_smooth_nogate_h192/bg_pdr_fm_smooth_nogate_h192_e200_full/summary.json` | Only if selected as final Ours; otherwise no. | Yes. | Capacity sweep variant; do not compare as an external method. |
| PDR-FM h224 e200 | Proposed-method residual-capacity variant. | `bg_pdr_fm/configs/experiments/aaai27/formal_bg_pdr_fm_smooth_nogate_h224_e200_full_eval.yaml`; `bg_pdr_fm/configs/openfwi_lmdb_residual_smooth_hc64_nogate_h224_e200.yaml` | `logs/bg_pdr_fm/aaai27/eval_interim_best_smooth_nogate_h224/bg_pdr_fm_smooth_nogate_h224_e200_full/summary.json` | Only if selected as final Ours; otherwise no. | Yes. | Capacity sweep variant; do not compare as an external method. |
| PDR-FM h256 e200 | Proposed-method residual-capacity variant. | `bg_pdr_fm/configs/experiments/aaai27/formal_bg_pdr_fm_smooth_nogate_h256_e200_full_eval.yaml`; `bg_pdr_fm/configs/openfwi_lmdb_residual_smooth_hc64_nogate_h256_e200.yaml` | `logs/bg_pdr_fm/aaai27/eval_interim_best_smooth_nogate_h256/bg_pdr_fm_smooth_nogate_h256_e200_full/summary.json` | Candidate final Ours row if chosen. | Yes if not final. | Current completed full-modality eval candidate. Still a proposed-method capacity variant, not an external baseline. |
| PDR-FM h256 lf02 e100 | Proposed-method loss-weight/capacity diagnostic branch. | `bg_pdr_fm/configs/experiments/aaai27/formal_bg_pdr_fm_smooth_nogate_h256_lf02_e100_full_eval.yaml`; `bg_pdr_fm/configs/openfwi_lmdb_residual_smooth_hc64_nogate_h256_lf02_e100.yaml` | Active training/eval artifacts under `logs/bg_pdr_fm/residual_train_smooth_hc64_nogate_h256_lf02_e100/` as of inspection. | No until completed and selected as final Ours. | Yes. | Low-frequency-loss diagnostic branch. |
| PDR-FM h288 e200 | Proposed-method residual-capacity variant. | `bg_pdr_fm/configs/experiments/aaai27/formal_bg_pdr_fm_smooth_nogate_h288_e200_full_eval.yaml`; `bg_pdr_fm/configs/openfwi_lmdb_residual_smooth_hc64_nogate_h288_e200.yaml` | No completed `logs/bg_pdr_fm/aaai27/...h288.../summary.json` was found during inspection. | No until completed and selected as final Ours. | Yes. | Capacity branch; not part of external-baseline reproduction. |

## Current Completed Full-Modality Signals

These numbers are for experiment routing only; logs are not committed paper
artifacts.

| Method variant | Eval path | Samples | MAE | RMSE | SSIM | MAE_L | MAE_H |
|---|---|---:|---:|---:|---:|---:|---:|
| PDR-FM h192 e200 | `logs/bg_pdr_fm/aaai27/eval_interim_best_smooth_nogate_h192/bg_pdr_fm_smooth_nogate_h192_e200_full/summary.json` | 33600 | 0.074201 | 0.103007 | 0.864961 | 0.062223 | 0.022000 |
| PDR-FM h224 e200 | `logs/bg_pdr_fm/aaai27/eval_interim_best_smooth_nogate_h224/bg_pdr_fm_smooth_nogate_h224_e200_full/summary.json` | 33600 | 0.072754 | 0.101148 | 0.868850 | 0.060919 | 0.021649 |
| PDR-FM h256 e200 | `logs/bg_pdr_fm/aaai27/eval_interim_best_smooth_nogate_h256/bg_pdr_fm_smooth_nogate_h256_e200_full/summary.json` | 33600 | 0.070727 | 0.098750 | 0.874148 | 0.059067 | 0.021287 |

## Reporting Rules

1. Do not call Concat-FM, CNCS-FM, historical gate/no-gate, h192/h224/h256/h288, or
   low-frequency-loss variants "external baselines".
2. The main full-modality SOTA table should include one final Ours row, not
   every capacity branch.
3. Internal variants belong in an ablation table whose caption states that all
   rows are controlled variants under the same OpenFWI split and evaluator.
4. External baselines must carry citation, venue, year, implementation source,
   and a clear statement of input/output adaptation.
