# AAAI27 Constructed Method Inventory

This document lists methods that are already constructed inside this repository
or already wired into the current AAAI27 evaluation tree. They should not be
placed in the external-method reproduction queue.

Use this inventory for controlled comparisons, ablations, and final-method
selection. Use `docs/paper/external_baseline_reproduction_plan.md` only for
methods whose source of truth is an external published paper/codebase.

## Policy

- **Do not reproduce these as external methods.** They are already implemented
  or adapted in this repository.
- **Do not present internal variants as published baselines.** Concat-FM,
  CNCS-FM, historical gate/no-gate, and h192/h224/h256/h288 branches are controlled
  variants or proposed-method candidates.
- **Main paper table:** use one final selected PDR-FM row, plus eligible
  external/adapted baselines with clear provenance.
- **Ablation table:** use the internal variants below to explain design choices.

## Adapted Reference Baselines Already Built Or Wired

| Method | Status | Config evidence | Evaluation evidence | Paper-table role |
|---|---|---|---|---|
| Smooth-Dix | Wired as parameter-free traditional reference. | `bg_pdr_fm/configs/experiments/aaai27/formal_smooth_dix.yaml` | Pending updated evaluation. | Main-table traditional/physics reference. |
| SV_Inv_Net | Wired as paper-faithful adapted baseline; training pending. | `bg_pdr_fm/configs/experiments/aaai27/formal_sv_inv_net.yaml` | Pending training/evaluation. | Main-table adapted external architecture. |
| VelocityGAN | Wired as adapted conditional GAN baseline; training pending. | `bg_pdr_fm/configs/experiments/aaai27/formal_velocity_gan.yaml` | Pending training/evaluation. | Main-table adapted generative baseline. |
| Conditional DDPM | Wired as diffusers conditional DDPM baseline; training pending. | `bg_pdr_fm/configs/experiments/aaai27/formal_conditional_ddpm.yaml` | Pending training/evaluation. | Main-table adapted generative baseline replacing legacy two-stage DDPM. |
| Adapted GFI | Built under AAAI27 protocol. | `bg_pdr_fm/configs/experiments/aaai27/formal_adapted_gfi_multimodal.yaml` | Existing eval artifacts may predate this registry; rerun before final paper table. | Main-table adapted published architecture reference. |
| Adapted Auto-Linear | Rewired to masked-AE Auto-Linear style; retraining required. | `bg_pdr_fm/configs/experiments/aaai27/formal_adapted_auto_linear_multimodal.yaml` | Pending retraining/evaluation under rewritten implementation. | Main-table adapted published architecture reference. |
| MM-InvNet | Legacy lightweight baseline. | `bg_pdr_fm/configs/experiments/aaai27/formal_mm_invnet.yaml` | `logs/bg_pdr_fm/aaai27/eval_interim_best/mm_invnet/summary.json` | Removed from main table; legacy reference only. |
| Two-stage DDPM | Legacy lightweight/two-stage scaffold. | `bg_pdr_fm/configs/experiments/aaai27/formal_two_stage_ddpm.yaml` | `logs/bg_pdr_fm/aaai27/eval_interim_best/two_stage_ddpm/summary.json` | Removed from main table; replaced by Conditional DDPM. |

## Internal Controlled Variants Already Built

| Method | Status | Config evidence | Evaluation evidence | Paper-table role |
|---|---|---|---|---|
| Concat-FM / Concat-FM-Strong | Built and evaluated. | `bg_pdr_fm/configs/experiments/aaai27/formal_concat_fm_strong.yaml` | `logs/bg_pdr_fm/aaai27/eval_interim_best/concat_fm_strong/summary.json` | Internal single-stage flow control; ablation/control table only. |
| CNCS-FM / CNCS-FM-Strong | Built and evaluated. | `bg_pdr_fm/configs/experiments/aaai27/formal_cncs_fm_strong.yaml` | `logs/bg_pdr_fm/aaai27/eval_interim_best/cncs_fm_strong/summary.json` | Internal decoupled-condition flow control; ablation/control table only. |
| PDR-FM legacy/gate | Built and evaluated. | `bg_pdr_fm/configs/experiments/aaai27/formal_bg_pdr_fm.yaml`; `bg_pdr_fm/configs/experiments/aaai27/formal_bg_pdr_fm_smooth_gate_e50_eval.yaml` | `logs/bg_pdr_fm/aaai27/eval_interim_best/bg_pdr_fm/summary.json`; `logs/bg_pdr_fm/aaai27/eval_interim_best_smooth_fixed/bg_pdr_fm_smooth_gate_e50/summary.json` | Historical gate ablation unless selected after relabeling. |
| PDR-FM direct residual e50 | Built and evaluated. | `bg_pdr_fm/configs/experiments/aaai27/formal_bg_pdr_fm_smooth_nogate_e50_eval.yaml` | `logs/bg_pdr_fm/aaai27/eval_interim_best_smooth_nogate/bg_pdr_fm_smooth_nogate_e50/summary.json` | Proposed-family ablation or early checkpoint reference. |

## Proposed-Method Capacity Branches

These are model-selection candidates for the final PDR-FM row. They are not
external baselines.

| Method | Status | Config evidence | Evaluation evidence | Notes |
|---|---|---|---|---|
| PDR-FM h192 e200 | Built and formally evaluated. | `bg_pdr_fm/configs/experiments/aaai27/formal_bg_pdr_fm_smooth_nogate_h192_e200_full_eval.yaml` | `logs/bg_pdr_fm/aaai27/eval_interim_best_smooth_nogate_h192/bg_pdr_fm_smooth_nogate_h192_e200_full/summary.json` | Capacity sweep candidate. |
| PDR-FM h224 e200 | Built and formally evaluated. | `bg_pdr_fm/configs/experiments/aaai27/formal_bg_pdr_fm_smooth_nogate_h224_e200_full_eval.yaml` | `logs/bg_pdr_fm/aaai27/eval_interim_best_smooth_nogate_h224/bg_pdr_fm_smooth_nogate_h224_e200_full/summary.json` | Capacity sweep candidate. |
| PDR-FM h256 e200 | Built and formally evaluated. | `bg_pdr_fm/configs/experiments/aaai27/formal_bg_pdr_fm_smooth_nogate_h256_e200_full_eval.yaml` | `logs/bg_pdr_fm/aaai27/eval_interim_best_smooth_nogate_h256/bg_pdr_fm_smooth_nogate_h256_e200_full/summary.json` | Current strong final-method candidate. |
| PDR-FM h256 lf02 e100 | Active or diagnostic branch at latest inspection. | `bg_pdr_fm/configs/experiments/aaai27/formal_bg_pdr_fm_smooth_nogate_h256_lf02_e100_full_eval.yaml` | `logs/bg_pdr_fm/residual_train_smooth_hc64_nogate_h256_lf02_e100/` | Low-frequency-loss diagnostic branch; do not interrupt active runs. |
| PDR-FM h288 e200 | Configured candidate branch. | `bg_pdr_fm/configs/experiments/aaai27/formal_bg_pdr_fm_smooth_nogate_h288_e200_full_eval.yaml` | No completed formal summary was confirmed at latest inspection. | Capacity sweep candidate; not external. |

## External Reproduction Queue Exclusion

The following names must stay out of the external reproduction queue:

- MM-InvNet
- Two-stage DDPM
- Concat-FM / Concat-FM-Strong
- CNCS-FM / CNCS-FM-Strong
- PDR-FM legacy/gate/direct-residual branches
- PDR-FM h192/h224/h256/h288 capacity branches
- Any loss-weight, gate, contrastive-checkpoint, inference-step, or residual
  backend diagnostic branch created inside this repository

The external reproduction queue currently starts with:

- GFI, ICLR 2025
- Auto-Linear, ICML 2024, only if a fair supervised OpenFWI adaptation is
  defined after audit
