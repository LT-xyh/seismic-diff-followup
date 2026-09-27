# External Baseline Reproduction Plan

This plan is limited to external published methods. It does not add, retrain,
or relabel already constructed repository methods such as MM-InvNet,
Two-stage DDPM, Concat-FM, CNCS-FM, gate/no-gate, or h192/h224/h256/h288
capacity branches.

Already constructed repository methods are archived separately in
`docs/paper/aaai27_constructed_method_inventory.md`. Treat that file as the
inventory for internal controls, adapted references, ablations, and
proposed-method capacity branches. The reproduction queue in this file should
only contain external-paper methods whose source of truth is outside this
repository.

Primary objective: strengthen the full-modality SOTA comparison with recent
top-conference baselines while keeping missing-modality robustness as a
secondary analysis.

## Reproduction Scope

Allowed in this queue:

- external published methods with official code or a paper-faithful
  implementation route;
- explicitly disclosed adaptations of external methods when input/output
  contracts do not match OpenFWI multimodal velocity modeling;
- smoke tests and formal runs needed to verify those external methods.

Excluded from this queue:

- already built adapted references: MM-InvNet and Two-stage DDPM;
- internal controlled variants: Concat-FM and CNCS-FM;
- proposed-method branches: BG-PDR-FM gate/no-gate and h192/h224/h256/h288
  capacity sweeps;
- diagnostic branches for loss weights, contrastive checkpoints, inference
  steps, residual backend capacity, or Stage-1 variants.

## Priority Summary

| Priority | Method | Venue/year | Task match | Reproduction decision |
|---|---|---|---|---|
| P0 | GFI / Latent U-Net / Invertible X-Net | ICLR 2025 | Direct for OpenFWI waveform-to-velocity; partial for this paper's multimodal condition-to-velocity task | Reproduce first. Official code exists; use external clone + adapter route. |
| P1 | Auto-Linear | ICML 2024 | Partial; latent translation/self-supervised subsurface imaging | Audit next; likely appendix or method-reference baseline unless a fair supervised OpenFWI adaptation is built. |
| P2 | E-FWI | NeurIPS 2023 Datasets and Benchmarks | Partial; elastic FWI benchmark differs from current acoustic OpenFWI setting | Related work or appendix unless physics/input mismatch is resolved. |
| P2 | CR-FWI | ICLR 2026 | Partial; continuous-representation/physics-optimization style FWI | Related work or appendix; not a direct multimodal velocity-generation baseline. |
| Related only | GlobalTomo | NeurIPS 2025 | Related dataset; global/3D tomography differs from current 2D OpenFWI | Related work only for current paper table. |
| Related only | SIGMA | CVPR 2026 | Related seismic image benchmark, gas-chimney task | Related work only; not a velocity inversion baseline. |

## Candidate Methods

### P0: GFI / Latent U-Net / Invertible X-Net

- **Citation / source:** ICLR 2025, "A Generalized Forward-Inverse Framework
  for Deep Learning Methodologies in Subsurface Imaging".
- **Public artifacts:** OpenReview page, project page, and official GitHub
  repository are available.
- **Reference URLs:**
  - OpenReview: `https://openreview.net/forum?id=yIlyHJdYV3`
  - Project page: `https://kgml-lab.github.io/projects/GFI-framework/`
  - Official repository:
    `https://github.com/KGML-lab/Generalized-Forward-Inverse-Framework-for-DL4SI`
- **Audited repository commit:** `cd0443b6f6463196e89ef2ea4ac775baf753ef39`
  from `https://github.com/KGML-lab/Generalized-Forward-Inverse-Framework-for-DL4SI`.
- **Official code contract:** the repository provides `dataset.py`,
  `src/train_inverse.py`, `src/train_invertible_x_net.py`, `networks/unet.py`,
  and `networks/inverse_network.py`. The inverse pipeline loads OpenFWI-style
  annotation text files through `FWIDataset`, reads waveform amplitude arrays
  and velocity labels from `.npy`, and trains mappings of the form
  `amp -> vel`.
- **Official tensor contract:** inverse models consume waveform amplitudes shaped
  like `[B, 5, 1000, 70]` and predict velocity maps shaped `[B, 1, 70, 70]`.
  The label range is dataset-specific but commonly normalized from
  `1500-4500` m/s to the training range.
- **Task match:** Direct for canonical OpenFWI waveform-to-velocity inversion.
  Partial for the current `bg_pdr_fm` paper task, whose inputs are
  PSTM/migrated image, horizon, RMS velocity, and well-log-derived conditions
  rather than raw waveform amplitudes.
- **Input modality compatibility:** Not native to the current multimodal
  condition set. A fair main-table use must either:
  1. define a waveform-to-velocity external table where GFI uses its native
     waveform input, or
  2. define and disclose a multimodal adapter that converts the current
     condition stack into the GFI input contract without pretending this is the
     original GFI setting.
- **Output compatibility:** Must be adapted to output a velocity field shaped
  `[B, 1, 70, 70]` for the current OpenFWI evaluator.
- **Fair OpenFWI comparison:** Feasible, but not plug-and-play. The most
  defensible first comparison is GFI-native waveform-to-velocity on the same
  OpenFWI dataset families and metrics. A direct comparison to BG-PDR-FM's
  multimodal condition-to-velocity task must be labeled as an adaptation.
- **Expected engineering cost:** Medium for native GFI reproduction; medium-high
  for a same-input multimodal adaptation.
- **Priority:** P0.

**Reproduction route:** Minimal Adaptation, not in-repo reimplementation.

1. Clone the official GFI repository outside this repository, for example under
   an ignored external workspace:

   ```bash
   mkdir -p external_repos
   git clone https://github.com/KGML-lab/Generalized-Forward-Inverse-Framework-for-DL4SI.git \
     external_repos/gfi
   ```

2. Record license, environment, data contract, target shape, and training/eval
   entry points. The initial audit found MIT license badges in the README,
   `environment.yaml`, native OpenFWI `.npy` annotation files, and inverse
   training commands under `src/train_inverse.py`.
3. Add a thin adapter only after choosing the fairness protocol:
   - **Native GFI protocol:** export or point to OpenFWI waveform `.npy` files
     and train/evaluate `amp -> vel` using the official scripts.
   - **Same-input multimodal protocol:** implement a clearly labeled adapter
     from the current condition stack to a GFI-compatible input tensor; this is
     no longer exact GFI and must be described as an adapted GFI baseline.
4. First smoke target:

   - one train batch
   - one validation/eval batch
   - finite loss
   - input shape `[B, 5, 1000, 70]` for native GFI, or a documented adapted
     multimodal input tensor
   - output shape `[B, 1, 70, 70]`
   - MAE/RMSE/SSIM/MAE_L/MAE_H finite under current metric code

5. A bridge skeleton now exists at `bg_pdr_fm/external_baselines/gfi/`, with
   metadata recorded in `bg_pdr_fm/configs/experiments/aaai27/formal_gfi.yaml`.
   The local bridge can generate an official-style annotation file for the
   available 500-sample smoke arrays:

   ```bash
   python -m bg_pdr_fm.external_baselines.gfi.prepare_gfi_annotations \
     --data /public/home/xuyinghao/workspace/datasets/Datasets/data1.npy \
     --label /public/home/xuyinghao/workspace/datasets/Datasets/model1.npy \
     --output /tmp/gfi_smoke/smoke.txt
   ```

   Verified smoke shapes:

   - data: `(500, 5, 1000, 70)`
   - label: `(500, 1, 70, 70)`

**Bridge scope:** the local skeleton only prepares annotations and records
metadata for the official external implementation. It also provides a post-hoc
metric adapter, `bg_pdr_fm.external_baselines.gfi.evaluate_gfi_predictions`, for
official GFI prediction arrays shaped `[N,1,70,70]`. It does not train or
reimplement GFI. The official GFI repository remains the source of method truth.
The audited code expects raw OpenFWI waveform amplitudes, while the current
`bg_pdr_fm` task uses derived multimodal conditions; any same-input multimodal
GFI row must therefore be labeled as an adapted GFI baseline.

**Current smoke status:** official-code smoke passed on 2026-06-15.

- Official source was fixed to commit
  `cd0443b6f6463196e89ef2ea4ac775baf753ef39` under the external workspace
  `/public/home/xuyinghao/workspace/external_repos/gfi`. Direct `git clone`
  timed out against `github.com`, so the commit zip was downloaded from
  `codeload.github.com` and unpacked without modifying the official source.
- The official `conda env create` path was blocked by Anaconda channel Terms of
  Service acceptance. To keep `conda seg` untouched, the smoke used an isolated
  venv at `/public/home/xuyinghao/workspace/external_repos/gfi_venv` with
  `iunets==0.1` and `memcnn==1.5.2` installed on top of the existing PyTorch
  stack.
- Official `src/train_inverse.py` completed one train batch and one validation
  batch with `InversionNet`, `batch_size=2`, `epoch_block=1`, `num_block=1`,
  and no source edits. The smoke used a two-sample trusted `.pt` dataset to
  avoid the official parser's `--file-size` string-type issue under the current
  runtime.
- Smoke evidence from `/tmp/gfi_smoke/official_smoke.log`: finite training loss
  `0.9460`, finite validation metrics
  `vel_sum_abs_error=0.4733978509902954` and
  `vel_sum_squared_error=0.33439409732818604`, and checkpoint outputs
  `checkpoint.pth`, `latest_checkpoint.pth`, and `model_1.pth`.
- The post-hoc metric bridge accepted exported prediction/target arrays shaped
  `(2, 1, 70, 70)` and wrote `/tmp/gfi_smoke/metric_adapter/metrics.csv` plus
  `/tmp/gfi_smoke/metric_adapter/summary.json` with finite
  MAE/RMSE/SSIM/MAE_L/MAE_H.

Do not interpret this smoke as a benchmark result. It only proves that the
official GFI inverse training path and the local metric bridge can run end to
end on the native waveform-to-velocity contract.

**Current mini-probe status:** external runner and 500-sample pipeline probe
passed on 2026-06-16.

- Added `bg_pdr_fm.external_baselines.gfi.run_official_gfi`, a thin external
  runner that prepares GFI annotation files, launches official
  `src/train_inverse.py`, exports prediction/target arrays, runs the local
  metric adapter, and writes `run_manifest.json`.
- The runner was checked with official `UNetInverseModel`, `curvevel-b`
  dataset metadata, `batch_size=8`, `epoch_block=1`, `num_blocks=1`,
  `unet_repeat_blocks=1`, and the local 500-sample smoke arrays
  `data1.npy/model1.npy`.
- Outputs landed under
  `logs/bg_pdr_fm/aaai27/external_gfi_native/curvevel_b_probe/run/`,
  including `train.log`, `checkpoint.pth`, `gfi_predictions.npy`,
  `gfi_targets.npy`, `gfi_metrics/summary.json`, and `run_manifest.json`.
- Metric bridge output over 500 samples was finite:
  `mae=0.6081101673841477`, `rmse=0.6889032337963581`,
  `ssim=0.16329611928388477`, `mae_l=0.565067267358303`,
  `mae_h=0.08103190580010414`.

Do not use the mini-probe as a paper result. It is a pipeline validation on the
available smoke arrays, not a full GFI training/evaluation on the intended
OpenFWI benchmark split.

**Current formal-data audit:** blocked pending full raw waveform split
(checked on 2026-06-16).

- Confirmed available native waveform arrays:
  `/public/home/xuyinghao/workspace/datasets/Datasets/data1.npy` with shape
  `(500, 5, 1000, 70)` and
  `/public/home/xuyinghao/workspace/datasets/Datasets/model1.npy` with shape
  `(500, 1, 70, 70)`.
- No complete `FlatVelA/FlatVelB/CurveVelA/CurveVelB` raw waveform `.npy`
  split was confirmed in the current workspace during this audit.
- The existing `openfwi_lmdb` tree is the BG-PDR-FM multimodal LMDB storage,
  not a verified official GFI raw waveform split.

Until a full raw waveform split is available, GFI-native must remain labeled as
`official native probe only` and must not enter the paper result table. The
same-input multimodal route is now tracked separately as `Adapted GFI`, which
uses a GFI/InversionNet-style inverse backbone on the current PSTM/horizon/RMS/
well-log input protocol. This adapted row may enter the main same-input table
after training/evaluation, but it must not be described as official waveform-to-
velocity GFI.

The reusable full-split audit entrypoint is:

```bash
python -m bg_pdr_fm.external_baselines.gfi.audit_gfi_raw_split \
  --raw-root /public/home/xuyinghao/workspace/datasets/openfwi_raw \
  --output-dir logs/bg_pdr_fm/aaai27/external_gfi_native/inputs \
  --strict
```

It remaps the official GFI `train_test_splits/*.txt` files to the local raw
root, verifies `[N,5,1000,70]` waveform and `[N,1,70,70]` label shards, and
writes local train/val annotations. The current workspace audit reports
`blocked_missing_or_invalid_raw_shards`, which is expected until the raw
OpenFWI shards are downloaded or mounted.

### P1: Auto-Linear / Adapted Auto-Linear

- **Citation / source:** ICML 2024, "The Auto-Linear Phenomenon in Subsurface
  Imaging", PMLR.
- **Reference URLs:**
  - PMLR: `https://proceedings.mlr.press/v235/feng24a.html`
  - arXiv: `https://arxiv.org/abs/2305.13314`
- **Task match:** Partial. It is highly relevant to latent-space subsurface
  imaging, but its native self-supervised/latent-translation assumptions differ
  from the current supervised multimodal velocity modeling setup.
- **Adapted implementation:** `bg_pdr_fm/external_baselines/auto_linear/adapted_multimodal.py`
  defines a same-input adaptation with two autoencoders and a frozen latent
  linear map. The measurement domain is the current five-channel multimodal
  stack; the velocity domain is `[B,1,70,70]`.
- **Training configs:** `adapted_auto_linear_ae_pretrain.yaml` trains the two
  autoencoders; `adapted_auto_linear_multimodal.yaml` loads that checkpoint,
  freezes the autoencoders, and trains only the latent linear map.
- **Fair OpenFWI comparison:** Allowed only with explicit adapted-baseline
  labeling. Table captions must state: "Auto-Linear architecture adapted to the
  multimodal PSTM/horizon/RMS/well-log input protocol; not official Auto-Linear
  reproduction."
- **Expected engineering cost:** Medium to high, depending on code availability
  and input/output mismatch.
- **Priority:** P1.

Recommended use: same-input adapted external baseline after full train/eval
passes; appendix/method-reference only if it underperforms MM-InvNet.

### P2: E-FWI

- **Citation / source:** NeurIPS 2023 Datasets and Benchmarks, elastic FWI
  benchmark.
- **Reference URLs:**
  - OpenReview: `https://openreview.net/forum?id=3BQaMV9jxK`
  - NeurIPS proceedings:
    `https://proceedings.neurips.cc/paper_files/paper/2023/hash/4aa8d18aad014fb3d0076e0afd2e3b2e-Abstract-Datasets_and_Benchmarks.html`
- **Task match:** Partial. It is a strong recent benchmark reference, but its
  elastic-wave setting does not match the current acoustic OpenFWI-style
  velocity target.
- **Input modality compatibility:** Not aligned with the current multimodal
  condition set without substantial adaptation.
- **Output compatibility:** Likely different physical targets or wavefield
  assumptions.
- **Fair OpenFWI comparison:** Not fair as a main-table method unless the
  physics mismatch is resolved and clearly documented.
- **Expected engineering cost:** High for a fair direct comparison.
- **Priority:** P2 / related work.

### P2: CR-FWI

- **Citation / source:** ICLR 2026, continuous-representation FWI direction.
- **Reference URL:** `https://openreview.net/forum?id=blqYa21WOv`
- **Task match:** Partial. It is a recent top-conference FWI method but closer
  to physics-based optimization/continuous representation than to multimodal
  generative initial velocity modeling.
- **Input modality compatibility:** Not native to PSTM/horizon/RMS/well-log
  multimodal conditioning.
- **Output compatibility:** Needs task-specific audit.
- **Fair OpenFWI comparison:** Appendix only unless a matched OpenFWI protocol
  can be defined.
- **Expected engineering cost:** High.
- **Priority:** P2 / related work.

### Related Only: GlobalTomo

- **Citation / source:** NeurIPS 2025 Datasets and Benchmarks candidate for
  global tomography / seismic wavefield modeling.
- **Reference URL:** `https://openreview.net/forum?id=XUpwWaH7HD`
- **Task match:** Related only. Global/3D tomography data and objectives differ
  from current 2D OpenFWI full-modality velocity modeling.
- **Fair OpenFWI comparison:** No for the current table.
- **Priority:** Related work only.

### Related Only: SIGMA

- **Citation / source:** CVPR 2026, physics-based benchmark for gas chimney
  understanding in seismic images.
- **Reference URL:**
  `https://openaccess.thecvf.com/content/CVPR2026/html/Truong_SIGMA_A_Physics-Based_Benchmark_for_Gas_Chimney_Understanding_in_Seismic_CVPR_2026_paper.html`
- **Task match:** Related only. It is a seismic image understanding benchmark,
  not a direct velocity inversion/generation method.
- **Fair OpenFWI comparison:** No for the current table.
- **Priority:** Related work only.

## Proposed Paper Tables

### Main Full-Modality SOTA Table

Eligible rows:

- Final proposed BG-PDR-FM variant only.
- Adapted GFI after a verified same-split adaptation.
- MM-InvNet only as an adapted InversionNet-style baseline.
- Two-stage DDPM only as an adapted generative baseline.
- Auto-Linear only if a fair supervised OpenFWI adaptation is implemented.

Not eligible as external rows:

- Concat-FM / Concat-FM-Strong
- CNCS-FM / CNCS-FM-Strong
- Gate/No-Gate branch rows
- h192/h224/h256/h288 capacity rows

### Architecture Ablation Table

Eligible rows:

- Background-only smooth bottleneck
- Gate vs No-Gate
- h192/h224/h256/h288 capacity variants
- Concat-FM and CNCS-FM as controlled variants
- Contrastive checkpoint or structural-condition ablations

### Missing-Modality Robustness Table

Eligible rows:

- Final proposed BG-PDR-FM variant
- GFI only after a documented missing-modality adaptation
- Adapted MM-InvNet and Two-stage DDPM
- Internal variants only if the caption clearly states this is a controlled
  robustness analysis, not an external SOTA table

## GFI Formal Reproduction Checklist

Before starting formal GFI training:

1. Confirm current BG-PDR-FM training/evaluation GPU use with `hy-smi`; do not
   interrupt active runs.
2. Clone official GFI code into an ignored external directory.
3. Record commit hash, license, dependency versions, and original command line.
4. Map data contract:
   - official input tensor names and shapes
   - official target tensor shape
   - normalization range
   - train/val/test split assumptions
5. Build adapter:
   - for native GFI, map current OpenFWI storage to the official annotation-file
     convention expected by `FWIDataset`
   - for same-input comparison, document any conversion from PSTM/horizon/RMS
     velocity/well-log conditions to the tensor consumed by GFI
   - output converted to `[B, 1, 70, 70]`
6. Run smoke:
   - one train batch
   - one eval batch
   - finite loss and metrics
7. Add formal config only after smoke passes.
8. Run full-modality formal eval first. Missing-modality eval is secondary and
   should only run after full-modality output is credible.

## Suggested Next Commands

These commands are intentionally not executed by this documentation pass.

```bash
hy-smi
mkdir -p external_repos
git clone https://github.com/KGML-lab/Generalized-Forward-Inverse-Framework-for-DL4SI.git external_repos/gfi
cd external_repos/gfi
git rev-parse HEAD
```

The local bridge already verifies the smoke annotation shape. The next step is
to run the official GFI `src/train_inverse.py` against `/tmp/gfi_smoke/smoke.txt`
for a one-batch smoke on a free high-index GPU. Do not launch a formal long run
until the official script proves loss and prediction compatibility.

## References Used For Planning

- GFI / ICLR 2025: OpenReview, project page, official GitHub repository.
- Auto-Linear / ICML 2024: PMLR paper page.
- E-FWI / NeurIPS 2023 Datasets and Benchmarks: benchmark/reference method.
- CR-FWI / ICLR 2026: recent continuous-representation FWI direction.
- GlobalTomo / NeurIPS 2025 and SIGMA / CVPR 2026: related seismic benchmark
  directions, not direct OpenFWI velocity-model baselines for the current table.
