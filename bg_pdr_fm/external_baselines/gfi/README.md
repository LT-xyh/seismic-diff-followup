# GFI External Baseline Bridge

This directory is a narrow bridge to the external GFI implementation:

- Paper: "A Unified Framework for Forward and Inverse Problems in Subsurface Imaging using Latent Space Translations", ICLR 2025.
- Official repository: `https://github.com/KGML-lab/Generalized-Forward-Inverse-Framework-for-DL4SI`
- Audited official commit: `cd0443b6f6463196e89ef2ea4ac775baf753ef39`

The official GFI inverse setting is waveform-to-velocity:

- input: waveform amplitude `[B, 5, 1000, 70]`
- output: velocity `[B, 1, 70, 70]`

This is not the same input contract as BG-PDR-FM's multimodal condition stack
(`migrated_image`, `horizon`, `rms_vel`, sparse `well_log`). Therefore, a GFI
row in the paper must be labeled either:

- **GFI-native waveform-to-velocity**, if trained/evaluated with raw waveform
  amplitudes; or
- **adapted GFI**, if a documented adapter maps the current multimodal inputs to
  a GFI-compatible tensor.

## Smoke Annotation

If the local smoke arrays exist at
`/public/home/xuyinghao/workspace/datasets/Datasets/data1.npy` and
`/public/home/xuyinghao/workspace/datasets/Datasets/model1.npy`, generate a GFI
annotation file with:

```bash
python -m bg_pdr_fm.external_baselines.gfi.prepare_gfi_annotations \
  --data /public/home/xuyinghao/workspace/datasets/Datasets/data1.npy \
  --label /public/home/xuyinghao/workspace/datasets/Datasets/model1.npy \
  --output /tmp/gfi_smoke/smoke.txt
```

The output file follows the official GFI `FWIDataset` annotation convention:

```text
/abs/path/to/data.npy<TAB>/abs/path/to/model.npy
```

## Official-Reproduction Path

Clone the external repository outside this repo:

```bash
git clone https://github.com/KGML-lab/Generalized-Forward-Inverse-Framework-for-DL4SI.git external_repos/gfi
```

If direct GitHub clone is unavailable, use the audited commit zip from
`codeload.github.com` and record the commit id:

```bash
mkdir -p /public/home/xuyinghao/workspace/external_repos /tmp/gfi_download
curl -L -o /tmp/gfi_download/gfi_cd0443b6.zip \
  https://codeload.github.com/KGML-lab/Generalized-Forward-Inverse-Framework-for-DL4SI/zip/cd0443b6f6463196e89ef2ea4ac775baf753ef39
unzip -q /tmp/gfi_download/gfi_cd0443b6.zip -d /public/home/xuyinghao/workspace/external_repos
mv /public/home/xuyinghao/workspace/external_repos/Generalized-Forward-Inverse-Framework-for-DL4SI-cd0443b6f6463196e89ef2ea4ac775baf753ef39 \
  /public/home/xuyinghao/workspace/external_repos/gfi
printf '%s\n' cd0443b6f6463196e89ef2ea4ac775baf753ef39 \
  > /public/home/xuyinghao/workspace/external_repos/gfi/.source_commit
```

Use the generated annotation file as `-t/-v` input to GFI's official
`src/train_inverse.py` for smoke checks. Do not treat this bridge as a local
reimplementation of GFI.

The 2026-06-15 smoke used an isolated venv rather than modifying `conda seg`:

```bash
source /public/home/xuyinghao/miniconda3/etc/profile.d/conda.sh
conda run -n seg python -m venv --system-site-packages \
  /public/home/xuyinghao/workspace/external_repos/gfi_venv
/public/home/xuyinghao/workspace/external_repos/gfi_venv/bin/python \
  -m pip install --no-deps iunets==0.1 memcnn==1.5.2
```

Two official-code compatibility details were needed for smoke only:

- add `PYTHONPATH=/tmp/gfi_smoke/pyshim:/public/home/xuyinghao/workspace/external_repos/gfi:/public/home/xuyinghao/workspace/external_repos/gfi/src`;
- create a root-level `rainbow256.npy` symlink in the external GFI checkout,
  because the official utilities load it from the current working directory.

The official script has no `max_batches` flag. For a true one-batch smoke, use
a two-sample trusted `.pt` dataset and run with `batch_size=2`. The smoke used:

```bash
cd /public/home/xuyinghao/workspace/external_repos/gfi
PYTHONPATH=/tmp/gfi_smoke/pyshim:/public/home/xuyinghao/workspace/external_repos/gfi:/public/home/xuyinghao/workspace/external_repos/gfi/src \
CUDA_VISIBLE_DEVICES=3 /public/home/xuyinghao/workspace/external_repos/gfi_venv/bin/python -u src/train_inverse.py \
  -d cuda \
  -ds flatvel-a \
  -ap '' \
  -t /tmp/gfi_smoke/smoke_dataset2_indexable.pt \
  -v /tmp/gfi_smoke/smoke_dataset2_indexable.pt \
  -m InversionNet \
  -b 2 \
  -j 0 \
  -eb 1 \
  -nb 1 \
  --print-freq 1 \
  --plot_interval 999 \
  --num_images 2 \
  --lambda_vgg_vel 0 \
  --lambda_recons 0 \
  --lr_scheduler None \
  -o /tmp/gfi_smoke/results \
  -l /tmp/gfi_smoke/logs \
  -n smoke \
  -s run_pt_indexable_n2
```

The completed smoke ran one train batch and one validation batch through
official `src/train_inverse.py` with `InversionNet`. The smoke log is
`/tmp/gfi_smoke/official_smoke.log`; the key finite values were:

- training loss: `0.9460`
- validation `vel_sum_abs_error`: `0.4733978509902954`
- validation `vel_sum_squared_error`: `0.33439409732818604`
- output checkpoint: `/tmp/gfi_smoke/results/smoke/run_pt_indexable_n2/checkpoint.pth`

This is not a benchmark result. It only verifies the official inverse training
path and local metric bridge.

## Metric Adapter

After the official GFI repository exports velocity predictions with shape
`[N, 1, 70, 70]`, evaluate them with the local AAAI27 metric schema:

```bash
python -m bg_pdr_fm.external_baselines.gfi.evaluate_gfi_predictions \
  --pred /path/to/gfi_predictions.npy \
  --target /path/to/velocity_targets.npy \
  --output-dir logs/bg_pdr_fm/aaai27/external_gfi_eval
```

This adapter writes `metrics.csv` and `summary.json` with MAE, RMSE, SSIM,
MAE_L, and MAE_H. It is a post-hoc metric bridge only; it does not train or
reimplement the GFI model.

Smoke bridge output on 2026-06-15:

- prediction/target arrays: `(2, 1, 70, 70)`
- metric output: `/tmp/gfi_smoke/metric_adapter/summary.json`
- finite means: `mae=0.47339780628681183`,
  `rmse=0.5717177242040634`, `ssim=0.00180062121944502`,
  `mae_l=0.45025773346424103`, `mae_h=0.04991407319903374`

## Skeleton Modules

- `adapter.py`: annotation parser, dataset adapter, and dataloader builder for
  the official GFI `[B,5,1000,70] -> [B,1,70,70]` contract.
- `model_wrapper.py`: shape-checking wrapper for any externally supplied GFI
  predictor.
- `run_official_gfi.py`: external-repo runner that prepares annotations,
  launches official `src/train_inverse.py`, exports prediction arrays, runs the
  metric adapter, and writes `run_manifest.json`.
- `audit_gfi_raw_split.py`: verifies local OpenFWI raw waveform shards against
  the official GFI split files and writes local train/val annotations.
- `adapted_multimodal.py`: disclosed same-input adaptation of a GFI/InversionNet
  style inverse CNN for PSTM/horizon/RMS/well-log inputs. It is not the official
  waveform-to-velocity GFI protocol.

## Mini Probe

The reusable runner has been checked on the local 500-sample waveform smoke
arrays:

```bash
CUDA_VISIBLE_DEVICES=7 python -m bg_pdr_fm.external_baselines.gfi.run_official_gfi \
  --data /public/home/xuyinghao/workspace/datasets/Datasets/data1.npy \
  --label /public/home/xuyinghao/workspace/datasets/Datasets/model1.npy \
  --output-root logs/bg_pdr_fm/aaai27/external_gfi_native \
  --run-name curvevel_b_probe \
  --suffix run \
  --model UNetInverseModel \
  --dataset-name curvevel-b \
  --batch-size 8 \
  --epoch-block 1 \
  --num-blocks 1 \
  --unet-repeat-blocks 1
```

The run wrote:

- `logs/bg_pdr_fm/aaai27/external_gfi_native/curvevel_b_probe/run/train.log`
- `logs/bg_pdr_fm/aaai27/external_gfi_native/curvevel_b_probe/run/checkpoint.pth`
- `logs/bg_pdr_fm/aaai27/external_gfi_native/curvevel_b_probe/run/gfi_predictions.npy`
- `logs/bg_pdr_fm/aaai27/external_gfi_native/curvevel_b_probe/run/gfi_metrics/summary.json`
- `logs/bg_pdr_fm/aaai27/external_gfi_native/curvevel_b_probe/run/run_manifest.json`

Mini-probe metrics over 500 samples were finite:

- `mae=0.6081101673841477`
- `rmse=0.6889032337963581`
- `ssim=0.16329611928388477`
- `mae_l=0.565067267358303`
- `mae_h=0.08103190580010414`

This is still not a formal benchmark result. It uses the local smoke arrays and
one epoch only; use it as a pipeline check before a full native GFI run on the
intended OpenFWI split.

## Current Formal Status

As of 2026-06-16, the local data audit only confirmed:

- `/public/home/xuyinghao/workspace/datasets/Datasets/data1.npy`:
  `(500, 5, 1000, 70)`, `float32`
- `/public/home/xuyinghao/workspace/datasets/Datasets/model1.npy`:
  `(500, 1, 70, 70)`, `float32`

These arrays are sufficient for smoke/probe execution, but they are not the
full `FlatVelA/FlatVelB/CurveVelA/CurveVelB` raw waveform split needed for a
paper-table GFI-native result. The `openfwi_lmdb` tree contains the BG-PDR-FM
derived multimodal LMDB datasets; that storage is not the official GFI raw
waveform contract unless a separate, documented adapter is built.

Therefore the current GFI status is **official native probe only**. Do not use
the 500-sample probe metrics in the main table.

When the full raw OpenFWI waveform tree is available, place it under
`/public/home/xuyinghao/workspace/datasets/openfwi_raw/` with the official
directory names such as `FlatVel_A/data/data1.npy` and run:

```bash
python -m bg_pdr_fm.external_baselines.gfi.audit_gfi_raw_split \
  --raw-root /public/home/xuyinghao/workspace/datasets/openfwi_raw \
  --output-dir logs/bg_pdr_fm/aaai27/external_gfi_native/inputs \
  --strict
```

Expected official split size is 48 train shards and 12 val shards per family,
with 500 samples per shard. If this audit fails, do not launch GFI formal
training.
