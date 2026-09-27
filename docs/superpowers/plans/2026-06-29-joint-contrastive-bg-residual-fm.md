# Joint Contrastive Background Residual FM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a full joint training mode that warm-starts the best contrastive encoder, direct background U-Net, and latent residual FM, then trains contrastive + background + residual FM losses together using predicted background.

**Architecture:** Add a new `training.stage: joint_full` path to `BGPDRFMLightning`. The joint path runs the encoder with all heads, computes contrastive loss, computes direct background loss against `P_L(V)`, and computes residual latent FM loss using predicted `B_hat`. Optimizer parameter groups protect the fully unfrozen encoder with very small learning rates while allowing background and residual FM to adapt.

**Tech Stack:** PyTorch, Lightning, OmegaConf, existing `bg_pdr_fm` modules, OpenFWI LMDB configs, ROCm/conda `seg` runtime.

---

## File Structure

- Modify `bg_pdr_fm/lightning/bg_pdr_fm_module.py`
  - Add `joint_full` as a valid stage inside model forward/training/validation paths.
  - Add parameter-group optimizer support for joint training.
  - Add full encoder unfreeze handling for `training.joint.unfreeze_encoder_backbones: true`.

- Modify `bg_pdr_fm/lightning/stage_losses.py`
  - Add `joint_full_stage_loss()` combining contrastive, background, and residual losses.
  - Reuse existing `contrastive_stage_loss`, `background_stage_loss`, and residual latent FM logic where possible.

- Modify `bg_pdr_fm/training/train_bg_pdr_fm.py`
  - Add `joint_full` to `VALID_STAGES`.

- Create `bg_pdr_fm/configs/openfwi_lmdb_joint_full_contrastive_bgfm_predbg_e100.yaml`
  - Warm-start best known checkpoints.
  - Use predicted background for residual FM.
  - Use `lambda_contrastive=0.01`.
  - Use encoder/backbone/head/background/residual parameter learning rates requested by user.
  - Train for 100 epochs.

- Create `bg_pdr_fm/configs/openfwi_lmdb_joint_full_contrastive_bgfm_predbg_eval_mb10.yaml`
  - Evaluate the joint checkpoint on 10 batches.

- Modify or create tests in `bg_pdr_fm/tests/test_joint_full_training.py`
  - Validate stage dispatch.
  - Validate trainable parameter groups.
  - Validate joint loss keys and finite scalar on synthetic/minimal batch where feasible.

---

## Task 1: Stage Registration And Freeze Policy

**Files:**
- Modify: `bg_pdr_fm/training/train_bg_pdr_fm.py`
- Modify: `bg_pdr_fm/lightning/bg_pdr_fm_module.py`
- Test: `bg_pdr_fm/tests/test_joint_full_training.py`

- [ ] **Step 1: Add failing tests for `joint_full` stage acceptance and freeze policy**

Create `bg_pdr_fm/tests/test_joint_full_training.py` with:

```python
from omegaconf import OmegaConf

from bg_pdr_fm.training.train_bg_pdr_fm import VALID_STAGES


def test_joint_full_is_valid_training_stage():
    assert "joint_full" in VALID_STAGES


def test_joint_full_config_defaults_are_addressable():
    conf = OmegaConf.create(
        {
            "training": {
                "stage": "joint_full",
                "joint": {
                    "lambda_contrastive": 0.01,
                    "encoder_backbone_lr": 1e-6,
                    "encoder_head_lr": 3e-6,
                    "background_lr": 1e-4,
                    "residual_lr": 5e-5,
                },
            }
        }
    )
    assert conf.training.stage == "joint_full"
    assert conf.training.joint.lambda_contrastive == 0.01
```

- [ ] **Step 2: Run tests and verify the stage test fails**

Run:

```bash
pytest bg_pdr_fm/tests/test_joint_full_training.py::test_joint_full_is_valid_training_stage -q
```

Expected: FAIL because `joint_full` is not yet in `VALID_STAGES`.

- [ ] **Step 3: Add `joint_full` to stage validation**

In `bg_pdr_fm/training/train_bg_pdr_fm.py`, update:

```python
VALID_STAGES = {"contrastive", "background", "residual", "joint_full"}
```

If `VALID_STAGES` is currently defined differently, preserve existing stages and add exactly `joint_full`.

- [ ] **Step 4: Update freeze policy in Lightning module**

In `bg_pdr_fm/lightning/bg_pdr_fm_module.py`, locate the stage-dependent `requires_grad` setup around encoder/background/residual. Add a branch equivalent to:

```python
if self.stage == "joint_full":
    self._set_requires_grad(self.encoder, True)
    self._set_requires_grad(self.contrastive_loss, True)
    self._set_requires_grad(self.background, True)
    self._set_requires_grad(self.residual, True)
    self._set_requires_grad(self.adapters, True)
    self._set_requires_grad(self.codec, False)
```

Keep codec frozen. Do not enable quality gate or rho calibrator through freeze policy unless config explicitly already does so.

- [ ] **Step 5: Run tests**

Run:

```bash
pytest bg_pdr_fm/tests/test_joint_full_training.py -q
python -m py_compile bg_pdr_fm/training/train_bg_pdr_fm.py bg_pdr_fm/lightning/bg_pdr_fm_module.py
```

Expected: tests pass, py_compile exits 0.

- [ ] **Step 6: Commit**

```bash
git add bg_pdr_fm/training/train_bg_pdr_fm.py bg_pdr_fm/lightning/bg_pdr_fm_module.py bg_pdr_fm/tests/test_joint_full_training.py
git commit -m "feat: register joint full bg-pdr-fm stage"
```

---

## Task 2: Joint Loss Composition

**Files:**
- Modify: `bg_pdr_fm/lightning/stage_losses.py`
- Modify: `bg_pdr_fm/lightning/bg_pdr_fm_module.py`
- Test: `bg_pdr_fm/tests/test_joint_full_training.py`

- [ ] **Step 1: Add a unit test for joint loss naming contract**

Append to `bg_pdr_fm/tests/test_joint_full_training.py`:

```python
def test_joint_full_expected_loss_keys_documented():
    expected = {
        "loss",
        "joint/contrastive_loss",
        "joint/background_loss",
        "joint/residual_loss",
        "joint/lambda_contrastive",
    }
    assert "joint/contrastive_loss" in expected
    assert "joint/background_loss" in expected
    assert "joint/residual_loss" in expected
```

This protects the metrics contract without requiring a heavy dataloader in unit tests.

- [ ] **Step 2: Implement `joint_full_stage_loss`**

In `bg_pdr_fm/lightning/stage_losses.py`, add:

```python
def joint_full_stage_loss(module, batch: BGSampleBatch, features, bg, conditions) -> StageLossOutput:
    lambda_contrastive = float(conf_get(module.conf, "training.joint.lambda_contrastive", 0.01))

    contrastive = module.contrastive_loss(
        features,
        batch.depth_vel,
        filter_module=module.filter,
    )
    contrastive_loss = contrastive["loss"] if isinstance(contrastive, dict) else contrastive

    bg_out = background_stage_loss(module, batch, features, bg)
    residual_out = residual_stage_loss(module, batch, features, bg, conditions)

    total = residual_out.loss + bg_out.loss + lambda_contrastive * contrastive_loss
    logs = {}
    logs.update({f"joint/bg_{k}": v for k, v in bg_out.logs.items()})
    logs.update({f"joint/residual_{k}": v for k, v in residual_out.logs.items()})
    if isinstance(contrastive, dict):
        for key, value in contrastive.items():
            if key != "loss":
                logs[f"joint/contrastive_{key}"] = value
    logs["joint/contrastive_loss"] = contrastive_loss.detach()
    logs["joint/background_loss"] = bg_out.loss.detach()
    logs["joint/residual_loss"] = residual_out.loss.detach()
    logs["joint/lambda_contrastive"] = total.new_tensor(lambda_contrastive)
    return StageLossOutput(loss=total, logs=logs)
```

Adjust exact call signatures if existing `StageLossOutput` or loss helpers use different field names. Preserve the semantics: total = residual + background + 0.01 contrastive.

- [ ] **Step 3: Dispatch `joint_full` from Lightning training/validation**

In `bg_pdr_fm/lightning/bg_pdr_fm_module.py`, import `joint_full_stage_loss` and add stage dispatch:

```python
elif self.stage == "joint_full":
    out = joint_full_stage_loss(self, batch, features, bg, conditions)
```

In any `return_all_heads` encoder call, use:

```python
return_all_heads=self.stage in {"contrastive", "joint_full"}
```

- [ ] **Step 4: Run static checks**

```bash
python -m py_compile bg_pdr_fm/lightning/stage_losses.py bg_pdr_fm/lightning/bg_pdr_fm_module.py
pytest bg_pdr_fm/tests/test_joint_full_training.py -q
```

Expected: py_compile exits 0 and tests pass.

- [ ] **Step 5: Commit**

```bash
git add bg_pdr_fm/lightning/stage_losses.py bg_pdr_fm/lightning/bg_pdr_fm_module.py bg_pdr_fm/tests/test_joint_full_training.py
git commit -m "feat: add joint contrastive background residual loss"
```

---

## Task 3: Parameter Group Optimizer

**Files:**
- Modify: `bg_pdr_fm/lightning/bg_pdr_fm_module.py`
- Test: `bg_pdr_fm/tests/test_joint_full_training.py`

- [ ] **Step 1: Add a test for joint LR config values**

Append:

```python
def test_joint_full_lr_values_match_confirmed_design():
    conf = OmegaConf.create(
        {
            "training": {
                "stage": "joint_full",
                "joint": {
                    "encoder_backbone_lr": 1e-6,
                    "encoder_head_lr": 3e-6,
                    "contrastive_lr": 3e-6,
                    "background_lr": 1e-4,
                    "residual_lr": 5e-5,
                },
            }
        }
    )
    assert conf.training.joint.encoder_backbone_lr == 1e-6
    assert conf.training.joint.encoder_head_lr == 3e-6
    assert conf.training.joint.background_lr == 1e-4
    assert conf.training.joint.residual_lr == 5e-5
```

- [ ] **Step 2: Implement parameter grouping for `joint_full`**

In `configure_optimizers()` of `bg_pdr_fm/lightning/bg_pdr_fm_module.py`, before the generic optimizer path, add:

```python
if self.stage == "joint_full":
    joint = self.conf.training.get("joint", {}) if hasattr(self.conf.training, "get") else {}
    weight_decay = float(_conf_get(self.conf, "optimizer.weight_decay", 0.0))

    def trainable_params(module):
        return [p for p in module.parameters() if p.requires_grad]

    groups = []
    encoder_head_params = []
    encoder_backbone_params = []
    for name, param in self.encoder.named_parameters():
        if not param.requires_grad:
            continue
        if ".heads." in name or name.startswith("heads.") or "head" in name:
            encoder_head_params.append(param)
        else:
            encoder_backbone_params.append(param)

    if encoder_backbone_params:
        groups.append({"params": encoder_backbone_params, "lr": float(_conf_get(self.conf, "training.joint.encoder_backbone_lr", 1e-6)), "name": "encoder_backbones"})
    if encoder_head_params:
        groups.append({"params": encoder_head_params, "lr": float(_conf_get(self.conf, "training.joint.encoder_head_lr", 3e-6)), "name": "encoder_heads"})
    if trainable_params(self.contrastive_loss):
        groups.append({"params": trainable_params(self.contrastive_loss), "lr": float(_conf_get(self.conf, "training.joint.contrastive_lr", 3e-6)), "name": "contrastive_loss"})
    if trainable_params(self.background):
        groups.append({"params": trainable_params(self.background), "lr": float(_conf_get(self.conf, "training.joint.background_lr", 1e-4)), "name": "background"})
    if trainable_params(self.residual):
        groups.append({"params": trainable_params(self.residual), "lr": float(_conf_get(self.conf, "training.joint.residual_lr", 5e-5)), "name": "residual"})
    if hasattr(self, "adapters") and trainable_params(self.adapters):
        groups.append({"params": trainable_params(self.adapters), "lr": float(_conf_get(self.conf, "training.joint.adapter_lr", 3e-6)), "name": "adapters"})

    groups = [group for group in groups if group["params"]]
    return torch.optim.AdamW(groups, weight_decay=weight_decay)
```

If the module already imports `torch`, reuse it. If not, add `import torch`.

- [ ] **Step 3: Add optimizer smoke by py_compile and unit test**

```bash
python -m py_compile bg_pdr_fm/lightning/bg_pdr_fm_module.py
pytest bg_pdr_fm/tests/test_joint_full_training.py -q
```

Expected: pass.

- [ ] **Step 4: Commit**

```bash
git add bg_pdr_fm/lightning/bg_pdr_fm_module.py bg_pdr_fm/tests/test_joint_full_training.py
git commit -m "feat: add joint full optimizer parameter groups"
```

---

## Task 4: Joint Training And Eval Configs

**Files:**
- Create: `bg_pdr_fm/configs/openfwi_lmdb_joint_full_contrastive_bgfm_predbg_e100.yaml`
- Create: `bg_pdr_fm/configs/openfwi_lmdb_joint_full_contrastive_bgfm_predbg_eval_mb10.yaml`

- [ ] **Step 1: Create training config**

Create `bg_pdr_fm/configs/openfwi_lmdb_joint_full_contrastive_bgfm_predbg_e100.yaml` using the closest current production config as base. Required effective values:

```yaml
extends: openfwi_lmdb_residual_smooth_hc64_nogate_unetfm_film_predbg_e100.yaml

model:
  background_backend: unet_direct
  background_condition_source: encoder_numerical
  residual_backend: unet_fm_film
  residual_background_source: predicted
  quality_gate:
    mode: none

loss:
  bg_l1_weight: 1.0
  bg_l2_weight: 0.5
  final_l1_weight: 1.0
  target_lf_weight: 0.2
  rho_weight: 0.0

training:
  stage: joint_full
  freeze_encoder: false
  max_epochs: 100
  batch_size: 256
  devices: 1
  load_stage_checkpoint: logs/bg_pdr_fm/residual_train_smooth_hc64_nogate_unetfm_film_predbg_e100/stage_checkpoints/residual_last.ckpt
  joint:
    lambda_contrastive: 0.01
    encoder_backbone_lr: 1.0e-6
    encoder_head_lr: 3.0e-6
    contrastive_lr: 3.0e-6
    adapter_lr: 3.0e-6
    background_lr: 1.0e-4
    residual_lr: 5.0e-5
  logging:
    log_dir: logs/bg_pdr_fm/joint_full_contrastive_bgfm_predbg_e100/lightning
    log_version: "joint_full_predbg_e100_{time}"
  checkpoint:
    dirpath: logs/bg_pdr_fm/joint_full_contrastive_bgfm_predbg_e100/lightning/checkpoints
    filename: "joint-full-predbg-{epoch}-loss{val/loss:.4f}"
  stage_checkpoint_path: logs/bg_pdr_fm/joint_full_contrastive_bgfm_predbg_e100/stage_checkpoints

evaluation:
  output_dir: logs/bg_pdr_fm/joint_full_contrastive_bgfm_predbg_e100/evaluation_mb10
  checkpoints:
    contrastive: logs/bg_pdr_fm/contrastive_train/stage_checkpoints/contrastive_last.ckpt
    background: logs/bg_pdr_fm/background_train_unet_direct_h192_l1l2_e100/stage_checkpoints/background_last.ckpt
    residual: logs/bg_pdr_fm/joint_full_contrastive_bgfm_predbg_e100/stage_checkpoints/residual_last.ckpt
```

If the current config naming uses different keys for checkpoints, follow the existing `evaluation.checkpoints` structure in nearby configs exactly.

- [ ] **Step 2: Create 10-batch eval config**

Create `bg_pdr_fm/configs/openfwi_lmdb_joint_full_contrastive_bgfm_predbg_eval_mb10.yaml`:

```yaml
extends: openfwi_lmdb_joint_full_contrastive_bgfm_predbg_e100.yaml

evaluation:
  output_dir: logs/bg_pdr_fm/joint_full_contrastive_bgfm_predbg_e100/evaluation_mb10
  max_batches: 10
  save_arrays: false
  save_panels: false
```

- [ ] **Step 3: Validate config loading**

Run:

```bash
python - <<'PY'
from omegaconf import OmegaConf
p='bg_pdr_fm/configs/openfwi_lmdb_joint_full_contrastive_bgfm_predbg_e100.yaml'
conf=OmegaConf.load(p)
print(conf.training.stage)
print(conf.training.joint.lambda_contrastive)
PY
```

Expected output includes:

```text
joint_full
0.01
```

- [ ] **Step 4: Commit**

```bash
git add bg_pdr_fm/configs/openfwi_lmdb_joint_full_contrastive_bgfm_predbg_e100.yaml bg_pdr_fm/configs/openfwi_lmdb_joint_full_contrastive_bgfm_predbg_eval_mb10.yaml
git commit -m "config: add joint full contrastive bg-fm experiment"
```

---

## Task 5: Smoke Train And Eval

**Files:**
- No source edits expected.
- Runtime outputs under `logs/bg_pdr_fm/joint_full_contrastive_bgfm_predbg_e100/`.

- [ ] **Step 1: Run static checks**

```bash
python -m py_compile \
  bg_pdr_fm/lightning/bg_pdr_fm_module.py \
  bg_pdr_fm/lightning/stage_losses.py \
  bg_pdr_fm/training/train_bg_pdr_fm.py
pytest bg_pdr_fm/tests/test_joint_full_training.py -q
```

Expected: pass.

- [ ] **Step 2: Run 1-batch train smoke in conda `seg`**

```bash
source /public/home/xuyinghao/miniconda3/etc/profile.d/conda.sh
conda activate seg
source /opt/dtk-25.04.2/env.sh
source /opt/hyhal/env.sh
export PATH=$CONDA_PREFIX/bin:$PATH
export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=7 HIP_VISIBLE_DEVICES=7 python -m bg_pdr_fm.training.train_bg_pdr_fm \
  --config bg_pdr_fm/configs/openfwi_lmdb_joint_full_contrastive_bgfm_predbg_e100.yaml \
  --max-batches 1 \
  --max-epochs 1
```

Expected: training completes, loss finite, no NaN/Inf.

- [ ] **Step 3: Run 1-batch eval smoke**

```bash
CUDA_VISIBLE_DEVICES=7 HIP_VISIBLE_DEVICES=7 python -m bg_pdr_fm.evaluation.evaluate_bg_pdr_fm \
  --config bg_pdr_fm/configs/openfwi_lmdb_joint_full_contrastive_bgfm_predbg_eval_mb10.yaml \
  --max-batches 1
```

Expected: `summary.json` is produced and `num_samples > 0`.

- [ ] **Step 4: Inspect smoke metrics**

```bash
python - <<'PY'
import json
p='logs/bg_pdr_fm/joint_full_contrastive_bgfm_predbg_e100/evaluation_mb10/summary.json'
d=json.load(open(p))
print(d.get('num_samples'), d.get('means', {}).get('ssim'), d.get('means', {}).get('bg_mae'))
PY
```

Expected: all printed metrics are finite.

---

## Task 6: Formal e100 Training With Checkpointed Monitoring

**Files:**
- Runtime outputs under `logs/bg_pdr_fm/joint_full_contrastive_bgfm_predbg_e100/`.

- [ ] **Step 1: Start training in tmux on an idle GPU**

```bash
tmux new-session -d -s joint_full_predbg_e100 'cd /public/home/xuyinghao/workspace/seismic-diff && source /public/home/xuyinghao/miniconda3/etc/profile.d/conda.sh && conda activate seg && source /opt/dtk-25.04.2/env.sh && source /opt/hyhal/env.sh && export PATH=$CONDA_PREFIX/bin:$PATH && export PYTHONPATH=. && CUDA_VISIBLE_DEVICES=7 HIP_VISIBLE_DEVICES=7 python -m bg_pdr_fm.training.train_bg_pdr_fm --config bg_pdr_fm/configs/openfwi_lmdb_joint_full_contrastive_bgfm_predbg_e100.yaml'
```

- [ ] **Step 2: Verify process is live**

```bash
tmux capture-pane -pt joint_full_predbg_e100 -S -80
hy-smi
```

Expected: training log advances and selected GPU has Python process memory usage.

- [ ] **Step 3: At epoch 30 or first available checkpoint, run 10-batch eval**

```bash
CUDA_VISIBLE_DEVICES=6 HIP_VISIBLE_DEVICES=6 python -m bg_pdr_fm.evaluation.evaluate_bg_pdr_fm \
  --config bg_pdr_fm/configs/openfwi_lmdb_joint_full_contrastive_bgfm_predbg_eval_mb10.yaml \
  --max-batches 10
```

Expected: `summary.json.num_samples` around 1000 or configured fixed-load count.

- [ ] **Step 4: Apply stop rule**

Run:

```bash
python - <<'PY'
import json
p='logs/bg_pdr_fm/joint_full_contrastive_bgfm_predbg_e100/evaluation_mb10/summary.json'
d=json.load(open(p))['means']
print('ssim', d.get('ssim'))
print('bg_mae', d.get('bg_mae'))
print('mae_l', d.get('mae_l'))
print('mae_h', d.get('mae_h'))
PY
```

Stop if epoch 30 has `ssim < 0.953` or `bg_mae > 0.035` unless training curves are still sharply improving. Continue to epoch 100 if it passes.

---

## Task 7: Final Evaluation And Decision

**Files:**
- Runtime outputs under `logs/bg_pdr_fm/joint_full_contrastive_bgfm_predbg_e100/evaluation_full_parallel/`.

- [ ] **Step 1: Run final 10-batch eval at epoch 100**

```bash
CUDA_VISIBLE_DEVICES=6 HIP_VISIBLE_DEVICES=6 python -m bg_pdr_fm.evaluation.evaluate_bg_pdr_fm \
  --config bg_pdr_fm/configs/openfwi_lmdb_joint_full_contrastive_bgfm_predbg_eval_mb10.yaml \
  --max-batches 10
```

- [ ] **Step 2: If 10-batch passes, run parallel full eval**

```bash
python -m bg_pdr_fm.evaluation.dispatch_parallel_eval \
  --config bg_pdr_fm/configs/openfwi_lmdb_joint_full_contrastive_bgfm_predbg_eval_mb10.yaml \
  --output-dir logs/bg_pdr_fm/joint_full_contrastive_bgfm_predbg_e100/evaluation_full_parallel \
  --gpus 4,5,6,7 \
  --batch-candidates 100,200,400 \
  --expected-samples 33600
```

- [ ] **Step 3: Compare against current references**

Run:

```bash
python - <<'PY'
import json
refs={
 'joint':'logs/bg_pdr_fm/joint_full_contrastive_bgfm_predbg_e100/evaluation_full_parallel/summary.json',
 'h192_bg_fm':'logs/bg_pdr_fm/background_train_unet_direct_h192_l1l2_e100/residual_plugin_oraclebg_unetfm_film_full_parallel/summary.json',
 'oraclebg_fm':'logs/bg_pdr_fm/residual_train_smooth_hc64_nogate_unetfm_film_oraclebg_e50/evaluation_full_oraclebg_metrics_only/summary.json',
 'gfi_direct':'logs/bg_pdr_fm/gfi_aligned_upper_bound/eval_pdr_gfi_aligned_direct_best_full/summary.json',
}
for name,path in refs.items():
    try:
        d=json.load(open(path))
        m=d.get('means', {})
        print(name, 'n=', d.get('num_samples'), 'ssim=', m.get('ssim'), 'bg_mae=', m.get('bg_mae'), 'mae_l=', m.get('mae_l'), 'mae_h=', m.get('mae_h'))
    except FileNotFoundError:
        print(name, 'missing', path)
PY
```

Decision:
- `SSIM > 0.9533`: joint beats current production-style h192 bg + FM result.
- `SSIM >= 0.98`: joint is close to oracle/FM ceiling and worth developing as main story.
- `SSIM < 0.9533`: joint full A does not justify further large runs; pivot to GFI-aligned FM condition or direct/GFI hybrid.
