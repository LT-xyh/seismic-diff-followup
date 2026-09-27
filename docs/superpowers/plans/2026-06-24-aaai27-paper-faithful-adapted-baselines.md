# AAAI27 Paper-Faithful Adapted Baselines Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace lightweight/weak AAAI27 main-table baselines with paper-faithful same-input adapted baselines: Smooth-Dix, SV_Inv_Net, VelocityGAN, Conditional DDPM, and paper-faithful Adapted Auto-Linear.

**Architecture:** Add focused external-baseline modules with a shared multimodal input adapter, connect them through `AAAI27BenchmarkLightning`, and keep internal BG-PDR-FM ablations separate from main-table methods. Evaluation continues through the existing unified benchmark evaluator, but table parameter reporting uses method-specific effective parameters rather than wrapper-level totals.

**Tech Stack:** PyTorch, Lightning, diffusers `UNet2DConditionModel` and `DDPMScheduler`, OmegaConf configs, existing BG-PDR-FM dataset/evaluation utilities, pytest.

---

## File Structure

- Create `bg_pdr_fm/external_baselines/common.py`
  - Shared five-channel multimodal input builder and SSIM loss helper.
- Create `bg_pdr_fm/external_baselines/sv_inv_net.py`
  - Full-capacity adapted SVInvNet-style dense-block encoder-decoder.
- Create `bg_pdr_fm/external_baselines/conditional_ddpm.py`
  - Diffusers conditional DDPM baseline with condition encoder and scheduler.
- Create `bg_pdr_fm/external_baselines/velocity_gan.py`
  - Conditional GAN generator, discriminator, and losses.
- Replace internals in `bg_pdr_fm/external_baselines/auto_linear/adapted_multimodal.py`
  - Masked autoencoder style implementation plus low-rank/two-layer latent converter.
- Modify `bg_pdr_fm/lightning/benchmark_module.py`
  - Add variants: `smooth_dix`, `sv_inv_net`, `conditional_ddpm`, `velocity_gan`.
  - Remove main-table use of `mm_invnet` and `two_stage_ddpm`; keep old variants only if needed for backward-compatible old checkpoints, but do not use them in new configs.
  - Add variant-specific loss/predict/metadata paths.
- Modify `bg_pdr_fm/training/train_aaai27_benchmark.py`
  - Extend config validation.
- Modify `bg_pdr_fm/evaluation/compare_experiments.py`
  - Treat `smooth_dix` as no-checkpoint/allow-untrained by default.
  - Ensure `has_background` excludes non-background baselines.
  - Preserve per-dataset and missing-mode CSV outputs.
- Modify `bg_pdr_fm/evaluation/run_aaai27_openfwi_eval.py`
  - Replace old main-table method list.
- Modify `bg_pdr_fm/evaluation/run_aaai27_marmousi_eval.py`
  - Replace old main-table method list and keep PDR-FM opt-out support.
- Create configs under `bg_pdr_fm/configs/experiments/aaai27/`
  - `formal_smooth_dix.yaml`
  - `sv_inv_net.yaml`
  - `formal_sv_inv_net.yaml`
  - `velocity_gan.yaml`
  - `formal_velocity_gan.yaml`
  - `conditional_ddpm.yaml`
  - `formal_conditional_ddpm.yaml`
  - revised `adapted_auto_linear_ae_pretrain.yaml`
  - revised `adapted_auto_linear_multimodal.yaml`
  - revised `formal_adapted_auto_linear_multimodal.yaml`
- Modify docs:
  - `docs/paper/aaai27_method_registry.md`
  - `docs/paper/aaai27_constructed_method_inventory.md`
  - `docs/paper/aaai_LaTeX/seismic_diff_method_appendix.tex`
- Modify tests:
  - `bg_pdr_fm/tests/test_smoke.py`

## Task 1: Shared Multimodal Input Adapter

**Files:**
- Create: `bg_pdr_fm/external_baselines/common.py`
- Modify: `bg_pdr_fm/external_baselines/gfi/adapted_multimodal.py`
- Modify: `bg_pdr_fm/external_baselines/auto_linear/adapted_multimodal.py`
- Test: `bg_pdr_fm/tests/test_smoke.py`

- [ ] **Step 1: Write failing tests for shared adapter**

Add tests to `bg_pdr_fm/tests/test_smoke.py`:

```python
def test_multimodal_condition_image_builder_outputs_five_channels():
    from bg_pdr_fm.external_baselines.common import build_multimodal_condition_image

    batch = _dummy_bg_batch(batch_size=2)
    x = build_multimodal_condition_image(batch, input_hw=(1000, 70))

    assert tuple(x.shape) == (2, 5, 1000, 70)
    assert torch.isfinite(x).all()


def test_multimodal_condition_image_builder_respects_missing_well_log():
    from bg_pdr_fm.evaluation.missing_modalities import apply_missing_modality_mode
    from bg_pdr_fm.external_baselines.common import build_multimodal_condition_image

    batch = _dummy_bg_batch(batch_size=2)
    no_well = apply_missing_modality_mode(batch, "w/o well_log")
    x = build_multimodal_condition_image(no_well, input_hw=(1000, 70))

    assert torch.count_nonzero(x[:, 3]).item() == 0
    assert torch.count_nonzero(x[:, 4]).item() == 0
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
pytest bg_pdr_fm/tests/test_smoke.py::test_multimodal_condition_image_builder_outputs_five_channels bg_pdr_fm/tests/test_smoke.py::test_multimodal_condition_image_builder_respects_missing_well_log -q
```

Expected: fail because `bg_pdr_fm.external_baselines.common` does not exist.

- [ ] **Step 3: Implement shared adapter**

Create `bg_pdr_fm/external_baselines/common.py`:

```python
"""Shared utilities for same-input adapted external baselines."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from bg_pdr_fm.data.batch import BGSampleBatch


def resize_channel(x: torch.Tensor, size: tuple[int, int], *, mode: str) -> torch.Tensor:
    if x.ndim != 4:
        raise ValueError(f"expected BCHW tensor, got {tuple(x.shape)}")
    if mode == "nearest":
        return F.interpolate(x, size=tuple(size), mode=mode)
    return F.interpolate(x, size=tuple(size), mode=mode, align_corners=False)


def build_multimodal_condition_image(
    batch: BGSampleBatch,
    input_hw: tuple[int, int] = (1000, 70),
) -> torch.Tensor:
    input_hw = tuple(int(v) for v in input_hw)
    migrated = resize_channel(batch.migrated_image, input_hw, mode="bilinear")
    rms = resize_channel(batch.rms_vel, input_hw, mode="bilinear")
    horizon = resize_channel(batch.horizon, input_hw, mode="nearest")
    well_mask = resize_channel(batch.well_mask, input_hw, mode="nearest")
    well_log = resize_channel(batch.well_log * batch.well_mask, input_hw, mode="bilinear")
    return torch.cat([migrated, rms, horizon, well_log, well_mask], dim=1).contiguous()
```

- [ ] **Step 4: Reuse adapter in GFI and Auto-Linear**

In `bg_pdr_fm/external_baselines/gfi/adapted_multimodal.py`, replace local `_resize_channel` and `build_adapted_gfi_input` body with:

```python
from bg_pdr_fm.external_baselines.common import build_multimodal_condition_image


def build_adapted_gfi_input(batch: BGSampleBatch, input_hw: tuple[int, int] = (1000, 70)) -> torch.Tensor:
    """Convert BG-PDR-FM multimodal conditions into a GFI-compatible tensor."""
    return build_multimodal_condition_image(batch, input_hw)
```

In `bg_pdr_fm/external_baselines/auto_linear/adapted_multimodal.py`, make `build_auto_linear_measurement_input` call `build_multimodal_condition_image`.

- [ ] **Step 5: Run tests**

Run:

```bash
pytest bg_pdr_fm/tests/test_smoke.py::test_multimodal_condition_image_builder_outputs_five_channels bg_pdr_fm/tests/test_smoke.py::test_multimodal_condition_image_builder_respects_missing_well_log -q
```

Expected: pass.

## Task 2: Smooth-Dix Baseline

**Files:**
- Create: `bg_pdr_fm/external_baselines/smooth_dix.py`
- Modify: `bg_pdr_fm/lightning/benchmark_module.py`
- Create: `bg_pdr_fm/configs/experiments/aaai27/formal_smooth_dix.yaml`
- Test: `bg_pdr_fm/tests/test_smoke.py`

- [ ] **Step 1: Write failing tests**

Add:

```python
def test_smooth_dix_predicts_velocity_shape_and_zero_params():
    from bg_pdr_fm.external_baselines.smooth_dix import SmoothDixBaseline

    batch = _dummy_bg_batch(batch_size=2)
    model = SmoothDixBaseline(kernel_size=9)
    y = model(batch)

    assert tuple(y.shape) == tuple(batch.depth_vel.shape)
    assert torch.isfinite(y).all()
    assert sum(p.numel() for p in model.parameters()) == 0
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```bash
pytest bg_pdr_fm/tests/test_smoke.py::test_smooth_dix_predicts_velocity_shape_and_zero_params -q
```

Expected: fail because module does not exist.

- [ ] **Step 3: Implement Smooth-Dix**

Create `bg_pdr_fm/external_baselines/smooth_dix.py`:

```python
"""Traditional smooth RMS/Dix-style baseline for AAAI27 evaluation."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from bg_pdr_fm.data.batch import BGSampleBatch


class SmoothDixBaseline(nn.Module):
    def __init__(self, kernel_size: int = 9) -> None:
        super().__init__()
        if kernel_size <= 0 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        self.kernel_size = int(kernel_size)

    def forward(self, batch: BGSampleBatch) -> torch.Tensor:
        x = F.interpolate(batch.rms_vel, size=batch.depth_vel.shape[-2:], mode="bilinear", align_corners=False)
        pad = self.kernel_size // 2
        x = F.avg_pool2d(F.pad(x, (pad, pad, pad, pad), mode="reflect"), kernel_size=self.kernel_size, stride=1)
        return torch.clamp(x, -1.0, 1.0)
```

- [ ] **Step 4: Wire into benchmark module**

In `AAAI27_VARIANTS`, add `"smooth_dix"`.

In `__init__`, instantiate:

```python
self.smooth_dix = SmoothDixBaseline(kernel_size=int(conf_get(self.conf, "benchmark.smooth_dix_kernel_size", 9)))
```

In `_apply_variant_trainability`, set it frozen and add:

```python
elif self.variant == "smooth_dix":
    pass
```

In `_step`, add:

```python
if self.variant == "smooth_dix":
    recon = self.smooth_dix(batch)
    loss = F.l1_loss(recon, batch.depth_vel)
    self._log_epoch_metric(f"{prefix}/loss", loss, prog_bar=True)
    self._log_epoch_metric(f"{prefix}/mae", loss)
    return loss
```

In `predict_batch`, return a direct prediction branch matching `adapted_gfi`.

In `benchmark_metadata`, return `[self.smooth_dix]` for `smooth_dix`.

- [ ] **Step 5: Create config**

Create `bg_pdr_fm/configs/experiments/aaai27/formal_smooth_dix.yaml`:

```yaml
extends: _base_formal.yaml

benchmark:
  variant: smooth_dix
  capacity_tier: traditional
  smooth_dix_kernel_size: 9

evaluation:
  output_dir: logs/bg_pdr_fm/aaai27/eval_smooth_dix/smooth_dix
  allow_untrained: true
  missing_modes: [full]
```

- [ ] **Step 6: Run tests**

Run:

```bash
pytest bg_pdr_fm/tests/test_smoke.py::test_smooth_dix_predicts_velocity_shape_and_zero_params -q
python -m py_compile bg_pdr_fm/external_baselines/smooth_dix.py bg_pdr_fm/lightning/benchmark_module.py
```

Expected: pass.

## Task 3: SV_Inv_Net Baseline

**Files:**
- Create: `bg_pdr_fm/external_baselines/sv_inv_net.py`
- Modify: `bg_pdr_fm/lightning/benchmark_module.py`
- Modify: `bg_pdr_fm/training/train_aaai27_benchmark.py`
- Create: `bg_pdr_fm/configs/experiments/aaai27/sv_inv_net.yaml`
- Create: `bg_pdr_fm/configs/experiments/aaai27/formal_sv_inv_net.yaml`
- Test: `bg_pdr_fm/tests/test_smoke.py`

- [ ] **Step 1: Write failing tests**

Add:

```python
def test_sv_inv_net_forward_shape_and_capacity():
    from bg_pdr_fm.external_baselines.sv_inv_net import SVInvNetMultimodal

    batch = _dummy_bg_batch(batch_size=1)
    model = SVInvNetMultimodal(base_channels=32, input_hw=(1000, 70))
    y = model(batch)
    params = sum(p.numel() for p in model.parameters())

    assert tuple(y.shape) == tuple(batch.depth_vel.shape)
    assert torch.isfinite(y).all()
    assert params > 1_000_000
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```bash
pytest bg_pdr_fm/tests/test_smoke.py::test_sv_inv_net_forward_shape_and_capacity -q
```

Expected: fail because module does not exist.

- [ ] **Step 3: Implement SVInvNet module**

Create `bg_pdr_fm/external_baselines/sv_inv_net.py` with:

```python
"""Adapted SVInvNet baseline for multimodal velocity prediction."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from bg_pdr_fm.data.batch import BGSampleBatch
from bg_pdr_fm.external_baselines.common import build_multimodal_condition_image


class DenseLayer(nn.Module):
    def __init__(self, in_channels: int, growth_rate: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([x, self.net(x)], dim=1)


class DenseBlock(nn.Module):
    def __init__(self, in_channels: int, growth_rate: int, layers: int) -> None:
        super().__init__()
        modules = []
        channels = in_channels
        for _ in range(layers):
            modules.append(DenseLayer(channels, growth_rate))
            channels += growth_rate
        self.net = nn.Sequential(*modules)
        self.out_channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, growth_rate: int, layers: int) -> None:
        super().__init__()
        self.down = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.dense = DenseBlock(out_channels, growth_rate, layers)
        self.project = nn.Conv2d(self.dense.out_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.project(self.dense(self.down(x)))


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat([x, skip], dim=1))


class SVInvNetMultimodal(nn.Module):
    def __init__(self, base_channels: int = 32, growth_rate: int = 16, dense_layers: int = 4, input_hw: tuple[int, int] = (1000, 70)) -> None:
        super().__init__()
        self.input_hw = tuple(int(v) for v in input_hw)
        c = int(base_channels)
        self.stem = nn.Sequential(
            nn.Conv2d(5, c, kernel_size=7, stride=(4, 1), padding=(3, 3), bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        )
        self.down1 = DownBlock(c, c * 2, growth_rate, dense_layers)
        self.down2 = DownBlock(c * 2, c * 4, growth_rate, dense_layers)
        self.down3 = DownBlock(c * 4, c * 8, growth_rate, dense_layers)
        self.down4 = DownBlock(c * 8, c * 8, growth_rate, dense_layers)
        self.bottleneck = DenseBlock(c * 8, growth_rate, dense_layers)
        self.bottleneck_project = nn.Conv2d(self.bottleneck.out_channels, c * 8, kernel_size=1, bias=False)
        self.up4 = UpBlock(c * 8, c * 8, c * 4)
        self.up3 = UpBlock(c * 4, c * 4, c * 2)
        self.up2 = UpBlock(c * 2, c * 2, c)
        self.up1 = UpBlock(c, c, c)
        self.head = nn.Sequential(nn.Conv2d(c, 1, kernel_size=3, padding=1), nn.Tanh())

    def forward(self, batch: BGSampleBatch) -> torch.Tensor:
        x = build_multimodal_condition_image(batch, self.input_hw)
        s0 = self.stem(x)
        s1 = self.down1(s0)
        s2 = self.down2(s1)
        s3 = self.down3(s2)
        x = self.down4(s3)
        x = self.bottleneck_project(self.bottleneck(x))
        x = self.up4(x, s3)
        x = self.up3(x, s2)
        x = self.up2(x, s1)
        x = self.up1(x, s0)
        x = F.interpolate(x, size=batch.depth_vel.shape[-2:], mode="bilinear", align_corners=False)
        return self.head(x)
```

- [ ] **Step 4: Wire variant**

Add `SVInvNetMultimodal` import and instantiate:

```python
self.sv_inv_net = SVInvNetMultimodal(
    base_channels=int(conf_get(self.conf, "benchmark.sv_inv_net_base_channels", 32)),
    growth_rate=int(conf_get(self.conf, "benchmark.sv_inv_net_growth_rate", 16)),
    dense_layers=int(conf_get(self.conf, "benchmark.sv_inv_net_dense_layers", 4)),
    input_hw=tuple(conf_get(self.conf, "benchmark.sv_inv_net_input_hw", (1000, 70))),
)
```

Add trainability branch for `sv_inv_net`, supervised loss branch with:

```python
recon = self.sv_inv_net(batch)
l1 = F.l1_loss(recon, batch.depth_vel)
ssim_loss = 1.0 - compute_batch_ssim_like(recon, batch.depth_vel)
loss = l1 + float(conf_get(self.conf, "loss.ssim_weight", 0.1)) * ssim_loss
```

If no SSIM helper exists in scope, use a local differentiable proxy first:

```python
ssim_loss = F.l1_loss(self.filter.lowpass(recon), self.filter.lowpass(batch.depth_vel))
```

Do not block the implementation on noncritical SSIM helper placement.

- [ ] **Step 5: Add configs**

Create `sv_inv_net.yaml`:

```yaml
extends: _base_formal.yaml

benchmark:
  variant: sv_inv_net
  capacity_tier: adapted_external
  sv_inv_net_base_channels: 32
  sv_inv_net_growth_rate: 16
  sv_inv_net_dense_layers: 4
  sv_inv_net_input_hw: [1000, 70]

training:
  max_epochs: 200
  batch_size: 32
  lr: 0.0001
  logger:
    log_dir: logs/bg_pdr_fm/aaai27/formal/sv_inv_net/lightning
    log_version: "sv_inv_net_{time}"
  checkpoint:
    dirpath: logs/bg_pdr_fm/aaai27/formal/sv_inv_net/checkpoints

loss:
  ssim_weight: 0.1

evaluation:
  output_dir: logs/bg_pdr_fm/aaai27/eval_sv_inv_net/sv_inv_net
  checkpoint: logs/bg_pdr_fm/aaai27/formal/sv_inv_net/checkpoints/last.ckpt
```

Create `formal_sv_inv_net.yaml` extending `sv_inv_net.yaml` and setting full eval output/checkpoint paths.

- [ ] **Step 6: Run tests**

Run:

```bash
pytest bg_pdr_fm/tests/test_smoke.py::test_sv_inv_net_forward_shape_and_capacity -q
python -m py_compile bg_pdr_fm/external_baselines/sv_inv_net.py bg_pdr_fm/lightning/benchmark_module.py bg_pdr_fm/training/train_aaai27_benchmark.py
```

Expected: pass.

## Task 4: Conditional DDPM with diffusers UNet2DConditionModel

**Files:**
- Create: `bg_pdr_fm/external_baselines/conditional_ddpm.py`
- Modify: `bg_pdr_fm/lightning/benchmark_module.py`
- Modify: `bg_pdr_fm/training/train_aaai27_benchmark.py`
- Create: `bg_pdr_fm/configs/experiments/aaai27/conditional_ddpm.yaml`
- Create: `bg_pdr_fm/configs/experiments/aaai27/formal_conditional_ddpm.yaml`
- Test: `bg_pdr_fm/tests/test_smoke.py`

- [ ] **Step 1: Write failing tests**

Add:

```python
def test_conditional_ddpm_training_loss_and_prediction_shape():
    from bg_pdr_fm.external_baselines.conditional_ddpm import ConditionalDDPMBaseline

    batch = _dummy_bg_batch(batch_size=2)
    model = ConditionalDDPMBaseline(
        sample_size=70,
        condition_input_hw=(1000, 70),
        condition_dim=128,
        train_timesteps=10,
        inference_steps=2,
        block_out_channels=(32, 64),
    )
    loss_dict = model.training_loss(batch)
    y = model.sample(batch)

    assert loss_dict["loss"].ndim == 0
    assert torch.isfinite(loss_dict["loss"])
    assert tuple(y.shape) == tuple(batch.depth_vel.shape)
    assert torch.isfinite(y).all()
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```bash
pytest bg_pdr_fm/tests/test_smoke.py::test_conditional_ddpm_training_loss_and_prediction_shape -q
```

Expected: fail because module does not exist.

- [ ] **Step 3: Implement Conditional DDPM**

Create `bg_pdr_fm/external_baselines/conditional_ddpm.py`:

```python
"""Diffusers conditional DDPM baseline for multimodal velocity prediction."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDPMScheduler, UNet2DConditionModel

from bg_pdr_fm.data.batch import BGSampleBatch
from bg_pdr_fm.external_baselines.common import build_multimodal_condition_image


class ConditionTokenEncoder(nn.Module):
    def __init__(self, condition_dim: int = 256, input_hw: tuple[int, int] = (1000, 70), num_tokens: int = 16) -> None:
        super().__init__()
        self.input_hw = tuple(int(v) for v in input_hw)
        self.num_tokens = int(num_tokens)
        self.net = nn.Sequential(
            nn.Conv2d(5, 32, kernel_size=7, stride=(4, 2), padding=3),
            nn.SiLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=(4, 2), padding=1),
            nn.SiLU(),
            nn.Conv2d(64, condition_dim, kernel_size=3, stride=(4, 2), padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((self.num_tokens, 1)),
        )

    def forward(self, batch: BGSampleBatch) -> torch.Tensor:
        x = build_multimodal_condition_image(batch, self.input_hw)
        x = self.net(x).squeeze(-1).transpose(1, 2).contiguous()
        return x


class ConditionalDDPMBaseline(nn.Module):
    def __init__(
        self,
        sample_size: int = 70,
        condition_input_hw: tuple[int, int] = (1000, 70),
        condition_dim: int = 256,
        num_condition_tokens: int = 16,
        train_timesteps: int = 1000,
        inference_steps: int = 50,
        block_out_channels: tuple[int, ...] = (64, 128, 256),
    ) -> None:
        super().__init__()
        self.condition_encoder = ConditionTokenEncoder(condition_dim, condition_input_hw, num_condition_tokens)
        self.scheduler = DDPMScheduler(num_train_timesteps=int(train_timesteps), prediction_type="epsilon")
        self.inference_steps = int(inference_steps)
        self.unet = UNet2DConditionModel(
            sample_size=int(sample_size),
            in_channels=1,
            out_channels=1,
            layers_per_block=2,
            block_out_channels=tuple(int(v) for v in block_out_channels),
            down_block_types=("DownBlock2D",) * len(block_out_channels),
            up_block_types=("UpBlock2D",) * len(block_out_channels),
            cross_attention_dim=int(condition_dim),
            norm_num_groups=8,
        )

    def training_loss(self, batch: BGSampleBatch) -> dict[str, torch.Tensor]:
        clean = batch.depth_vel
        noise = torch.randn_like(clean)
        timesteps = torch.randint(
            0,
            self.scheduler.config.num_train_timesteps,
            (clean.shape[0],),
            device=clean.device,
            dtype=torch.long,
        )
        noisy = self.scheduler.add_noise(clean, noise, timesteps)
        cond = self.condition_encoder(batch)
        pred = self.unet(noisy, timesteps, encoder_hidden_states=cond).sample
        loss = F.mse_loss(pred, noise)
        return {"loss": loss, "noise_mse": loss}

    @torch.no_grad()
    def sample(self, batch: BGSampleBatch) -> torch.Tensor:
        cond = self.condition_encoder(batch)
        sample = torch.randn_like(batch.depth_vel)
        self.scheduler.set_timesteps(self.inference_steps, device=sample.device)
        for timestep in self.scheduler.timesteps:
            model_out = self.unet(sample, timestep, encoder_hidden_states=cond).sample
            sample = self.scheduler.step(model_out, timestep, sample).prev_sample
        return torch.clamp(sample, -1.0, 1.0)
```

- [ ] **Step 4: Wire variant**

Add import, instantiate `self.conditional_ddpm`, trainability branch, `_step` branch using `training_loss`, and `predict_batch` branch using `sample`.

Add metadata generator modules `[self.conditional_ddpm]`.

Remove new main-table dependency on `two_stage_ddpm`.

- [ ] **Step 5: Add configs**

Create `conditional_ddpm.yaml`:

```yaml
extends: _base_formal.yaml

benchmark:
  variant: conditional_ddpm
  capacity_tier: adapted_external
  ddpm_condition_dim: 256
  ddpm_num_condition_tokens: 16
  ddpm_block_out_channels: [64, 128, 256]
  ddpm_inference_steps: 50

model:
  residual_num_train_timesteps: 1000

training:
  max_epochs: 200
  batch_size: 32
  lr: 0.0001
  logger:
    log_dir: logs/bg_pdr_fm/aaai27/formal/conditional_ddpm/lightning
    log_version: "conditional_ddpm_{time}"
  checkpoint:
    dirpath: logs/bg_pdr_fm/aaai27/formal/conditional_ddpm/checkpoints

evaluation:
  output_dir: logs/bg_pdr_fm/aaai27/eval_conditional_ddpm/conditional_ddpm
  checkpoint: logs/bg_pdr_fm/aaai27/formal/conditional_ddpm/checkpoints/last.ckpt
```

Create `formal_conditional_ddpm.yaml` extending it.

- [ ] **Step 6: Run tests**

Run:

```bash
pytest bg_pdr_fm/tests/test_smoke.py::test_conditional_ddpm_training_loss_and_prediction_shape -q
python -m py_compile bg_pdr_fm/external_baselines/conditional_ddpm.py bg_pdr_fm/lightning/benchmark_module.py
```

Expected: pass.

## Task 5: VelocityGAN Baseline

**Files:**
- Create: `bg_pdr_fm/external_baselines/velocity_gan.py`
- Modify: `bg_pdr_fm/lightning/benchmark_module.py`
- Modify: `bg_pdr_fm/training/train_aaai27_benchmark.py`
- Create: `bg_pdr_fm/configs/experiments/aaai27/velocity_gan.yaml`
- Create: `bg_pdr_fm/configs/experiments/aaai27/formal_velocity_gan.yaml`
- Test: `bg_pdr_fm/tests/test_smoke.py`

- [ ] **Step 1: Write failing tests**

Add:

```python
def test_velocity_gan_generator_and_losses_are_finite():
    from bg_pdr_fm.external_baselines.velocity_gan import VelocityGANBaseline

    batch = _dummy_bg_batch(batch_size=2)
    model = VelocityGANBaseline(base_channels=32, input_hw=(1000, 70))
    losses = model.generator_loss(batch)
    y = model(batch)

    assert tuple(y.shape) == tuple(batch.depth_vel.shape)
    assert torch.isfinite(losses["loss"])
    assert torch.isfinite(losses["adv_loss"])
    assert torch.isfinite(losses["l1"])
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```bash
pytest bg_pdr_fm/tests/test_smoke.py::test_velocity_gan_generator_and_losses_are_finite -q
```

Expected: fail because module does not exist.

- [ ] **Step 3: Implement VelocityGAN**

Create `bg_pdr_fm/external_baselines/velocity_gan.py`:

```python
"""Adapted VelocityGAN-style conditional GAN baseline."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from bg_pdr_fm.data.batch import BGSampleBatch
from bg_pdr_fm.external_baselines.common import build_multimodal_condition_image, resize_channel


class UNetGenerator(nn.Module):
    def __init__(self, base_channels: int = 64, input_hw: tuple[int, int] = (1000, 70)) -> None:
        super().__init__()
        self.input_hw = tuple(int(v) for v in input_hw)
        c = int(base_channels)
        self.enc1 = nn.Sequential(nn.Conv2d(5, c, 4, 2, 1), nn.LeakyReLU(0.2, inplace=True))
        self.enc2 = nn.Sequential(nn.Conv2d(c, c * 2, 4, 2, 1), nn.BatchNorm2d(c * 2), nn.LeakyReLU(0.2, inplace=True))
        self.enc3 = nn.Sequential(nn.Conv2d(c * 2, c * 4, 4, 2, 1), nn.BatchNorm2d(c * 4), nn.LeakyReLU(0.2, inplace=True))
        self.enc4 = nn.Sequential(nn.Conv2d(c * 4, c * 8, 4, 2, 1), nn.BatchNorm2d(c * 8), nn.LeakyReLU(0.2, inplace=True))
        self.dec3 = nn.Sequential(nn.Conv2d(c * 8 + c * 4, c * 4, 3, padding=1), nn.BatchNorm2d(c * 4), nn.ReLU(inplace=True))
        self.dec2 = nn.Sequential(nn.Conv2d(c * 4 + c * 2, c * 2, 3, padding=1), nn.BatchNorm2d(c * 2), nn.ReLU(inplace=True))
        self.dec1 = nn.Sequential(nn.Conv2d(c * 2 + c, c, 3, padding=1), nn.BatchNorm2d(c), nn.ReLU(inplace=True))
        self.head = nn.Sequential(nn.Conv2d(c, 1, 3, padding=1), nn.Tanh())

    def forward(self, batch: BGSampleBatch) -> torch.Tensor:
        x = build_multimodal_condition_image(batch, self.input_hw)
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        d3 = F.interpolate(e4, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = F.interpolate(d3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        y = F.interpolate(d1, size=batch.depth_vel.shape[-2:], mode="bilinear", align_corners=False)
        return self.head(y)


class PatchDiscriminator(nn.Module):
    def __init__(self, base_channels: int = 64) -> None:
        super().__init__()
        c = int(base_channels)
        self.net = nn.Sequential(
            nn.Conv2d(6, c, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c, c * 2, 4, 2, 1),
            nn.BatchNorm2d(c * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c * 2, c * 4, 4, 2, 1),
            nn.BatchNorm2d(c * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(c * 4, 1, 3, padding=1),
        )

    def forward(self, condition_70: torch.Tensor, velocity: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([condition_70, velocity], dim=1))


class VelocityGANBaseline(nn.Module):
    def __init__(self, base_channels: int = 64, input_hw: tuple[int, int] = (1000, 70), l1_weight: float = 100.0) -> None:
        super().__init__()
        self.generator = UNetGenerator(base_channels, input_hw)
        self.discriminator = PatchDiscriminator(base_channels)
        self.l1_weight = float(l1_weight)

    def condition_70(self, batch: BGSampleBatch) -> torch.Tensor:
        return build_multimodal_condition_image(batch, batch.depth_vel.shape[-2:])

    def forward(self, batch: BGSampleBatch) -> torch.Tensor:
        return self.generator(batch)

    def generator_loss(self, batch: BGSampleBatch) -> dict[str, torch.Tensor]:
        fake = self.generator(batch)
        pred_fake = self.discriminator(self.condition_70(batch), fake)
        adv = F.binary_cross_entropy_with_logits(pred_fake, torch.ones_like(pred_fake))
        l1 = F.l1_loss(fake, batch.depth_vel)
        return {"loss": adv + self.l1_weight * l1, "adv_loss": adv, "l1": l1, "velocity_hat": fake}

    def discriminator_loss(self, batch: BGSampleBatch) -> torch.Tensor:
        cond = self.condition_70(batch)
        with torch.no_grad():
            fake = self.generator(batch)
        pred_real = self.discriminator(cond, batch.depth_vel)
        pred_fake = self.discriminator(cond, fake)
        real_loss = F.binary_cross_entropy_with_logits(pred_real, torch.ones_like(pred_real))
        fake_loss = F.binary_cross_entropy_with_logits(pred_fake, torch.zeros_like(pred_fake))
        return 0.5 * (real_loss + fake_loss)
```

- [ ] **Step 4: Wire variant**

Add import, instantiate `self.velocity_gan`, trainability branch, `_step` branch using `generator_loss`. For first implementation use one optimizer for generator and discriminator parameters through `loss = generator_loss + discriminator_loss` to keep Lightning integration simple. Later split optimizers if needed.

Prediction uses `self.velocity_gan(batch)`.

Metadata main generator params use `self.velocity_gan.generator`; add discriminator params to metadata as `discriminator_params`.

- [ ] **Step 5: Add configs**

Create `velocity_gan.yaml`:

```yaml
extends: _base_formal.yaml

benchmark:
  variant: velocity_gan
  capacity_tier: adapted_external
  velocity_gan_base_channels: 64
  velocity_gan_l1_weight: 100.0

training:
  max_epochs: 200
  batch_size: 32
  lr: 0.0001
  logger:
    log_dir: logs/bg_pdr_fm/aaai27/formal/velocity_gan/lightning
    log_version: "velocity_gan_{time}"
  checkpoint:
    dirpath: logs/bg_pdr_fm/aaai27/formal/velocity_gan/checkpoints

evaluation:
  output_dir: logs/bg_pdr_fm/aaai27/eval_velocity_gan/velocity_gan
  checkpoint: logs/bg_pdr_fm/aaai27/formal/velocity_gan/checkpoints/last.ckpt
```

Create `formal_velocity_gan.yaml` extending it.

- [ ] **Step 6: Run tests**

Run:

```bash
pytest bg_pdr_fm/tests/test_smoke.py::test_velocity_gan_generator_and_losses_are_finite -q
python -m py_compile bg_pdr_fm/external_baselines/velocity_gan.py bg_pdr_fm/lightning/benchmark_module.py
```

Expected: pass.

## Task 6: Paper-Faithful Adapted Auto-Linear

**Files:**
- Modify: `bg_pdr_fm/external_baselines/auto_linear/adapted_multimodal.py`
- Modify: `bg_pdr_fm/lightning/benchmark_module.py`
- Modify configs:
  - `bg_pdr_fm/configs/experiments/aaai27/adapted_auto_linear_ae_pretrain.yaml`
  - `bg_pdr_fm/configs/experiments/aaai27/adapted_auto_linear_multimodal.yaml`
  - `bg_pdr_fm/configs/experiments/aaai27/formal_adapted_auto_linear_multimodal.yaml`
- Test: `bg_pdr_fm/tests/test_smoke.py`

- [ ] **Step 1: Write failing tests for masked AE and low-rank converter**

Add:

```python
def test_adapted_auto_linear_uses_masked_autoencoders_and_low_rank_converter():
    from bg_pdr_fm.external_baselines.auto_linear.adapted_multimodal import AdaptedAutoLinearMultimodal

    model = AdaptedAutoLinearMultimodal(
        embed_dim=128,
        depth=2,
        num_heads=4,
        latent_tokens=16,
        converter_rank=32,
        input_hw=(1000, 70),
    )

    assert hasattr(model, "measurement_mae")
    assert hasattr(model, "velocity_mae")
    assert hasattr(model, "latent_converter")
    assert hasattr(model.latent_converter, "down")
    assert hasattr(model.latent_converter, "up")


def test_adapted_auto_linear_losses_are_finite():
    from bg_pdr_fm.external_baselines.auto_linear.adapted_multimodal import AdaptedAutoLinearMultimodal

    batch = _dummy_bg_batch(batch_size=2)
    model = AdaptedAutoLinearMultimodal(
        embed_dim=64,
        depth=1,
        num_heads=4,
        latent_tokens=8,
        converter_rank=16,
        input_hw=(1000, 70),
    )
    ae = model.ae_pretrain_loss(batch)
    inv = model.inverse_linear_loss(batch)
    y = model(batch)

    assert torch.isfinite(ae["loss"])
    assert torch.isfinite(inv["loss"])
    assert tuple(y.shape) == tuple(batch.depth_vel.shape)
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
pytest bg_pdr_fm/tests/test_smoke.py::test_adapted_auto_linear_uses_masked_autoencoders_and_low_rank_converter bg_pdr_fm/tests/test_smoke.py::test_adapted_auto_linear_losses_are_finite -q
```

Expected: fail because current implementation uses CNN encoders and `latent_map`.

- [ ] **Step 3: Replace implementation with masked transformer-style AEs**

Implement:

- `PatchEmbed2D`
- `MaskedAutoEncoder2D`
- `LowRankLatentConverter`
- `AdaptedAutoLinearMultimodal`

Minimum required API compatibility:

```python
def ae_pretrain_loss(self, batch: BGSampleBatch) -> dict[str, torch.Tensor]
def inverse_linear_loss(self, batch: BGSampleBatch) -> dict[str, torch.Tensor]
def forward(self, batch: BGSampleBatch) -> torch.Tensor
```

The AE pretrain loss must reconstruct measurement and velocity tensors. The inverse phase must:

```python
z_c = self.measurement_mae.encode(self.measurement_input(batch), mask_ratio=0.0)
with torch.no_grad():
    z_v = self.velocity_mae.encode(batch.depth_vel, mask_ratio=0.0)
z_v_hat = self.latent_converter(z_c)
velocity_hat = self.velocity_mae.decode(z_v_hat, out_hw=batch.depth_vel.shape[-2:])
loss = mse(z_v_hat, z_v) + l1(velocity_hat, batch.depth_vel)
```

- [ ] **Step 4: Update benchmark trainability**

In `benchmark_module.py`, update Auto-Linear phase trainability:

For `ae_pretrain`, enable:

```python
self.adapted_auto_linear.measurement_mae
self.adapted_auto_linear.velocity_mae
```

For `inverse_linear`, enable only:

```python
self.adapted_auto_linear.latent_converter
```

Keep the two AEs in eval mode during inverse training.

- [ ] **Step 5: Update configs**

In AE config, use:

```yaml
benchmark:
  variant: adapted_auto_linear
  capacity_tier: adapted_external
  auto_linear_phase: ae_pretrain
  auto_linear_embed_dim: 256
  auto_linear_depth: 4
  auto_linear_num_heads: 8
  auto_linear_latent_tokens: 16
  auto_linear_converter_rank: 64
  auto_linear_mask_ratio: 0.5
```

In inverse config, use same architecture keys and:

```yaml
benchmark:
  auto_linear_phase: inverse_linear
training:
  load_stage_checkpoint: logs/bg_pdr_fm/aaai27/formal/adapted_auto_linear_ae_pretrain/checkpoints/last.ckpt
```

- [ ] **Step 6: Run tests**

Run:

```bash
pytest bg_pdr_fm/tests/test_smoke.py::test_adapted_auto_linear_uses_masked_autoencoders_and_low_rank_converter bg_pdr_fm/tests/test_smoke.py::test_adapted_auto_linear_losses_are_finite bg_pdr_fm/tests/test_smoke.py::test_adapted_auto_linear_phase_loss_and_trainability -q
python -m py_compile bg_pdr_fm/external_baselines/auto_linear/adapted_multimodal.py bg_pdr_fm/lightning/benchmark_module.py
```

Expected: pass.

## Task 7: Benchmark Variant Registry and Backward Compatibility

**Files:**
- Modify: `bg_pdr_fm/lightning/benchmark_module.py`
- Modify: `bg_pdr_fm/training/train_aaai27_benchmark.py`
- Modify: `bg_pdr_fm/evaluation/compare_experiments.py`
- Modify: `bg_pdr_fm/tests/test_smoke.py`

- [ ] **Step 1: Write failing variant registry test**

Update parametrized variant tests to include:

```python
["smooth_dix", "sv_inv_net", "velocity_gan", "conditional_ddpm", "adapted_gfi", "adapted_auto_linear", "concat_fm", "cncs_fm", "bg_pdr_fm"]
```

Remove `mm_invnet` and `two_stage_ddpm` from new main registry tests. If old compatibility tests remain, name them `legacy`.

- [ ] **Step 2: Run tests and verify failure**

Run:

```bash
pytest bg_pdr_fm/tests/test_smoke.py -k "benchmark_variant or main_table" -q
```

Expected: fail on unknown new variants.

- [ ] **Step 3: Update validation**

In `train_aaai27_benchmark.py`, allowed variants become:

```python
allowed = {
    "smooth_dix",
    "sv_inv_net",
    "velocity_gan",
    "conditional_ddpm",
    "adapted_gfi",
    "adapted_auto_linear",
    "concat_fm",
    "cncs_fm",
    "bg_pdr_fm",
}
```

If old config loading is needed, retain `mm_invnet` and `two_stage_ddpm` only behind a clear `legacy` comment and do not include them in launch/eval method lists.

- [ ] **Step 4: Update evaluator checkpoint loading for Smooth-Dix**

In `_load_checkpoint`, allow no checkpoint when `variant == "smooth_dix"`:

```python
variant = str(_conf_get(conf, "benchmark.variant", "")).lower()
if variant == "smooth_dix" and (checkpoint_path is None or str(checkpoint_path).strip() == ""):
    return {"mode": "analytic", "path": ""}
```

- [ ] **Step 5: Run static checks**

Run:

```bash
python -m py_compile bg_pdr_fm/lightning/benchmark_module.py bg_pdr_fm/training/train_aaai27_benchmark.py bg_pdr_fm/evaluation/compare_experiments.py
```

Expected: pass.

## Task 8: Main-Table Evaluation Runner Updates

**Files:**
- Modify: `bg_pdr_fm/evaluation/run_aaai27_openfwi_eval.py`
- Modify: `bg_pdr_fm/evaluation/run_aaai27_marmousi_eval.py`
- Test: `bg_pdr_fm/tests/test_smoke.py`

- [ ] **Step 1: Write test for method list**

Add:

```python
def test_openfwi_main_table_method_list_uses_paper_faithful_baselines():
    from bg_pdr_fm.evaluation.run_aaai27_openfwi_eval import METHODS

    names = [item["name"] for item in METHODS]
    assert "mm_invnet" not in names
    assert "two_stage_ddpm" not in names
    assert names[:6] == [
        "smooth_dix",
        "sv_inv_net",
        "velocity_gan",
        "conditional_ddpm",
        "adapted_gfi",
        "adapted_auto_linear",
    ]
```

- [ ] **Step 2: Run test and verify failure**

Run:

```bash
pytest bg_pdr_fm/tests/test_smoke.py::test_openfwi_main_table_method_list_uses_paper_faithful_baselines -q
```

Expected: fail because old list includes `mm_invnet` and `two_stage_ddpm`.

- [ ] **Step 3: Update OpenFWI runner**

Set `METHODS` to:

```python
METHODS = [
    {"name": "smooth_dix", "config": "bg_pdr_fm/configs/experiments/aaai27/formal_smooth_dix.yaml", "checkpoint_hint": ""},
    {"name": "sv_inv_net", "config": "bg_pdr_fm/configs/experiments/aaai27/formal_sv_inv_net.yaml", "checkpoint_hint": "logs/bg_pdr_fm/aaai27/formal/sv_inv_net/checkpoints/last.ckpt"},
    {"name": "velocity_gan", "config": "bg_pdr_fm/configs/experiments/aaai27/formal_velocity_gan.yaml", "checkpoint_hint": "logs/bg_pdr_fm/aaai27/formal/velocity_gan/checkpoints/last.ckpt"},
    {"name": "conditional_ddpm", "config": "bg_pdr_fm/configs/experiments/aaai27/formal_conditional_ddpm.yaml", "checkpoint_hint": "logs/bg_pdr_fm/aaai27/formal/conditional_ddpm/checkpoints/last.ckpt"},
    {"name": "adapted_gfi", "config": "bg_pdr_fm/configs/experiments/aaai27/formal_adapted_gfi_multimodal.yaml", "checkpoint_hint": "logs/bg_pdr_fm/aaai27/formal/adapted_gfi_multimodal/checkpoints/last.ckpt"},
    {"name": "adapted_auto_linear", "config": "bg_pdr_fm/configs/experiments/aaai27/formal_adapted_auto_linear_multimodal.yaml", "checkpoint_hint": "logs/bg_pdr_fm/aaai27/formal/adapted_auto_linear_multimodal/checkpoints/last.ckpt"},
]
```

Keep Ours handled separately if the script already supports leaving PDR-FM blank.

- [ ] **Step 4: Update Marmousi runner**

Use the same non-PDR-FM method list unless the user explicitly asks to evaluate PDR-FM.

- [ ] **Step 5: Run tests**

Run:

```bash
pytest bg_pdr_fm/tests/test_smoke.py::test_openfwi_main_table_method_list_uses_paper_faithful_baselines -q
python -m py_compile bg_pdr_fm/evaluation/run_aaai27_openfwi_eval.py bg_pdr_fm/evaluation/run_aaai27_marmousi_eval.py
```

Expected: pass.

## Task 9: Documentation and TeX Table Framing

**Files:**
- Modify: `docs/paper/aaai27_method_registry.md`
- Modify: `docs/paper/aaai27_constructed_method_inventory.md`
- Modify: `docs/paper/aaai_LaTeX/seismic_diff_method_appendix.tex`

- [ ] **Step 1: Update method registry**

Replace old main-table rows with:

```markdown
| Smooth-Dix | Traditional/physics RMS-smooth/Dix-style baseline. | `formal_smooth_dix.yaml` | Pending updated eval. | Yes, as traditional reference. | No. | No learned parameters. |
| SV_Inv_Net | SVInvNet architecture adapted to current multimodal input. | `formal_sv_inv_net.yaml` | Pending training/evaluation. | Yes. | No. | Source: arXiv:2312.08194; adapted input protocol. |
| VelocityGAN | VelocityGAN-style conditional GAN adapted to current multimodal input. | `formal_velocity_gan.yaml` | Pending training/evaluation. | Yes. | No. | Generator params reported in main table. |
| Conditional DDPM | diffusers `UNet2DConditionModel` conditional DDPM adapted to current multimodal input. | `formal_conditional_ddpm.yaml` | Pending training/evaluation. | Yes. | No. | Replaces old two-stage DDPM scaffold. |
```

Mark `MM-InvNet` and `Two-stage DDPM` as legacy removed from main table.

- [ ] **Step 2: Update TeX caption**

In `seismic_diff_method_appendix.tex`, update the main table caption to state:

```tex
SV\_Inv\_Net, VelocityGAN, Conditional DDPM, GFI, and Auto-Linear are architecture-level adaptations to the same multimodal input protocol; official native reproductions are not claimed.
```

Remove `Trainable Params` if present. Keep columns:

```text
MAE / RMSE / SSIM / MAE_L / MAE_H / Params
```

- [ ] **Step 3: Do not overwrite old numeric rows with fake results**

Replace old `MM-InvNet` and `Two-stage DDPM` rows with placeholders only if no new evaluation exists:

```tex
SV\_Inv\_Net & -- & -- & -- & -- & -- & -- \\
VelocityGAN & -- & -- & -- & -- & -- & -- \\
Conditional DDPM & -- & -- & -- & -- & -- & -- \\
```

Do not invent numbers.

- [ ] **Step 4: Check docs diff**

Run:

```bash
git diff -- docs/paper/aaai27_method_registry.md docs/paper/aaai27_constructed_method_inventory.md docs/paper/aaai_LaTeX/seismic_diff_method_appendix.tex
```

Expected: only provenance/table framing changes.

## Task 10: Smoke Training and Evaluation Protocol

**Files:**
- No new files unless a smoke helper already exists and is appropriate.

- [ ] **Step 1: Static compile**

Run:

```bash
python -m py_compile \
  bg_pdr_fm/external_baselines/common.py \
  bg_pdr_fm/external_baselines/smooth_dix.py \
  bg_pdr_fm/external_baselines/sv_inv_net.py \
  bg_pdr_fm/external_baselines/conditional_ddpm.py \
  bg_pdr_fm/external_baselines/velocity_gan.py \
  bg_pdr_fm/external_baselines/auto_linear/adapted_multimodal.py \
  bg_pdr_fm/lightning/benchmark_module.py \
  bg_pdr_fm/training/train_aaai27_benchmark.py \
  bg_pdr_fm/evaluation/compare_experiments.py
```

Expected: no output and exit code 0.

- [ ] **Step 2: Run focused unit tests**

Run:

```bash
pytest bg_pdr_fm/tests/test_smoke.py -k "smooth_dix or sv_inv_net or conditional_ddpm or velocity_gan or auto_linear or main_table_method_list or multimodal_condition_image" -q
```

Expected: pass.

- [ ] **Step 3: Fast-run learned baselines**

Run each:

```bash
CUDA_VISIBLE_DEVICES=7 python -m bg_pdr_fm.training.train_aaai27_benchmark --config bg_pdr_fm/configs/experiments/aaai27/sv_inv_net.yaml --fast-run
CUDA_VISIBLE_DEVICES=7 python -m bg_pdr_fm.training.train_aaai27_benchmark --config bg_pdr_fm/configs/experiments/aaai27/velocity_gan.yaml --fast-run
CUDA_VISIBLE_DEVICES=7 python -m bg_pdr_fm.training.train_aaai27_benchmark --config bg_pdr_fm/configs/experiments/aaai27/conditional_ddpm.yaml --fast-run
CUDA_VISIBLE_DEVICES=7 python -m bg_pdr_fm.training.train_aaai27_benchmark --config bg_pdr_fm/configs/experiments/aaai27/adapted_auto_linear_ae_pretrain.yaml --fast-run
```

Expected: each completes one small train/val pass with finite loss.

- [ ] **Step 4: One-batch evaluation smoke**

For each formal config, temporarily override or create smoke configs with:

```yaml
evaluation:
  max_batches: 1
  missing_modes: [full]
```

Run:

```bash
CUDA_VISIBLE_DEVICES=7 python -m bg_pdr_fm.evaluation.compare_experiments --config bg_pdr_fm/configs/experiments/aaai27/formal_smooth_dix.yaml
```

For learned baselines, run this only after a checkpoint exists.

Expected: `summary.json`, `metrics.csv`, `dataset_summary.csv` generated with finite metrics.

## Task 11: Git Hygiene

**Files:**
- All files touched by this implementation only.

- [ ] **Step 1: Inspect dirty worktree**

Run:

```bash
git status --short
```

Expected: there may be unrelated existing dirty files under paper images and internal experiment diagnostics. Do not stage them.

- [ ] **Step 2: Stage only implementation files**

Run:

```bash
git add \
  bg_pdr_fm/external_baselines/common.py \
  bg_pdr_fm/external_baselines/smooth_dix.py \
  bg_pdr_fm/external_baselines/sv_inv_net.py \
  bg_pdr_fm/external_baselines/conditional_ddpm.py \
  bg_pdr_fm/external_baselines/velocity_gan.py \
  bg_pdr_fm/external_baselines/auto_linear/adapted_multimodal.py \
  bg_pdr_fm/lightning/benchmark_module.py \
  bg_pdr_fm/training/train_aaai27_benchmark.py \
  bg_pdr_fm/evaluation/compare_experiments.py \
  bg_pdr_fm/evaluation/run_aaai27_openfwi_eval.py \
  bg_pdr_fm/evaluation/run_aaai27_marmousi_eval.py \
  bg_pdr_fm/configs/experiments/aaai27/formal_smooth_dix.yaml \
  bg_pdr_fm/configs/experiments/aaai27/sv_inv_net.yaml \
  bg_pdr_fm/configs/experiments/aaai27/formal_sv_inv_net.yaml \
  bg_pdr_fm/configs/experiments/aaai27/velocity_gan.yaml \
  bg_pdr_fm/configs/experiments/aaai27/formal_velocity_gan.yaml \
  bg_pdr_fm/configs/experiments/aaai27/conditional_ddpm.yaml \
  bg_pdr_fm/configs/experiments/aaai27/formal_conditional_ddpm.yaml \
  bg_pdr_fm/configs/experiments/aaai27/adapted_auto_linear_ae_pretrain.yaml \
  bg_pdr_fm/configs/experiments/aaai27/adapted_auto_linear_multimodal.yaml \
  bg_pdr_fm/configs/experiments/aaai27/formal_adapted_auto_linear_multimodal.yaml \
  bg_pdr_fm/tests/test_smoke.py \
  docs/paper/aaai27_method_registry.md \
  docs/paper/aaai27_constructed_method_inventory.md \
  docs/paper/aaai_LaTeX/seismic_diff_method_appendix.tex
```

- [ ] **Step 3: Verify staged diff**

Run:

```bash
git diff --cached --stat
git diff --cached --check
```

Expected: no whitespace errors; staged files match this implementation scope.

- [ ] **Step 4: Commit**

Run:

```bash
git commit -m "feat: add paper-faithful adapted AAAI27 baselines"
```

Expected: commit succeeds.

## Execution Order

1. Task 1 shared adapter.
2. Task 2 Smooth-Dix.
3. Task 3 SV_Inv_Net.
4. Task 4 Conditional DDPM.
5. Task 5 VelocityGAN.
6. Task 6 Auto-Linear rewrite.
7. Task 7 registry/evaluator compatibility.
8. Task 8 runners.
9. Task 9 docs/TeX.
10. Task 10 verification.
11. Task 11 selective commit.

Do not launch full 200-epoch training until Tasks 1-10 pass.

## Self-Review

- Spec coverage: Smooth-Dix, SV_Inv_Net, Conditional DDPM, VelocityGAN, Auto-Linear, parameter reporting, evaluation runners, and docs are covered.
- Placeholder scan: no `TBD` or invented metric values. Table rows without results must use `--`.
- Type consistency: new variants use snake-case identifiers in configs and benchmark code; paper display names can use formatted names.
