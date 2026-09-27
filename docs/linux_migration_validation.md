# Linux 服务器迁移后校正文档

这份文档用于把本仓库从 Windows 工作站迁移到 Linux 服务器后做路径、环境、数据、LMDB、训练入口和评估入口的校正验收。目标不是重新设计代码，而是确认迁移后的 Linux 环境和本地开发环境遵守同一套 BG-PDR-FM 数据与运行契约。

## 0. 迁移目标状态

迁移完成后应满足：

- 代码从 Git 获取，服务器上以 `main` 分支为主线。
- `bg_pdr_fm/` 可以独立运行，不依赖把旧实验目录加入 `PYTHONPATH`。
- OpenFWI 源数据和 OpenFWI LMDB 放在服务器数据盘，不放进 Git。
- Marmousi 保持 file-per-sample 的 `.npy` 布局。
- `well_log` 不存盘，运行时由 raw `depth_vel` 动态生成。
- 默认 batch schema 固定包含：

```text
depth_vel, migrated_image, horizon, rms_vel, well_log, well_mask,
modality_mask, modality_quality
```

## 1. 路径校正

建议服务器路径：

```text
/data/seismic/openfwi
/data/seismic/openfwi_lmdb
/data/seismic/marmousi
/data/seismic/checkpoints/bg_pdr_fm
```

仓库内配置需要把 Windows 路径改成 Linux 路径。重点检查：

```yaml
data:
  root_dir: /data/seismic/openfwi
  lmdb_root: /data/seismic/openfwi_lmdb

model:
  codec_checkpoint: /data/seismic/checkpoints/bg_pdr_fm/seismic_autoencoder_kl.ckpt
```

Marmousi 配置：

```yaml
data:
  root_dir: /data/seismic/marmousi
```

必须清除或替换这些 Windows 路径形态：

```text
D:/...
I:/...
D:\...
I:\...
```

校正命令：

```bash
grep -RIn "D:/\|I:/\|D:\\\\\|I:\\\\" bg_pdr_fm docs tools || true
```

如果还有命中，先判断是否只是说明文字；配置里的真实路径必须改掉。

## 2. 环境校正

不要直接复用 Windows 的 conda 环境。服务器上重新建 Linux 环境：

```bash
conda create -n seismic python=3.10 -y
conda activate seismic
```

先确认 GPU 和驱动：

```bash
nvidia-smi
```

然后按服务器 CUDA/驱动安装匹配的 PyTorch。安装完至少确认：

```bash
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
print("cuda version", torch.version.cuda)
PY
```

基础依赖检查：

```bash
python - <<'PY'
mods = [
    "lightning",
    "omegaconf",
    "diffusers",
    "lmdb",
    "matplotlib",
    "numpy",
    "torchmetrics",
    "ignite",
]
for name in mods:
    __import__(name)
    print("OK", name)
PY
```

如果 `ignite` 失败，通常需要安装 `pytorch-ignite`。

## 3. 代码独立性校正

在仓库根目录运行：

```bash
python - <<'PY'
import importlib
mods = [
    "bg_pdr_fm",
    "bg_pdr_fm.data",
    "bg_pdr_fm.models",
    "bg_pdr_fm.lightning.bg_pdr_fm_module",
    "bg_pdr_fm.training.train_bg_pdr_fm",
    "bg_pdr_fm.evaluation.evaluate_bg_pdr_fm",
    "bg_pdr_fm.training.train_autoencoder",
    "bg_pdr_fm.evaluation.evaluate_autoencoder",
]
for name in mods:
    importlib.import_module(name)
    print("OK", name)
PY
```

期望：全部输出 `OK`。

再做编译检查：

```bash
python - <<'PY'
import pathlib
import py_compile
files = list(pathlib.Path("bg_pdr_fm").rglob("*.py"))
for path in files:
    py_compile.compile(str(path), doraise=True)
print("compiled", len(files), "files")
PY
```

期望：无异常。

## 4. 数据结构校正

OpenFWI 源目录每个子集应至少包含：

```text
FlatVelA/
  depth_vel/
  migrated_image/
  horizon/
  rms_vel/
```

不应要求存在：

```text
well_log/
time_vel/
shot_data/
```

LMDB 目录每个子集一个：

```text
/data/seismic/openfwi_lmdb/FlatVelA.lmdb/manifest.json
/data/seismic/openfwi_lmdb/CurveVelA.lmdb/manifest.json
```

校正命令：

```bash
find /data/seismic/openfwi_lmdb -maxdepth 2 -name manifest.json -print
```

如果新增 OpenFWI 子集，只新增对应的 `<SubsetName>.lmdb`，不要和旧子集合并成一个大 LMDB。

## 5. Dataset schema 校正

先用 synthetic 验证 batch 契约：

```bash
python - <<'PY'
from torch.utils.data import DataLoader
from bg_pdr_fm.data import SyntheticBGDataset, collate_bg_samples, validate_bg_batch

loader = DataLoader(SyntheticBGDataset(length=2), batch_size=2, collate_fn=collate_bg_samples)
batch = next(iter(loader))
validate_bg_batch(batch, context="linux synthetic schema check")
for key, value in batch.__dict__.items():
    if hasattr(value, "shape"):
        print(key, tuple(value.shape), value.dtype)
PY
```

期望：

```text
depth_vel / horizon / well_log / well_mask: B x 1 x 70 x 70
modality_mask / modality_quality: B x 4
```

真实 OpenFWI/LMDB 检查：

```bash
python - <<'PY'
from bg_pdr_fm.data import OpenFWIBGDataset

dataset = OpenFWIBGDataset(
    root_dir="/data/seismic/openfwi",
    lmdb_root="/data/seismic/openfwi_lmdb",
    datasets=["FlatVelA"],
    split="train",
    storage_backend="auto",
    well_random=False,
)
item = dataset[0]
for key, value in item.items():
    if hasattr(value, "shape"):
        print(key, tuple(value.shape), value.dtype)
print("well observed pixels", int(item["well_mask"].sum().item()))
if hasattr(dataset, "close"):
    dataset.close()
PY
```

必须确认：

- `well_log.shape == depth_vel.shape`
- `well_mask.shape == depth_vel.shape`
- `well_log[well_mask == 0]` 全为 0
- `modality_mask` 表示模态存在性
- `well_mask` 表示井空间观测位置，二者不混用

## 6. Fast-run 校正

先跑不依赖真实数据的三阶段 smoke：

```bash
python -m bg_pdr_fm.training.train_bg_pdr_fm --config bg_pdr_fm/configs/fast_run.yaml
```

期望：contrastive、background、residual 三个 stage 都完成 1 个 batch。

再跑 synthetic evaluation：

```bash
python -m bg_pdr_fm.evaluation.evaluate_bg_pdr_fm --config bg_pdr_fm/configs/fast_run.yaml
```

期望：无 checkpoint 缺失错误，因为 `fast_run.yaml` 中 `evaluation.allow_untrained: true`。

Autoencoder fast-run：

```bash
python -m bg_pdr_fm.training.train_autoencoder bg_pdr_fm/configs/autoencoder_fast_run.yaml
```

期望：完成 1 个 train batch 和 1 个 val batch。

## 7. 真实数据小批量校正

OpenFWI LMDB 小批量训练前，建议临时把配置改成：

```yaml
training:
  max_epochs: 1
  batch_size: 2
  num_workers: 4
  limit_train_batches: 2
  limit_val_batches: 1
```

运行：

```bash
python -m bg_pdr_fm.training.train_bg_pdr_fm --config bg_pdr_fm/configs/openfwi_lmdb_train.yaml
```

Marmousi 小批量：

```bash
python -m bg_pdr_fm.training.train_bg_pdr_fm --config bg_pdr_fm/configs/marmousi_train.yaml
```

如果服务器 GPU 正常，正式训练再把：

```yaml
training:
  accelerator: gpu
  devices: 1
  num_workers: 4
  persistent_workers: true
  prefetch_factor: 3
  pin_memory: true
```

## 8. 性能校正

OpenFWI 优先使用 LMDB：

```yaml
data:
  storage_backend: auto
```

如果想强制检查 LMDB 是否齐全：

```yaml
data:
  storage_backend: lmdb
```

若强制 LMDB 后报缺失，说明 `/data/seismic/openfwi_lmdb/<SubsetName>.lmdb/manifest.json` 或对应子集没有准备好。

DataLoader 建议起点：

```yaml
training:
  num_workers: 4
  persistent_workers: true
  prefetch_factor: 3
  pin_memory: true
```

服务器稳定后可试 `num_workers: 8`。如果 CPU 占用过高、batch wait time 变长，退回 4。

## 9. 验收清单

- [ ] `git branch --show-current` 输出 `main`
- [ ] `git status --short` 无意外改动
- [ ] Linux conda 环境可 import `torch/lightning/lmdb/diffusers`
- [ ] `torch.cuda.is_available()` 为预期值
- [ ] 配置中无 Windows 绝对路径
- [ ] `bg_pdr_fm` 核心模块可独立 import
- [ ] `bg_pdr_fm` 全部 `.py` 可 `py_compile`
- [ ] OpenFWI LMDB 每个子集都有 `manifest.json`
- [ ] OpenFWI/Marmousi 不依赖本地 `well_log/*.npy`
- [ ] synthetic BG-PDR-FM fast-run 通过
- [ ] synthetic BG-PDR-FM evaluation 通过
- [ ] autoencoder fast-run 通过
- [ ] OpenFWI LMDB 小批量训练通过
- [ ] Marmousi 小批量训练通过
- [ ] 日志输出目录和 checkpoint 输出目录可写

## 10. 常见问题

### `ModuleNotFoundError: No module named 'bg_pdr_fm'`

没有在仓库根目录运行，或者没有用模块方式启动。切到仓库根目录后运行：

```bash
python -m bg_pdr_fm.training.train_bg_pdr_fm --config bg_pdr_fm/configs/fast_run.yaml
```

### `ModuleNotFoundError: No module named 'lmdb'`

安装：

```bash
pip install lmdb
```

### `Missing OpenFWI LMDB datasets`

检查：

- `data.lmdb_root` 是否是 Linux 绝对路径
- 子集名是否和 LMDB 目录名一致
- 是否存在 `<SubsetName>.lmdb/manifest.json`

### GPU 可见但训练没用 GPU

检查配置：

```yaml
training:
  accelerator: gpu
  devices: 1
```

同时确认 PyTorch CUDA 可用：

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
```

### `well_log` 文件夹不存在

这是预期行为。当前实现不读取 stored well log，井约束由 `depth_vel` 动态抽列生成，并由 `well_mask` 标记观测位置。
