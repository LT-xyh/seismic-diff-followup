# OpenFWI LMDB 打包说明

这份文档用于在服务器上把 OpenFWI 的 `.npy` 小文件数据打包成 LMDB。当前代码只对 OpenFWI 使用 LMDB；Marmousi 继续使用 file-per-sample 的 `.npy` 布局。

## 1. 目标格式

每个 OpenFWI 子集单独打包成一个 LMDB 目录：

```text
<lmdb-root>/
  FlatVelA.lmdb/
    data.mdb
    lock.mdb
    manifest.json
  FlatVelB.lmdb/
    data.mdb
    lock.mdb
    manifest.json
  CurveVelA.lmdb/
    data.mdb
    lock.mdb
    manifest.json
```

默认 LMDB 根目录建议为：

```text
I:/Datasets/openfwi_lmdb
```

在 Linux 服务器上可换成类似：

```text
/data/Datasets/openfwi_lmdb
/mnt/i/Datasets/openfwi_lmdb
```

当前实现是 map-style Dataset，训练时仍然通过 PyTorch `DataLoader(..., shuffle=True)` 按样本打乱读取。多个子集会先在 Dataset 内拼成一个 index 列表，再由 DataLoader 对全局样本 index 打乱。

## 2. 源数据目录

LMDB 打包脚本要求源目录采用下面的结构：

```text
<source-root>/
  FlatVelA/
    depth_vel/
      0.npy
      1.npy
      ...
    migrated_image/
      0.npy
      1.npy
      ...
    horizon/
      0.npy
      1.npy
      ...
    rms_vel/
      0.npy
      1.npy
      ...
  FlatVelB/
    depth_vel/
    migrated_image/
    horizon/
    rms_vel/
```

默认需要的四个源模态是：

```text
depth_vel
migrated_image
horizon
rms_vel
```

不打包也不读取：

```text
well_log
well_mask
time_vel
shot_data
```

`well_log` 在训练时由 raw `depth_vel` 随机抽取速度列动态生成；`well_mask` 同时动态生成，用于标记真实观测到的井位置。

## 3. 单样本内容与 Shape

LMDB 中保存的是原始 `.npy` 文件字节，不做归一化、不做 resize、不做 dtype 转换。因此 LMDB 后端和 `.npy` 后端读取到的 raw 数值应完全一致。

推荐每个 `.npy` 文件为：

| 模态 | 含义 | 推荐 shape | 可接受 shape | 推荐 dtype | 数值含义 |
| --- | --- | --- | --- | --- | --- |
| `depth_vel` | 深度域真实速度 | `[1, 70, 70]` | `[70, 70]` 或 `[1, H, W]` | `float32` | m/s，OpenFWI profile 默认 `[1500, 4500]` |
| `migrated_image` | 偏移/成像结果 | `[1, 1000, 70]` | `[70, 70]` 或 `[1, H, W]` | `float32` | 振幅，OpenFWI profile 默认 `[-200, 200]` |
| `horizon` | 层位/界面 mask | `[1, 70, 70]` | `[70, 70]` 或 `[1, H, W]` | `float32`、`int32` 或 `int64` | 离散 0/1，不做连续归一化 |
| `rms_vel` | RMS 速度 | `[1, 1000, 70]` | `[70, 70]` 或 `[1, H, W]` | `float32` 或 `float64` | m/s，OpenFWI profile 默认 `[1500, 4500]` |

训练入口默认会把样本标准化为：

```text
depth_vel:       [B, 1, 70, 70]
migrated_image:  [B, 1, 70, 70]
horizon:         [B, 1, 70, 70]
rms_vel:         [B, 1, 70, 70]
well_log:        [B, 1, 70, 70]
well_mask:       [B, 1, 70, 70]
```

BG-PDR-FM 额外返回：

```text
modality_mask:    [B, 4]
modality_quality: [B, 4]
```

注意：LMDB 只保存 raw source array。归一化发生在 Dataset 读取阶段，当前 OpenFWI profile 为：

```text
depth_vel / rms_vel / well_log: [1500, 4500] -> [-1, 1] 后 clamp
migrated_image:                 [-200, 200] -> [-1, 1] 后 clamp
horizon / well_mask:            保持 0/1
```

## 4. LMDB Key/Value 约定

每个样本每个模态一条 key：

```text
{sample_index:08d}:{modality}
```

示例：

```text
00000000:depth_vel
00000000:migrated_image
00000000:horizon
00000000:rms_vel
00000001:depth_vel
```

value 是对应 `.npy` 文件的原始 bytes，可直接用 `np.load(io.BytesIO(value), allow_pickle=False)` 读回。

LMDB 内还会写入一个特殊 key：

```text
__manifest__
```

同时 LMDB 目录下也会生成一份便于人工检查的：

```text
manifest.json
```

`manifest.json` 示例：

```json
{
  "format": "openfwi_lmdb",
  "version": 1,
  "dataset_name": "FlatVelA",
  "sample_count": 27000,
  "modalities": ["depth_vel", "migrated_image", "horizon", "rms_vel"],
  "modality_meta": {
    "depth_vel": {"shape": [1, 70, 70], "dtype": "float32"},
    "migrated_image": {"shape": [1, 1000, 70], "dtype": "float32"},
    "horizon": {"shape": [1, 70, 70], "dtype": "int32"},
    "rms_vel": {"shape": [1, 1000, 70], "dtype": "float64"}
  },
  "source_root": "/abs/path/to/openfwi/FlatVelA",
  "normalization_profile": "openfwi",
  "key_format": "{sample_index:08d}:{modality}",
  "stores_well_log": false
}
```

`sample_count` 由四个模态目录的 `.npy` 文件数量决定。四个目录数量必须一致，否则脚本会报错。

## 5. 打包命令

在仓库根目录运行。先确认服务器环境里已经安装：

```bash
python -c "import lmdb; print(lmdb.__version__)"
```

Linux 服务器示例：

```bash
python tools/build_openfwi_lmdb.py \
  --source-root /data/Datasets/openfwi \
  --lmdb-root /data/Datasets/openfwi_lmdb \
  --datasets FlatVelA FlatVelB CurveVelA CurveVelB CurveFaultA \
  --overwrite
```

Windows / PowerShell 示例：

```powershell
python tools\build_openfwi_lmdb.py `
  --source-root I:\Datasets\openfwi `
  --lmdb-root I:\Datasets\openfwi_lmdb `
  --datasets FlatVelA FlatVelB CurveVelA CurveVelB CurveFaultA `
  --overwrite
```

可选参数：

```text
--datasets              只打包指定子集
--modalities            默认 depth_vel migrated_image horizon rms_vel，一般不要改
--normalization-profile 默认 openfwi，只写入 manifest，不改变 raw 数据
--map-size-gb           手动指定每个子集 LMDB 的 map_size
--commit-interval       每多少个 sample 提交一次写事务，默认 512
--overwrite             目标 LMDB 已存在时先删除再重建
```

如果遇到 `MDB_MAP_FULL`，说明 map size 不够，增大 `--map-size-gb` 后用 `--overwrite` 重打：

```bash
python tools/build_openfwi_lmdb.py \
  --source-root /data/Datasets/openfwi \
  --lmdb-root /data/Datasets/openfwi_lmdb \
  --datasets FlatVelA \
  --map-size-gb 8 \
  --overwrite
```

## 6. 添加新的 OpenFWI 子集

新增子集不需要合并进旧 LMDB。每个子集仍然单独一个 LMDB，例如：

```text
/data/Datasets/openfwi/NewSubset/depth_vel/*.npy
/data/Datasets/openfwi/NewSubset/migrated_image/*.npy
/data/Datasets/openfwi/NewSubset/horizon/*.npy
/data/Datasets/openfwi/NewSubset/rms_vel/*.npy
```

只打包新增子集：

```bash
python tools/build_openfwi_lmdb.py \
  --source-root /data/Datasets/openfwi \
  --lmdb-root /data/Datasets/openfwi_lmdb \
  --datasets NewSubset \
  --overwrite
```

然后在训练配置里加入：

```yaml
data:
  openfwi_datasets:
    - FlatVelA
    - FlatVelB
    - CurveVelA
    - CurveVelB
    - CurveFaultA
    - NewSubset
```

保持每个子集一个 LMDB 的原因：

1. 新增/重建某个子集时不需要重写所有数据。
2. manifest 能清楚记录每个子集自己的 shape、dtype、sample count 和 source root。
3. PyTorch 训练时仍然是全局 sample-level shuffle，不会变成“先读完一个子集再读另一个子集”。

## 7. 验证打包结果

检查 manifest：

```bash
python - <<'PY'
import json
from pathlib import Path

root = Path("/data/Datasets/openfwi_lmdb")
for path in sorted(root.glob("*.lmdb/manifest.json")):
    manifest = json.loads(path.read_text())
    print(path)
    print("  dataset:", manifest["dataset_name"])
    print("  samples:", manifest["sample_count"])
    print("  modalities:", manifest["modalities"])
    print("  meta:", manifest["modality_meta"])
PY
```

比较 `.npy` 与 LMDB raw 数值是否完全一致：

```bash
python - <<'PY'
from pathlib import Path
import numpy as np

from bg_pdr_fm.data.openfwi_lmdb import OpenFWILMDBReader, openfwi_lmdb_path

source_root = Path("/data/Datasets/openfwi")
lmdb_root = Path("/data/Datasets/openfwi_lmdb")
dataset = "FlatVelA"
idx = 0

reader = OpenFWILMDBReader(openfwi_lmdb_path(lmdb_root, dataset))
for modality in ("depth_vel", "migrated_image", "horizon", "rms_vel"):
    raw_lmdb = reader.get(idx, modality)
    raw_npy = np.load(source_root / dataset / modality / f"{idx}.npy", allow_pickle=False)
    print(modality, raw_lmdb.shape, raw_lmdb.dtype, np.array_equal(raw_lmdb, raw_npy))
PY
```

检查 Dataset 能否使用 LMDB：

```bash
python - <<'PY'
from bg_pdr_fm.data.raw_datasets import DEFAULT_OPENFWI_SCHEMA, OpenFWI

ds = OpenFWI(
    root_dir="/data/Datasets/openfwi",
    lmdb_root="/data/Datasets/openfwi_lmdb",
    datasets=("FlatVelA",),
    use_data=DEFAULT_OPENFWI_SCHEMA,
    storage_backend="lmdb",
    use_normalize="-1_1",
    well_random=False,
)

item = ds[0]
print("backend:", ds.storage_backend)
for key, value in item.items():
    print(key, tuple(value.shape), value.dtype, float(value.min()), float(value.max()))
print("well observed pixels:", int(item["well_mask"].sum().item()))
PY
```

跑 IO benchmark：

```bash
python tools/benchmark_dataset_io.py \
  --root-dir /data/Datasets/openfwi \
  --lmdb-root /data/Datasets/openfwi_lmdb \
  --datasets FlatVelA \
  --backend lmdb \
  --batch-size 32 \
  --num-workers 4 \
  --batches 50
```

也可以对比 `.npy`：

```bash
python tools/benchmark_dataset_io.py \
  --root-dir /data/Datasets/openfwi \
  --lmdb-root /data/Datasets/openfwi_lmdb \
  --datasets FlatVelA \
  --backend npy \
  --batch-size 32 \
  --num-workers 4 \
  --batches 50
```

## 8. 训练时如何使用

OpenFWI 普通 Dataset：

```python
from bg_pdr_fm.data.raw_datasets import OpenFWI, DEFAULT_OPENFWI_SCHEMA

dataset = OpenFWI(
    root_dir="/data/Datasets/openfwi",
    lmdb_root="/data/Datasets/openfwi_lmdb",
    datasets=("FlatVelA", "FlatVelB"),
    use_data=DEFAULT_OPENFWI_SCHEMA,
    storage_backend="auto",
)
```

BG-PDR-FM 配置：

```yaml
data:
  name: openfwi
  root_dir: /data/Datasets/openfwi
  storage_backend: auto
  lmdb_root: /data/Datasets/openfwi_lmdb
  normalization_profile: auto
  normalize_clamp: true
  target_shape: [70, 70]
  well_count_range: [0, 3]
  well_seed: 1234
  openfwi_datasets:
    - FlatVelA
    - FlatVelB
    - CurveVelA
    - CurveVelB
    - CurveFaultA

training:
  num_workers: 4
  persistent_workers: true
  prefetch_factor: 3
  pin_memory: true
```

`storage_backend: auto` 的行为：

1. 如果指定子集对应的 `<lmdb-root>/<SubsetName>.lmdb/manifest.json` 都存在，并且 Python 能 import `lmdb`，则使用 LMDB。
2. 否则回退到 `.npy` 小文件读取。

如果想强制使用 LMDB，把配置改成：

```yaml
data:
  storage_backend: lmdb
```

此时缺少任何子集的 LMDB 都会直接报错，便于发现路径或打包遗漏。

## 9. 常见错误

`ModuleNotFoundError: No module named 'lmdb'`

安装 LMDB：

```bash
pip install lmdb
```

`Mismatched OpenFWI modality lengths`

四个模态目录下 `.npy` 文件数量不一致。需要补齐或删除不完整样本。

`Missing modality directory`

源目录缺少 `depth_vel/migrated_image/horizon/rms_vel` 中某个目录。当前默认 LMDB 不接受只含部分模态的源数据。

`FileExistsError: LMDB path already exists and is not empty`

目标 LMDB 已存在。确认要重建后加 `--overwrite`。

`MDB_MAP_FULL`

LMDB map size 不够。加大 `--map-size-gb` 并重建。

Dataset 打印 `backend: npy`

说明 `storage_backend="auto"` 没有找到完整 LMDB。检查：

```text
lmdb_root 是否正确
<lmdb-root>/<SubsetName>.lmdb/manifest.json 是否存在
config 里的 openfwi_datasets 名字是否和 LMDB 目录名完全一致
```
