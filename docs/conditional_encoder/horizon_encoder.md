# Horizon Encoder

> Legacy note: this document records the old `src/` horizon-encoder design.
> The local `src/` tree has been removed from the active project; current
> BG-PDR-FM implementations live under `bg_pdr_fm/`.

## 1. 设计目标

`HorizonEncoderA` 不再把 horizon mask 当作普通稠密图像处理，而是把它视为一组稀疏、细薄、具有地层边界物理意义的界面信号。

这次重构的核心目标是：

- 支持变尺寸输入 `x: [B, 1, H, W]`
- 去除对 `70 x 70` 的硬编码依赖
- 优先保留 horizon 的细界面定位能力
- 同时建立跨全图的结构连续性建模能力
- 输出统一的 2D 特征图，供后续多模态融合使用

输出接口为：

```python
y: [B, C_out, H_out, W_out]
```

其中 `H_out, W_out` 由 `out_hw` 控制，默认是 `(70, 70)`。

## 2. 整体思路

新编码器采用一条明确的“稀疏界面建模”路径，而不是简单堆叠标准 CNN 残差块：

1. `InterfaceEnhancer`
   先把原始二值 horizon mask 转成更稳定、更可学习的 sparse-band 表示。
2. `InterfacePatchStem + LocalInterfaceBlock`
   在原始分辨率上提取局部 horizon fragment，尽量保留细界面位置。
3. `SparseBandTokenizer`
   通过 band-weighted pooling，把局部有效区域汇聚成紧凑 token grid。
4. `GlobalTokenEncoder`
   用轻量 Transformer 在 token 之间建模长程结构连续性。
5. `ProjectionHead`
   把局部特征和全局上下文重新投影回统一 2D feature map。


## 3. 模块分解

### 3.1 Interface Enhancement Stage

模块：

- `FixedGaussianBlur2d`
- `FixedSobelMagnitude2d`
- `InterfaceEnhancer`

这一阶段不依赖可学习参数的复杂稀疏索引，而是用稳定、确定性的局部算子把细线 horizon 扩展成更可学习的界面带。

构造出的增强特征为：

```text
[mask, soft_band, wide_band, local_density, grad_mag, y, x]
```

含义：

- `mask`：原始二值 horizon
- `soft_band`：高斯模糊后的软界面带
- `wide_band`：进一步扩展的带状激活
- `local_density`：局部 horizon 活跃程度
- `grad_mag`：界面边缘强度
- `y, x`：归一化空间坐标

此外还构造了一个 `guide`：

```text
guide = wide_band + 0.5 * grad_mag + 0.25 * local_density
```

它用于后续引导局部编码和 token 提取，让网络把注意力更多放在界面附近。

对应代码：

- [HorizonEncoder_70x70.py](/d:/Research/seismic-diff/src/models/conditional_encoder/HorizonEncoder_70x70.py:163)

### 3.2 Local Structure Encoder

模块：

- `InterfacePatchStem`
- `LocalInterfaceBlock`

这里的设计重点不是“做更深的标准卷积”，而是显式区分不同类型的局部结构：

- `thin_branch`：强调像素级细界面
- `lateral_branch`：强调沿 horizon 方向的横向连续性
- `fragment_branch`：强调局部带状 fragment

`LocalInterfaceBlock` 内部也沿这个思路组织：

- `(5, 1)` 分支偏向薄层垂向支撑
- `(1, 7)` 分支偏向横向连续
- `(3, 5)` 分支负责局部片段混合

并且每个 block 都由 `guide` 做显式门控，不再把所有像素看作同等重要。

对应代码：

- [HorizonEncoder_70x70.py](/d:/Research/seismic-diff/src/models/conditional_encoder/HorizonEncoder_70x70.py:194)
- [HorizonEncoder_70x70.py](/d:/Research/seismic-diff/src/models/conditional_encoder/HorizonEncoder_70x70.py:227)

### 3.3 Sparse Region Extraction / Tokenization

模块：

- `SparseBandTokenizer`

这一层是本次重构和旧版 CNN 最大的区别之一。

旧版做法本质上还是在整张 2D 图上密集卷积。新版先利用 `guide` 作为带状权重，对局部特征做加权池化，把原始稀疏界面信息压缩成一个小的 token grid。

默认 token 分辨率：

```python
token_hw = (14, 14)
```

token 化时会汇聚：

- 局部特征 `local_map`
- mask 占据率
- guide 强度
- 坐标信息

这样每个 token 不只是一个普通 patch embedding，而是一个“界面带摘要”。

对应代码：

- [HorizonEncoder_70x70.py](/d:/Research/seismic-diff/src/models/conditional_encoder/HorizonEncoder_70x70.py:269)

### 3.4 Global Context Encoder

模块：

- `GlobalTokenEncoder`

在 token grid 上使用轻量 `TransformerEncoder` 做全局建模。每个 token 的输入由三部分组成：

- 局部稀疏带表示
- `guide` 的池化结果
- token 对应的二维坐标

这一步主要负责建模：

- horizon 片段之间的延续关系
- 全图尺度的结构组织
- 局部缺失或断裂情况下的上下文补偿

相比直接在高分辨率图上做自注意力，这种做法计算量更稳，也更符合“先聚焦活跃界面，再建模全局结构”的思路。

对应代码：

- [HorizonEncoder_70x70.py](/d:/Research/seismic-diff/src/models/conditional_encoder/HorizonEncoder_70x70.py:300)

### 3.5 2D Projection Head

模块：

- `ProjectionHead`

最后将两条信息流融合：

- 原始分辨率的 `local_map`
- token 级全局上下文 `token_map`

再辅以 `guide_pack`：

```text
[mask, soft_band, guide]
```

投影回统一的 2D 输出特征图，并通过 `out_hw` 控制输出大小。

对应代码：

- [HorizonEncoder_70x70.py](/d:/Research/seismic-diff/src/models/conditional_encoder/HorizonEncoder_70x70.py:339)

## 4. 前向传播中的张量形状

设输入为：

```python
x: [B, 1, H, W]
```

主要张量变化如下：

1. 增强阶段

```python
enhanced: [B, 7, H, W]
guide:    [B, 1, H, W]
```

2. 局部编码阶段

```python
local_map: [B, c1, H, W]
```

3. 稀疏 token 提取阶段

```python
token_grid: [B, c2, Gh, Gw]
guide_grid: [B, 1, Gh, Gw]
coord_grid: [B, 2, Gh, Gw]
```

其中 `(Gh, Gw) = token_hw`

4. 全局上下文编码后

```python
token_grid: [B, c2, Gh, Gw]
```

5. 投影输出

```python
y: [B, C_out, H_out, W_out]
```

其中 `(H_out, W_out) = out_hw`

## 5. 可配置项

`HorizonEncoderA` 当前主要参数：

```python
HorizonEncoderA(
    C_out=16,
    c1=32,
    c2=48,
    c3=64,
    k_gauss=7,
    sigma_gauss=1.5,
    use_se=True,
    use_spatial_attn=True,
    dropout=0.0,
    out_hw=(70, 70),
    token_hw=(14, 14),
    num_global_layers=2,
    num_heads=4,
)
```

建议理解如下：

- `c1`：原始分辨率局部结构编码通道数
- `c2`：token/grid 表示通道数
- `c3`：最终 2D 投影融合通道数
- `out_hw`：输出给融合模块的目标尺寸
- `token_hw`：全局上下文建模时的 token grid 尺寸
- `num_global_layers` / `num_heads`：Transformer 上下文建模强度

## 6. 相比旧版的关键改进

### 6.1 不再固定输入为 70x70

旧版在 `forward()` 里直接断言：

```python
(B, 1, 70, 70)
```

新版支持任意 `H, W`，只要求输入是单通道 horizon mask。

### 6.2 不再把 horizon 当作普通图像

旧版主体仍然是“高斯模糊 + dilated residual CNN + attention”。

新版明确区分：

- 稀疏界面增强
- 界面局部片段编码
- 稀疏带 token 提取
- token 级全局连续性建模
- 再投影回 dense 2D map

这在架构意图上更清晰，也更贴近 horizon 的物理含义。

### 6.3 局部与全局建模分工更明确

旧版全靠卷积扩大感受野。

新版将建模职责拆开：

- 局部模块负责保留界面位置和片段细节
- token/Transformer 模块负责连接长程结构关系

### 6.4 计算重心更偏向活跃区域

虽然新版没有使用脆弱的显式稀疏索引，但通过 `guide` 加权池化，已经把编码重心从“整张图均匀计算”转向“界面带附近优先聚合”。

## 7. 为什么这更符合 horizon 的物理意义

horizon mask 本质上代表地层界面，而不是自然图像中的纹理分布。

因此更合理的做法应该是：

- 先稳定和扩展细界面
- 再提取局部界面片段
- 再理解片段之间的连续关系和全局组织
- 最后把这些信息投影成统一特征图供下游融合

这正是当前实现的结构逻辑。

从建模语义上看，当前编码器更接近：

- sparse interface encoder
- contour-fragment encoder
- band-token + global context encoder

而不再是一个普通的 2D CNN 图像编码器。

## 8. 测试

文件末尾 `__main__` 中提供了 3 组测试：

```python
(2, 1, 70, 70)
(2, 1, 96, 96)
(1, 1, 80, 48)
```

每组测试都会：

- 构造若干 toy horizon line / fragment
- 前向运行编码器
- 检查输出 shape 是否等于目标 `out_hw`

对应代码：

- [HorizonEncoder_70x70.py](/d:/Research/seismic-diff/src/models/conditional_encoder/HorizonEncoder_70x70.py:525)

## 9. 建议的后续演进方向

如果后续还想继续增强这个模块，比较自然的方向有：

- 将 `guide` 从固定公式扩展为“固定算子 + 小型可学习校正”
- 尝试 axial attention 或 row/column mixer 替代标准 Transformer
- 在 tokenization 阶段加入更显式的 interface-strip pooling
- 在 projection head 中加入多尺度 token 回注入

当前版本已经完成从“固定尺寸 CNN”到“稀疏界面 token encoder”的架构转向，适合作为后续继续演进的基础版本。
