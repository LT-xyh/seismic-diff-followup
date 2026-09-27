# Well Encoder

## 1. 设计目标

`WellLogEncoderA` 用于编码稀疏井约束输入，输入张量形状为：

```text
x: [B, 1, H, W]
```

其中：
- `B` 是 batch size
- `H` 是垂向采样长度
- `W` 是横向位置数

这一模态的物理含义不是“稀疏 2D 图像”，而是“若干个横向位置上的高可靠垂向速度曲线”。因此当前版本的核心设计原则是：

- 先把每个横向位置看成一条独立的垂向 trace
- 显式区分有效井值与归一化后的背景常数
- 再在横向轴上建模不同井之间的交互
- 最后把稀疏 trace 表示投影成统一的 2D dense feature map 供后续多模态融合使用

输出张量形状为：

```text
y: [B, C_out, H_out, W_out]
```

其中 `H_out, W_out` 由 `out_hw` 控制，不再固定为 `70 x 70`。


## 2. 数据约定与掩码语义

当前 well-log 数据的原始约定是：

- 原始输入中，井所在位置是物理速度值
- 非井位置是 `0`
- 之后整个图使用固定 min-max 做归一化：
  - `vmin = 1500`
  - `vmax = 4500`

因此原始背景值 `0` 在归一化后会变成一个常数：

```text
background_norm = (0 - 1500) / (4500 - 1500) = -0.5
```

这意味着：

- 非井位置在归一化后不是 0
- 不能通过 `x > 0`、`x != 0`、`abs(x) > 0` 来判断井是否存在

当前实现提供了两个相关工具：

- `normalized_background_value(...)`
- `build_valid_well_mask(...)`

默认逻辑为：

```text
valid_mask = abs(x - background_norm) > tol
```

如果外部已经有显式 mask，也可以通过 `forward(x, valid_mask=...)` 传入，但默认实现已经正确支持当前单输入约定。


## 3. 总体结构

当前 `WellLogEncoderA` 可以分成 4 个阶段：

1. `prepare_input`
2. `trace_encoder`
3. `lateral_mixer`
4. `projector`

整体数据流为：

```text
Normalized Well Map
-> Mask / Support Construction
-> Per-trace 1D Encoding
-> Lateral Trace Interaction
-> Sparse-to-Dense Projection
-> Resize to out_hw
```


## 4. 模块分解

### 4.1 输入准备阶段

`prepare_input(...)` 负责把单通道归一化输入拆成更适合后续处理的稀疏表示。

它会构造：

- `point_mask`
  - 点级有效井掩码，形状 `[B, 1, H, W]`
- `centered_value`
  - 将输入减去归一化背景常数后，再乘以 `point_mask`
  - 非井位置会被显式清零
- `column_support`
  - 沿垂向聚合后的列级 support，表示该横向位置是否存在井
- `soft_support`
  - 对 `column_support` 仅沿宽度方向做高斯扩散，形成软支撑先验

这一阶段的意义是：

- 明确“有效井值”和“背景常数”是两种不同语义
- 保留硬约束信息
- 同时为后续 lateral 交互提供局部软支持范围


### 4.2 逐 trace 的 1D 编码

当前版本不是对 `x` 直接做 2D 卷积，而是先把每个横向位置单独抽成一条垂向信号：

```text
[B, 2, H, W]
-> permute + reshape
-> [B * W, 2, H]
```

这里的两个通道分别是：

- `centered_value`
- `point_mask`

这样做有两个目的：

1. 让编码器看到真实井值与有效性信息
2. 把“每个横向位置是一条垂向 trace”这一物理结构显式写进网络

`trace_encoder` 由以下部分组成：

- `ConvBNAct1d`
- 多个 `TraceResBlock1d`
- `AdaptiveAvgPool1d(trace_latent_steps)`

输出变为：

```text
[B * W, 2, H]
-> [B * W, C_t, Lt]
```

其中：

- `C_t` 是 trace feature channel 数
- `Lt` 是压缩后的 latent vertical steps

这个阶段主要负责学习单条井曲线内部的垂向速度模式，而不是横向空间纹理。


### 4.3 横向 trace 交互阶段

逐 trace 编码之后，网络会对每个横向位置形成一个 trace token：

```text
trace_map: [B, C_t, Lt, W]
-> mean over Lt
-> trace_tokens: [B, C_t, W]
```

然后送入 `LateralTraceMixer`。

该模块输入除了 `trace_tokens` 之外，还显式使用：

- `column_mask_1d`
- `soft_support_1d`

也就是说，横向交互不是盲目地在所有位置上传播，而是受到井位置和软支持区域的约束。

`LateralTraceMixer` 内部包含：

- `1x1` 预融合
- 多个 `LateralMixerBlock`
- 一个基于 support 的 gating 分支

这个阶段的作用是：

- 建模井与井之间的横向影响
- 让稀疏井约束对邻近区域产生可控传播
- 仍然保留“井本身是硬约束”的中心地位


### 4.4 稀疏到稠密的 2D 投影

经过横向交互后，网络已经有了一个带上下文的 trace latent map：

```text
trace_map: [B, C_t, Lt, W]
```

此时进入 `SparseTraceProjector`，它会把以下信息一起送入 2D 模块：

- `trace_map`
- `sparse_value`
- `point_mask`
- `column_support`
- `soft_support`
- 可选坐标通道

这里的设计重点不是把 sparse 输入当成普通图像处理，而是把“trace latent 表示 + 稀疏支持先验”联合映射成 dense context。

输出流程为：

```text
[B, C_t + 4 (+2), Lt, W]
-> stem conv
-> residual refine blocks
-> fuse with trace_map
-> [B, C2d_2, Lt, W]
-> interpolate to out_hw
-> out_proj
-> [B, C_out, H_out, W_out]
```


## 5. 典型形状流

给定输入：

```text
x: [B, 1, H, W]
```

前向传播中的主要形状变化为：

```text
x                       : [B, 1, H, W]
point_mask              : [B, 1, H, W]
centered_value          : [B, 1, H, W]
column_support          : [B, 1, H, W]
soft_support            : [B, 1, H, W]

trace_input             : [B, 2, H, W]
trace_batch             : [B * W, 2, H]
trace_features          : [B * W, C_t, Lt]
trace_map               : [B, C_t, Lt, W]
trace_tokens            : [B, C_t, W]
mixed_tokens            : [B, C_t, W]

sparse_value            : [B, 1, Lt, W]
point_mask_pooled       : [B, 1, Lt, W]
column_support_pooled   : [B, 1, Lt, W]
soft_support_pooled     : [B, 1, Lt, W]

dense_feature           : [B, C2d_2, Lt, W]
interpolate             : [B, C2d_2, H_out, W_out]
out_proj                : [B, C_out, H_out, W_out]
```


## 6. 为什么这比旧版更合适

相对旧版，当前设计有几个关键变化。

### 6.1 不再假设输入固定为 70x70

旧版直接断言输入必须是：

```text
[B, 1, 70, 70]
```

当前版本支持任意 `[B, 1, H, W]`，仅在最终输出阶段通过 `out_hw` 对齐到统一尺寸。


### 6.2 不再用“是否为零”推断井位置

旧版的掩码逻辑本质上依赖于：

```text
x.abs() > 0
```

这与当前数据约定不兼容，因为归一化后背景本来就是非零常数。新版本显式使用归一化背景值和容差来构造 mask，语义是正确的。


### 6.3 不再把 well-log 当作普通稀疏图像

旧版虽然有列方向的 1D 分支，但整体仍然偏向“2D 稀疏图像 + 轻量列增强”的思路。

新版本从主干设计上改成：

- 先逐 trace 编码
- 再横向混合
- 再稀疏到稠密投影

这与 well-log 的物理结构更一致。


### 6.4 硬约束与软支持被分开建模

新版本中：

- `point_mask` / `column_support` 表示硬约束
- `soft_support` 表示邻域扩散后的软支持先验

这种拆分有助于网络区分：

- 哪些位置是井真实观测
- 哪些位置只是靠近井、可受井约束影响


## 7. 为什么这更符合 sparse well constraint 的物理意义

井约束的本质不是“某张图上有几个亮点”，而是：

- 在少数横向位置上，存在高可信的垂向速度曲线
- 这些曲线对所在位置是硬约束
- 对邻近位置则应形成有限、可控的影响

因此一个更合理的编码器应该：

- 优先学习单井的垂向结构
- 再学习井与井之间的横向关系
- 最后再把这些稀疏约束传播到 dense 表示空间

当前 `WellLogEncoderA` 正是按这个顺序组织的，所以比“直接把输入当成稀疏 2D 图像做卷积”更贴近物理意义。


## 8. 可配置项

当前版本的主要可配置参数包括：

- `C_t`
  - trace latent channel 数
- `trace_latent_steps`
  - 垂向 latent 压缩长度
- `n_trace_blocks`
  - 逐 trace 1D 编码深度
- `n_lateral_blocks`
  - 横向交互深度
- `n_dense_blocks`
  - dense projection refine 深度
- `k_soft`, `sigma_soft`
  - 软支持图的横向高斯扩散参数
- `out_hw`
  - 最终输出尺寸
- `mask_tol`
  - 背景常数比较时的容差
- `vmin`, `vmax`, `raw_background`
  - 数据归一化与背景值定义


## 9. 测试

`__main__` 中包含了 3 组测试：

1. 典型输入
   - `B=2, H=70, W=70`
2. 更大输入
   - `B=2, H=96, W=96`
3. 矩形输入
   - `B=1, H=80, W=48`

每组测试都做了以下检查：

- 构造原始 raw well map
- 按固定 `vmin=1500, vmax=4500` 归一化
- 验证背景值归一化后不是 0
- 验证 `infer_valid_mask(...)` 与原始井位置一致
- 运行 encoder
- 验证输出形状是否符合 `out_hw`


## 10. 一句话总结

当前 well encoder 的核心思想是：先把 sparse well 当作一组可靠的垂向 trace 来编码，再做横向交互，最后把稀疏硬约束投影为 dense fusion feature，而不是把它当作普通稀疏 2D 图像处理。
