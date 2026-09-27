# PSTM Encoder

## 1. 设计目标

PSTM 编码器用于处理 post-stack time-migrated seismic image，输入为：

```text
x: [B, 1, T, W]
```

其中：
- `B` 为 batch size
- `T` 为时间轴长度
- `W` 为横向道轴长度

该编码器的设计目标不是把 PSTM 当作普通自然图像处理，而是显式利用它的物理结构特征：
- 时间轴和横向轴具有不同物理含义
- 有价值的信息主要来自局部反射纹理、反射轴连续性、层位几何、断层与终止关系
- 输出仍需保持为统一的 2D feature map，便于后续多模态融合

输出形式为：

```text
y: [B, C_out, H_out, W_out]
```

默认输出尺寸为：

```text
out_hw = (70, 70)
```


## 2. 总体结构

当前 PSTM 编码器由以下几部分组成：

1. `SeismicPatchStem`
2. `StripPatchEmbedding`
3. `Stage1 ~ Stage4` 局部结构编码
4. `LateralContextBlock`
5. `ProjectionHead`

整体数据流为：

```text
Input PSTM
-> Anisotropic Stem
-> Strip Patch Embedding
-> Local Seismic Texture Encoding
-> Lateral Context Modeling
-> Projection to Unified 2D Map
```


## 3. 数据流

### 3.1 输入

输入张量为：

```text
x: [B, 1, T, W]
```

编码器只要求：
- 输入为 4D tensor
- 通道数为 1

不再对 `T=1000`、`W=70` 做硬编码假设。


### 3.2 SeismicPatchStem

这一层是面向 PSTM 的各向异性前端，用于在早期阶段分别提取不同类型的局部结构线索。

包含三个并行分支：

- `temporal_branch`
  - 卷积核为 `(11, 1)`
  - 强调沿时间轴的局部波形和反射纹理

- `event_branch`
  - 卷积核为 `(7, 5)`
  - 提取局部反射事件、层状结构和小尺度构造几何

- `lateral_branch`
  - 卷积核为 `(3, 9)`
  - 强调横向连续性与跨 trace 的局部结构联系

三路特征在通道维拼接后，再通过 `1x1` 卷积融合：

```text
[B, 1, T, W]
-> 3 branches
-> concat on channel
-> 1x1 fuse
-> [B, c1, T, W]
```

这一层的作用是先把 PSTM 分解为“时间纹理 / 局部几何 / 横向连续性”三个方向的局部表征，而不是直接进入统一的通用卷积堆叠。


### 3.3 StripPatchEmbedding

在完成初步局部建模后，使用 `StripPatchEmbedding` 将原始高分辨率输入压缩为更稳定的局部 patch 特征。

该模块包括：
- 一个预处理卷积 `ConvBNAct(channels, channels, k=(7, 3), s=1)`
- 一个带 stride 的 patch embedding 卷积 `ConvBNAct(channels, channels, k=(7, 5), s=(2, 2))`

形状变化大致为：

```text
[B, c1, T, W]
-> [B, c1, T/2, W/2]
```

这里的重点不是图像 patch tokenization，而是保留时间-横向二维网格的前提下，把局部 seismic pattern 编码成低分辨率、可继续建模的结构特征。


### 3.4 Stage1 ~ Stage4：局部结构编码

主干由 4 个阶段组成，每个阶段都包含若干 `SeismicTextureBlock`。

#### SeismicTextureBlock 的内部结构

每个 block 将局部建模拆成 3 个方向：

- `temporal`
  - depthwise conv
  - 卷积核 `(7, 1)`
  - 重点建模沿时间轴的反射纹理

- `lateral`
  - depthwise conv
  - 卷积核 `(1, 9)`
  - 重点建模横向连续性

- `structure`
  - depthwise conv
  - 卷积核 `(3, 3)`
  - 用于补充局部二维构造纹理

三路结果拼接后，再通过 `1x1` 通道混合、SE 门控和残差连接融合：

```text
x
-> temporal branch
-> lateral branch
-> structure branch
-> concat
-> 1x1 mixing
-> SE
-> residual add
```

这种设计比普通 `3x3 CNN` 更适合 PSTM，因为它显式区分了时间方向与横向方向的建模职责。


### 3.5 分阶段下采样

stage 之间使用 `DownsampleBlock` 逐步压缩特征图，但不是简单重复同一种池化逻辑。

各阶段的 stride 设计如下：

- `patch_embed`: `(2, 2)`
- `down1`: `(2, 1)`
- `down2`: `(2, 2)`
- `down3`: `(2, 2)`

大致形状流如下：

```text
x                : [B, 1,   T,    W]
stem             : [B, c1,  T,    W]
patch_embed      : [B, c1,  T/2,  W/2]
stage1 -> h1     : [B, c1,  T/2,  W/2]
down1 + stage2   : [B, c2,  T/4,  W/2]
down2 + stage3   : [B, c3,  T/8,  W/4]
down3 + stage4   : [B, c4,  T/16, W/8]
```

说明：
- 第一阶段之后先优先压缩时间轴，尽量保留横向结构连续性
- 更深层再逐步联合压缩时间轴和横向轴
- 整体目标是在保持结构语义的前提下控制计算量


### 3.6 LateralContextBlock：横向全局交互

`LateralContextBlock` 是该编码器中最能体现 PSTM 物理意义的模块之一。

核心思想：
- 对深层特征 `h4`，保留每个横向位置
- 沿时间轴压缩为固定数量的竖向 strip summary
- 将每个横向位置视为一个 token
- 在横向序列上做轻量级自注意力

其流程为：

```text
h4: [B, c4, H, W]
-> adaptive average pool over time axis
-> [B, c4, context_bins, W]
-> channel projection
-> one token per lateral location
-> self-attention over W
-> broadcast back to 2D feature map
-> residual fusion with h4
```

对应的建模含义是：
- 每个横向位置对应一个局部竖向地震条带的摘要
- 横向 token 之间的 attention 用于建模反射轴跨 trace 的连续性
- 对断层错断、反射终止、横向结构变化等关系更敏感

它不是完整的 Vision Transformer，而是一个明确服务于“横向结构交互”的轻量模块。


### 3.7 ProjectionHead：输出统一融合特征图

最后的 `ProjectionHead` 将中层和深层特征融合，并映射到统一的 2D 输出尺寸。

输入包括：
- `h3`：中层结构特征
- `h4`：更深层、带有横向上下文的特征

处理流程：

1. 分别使用 `1x1` 卷积调整通道
2. 分别通过 `AdaptiveAvgPool2d` 对齐到 `out_hw`
3. 在通道维拼接
4. 再经过一层 `SeismicTextureBlock`
5. 通过 `1x1` 卷积输出最终 `C_out`

输出形状为：

```text
y: [B, C_out, H_out, W_out]
```

默认情况下：

```text
y: [B, C_out, 70, 70]
```

这一层的作用是把编码器内部多尺度、各向异性的 PSTM 结构表示，整理成后续 multimodal fusion 可以直接使用的统一二维特征图。


## 4. 为什么这个结构更适合 PSTM

相比旧版固定形状、重复垂向下采样的 CNN，这个版本更适合 PSTM 的原因在于：

- 它不再假设固定输入尺寸，支持任意 `T` 和 `W`
- 它在前端显式区分时间轴纹理、局部反射结构和横向连续性
- 它没有过早地通过单一池化机制破坏局部 seismic texture
- 它在深层加入了横向序列交互，而不是只依赖局部卷积感受野
- 它最终仍输出规则的 2D feature map，方便接入后续多模态融合网络

简而言之，这个编码器不是“把 PSTM 当图像分类输入”，而是把它当作具有明确时间轴、横向结构和反射连续性的地震剖面来建模。


## 5. 测试配置

当前实现中在 `__main__` 里加入了 3 组形状测试：

```text
(2, 1, 1000, 70)
(2, 1, 1200, 96)
(1, 1, 800, 48)
```

验证目标是：
- 编码器能正常接受不同 `T` 和 `W`
- 输出始终满足配置的 `out_hw`
- 不依赖任何固定输入形状假设


## 6. 一句话总结

这个 PSTM 编码器的数据流可以概括为：

```text
原始 PSTM
-> 各向异性局部前端
-> strip patch embedding
-> 局部 seismic texture / structure 编码
-> 横向上下文建模
-> 投影到统一 2D fusion map
```

它的核心优势在于：既保留了 PSTM 的物理结构含义，又保持了工程上的轻量、模块化和易于接入后续融合网络的特性。
