# RMS Encoder

## 1. 设计目标

RMS 编码器用于处理 RMS velocity 剖面，输入为：

```text
x: [B, 1, T, W]
```

其中：
- `B` 为 batch size
- `T` 为时间轴长度
- `W` 为横向道轴长度

RMS velocity 与 PSTM 的物理含义不同。它不是反射纹理图像，而更接近一组按横向排列的速度曲线集合。因此该编码器的设计重点是：

- 先把每条 lateral trace 当作独立的 1D 速度曲线建模
- 再在横向方向上进行 trace 间交互
- 最后投影为统一的 2D 特征图，供后续多模态融合使用

输出形式为：

```text
y: [B, C_out, H_out, W_out]
```

默认输出尺寸为：

```text
out_hw = (70, 70)
```


## 2. 总体结构

当前 RMS 编码器 `RMSVelocityEncoderA` 由以下 4 个阶段组成：

1. `trace_stem`：逐道 1D 编码
2. `temporal_pool`：每条道的时间压缩
3. `lateral_mixer`：横向 trace 交互
4. `projection_head`：映射到统一 2D feature map

整体数据流为：

```text
Input RMS
-> Per-trace 1D Encoding
-> Temporal Compression
-> Lateral Trace Mixing
-> 2D Projection for Fusion
```


## 3. 数据流

### 3.1 输入

输入张量为：

```text
x: [B, 1, T, W]
```

编码器只要求：
- 输入必须是 4D tensor
- 通道数必须为 1

不再假设固定输入形状，例如：
- `T = 1000`
- `W = 70`


### 3.2 Step 1：逐道 1D 编码

RMS 编码器的第一步不是做二维卷积，而是把每一个横向位置对应的一条 RMS trace 单独拿出来处理。

在 `forward` 中：

```text
x: [B, 1, T, W]
-> permute
-> reshape
-> trace_batch: [B * W, 1, T]
```

这样做的意义是：
- 每条 trace 被看作一条独立的 1D 速度曲线
- 所有 lateral position 共享同一套 1D 编码器参数
- 更符合 RMS velocity 的物理结构

随后进入 `trace_stem`。

`trace_stem` 包括：
- 一个 `ConvBNGELU1d(1, trace_channels, k=7)`
- 多个 `TraceResBlock1d`

输出形状为：

```text
[B * W, 1, T]
-> [B * W, C_trace, T]
```

其中 `TraceResBlock1d` 的作用是：
- 用 depthwise 1D 卷积提取每条速度曲线的局部纵向模式
- 用 pointwise 卷积做通道扩展与回投影
- 通过残差连接稳定训练

这一阶段主要学习的是单条 RMS 曲线内部的时间结构，而不是横向结构。


### 3.3 Step 2：时间压缩

对每条 trace 完成 1D 编码后，使用：

```text
AdaptiveAvgPool1d(latent_time_steps)
```

把时间轴压缩到固定长度 `Lt`。

形状变化为：

```text
[B * W, C_trace, T]
-> [B * W, C_trace, Lt]
```

随后再恢复回 batch 结构：

```text
[B * W, C_trace, Lt]
-> [B, W, C_trace, Lt]
```

这里的作用是：
- 让不同长度的输入时间轴映射到统一的 latent temporal representation
- 保留每条 trace 的时间结构摘要
- 为后续跨 trace 建模提供稳定接口


### 3.4 Step 3：横向 trace 混合

在 RMS 编码器中，横向关系不是通过 2D CNN 直接提取，而是先将每条 trace 压缩成 token，再沿横向轴做交互。

当前实现中：

```text
trace_features: [B, W, C_trace, Lt]
-> mean over Lt
-> trace_tokens: [B, W, C_trace]
-> permute
-> [B, C_trace, W]
```

也就是说，每条 trace 会被压缩成一个横向 token，用于表示该 trace 的整体速度特征。

然后送入 `lateral_mixer`，它由多个 `LateralMixerBlock` 组成。

每个 `LateralMixerBlock` 本质上是一个沿横向轴的一维残差混合器：
- 输入形状为 `[B, C_trace, W]`
- 仅在横向维 `W` 上做 depthwise 卷积
- 不在时间轴上再次卷积

这样做的含义是：
- 建模相邻 trace 之间的 lateral continuity
- 捕获横向速度变化趋势
- 让单条 trace 的表示能够吸收邻近位置的信息

混合后的 token 再加回原始 `trace_features`：

```text
trace_tokens: [B, C_trace, W]
-> permute + unsqueeze
-> [B, W, C_trace, 1]
-> add to trace_features
```

从而完成“逐道编码 + 横向交互”的结合。


### 3.5 Step 4：转成统一 2D 特征图

在完成逐道建模和横向混合后，编码器需要把内部表示转换为后续融合模块可直接使用的 2D feature map。

当前实现中：

```text
trace_features: [B, W, C_trace, Lt]
-> permute
-> feature_map: [B, C_trace, Lt, W]
```

这一步很关键，因为它把“trace set representation”重新组织成二维网格：
- 高度维对应 latent time
- 宽度维对应 lateral trace position

接着做：

```text
F.interpolate(feature_map, size=out_hw, mode="bilinear")
```

得到统一尺寸：

```text
[B, C_trace, Lt, W]
-> [B, C_trace, H_out, W_out]
```

然后送入 `projection_head`：
- 一个 `ConvBNGELU2d`
- 多个 `ResBlock2d`

最后通过 `1x1` 卷积输出最终通道数：

```text
[B, C_mid, H_out, W_out]
-> [B, C_out, H_out, W_out]
```


## 4. 典型形状流

如果输入为：

```text
x: [B, 1, T, W]
```

则整体数据流可概括为：

```text
x                      : [B, 1, T, W]
trace_batch            : [B * W, 1, T]
trace_stem             : [B * W, C_trace, T]
temporal_pool          : [B * W, C_trace, Lt]
restore trace layout   : [B, W, C_trace, Lt]
trace_tokens           : [B, C_trace, W]
lateral_mixer          : [B, C_trace, W]
feature_map            : [B, C_trace, Lt, W]
interpolate            : [B, C_trace, H_out, W_out]
projection_head        : [B, C_mid, H_out, W_out]
out_proj               : [B, C_out, H_out, W_out]
```

默认设置下：

```text
out_hw = (70, 70)
```

因此最终输出通常为：

```text
y: [B, C_out, 70, 70]
```


## 5. 为什么这个结构更适合 RMS Velocity

这个结构比直接使用普通 2D CNN 更适合 RMS velocity，原因在于：

- RMS velocity 更像一组 1D 速度曲线，而不是自然图像纹理
- 每条 trace 的纵向结构本身就有独立意义，适合先做 shared 1D 编码
- trace 之间的关系主要体现在横向变化和横向连续性，因此横向混合应被单独建模
- 最终再统一投影成 2D 特征图，可以兼顾物理合理性和工程可接入性

一句话概括：

```text
先逐道理解速度曲线，再建模道间关系，最后转成统一二维表示。
```


## 6. Decoder 说明

文件中还包含 `RMS2DepthDecoder`，用于把 RMS 编码器输出的二维特征进一步解码为深度域预测图：

```text
f_rms: [B, C_in, H, W]
-> stem
-> residual refinement
-> context branch
-> fuse
-> head
-> [B, 1, H, W]
```

其中支持可选的坐标通道：
- `yy`
- `xx`

用于增强空间位置感知能力。


## 7. 测试配置

当前实现中在 `__main__` 里加入了以下测试输入：

```text
(2, 1, 1000, 70)
(2, 1, 1200, 96)
(1, 1, 800, 48)
```

验证目标是：
- 编码器支持可变 `T` 和 `W`
- 编码器输出固定为 `out_hw`
- 解码器输出与编码器输出空间尺寸一致


## 8. 一句话总结

这个 RMS 编码器的数据流可以概括为：

```text
原始 RMS velocity
-> 每条 trace 的共享 1D 编码
-> 时间轴压缩
-> 横向 trace 混合
-> 投影到统一 2D fusion map
```

它的核心优势在于：既保持了 RMS velocity 作为“trace set”的物理结构，又输出了后续多模态融合网络容易使用的规则二维特征图。
