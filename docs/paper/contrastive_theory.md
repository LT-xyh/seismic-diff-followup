# Physics-Anchored Reliability-Guided Symile Decoupling

本文档整理最终版对比学习理论：**Physics-Anchored Reliability-Guided Symile Decoupling**，即**物理锚定、可靠性引导的 Symile 多模态解耦表征学习**。

需要首先明确：本文给出的不是严格数学意义上的“证明”，而是一个面向方法章节的**物理启发式推导与可验证命题**。它说明为什么该目标函数是合理的、每个损失项试图优化什么，以及必须通过哪些实验来证明它没有退化为捷径学习或事后合理化。

---

## 1. Motivation

多模态地球物理反演中的输入模态具有明显的物理不对称性：

- `PSTM` 主要携带反射结构、振幅纹理与部分运动学线索。
- `Horizon` 主要携带层位边界与几何拓扑，几乎不直接携带绝对速度数值。
- `RMS` 主要携带平滑低频速度背景，同时包含弱结构阶跃。
- `Well` 主要携带稀疏但高置信度的绝对速度数值。

因此，合理的表征学习目标不是让所有模态在同一个潜空间中完全混合，而是学习：

```text
S: shared structural subspace
N: shared numerical subspace
U: modality-specific residual subspace
```

其中 `S` 应主要解释高频结构，`N` 应主要解释低频速度背景，`U` 保留无法被共享结构/数值空间解释的模态残差信息。

核心假设是：

```text
高质量条件表征 = 物理可解释性 + 多模态互补性 + 空间拓扑保持 + 信息不丢失
```

---

## 2. Wavelet Anchors

给定目标深度速度模型：

$$
V = V_{depth}
$$

使用二维离散小波变换将其分解为低频近似分量与高频细节分量：

$$
A_N = W_L(V), \qquad A_S = W_H(V)
$$

其中：

- $A_S$ 是结构锚点，主要对应速度模型中的高频界面、边界与局部突变。
- $A_N$ 是数值锚点，主要对应速度模型中的低频背景与平滑趋势。

由于小波分解近似满足：

$$
V \approx W_L(V) + W_H(V)
$$

因此 $A_N$ 与 $A_S$ 可以作为低频/高频物理方向的显式锚点。

重要的是，本文不把 $A_S$ 与 $A_N$ 当成简单回归标签，而将其视为**物理校准方向**。即 anchor 的作用是约束潜空间的物理语义，而不是替代多模态联合学习。

---

## 3. Feature Decomposition

对每个模态：

$$
m \in \{\mathrm{pstm}, \mathrm{horizon}, \mathrm{rms}, \mathrm{well}\}
$$

先经过模态编码器得到基础特征：

$$
F_m = E_m(x_m)
$$

再通过三个并行的 $1 \times 1$ projection heads 得到：

$$
S_m = P_m^S(F_m), \qquad
N_m = P_m^N(F_m), \qquad
U_m = P_m^U(F_m)
$$

其中：

- $S_m$ 表示模态 $m$ 对结构空间的贡献。
- $N_m$ 表示模态 $m$ 对数值空间的贡献。
- $U_m$ 表示模态 $m$ 中无法被共享结构/数值空间解释的残差信息。

在当前代码边界中，对应模块为：

```text
DepthVelocityWaveletTransform       -> anchor_struct / anchor_num
FeatureExtractionDecouplingEncoder  -> base / structure / numerical / unique
ContrastivePretrainingLoss          -> contrastive and regularization losses
```

---

## 4. Reliability-Guided Multimodal Sets

不同模态对结构空间和数值空间的可靠性不同，因此引入低维物理可靠性权重：

$$
w_{m,S}, \qquad w_{m,N}
$$

它们表达的是“模态 $m$ 对结构/数值属性的可信贡献程度”。例如：

```text
Horizon: high reliability for S, low reliability for N
RMS:     high reliability for N, medium reliability for S
Well:    high reliability for N, low reliability for dense S
PSTM:    high reliability for S, weak-to-medium reliability for N
```

为避免纯硬编码或完全自由 attention，可靠性权重可以采用物理初始化并允许小范围学习：

$$
\mathcal{L}_{rel}
=
\sum_m
\left[
(w_{m,S} - w^0_{m,S})^2
+
(w_{m,N} - w^0_{m,N})^2
\right]
$$

其中 $w^0$ 是物理先验初始化值。

需要注意一个工程细节：若将 $w$ 直接乘到未归一化特征上，网络可能通过放大特征范数抵消门控。因此更稳妥的实现方式是：

```text
1. 先对投影向量做 L2 normalization，再施加 reliability 权重；
2. 或者将 reliability 作为 loss-level weighting，而不是 feature amplitude scaling。
```

因此，本文的 reliability 不应被理解为普通 attention，而是**物理约束下的低维贡献调节**。

---

## 5. Symile as High-Order Multimodal Fusion

结构空间的联合集合定义为：

$$
X_S =
\{
w_{\mathrm{pstm},S} S_{\mathrm{pstm}},
w_{\mathrm{horizon},S} S_{\mathrm{horizon}},
w_{\mathrm{rms},S} S_{\mathrm{rms}},
w_{\mathrm{well},S} S_{\mathrm{well}},
A_S
\}
$$

数值空间的联合集合定义为：

$$
X_N =
\{
w_{\mathrm{pstm},N} N_{\mathrm{pstm}},
w_{\mathrm{horizon},N} N_{\mathrm{horizon}},
w_{\mathrm{rms},N} N_{\mathrm{rms}},
w_{\mathrm{well},N} N_{\mathrm{well}},
A_N
\}
$$

对每个集合使用 official MIP-shuffle Symile。给定 $K$ 个嵌入向量，Symile 使用多线性内积：

$$
\mathrm{MIP}(z_1,\dots,z_K)
=
\sum_d \prod_{k=1}^{K} z_{k,d}
$$

正样本是同一样本内的多模态组合；负样本是 batch shuffle 后形成的跨样本错配组合。

因此，结构空间与数值空间的 Symile 损失为：

$$
\mathcal{L}_{symile,S}
=
\mathrm{Symile}(X_S)
$$

$$
\mathcal{L}_{symile,N}
=
\mathrm{Symile}(X_N)
$$

从信息论角度看，InfoNCE 类目标可作为互信息下界的估计：

$$
I(Z_1,\dots,Z_K)
\gtrsim
\log B - \mathcal{L}_{NCE}
$$

因此，最小化 $\mathcal{L}_{symile,S}$ 可被解释为提升结构空间中多模态与结构锚点之间的联合一致性；最小化 $\mathcal{L}_{symile,N}$ 可被解释为提升数值空间中多模态与数值锚点之间的联合一致性。

这里必须保持一个保守表述：

```text
Symile is hypothesized to improve high-order multimodal complementarity beyond independent anchor alignment.
```

即 Symile 的必要性不能只靠推导证明，必须通过 `Anchor only`、`Symile only`、`Anchor + Symile` 消融实验验证。

---

## 6. Anchor Calibration

Anchor calibration 的目标是让共享子空间具有物理方向，而不是进行强逐像素回归。

可以使用潜空间对齐：

$$
\mathcal{L}_{anchor,S}
=
\mathrm{InfoNCE}(q(S_m), q(A_S))
$$

$$
\mathcal{L}_{anchor,N}
=
\mathrm{InfoNCE}(q(N_m), q(A_N))
$$

也可以使用轻量空间校准：

$$
\mathcal{L}_{spatial,S}
=
1 - \cos(\phi(S_m), \phi(A_S))
$$

$$
\mathcal{L}_{spatial,N}
=
1 - \cos(\phi(N_m), \phi(A_N))
$$

其中 $q(\cdot)$ 可以是 `GAP -> projector`，$\phi(\cdot)$ 可以是 feature energy map、patch representation 或低分辨率 pooled map。

Anchor calibration 与 Symile 的区别是：

```text
Anchor: provides physical direction.
Symile: learns high-order multimodal consistency.
```

但二者可能存在传递性冗余：如果所有模态都被强力拉向同一个 anchor，则模态间自然靠近，Symile 的边际贡献会下降。因此 $\lambda_A$ 应该是校准权重而非主导权重，并且必须通过消融实验确认 Symile 的额外收益。

---

## 7. Orthogonality as Redundancy Regularization

为降低结构空间与数值空间的冗余，可以加入弱正交项：

$$
\mathcal{L}_{SN}
=
\sum_m
\left|
\cos(\mathrm{GAP}(S_m), \mathrm{GAP}(N_m))
\right|
$$

对特异空间也可加入：

$$
\mathcal{L}_{U}
=
\sum_m
\left(
\left|\cos(\mathrm{GAP}(U_m), \mathrm{GAP}(S_m))\right|
+
\left|\cos(\mathrm{GAP}(U_m), \mathrm{GAP}(N_m))\right|
\right)
$$

这里需要避免过度声称。GAP 后的余弦正交只能说明全局 pooled 表征的线性冗余降低，不能证明二维空间上的物理解耦。因此该项应被表述为：

```text
global redundancy regularization
```

而不是严格意义上的 physical orthogonality proof。

真正的物理解耦需要通过频域能量、空间相关性与下游任务表现验证。

---

## 8. Unique Subspace as Compressed Residual

特异空间定义为：

$$
U_m
\approx
\mathrm{Residual}(F_m \mid S_m, N_m)
$$

它不是第三个共享物理空间，而是模态特异残差信息，例如：

- PSTM 的局部振幅异常、成像伪影、反射响应。
- Horizon 的标注不确定性、边界残差、采样误差。
- RMS 的平滑假设误差与速度转换偏差。
- Well 的稀疏采样偏差与井点局部异常。

如果 $U$ 只有正交和 bottleneck，而没有任何正向驱动力，它可能退化为死区或噪声。因此 $U$ 的使用必须谨慎：

```text
U can be used as a bandwidth-limited residual path, not as an unconstrained main condition.
```

用于下游 Diffusion 时，条件可以写成：

$$
C =
\mathrm{Concat}
(
F_{base},
S,
N,
\mathrm{Compress}(U)
)
$$

其中 $\mathrm{Compress}(U)$ 可以由 $1 \times 1$ bottleneck、dropout 或小权重 adapter 实现，以避免 $U$ 成为绕过 $S/N$ 解耦的旁路。

若实验发现 $U$ 不提供稳定增益，应将其降级为消融项或诊断项。

---

## 9. Missing-Modality Handling

真实地球物理场景中，多模态数据并不总是完整。尤其是 `Well` 可能稀疏或缺失，`Horizon` 可能只在局部区域可靠，`RMS` 也可能存在质量退化。因此，缺失模态不能被简单地 zero-fill，也不应使用 learned missing embedding 或训练集均值向量伪造一个“伪观测”。

本文采用 **mask-aware Subset-Symile**：

```text
Missing modality is unobserved evidence, not a learnable pseudo-observation.
```

也就是说，缺失模态不参与 Symile、不参与 anchor calibration、不参与 orthogonality，也不作为负样本。

### 9.1 Availability Mask

对每个样本定义模态可用性掩码：

$$
p_m \in \{0,1\},
\qquad
m \in \{\mathrm{pstm}, \mathrm{horizon}, \mathrm{rms}, \mathrm{well}\}.
$$

其中 $p_m=1$ 表示模态 $m$ 可用，$p_m=0$ 表示模态缺失。

带 mask 的 reliability weights 写为：

$$
\tilde w_{m,S}
=
\frac{p_m w_{m,S}}
{\sum_j p_j w_{j,S}+\epsilon},
\qquad
\tilde w_{m,N}
=
\frac{p_m w_{m,N}}
{\sum_j p_j w_{j,N}+\epsilon}.
$$

这样缺失模态权重为 0，剩余可用模态的贡献会重新归一化，避免条件特征的尺度因模态缺失而剧烈漂移。

### 9.2 Subset-Symile

在结构空间中，只使用观测到的模态：

$$
X_S^{obs}
=
\{
\tilde w_{m,S} S_m \mid p_m=1
\}
\cup
\{A_S\}.
$$

在数值空间中：

$$
X_N^{obs}
=
\{
\tilde w_{m,N} N_m \mid p_m=1
\}
\cup
\{A_N\}.
$$

然后只在观测子集上计算：

$$
\mathcal{L}_{symile,S}^{miss}
=
\mathrm{Symile}(X_S^{obs}),
\qquad
\mathcal{L}_{symile,N}^{miss}
=
\mathrm{Symile}(X_N^{obs}).
$$

如果某个子空间中的真实可用模态数量少于 2，即：

$$
\left|X_S^{obs} - \{A_S\}\right| < 2
\quad \text{or} \quad
\left|X_N^{obs} - \{A_N\}\right| < 2,
$$

则该子空间的高阶 Symile 项应被 mask out。此时样本退化为 **anchor-only physical calibration**：

```text
No pseudo modality is created.
No cross-modal Symile is computed.
Only observed modalities are calibrated against the physical anchor.
```

这点很重要。单个观测模态无法构成真正的跨模态互补关系，因此不应把退化情形描述为 cross-modal InfoNCE 或 Symile。

### 9.3 Anchor and Regularization Under Missingness

Anchor calibration 只作用于可用模态：

$$
\mathcal{L}_{anchor,S}^{miss}
=
\sum_{m:p_m=1}
\tilde w_{m,S}
\mathcal{L}_{anchor}(S_m,A_S),
$$

$$
\mathcal{L}_{anchor,N}^{miss}
=
\sum_{m:p_m=1}
\tilde w_{m,N}
\mathcal{L}_{anchor}(N_m,A_N).
$$

同理，orthogonality、$U$ residual regularization 和任何可选的 modality adversarial loss 都只在存在模态上计算。对缺失模态计算这些损失会把“缺失模式”误当成物理观测，从而诱导 hallucination。

### 9.4 Modality Dropout

即使训练数据中模态完整，也应在训练期使用 modality dropout：

```text
randomly mask PSTM / Horizon / RMS / Well according to deployment priors
```

dropout 概率不应完全均匀，而应符合实际部署场景。例如：

```text
Well:    more frequently missing
Horizon: partially or regionally missing
RMS:     quality degradation or absent in some surveys
PSTM:    usually available but may be noisy
```

这使模型学习：

```text
any observed modality subset -> stable S/N conditions
```

而不是只适配完整四模态输入。

### 9.5 Downstream Uncertainty Conditioning

推理时真实速度模型不可见，因此没有 $A_S,A_N$。下游 Diffusion 或 Flow Matching 只能接收由可用模态聚合得到的条件：

$$
C_N = g_N(\{N_m \mid p_m=1\}, p),
\qquad
C_S = g_S(\{S_m \mid p_m=1\}, p).
$$

为了让下游生成器知道当前条件的不确定性，推荐把模态掩码和可靠性摘要映射为调制参数，而不是直接作为 0/1 向量 concat 到空间特征中：

$$
r = h_p(p,\bar w_S,\bar w_N,q),
$$

$$
\gamma_p,\beta_p = \mathrm{MLP}(r).
$$

然后使用 FiLM/AdaIN 风格调制时间步嵌入、condition adapter 或残差生成器中间层：

$$
e_t' = \gamma_p \odot e_t + \beta_p.
$$

其中 $q$ 可以表示模态质量摘要，例如 RMS 质量、Well 稀疏度或 Horizon 覆盖率。这样，缺失模式被解释为**条件可靠性和不确定性上下文**，而不是被错误地当成空间图像特征。

### 9.6 Implementation Note

在 batch 内，不同样本可能对应不同的可用模态子集。因此，Subset-Symile 的工程实现应按 observed subset 分组：

```text
1. group samples by observed modality subset;
2. compute subset-specific Symile inside each group;
3. skip Symile if the group is too small or has fewer than two observed modalities;
4. fall back to anchor-only calibration for observed modalities.
```

该策略与普通 NLP/vision 多模态中的 learned missing embedding 不同。对于地球物理反演，缺失 `Well` 意味着没有局部绝对速度约束，缺失 `Horizon` 意味着没有对应的显式边界观测。伪造缺失嵌入会使网络产生虚假的物理自信，因此应避免。

---

## 10. Final Objective

最终目标函数为：

$$
\mathcal{L}
=
\mathcal{L}_{symile,S}^{miss}
+
\mathcal{L}_{symile,N}^{miss}
+
\lambda_A \mathcal{L}_{anchor}
+
\lambda_R \mathcal{L}_{rel}
+
\lambda_O \mathcal{L}_{ortho}
+
\lambda_U \mathcal{L}_{U}
$$

其中：

$$
\mathcal{L}_{anchor}
=
\mathcal{L}_{anchor,S}^{miss}
+
\mathcal{L}_{anchor,N}^{miss}
$$

$$
\mathcal{L}_{ortho}
=
\mathcal{L}_{SN}
+
\mathcal{L}_{U}
$$

`Modality adversarial loss` 可以作为消融项保留，但不建议作为主理论的必要项。原因是强模态对抗可能误伤地球物理信息，尤其当“模态身份”本身与物理信息高度耦合时。

---

## 11. Theoretical Claim

在以下条件近似成立时：

1. 小波锚点 $A_S, A_N$ 能有效分离速度模型的高频结构与低频背景；
2. Symile 的 batch shuffle 负样本能够形成有效跨样本错配；
3. reliability 权重不会被特征范数抵消，也不会塌陷；
4. anchor calibration 不强到完全淹没 Symile；
5. $U$ 受到压缩限制，不成为主旁路；
6. 缺失模态被显式 mask out，而不是被伪造为有效观测；
7. 下游生成器接收模态可用性和可靠性调制，以感知条件不确定性；

最小化上述目标有望得到：

```text
S: 与 A_S 高一致、跨模态互补的结构表征；
N: 与 A_N 高一致、跨模态互补的数值表征；
U: 与 S/N 低冗余、带宽受限的模态残差表征。
```

因此，下游条件生成模型接收的条件：

$$
C =
F_{base} + S + N + \mathrm{Compress}(U)
$$

比原始多模态直接拼接更可能具备：

- 物理可解释性；
- 多模态互补性；
- 空间拓扑保持；
- 模态残差信息保留。

---

## 12. What Is Proven and What Is Not

本文推导中可以相对稳妥主张的是：

- DWT anchor 为结构/数值分解提供物理启发；
- InfoNCE/Symile 类目标可以被解释为互信息下界优化；
- reliability prior 表达模态物理不对称；
- orthogonality 可以降低 pooled 表征冗余；
- compressed $U$ 可以作为模态残差的受限信息通道。
- availability mask 可以避免缺失模态被误当成真实物理观测。

本文不能直接证明的是：

- Symile 一定比 anchor-only 更有效；
- GAP 正交一定等价于二维物理空间解耦；
- reliability gates 一定会学到真实物理可靠性；
- $U$ 一定携带有用残差而不是噪声；
- 多个损失项在深度网络优化中不会发生梯度冲突。
- Subset-Symile 一定能在极端缺失模态场景下保持完整模态性能。

因此，本文理论应被表述为：

```text
physics-guided contrastive decoupling hypothesis
```

而不是：

```text
mathematically proven decoupling framework
```

---

## 13. Required Empirical Verification

为了使该理论在学术上站住，需要用实验填补推导中的不可证部分。

最低限度应包含以下消融：

```text
1. Anchor only
2. Symile only
3. Anchor + Symile
4. Anchor + Symile + reliability
5. with / without U branch
6. with / without S-N orthogonality
7. with / without modality adversarial loss
8. complete modalities vs modality dropout
9. learned missing embedding vs mask-aware Subset-Symile
```

表征诊断应包含：

```text
1. S/N/U feature map heatmap
2. S/N/U t-SNE or UMAP
3. S 与 A_S 的 spatial correlation
4. N 与 A_N 的 spatial correlation
5. S 的高频能量比例
6. N 的低频能量比例
7. S/N feature map overlap
8. modality classifier accuracy on S/N/U
9. feature stability under each missing-modality pattern
```

下游验证应包含：

```text
1. Diffusion / Flow Matching reconstruction metrics
2. background velocity error
3. structural boundary error
4. well consistency error
5. ablation under missing or degraded modalities
6. performance under missing Well / missing Horizon / missing RMS
7. calibration of mask-conditioned uncertainty indicators
```

只有当 `Anchor + Symile` 在表征诊断和下游生成指标上稳定优于 `Anchor only`，并且 $S/N/U$ 的频域与空间统计符合预期时，才能声称该理论有效。

---

## 14. Final Summary

最终理论可以概括为：

```text
以 Symile 学习高阶多模态互补，
以 DWT anchor 提供结构/数值物理方向，
以 reliability weights 处理模态物理不对称，
以 Subset-Symile 处理缺失模态，
以 compressed U 保留不可共享残差信息，
最终为 Diffusion / Flow Matching 提供物理可解释、空间保真、信息较完整的条件表征。
```

这不是严格证明出来的确定性解耦，而是一个需要实验验证的物理启发表征学习假设。它的价值不在于声称神经网络一定会按人类直觉工作，而在于提供一套可诊断、可消融、可被证伪的多模态物理表征学习框架。
