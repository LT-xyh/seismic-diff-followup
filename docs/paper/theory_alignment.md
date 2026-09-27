# 对比学习理论与残差生成理论的一致性整理

本文档用于整理 `contrastive_theory.md` 与 `diffusion_theory.md` 之间的理论关系。核心目的不是重新推导两套理论，而是说明二者是否匹配、如何衔接、哪些变量需要统一，以及后续写论文时应如何组织叙事。

## 1. 总体判断

两份理论总体是匹配的。

`contrastive_theory.md` 解决的是上游条件表征问题：如何从 PSTM、Horizon、RMS、Well 等多模态输入中学习物理可解释的结构表征与数值表征。

`diffusion_theory.md` 解决的是下游生成建模问题：在已经获得结构条件与数值条件后，如何用数值条件估计低频背景，再用 Flow Matching 或 Diffusion 建模背景无法解释的残差。

因此，两者可以形成如下方法链：

```text
多模态输入 X
    ↓
物理锚定对比学习
    ↓
结构条件 C_S 与数值条件 C_N
    ↓
低频背景估计 B_hat = f_B(C_N)
    ↓
残差生成 R = V - B_hat
    ↓
最终速度模型 V_hat = B_hat + R_hat
```

也就是说，对比学习不是一个孤立模块，而是为残差生成模型提供物理锚定的条件空间；残差生成模型也不是普通条件生成器，而是利用该条件空间进行低频背景与结构残差的分工建模。

## 2. 共同物理基础

两份理论共享同一个物理分解思想：速度模型可以近似分解为低频背景与高频结构。

在对比学习理论中，速度模型通过小波锚点分解为

$$
A_N = W_L(V), \qquad A_S = W_H(V).
$$

其中 \(A_N\) 表示数值锚点，主要对应低频背景与平滑趋势；\(A_S\) 表示结构锚点，主要对应高频界面、边界和局部突变。

在残差生成理论中，速度模型被写为

$$
B = P_L(V), \qquad S = P_H(V).
$$

其中 \(B\) 是真实低频背景，\(S\) 是真实高频结构或细节残差。

因此，两个文档中的变量可以建立如下对应关系：

```text
A_N  ≈  B  = P_L(V)
A_S  ≈  S  = P_H(V)
```

需要注意的是，这里的等价是物理语义层面的近似对应，而不是严格数学恒等。因为 \(W_L,W_H\) 与 \(P_L,P_H\) 可以来自不同的小波、多尺度滤波或实现细节。论文中应使用“corresponds to”或“approximates”，避免声称二者完全相同。

但这种近似对应不能停留在文字层面。若 \(W_L\) 与 \(P_L\) 的频谱截断、核函数形状或边界处理差异过大，上游对比学习和下游背景估计可能出现目标不一致：上游 \(N\) 被拉向 \(W_L(V)\)，下游 \(f_B\) 却被 \(P_L(V)\) 和 \(P_H(\hat B)\) 约束。此时中频信息可能既不被低频背景解释，也不被结构残差稳定吸收。

因此，推荐把低频算子设为同源算子：

$$
P_L = \mathcal D \circ W_L,
\qquad
P_H = I-P_L,
$$

其中 \(\mathcal D\) 表示必要的重建、尺度对齐或分辨率适配。若工程上必须使用不同算子，则应报告二者的一致性诊断：

$$
\Delta_L
=
\frac{
\mathbb E\|\mathcal D(W_L(V))-P_L(V)\|_1
}{
\mathbb E\|P_L(V)\|_1+\epsilon
}.
$$

当 \(\Delta_L\) 较大时，论文中不应声称 \(A_N\) 与 \(B\) 是同一低频目标，而应显式引入中频缓冲带：

```text
low-frequency background: assigned to N and B_hat
high-frequency structure: assigned to S and residual generation
mid-frequency transition: reported as ambiguity band and handled by residual correction
```

这可以避免审稿人质疑：上游 anchor 和下游正则是否在频域上“左右互搏”。

## 3. 上游对比学习理论的作用

对比学习理论的核心输出不是直接生成速度模型，而是学习两个主要条件空间：

```text
N: numerical condition representation
S: structural condition representation
U: modality-specific residual condition
```

其中：

- \(N\) 与 \(A_N\) 对齐，用于表达低频速度背景、绝对速度趋势和数值约束；
- \(S\) 与 \(A_S\) 对齐，用于表达界面、断层、边界和局部结构突变；
- \(U\) 保留无法被共享结构/数值空间解释的模态特异信息，但必须被压缩或限制，不能成为绕过 \(S/N\) 解耦的主通道。

对比学习理论中的 Symile、Anchor Calibration、Reliability Weights 和 Orthogonality 的作用可以概括为：

```text
Anchor Calibration: 提供结构/数值物理方向
Symile: 学习高阶多模态互补关系
Reliability Weights: 表达不同模态对结构/数值属性的贡献差异
Orthogonality: 降低 S/N/U 之间的全局冗余
```

这部分理论的目标是让下游生成模型接收到更可解释、更稳定、更接近物理分解的条件，而不是让多模态特征在一个黑箱空间中直接拼接。

## 4. 下游残差生成理论的作用

残差生成理论接收对比学习阶段产生的条件表征，并进行两阶段建模。

Stage 1 使用数值条件估计低频背景：

$$
\hat B = f_B(C_N).
$$

Stage 2 使用结构条件和背景估计结果建模残差：

$$
R_{\hat B}=V-\hat B.
$$

残差 Flow Matching 学习：

$$
v_\theta(R_t,t,C_S,\hat B)
\approx
\tilde R_{\hat B}-\xi.
$$

这里还存在一个实现假设：\(\hat B\) 作为条件输入时，不应以无约束高维特征的形式直接喂给残差生成器。因为 Stage 2 的目标正是 \(V-\hat B\)，如果网络能完整访问 \(\hat B\)，它可能学到捷径映射，用背景的空间梯度或边缘伪迹替代 \(C_S\) 的结构指导，从而削弱上游结构条件的必要性。

更稳妥的写法是让 \(\hat B\) 只提供背景上下文和质量调制，而不是成为结构主条件。例如：

$$
\psi_B(\hat B)
=
\mathrm{LowPassEmbed}(\hat B),
$$

$$
v_\theta
=
v_\theta(R_t,t,C_S,\psi_B(\hat B)).
$$

其中 \(\psi_B\) 应满足低带宽或低分辨率约束，避免把 \(\hat B\) 的局部高频梯度直接暴露给残差网络。也可以只输入背景质量标量或调制向量：

$$
q_B = h_B(\hat B),
\qquad
v_\theta=v_\theta(R_t,t,C_S;q_B).
$$

推荐的消融包括：

```text
1. Stage 2 without B_hat condition.
2. Stage 2 with scalar/vector background-quality condition.
3. Stage 2 with low-pass B_hat embedding.
4. Stage 2 with full-resolution B_hat.
```

如果第 4 项显著优于其他项但 \(C_S\) 消融后性能不下降，则说明模型可能发生了捷径学习。此时主论文应采用第 2 或第 3 项作为更可解释的设计。

最终输出为：

$$
\hat V=\hat B+\hat R.
$$

这套理论的关键主张不是“两阶段一定优于单阶段”，而是一个可检验的充分条件：

$$
\mathbb E\|V-\hat B\|^2
<
\mathbb E\|V\|^2.
$$

如果数值条件 \(C_N\) 能够有效解释低频背景，那么残差生成面对的传输任务更轻；如果背景估计误差不可忽略，则残差流必须允许低频校正，而不能强行把残差限制为纯高频。

## 5. 必须补充的接口定义

当前两份文档最需要补强的是接口层。

对比学习文档中，条件被写为：

$$
C =
\mathrm{Concat}
(
F_{base},
S,
N,
\mathrm{Compress}(U)
).
$$

但残差生成文档中需要两个分开的条件：

$$
C_N = \phi_N(X),
\qquad
C_S = \phi_S(X).
$$

更严格地说，下游接口必须避免表征旁路。主理论路径不应把 \(F_{base}\) 直接送入生成器，因为 \(F_{base}\) 是尚未解耦的混合表征。如果下游的 Diffusion 或 Flow Matching 模型可以直接访问高维 \(F_{base}\)，它可能绕过 \(S/N\) 子空间，从而削弱对比学习理论中 Symile、Anchor Calibration、Reliability Weights 和 Orthogonality 的必要性。

因此，推荐把主接口收紧为：

$$
C_N
=
g_N(N),
$$

$$
C_S
=
g_S(S).
$$

其中 \(g_N\) 和 \(g_S\) 可以是 \(1\times1\) convolution、MLP、adapter 或轻量条件投影模块。

这里的原则是：

```text
N is the only main path to C_N.
S is the only main path to C_S.
F_base is not a downstream condition in the main method.
```

如果希望保留 \(U\)，也不应引入当前理论中不存在的 \(U_N\) 或 \(U_S\)。在 `contrastive_theory.md` 中，\(U\) 是按模态定义的特异残差表征 \(U_m\)，并没有被进一步拆成数值特异残差和结构特异残差。

因此，更稳妥的写法是定义一个受限的辅助条件：

$$
C_U = g_U(\mathrm{Compress}(U)).
$$

其中 \(C_U\) 只能作为低带宽、低权重、可消融的辅助条件使用。需要注意的是，直接通道拼接并不是最稳妥的主方案，因为 \(U\) 表示模态特异异常或偏差，若直接进入后续卷积层，可能污染主结构条件 \(C_S\) 的边界表达。

因此，推荐使用门控或调制式注入，而不是无约束 concat：

$$
\gamma_U,\beta_U
=
h_U(C_U),
$$

$$
\tilde C_S
=
C_S
+
\eta\,
\gamma_U \odot \mathrm{Norm}(C_S)
+
\eta\,\beta_U,
\qquad
0 \leq \eta \ll 1.
$$

或者使用残差门控：

$$
\tilde C_S
=
C_S
+
\eta\,
\sigma(h_U(C_U))\odot r_U(C_U).
$$

这样 \(U\) 只能调制或小幅修正结构条件，而不能直接改写主结构通道。

如果使用通道拼接，应把它作为风险基线，而不是推荐主路径：

$$
C_S^+
=
\mathrm{Concat}(C_S,\eta C_U),
\qquad
0 \leq \eta \ll 1.
$$

更保守地，也可以把 \(U\) 完全降级为诊断项和消融项，而不进入主生成路径。

推荐的保守表述是：

```text
The contrastive encoder produces physics-decoupled numerical and structural
representations. The numerical adapter maps N to C_N for background estimation,
whereas the structural adapter maps S to C_S for residual generation.
The mixed base feature F_base is not directly exposed to the generator.
The unique representation U is used only as a compressed optional residual
condition through gated or modulated injection, or as a diagnostic branch.
```

这样可以避免审稿人质疑：既然下游可以直接使用 \(F_{base}\)，为什么还需要复杂的结构/数值解耦？

若当前工程实现中已经存在 `base / structure / numerical / unique` 的拼接条件，也应在论文主线中把 `base` 解释为消融或过渡实现，而不是理论主张的必要输入。最稳健的实验设置应包含：

```text
1. N -> C_N, S -> C_S only.
2. N/S plus gated compressed U.
3. N/S plus concatenated U.
4. N/S plus F_base bypass.
```

如果第 4 项显著更好，则说明当前解耦表征还不够强；如果第 1 或第 2 项已经足够好，理论闭环才更可信。若第 3 项优于第 2 项但破坏结构边界指标，则说明 \(U\) 的信息有用但注入方式过于粗暴。

## 6. Anchor 的训练期与推理期角色

必须明确区分 anchor 在训练期和推理期的角色。

训练时，\(A_N,A_S\) 可以由真实速度模型 \(V\) 通过小波或多尺度分解得到，用作物理校准方向：

$$
A_N=W_L(V), \qquad A_S=W_H(V).
$$

但推理时，真实速度模型 \(V\) 不存在，因此 \(A_N,A_S\) 不能作为下游生成模型的输入。

正确表述应为：

```text
A_N and A_S are training-time calibration anchors, not inference-time conditions.
At inference time, the generator only receives C_N and C_S predicted from multimodal inputs.
```

这点非常重要。否则审稿人可能质疑方法是否在推理阶段使用了目标速度信息。

## 7. 两种 Residual 的概念区分

两份文档中都出现了 residual，但含义不同。

对比学习理论中的 \(U\) 是模态特异残差信息：

$$
U_m \approx \mathrm{Residual}(F_m \mid S_m,N_m).
$$

它表示某个模态中无法被共享结构空间和数值空间解释的剩余信息，例如 PSTM 的振幅异常、Horizon 的标注误差、RMS 的平滑偏差、Well 的局部采样异常。

残差生成理论中的 \(R_{\hat B}\) 是速度场残差：

$$
R_{\hat B}=V-\hat B.
$$

它表示完整速度场中无法被背景估计器解释的目标生成部分。

因此，两者不应混为一谈。建议统一命名为：

```text
U: modality-specific residual condition
R_Bhat: velocity residual target
```

论文中应避免写成“the residual branch is used to generate the residual”这种模糊表述，因为前一个 residual 可能指 \(U\)，后一个 residual 指 \(R_{\hat B}\)。

更清晰的表述是：

```text
The U branch provides a compressed modality-specific condition,
whereas the residual generator models the velocity residual V - B_hat.
```

同时需要避免使用尚未定义的 \(U_N\) 和 \(U_S\)。除非模型中显式增加两个 projection heads：

$$
U_N=P^U_N(F_m),
\qquad
U_S=P^U_S(F_m),
$$

否则论文中不应把 \(U\) 自然拆分给 \(C_N\) 和 \(C_S\)。当前更合理的处理是：

```text
U is a modality-specific residual condition, not a numerical or structural
condition by default.
```

## 8. 低频约束的理论闭环

还需要解释一个看似矛盾的问题：如果上游 \(N\) 已经通过 Anchor Calibration 对齐了低频锚点 \(A_N\)，为什么下游背景估计器 \(f_B(C_N)\) 仍然需要高频抑制项？

这个问题的答案是：上游 \(N\) 的低频性是条件表征层面的语义约束，而不是输出速度场层面的严格频谱约束。

对比学习中的 anchor calibration 只能说明 \(N\) 与 \(A_N\) 在投影空间、互信息下界或 pooled representation 上更一致。它不能保证经过 nonlinear adapter 和背景估计器之后，输出 \(\hat B=f_B(C_N)\) 必然落在低频子空间中。

因此，下游仍然需要输出空间约束：

$$
\mathcal L_B
=
\|\hat B-P_L(V)\|_1
+
\lambda_H\|P_H(\hat B)\|_1.
$$

其中第一项要求背景估计接近真实低频背景，第二项抑制背景分支生成高频结构。

两者关系可以写成：

```text
Anchor calibration makes N semantically low-frequency.
The high-frequency penalty makes B_hat spectrally low-frequency.
```

也就是说，上游约束的是“条件空间的物理方向”，下游约束的是“生成结果的频谱归属”。这不是重复设计，而是从表征层到输出层的闭环约束。

如果实验中 \(P_H(\hat B)\) 仍然很大，则说明 \(N\) 的低频解耦或 \(f_B\) 的输出约束不充分。此时应报告：

$$
\epsilon_H^B
=
\mathbb E\|P_H(\hat B)\|_1.
$$

该指标可以作为背景分支是否污染结构残差的诊断量。

## 9. Reliability 与背景质量门控的跨理论互动

两份理论之间还存在一个重要互动：上游的数值可靠性权重 \(w_{m,N}\) 与下游背景质量指标 \(\rho_B\) 在物理上应当相关。

在对比学习理论中，\(w_{m,N}\) 表示模态 \(m\) 对数值空间的可信贡献。例如 RMS 和 Well 通常对数值背景更可靠，而 PSTM 和 Horizon 对结构空间更可靠。

可以定义一个样本级或 batch 级的数值可靠性摘要：

$$
\bar w_N
=
\sum_m \pi_m w_{m,N},
$$

其中 \(\pi_m\) 可以是模态存在性、质量评分或固定先验权重。

在残差生成理论中，背景误差和残差低频比例为：

$$
\epsilon_B
=
\mathbb E\|P_L(V)-\hat B\|_1,
$$

$$
\rho_B
=
\frac{
\mathbb E\|P_L(V-\hat B)\|_1
}{
\mathbb E\|P_H(V)\|_1+\epsilon
}.
$$

理论上，如果提供绝对速度信息的模态缺失或质量较差，则 \(\bar w_N\) 应降低，\(\epsilon_B\) 和 \(\rho_B\) 应升高。此时残差生成器不能被强行限制为纯高频，而应允许低频校正。

可以把这个关系写成保守的可检验假设：

```text
Lower numerical reliability should correlate with larger background error
and a higher low-frequency ratio in the residual target.
```

即：

$$
\bar w_N \downarrow
\quad\Rightarrow\quad
\epsilon_B \uparrow,\ \rho_B \uparrow.
$$

这不是必须严格单调成立的定理，而是一个跨模块诊断假设。它可以带来两个好处。

第一，训练期可以报告 \(\bar w_N\)、\(\epsilon_B\)、\(\rho_B\) 之间的相关性，证明上游 reliability 不是装饰项。

第二，推理期真实 \(V\) 不可见，无法直接计算 \(\rho_B\)。因此不能把 \(\bar w_N\) 直接代入 \(\alpha(\rho_B)\)，因为二者量纲和单调方向都不同：\(\bar w_N\) 越大表示越可靠，而 \(\rho_B\) 越大表示残差低频泄漏越严重。

更稳妥的做法是训练一个校准器，将推理期可见信号映射到预测的背景泄漏比例：

$$
\hat\rho_B
=
\kappa_\omega
(
\bar w_N,
q_{\mathrm{miss}},
q_{\mathrm{rms}},
q_{\mathrm{well}},
\epsilon_H^B
),
$$

其中 \(q_{\mathrm{miss}}\) 表示模态缺失模式，\(q_{\mathrm{rms}}\) 和 \(q_{\mathrm{well}}\) 表示 RMS 与 Well 的质量摘要，\(\epsilon_H^B\) 是背景分支高频污染的可观测诊断。训练期用真实 \(\rho_B\) 监督：

$$
\mathcal L_{\rho}
=
\|\kappa_\omega(\cdot)-\rho_B\|_1.
$$

推理期门控不再使用真实 \(\rho_B\)，而是使用预测值：

$$
\hat\alpha
=
\max\left(0,1-\frac{\hat\rho_B}{\tau}\right).
$$

如果希望直接从可靠性得到门控，也必须经过单调校准函数：

$$
\hat\rho_B
=
\mathrm{softplus}(a-b\bar w_N),
\qquad b\ge 0,
$$

再代入 \(\hat\alpha\)。这样既保持“可靠性越高，低频泄漏越小”的物理单调性，又避免把不同量纲的变量强行替换。

推荐报告校准误差：

$$
\mathrm{MAE}_\rho
=
\mathbb E|\hat\rho_B-\rho_B|,
$$

以及分桶可靠性图：

```text
predicted rho_B bin -> empirical rho_B mean -> downstream residual performance
```

只有当 \(\hat\rho_B\) 能够稳定预测真实 \(\rho_B\) 的区间，推理期背景质量门控才是闭合的。

因此，完整系统不只是顺序执行：

```text
contrastive encoder -> residual generator
```

而应被表述为：

```text
contrastive reliability estimates the trustworthiness of numerical conditions,
and residual generation uses background-quality diagnostics or reliability
proxies to decide how much low-frequency correction should remain in the residual.
```

这会让两套理论从“模块串联”变成“物理闭环”。

## 10. 二者匹配后的完整理论链条

将两个文档合并理解后，完整理论链条可以写成：

1. 给定目标速度模型 \(V\)，通过小波或多尺度投影得到训练期物理锚点 \(A_N,A_S\)。
2. 对多模态输入 \(X\) 进行编码，得到结构表征 \(S\)、数值表征 \(N\) 和受限模态残差表征 \(U\)。
3. 通过 Anchor Calibration 使 \(N\) 与 \(A_N\) 对齐、\(S\) 与 \(A_S\) 对齐。
4. 通过 Symile 使多模态结构/数值表征形成高阶互补，而不是独立两两对齐。
5. 通过 Reliability Weights 表达不同模态对结构和数值属性的可信贡献，并形成数值可靠性摘要 \(\bar w_N\)。
6. 将 \(N\) 映射为 \(C_N\)，用于估计低频背景 \(\hat B=f_B(C_N)\)，不将 \(F_{base}\) 作为主路径输入。
7. 使用 \(\|P_H(\hat B)\|_1\) 约束背景分支，保证输出层面仍然属于低频背景。
8. 将 \(S\) 映射为 \(C_S\)，并用低带宽 \(\psi_B(\hat B)\) 或背景质量向量调制残差生成，而不是直接暴露完整 \(\hat B\)。
9. 如果使用 \(U\)，采用门控或调制式注入，而不是无约束通道拼接。
10. 训练 \(\kappa_\omega\) 从推理期可见可靠性信号预测 \(\hat\rho_B\)，并用 \(\hat\alpha=\max(0,1-\hat\rho_B/\tau)\) 控制残差低频校正强度。
11. 使用 \(\bar w_N\)、\(\epsilon_B\)、\(\rho_B\)、\(\hat\rho_B\) 的相关性和校准误差诊断上游可靠性与下游背景质量是否一致。
12. 最终输出 \(\hat V=\hat B+\hat R\)。

这个链条说明：上游对比学习理论和下游残差生成理论不是两个独立技巧，而是“物理锚定条件学习 + 背景质量门控残差生成”的连续框架。

## 11. 对快速投稿与合并投稿的启示

如果先写快速投稿的小论文，可以只保留对比学习部分，重点强调：

```text
Physics-Anchored Reliability-Guided Symile Decoupling
```

主贡献是：

- 使用 \(A_N,A_S\) 作为训练期物理锚点；
- 学习结构/数值解耦的多模态条件表征；
- 使用 Symile 建模高阶多模态互补；
- 使用 reliability weights 处理模态物理不对称；
- 通过诊断实验验证 \(S/N/U\) 的物理可解释性。

此时，Diffusion 或 Flow Matching 只应作为下游验证或未来工作出现，不宜展开完整 BG-PDR-FM 理论，以免后续合并投稿时形成重复发表风险。

如果后续退稿或准备更完整的大论文，则可以把两份理论合并为：

```text
Physics-Anchored Contrastive Residual Generation for Seismic Velocity Modeling
```

或更具体地写为：

```text
Physics-Anchored Contrastive Background-Gated Residual Flow Matching
```

大论文的主线应是：

```text
先学习物理锚定的结构/数值条件空间，
再利用数值条件估计低频背景，
最后用结构条件、背景质量门控和可靠性诊断生成残差。
```

## 12. 建议补入原文档的最小修改

为了让两份文档更匹配，建议至少做九处补充。

第一，在 `contrastive_theory.md` 的下游条件部分补充：

```text
For downstream residual generation, the learned representation is split into
a numerical condition C_N and a structural condition C_S.
C_N is used for background estimation, while C_S is used for residual generation.
The mixed base feature F_base is not directly exposed to the generator in
the main theoretical path, to avoid bypassing the decoupled S/N spaces.
```

第二，在 `contrastive_theory.md` 的 anchor 部分补充：

```text
The wavelet anchors A_N and A_S are only used during training as calibration targets.
They are not available during inference.
```

第三，在 `diffusion_theory.md` 的记号定义部分补充：

```text
C_N and C_S are produced by the upstream physics-anchored contrastive encoder.
C_N corresponds to the numerical representation aligned with A_N,
and C_S corresponds to the structural representation aligned with A_S.
```

第四，在 `diffusion_theory.md` 的背景估计部分补充：

```text
Although N is calibrated toward the low-frequency anchor A_N, this does not
guarantee that the decoded background B_hat is spectrally low-frequency.
Therefore the output-space penalty ||P_H(B_hat)|| is still needed to prevent
the background branch from absorbing structural details.
```

第五，在两份文档的实验诊断部分补充：

```text
Report the correlation between numerical reliability, background error, and
the residual low-frequency ratio. In particular, lower numerical reliability
should correspond to larger epsilon_B and rho_B, indicating that the residual
generator must allow low-frequency correction.
```

第六，在低频算子定义处补充：

```text
The low-frequency anchor operator W_L and the background projection P_L should
be either shared or explicitly calibrated. If different operators are used,
report Delta_L = ||D(W_L(V))-P_L(V)|| / ||P_L(V)|| after reconstruction or
scale alignment, and treat large discrepancies as a mid-frequency ambiguity
band rather than assuming exact equivalence.
```

第七，在推理期门控部分补充：

```text
At inference time rho_B is unavailable. A calibrated predictor rho_hat_B should
be learned from reliability and modality-quality signals, and the gate should
use alpha_hat = max(0, 1-rho_hat_B/tau). Directly replacing rho_B with w_N is
not valid because they have different scales and opposite monotonic directions.
```

第八，在 Stage 2 条件注入部分补充：

```text
B_hat should not be injected as an unconstrained full-resolution condition.
Use a low-pass embedding or a background-quality modulation vector to avoid
shortcut learning where the residual generator ignores C_S.
```

第九，在 \(U\) 分支部分补充：

```text
The unique condition U should be injected through gated or modulated residual
conditioning. Plain channel concatenation is kept as a risk baseline because
it may contaminate the structural condition C_S.
```

补完这些内容后，两份理论就可以稳定衔接，并且更能抵抗“表征旁路”“低频重复约束”“可靠性只是装饰项”“推理期门控断层”“低频算子双重标准”“背景条件捷径学习”等审稿质疑。

## 13. 最终结论

`contrastive_theory.md` 与 `diffusion_theory.md` 在理论方向上是匹配的。它们共享低频/高频物理分解假设，前者学习物理锚定的条件表征，后者利用这些条件进行背景估计和残差生成。

当前主要问题不是理论冲突，而是接口定义和推理期机制尚未完全显式化。只要补清楚 \(A_N/A_S\) 与 \(B/S\) 的对应关系、\(W_L/P_L\) 的算子一致性、\(C_N/C_S\) 的来源、anchor 的训练期角色、\(U\) 与 \(R_{\hat B}\) 的概念区别、\(F_{base}\) 旁路风险、输出层低频约束的必要性、\(\rho_B\) 的推理期校准代理、\(\hat B\) 条件注入的防捷径机制，以及 reliability 与 \(\rho_B\) 的跨模块关系，两份理论就可以自然合并为一个完整框架。

推荐最终统一表述为：

```text
The proposed framework first learns physics-anchored numerical and structural
condition representations through reliability-guided contrastive decoupling.
The numerical condition estimates the low-frequency background, while the
structural condition guides residual generation. To avoid bypassing the
decoupled condition spaces, the mixed base representation is not directly
exposed to the generator in the main path. Background reliability diagnostics
are calibrated into an inference-time estimate of residual low-frequency
leakage, which determines whether the residual generator should focus on
structural high-frequency details or retain low-frequency correction ability.
The background estimate is injected only through low-bandwidth embeddings or
quality modulation to avoid shortcut learning. This turns seismic velocity
modeling from direct conditional generation into background-aware residual
transport under physically interpretable conditions.
```
