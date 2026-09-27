
## Overview

给定多模态地球物理观测：

$$
X=\{x_m\}_{m\in\mathcal M},
$$

其中：

$$
\mathcal M=\{\mathrm{PSTM},\mathrm{Horizon},\mathrm{RMS},\mathrm{Well}\},
$$

目标是估计深度域速度模型：

$$
V\in\mathbb R^{H\times W}.
$$

直接将所有观测模态作为条件输入生成器，容易模糊平滑速度趋势与结构不连续之间的区别。因此，本文将任务分解为两个耦合阶段：**物理锚定的条件学习**和**物理分解的残差生成**。第一阶段将异构观测组织为数值条件和结构条件；第二阶段使用数值条件估计背景场，并通过 Flow Matching 生成剩余残差。

本文采用低频/高频分解作为组织原则：

$$
B=P_L(V),\qquad S=P_H(V),\qquad P_L(V)+P_H(V)\approx V,
$$

其中 $P_L$ 和 $P_H$ 分别表示低通和高通算子。该分解不被视为严格的物理恒等式，而是为两个分支提供清晰的职责划分。
这种职责划分通过一个连接表征学习与生成建模的接口契约来执行：数值条件应当支撑低频背景估计，结构条件应当支撑残差结构生成。

## Physics-Anchored Condition Learning

条件学习器不是通用的特征预训练模块。它的作用是构造两个下游接口 $C_N$ 和 $C_S$，使它们既保留观测模态中的联合证据，又遵守上述背景/结构职责契约。本文将其视为一个受约束的、物理感知的互信息最大化问题：条件空间需要保留观测模态子集中的联合信息，同时被校准到物理可解释的频率锚点，并避免数值角色和结构角色之间的信息冗余。由于精确求解该约束目标不可行，本文使用对比学习代理目标来对应这些约束。

为了使学习到的条件空间与目标物理属性对齐，训练阶段从速度模型中构造小波锚点：

$$
A_N=W_L(V),\qquad A_S=W_H(V).
$$

$A_N$ 作为数值锚点，用于捕捉平滑速度趋势；$A_S$ 作为结构锚点，用于捕捉界面和局部不连续。这些锚点只在训练阶段使用，推理阶段不会提供给模型。

每个模态 $x_m$ 首先被编码为基础特征：

$$
F_m=E_m(x_m),
$$

然后被投影到结构、数值和模态特异子空间：

$$
S_m=P_m^S(F_m),\qquad
N_m=P_m^N(F_m),\qquad
U_m=P_m^U(F_m).
$$

其中，$S_m$ 捕捉结构证据，$N_m$ 捕捉数值证据，$U_m$ 保留压缩后的模态特异信息。模态嵌入还会被融合为内部表示：

$$
F_{\mathrm{base}}=\Phi(\{F_m\}_{m\in\mathcal M}),
$$

但主生成器不会直接使用 $F_{\mathrm{base}}$。相反，它只使用下文定义的解耦数值接口和结构接口作为条件。

表征学习目标结合了四类互补约束：

$$
\mathcal L_{\mathrm{rep}}
=
\mathcal L_{\mathrm{symile}}
+ \lambda_A\mathcal L_A
+ \mathcal L_{\mathrm{pair}}
+ \lambda_F\mathcal L_{\mathrm{freq}}
+ \lambda_O\mathcal L_O.
$$

其中，$\mathcal L_{\mathrm{symile}}$ 鼓励联合跨模态一致性，$\mathcal L_A$ 提供物理语义校准，$\mathcal L_{\mathrm{pair}}$ 在每个角色内部保留实例级判别性，$\mathcal L_{\mathrm{freq}}$ 和 $\mathcal L_O$ 抑制频率泄漏以及数值/结构冗余编码。这些项不是彼此可替换的 loss 堆叠：仅做多模态对齐可能混合背景与结构，仅做 anchor 对齐可能忽视跨模态互补性，而仅做分离正则可能丢弃有用信号。它们共同把接口契约变成可优化、可诊断的目标。

Symile 项通过对比匹配的联合模态元组和 batch 内错配元组，鼓励高阶多模态一致性。对样本 $i$，令 $\Omega_i$ 表示其观测到的模态集合，$r_{i,m}\in\mathbb R^D$ 表示模态 $m$ 的投影表征。候选元组 $\mathbf j$ 的多线性交互得分为：

$$
s(i,\mathbf j;\Omega_i)=
\sum_{d=1}^{D}\prod_{m\in\Omega_i} r_{j_m,m,d}.
$$

紧凑的 subset-Symile 目标写为：

$$
\mathcal L_{\mathrm{symile}}
=
-\frac{1}{B}\sum_i
\log
\frac{\exp s(i,\mathbf i;\Omega_i)}
{\sum_{\mathbf j\in\mathcal J_i}\exp s(i,\mathbf j;\Omega_i)},
$$

其中 $\mathbf i$ 是匹配元组，$\mathcal J_i$ 表示 batch 内形成的候选元组集合。缺失模态只会改变 $\Omega_i$，不会被伪观测替代。完整的 MIP-shuffle 构造和缺失模态处理放入附录。

Anchor 项将 $N_m$ 与 $A_N$ 对齐，将 $S_m$ 与 $A_S$ 对齐；orthogonality 项抑制数值子空间和结构子空间之间的冗余信息。模态可靠性在这里不仅是缺失模态兜底机制，也是接口契约的自适应执行方式：当某个模态缺失或不可靠时，权重 $w_{m,N}$ 和 $w_{m,S}$ 会降低该模态对相应角色的贡献，避免受损证据污染数值或结构接口。

紧凑形式下，anchor loss 写为：

$$
\mathcal L_A
=
\sum_{m\in\mathcal M}
\Bigl(
w_{m,N}\,\ell(q(N_m),q(A_N))
+w_{m,S}\,\ell(q(S_m),q(A_S))
\Bigr),
$$

其中 $\ell$ 表示 InfoNCE loss，$q(\cdot)$ 表示用于对比对齐的投影头。当模态缺失时，损失只在观测到的子集上计算；详细的 subset-Symile 构造放入附录。

为了在频率解耦下保留实例级判别性，本文在每个表征子空间中加入轻量 pairwise contrastive regularizer：

$$
\mathcal L_{\mathrm{pair}}
=
\lambda_{P,N}\mathcal L_{\mathrm{pair},N}
+
\lambda_{P,S}\mathcal L_{\mathrm{pair},S}.
$$

对于每个观测到的模态对 $(m,m')$ 且 $m\ne m'$，$\mathcal L_{\mathrm{pair},N}$ 在同一速度样本的 $q(N_m)$ 和 $q(N_{m'})$ 之间施加对称 InfoNCE loss，并使用 batch 内样本作为负样本；$\mathcal L_{\mathrm{pair},S}$ 对 $q(S_m)$ 和 $q(S_{m'})$ 以相同方式定义。pairwise 项在计算每个样本的损失后按可靠性加权。该辅助项只用于表征训练，推理阶段不需要小波锚点，也不会改变生成器结构。

## Numerical and Structural Condition Interface

模态级表征被聚合为两个下游接口：

$$
C_N=g_N(\{N_m\}_{m\in\mathcal M}),\qquad
C_S=g_S(\{S_m\}_{m\in\mathcal M}).
$$

$C_N$ 总结用于背景估计的数值条件，$C_S$ 总结用于残差生成的结构条件。它们不是普通中间特征，而是从表征学习传递给生成器的接口契约：背景分支可以将 $C_N$ 视为低频数值趋势的来源，残差生成器可以将 $C_S$ 视为结构证据的来源。

在主路径中，$U_m$ 不作为生成器条件使用；它仅作为压缩的模态特异分支保留，用于辅助诊断和消融。PDR-FM 生成器以 $C_N$、$C_S$ 和低带宽背景嵌入作为条件。它不允许直接接收 $F_{\mathrm{base}}$ 作为捷径输入，因为这种旁路会让生成器绕过数值/结构契约，使物理锚定条件学习器变得不可验证。

## Physics-Decomposed Residual Generation

背景分支根据数值条件估计低频分量：

$$
\hat B=f_B(C_N).
$$

该分支使用低频目标进行训练，同时抑制不希望出现的高频泄漏：

$$
\mathcal L_B=\|\hat B-P_L(V)\|_1+\lambda_H\|P_H(\hat B)\|_1.
$$

随后定义残差目标：

$$
R_{\hat B}=V-\hat B.
$$

该残差形式使生成器面对更简单的目标：它不再需要一次性合成完整速度场，而只需要建模背景解释之后剩余的部分。

本文使用 Flow Matching 训练残差生成器。给定高斯噪声 $\xi\sim\mathcal N(0,I)$，在噪声和残差目标之间进行插值：

$$
R_t=(1-t)\xi+tR_{\hat B},
\qquad
u_t=R_{\hat B}-\xi.
$$

残差模型通过以下目标优化：

$$
\mathcal L_{\mathrm{FM}}
=
\mathbb E_{t,\xi,V}
\left[
\left\|
v_\theta(R_t,t,C_S,\psi_B(\hat B))-u_t
\right\|_2^2
\right],
$$

其中 $\psi_B(\hat B)$ 是低带宽背景嵌入。关键设计选择是：生成器接收结构条件和紧凑背景摘要，而不是完整且无约束的背景图像。

为了诊断背景-残差分工是否合理，本文在评估阶段报告残差低频比例：

$$
\rho_B=
\frac{\|P_L(V-\hat B)\|_1}{\|V-\hat B\|_1+\epsilon}.
$$

该统计量只用于分析背景估计是否留下结构化低频误差，不作为主方法推理期条件。主训练目标可以加入固定的残差低频正则：

$$
\mathcal L_{\mathrm{R}}
=
\mathcal L_{\mathrm{FM}}
+\lambda_{R,L}\|P_L(\hat R)\|_1,
$$

其中 $\lambda_{R,L}$ 是固定超参数，而不是依赖目标速度或预测质量门控的动态权重。这里 $\hat R$ 表示生成的残差样本。这样主方法保持简单的背景-残差分解；动态背景质量门控仅作为消融或附录诊断，不作为核心叙事。

## Optimization and Inference

本文采用三阶段训练，以降低表征对齐、背景拟合和残差传输之间的梯度冲突。第一阶段学习物理锚定表征，使 $N_m$ 与 $A_N$ 对齐，$S_m$ 与 $A_S$ 对齐，并将表征空间组织为互补分支。第二阶段使用 $C_N$ 训练背景估计器。第三阶段固定或加载背景分支，训练残差 Flow Matching 模型。端到端微调可作为消融进行评估，但不是主方法公式所必需的。

推理阶段只使用观测模态 $X$。目标速度 $V$ 以及由目标导出的锚点 $A_N$ 和 $A_S$ 从不参与推理。模型预测 $C_N$ 和 $C_S$，估计 $\hat B$，从学习到的 flow 中采样残差，并重建：

$$
\hat V=\hat B+\hat R.
$$

这种分离使锚点仅作为训练期校准机制，并保证推理路径不存在目标泄漏。
