# BG-PDR-FM: 背景质量门控的物理分解残差流匹配

本文整理最终版本的扩散/流匹配理论。相比早期的 EAO-FM 和 CDO-FM，本版本不再声称两阶段模型在数学上必然优于单阶段模型，而是给出一个更严谨、可检验、可被审稿人追问的理论框架：

> 当数值条件能够有效解释速度场的低频背景时，先用确定性背景估计器移除低熵低频部分，再用 Flow Matching 建模剩余高熵残差，可以降低生成任务的传输能量和建模复杂度。若背景估计误差不可忽略，残差流必须允许低频校正，而不能被强行约束为纯高频。

因此，最终方法命名为 **Background-Gated Physics-Decomposed Residual Flow Matching, BG-PDR-FM**，中文为 **背景质量门控的物理分解残差流匹配**。

## 1. 记号定义

令完整深度速度场为

$$
V \in \mathbb R^{H \times W}.
$$

给定多模态条件输入后，对比学习编码器提取两类主要条件：

$$
C_N = \phi_N(X),
\qquad
C_S = \phi_S(X),
$$

其中 \(C_N\) 表示数值条件特征，主要来自 RMS velocity、well log 等低频数值约束；\(C_S\) 表示结构条件特征，主要来自 PSTM、horizon、edge/interface 等结构约束。

令 \(P_L\) 和 \(P_H\) 为由小波或其他多尺度算子定义的低频与高频投影，并满足近似互补关系：

$$
P_L(V)+P_H(V)\approx V.
$$

定义真实低频背景和高频结构为

$$
B=P_L(V),
\qquad
S=P_H(V).
$$

背景估计器为

$$
\hat B=f_B(C_N).
$$

基于预测背景定义残差目标：

$$
R_{\hat B}=V-\hat B.
$$

最终生成结果为

$$
\hat V=\hat B+\hat R.
$$

这里 \(\hat R\) 由 Flow Matching 模型生成。

## 2. 方法假设

BG-PDR-FM 的核心不是假设背景速度完全确定，而是提出一个可检验的条件：

$$
\mathbb E\|B-\hat B\|^2
<
\mathbb E\|B\|^2.
$$

也就是说，数值条件 \(C_N\) 至少能够解释低频背景中的显著部分。如果该条件不成立，两阶段分解不会天然带来收益。

因此，本文不使用过强的 Dirac delta 假设

$$
p(B|C_N)\approx \delta(B-f_B(C_N)).
$$

更严谨的表述是：背景网络学习给定损失下的 Bayes estimator：

$$
f_B^*
=
\arg\min_f
\mathbb E[\ell(f(C_N),B)].
$$

当 \(\ell\) 为 L2 损失时，

$$
f_B^*(C_N)=\mathbb E[B|C_N].
$$

当 \(\ell\) 为 L1 损失时，\(f_B^*(C_N)\) 对应条件中位数。因此，背景分支是低频条件估计器，而不是低频不确定性的数学消除器。

## 3. Stage 1: 低频背景估计

Stage 1 使用数值特征预测低频背景：

$$
\hat B=f_B(C_N).
$$

训练目标为

$$
\mathcal L_B
=
\|\hat B-P_L(V)\|_1
+
\lambda_H\|P_H(\hat B)\|_1.
$$

其中第一项保证背景预测接近真实低频分量，第二项抑制背景分支生成明显高频结构。

需要注意的是，\(\lambda_H\|P_H(\hat B)\|_1\) 不是严格物理证明，而是一个优化代理，用于鼓励背景预测集中在低频子空间中。

## 4. Stage 2: 残差 Flow Matching

冻结背景估计器 \(f_B\)，构造固定残差目标：

$$
R_{\hat B}=V-\hat B.
$$

可选地对残差进行标准化：

$$
\tilde R_{\hat B}
=
\frac{R_{\hat B}-\mu_R}{\sigma_R+\epsilon}.
$$

线性 Flow Matching 路径定义为

$$
R_t=(1-t)\xi+t\tilde R_{\hat B},
\qquad
\xi\sim \mathcal N(0,I),
\qquad
t\sim \mathcal U(0,1).
$$

对应的目标速度场为

$$
u_t=\tilde R_{\hat B}-\xi.
$$

残差流模型学习

$$
v_\theta(R_t,t,C_S,\hat B)
\approx
\tilde R_{\hat B}-\xi.
$$

训练损失为

$$
\mathcal L_{FM}
=
\mathbb E_{t,\xi,V}
\left[
\left\|
v_\theta(R_t,t,C_S,\hat B)
-
(\tilde R_{\hat B}-\xi)
\right\|_2^2
\right].
$$

推理时从噪声出发，通过 ODE 求解：

$$
\frac{dR_t}{dt}
=
v_\theta(R_t,t,C_S,\hat B),
\qquad
R_0=\xi.
$$

得到 \(\hat R\) 后输出

$$
\hat V=\hat B+\hat R.
$$

## 5. 背景误差不会消失

早期版本容易被误解为：只要学习 \(V-\hat B\)，背景误差就不会影响最终结果。这个说法不严谨。

事实上，

$$
R_{\hat B}
=
V-\hat B
=
B+S-\hat B
=
S+(B-\hat B).
$$

因此，残差目标同时包含结构残差 \(S\) 和背景误差补偿项 \(B-\hat B\)。

这说明：背景误差并没有消失，而是进入了残差目标。如果强行要求残差流完全不包含低频成分，就会与该目标发生冲突。

因此 BG-PDR-FM 引入背景质量诊断指标：

$$
\rho_B
=
\frac{
\mathbb E\|P_L(V-\hat B)\|_1
}{
\mathbb E\|P_H(V)\|_1+\epsilon
}.
$$

若 \(\rho_B\) 较小，残差主要由结构高频组成，可以使用较强的低频抑制。若 \(\rho_B\) 较大，残差流必须允许低频校正。

## 6. 背景质量门控正则

旧版本使用严格低频抑制：

$$
\|P_L(\hat R)\|_1.
$$

这个约束只在背景估计非常准确时合理。更稳妥的做法是使用目标感知低频约束：

$$
\mathcal L_{LF}
=
\|P_L(\hat R)-P_L(R_{\hat B})\|_1.
$$

它允许残差流生成必要的低频校正，但抑制无根据的低频漂移。

另一种做法是背景质量门控的低频抑制：

$$
\mathcal L_{gate}
=
\alpha(\rho_B)\|P_L(\hat R)\|_1,
$$

其中

$$
\alpha(\rho_B)
=
\max\left(0,1-\frac{\rho_B}{\tau}\right).
$$

当背景误差较小时，\(\alpha(\rho_B)\) 较大，鼓励残差集中在结构高频；当背景误差较大时，\(\alpha(\rho_B)\) 下降，避免残差流被错误地禁止低频校正。

最终训练目标可以写为

$$
\mathcal L
=
\mathcal L_B
+
\mathcal L_{FM}
+
\lambda_{LF}\mathcal L_{LF}
+
\lambda_{gate}\mathcal L_{gate}.
$$

实际训练中建议分阶段优化：

1. 先训练 \(f_B\)。
2. 冻结 \(f_B\)，统计 \(\rho_B\)。
3. 根据 \(\rho_B\) 选择残差正则强度。
4. 再训练残差 Flow Matching。

## 7. 命题与证明

### 命题 1: 残差低频能量由背景误差控制

若 \(P_L\) 为线性低频投影，且 \(\hat B\in \mathrm{Range}(P_L)\)，则残差目标的低频部分为

$$
P_L(R_{\hat B})=B-\hat B.
$$

**证明：**

由定义

$$
R_{\hat B}=V-\hat B.
$$

对两侧施加 \(P_L\)：

$$
P_L(R_{\hat B})
=
P_L(V-\hat B).
$$

由于 \(P_L\) 线性，

$$
P_L(R_{\hat B})
=
P_L(V)-P_L(\hat B).
$$

又因为

$$
B=P_L(V),
$$

且 \(\hat B\in \mathrm{Range}(P_L)\)，所以

$$
P_L(\hat B)=\hat B.
$$

因此

$$
P_L(R_{\hat B})
=
B-\hat B.
$$

证毕。

该命题说明，残差流是否可以被视为结构主导，并不是先验成立的，而取决于背景估计误差 \(\|B-\hat B\|\)。这也是引入 \(\rho_B\) 的原因。

### 命题 2: 背景估计有效时，残差流的目标速度能量更低

设 Flow Matching 使用线性路径。直接生成完整速度时，目标样本为 \(X_1=V\)，目标速度为

$$
u_V=V-\xi.
$$

残差生成时，目标样本为 \(X_1=R_{\hat B}=V-\hat B\)，目标速度为

$$
u_R=R_{\hat B}-\xi.
$$

若 \(\xi\sim\mathcal N(0,I)\) 与数据独立且零均值，则

$$
\mathbb E\|u_V\|^2-\mathbb E\|u_R\|^2
=
\mathbb E\|V\|^2-\mathbb E\|V-\hat B\|^2.
$$

**证明：**

由于 \(\xi\) 与 \(V\) 独立且 \(\mathbb E[\xi]=0\)，有

$$
\mathbb E\|V-\xi\|^2
=
\mathbb E\|V\|^2+\mathbb E\|\xi\|^2.
$$

同理，

$$
\mathbb E\|V-\hat B-\xi\|^2
=
\mathbb E\|V-\hat B\|^2+\mathbb E\|\xi\|^2.
$$

两式相减得到

$$
\mathbb E\|u_V\|^2-\mathbb E\|u_R\|^2
=
\mathbb E\|V\|^2-\mathbb E\|V-\hat B\|^2.
$$

证毕。

因此，当背景估计器捕获了稳定低频成分，使得

$$
\mathbb E\|V-\hat B\|^2<\mathbb E\|V\|^2,
$$

残差 Flow Matching 的目标速度能量低于直接生成完整速度的目标速度能量。

这不是证明残差模型一定更准确，而是证明在背景估计有效时，残差流面对的传输任务更轻。

### 命题 3: 低频抑制只有在背景误差足够小时才合理

若使用严格低频惩罚

$$
\|P_L(\hat R)\|_1,
$$

而真实残差满足

$$
P_L(R_{\hat B})=B-\hat B,
$$

则当 \(\|B-\hat B\|\) 不可忽略时，严格低频惩罚会与残差目标发生冲突。

**证明：**

残差流的理想输出应满足

$$
\hat R=R_{\hat B}.
$$

对低频部分有

$$
P_L(\hat R)=P_L(R_{\hat B})=B-\hat B.
$$

若同时最小化严格低频惩罚 \(\|P_L(\hat R)\|_1\)，则优化目标倾向于

$$
P_L(\hat R)\rightarrow 0.
$$

当

$$
B-\hat B \neq 0
$$

时，两个目标不一致。因此严格低频惩罚只有在

$$
\|B-\hat B\|\approx 0
$$

时才合理。

证毕。

该命题直接修正了早期理论中“残差要吸收背景误差，但又禁止残差包含低频”的矛盾。

### 命题 4: 目标感知低频约束不会阻止背景误差校正

若使用目标感知低频约束

$$
\mathcal L_{LF}
=
\|P_L(\hat R)-P_L(R_{\hat B})\|_1,
$$

则理想残差输出 \(\hat R=R_{\hat B}\) 同时使 Flow Matching 目标和低频约束达到一致。

**证明：**

当

$$
\hat R=R_{\hat B}
$$

时，

$$
P_L(\hat R)-P_L(R_{\hat B})=0.
$$

因此

$$
\mathcal L_{LF}=0.
$$

这说明目标感知低频约束不会禁止必要的低频校正，而是要求残差的低频部分与真实背景误差一致。

证毕。

## 8. 与单阶段 Flow Matching 的区别

单阶段 Flow Matching 直接建模

$$
p(V|C_N,C_S).
$$

BG-PDR-FM 建模的是

$$
\hat B=f_B(C_N),
\qquad
p(R_{\hat B}|C_S,\hat B).
$$

二者的根本差别不在于恒等式

$$
V=\hat B+(V-\hat B),
$$

而在于是否满足以下经验假设：

$$
H(R_{\hat B}|C_S,\hat B)
<
H(V|C_N,C_S),
$$

或更容易验证的能量条件：

$$
\mathbb E\|V-\hat B\|^2
<
\mathbb E\|V\|^2.
$$

因此，BG-PDR-FM 的论文主张应该写成：

> 我们提出一种熵引导的生成资源分配策略。低频背景由数值条件估计，高频和未解释残差由 Flow Matching 建模。该方法的收益不是无条件保证的，而是在背景估计器能够显著降低残差能量和低频不确定性时成立。

## 9. 必须报告的诊断指标

为了避免理论空转，实验中必须报告以下诊断指标：

背景误差：

$$
\epsilon_B
=
\mathbb E\|P_L(V)-\hat B\|_1.
$$

残差低频泄漏比例：

$$
\rho_B
=
\frac{
\mathbb E\|P_L(V-\hat B)\|_1
}{
\mathbb E\|P_H(V)\|_1+\epsilon
}.
$$

完整目标速度能量：

$$
E_V
=
\mathbb E\|V-\xi\|^2.
$$

残差目标速度能量：

$$
E_R
=
\mathbb E\|V-\hat B-\xi\|^2.
$$

若实验显示

$$
E_R<E_V
$$

且 \(\rho_B\) 较低，则说明残差 Flow 的任务确实更简单。若 \(\rho_B\) 较高，则应启用目标感知低频校正，而不是使用严格低频抑制。

## 10. 推荐实验设置

主实验应比较：

1. Single DDPM。
2. Two-stage DDPM。
3. Single Flow Matching。
4. PDR-FM with strict low-frequency penalty。
5. BG-PDR-FM with target-aware low-frequency constraint。

关键消融包括：

1. 是否使用背景估计器 \(f_B(C_N)\)。
2. 是否冻结背景估计器。
3. 严格低频抑制 vs 目标感知低频约束。
4. 不同 \(\rho_B\) 区间下的性能变化。
5. 不同 ODE step 数下的速度与精度权衡。

核心评价指标包括：

1. 完整速度 MAE、MSE、SSIM。
2. 低频背景误差 \(\epsilon_B\)。
3. 高频结构误差 \(\|P_H(\hat V)-P_H(V)\|\)。
4. 残差低频比例 \(\rho_B\)。
5. 采样时间与精度。
6. Well-log 位置的 MAE、MSE、相关系数。

## 11. 最终学术表述

BG-PDR-FM 不证明确定性背景在所有情况下最优，也不声称背景误差会自动消失。它提出的是一个可检验的充分条件：

> 如果数值条件能够有效解释低频背景，则将低频背景从生成目标中显式移除，可以降低 Flow Matching 的残差传输能量；如果背景误差不可忽略，则残差流需要通过目标感知低频约束显式校正该误差。

这种表述比“误差吸收”“正交流分解”更稳健，因为它承认方法的适用边界，并把理论主张转化为可以在实验中验证的诊断量。

最终一句话总结：

> BG-PDR-FM 用确定性网络估计低熵背景，用 Flow Matching 生成背景无法解释的残差，并通过背景质量门控决定残差是否需要低频校正，从而在理论上避免过度设计，在工程上保持一次 Flow 采样的效率。
