# AAAI 2027 实验对比与诊断方案 v2

本文档用于指导 **Physics-Decomposed Residual Flow Matching for Seismic Velocity Modeling** 的 AAAI 2027 实验设计与论文写作。目标不是堆叠大量外部模型，而是用清晰的归因链证明三个核心设计为何必要：

1. 物理锚定条件解耦是否有用。
2. 背景-残差分离是否有用。
3. 低/高频误差分解是否能解释背景与残差的职责划分。

主文实验建议采用 **5 行主表 + 4 行核心消融 + 3 类诊断**。附录承载容量版本、历史 gate 扩展、传统参考方法、完整组件消融和计算预算。

## 1. 实验叙事原则

论文主张应聚焦于：

```text
raw multimodal evidence
-> disentangled numerical/structural conditions
-> deterministic low-frequency background
-> physics-decomposed residual Flow Matching
```

不要写成“Flow Matching 一定优于确定性回归”或“两阶段一定优于单阶段”。更审稿安全的表述是：

> PDR-FM is designed to improve reconstruction accuracy and robustness under missing modalities and frequency-dependent errors by assigning numerical and structural conditions to separate downstream roles.

也就是说，本文要证明的不是生成式模型在所有指标上压倒回归模型，而是 PDR-FM 在多模态不完整和低/高频误差分离的场景下更稳健、更可解释。

## 2. Main Table

主表只放同输入、同数据划分、同缺失模态协议的参考/适配外部方法和最终 Ours。不要把 `Concat-FM`、`CNCS-FM`、容量分支或历史 gate 分支放入主表；这些方法只作为内部消融或附录参考。

| Method | Definition | Purpose |
|---|---|---|
| `MM-InvNet` | Multi-modal InversionNet / MultiConstraint-InversionNet。将 PSTM、horizon、RMS velocity、well log 作为多通道输入，直接回归速度模型 `V`。 | 受控多模态确定性回归基线。 |
| `Two-stage DDPM` | 使用当前多模态输入协议的两阶段生成式参考方法。 | 生成式参考基线。 |
| `Adapted GFI` | GFI/ICLR-style inverse backbone 适配到 PSTM/horizon/RMS/well-log 输入协议。 | 近三年外部架构的 same-input adapted reference。 |
| `Adapted Auto-Linear` | Auto-Linear/ICML-style latent translation 适配到当前多源输入协议。 | 潜空间线性迁移外部架构参考。 |
| `PDR-FM (Ours)` | 完整方法：`C_N -> f_B -> \hat B`，`C_S + \psi_B(\hat B) -> residual FM`，最后 `\hat V=\hat B+\hat R`。 | 验证物理分解残差框架。 |

### 2.1 Concat-FM 的强制定义

`Concat-FM` 不能实现为“随便拼接原始输入 + U-Net”。它必须使用与主方法可比的 modality backbone，并输出未解耦的混合条件 `F_base`。推荐定义为：

```text
F_base = Phi({F_m}_{m in M})
V_hat = FM(F_base)
```

其中 `F_m` 来自与主方法相同类型的模态编码器，`Phi` 是 mask-aware fusion。`Concat-FM` 不使用训练期锚点 `A_N/A_S`，也不生成 `C_N/C_S`。这样 `Concat-FM -> CNCS-FM` 的差异才主要来自“未解耦融合条件 vs 解耦条件”，而不是来自完全不同的编码器容量。

### 2.2 主表指标

主表建议报告：

| Metric | Meaning |
|---|---|
| `MAE` / `RMSE` | 全场速度误差。 |
| `SSIM` | 结构相似性。 |
| `MAE_L` | 低频误差，使用与训练一致的低通算子计算。 |
| `MAE_H` | 高频误差，使用与训练一致的高通算子计算。 |
| Missing-modality average | 对主要缺失模态模式求平均。 |
| Inference time | 单样本或固定 batch 的平均推理时间。 |

低/高频误差建议定义为：

```text
MAE_L = || P_L(V_hat - V) ||_1
MAE_H = || P_H(V_hat - V) ||_1
```

其中 `P_L` 和 `P_H` 使用代码中的 `LowHighPassFilter`，不要在诊断中另行改用不一致的 FFT 或 DWT 划分。

## 3. Core Ablation

正文核心消融保留 4 行，用于解释主方法的内部设计选择。

| Variant | Definition |
|---|---|
| `Concat-FM` | 使用未解耦融合条件 `F_base` 直接生成完整速度场。 |
| `CNCS-FM` | 从 `C_N,C_S` 直接生成 `V`；无背景预测、无残差目标。等价于 no-background-branch counterpart。 |
| `PDR-FM w/o residual frequency regularization` | 保留背景-残差分解，但不使用固定低频残差正则，用于验证低/高频误差权衡。 |
| `PDR-FM (Ours)` | 使用 `C_N` 预测低频背景，并用 `C_S+\psi_B(\hat B)` 生成残差；动态 gate 不作为主方法组件。 |

核心消融表注建议写：

> CNCS-FM is identical to the disentangled full-field FM baseline in Table 1. It directly generates `V` from `C_N` and `C_S`, without `f_B` or residual targets.

### 3.1 解释口径

`CNCS-FM` 不必须优于 `Concat-FM`。如果 `CNCS-FM` 在 MAE/RMSE 上弱于 `Concat-FM`，不要解释为“解耦失败”。更稳妥的写法是：

> CNCS-FM tests whether disentangled conditions alone improve direct full-field generation. PDR-FM further tests whether these conditions become useful when assigned separate background and residual roles.

也就是说，条件解耦的价值不一定表现为直接生成完整 `V` 的单点提升，而可能表现为：

1. 为 `f_B(C_N)` 提供更干净的低频数值接口。
2. 为 residual FM 提供更集中的结构条件 `C_S`。
3. 在缺失模态下通过 reliability 和 subset-Symile 保持条件稳定。

## 4. Diagnostics

正文精选 3 类诊断，每类对应一个 contribution。其余机制分析放附录。

| Diagnostic | Required Output | Contribution Supported |
|---|---|---|
| Missing-modality matrix | `full`、`w/o well_log`、`w/o horizon`、`w/o rms_vel`、`w/o well+rms`、`PSTM only`；报告 MAE/RMSE/SSIM、`MAE_L`、`MAE_H`。 | 多模态可靠性与缺失模态鲁棒性。 |
| Frequency separation | 使用 `LowHighPassFilter` 计算 `MAE_L`、`MAE_H`；对 `S` 分支报告高频能量占比，对 `N` 分支报告低频能量占比。 | 物理锚定 S/N 解耦。 |
| Background-residual diagnostics | 报告 `bg_mae`、`epsilon_H_B`、`\rho_B`、`E_R/E_V`，并按数据族和缺失模态分组。 | 背景分支是否承担低频趋势、残差分支是否承担剩余结构。 |

### 4.1 Missing-Modality Matrix

建议至少报告以下模式：

| Mode | Observed Modalities |
|---|---|
| `full` | PSTM + horizon + RMS velocity + well log |
| `w/o well_log` | PSTM + horizon + RMS velocity |
| `w/o horizon` | PSTM + RMS velocity + well log |
| `w/o rms_vel` | PSTM + horizon + well log |
| `w/o well+rms` | PSTM + horizon |
| `PSTM only` | PSTM |

不要只给一个缺失模态平均数。主文可以展示平均数，但附录必须给完整矩阵。

### 4.2 Frequency Separation

为了保持训练目标和诊断指标一致，低/高频诊断统一使用 `bg_pdr_fm.models.filters.LowHighPassFilter`：

```text
P_L(x) = LowHighPassFilter.lowpass(x)
P_H(x) = x - P_L(x)
```

对预测速度误差：

```text
MAE_L = || P_L(V_hat - V) ||_1
MAE_H = || P_H(V_hat - V) ||_1
```

对条件分支能量：

```text
N_low_ratio = || P_L(C_N) ||_2^2 / (|| C_N ||_2^2 + eps)
S_high_ratio = || P_H(C_S) ||_2^2 / (|| C_S ||_2^2 + eps)
```

如果特征图通道数较多，先对每个通道计算能量，再对通道和样本求平均。不要对 `S/N` 诊断另行使用 DWT，除非论文明确说明训练和诊断采用不同算子。

### 4.3 Background-Residual Diagnostics

评估期可由目标速度计算：

```text
rho_B = || P_L(V - B_hat) ||_1 / (|| V - B_hat ||_1 + eps)
```

该指标不作为主方法推理期条件，而是用于解释背景是否留下结构化低频残差。诊断图建议包括：

1. 不同数据族下的 `bg_mae`、`MAE_L`、`MAE_H`。
2. `rho_B` 与最终 `MAE_L/MAE_H` 的关系。
3. `E_R/E_V` 与残差生成难度的关系。
4. 背景高频泄漏 `epsilon_H_B` 与结构误差 `MAE_H` 的关系。

## 5. Appendix Plan

附录放置不影响主文归因链但有审稿兜底价值的内容。

| Item | Rule |
|---|---|
| `Cond-DDPM` | 相同多模态条件、相同数据划分、可比训练协议。主文实验段落保留一句指向附录。 |
| Residual-DDPM replacement | 在相同背景-残差框架中用 DDPM 替换 FM，验证框架不完全依赖 FM。 |
| Multi-modal VelocityGAN | 仅当条件注入和训练预算能公平说明时加入，标注为 adapted generative baseline。 |
| Smooth Dix / single-modal InversionNet | 仅作为 lower-bound reference，不进入主表排名。 |
| Full component ablations | `w/o anchors`、`w/o Symile`、`anchor-only`、`Symile without anchors`、`w/o reliability`、`w/o orthogonality`、`C_N only`、`C_S only`、`F_base bypass`、`unconstrained coarse predictor`、`dynamic gate extension`。 |
| Representation diagnostics | CKA/cosine 相似度、模态分类器在 `S/N/U` 上的准确率、`S/N/U` 可视化。 |
| Compute budget | 参数量、训练阶段步数、表示预训练成本、推理时间、显存。 |

主文中建议加入一句：

> Additional diffusion-based comparisons under the same multimodal condition are reported in the appendix.

## 6. Fairness Protocol

为了避免审稿人质疑比较不公平，所有主表模型必须满足：

1. 使用相同训练/验证/测试划分。
2. 使用相同可观测模态集合。
3. 使用相同缺失模态协议。
4. 使用相同验证集 checkpoint selection。
5. `Concat-FM`、`CNCS-FM`、`PDR-FM` 使用可比的 FM 生成主干容量。
6. 生成阶段训练步数保持一致；表示预训练成本单独报告。
7. 推理期不访问 `V`、`A_N`、`A_S` 或 evaluation-only residual statistics。

不要求所有方法总 GPU 小时完全一致，因为 `CNCS-FM` 和 `PDR-FM` 的表示预训练是方法本身的一部分。但必须透明报告预训练成本。

## 7. Writing Rules

论文写作时遵循以下规则：

1. `MM-InvNet` 只称为 `controlled multi-modal deterministic baseline`，不称为上界。
2. `Concat-FM` 必须明确使用同一类 modality backbone 和未解耦融合条件 `F_base`。
3. `CNCS-FM` 表注必须写明：不使用 `f_B` 或残差目标。
4. 不声称 `CNCS-FM` 必须优于 `Concat-FM`。
5. 不声称 PDR-FM 严格证明两阶段一定优于单阶段。
6. 不把 `Smooth Dix` 或单模态 `InversionNet` 包装成主文核心证据。
7. 不把附录 `VelocityGAN` 或 `Cond-DDPM` 结果作为主贡献成立的必要条件。

推荐贡献解释：

> The disentangled conditions are not introduced merely to improve full-field generation in isolation. Their main role is to provide role-specific interfaces: `C_N` for background estimation and `C_S` for residual structural generation.

## 8. Acceptance Checks

实验执行和论文整理前进行以下检查：

- 主表只包含 `MM-InvNet`、`Two-stage DDPM`、`Adapted GFI`、`Adapted Auto-Linear`、`PDR-FM (Ours)`。
- `Concat-FM` 使用未解耦 `F_base`，不是随意拼接原始输入的弱基线。
- `CNCS-FM` 明确等价于 no-background-branch counterpart。
- 核心消融表包含 `Concat-FM`、`CNCS-FM`、去固定残差频率正则的 PDR-FM、最终 `PDR-FM (Ours)`。
- 低/高频指标和 S/N 能量诊断统一使用 `LowHighPassFilter`。
- 推理期不访问 `V`、`A_N`、`A_S` 或 evaluation-only residual statistics。
- 附录包含 DDPM、传统 lower-bound reference、完整组件消融和计算预算。
- 论文正文保留一句扩散对比提示，但不让扩散对比抢占主表归因链。

## 9. Suggested Results Layout

建议最终论文实验部分按以下顺序组织：

1. **Experimental Setup**：数据、模态、训练协议、指标和公平性说明。
2. **Main Comparison**：5 行主表。
3. **Core Ablation**：4 行内部消融。
4. **Missing-Modality Robustness**：缺失模态矩阵。
5. **Frequency and Residual Diagnostics**：低/高频误差、S/N 能量、背景-残差分工诊断。
6. **Appendix References**：扩散、传统参考、完整消融、预算。

这样主文读者可以按同一条逻辑读完：先看主效果，再看每个设计为何必要，最后看机制诊断是否支持论文解释。
