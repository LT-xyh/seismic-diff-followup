# Figure 1 作图文档：Motivation and Design Principle

## 1. Figure 定位

推荐标题：

```text
Figure 1: Motivation and Design Principle of Physics-Decomposed Residual Generation
```

图的功能：

1. 在 Introduction 阶段建立审稿人直觉。
2. 连接论文动机与 Method 里的核心变量。
3. 为后续实验归因链做视觉铺垫：
   `Concat-FM -> CNCS-FM -> PDR-FM`。

图不应展示完整网络细节，例如具体 encoder 层数、投影头、Symile shuffle、完整损失函数或 U-Net 结构。这些内容属于 Method figure 或 appendix。

## 2. 推荐放置位置

优先放在 Introduction 中：

```text
方法概述段之后
contributions list 之前
```

若版面紧张，可放在 Method 的 Overview 开头，并在 Introduction 中加入一句：

```latex
Figure~\ref{fig:motivation_design} summarizes the motivation and design principle.
```

由于当前 TeX 使用 `\setcounter{secnumdepth}{0}`，caption 中暂时不要写 `Section~\ref{sec:method}`；建议使用自然语言 `the Method section`。

## 3. 三栏结构

整体采用三栏横向排布：

```text
(a) Asymmetric multimodal evidence
(b) Physical decomposition
(c) Design principle
```

建议视觉流向为从左到右，使用箭头连接三栏。

### (a) Asymmetric Multimodal Evidence

目的：说明不同模态天然提供不同类型的信息，直接拼接会混合数值背景、结构界面和模态噪声。

内容建议：

```text
PSTM image       -> structural cues
interpreted horizon -> structural interfaces
RMS velocity     -> numerical background cues
well log         -> absolute velocity scale
```

视觉建议：

- PSTM / horizon 用绿色或青绿色表示 structural evidence。
- RMS / well log 用蓝色表示 numerical evidence。
- 可以用简化小图标或小型示意块，不必使用真实实验结果。
- 若使用真实或合成视觉示例，必须避免暗示其为 quantitative result。

推荐标签：

```text
Heterogeneous observations provide asymmetric evidence.
```

### (b) Physical Decomposition

目的：说明速度场可以按低频背景和高频结构承担不同角色。

核心公式：

```text
V = B + S
B = P_L(V)
S = P_H(V)
```

或更紧凑地写：

```text
V -> low-frequency background B + high-frequency structure S
```

视觉建议：

- 中间放一个简化速度场 `V`。
- 向右分成两个分支：
  - smooth low-frequency background `B`
  - sharp high-frequency structure `S`
- `B` 使用蓝色；`S` 使用绿色。

推荐标签：

```text
Velocity modeling mixes smooth background and sharp structures.
```

### (c) Design Principle

目的：展示本文如何将前两栏的动机落到最小方法闭环。这里不是完整架构图，只保留核心变量链。

必须使用“自然语言 + 公式”的双层标签，避免 Introduction 阶段符号未定义导致理解困难。

推荐内容：


Numerical condition
$$C_N -> background \hat B$$

Structural condition + background summary
$$C_S + \psi_B(\hat B) -> residual \hat R$$

Final velocity
$$\hat V = \hat B + \hat R$$


视觉建议：

- $C_N$ -> $\hat B$ 用蓝色路径。
- `C_S -> \hat R` 用绿色路径。
- 最终输出 `\hat V` 用中性深灰或蓝绿组合。
- 不要画 `F_base` 进入 generator；如果需要表达防止 bypass，可用一个很小的 muted note：`no direct F_base shortcut`，但不建议让它成为主视觉元素。

推荐标签：

```text
Assign numerical and structural conditions to separate generation roles.
```

## 4. Caption 建议

推荐 LaTeX caption：

```latex
\caption{
Motivation and design principle of physics-decomposed residual generation.
Heterogeneous modalities provide asymmetric evidence: PSTM images and interpreted horizons mainly constrain structural interfaces, while RMS velocity and well logs provide numerical background cues.
This motivates a decomposition of velocity modeling into background estimation and residual generation.
The figure illustrates the design principle that connects the motivation to the method; the full network architecture and training objectives are described in the Method section.
}
```

推荐 label：

```latex
\label{fig:motivation_design}
```

## 5. 与实验归因链的对应关系

| 图中层次 | 对应实验 | 审稿人应理解的点 |
|---|---|---|
| 模态不对称性 | `Concat-FM` vs `CNCS-FM` | 不同模态提供不同证据，因此需要解耦条件。 |
| 物理分解 | `CNCS-FM` vs `PDR-FM` | 速度场可分成背景和结构，因此直接生成完整 `V` 不是最合适的职责分配。 |
| 频率分解诊断 | `PDR-FM` 的 `MAE_L/MAE_H/rho_B` | 低频趋势和高频结构误差可以被分开审计，而不是只看单一 MAE。 |

这张图的作用是：**图给直觉，主表量化贡献，消融隔离变量。**

## 6. 视觉规范

推荐风格：

- AAAI 双栏论文可读，优先适配 `figure*` 跨双栏。
- 白底或近白底。
- 颜色克制：蓝色表示 numerical/background，绿色表示 structural/residual。
- 箭头从左到右，避免交叉。
- 文本短句化，不放长段解释。
- 字体大小保证缩放到论文宽度后仍可读。

避免：

- 不要画完整 U-Net 或 encoder 细节。
- 不要放过多公式。
- 不要使用实验结果热图伪装成定量结果。
- 不要使用纯装饰性渐变、复杂背景或大面积彩色块。
- 不要在 Introduction 图中让未解释符号单独出现；每个关键符号都要配自然语言标签。

## 7. 推荐 LaTeX 插入骨架

若图放在 Introduction 中，可使用：

```latex
\begin{figure*}[t]
\centering
\includegraphics[width=\textwidth]{images/figure1_motivation_design.pdf}
\caption{
Motivation and design principle of physics-decomposed residual generation.
Heterogeneous modalities provide asymmetric evidence: PSTM images and interpreted horizons mainly constrain structural interfaces, while RMS velocity and well logs provide numerical background cues.
This motivates a decomposition of velocity modeling into background estimation and residual generation.
The figure illustrates the design principle that connects the motivation to the method; the full network architecture and training objectives are described in the Method section.
}
\label{fig:motivation_design}
\end{figure*}
```

若版面紧张，可改为单栏图：

```latex
\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{images/figure1_motivation_design.pdf}
\caption{...}
\label{fig:motivation_design}
\end{figure}
```

单栏版本需要进一步简化文字，尤其是第三栏的公式链。

## 8. 交付文件建议

建议最终图文件放在 `docs/paper/images/`：

```text
figure1_motivation_design.vsdx   # 可编辑源文件，若使用 Visio
figure1_motivation_design.pptx   # 可编辑源文件，若使用 PowerPoint
figure1_motivation_design.pdf    # 推荐插入 LaTeX
figure1_motivation_design.png    # 预览与备用
```

如果只交付一种格式，优先交付 PDF；如果还需要后续修改，必须保留 VSDX 或 PPTX 源文件。

## 9. Acceptance Checks

作图完成后检查：

- 图题为 motivation/design principle，而不是 full architecture。
- 图中包含三栏：模态不对称性、物理分解、设计原则。
- 第三栏包含 `C_N -> \hat B`、`C_S + \psi_B(\hat B) -> \hat R`、`\hat V = \hat B + \hat R`。
- 每个方法符号都配有自然语言标签。
- 图中没有完整网络结构、长损失函数或过多训练细节。
- Caption 明确说明该图连接动机和方法，完整架构与训练目标见 Method section。
- 缩放到论文宽度后文字仍可读。
