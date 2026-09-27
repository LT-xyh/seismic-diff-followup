# BG-PDR-FM 代码规范

本文档约束 `bg_pdr_fm` 包内的工程代码。目标是让训练、评估、诊断和论文实验路径长期可维护，而不是追求一次性的大规模格式化。

## 包边界

- `bg_pdr_fm` 是自包含训练/eval 包，运行时不得依赖 `src.models`、`models.conditional_encoder` 或旧实验路径。
- 旧实验代码可以留在仓库中，但不能被 `bg_pdr_fm` 的模型、数据、训练或评估入口导入。
- 包内公共导出集中放在各子包 `__init__.py`；新增导出必须保持向后兼容，不随意删除已有名字。

## 数据合同

- 每个 batch 必须包含 `depth_vel, migrated_image, horizon, rms_vel, well_log, well_mask, modality_mask, modality_quality`。
- `depth_vel`、`horizon`、`well_log`、`well_mask` 是深度域张量；`migrated_image` 和 `rms_vel` 保留数据集原生物理形状。
- 数据适配器不得静默把时间域模态 resize 到 `70 x 70`；需要对齐网格时在模型编码器或 adapter 内完成。
- `well_log` 由 `depth_vel + well_mask` 动态生成，不存入 OpenFWI LMDB。
- `modality_mask` 表示模态是否观测，`modality_quality` 表示观测质量；缺失模态不能参与对比 loss 或负样本。

## 配置约定

- 正式训练参数以 YAML 为唯一来源；脚本不能覆盖 batch size、epoch、fast_run、precision、checkpoint 路径等调参项。
- 三阶段正式配置分别为 contrastive、background、residual；fast-run 配置只用于 smoke。
- `training.precision` 和 `training.matmul_precision` 必须显式写入正式训练配置。
- `training.logging` 控制 Lightning CSV/TensorBoard 日志；`diagnostics` 只控制论文诊断产物。
- stage handoff checkpoint 使用 `contrastive_last.ckpt`、`background_last.ckpt`、`residual_last.ckpt` 这类稳定文件名。

## 训练入口

- 每个阶段可有独立入口，但入口必须是薄封装：只加载对应 YAML、校验 `training.stage`、调用共享训练逻辑。
- 共享训练逻辑由 `bg_pdr_fm.training.train_bg_pdr_fm` 维护，避免各入口复制 DataLoader、Trainer 或 checkpoint 逻辑。
- stage wrapper 不得修改 YAML 中的训练参数；需要调参时直接改 YAML。
- autoencoder 训练入口与 BG-PDR-FM 三阶段入口分离，checkpoint 交接通过配置路径完成。

## 日志与诊断

- Lightning CSV/TensorBoard 是训练日志主路径，必须由 Trainer 标准 logger 产生。
- `diagnostics/*.csv` 和 PNG 面板是论文诊断路径，用于频谱、可靠性、rho/gate、可视化，不替代训练日志。
- 新增指标应同时考虑 Lightning log 和 diagnostics 是否都需要记录。
- 不使用裸 `print` 作为训练/评估输出；需要输出时使用模块级 `logging.getLogger(__name__)`。

## 代码风格

- 当前项目不强制 ruff、black、isort；本阶段采用手工低风险规范化。
- 新代码优先控制普通代码行不超过 119 字符；复杂数学表达式可为可读性保留例外。
- 导入顺序为标准库、第三方、包内模块，中间用空行分隔。
- 类型标注用于公共函数、配置入口、数据结构和张量合同；内部短 helper 可保持简洁。
- 异常信息要说明配置路径、文件路径、stage 或模态名，便于训练中断时定位。
- 注释只解释不显然的物理假设、数据合同或训练约束，避免重复代码本身。

## 测试要求

- 修改训练入口、配置、数据合同、checkpoint、loss 或评估加载时，必须补充或更新 smoke 测试。
- 基础静态检查为 `python -m compileall -q bg_pdr_fm`。
- 回归测试优先覆盖：配置加载、stage launcher、Trainer 参数透传、三阶段 fast-run、checkpoint 串联、推理无 target 泄漏。
- 没有真实数据集时，使用 synthetic smoke；真实 OpenFWI/Marmousi 测试必须能在数据缺失时安全跳过。

## 文档要求

- `README.md` 保持运行入口、配置、输出目录说明；不承载过长工程规范。
- 论文理论、实验 TODO 和工程规范分开维护，避免把代码约定写入理论正文。
- 新增配置或入口时，同步更新 README 和本规范中相关约定。
