# BG-PDR-FM Contrastive Learning TODO

| 类别 | TODO | 完成标志 |
| --- | --- | --- |
| 代码修改 | 新建 contrastive-only 配置 | 可单独启动 `stage: contrastive` |
| 代码修改 | 检查 Symile MIP-shuffle 实现 | 完整模态 loss 有限且可反传 |
| 代码修改 | 检查 Subset-Symile mask 逻辑 | 缺失模态不参与 loss 和负样本 |
| 代码修改 | 检查 wavelet anchor 训练期使用 | 推理路径不访问 `depth_vel` anchor |
| 代码修改 | 增加 contrastive 日志字段 | metrics 含 Symile、anchor、reliability、orthogonality |
| 代码修改 | 增加 checkpoint 命名区分 | contrastive checkpoint 不覆盖其他 stage |
| 训练启动时机 | 开始 contrastive smoke 训练 | contrastive-only 配置可启动 |
| 训练启动时机 | 开始短程试训 | encoder、anchor、mask、backward 测试通过 |
| 训练启动时机 | 开始正式 contrastive 预训练 | 前若干 epoch 无 NaN 且 loss 稳定下降 |
| 训练启动时机 | 开始 background 训练 | contrastive checkpoint 已保存且 S/N 诊断合理 |
| 训练启动时机 | 开始 residual 训练 | background 误差低于无预训练 baseline |
| 单元与流程验证 | 跑 encoder all-heads 测试 | 四模态 S/N/U head 形状正确 |
| 单元与流程验证 | 跑 wavelet anchor 测试 | `anchor_struct/anchor_num` 形状与输入一致 |
| 单元与流程验证 | 跑单模态缺失测试 | 退化为 anchor-only 且 loss 有限 |
| 单元与流程验证 | 跑混合缺失 batch 测试 | 按观测子集分组无异常 |
| 单元与流程验证 | 跑 contrastive backward smoke | loss 可反传且无 NaN |
| 单元与流程验证 | 跑三阶段 smoke | contrastive/background/residual 顺序完成 |
| 调参 | 调 `temperature` | Symile loss 稳定下降 |
| 调参 | 调 `anchor_weight` | anchor 不淹没 Symile |
| 调参 | 调 `reliability_prior_weight` | 可靠性不塌陷 |
| 调参 | 调 `sn_ortho_weight` | S/N 余弦相关降低 |
| 调参 | 调 `unique_ortho_weight` | U 与 S/N 冗余降低 |
| 调参 | 调 `embed_dim` | 表征指标和显存开销平衡 |
| 调参 | 调 batch size | Subset-Symile group 数量足够 |
| 调参 | 调 wavelet level | anchor 频谱分离清晰 |
| 表征诊断实验 | 画 S/N 频谱分布 | N 偏低频，S 偏高频 |
| 表征诊断实验 | 画 anchor alignment 曲线 | S 对齐结构锚，N 对齐数值锚 |
| 表征诊断实验 | 统计 S/N cosine | 分支相关性低于 baseline |
| 表征诊断实验 | 统计 reliability 分布 | 模态权重符合物理预期 |
| 表征诊断实验 | 检查 U 分支贡献 | 确认保留、降权或移除策略 |
| 表征诊断实验 | 做 t-SNE/UMAP 可视化 | 不同物理属性有可解释聚类 |
| 消融实验 | Anchor only | 得到基础物理校准性能 |
| 消融实验 | Symile only | 验证无 anchor 时的互补学习能力 |
| 消融实验 | Anchor + Symile | 优于单独项 |
| 消融实验 | Anchor + Symile + reliability | 验证可靠性收益 |
| 消融实验 | 加/不加 orthogonality | 验证解耦正则收益 |
| 消融实验 | 加/不加 U 分支 | 判断 U 是否稳定增益 |
| 消融实验 | Pairwise InfoNCE 对比 Symile | 验证高阶对齐收益 |
| 消融实验 | Learned missing embedding 对比 mask-aware | 验证缺失模态处理策略 |
| 缺失与退化实验 | 缺失 well | 性能下降可控 |
| 缺失与退化实验 | 缺失 horizon | 结构表征仍稳定 |
| 缺失与退化实验 | 缺失 RMS | 数值表征下降符合预期 |
| 缺失与退化实验 | 单模态输入 | 不会崩溃且诊断合理 |
| 缺失与退化实验 | 随机模态 dropout | 泛化优于完整模态专训 |
| 缺失与退化实验 | 降低 modality quality | reliability 随质量下降 |
| 缺失与退化实验 | 稀疏 well mask 测试 | well 编码不伪造 dense 信息 |
| 下游验证 | 固定 contrastive 训练 background | 背景误差低于无预训练 |
| 下游验证 | 固定 background 训练 residual | 最终速度指标提升 |
| 下游验证 | 比较无 contrastive 下游 | 证明对比学习贡献 |
| 下游验证 | 统计 numerical reliability vs background error | 负相关或可解释趋势 |
| 下游验证 | 统计 background error vs residual 低频比例 | 门控假设成立 |
| 下游验证 | 检查推理无 target 泄漏 | inference 不使用 `depth_vel` anchor |
| 论文实验产物 | 汇总主表指标 | 含 MAE、RMSE、SSIM 或项目标准指标 |
| 论文实验产物 | 汇总消融表 | 覆盖 anchor、Symile、reliability、orthogonality |
| 论文实验产物 | 汇总缺失模态表 | 覆盖完整、缺失、单模态设置 |
| 论文实验产物 | 输出频谱诊断图 | 能支撑 S/N 分工论述 |
| 论文实验产物 | 输出 reliability 诊断图 | 能支撑物理可靠性论述 |
| 论文实验产物 | 输出失败案例 | 说明方法边界 |
