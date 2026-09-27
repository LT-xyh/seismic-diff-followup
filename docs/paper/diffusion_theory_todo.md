# BG-PDR-FM Diffusion Theory TODO

## 理论整理

- [x] 明确 BG-PDR-FM 主张边界；完成标志：正文写成可检验条件而非必然优越。
- [x] 区分 \(U_m\) 与 \(R_{\hat B}\)；完成标志：表征残差和速度残差不混用。
- [x] 对齐 \(W_L/W_H\) 与 \(P_L/P_H\)；完成标志：说明 anchor 分解与背景投影的关系。
- [x] 补充推理期无 target 说明；完成标志：inference 不访问 \(V\)、wavelet anchor 或真实 \(\rho_B\)。
- [ ] 精简正文证明；完成标志：复杂推导移动到 Appendix。 **下一步**
- [x] 写清背景误差不会消失；完成标志：\(R_{\hat B}=S+(B-\hat B)\) 在正文中明确。
- [x] 写清低频正则适用条件；完成标志：严格低频抑制仅在背景可靠时成立。

## 代码修改

- [x] 检查 background-only 配置；完成标志：可单独启动 `stage: background`。
- [ ] 检查 residual-only 配置；完成标志：可加载 background checkpoint 后启动 `stage: residual`。 **下一步**
- [ ] 给 downstream 配置补 precision；完成标志：background/residual Trainer 使用 YAML 精度配置。 **下一步**
- [x] 确认 \(f_B(C_N)\) 输入合同；完成标志：BackgroundEstimator 只接收 numerical condition。
- [x] 确认 \(C_S\) 输入残差生成器；完成标志：ResidualFlowGenerator 主要消费 structural condition。
- [x] 检查 \(\hat B\) 注入带宽；完成标志：residual 只接收 low-pass background context。
- [x] 检查背景冻结逻辑；完成标志：residual stage 不更新 background 参数。
- [ ] 增加 target-aware 低频约束；完成标志：\(\mathcal L_{LF}\) 可配置开关且可反传。 **下一步**
- [ ] 增加 residual 频谱日志；完成标志：metrics 含 residual low/high energy ratio。 **下一步**
- [ ] 增加 \(E_V/E_R\) 日志；完成标志：可统计残差目标是否降低 transport energy。 **下一步**
- [x] 增加 \(\hat\rho_B\) 诊断；完成标志：predicted gate 与真实 \(\rho_B\) 可比较。
- [x] 增加 checkpoint 串联检查；完成标志：contrastive/background/residual checkpoint 不互相覆盖。

## 流程验证

- [x] 跑 background backward smoke；完成标志：\(\mathcal L_B\) 有限且无 NaN。
- [x] 跑 residual backward smoke；完成标志：Flow loss 有限且可反传。
- [x] 跑三阶段 smoke；完成标志：contrastive/background/residual 顺序完成。
- [x] 验证推理无 target 泄漏；完成标志：`predict_batch` 不访问 wavelet anchor 或真实 \(V\) 指标。
- [x] 验证频域算子稳定；完成标志：\(P_L(V)+P_H(V)\) 近似重构。
- [x] 验证 checkpoint 加载；完成标志：residual stage 可从 background checkpoint 初始化。
- [x] 验证 BF16 训练入口；完成标志：Trainer 收到 `precision: bf16-mixed`。

## 调参

- [ ] 调 `bg_high_weight`；完成标志：\(\hat B\) 高频污染下降。 **下一步**
- [ ] 调 `rho_weight`；完成标志：\(\rho_B\) 校准趋势稳定。
- [ ] 调 `final_l1_weight`；完成标志：residual 重建不过度牺牲 FM loss。
- [ ] 调 residual loss type；完成标志：MSE/L1/Huber 中选稳定项。
- [ ] 调 `rho_tau`；完成标志：背景质量门控分组趋势清晰。 **下一步**
- [ ] 调 ODE steps；完成标志：速度和精度达到平衡。
- [ ] 调 latent size；完成标志：显存、速度和重建质量平衡。
- [ ] 调 batch size；完成标志：Flow loss 方差可接受。
- [ ] 调 learning rate；完成标志：background/residual 均稳定下降。

## 主实验

- [ ] 跑 Single DDPM baseline；完成标志：得到完整速度生成基线。
- [ ] 跑 Two-stage DDPM baseline；完成标志：得到传统两阶段扩散基线。
- [ ] 跑 Single Flow Matching；完成标志：得到单阶段 FM 基线。
- [ ] 跑 strict PDR-FM；完成标志：量化严格低频抑制效果。
- [ ] 跑 BG-PDR-FM；完成标志：主方法指标完整。

## 消融实验

- [ ] 去掉背景估计器；完成标志：验证 \(f_B\) 对下游的贡献。
- [ ] 冻结 vs 不冻结背景估计器；完成标志：比较 staged 与 joint 训练稳定性。
- [ ] strict vs target-aware LF；完成标志：验证目标感知低频约束收益。
- [ ] 加/不加 background gate；完成标志：验证门控是否改善残差频谱。
- [ ] 改变 ODE step 数；完成标志：形成速度-精度曲线。
- [ ] simple codec vs AE codec；完成标志：验证 latent codec 对残差质量影响。

## 缺失实验

- [ ] 缺失 RMS velocity；完成标志：\(\epsilon_B,\rho_B\) 合理升高。
- [ ] 缺失 well log；完成标志：well 位置指标下降可解释。
- [ ] 缺失 horizon/PSTM；完成标志：结构误差变化可解释。
- [ ] 随机 modality dropout；完成标志：泛化优于完整模态专训。
- [ ] 降低 modality quality；完成标志：reliability/gate 随质量下降。

## 诊断实验

- [ ] 报告 \(\epsilon_B\)；完成标志：背景误差低于平凡预测。
- [ ] 报告 \(\rho_B\)；完成标志：残差低频比例可解释。
- [ ] 报告 \(E_R<E_V\)；完成标志：支撑残差任务更轻。
- [ ] 统计 \(\hat\rho_B\) 校准；完成标志：预测分组与真实分组一致。
- [ ] 画残差频谱图；完成标志：展示门控前后频率变化。
- [ ] 画背景误差热图；完成标志：显示失败区域与结构复杂度关系。

## 下游验证

- [ ] 固定 contrastive 训练 background；完成标志：背景误差低于无预训练。
- [ ] 固定 background 训练 residual；完成标志：最终速度指标提升。
- [ ] 比较无 contrastive 下游；完成标志：证明表征学习贡献。
- [ ] 统计 numerical reliability vs background error；完成标志：呈负相关或可解释趋势。
- [ ] 统计 background error vs residual LF ratio；完成标志：门控假设成立。

## 论文产物

- [ ] 汇总主结果表；完成标志：覆盖 MAE、MSE、RMSE、SSIM。
- [ ] 汇总频域诊断表；完成标志：覆盖 \(\epsilon_B,\rho_B,E_R/E_V\)。
- [ ] 汇总消融表；完成标志：覆盖背景、冻结、门控、LF 约束。
- [ ] 汇总缺失模态表；完成标志：覆盖完整、缺失、单模态设置。
- [ ] 输出速度剖面对比图；完成标志：展示背景、残差和最终速度。
- [ ] 输出失败案例；完成标志：说明方法适用边界。
- [ ] 更新 Method 正文；完成标志：与代码接口和实验设置一致。
