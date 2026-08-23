# 阶段4：不确定性门控与加权实施方案

日期：2026-08-22

## 1. 阶段目标

阶段4把阶段2冻结的掩码失败概率与阶段3冻结的位姿失败概率转化为关键帧
“接受、降权、拒绝”决策，减少不可靠观测进入形状更新闭环。实施顺序是先影子
评估、再离线回放、最后在线接入；影子模式只记录决策，不改变BIT输出。

## 2. 概率语义

冻结模型输出均为失败概率：

```text
p_mask_fail = P(mask IoU < 0.90)
p_pose_fail = P(pose fails 5deg/5cm)
q_mask = 1 - p_mask_fail
q_pose = 1 - p_pose_fail
q_base = q_mask * q_pose
```

阶段4 v1禁止直接把两个“失败概率”相乘后当作可靠度。`q_base`是两个可靠度
相乘，等价于在条件独立近似下要求掩码与位姿同时可靠。

## 3. 分阶段优先级

### P0：影子门控基线

改动位置：

- `scripts/build_stage4_shadow_gate.py`
- `tests/test_stage4_shadow_gate.py`

输入使用阶段2/3严格嵌套LOSO的冻结OOF概率，同一对象/序列的三个seed保持同折，
避免在训练数据上报告门控效果。首轮诊断策略为：

```text
q_base >= 0.80       -> accept, weight=1
0.30 <= q_base < .80 -> downweight, weight=q_base/0.80
q_base < 0.30        -> reject, weight=0
```


该阈值仅用于验证管线和方向，不作为冻结上线阈值。首轮83帧结果显示其没有接受
任何关键帧，安全帧误拒率53.8%，因此被明确判定为过度保守的负向控制。风险
排序本身有效：unsafe risk AUROC为0.854，危险帧非接受覆盖率为100%。

随后使用`select_stage4_gate_thresholds_loso.py`在不查看外层对象/序列的情况下
选择阈值。内层约束为安全帧拒绝率不超过10%、安全帧非接受率不超过50%，
目标为最大化危险帧拒绝召回并兼顾降权覆盖。该结果用于判断是否存在可行工作点，
仍需独立序列验证后才能冻结在线阈值。
每个关键帧输出当前mask/pose失败标签、未来5帧位姿失败、未来10帧内模型更新事件，
以及更新前后各3帧失败率。主要验证指标为：

- unsafe risk AUROC；
- reject unsafe recall与reject safe rate；
- accept/downweight/reject各组的当前和未来失败率；
- 被门控覆盖的退化模型更新数量。

### P1：可见性与信息量消融

首版记录但不启用：

```text
q_visibility = geometric_mean(
    visible_boundary_ratio,
    effective_contour_ratio,
    1 - occlusion_ratio)
q_information = clip(min_view_angle_deg / 20deg, 0, 1)
```

原因是阶段3已证明可见性特征在`toy_plane/02`上能改善排序，但全局融合会恶化
校准。只有严格LOSO消融证明其降低错误更新且不提高安全帧拒绝率后，才允许进入
主门控。

P1已完成12种消融。`q_mask*q_pose`保持最高的确认性AUROC 0.854和3.8%的
安全误拒率；绝对可见性/信息量未稳定改善两个问题序列。因果相对信息量点估计
达到0.864，但bootstrap增益区间跨0，属于探索候选。当前不冻结阈值；进入P2
前需使用独立新增关键帧确认。详见`STAGE4_P1_GATE_ABLATION_REPORT.md`。

独立确认及预注册扩样现已完成：36/36次运行完整，得到95个预热后对齐关键帧。
因果相对信息量的总体AUROC为0.745，高于固定`q_mask*q_pose`基线的0.713，且
总体安全误拒率/安全非接受率为0%/30%；但它在`duplo_dude/01`和
`toy_plane/02`上没有改变基线决策，未满足“至少一个问题序列严格改善”和三seed
方向一致要求。因此`enter_p2=false`，P2继续阻塞，转入P1-v2特征修复。详见
`STAGE4_INDEPENDENT_CONFIRMATION_REPORT.md`。

### P2：离线回放与阈值冻结

在序列级嵌套交叉验证中比较：

1. `q_mask`单独门控；
2. `q_pose`单独门控；
3. `q_mask*q_pose`；
4. 加入`q_visibility/q_information`；
5. hard reject与soft weighting。

阈值选择目标为：安全关键帧拒绝率受控时最大化错误更新召回。阈值只能在内层
训练序列选择，外层序列只评估。完成后冻结阈值、数据哈希和策略版本。

### P3：在线影子记录

在Python/C++通信帧号对齐的基础上，在线计算冻结模型概率并记录：

```text
frame_index,p_mask_fail,p_pose_fail,q_base,
q_visibility,q_information,decision,weight,decision_time_ms
```

该步骤仍不改变关键帧发送、轮廓优化和模型发布。

### P4：在线加权和信赖域

仅在P0-P3通过后启用：

- reject：不进入形状训练缓冲；
- downweight：轮廓/掩码损失乘以shadow weight；
- accept：原权重；
- 低`q_pose`时缩小位姿优化信赖域，并禁止产生模型更新候选。

在线策略必须提供总开关和影子模式开关，以便与原BIT逐帧配对比较。

## 4. 验收标准

- 错误模型更新次数下降至少50%；
- 困难序列ADD-AUC提升至少3点；
- 5deg/5cm成功率不下降；
- 普通序列ADD-AUC退化不超过1点；
- 在线门控额外开销低于5%；
- 三个seed方向一致，并报告序列bootstrap 95%置信区间。

## 5. 当前边界

P0结果只能回答“冻结风险分数能否筛出危险关键帧”，不能直接证明跟踪精度已经
提升。只有在线门控与基线进行同序列、同seed配对实验后，才能报告ADD-AUC和
5deg/5cm的阶段4性能增益。

独立确认已经证明因果相对信息量具有总体排序信号，但没有证明它能稳定修复两个
问题序列。不得根据本次确认标签放宽阈值后直接进入P2；任何P1-v2公式都必须重新
冻结，并在未查看的新seed混合标签关键帧上确认。

P1-v2开发与公式冻结现已完成。冻结的“直接`q_pose`因果风险增幅”在131帧严格
序列LOSO中将危险保护率从60.9%提高到83.9%，安全误拒/安全非接受率为
4.5%/45.5%，两个问题序列的危险保护均达到100%。该结果仍属于开发证据；P2
继续阻塞，下一步严格使用冻结阈值在未查看的seed 53–58上独立再确认。详见
`STAGE4_P1V2_DEVELOPMENT_REPORT.md`与`config/stage4_p1v2_reconfirmation.json`。
