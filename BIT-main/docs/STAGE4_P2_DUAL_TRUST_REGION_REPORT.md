# 阶段4 P2：双候选信赖域开发报告

日期：2026-08-23

## 1. 方法

双候选信赖域在每个候选事件中，从同一稳定模型出发生成两种200步优化结果：

- `full`：使用原始全量训练观测；
- `loo`：先执行25步全量/逐观测删一，以
  `u_i=L_val(leave-one-out_i)-L_val(full)` 估计边际效用，仅保留
  `u_i>0` 且未被父门控拒绝的观测。

随后在完全相同的验证帧、相机参数和渲染配置上测量：

```text
L_val(stable), L_val(full), L_val(loo)
```

最终发布验证 IoU loss 最低的几何；相同 loss 时依次优先 stable、full、loo。
当 LOO 保留观测少于2帧时不生成 LOO 候选，但仍在 stable 与 full 间选择。
评估基线使用同次重新生成的 full 候选，不再使用跨回放候选值。

该设计提供两个逐事件硬保证：

- 最终验证 IoU loss 不高于同次全量候选；
- 最终验证 IoU loss 不高于稳定模型。

实现位置：

- `scripts/stage4_influence.py::select_lowest_validation_geometry`；
- `scripts/replay_stage4_p2_geometry.py::dual_trust_record`；
- `scripts/evaluate_stage4_p2_influence.py`；
- `config/stage4_p2_dual_trust_region_v1.json`。

## 2. 开发集

开发范围保持为 `duplo_dude/01`、`toy_plane/02` 的 seed 59–64，共15个
候选事件，不加入 seed 65–67，也没有根据结果搜索 LOO 阈值。

| 指标 | 双候选 v1 |
|---|---:|
| 选择 stable / full / LOO | 4 / 7 / 4 |
| 相对 full 的退化事件 | 0/15 |
| 相对 stable 的退化事件 | 0/15 |
| 相对 full 严格改善事件 | 8/15（53.3%） |
| 平均 ΔIoU loss vs full | -0.000646 |
| bootstrap 95% CI | [-0.001211, -0.000202] |
| 平均位姿不一致性变化 | -0.000260 |
| 平均时序标准差变化 | +0.000699 |
| 平均不确定性变化 | -0.00000699 |
| LOO 边际效用开销 | 24.11 s/事件 |
| 完整双候选回放开销 | 约88 s/事件 |

按 seed 的平均 IoU loss 变化：

- seed 59：-0.001901；
- seed 60：-0.000935；
- seed 61：-0.000365；
- seed 62：0；
- seed 63：0；
- seed 64：-0.000998。

## 3. 结论

双候选机制修复了单候选信赖域的主要缺陷。平均 IoU 收益由
`+0.001217` 的退化变为 `-0.000646` 的改善，bootstrap 区间完全低于0，
所有事件相对 full 和 stable 都实现零退化。LOO 候选在4个事件中胜出，证明短步
边际效用并非完全无效；另外4个事件通过回滚稳定模型阻止了有害全量更新。

但按此前冻结的多指标标准，当前仍为 `pass=false`：

- 严格改善事件为8/15，低于2/3；
- seed 62、63为持平，不满足每个 seed 严格改善；
- 选择目标只最小化 IoU，导致平均时序标准差增加0.000699；
- 约88秒/事件的双候选开销不具备实时性。

因此该版本标记为 `development_partial_pass_not_frozen`，暂不接入在线发布，
也不启动新的独立 seed。它可以作为高质量离线教师与安全上界。

## 4. 下一优先级

下一版应保持双候选和同次配对不变，只把选择集合限制为“相对 full 的位姿与时序
均不退化”的候选，再在可行集合中选择最低 IoU；full 永远在可行集合中，因此仍
保持零退化保证。该多指标可行域公式必须先固定，再决定是否使用全新的独立 seed。
