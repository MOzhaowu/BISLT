# 阶段2：掩码不确定性收尾报告

日期：2026-08-17

## 冻结结论

阶段2已达到验收条件。最终方法为 **SAM 候选分歧 + Platt（L2 logistic）校准**。方法不改变 BIT 使用的 `masks[0]`，仅旁路保存三个候选、scores、logits 和四类不确定性并输出低质量掩码概率。

## 数据、标签与协议

- 8 个 MOPED 对象/序列，每序列 seed 41/42/43，共 24 次运行、132 个按真实帧号对齐的去重关键帧。
- 主标签：Mask IoU < 0.90（30/132）；辅助标签：Boundary-F1@2px < 0.70（28/132）。IoU < 0.50 无失败样本，仅保留兼容字段。
- 首帧记录 `temporal_missing=1`，分析特征以投影不一致性回退；冻结模型不依赖该回退项。
- 严格嵌套 LOSO：外层留出整个 `object/sequence`，三个 seed 始终同折；内层从 `{0.01,0.1,1,10,100}` 选择 L2，避免序列泄漏。

阈值在本阶段冻结。跨数据集测试必须沿用，或仅在独立训练集预注册新阈值。

## 结果

| 标签/模型 | OOF AUROC | ECE | NLL |
|---|---:|---:|---:|
| IoU<0.90，候选分歧 | **0.991** | 0.030 | 0.091 |
| IoU<0.90，+投影 | 0.982 | 0.055 | 0.147 |
| IoU<0.90，+时序回退 | 0.987 | 0.031 | 0.112 |
| IoU<0.90，+边界熵 | 0.987 | **0.025** | 0.111 |
| Boundary-F1<0.70，候选分歧 | **0.978** | **0.041** | **0.204** |
| Boundary-F1<0.70，+时序回退 | 0.971 | 0.060 | 0.246 |

未校准候选分歧的主标签 AUROC 为 0.998，但 ECE 0.230、NLL 0.384；Platt 校准将 ECE 降到 0.030、NLL 降到 0.091。

按 `object/sequence` 聚类的 10,000 次 bootstrap：主标签 AUROC 95% CI `[0.868, 1.000]`，辅助标签 `[0.921, 1.000]`。加入时序后的 AUROC 差分别为 `-0.0024 [-0.0152, 0.0072]` 和 `-0.0052 [-0.0168, 0.0047]`，均无稳定增益，因此冻结单特征方案。

## 交付物

- 采集：`examples/sam_utils.py`、`examples/mask_uncertainty.py`
- 帧对齐/GT 标签：`scripts/label_mask_uncertainty.py`
- 嵌套 LOSO：`scripts/fit_stage2_mask_nested_loso.py`
- 聚类 bootstrap：`scripts/bootstrap_stage2_mask.py`
- 数据与结果：`baseline/moped/stage2_final/`

当前只有 8 个独立序列，置信区间仍较宽。阶段3应独立研究位姿扰动/Hessian 不确定性；阶段4再通过预注册规则融合 `q_mask` 与 `q_pose`。
