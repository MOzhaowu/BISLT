# Stage 4 P3 扩展教师数据质量报告

## 1. 结论

seed 71–73 的冻结教师流水线已完成，新增数据已与 seed 59–64 的原有教师数据合并。合并集覆盖 8 个对象/序列组、41 个教师事件和 107 个观测。

当前合并集 **不满足学生模型开发准入条件**。七项预注册门控中六项通过；唯一失败项是事件级 `full_harmful_vs_stable` 标签的序列内混合度：只有 2 个序列同时包含 full-safe 与 full-harmful 事件，低于要求的 4 个序列。

因此，当前数据可以作为冻结的扩展教师数据版本和后续补样依据，但不应据此选择学生模型、正则化强度或阈值。

## 2. 冻结产物

- 原有教师集：`baseline/moped/stage4_p3_distillation/development_s59_64/teacher_events.jsonl`
  - SHA-256：`a8906adf8c31a64123682f1a730cd247a0d7b3ad5727a2086f8d339b0268f30e`
- 新增教师集：`baseline/moped/stage4_p3_distillation/expansion_s71_73/teacher_events.jsonl`
  - SHA-256：`0118cc4aed41c075b9338f85d7bdf791bbcfb4bf77d6daf78f240f8b0719ab67`
- 八序列合并集：`baseline/moped/stage4_p3_distillation/combined_s59_73/teacher_events.jsonl`
  - SHA-256：`205790ec216aeb49f6586d632bc432ebc429e4d1788ca388f70f25fe595c8a2e`
- 机器可读审核：`baseline/moped/stage4_p3_distillation/combined_s59_73/quality_audit.json`

冻结流水线于 2026-08-24 17:16:22 +08:00 完成。新增教师采用 200 步受约束投影和冻结的 `p1v2_short_loo_control_iou_distillation_expansion` 策略，教师选择为 stable/full/LOO = 6/9/11。

## 3. 总体统计

| 指标 | 结果 |
|---|---:|
| 序列组 | 8 |
| 教师事件 | 41 |
| 观测 | 107 |
| 具备在线特征的观测 | 62 |
| 特征可用率 | 57.94% |
| keep/drop | 75/32 |
| full-harmful/full-safe | 15/26 |
| stable/full/LOO | 7/14/20 |
| 开发种子 | 59–64、71–73 |
| 锁定种子泄漏 | 0 |

特征缺失主要来自协议显式保留的 anchor 观测；数据集没有在构建阶段进行插补，后续只能在每个 LOSO 训练折内部学习插补统计量。

## 4. 预注册门控

| 门控 | 实际值 | 要求 | 结果 |
|---|---:|---:|---|
| 预注册序列全部存在 | 8/8 | 8/8 | 通过 |
| 有事件的序列组 | 8 | ≥6 | 通过 |
| 总序列组 | 8 | ≥8 | 通过 |
| 观测数 | 107 | ≥100 | 通过 |
| 同时含 keep/drop 的序列 | 7 | ≥4 | 通过 |
| 同时含 full-safe/full-harmful 的序列 | 2 | ≥4 | **未通过** |
| seed 68–70、74–79 泄漏 | 0 | 0 | 通过 |

事件级标签混合只出现在 `duplo_dude/01` 和 `toy_plane/02`。新增序列中：

- `black_drill/00`、`cheezit/00`、`cheezit/02` 全部为 full-harmful；
- `cheezit/04`、`duplo_dude/02`、`toy_plane/03` 全部为 full-safe。

这意味着严格 sequence-LOSO 的部分测试折只有单一事件标签，且训练分布容易通过序列身份间接决定标签。此时较高的总体样本数不能弥补事件级标签与序列的混杂。

## 5. 分序列统计

| 序列 | 事件 | 观测 | keep/drop | full-harmful/full-safe |
|---|---:|---:|---:|---:|
| black_drill/00 | 3 | 9 | 5/4 | 3/0 |
| cheezit/00 | 3 | 9 | 6/3 | 3/0 |
| cheezit/02 | 5 | 15 | 11/4 | 5/0 |
| cheezit/04 | 3 | 6 | 6/0 | 0/3 |
| duplo_dude/01 | 6 | 13 | 11/2 | 3/3 |
| duplo_dude/02 | 6 | 18 | 11/7 | 0/6 |
| toy_plane/02 | 9 | 19 | 14/5 | 1/8 |
| toy_plane/03 | 6 | 18 | 11/7 | 0/6 |

## 6. 后续处理约束

1. 保留当前合并集、质量审核及全部源哈希，不覆盖或重选 seed 71–73 结果。
2. 暂停学生模型、正则化和阈值选择，避免在标签与序列混杂的数据上过拟合。
3. 下一批补样应以“在至少两个当前单标签序列内获得相反事件标签”为目标，但不能通过查看锁定 seed 74–79 后再改变公式或选择序列。
4. 在补样协议预注册前，不开放 seed 74–76 公式确认集和 seed 77–79 外部独立集。
5. 补样后重新运行相同审核；只有所有门控通过，才进入嵌套 sequence-LOSO 学生开发。

## 7. 可复现命令

```bash
/root/miniconda3/envs/BIT_Track/bin/python \
  BIT-main/scripts/audit_stage4_p3_teacher_dataset.py \
  baseline/moped/stage4_p3_distillation/development_s59_64/teacher_events.jsonl \
  baseline/moped/stage4_p3_distillation/expansion_s71_73/teacher_events.jsonl \
  --base-protocol BIT-main/config/stage4_p3_distillation_protocol.json \
  --expansion-protocol BIT-main/config/stage4_p3_distillation_expansion_protocol.json \
  --combined-output baseline/moped/stage4_p3_distillation/combined_s59_73/teacher_events.jsonl \
  --output baseline/moped/stage4_p3_distillation/combined_s59_73/quality_audit.json
```
