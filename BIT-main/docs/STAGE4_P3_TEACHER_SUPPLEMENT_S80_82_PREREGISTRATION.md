# Stage 4 P3 教师补样预注册（seed 80–82）

## 目的

当前八序列教师集已达到序列数和观测数要求，但事件级 `full_harmful_vs_stable` 标签仍与序列强混杂：只有 `duplo_dude/01` 和 `toy_plane/02` 同时包含 full-safe 与 full-harmful。补样的唯一主要目标是让至少两个当前单标签序列获得反类事件，使混合标签序列总数从 2 增加到至少 4。

本文件和 `config/stage4_p3_teacher_supplement_s80_82_protocol.json` 必须在任何补样运行开始前冻结。seed 74–76 仍是公式确认集，seed 77–79 仍是外部独立集，补样不得读取或生成这些种子的结果。

## 固定设计

- 开发种子：80、81、82。
- 序列：`black_drill/00`、`cheezit/00`、`cheezit/02`、`cheezit/04`、`duplo_dude/02`、`toy_plane/03`。
- 总运行数：6 × 3 = 18。
- 18 组必须全部运行，不允许观察中间标签后提前停止。
- 所有成功运行均纳入；零事件运行保留且不得换 seed。
- 技术失败只能用完全相同的序列和 seed 重试。
- 补样期间禁止训练学生模型或查看 seed 74–79。

教师仍使用冻结的 25 步双候选参考和 200 步受约束投影，不改变 stable/full/LOO 定义、验证损失、正则权重或 `full_harmful_vs_stable` 标签定义。协议 JSON 已记录相关脚本和配置哈希。

## 预先固定的反类目标

| 当前状态 | 序列 | 补样后所需反类 |
|---|---|---|
| harmful-only | black_drill/00、cheezit/00、cheezit/02 | 至少一个 full-safe 事件 |
| safe-only | cheezit/04、duplo_dude/02、toy_plane/03 | 至少一个 full-harmful 事件 |

不会根据补样表现删除难以翻转的序列，也不会只汇报成功翻转的种子。

## 准入标准

补样完成后，将 seed 80–82 与冻结的 seed 59–64、71–73 教师数据合并，并重新执行相同质量审核。只有同时满足以下条件，才可进入嵌套 sequence-LOSO 学生开发：

1. 至少两个当前单标签序列获得反类事件；
2. 总混合 full-safe/full-harmful 序列不少于 4；
3. 18 个计划运行全部有明确状态；
4. 不包含 seed 68–70、74–79；
5. 原有序列数、观测数和 keep/drop 混合度门控继续通过。

如任一条件失败，应如实发布失败结论。进一步补样必须另写协议，不能顺延 seed，也不能开放 seed 74–79。

## 因果特征约束

anchor 和每个运行前 5 帧的 pose calibration warmup 可以显式缺失在线特征。不得使用后续帧计算的初始中位数回填 warmup 帧。所有缺失值插补只能在准入通过后的 LOSO 训练折内部完成，并必须同时保留缺失指示。

## 当前冻结依据

- 父合并集：41 个事件、107 个观测、8 个序列；SHA-256 `205790ec216aeb49f6586d632bc432ebc429e4d1788ca388f70f25fe595c8a2e`。
- 父质量审核 SHA-256：`74b202202a96a9edffab446d6fc9b12ca03f27da885eadaaf3bbec83e1e2ba27`。
- 预注册时 P3 目录中不存在 seed 74–79 的捕获或评价文件。
