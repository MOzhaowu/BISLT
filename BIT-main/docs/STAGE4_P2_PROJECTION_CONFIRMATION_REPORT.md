# 阶段4 P2：受约束投影200步确认与独立验证冻结报告

日期：2026-08-24

## 1. 冻结结论

控制IoU对齐的多指标受约束梯度投影已经完成预注册的两级确认：先在
seed 59--64上运行200步、每步刷新位姿扰动，再在未用于方法选择的
seed 68--70上进行独立验证。开发确认和独立验证的全部预注册验收项均通过。

阶段判定为：

- P2离线教师方法冻结；
- 固定候选集合为stable、full、LOO；
- 固定选择规则为先剔除位姿或时序退化候选，再选择控制帧IoU loss最低者；
- stable始终作为安全回退；
- 允许进入实时蒸馏和P3在线影子记录；
- 当前方法不得直接作为实时在线策略。

最终机器可读结论为`independent_validation_pass`。

## 2. 方法与数据隔离

候选优化固定为：

```text
training_iou + 0.1 * laplacian
+ control_iou
+ pose_hinge
+ temporal_hinge
```

实际Adam更新方向通过20轮Dykstra循环投影到以下下降半空间：

```text
g_control_iou dot d >= 0
g_pose dot d >= 0       （位姿约束激活时）
g_temporal dot d >= 0   （时序约束激活时）
```

开发确认使用`duplo_dude/01`和`toy_plane/02`的seed 59--64。只有开发确认
通过后，流水线才采集并打开seed 68--70。独立数据未参与公式、权重、阈值、
投影轮数或候选选择规则的修改。

200步实验固定参数：

- `proposal_steps=200`；
- `pose_probe_interval=1`，每个优化步重新评估12个SE(3)轴向扰动并选择探针；
- control IoU、pose hinge、temporal hinge权重均为1；
- 约束激活阈值`1e-8`；
- 投影可行容差`1e-7`；
- 20轮Dykstra投影；
- 控制帧参与候选优化，因此seed 59--64只属于开发证据。

## 3. seed 59--64开发确认

| 指标 | 200步确认结果 |
|---|---:|
| 候选事件 | 15 |
| stable/full/LOO选择 | 1/5/9 |
| 严格改善事件 | 10/15（66.7%） |
| 严格改善seed | 6/6 |
| 平均Delta IoU loss | -0.004572 |
| IoU bootstrap 95% CI | [-0.006874, -0.002462] |
| 平均Delta位姿不一致性 | -0.000861 |
| 平均Delta时序标准差 | -0.002958 |
| LOO可行候选 | 9/9 |
| 优化步数 | 1800 |
| 实际投影步数 | 45 |
| 一阶投影违规 | 0/1800 |

全部12项多指标验收条件通过，协议、回放和评估SHA-256已写入冻结JSON。

## 4. seed 68--70独立验证

| 指标 | 独立结果 |
|---|---:|
| 配对候选事件 | 7 |
| stable/LOO选择 | 4/3 |
| 严格改善事件 | 7/7（100%） |
| 严格改善seed | 3/3 |
| seed 68平均Delta IoU loss | -0.003077 |
| seed 69平均Delta IoU loss | -0.005255 |
| seed 70平均Delta IoU loss | -0.007883 |
| 总体平均Delta IoU loss | -0.005384 |
| IoU bootstrap 95% CI | [-0.008072, -0.002944] |
| 平均Delta位姿不一致性 | -0.000174 |
| 平均Delta时序标准差 | 0.000000 |
| 平均Delta扰动不确定性 | +0.001026 |
| 优化步数 | 800 |
| 实际投影步数 | 22 |
| 一阶投影违规 | 0/800 |

独立集7个事件均相对同次正常full候选改善。其中3个事件提交LOO候选，4个事件
回退stable。这表明收益来自“可行LOO改进 + 对有害full更新的安全回滚”两部分，
不能把7/7全部解释为LOO本身优于full。

独立集全部12项验收条件通过：IoU、位姿和时序逐事件零退化，bootstrap IoU
区间上界小于0，全部seed严格改善，安全误拒率为0，安全非接受率为44.4%。

## 5. 可复现性与冻结产物

入口命令：

```bash
CUDA_VISIBLE_DEVICES=0 \
BIT_PYTHON=/root/miniconda3/envs/BIT_Track/bin/python \
bash scripts/run_stage4_p2_projection_confirmation.sh
```

代码、协议与审计：

- `scripts/stage4_candidate_regularization.py`；
- `scripts/replay_stage4_p2_geometry.py`；
- `scripts/apply_stage4_p2_multimetric_trust_region.py`；
- `scripts/evaluate_stage4_p2_influence.py`；
- `scripts/finalize_stage4_p2_projection.py`；
- `scripts/run_stage4_p2_projection_confirmation.sh`；
- `config/stage4_p2_control_iou_projection_protocol.json`；
- `config/stage4_p2_control_iou_projection_frozen.json`。

最终冻结哈希：

```text
protocol   3054248cc0277e06e7804649089b0232adfb932b4e4801a5406f94a1c0405c34
dev replay 06c0a3ae7dc9eb665de0d65af1342d2f8338ba7996561fcec77757316fdaaa7e
ind replay 36a2b0af13ec6a85b39b0a23ca9a0d094b8ed4aacdb4bdffacd708d7f084aedf
ind eval   66803bf4beda20eae0addd2b8cb80c0cbc2d290ffb4bc2828aa238c7bf2acf55
```

预注册协议中的`status=screening_pass_pending_200_step_confirmation`保留不改，
因为修改该文件会破坏预注册哈希。最终状态由冻结JSON和独立评估JSON记录。

## 6. 局限与下一步

本轮只覆盖两个问题序列、三个独立seed和7个候选事件。独立集时序标准差均为0，
可以证明无退化，但不能充分证明复杂时序条件下的稳定收益。扰动不确定性平均增加
0.001026，虽不属于冻结硬约束，仍需在后续论文实验中报告。

短步LOO边际效用在独立实验中的平均耗时约24.3秒/事件，因此本方法冻结为离线
教师而不是在线算法。下一优先级是实时蒸馏：用低成本`q_mask`、`q_pose`、
Hessian、可见性、轮廓残差和历史变化特征预测教师的stable/full/LOO选择与
多指标可行性。学生模型通过全新独立seed验证且在线额外开销低于5%后，才进入
P3影子记录；P3通过后再进入P4在线闭环。
