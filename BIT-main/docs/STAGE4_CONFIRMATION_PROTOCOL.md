# 阶段4独立确认集冻结协议

冻结日期：2026-08-22

## 目的

本协议在查看确认集标签和方法结果之前冻结，用于验证P1因果相对特征是否具有
独立收益，并决定能否进入P2离线回放与阈值冻结。确认期间禁止修改特征公式、
阈值、标签和停止规则。

## 确认集

### 外部对象集

以下对象未参与阶段2/3和阶段4 P0/P1开发：

- `duster/00`
- `graphics_card/00`
- `orange_drill/00`
- `pouch/00`
- `remote/00`

每个序列运行seed 41/42/43，共15次。

### 问题序列定向确认

- `duplo_dude/01`
- `toy_plane/02`

使用开发阶段未使用的seed 44/45/46，共6次。该部分只验证预先识别的问题序列，
不替代外部对象泛化结果。

### 预注册扩展规则

完成上述21次运行后，若满足以下任一条件，则不查看方法优劣，直接追加外部5对象
的`02`序列、seed 41/42/43，共15次：

- 可对齐且通过5帧q_pose warmup的关键帧少于100；
- 安全关键帧少于20；
- 危险关键帧少于20。

危险标签固定为：

```text
mask_iou < 0.90 OR rotation_error > 5deg OR translation_error > 50mm
```

## 冻结模型

- `q_mask`：candidate disagreement单特征、全开发集冻结Platt模型；
- `q_pose`：阶段3冻结的`initial_median_covariance_trace`模型；
- 两个模型都输出失败概率，可靠度定义为`q=1-p_failure`。

## 固定方法与阈值

### 确认基线

```text
q_base = q_mask * q_pose
accept >= 0.2361967829315
reject < 0.05407334316641414
```

### 因果相对可见性

```text
q_vis_rel(t) = q_visibility(t) / max(q_visibility(<=t))
q_candidate = q_base * q_vis_rel
accept >= 0.23215984252580965
reject < 0.04907133706907316
```

### 因果相对信息量

```text
q_info_rel(t) = q_information(t) / max(q_information(<=t))
q_candidate = q_base * q_info_rel
accept >= 0.20693127944817635
reject < 0.04282146603854668
```

其中：

```text
q_visibility = geometric_mean(
    visible_boundary_ratio,
    effective_contour_ratio,
    1 - occlusion_ratio)
q_information = clip(min_view_angle_deg / 20deg, 0, 1)
```

运行内最大值只使用当前及历史关键帧，不允许使用未来帧。

## 固定验收标准

候选方法进入P2必须同时满足：

1. 确认集安全误拒率≤10%；
2. 确认集安全非接受率≤50%；
3. 危险保护率不低于`q_base`；
4. `duplo_dude/01`与`toy_plane/02`的危险保护率均不低于`q_base`；
5. 两个问题序列中至少一个严格改善，另一个不得退化；
6. 三个新seed的改善方向一致；
7. 序列bootstrap不显示安全误拒或危险保护的显著反向变化。

若无候选满足全部条件，则保持影子模式，不能进入P2阈值冻结。

## 禁止事项

- 不得根据确认集结果重新选择阈值或特征；
- 不得删除失败运行或不利对象；
- 不得把定向问题序列结果与外部对象结果混为同一泛化结论；
- 若发生工程故障，只允许修复数据产生或对齐错误，修复前后运行必须保留记录。
