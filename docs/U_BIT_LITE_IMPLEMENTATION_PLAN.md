# U-BIT Lite 方案说明与实施文档

## 0. 文档信息

| 项目 | 内容 |
| --- | --- |
| 文档名称 | U-BIT Lite 方案说明与实施文档 |
| 适用项目 | BIT - Prior-free 3D Object Tracking |
| 项目路径 | `/Users/bytedance/Desktop/BIT-main` |
| 详细设计 | `docs/U_BIT_LITE_PHASE1_DETAILED_DESIGN.md` |
| 方案版本 | 1.0 |
| 目标读者 | 算法工程师、C++ 工程师、测试人员、项目负责人、论文实验负责人 |
| 建议周期 | MVP 7-10 个工作日；完整交付 3-4 周 |
| 核心原则 | 最小改动、严格评测、无需训练、可回滚、可复现实验 |

---

## 1. 项目概述

### 1.1 项目背景

BIT 是一个面向未知刚性物体的单目 RGB 6DoF 跟踪与在线几何生成系统。系统不要求事先提供目标物体 CAD 模型，也不要求针对目标对象进行姿态标注训练。现有实现由以下两个闭环模块组成：

1. C++ 位姿优化模块：基于 Summer/SLCT 的颜色直方图、轮廓搜索线和可微解析优化进行位姿跟踪。
2. Python 几何生成模块：使用 SAM 生成目标轮廓，通过 Soft Rasterizer 优化固定拓扑网格，再将生成网格返回 C++ 跟踪器。

现有闭环能够逐步改善模型和跟踪结果，但所有观测基本被等权使用。错误分割、低质量姿态或不充分视角可能直接进入几何优化，并在后续跟踪中形成误差自增强。此外，当前评测流程包含每帧 GT 输入和失败后 GT 恢复逻辑，不适合作为严格的长期自主跟踪评测协议。

### 1.2 建设目标

本项目建设一个轻量级不确定性感知版本 U-BIT Lite，在不更换核心跟踪器、不训练新模型、不引入复杂三维表示的前提下，实现：

1. 建立只在首帧使用 GT 的严格评测基线。
2. 修复 C++/Python 磁盘通信的批次完整性问题。
3. 利用 SAM 多候选结果、SLCT 概率图和渲染轮廓计算观测置信度。
4. 对几何优化进行帧级加权，并拒绝明显不可靠的观测。
5. 在更新后误差恶化时回滚到最佳网格。
6. 使用置信度增强现有关键帧选择策略。
7. 输出完整日志、指标、消融结果和可复现实验配置。

### 1.3 项目范围

#### 范围内

- `examples/run_example.py` 几何优化流程。
- `examples/sam_utils.py` SAM 候选选择与置信度计算。
- `examples/communication.py` 批次读取与完整性校验。
- `tracker/example/eg_BIT.cpp` 评测协议和关键帧筛选。
- `tracker/interface/communication.*` 完成标记与元数据输出。
- `tracker/interface/tracker_summer/*` GT 使用边界和跟踪置信度输出。
- 配置文件生成、日志、指标统计、部署和实验脚本。
- RBOT、MOPED 上的基线与消融实验。

#### 范围外

- 将网格整体替换为 NeRF、SDF 或 3D Gaussian Splatting。
- 训练新的分割、姿态或置信度网络。
- 支持非刚性物体、多目标跟踪或机器人主动视角控制。
- 重写 Summer/SLCT 的主体跟踪算法。
- 将本地磁盘通信整体替换为共享内存或网络 RPC。
- 在本阶段承诺顶会论文录用或固定性能提升比例。

### 1.4 成功定义

项目成功需同时满足：

- 严格模式下除首帧初始化外不使用 GT 影响跟踪状态。
- 每个 Python 批次的数据类型、索引和数量完全一致。
- 低置信度观测不会直接污染几何模型。
- 所有改进均可通过配置开关独立启用和关闭。
- 在至少一个主数据集上取得稳定的跟踪或几何指标提升。
- 相同配置重复运行的主要指标波动处于验收范围。

---

## 2. 需求分析

### 2.1 功能需求

#### FR-01 严格评测模式

- 增加 `evaluation.strict_mode` 配置。
- 严格模式仅允许第 0 帧或明确指定初始化帧使用 GT。
- 跟踪失败后不得使用当前帧 GT 恢复。
- GT 仍可用于离线计算误差，但不得反馈到跟踪状态。
- 日志必须记录初始化来源和每次恢复来源。

#### FR-02 通信批次完整性

- C++ 完成同一帧所有文件写入后，最后原子创建 `<frame_id>.done`。
- Python 仅处理存在 `.done` 的帧。
- Python 按帧 ID 求所有模态文件的交集，不再只按 `K` 文件数判断完成。
- 批次应包含 `rgb/prob/mask/pose/K/origin_roi/target_roi/target_size`。
- 文件缺失、为空或无法解析时，当前帧不得进入优化。

#### FR-03 SAM 多候选 Mask 选择

- 使用 SAM `multimask_output=True` 的所有候选。
- 综合 SAM score、SLCT 概率图一致性、当前网格渲染轮廓一致性选择候选。
- 输出最优 Mask、候选评分、候选间分歧和最终分割置信度。
- 当全部候选低于阈值时，将帧标记为不可用于几何更新。

#### FR-04 观测置信度

每帧计算以下分量：

- `sam_confidence`：SAM 候选质量。
- `tracker_confidence`：SLCT 前景概率和轮廓残差质量。
- `view_confidence`：视角新颖性和历史覆盖情况。
- `render_confidence`：观测 Mask 与当前模型渲染轮廓的一致性。
- `frame_weight`：上述分量的融合权重。

所有分量应归一化到 `[0, 1]`，并写入结构化日志。

#### FR-05 加权几何优化

- 几何损失按帧权重加权。
- 启用已有但当前未加入总损失的 flatten loss。
- 支持低置信度帧剔除。
- 权重总和过低或有效帧不足时跳过当前几何更新。
- 保留原始 BIT 损失作为可切换基线。

#### FR-06 模型更新门控与回滚

- 每轮优化前保存当前模型状态。
- 优化过程中保存验证损失最优状态。
- 更新后在保留帧上重新计算轮廓误差。
- 若更新后误差恶化超过阈值，则拒绝新模型并回滚。
- 记录接受或拒绝原因。

#### FR-07 置信度关键帧筛选

- 保留现有视角夹角约束。
- 增加跟踪置信度、轮廓质量和视角覆盖判断。
- 支持强制最小关键帧数量，防止过度过滤。
- 记录每个候选帧的分数和拒绝原因。

#### FR-08 实验与指标输出

- 输出逐帧姿态、姿态误差、Mask 质量、置信度和处理耗时。
- 输出逐轮几何损失、模型接受状态和网格路径。
- 支持基线、单模块消融和完整方案配置。
- 汇总 ADD、ADD-S、AUC、旋转误差、平移误差、成功率和耗时。

### 2.2 非功能需求

#### NFR-01 性能

- 置信度计算不应使单帧 SAM 阶段耗时增加超过 20%。
- 除已有优化外，不引入额外网络推理。
- 日志写入不得阻塞跟踪主循环超过 10 ms/帧。

#### NFR-02 可靠性

- 不完整批次不得导致 `np.stack` 崩溃。
- C++ 或 Python 任一进程异常退出后，残留文件不得被误识别为完整批次。
- 模型更新失败时系统应继续使用上一个有效模型。

#### NFR-03 可复现性

- Python、NumPy、PyTorch 和 CUDA 随机种子统一配置。
- 每次实验保存完整配置副本、代码版本标识和环境信息。
- 相同环境下三次运行的 ADD(-S) AUC 标准差目标不超过 1 个百分点。

#### NFR-04 可配置性

- 新增功能必须通过 YAML 配置控制。
- 默认兼容原始配置；缺少新字段时使用保守默认值。
- 不允许将实验阈值散落硬编码在多个源码文件。

#### NFR-05 可维护性

- Python 新增逻辑使用小型纯函数和 `dataclass`。
- C++ 新增结构保持 POD 风格，避免跨模块复杂依赖。
- 日志采用 JSONL 或 CSV，不解析控制台文本作为实验数据。

#### NFR-06 兼容性

- 目标环境为 Ubuntu 20.04/22.04、Python 3.10、CUDA 11.8。
- 保持 PyTorch、Soft Rasterizer、OpenCV 4.x、Qt 5.x 现有依赖。
- 不要求修改数据集原始目录结构。

---

## 3. 系统架构设计

### 3.1 整体架构

```text
                        +-------------------------+
RGB Sequence ---------->| C++ Summer/SLCT Tracker |
                        +------------+------------+
                                     |
                         pose/prob/mask/ROI/K
                                     |
                        +------------v------------+
                        | Atomic Batch Publisher  |
                        | data files + .done      |
                        +------------+------------+
                                     |
                        +------------v------------+
                        | Python Batch Consumer   |
                        | validation/alignment    |
                        +------------+------------+
                                     |
              +----------------------+----------------------+
              |                                             |
    +---------v---------+                         +---------v---------+
    | SAM Mask Selector |                         | Confidence Fusion |
    | multi-mask score  |                         | frame weight      |
    +---------+---------+                         +---------+---------+
              |                                             |
              +----------------------+----------------------+
                                     |
                        +------------v------------+
                        | Weighted Mesh Optimizer |
                        | IoU/Lap/Flat/Temporal   |
                        +------------+------------+
                                     |
                        +------------v------------+
                        | Update Gate & Rollback  |
                        +------------+------------+
                                     |
                               accepted mesh
                                     |
                        +------------v------------+
                        | C++ Model Replacement   |
                        +-------------------------+
```

### 3.2 技术选型

| 领域 | 技术 | 选择原因 |
| --- | --- | --- |
| 语言 | Python 3.10 + C++17 | 沿用现有实现，避免重构 |
| 深度学习 | PyTorch + CUDA 11.8 | 现有 Soft Rasterizer 和 SAM 依赖 |
| 分割 | SAM ViT-B | 已集成，无新增训练成本 |
| 几何表示 | 固定拓扑三角网格 | 与现有跟踪器接口直接兼容 |
| 可微渲染 | Soft Rasterizer | 已具备 CUDA 扩展和项目实现 |
| 位姿跟踪 | Summer/SLCT | 保持现有论文基线 |
| 配置 | OpenCV YAML 兼容 YAML | 同时服务 C++ 和 Python |
| 进程通信 | 文件批次 + 完成标记 | 对现有方案最小修改 |
| 实验日志 | JSONL + CSV 汇总 | 便于增量写入和统计分析 |

### 3.3 模块划分

#### M1 StrictEvaluation

负责 GT 使用边界、初始化策略和评测状态隔离。

#### M2 BatchProtocol

负责数据原子发布、帧索引对齐、完整性校验和异常恢复。

#### M3 MaskSelector

负责 SAM 多候选推理、候选评分、Mask 选择和分割置信度。

#### M4 ConfidenceEstimator

负责跟踪、视角、渲染和分割置信度的归一化与融合。

#### M5 WeightedGeometryOptimizer

负责加权损失、有效帧筛选、最佳状态保存和优化统计。

#### M6 ModelUpdateGate

负责模型接受、拒绝、回滚和输出版本管理。

#### M7 KeyframeSelector

负责 C++ 侧低成本关键帧预筛选，以及 Python 侧最终质量筛选。

#### M8 ExperimentRecorder

负责配置快照、逐帧日志、运行摘要和指标汇总。

---

## 4. 详细设计

### 4.1 配置设计

建议在生成的任务 YAML 中增加以下字段：

```yaml
evaluation:
  strict_mode: true
  init_frame: 0
  allow_gt_recovery: false

communication:
  root: "/path/to/cmc/"
  protocol_version: 2
  require_done_marker: true
  batch_timeout_seconds: 30

confidence:
  enabled: true
  min_frame_weight: 0.25
  min_valid_frames: 2
  sam_weight: 0.35
  tracker_weight: 0.25
  render_weight: 0.25
  view_weight: 0.15

mask_selector:
  use_all_sam_candidates: true
  sam_score_weight: 0.45
  tracker_iou_weight: 0.30
  render_iou_weight: 0.25
  min_mask_score: 0.30

geometry_optimization:
  weighted_loss: true
  learning_rate: 0.01
  laplacian_weight: 0.10
  flatten_weight: 0.01
  temporal_weight: 0.02
  rollback_enabled: true
  max_relative_degradation: 0.03

keyframe:
  confidence_enabled: true
  min_angle_degrees: 3.0
  min_tracking_confidence: 0.25
  min_combined_score: 0.30
  force_accept_after_frames: 30

logging:
  run_dir: ""
  frame_jsonl: "frames.jsonl"
  optimization_jsonl: "optimizations.jsonl"
  save_debug_images: false
```

配置读取应支持字段缺失，旧配置默认表现为原始 BIT 模式。

### 4.2 严格评测模块

#### 4.2.1 状态规则

```text
frame == init_frame:
    tracker_pose <- ground_truth_pose
frame > init_frame:
    tracker_pose <- previous_estimated_pose
tracking_failure:
    keep predicted pose / enter lost state
    never assign current ground truth pose
```

#### 4.2.2 C++ 接口

建议在 `StdData` 或跟踪器配置中增加：

```cpp
struct EvaluationPolicy {
    bool strict_mode{true};
    int init_frame{0};
    bool allow_gt_recovery{false};
};
```

建议新增：

```cpp
void SetEvaluationPolicy(const EvaluationPolicy& policy);
bool CanUseGroundTruthForInitialization(int frame_index) const;
```

严格模式下，`TrackerBase::IsSuccess()` 可以继续读取 GT 计算离线成功率，但其返回值不得触发 GT 姿态写回。为避免误用，推荐将其拆分为：

```cpp
PoseError EvaluatePoseError(const cv::Matx44f& estimate,
                            const cv::Matx44f& ground_truth);
bool IsEstimateWithinThreshold(const PoseError& error);
```

### 4.3 通信协议设计

#### 4.3.1 文件布局

```text
cmc/
  img/0001.png
  prob/0001.png
  mask/0001.png
  pose/0001.pose
  K/0001.K
  originRoi/0001.roi
  targetRoi/0001.roi
  targetSize/0001.size
  ready/0001.done
```

#### 4.3.2 发布流程

1. C++ 将每个文件写入同目录临时文件，例如 `0001.pose.tmp`。
2. 写入成功后执行 rename 为正式文件。
3. 校验所有正式文件均存在且非空。
4. 最后写入 `ready/0001.done.tmp`。
5. rename 为 `ready/0001.done`，作为批次提交点。

同一文件系统内 rename 应视为原子操作。`.done` 内容建议为：

```json
{"version":2,"frame_id":1,"files":8}
```

#### 4.3.3 Python 消费接口

```python
@dataclass(frozen=True)
class FrameBatchPaths:
    frame_id: int
    rgb: str
    probability: str
    mask: str
    pose: str
    intrinsics: str
    origin_roi: str
    target_roi: str
    target_size: str
    done: str

def scan_complete_frames(root: str) -> list[int]: ...
def resolve_frame_paths(root: str, frame_id: int) -> FrameBatchPaths: ...
def validate_frame_batch(paths: FrameBatchPaths) -> list[str]: ...
def load_frame_batch(paths: FrameBatchPaths) -> "FrameObservation": ...
```

解析失败返回结构化错误，不直接调用 `np.stack`。批次构建时按 `frame_id` 排序，不依赖不同目录的列表切片位置。

### 4.4 数据结构设计

#### 4.4.1 Python 观测结构

```python
@dataclass
class FrameObservation:
    frame_id: int
    rgb: np.ndarray
    tracker_probability: np.ndarray
    tracker_mask: np.ndarray
    pose: np.ndarray
    intrinsics: np.ndarray
    origin_roi: np.ndarray
    target_roi: np.ndarray
    target_size: np.ndarray

@dataclass
class ConfidenceComponents:
    sam: float
    tracker: float
    render: float
    view: float
    combined: float

@dataclass
class SelectedMask:
    mask: np.ndarray
    candidate_index: int
    candidate_scores: list[float]
    disagreement: float
    confidence: float

@dataclass
class WeightedObservation:
    observation: FrameObservation
    selected_mask: SelectedMask
    confidence: ConfidenceComponents
    accepted: bool
    rejection_reason: str | None
```

#### 4.4.2 C++ 关键帧质量结构

```cpp
struct FrameQuality {
    int frame_id{-1};
    float angle_novelty{0.0f};
    float tracking_confidence{0.0f};
    float silhouette_quality{0.0f};
    float combined_score{0.0f};
    bool accepted{false};
};
```

第一期可只在 C++ 输出 `tracking_confidence` 和 `angle_novelty`，最终 `frame_weight` 在 Python 计算，降低跨语言改动成本。

### 4.5 Mask 选择算法

对第 `i` 个 SAM 候选计算：

```text
sam_quality_i     = clamp(SAM score_i, 0, 1)
tracker_iou_i     = IoU(mask_i, binarized tracker probability)
render_iou_i      = IoU(mask_i, rendered current mesh silhouette)
candidate_score_i = a*sam_quality_i + b*tracker_iou_i + c*render_iou_i
```

候选分歧：

```text
disagreement = mean(1 - IoU(mask_i, mask_j)), i != j
```

最终置信度：

```text
sam_confidence = best_candidate_score * (1 - disagreement)
```

边界情况：

- 首轮没有可用模型轮廓时，将 `render_iou_weight` 重新分配给其他项。
- SLCT 概率图为空时，仅使用 SAM 和渲染一致性。
- 所有候选低于阈值时保留最高分 Mask 用于可视化，但不参与几何更新。
- Mask 面积过小、过大或触碰 ROI 边界时施加惩罚。

### 4.6 置信度融合设计

#### 4.6.1 分量定义

```text
sam_confidence:
    来自最佳候选分数和候选间分歧。

tracker_confidence:
    前景区域平均 |p - 0.5| * 2，
    并乘以有效轮廓搜索线比例。

render_confidence:
    观测 Mask 与当前模型渲染轮廓的 IoU，
    首轮使用中性值 0.5。

view_confidence:
    与已有关键视角最小夹角的归一化值，
    并限制在 [0, 1]。
```

#### 4.6.2 融合

首版采用加权几何平均，避免某个极低分量被其他高分掩盖：

```text
frame_weight =
    sam_confidence^ws
  * tracker_confidence^wt
  * render_confidence^wr
  * view_confidence^wv
```

实现时使用 `epsilon=1e-6` 防止数值问题，并将权重重新缩放到当前批次均值约为 1，避免有效学习率随批次质量剧烈变化。

### 4.7 加权几何优化

#### 4.7.1 损失函数

```text
L_mask =
    sum_t(frame_weight_t * SoftIoU(render_t, mask_t))
    / sum_t(frame_weight_t)

L_total =
    L_mask
  + lambda_lap * L_laplacian
  + lambda_flat * L_flatten
  + lambda_temp * ||V_current - V_previous||_2^2
```

`L_temporal` 仅约束相邻几何更新轮次，不约束同一轮优化迭代。

#### 4.7.2 优化前置条件

满足以下条件才启动优化：

- 有效观测数量不少于 `min_valid_frames`。
- 有效权重总和大于最小阈值。
- 至少一个观测具备足够视角新颖性。
- 所有姿态矩阵、内参和 Mask 通过数值校验。

#### 4.7.3 最佳状态

```python
best_loss = float("inf")
best_state = deepcopy(model.state_dict())

for step in range(iteration_nums):
    loss = ...
    if is_finite(loss) and loss < best_loss:
        best_loss = loss.item()
        best_state = deepcopy(model.state_dict())

model.load_state_dict(best_state)
```

必须拒绝 NaN、Inf、网格顶点越界和退化三角形比例异常的状态。

### 4.8 更新门控设计

将当前批次拆分为优化集和门控集。数据较少时可采用 leave-one-out 轮换；MVP 可固定保留最低 20% 帧作为门控集。

```text
relative_degradation =
    (new_gate_loss - old_gate_loss) / max(old_gate_loss, epsilon)

accept if:
    relative_degradation <= max_relative_degradation
    and invalid_face_ratio <= threshold
    and vertex_extent is valid
```

模型文件版本：

```text
_candidate_03.obj
_accepted_03.obj
_rejected_03.obj       # 可选，仅调试保留
latest_accepted.obj
```

通过临时文件和 rename 更新 `latest_accepted.obj`，避免 C++ 读取半写入网格。

### 4.9 关键帧筛选设计

MVP 阶段计算：

```text
angle_score    = clamp(min_angle / target_angle, 0, 1)
tracking_score = foreground probability certainty
quality_score  = angle_score * tracking_score
```

接受规则：

```text
accept =
    not reference_frame
    and angle >= min_angle
    and tracking_score >= min_tracking_confidence
    and quality_score >= min_combined_score
```

如果连续 `force_accept_after_frames` 帧均未接受，则接受其中质量最高的一帧，避免几何更新永久饥饿。

第二阶段再加入 Python 返回的视角覆盖和渲染一致性，不作为 MVP 阻塞项。

### 4.10 日志与结果接口

#### 帧日志 `frames.jsonl`

```json
{
  "frame_id": 12,
  "pose_source": "tracker",
  "sam_confidence": 0.82,
  "tracker_confidence": 0.74,
  "render_confidence": 0.69,
  "view_confidence": 0.61,
  "frame_weight": 0.72,
  "accepted_for_geometry": true,
  "processing_ms": 38.4
}
```

#### 优化日志 `optimizations.jsonl`

```json
{
  "round": 2,
  "candidate_frames": 6,
  "valid_frames": 4,
  "loss_before": 0.321,
  "best_train_loss": 0.207,
  "gate_loss_before": 0.346,
  "gate_loss_after": 0.301,
  "accepted": true,
  "mesh_path": "_accepted_02.obj"
}
```

#### 运行目录

```text
runs/<dataset>/<object>/<timestamp>/
  config.yml
  environment.txt
  frames.jsonl
  optimizations.jsonl
  metrics.json
  meshes/
  debug/                 # 可选
```

---

## 5. 实施计划

### 5.1 阶段划分

#### 阶段 0：环境和基线冻结，1 天

- 准备 RBOT/MOPED 最小测试序列。
- 保存原始 BIT 输出和运行配置。
- 记录 GPU、CUDA、PyTorch、OpenCV、编译器版本。
- 固定至少一条快速回归序列。

交付：基线运行记录、环境清单、回归样本。

#### 阶段 1：严格评测和通信修复，2 天

- 增加严格评测配置。
- 移除失败后的 GT 状态写回。
- 实现 `.done` 完成标记。
- Python 按帧 ID 校验和加载批次。
- 增加通信异常测试。

交付：可信基线和稳定批次协议。

#### 阶段 2：Mask 选择和置信度，3 天

- 实现 SAM 多候选评分。
- 实现候选分歧。
- 计算四类置信度。
- 输出逐帧 JSONL 和调试图。

交付：可独立开关的置信度模块。

#### 阶段 3：加权优化和回滚，4 天

- 实现加权 IoU。
- 接入 flatten 和 temporal loss。
- 实现有效帧筛选。
- 实现最佳状态保存、门控和回滚。

交付：U-BIT Lite 核心算法。

#### 阶段 4：关键帧筛选和性能整理，3 天

- C++ 输出跟踪质量。
- 实现关键帧综合分数和保底策略。
- 分析额外耗时，优化重复渲染。

交付：完整闭环和性能报告。

#### 阶段 5：实验、消融和交付，5-7 天

- 跑基线与完整方案。
- 跑各模块消融。
- 三次重复实验。
- 生成指标表、失败案例和可视化。
- 补齐部署、测试和变更说明。

交付：发布候选版本、实验报告和技术总结。

### 5.2 时间安排

| 周期 | 工作内容 | 里程碑 |
| --- | --- | --- |
| 第 1 周 | 基线、严格评测、通信、Mask 选择 | MVP 可稳定运行 |
| 第 2 周 | 置信度、加权优化、回滚 | 核心方案完成 |
| 第 3 周 | 关键帧、实验、消融 | 主要结果完成 |
| 第 4 周 | 重复实验、文档、交付 | 发布候选版本 |

### 5.3 资源配置

#### 最小团队

- 1 名算法/全栈工程师：Python、PyTorch、C++，全程投入。
- 1 名评测支持人员：可兼职，负责数据、实验矩阵和结果检查。

#### 推荐团队

- Python 算法工程师 1 人。
- C++ 跟踪工程师 1 人，投入约 30%。
- 测试/实验工程师 1 人，投入约 30%。

#### 硬件

- NVIDIA RTX 3080 及以上，推荐 24 GB 显存。
- 至少 100 GB 可用磁盘；完整数据集按实际大小扩容。
- Ubuntu 20.04 或 22.04。

---

## 6. 部署流程

### 6.1 环境要求

推荐：

```text
Ubuntu 22.04
Python 3.10
CUDA 11.8
PyTorch CUDA 11.8 build
OpenCV 4.5
Eigen 3.4
Qt 5.12+
CMake 3.12+
GCC/G++ with C++17
```

### 6.2 Python 环境

```bash
conda create -n BIT_Track python=3.10
conda activate BIT_Track

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install git+https://github.com/facebookresearch/segment-anything.git
python setup.py install
```

检查 CUDA：

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)"
```

### 6.3 SAM 权重

```bash
mkdir -p checkpoints
cd checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
cd ..
```

### 6.4 C++ 依赖与构建

安装 OpenCV、Eigen、Qt、Glog、Gflags 和 OpenMP 后执行：

```bash
sh setup_cpp.sh
```

验证产物：

```bash
test -x tracker/bin/eg_BIT
test -x tracker/bin/gen_references
```

### 6.5 数据配置

编辑 `config/dataset.yml`：

```yaml
moped_dir: "/absolute/path/to/moped"
rbot_dir: "/absolute/path/to/RBOT_dataset"
```

生成任务配置：

```bash
sh gen_configs.sh
```

生成参考帧：

```bash
sh gen_refers.sh "moped" "black_drill/reference/00.yml"
sh gen_refers.sh "moped" "black_drill/evaluation/00.yml"
```

### 6.6 运行

原始兼容模式：

```bash
sh run_example.sh "moped" "black_drill/evaluation/00.yml"
```

U-BIT Lite 模式通过对应 YAML 开启：

```yaml
evaluation:
  strict_mode: true
confidence:
  enabled: true
geometry_optimization:
  weighted_loss: true
  rollback_enabled: true
```

### 6.7 部署验证

部署后依次验证：

1. Python 能导入 `torch`、`segment_anything` 和 `soft_renderer`。
2. `tracker/bin/eg_BIT` 可执行。
3. 配置文件中的数据、权重和输出目录存在。
4. C++ 启动后能生成完整帧和 `.done` 文件。
5. Python 不读取无 `.done` 的残留帧。
6. 严格模式日志中只有初始化帧的 `pose_source` 为 `ground_truth`。
7. 至少完成一轮网格优化并生成 accepted 或 rejected 记录。

---

## 7. 风险评估与应对措施

| 风险 | 概率 | 影响 | 应对措施 |
| --- | --- | --- | --- |
| 去掉 GT 恢复后基线显著下降 | 高 | 高 | 将严格协议作为真实基线；保留 lost 状态；后续开发自主重定位 |
| SAM score 与真实质量不一致 | 中 | 高 | 融合 SLCT 和渲染一致性；不单独使用 SAM score |
| 置信度阈值过多 | 高 | 中 | 集中配置；在固定验证序列选参；报告敏感性曲线 |
| 过滤过多导致无帧可优化 | 中 | 高 | 最小有效帧数、保底关键帧和阈值退火 |
| 加权后有效学习率变化 | 中 | 中 | 批次内权重归一化；记录梯度范数 |
| 网格出现自交或退化 | 中 | 高 | flatten、temporal、退化面检查和回滚 |
| 磁盘通信延迟 | 中 | 中 | 原子 rename；减少轮询；后续可迁移共享内存 |
| OpenCV GUI 在无显示环境失败 | 中 | 中 | 增加 headless 配置，关闭 `imshow/waitKey` |
| CUDA 扩展编译不兼容 | 中 | 高 | 固定 CUDA/PyTorch 版本；保存构建日志和环境镜像 |
| 数据集路径或序列不完整 | 中 | 中 | 部署前置校验脚本；快速失败并输出缺失列表 |
| 提升不足以形成论文结果 | 中 | 中 | 先保证可信工程结果，再升级为完整概率模型 |

### 7.1 回滚策略

- 所有功能均有配置开关。
- 保留 `legacy`、`strict_baseline`、`u_bit_lite` 三套标准配置。
- 网格更新永远保留上一份 accepted 模型。
- 发布时保留原始运行脚本，新增脚本不得覆盖用户数据。

---

## 8. 质量保障

### 8.1 测试策略

#### 单元测试

重点覆盖无 GPU 的纯函数：

- IoU 和候选评分。
- 候选分歧计算。
- 置信度归一化与融合。
- 帧权重归一化。
- 配置默认值和边界值。
- 批次文件交集与完整性验证。
- 更新门控接受/拒绝规则。

#### 集成测试

- C++ 写一帧，Python 完整读取。
- C++ 写到一半时 Python 不应读取。
- 损坏某类文件时 Python 应跳过并记录错误。
- Python 输出新模型后 C++ 能完整加载。
- 严格模式下失败后不发生 GT 恢复。

#### 回归测试

- 原始模式运行结果与修改前保持在允许随机波动内。
- 严格基线配置每次提交至少运行一条短序列。
- U-BIT Lite 至少完成一次完整三轮模型更新。

#### 实验测试

建议实验组：

```text
A. original BIT
B. strict BIT
C. B + SAM candidate selection
D. C + weighted geometry loss
E. D + update gate
F. E + confidence keyframe selection
```

### 8.2 核心测试用例

| 编号 | 场景 | 预期结果 |
| --- | --- | --- |
| TC-01 | `.done` 存在且所有文件完整 | 成功加载一帧 |
| TC-02 | `K` 已写入但 pose 未写入 | 不加载该帧，不崩溃 |
| TC-03 | pose 文件为空 | 标记损坏并跳过 |
| TC-04 | SAM 三个候选分数不同 | 选择综合评分最高候选 |
| TC-05 | SAM 候选互相高度冲突 | 分割置信度降低 |
| TC-06 | 某帧权重低于阈值 | 不参与几何优化 |
| TC-07 | 有效帧不足 | 跳过优化，保留当前模型 |
| TC-08 | 优化产生 NaN | 回滚且记录失败 |
| TC-09 | 门控损失恶化超阈值 | 拒绝 candidate 模型 |
| TC-10 | 连续无关键帧 | 保底策略接受最佳候选 |
| TC-11 | 严格模式跟踪失败 | 不写回 GT 姿态 |
| TC-12 | 旧 YAML 无新增字段 | 按兼容默认值运行 |

### 8.3 验收标准

#### 功能验收

- 所有 FR-01 至 FR-08 均完成。
- 新增模块可独立关闭。
- 严格模式不存在非初始化 GT 状态反馈。
- 通信异常不会造成 Python 崩溃。
- 模型回滚经过至少一个构造用例验证。

#### 性能验收

- 新增置信度逻辑导致的额外耗时不超过目标值。
- 相比 strict BIT，完整方案至少满足以下一项：
  - ADD(-S) AUC 有统计稳定提升；
  - 跟踪失败率降低至少 10%；
  - 相近精度下几何优化帧数减少至少 20%；
  - Mask 噪声实验中的性能下降显著减小。
- 三次重复实验结论一致。

#### 交付验收

- 代码、标准配置、部署文档、测试记录齐全。
- 至少提供一条可运行示例命令。
- 结果目录能够追溯到配置、环境和模型版本。
- 已知限制和未解决问题有明确记录。

---

## 9. 维护方案

### 9.1 日常维护

- 定期检查数据目录和运行目录磁盘使用量。
- 每次运行确认 `.done` 与数据文件数量一致。
- 保留最近一次 accepted 模型和完整日志。
- 配置变更必须记录版本，禁止直接在源码中调参。
- 依赖升级前保存当前可复现环境。

### 9.2 问题排查流程

#### 跟踪器无法启动

1. 检查 `tracker/bin/eg_BIT` 是否存在及可执行。
2. 检查 Qt/OpenCV/Glog 动态库。
3. 检查配置路径是否为绝对路径或相对当前工作目录可解析。
4. 查看 C++ stderr 和结果目录。

#### Python 一直等待数据

1. 检查 C++ 进程是否仍在运行。
2. 检查 `ready/*.done`。
3. 检查对应八类数据文件是否齐全。
4. 检查目录权限和剩余磁盘空间。
5. 查看批次验证日志中的缺失文件列表。

#### 几何优化出现 NaN

1. 检查姿态矩阵和相机内参是否有限。
2. 检查 Mask 是否为空。
3. 检查模型顶点是否接近 sigmoid 变换奇点。
4. 降低学习率并启用梯度裁剪。
5. 回滚到最后 accepted 模型。

#### 性能明显下降

1. 区分 original 与 strict 基线，避免协议混淆。
2. 查看每个置信度分量分布。
3. 检查是否过滤过多关键帧。
4. 检查 SAM 候选与渲染轮廓可视化。
5. 按消融配置逐项关闭模块定位问题。

### 9.3 版本计划

#### v0.1 - Strict Baseline

- 严格评测协议。
- 通信完整性。
- 结构化日志。

#### v0.2 - Confidence MVP

- SAM 多候选选择。
- 帧置信度。
- 低质量帧过滤。

#### v0.3 - U-BIT Lite

- 加权几何优化。
- 最佳模型保存。
- 更新门控和回滚。

#### v1.0 - Experimental Release

- 关键帧筛选。
- 完整测试和消融。
- 部署与实验文档。
- 稳定标准配置。

#### 后续 v2.0 候选

- 滑动窗口概率优化。
- 姿态 Hessian 协方差。
- 长期记忆与自主重定位。
- 共享内存或消息队列通信。

### 9.4 兼容与升级原则

- 配置新增字段必须提供默认值。
- 通信协议通过 `protocol_version` 区分。
- 旧版读取器不得读取新版不兼容批次。
- 模型文件继续使用 OBJ，确保 Summer/SLCT 可直接加载。
- 依赖大版本升级必须重新执行完整回归实验。

---

## 10. 最终交付清单

### 10.1 代码

- 严格评测模式实现。
- 原子批次通信实现。
- SAM Mask 选择器。
- 置信度估计与融合模块。
- 加权几何优化器。
- 更新门控和回滚模块。
- 关键帧筛选增强。
- 结构化实验日志。

### 10.2 配置

- `legacy.yml`：兼容原始 BIT。
- `strict_baseline.yml`：严格基线。
- `u_bit_lite.yml`：完整方案。
- 各消融实验配置。

### 10.3 测试与实验

- 单元测试和集成测试结果。
- 原始、严格基线和 U-BIT Lite 对比结果。
- 模块消融表。
- 三次重复实验统计。
- 典型成功和失败案例。

### 10.4 文档

- 本实施文档。
- 环境部署说明。
- 配置字段说明。
- 实验运行说明。
- 结果目录和指标说明。
- 已知问题与限制。

---

## 11. 实施决策摘要

本方案选择在现有 BIT 闭环上增加“观测质量控制层”，而不是替换跟踪器或三维表示。该决策的主要依据是：

1. 复用现有 C++ 跟踪器、SAM 和 Soft Rasterizer，开发周期最短。
2. 不需要新数据标注和网络训练，环境与算法风险较低。
3. 先修复 GT 恢复和通信问题，可以建立可信、稳定的实验基线。
4. 置信度加权、更新门控和关键帧筛选均可独立验证，方便消融和回滚。
5. 交付结果既可直接改善工程稳定性，也能作为后续完整概率 BIT 研究的基础。

建议团队严格按照“可信基线 -> 置信度 -> 加权优化 -> 更新门控 -> 关键帧 -> 完整实验”的顺序实施，避免同时修改多个闭环模块导致问题难以定位。
