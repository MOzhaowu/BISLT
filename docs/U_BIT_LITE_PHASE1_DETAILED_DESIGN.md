# U-BIT Lite 第一阶段详细设计文档

## 0. 文档控制

| 项目 | 内容 |
| --- | --- |
| 文档名称 | U-BIT Lite 第一阶段详细设计文档 |
| 上位文档 | `docs/U_BIT_LITE_IMPLEMENTATION_PLAN.md` |
| 设计版本 | 1.0 |
| 目标版本 | U-BIT Lite v0.1-v0.3 |
| 设计状态 | 可进入开发 |
| 目标周期 | 15-20 个工作日 |
| 设计原则 | 最小改动、可独立验证、默认兼容、保留研究扩展接口 |

---

## 1. 设计目标与约束

### 1.1 第一阶段目标

第一阶段在不替换 Summer/SLCT、不训练新网络、不改变 OBJ 模型接口的前提下，完成以下闭环：

```text
严格首帧初始化
  -> C++ 跟踪
  -> 原子发布完整观测
  -> SAM 多候选选择
  -> 分割/跟踪/渲染/视角置信度
  -> 加权几何优化
  -> 模型门控与回滚
  -> C++ 加载已接受模型
```

第一阶段不是完整概率 U-BIT。所有原始置信度信号必须被保留，后续可将启发式融合替换为概率推断，而无需修改上游数据接口。

### 1.2 强制约束

1. GT 只允许用于初始化和离线指标计算，不允许影响第 1 帧之后的跟踪状态。
2. Python 不得再通过某一种文件数量推断批次完成。
3. 每个观测必须具有稳定的 `frame_id`，所有模态按 `frame_id` 对齐。
4. 新功能必须可通过配置独立关闭。
5. 原始 BIT 模式必须继续可运行。
6. 候选模型未通过门控时，C++ 只能继续使用上一份已接受模型。
7. 所有置信度原始分量必须写入结构化日志。

### 1.3 非目标

- 本阶段不实现因子图、Hessian 姿态协方差或贝叶斯后验。
- 本阶段不实现完全遮挡后的全局重定位。
- 本阶段不引入共享内存、gRPC 或消息队列。
- 本阶段不实现新的 CUDA Kernel。
- 本阶段不重新设计 Summer/SLCT 的轮廓优化器。

---

## 2. 当前实现问题与设计决策

### 2.1 当前问题

| 编号 | 当前行为 | 影响 |
| --- | --- | --- |
| P-01 | 每帧 GT 被写入 `StdData` | 跟踪模块可访问未来真值 |
| P-02 | 失败后执行 `setPose(currData->gt)` | 破坏自主长期跟踪协议 |
| P-03 | Python 只按 `K` 文件数判断完成 | 可能早于 pose 写入，产生竞态 |
| P-04 | 不同模态按目录列表位置切片 | 文件缺失后会发生跨帧错配 |
| P-05 | SAM 无条件使用 `masks[0]` | 忽略候选质量和已有跟踪证据 |
| P-06 | 几何优化对所有帧等权 | 错误观测直接污染模型 |
| P-07 | `flatten_loss` 被计算但未使用 | 网格折叠约束不足 |
| P-08 | 保存最后一步而非最佳模型 | 优化后期震荡会降低结果 |
| P-09 | 模型更新没有门控和回滚 | 错误模型被直接注入跟踪器 |
| P-10 | 关键帧只使用角度阈值 | 低质量姿态也可能进入重建 |

### 2.2 核心设计决策

#### D-01 GT 与运行时状态隔离

`eg_BIT.cpp` 仍可持有 `gts` 用于离线评测，但第 1 帧之后不再把对应 GT 写入跟踪器输入。初始化姿态通过明确的 `SetInitPose()` 接口注入。

#### D-02 完成标记是唯一提交点

一帧所有模态写入并关闭后，C++ 最后原子创建 `.done`。Python 只消费 `.done` 对应帧。

#### D-03 Python 是质量融合唯一责任方

C++ 只负责生成现有概率图、姿态和视角。SAM 分数、渲染一致性和最终 `frame_weight` 全部在 Python 计算，避免两套融合逻辑。

#### D-04 Mask 选择不依赖当前 Python 渲染结果

Mask 候选首先基于 SAM score 和 SLCT probability 选择；选定后再与当前网格渲染轮廓比较，得到 `render_confidence`。这样避免 SAM 选择与几何渲染互相依赖。

#### D-05 更新门控使用保留观测

每轮观测按确定性规则划分优化集和门控集。门控集不参与梯度更新，只用于比较更新前后质量。

#### D-06 原始信号长期保留

结构化日志保存各置信度分量、候选分数和拒绝原因。后续完整 U-BIT 可以直接复用这些数据进行概率建模。

---

## 3. 代码组织与文件级改动

### 3.1 新增 Python 文件

```text
examples/u_bit/
  __init__.py
  config.py
  data_types.py
  batch_protocol.py
  mask_selector.py
  confidence.py
  geometry_optimizer.py
  update_gate.py
  experiment_logger.py
```

职责如下：

| 文件 | 职责 |
| --- | --- |
| `config.py` | 新配置默认值、兼容合并和校验 |
| `data_types.py` | dataclass 数据结构 |
| `batch_protocol.py` | 按 frame ID 扫描、校验和加载批次 |
| `mask_selector.py` | SAM 候选评分和选择 |
| `confidence.py` | 各置信度分量和融合 |
| `geometry_optimizer.py` | 加权损失、最佳状态和优化结果 |
| `update_gate.py` | 模型质量校验、接受和回滚 |
| `experiment_logger.py` | JSONL、环境和配置快照 |

### 3.2 修改 Python 文件

#### `examples/communication.py`

- 保留 `DataGroups` 供旧模式使用。
- 新增协议版本分支。
- 新模式委托 `u_bit.batch_protocol`。
- 删除新模式下基于目录列表位置对齐的逻辑。

#### `examples/sam_utils.py`

- `segment()` 改为返回结构化选择结果。
- 修复 `save_points` 使用错误配置键。
- 修复 `BinaringImgs()` 返回原图而非阈值结果。
- `LoadConfigSafety()` 改为内存解析，不再使用固定临时文件。

#### `examples/run_example.py`

- 抽离主流程为可测试函数。
- 接入配置、批次、Mask 选择、置信度、优化和门控。
- 将 GUI 显示放到 `runtime.headless` 开关之后。
- 保存 best state 和 accepted mesh。

#### `config/config_generator.py`

- 输出 U-BIT Lite 默认字段。
- 保留旧字段和 OpenCV YAML 头。

### 3.3 修改 C++ 文件

#### `tracker/definition/global_definition.h`

- `DataGroup` 增加 `frameId`。
- 可选增加低成本质量字段，不在第一提交中强制。

#### `tracker/interface/interface.h/.cc`

- 暴露 `SetInitPose()`。
- 暴露严格评测策略设置，或在 `eg_BIT.cpp` 内部完成最小策略控制。

#### `tracker/interface/system_manager.*`

- 将 `SetInitPose()` 转发给 tracker manager。

#### `tracker/interface/tracker_manager.*`

- 将初始化姿态转发给当前 tracker。

#### `tracker/interface/tracker_summer/tracker_summer.cc`

- `StartTracking()` 使用 `m_initPose`，不读取当前帧 GT。
- 删除失败后的 GT 姿态写回。
- 失败只设置状态和计数，不修改 pose。

#### `tracker/interface/communication.h/.cpp`

- 创建 `ready/`。
- 清理 ready 目录。
- `SentDatas2Py()` 聚合所有写入结果。
- 所有文件写完后调用 `PublishDoneMarker()`。
- 写入函数返回真实成功状态。

#### `tracker/example/eg_BIT.cpp`

- 首帧设置初始化姿态。
- 后续 `StdData` 不携带有效 GT 给 tracker。
- 离线指标继续从本地 `gts` 与 `poses` 计算。
- 接入 C++ 关键帧预筛选。

### 3.4 不修改文件

- `soft_renderer/cuda/*`
- Summer/SLCT 核心 Jacobian 和搜索线算法。
- OBJ 文件格式和模型缩放接口。
- SAM checkpoint 类型。

---

## 4. 配置详细设计

### 4.1 默认配置

新配置集中在 `DEFAULT_U_BIT_CONFIG`：

```python
DEFAULT_U_BIT_CONFIG = {
    "evaluation": {
        "strict_mode": False,
        "init_frame": 0,
        "allow_gt_recovery": True,
    },
    "communication": {
        "protocol_version": 1,
        "require_done_marker": False,
        "batch_timeout_seconds": 30,
    },
    "runtime": {
        "headless": False,
        "seed": 42,
    },
    "confidence": {
        "enabled": False,
        "min_frame_weight": 0.25,
        "min_valid_frames": 2,
        "epsilon": 1e-6,
        "weights": {
            "sam": 0.35,
            "tracker": 0.25,
            "render": 0.25,
            "view": 0.15,
        },
    },
    "mask_selector": {
        "use_all_sam_candidates": False,
        "sam_score_weight": 0.60,
        "tracker_iou_weight": 0.40,
        "min_mask_score": 0.30,
        "area_ratio_min": 0.01,
        "area_ratio_max": 0.90,
    },
    "geometry_optimization": {
        "weighted_loss": False,
        "laplacian_weight": 0.10,
        "flatten_weight": 0.00,
        "temporal_weight": 0.00,
        "gradient_clip_norm": 5.0,
        "save_best_state": False,
    },
    "update_gate": {
        "enabled": False,
        "holdout_ratio": 0.20,
        "max_relative_degradation": 0.03,
        "max_invalid_face_ratio": 0.01,
    },
    "logging": {
        "enabled": True,
        "run_root": "runs",
        "save_debug_images": False,
    },
}
```

旧配置合并后默认保持原始行为。标准 U-BIT Lite 配置显式开启新字段。

### 4.2 配置加载接口

```python
def load_opencv_yaml(path: str) -> dict:
    """忽略 OpenCV YAML 第一行并在内存中解析。"""

def deep_merge(defaults: dict, overrides: dict) -> dict:
    """递归合并，列表和标量整体替换。"""

def validate_u_bit_config(config: dict) -> None:
    """检查权重、阈值和互斥条件。"""
```

校验规则：

- 所有置信度权重非负且总和大于 0。
- 所有置信度阈值位于 `[0, 1]`。
- `holdout_ratio` 位于 `[0, 0.5]`。
- 严格模式下强制 `allow_gt_recovery=false`。
- 协议版本 2 时强制 `require_done_marker=true`。
- `min_valid_frames >= 1`。

---

## 5. 数据结构详细设计

### 5.1 Python 数据结构

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np


@dataclass(frozen=True)
class FrameBatchPaths:
    frame_id: int
    rgb: Path
    probability: Path
    tracker_mask: Path
    pose: Path
    intrinsics: Path
    origin_roi: Path
    target_roi: Path
    target_size: Path
    done: Path


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
class MaskCandidateScore:
    index: int
    sam_score: float
    tracker_iou: float
    area_score: float
    total: float


@dataclass
class SelectedMask:
    full_resolution_mask: np.ndarray
    candidate_index: int
    candidates: list[MaskCandidateScore]
    disagreement: float
    confidence: float


@dataclass
class ConfidenceComponents:
    sam: float
    tracker: float
    render: float
    view: float
    combined_raw: float
    combined_normalized: float = 0.0


@dataclass
class WeightedObservation:
    observation: FrameObservation
    selected_mask: SelectedMask
    rendered_mask: Optional[np.ndarray]
    confidence: ConfidenceComponents
    accepted: bool
    rejection_reason: Optional[str]


@dataclass
class OptimizationResult:
    attempted: bool
    accepted: bool
    reason: str
    valid_frame_ids: list[int]
    train_loss_before: Optional[float]
    train_loss_best: Optional[float]
    gate_loss_before: Optional[float]
    gate_loss_after: Optional[float]
    candidate_mesh_path: Optional[Path]
    accepted_mesh_path: Optional[Path]
```

### 5.2 DataGroups 兼容策略

旧 `DataGroups` 保持不删除。新增转换函数：

```python
def observations_to_data_groups(
    observations: list[FrameObservation],
) -> DataGroups:
    ...
```

新算法内部只使用 `FrameObservation`，仅在调用旧工具函数时转换，防止新的质量字段继续依赖多个平行数组。

### 5.3 C++ 数据结构

对 `DataGroup` 的最小修改：

```cpp
struct DataGroup
{
    int frameId{-1};
    bool valid{false};
    // existing fields remain unchanged
};
```

第一阶段不将 Python 最终置信度回传 C++。C++ 预筛选所需质量直接从 `DataGroup.prob` 计算。

### 5.4 日志 Schema

`frames.jsonl` 每帧一条，字段不得随运行动态缺失；不可计算项写 `null`：

```json
{
  "schema_version": 1,
  "frame_id": 12,
  "batch_valid": true,
  "mask_candidate_index": 1,
  "sam_scores": [0.61, 0.85, 0.72],
  "sam_confidence": 0.78,
  "tracker_confidence": 0.70,
  "render_confidence": 0.65,
  "view_confidence": 0.56,
  "frame_weight_raw": 0.66,
  "frame_weight_normalized": 0.94,
  "accepted_for_geometry": true,
  "rejection_reason": null
}
```

`optimizations.jsonl` 每轮一条：

```json
{
  "schema_version": 1,
  "round": 2,
  "valid_frame_ids": [0, 4, 9, 12],
  "holdout_frame_ids": [12],
  "train_loss_before": 0.34,
  "train_loss_best": 0.21,
  "gate_loss_before": 0.38,
  "gate_loss_after": 0.31,
  "relative_degradation": -0.184,
  "invalid_face_ratio": 0.0,
  "accepted": true,
  "reason": "gate_improved"
}
```

---

## 6. 严格评测详细设计

### 6.1 初始化流程

在 `eg_BIT.cpp` 加载 GT 后：

```cpp
const int init_frame = LoadOptionalInt(config, "evaluation.init_frame", 0);
const bool strict_mode =
    LoadOptionalBool(config, "evaluation.strict_mode", false);

SetInitPose(gts.at(init_frame));
```

`SetInitPose` 调用链：

```text
interface::SetInitPose
  -> SystemManager::SetInitPose
  -> TrackerManager::SetInitPose
  -> TrackerBase::SetInitPose
```

若当前已有内部调用链但未暴露到 `interface.h`，只补齐公开接口和转发，不创建第二套初始化变量。

### 6.2 跟踪输入规则

严格模式：

```cpp
psd->index = index;
psd->frame = frame;
psd->poseScale = poseScale;
psd->gt = cv::Matx44f::eye();  // tracker 不得依赖此字段
```

推荐后续将 `gt` 从运行时 `StdData` 拆出。本阶段为最小改动保留字段，但严格模式下不传真实值。

### 6.3 SummerTracker 修改

修改前：

```cpp
m_objects[0]->setPose(currData->gt);
```

修改后：

```cpp
m_objects[0]->setPose(m_initPose);
```

跟踪失败时：

```cpp
if (!success) {
    ++m_failCnts;
    SetTrackingResult(ResultType::kResValid, 0);
    // 不修改 m_objects[0] pose
}
```

严格模式不能调用依赖当前 GT 的 `IsSuccess(currPose)` 作为运行时失败检测。MVP 处理：

- 原始模式：继续使用现有 `IsSuccess`，保证兼容。
- 严格模式：运行时 `success` 只基于内部质量信号；第一提交可先不执行自动恢复，只记录有限性和 ROI 有效性。
- 离线成功率在 `eg_BIT.cpp` 保存 pose 后使用本地 `gts[index]` 计算。

### 6.4 内部失败判定

新增不依赖 GT 的基础判定：

```cpp
bool SummerTracker::IsInternallyValid(
    const cv::Matx44f& pose,
    const DataGroup& data_group) const;
```

MVP 条件：

- pose 所有元素有限。
- rotation determinant 位于合理范围。
- translation norm 有限。
- `data_group.valid == true`。
- ROI 面积大于 0 且不越界。
- probability map 非空。

该函数不是精确 lost detector，只用于防止明显无效状态继续污染下游。

### 6.5 离线指标

`eg_BIT.cpp` 保留：

```cpp
poses.push_back(estimated_pose);
```

序列结束后使用 `gts` 和 `poses` 计算 ADD、AUC 或当前已有指标。任何指标计算结果不得反馈到 `ChangeModels`、`SetPose` 或关键帧选择。

---

## 7. 原子批次通信详细设计

### 7.1 协议版本

| 版本 | 行为 |
| --- | --- |
| v1 | 当前目录扫描和列表切片，仅兼容 |
| v2 | 临时文件、原子 rename、`.done` 提交、按 frame ID 加载 |

### 7.2 C++ 目录成员

`LocalStorageCommunicator` 新增：

```cpp
std::string ready_dir_;

bool PublishDoneMarker(int frame_id);
bool AllFrameFilesExist(int frame_id) const;
std::string FrameStem(int frame_id) const;
```

`SetUp()` 创建 `ready/`，`ClearCache()` 清理该目录。

### 7.3 写入返回值

当前 `cv::imwrite` 分支无论是否成功都返回 true。修改为：

```cpp
return cv::imwrite(target_path, data);
```

文本写入必须检查：

```cpp
if (!ofs.is_open()) {
    return false;
}
ofs << ...;
ofs.flush();
const bool ok = ofs.good();
ofs.close();
return ok;
```

### 7.4 提交流程

```cpp
bool LocalStorageCommunicator::SentDatas2Py(
    TrackingResult* result,
    const int& index) {
    const DataGroup& group = result->dataGroup;

    const bool ok =
        WriteData(group.img, index, "img") &&
        WriteData(group.prob, index, "prob") &&
        WriteData(group.mask, index, "mask") &&
        WriteData(group.originRoi, index, "origin_roi") &&
        WriteData(group.targetRoi, index, "target_roi") &&
        WriteData(group.targetSize, index, "size") &&
        WriteData(group.K, index, "K") &&
        WriteData(normalized_pose, index, "pose");

    if (!ok || !AllFrameFilesExist(index)) {
        return false;
    }
    return PublishDoneMarker(index);
}
```

`DataNumsSent2PyAddOne()` 只能在 `SentDatas2Py()` 返回 true 后调用，避免计数先于提交。

### 7.5 原子文件写入

建议新增通用辅助函数：

```cpp
bool AtomicRename(const std::string& temp_path,
                  const std::string& final_path);
```

每个文本文件：

```text
write final_path.tmp
flush and close
rename final_path.tmp -> final_path
```

图像可先 `imwrite(final_path + ".tmp.png")`，再 rename 到正式 `.png`。临时扩展名需要保持 OpenCV 可识别。

`.done` 最后创建：

```text
ready/0012.done.tmp
rename -> ready/0012.done
```

内容采用简单文本，避免 C++ 引入 JSON 依赖：

```text
protocol_version=2
frame_id=12
file_count=8
```

### 7.6 Python 扫描

```python
FRAME_PATTERNS = {
    "rgb": ("img", ".png"),
    "probability": ("prob", ".png"),
    "tracker_mask": ("mask", ".png"),
    "pose": ("pose", ".pose"),
    "intrinsics": ("K", ".K"),
    "origin_roi": ("originRoi", ".roi"),
    "target_roi": ("targetRoi", ".roi"),
    "target_size": ("targetSize", ".size"),
}

def scan_complete_frame_ids(root: Path) -> list[int]:
    return sorted(parse_id(p) for p in (root / "ready").glob("*.done"))
```

加载前执行：

- 所有文件存在。
- 所有文件大小大于 0。
- 图像读取结果非 `None`。
- pose 恰有 12 个浮点数。
- K 恰有 4 个浮点数。
- ROI 恰有 4 个整数。
- target size 恰有 2 个正数。

### 7.7 消费状态

`LocalStorageCommunication` 新增：

```python
self.consumed_frame_ids: set[int] = set()
```

只返回：

```text
complete IDs - consumed IDs
```

同一批次成功交给优化流程后再标记 consumed。加载异常不标记 consumed，允许下一轮重试；超过 timeout 后记录并隔离。

---

## 8. SAM Mask 选择详细设计

### 8.1 接口

```python
def select_sam_mask(
    masks: np.ndarray,
    sam_scores: np.ndarray,
    tracker_probability: np.ndarray,
    origin_roi: np.ndarray,
    target_roi: np.ndarray,
    target_size: np.ndarray,
    final_size: tuple[int, int],
    config: dict,
) -> SelectedMask:
    ...
```

### 8.2 坐标统一

每个 SAM candidate 先通过现有 ROI 映射到最终原图坐标：

```python
candidate_full = resize_mask(
    candidate,
    origin_roi,
    target_roi,
    target_size,
    final_size,
)
```

SLCT probability 也必须位于相同最终尺寸。比较前统一二值化：

```python
tracker_binary = tracker_probability >= tracker_threshold
candidate_binary = candidate_full >= mask_threshold
```

### 8.3 候选评分

```python
def binary_iou(a: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> float:
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(intersection / (union + eps))


def area_score(mask: np.ndarray, min_ratio: float, max_ratio: float) -> float:
    ratio = np.count_nonzero(mask) / mask.size
    if ratio < min_ratio or ratio > max_ratio:
        return 0.0
    return 1.0
```

总分：

```text
total_i =
    ws * clamp(sam_score_i)
  + wt * IoU(candidate_i, tracker_binary)

total_i *= area_score_i
```

候选分歧：

```text
disagreement =
    mean_{i<j}(1 - IoU(candidate_i, candidate_j))
```

最终分割置信度：

```text
sam_confidence =
    clamp(best_total * (1 - disagreement), 0, 1)
```

### 8.4 回退规则

1. 只有一个候选：`disagreement=0`。
2. tracker probability 无效：将 tracker 权重归零并重新归一化。
3. 所有 candidate area 无效：选择 SAM score 最高者，但 `confidence=0`。
4. 所有候选为空：返回 rejected 结果，不抛出进程级异常。

### 8.5 segment 新返回值

```python
def segment(...) -> list[SelectedMask]:
    ...
```

旧模式下提供：

```python
def selected_masks_to_legacy_list(
    results: list[SelectedMask],
) -> list[np.ndarray]:
    return [item.full_resolution_mask for item in results]
```

---

## 9. 置信度详细设计

### 9.1 Tracker confidence

从 SLCT probability 计算，不依赖 GT：

```text
certainty(p) = 2 * |p - 0.5|
```

只在当前 candidate Mask 或 tracker ROI 内统计：

```text
tracker_confidence =
    mean(certainty(probability / 255))
```

增加前景有效比例惩罚：

```text
tracker_confidence *= min(valid_pixel_ratio / target_ratio, 1)
```

### 9.2 Render confidence

Mask 选择完成后，使用当前模型、对应 pose 和 K 渲染轮廓：

```python
def render_current_silhouettes(
    model: Model,
    transforms: "TransformBatch",
    rasterizer,
) -> np.ndarray:
    ...
```

渲染结果和 selected mask 统一执行：

1. 垂直翻转规则与当前几何损失保持一致。
2. 非方形输入按当前 `ExtendMatImgs` 规则补齐。
3. resize 到 `renderer_img_size`。
4. 二值化后计算 IoU。

首轮球体与真实目标差异可能很大，因此使用 warmup：

```text
round == 0:
    render_confidence = 0.5
    render exponent = 0
round > 0:
    render confidence enabled
```

### 9.3 View confidence

根据当前视角与已接受观测视角的最小夹角：

```text
min_angle = min(angle(view, accepted_view_i))
view_confidence = clamp(min_angle / target_angle, 0, 1)
```

无历史视角时为 1。视角从 pose 计算，必须复用 C++ 相同坐标约定并通过固定姿态测试。

### 9.4 融合

加权几何平均：

```python
def combine_confidence(
    components: dict[str, float],
    exponents: dict[str, float],
    eps: float,
) -> float:
    log_weight = 0.0
    exponent_sum = 0.0
    for name, exponent in exponents.items():
        if exponent <= 0:
            continue
        log_weight += exponent * np.log(max(components[name], eps))
        exponent_sum += exponent
    return float(np.exp(log_weight / exponent_sum))
```

批次归一化：

```text
normalized_weight_i =
    raw_weight_i / mean(raw_weights_of_accepted_frames)
```

限制到 `[0.25, 4.0]`，避免单帧梯度支配。

### 9.5 拒绝原因枚举

```python
class RejectionReason(str, Enum):
    INVALID_BATCH = "invalid_batch"
    EMPTY_MASK = "empty_mask"
    LOW_SAM_CONFIDENCE = "low_sam_confidence"
    LOW_TRACKER_CONFIDENCE = "low_tracker_confidence"
    LOW_COMBINED_WEIGHT = "low_combined_weight"
    INVALID_POSE = "invalid_pose"
    REDUNDANT_VIEW = "redundant_view"
```

日志必须使用稳定枚举值，不写自由文本作为统计字段。

---

## 10. 加权几何优化详细设计

### 10.1 输入

```python
def optimize_geometry(
    model: Model,
    observations: list[WeightedObservation],
    renderer_bundle: "RendererBundle",
    config: dict,
    round_index: int,
) -> OptimizationResult:
    ...
```

### 10.2 数据划分

按 `frame_id` 排序后确定性划分：

```python
def split_train_gate(items, holdout_ratio):
    if len(items) < 4:
        return items, items[-1:]
    stride = max(round(1 / holdout_ratio), 2)
    gate = items[stride - 1::stride]
    train = [item for item in items if item not in gate]
    return train, gate
```

数据少于 2 帧时不优化。门控集可以与训练集重叠的情况必须写入日志；论文正式实验建议至少 4 帧后使用独立门控。

### 10.3 加权 Soft IoU

```python
def per_frame_iou_loss(predict: torch.Tensor,
                       target: torch.Tensor) -> torch.Tensor:
    dims = tuple(range(1, predict.ndim))
    intersection = (predict * target).sum(dims)
    union = (predict + target - predict * target).sum(dims)
    return 1.0 - intersection / (union + 1e-6)


frame_losses = per_frame_iou_loss(predicted_alpha, target_alpha)
mask_loss = (frame_losses * weights).sum() / weights.sum().clamp_min(1e-6)
```

不得使用当前 `neg_iou_loss()` 内部对 batch 再整体求和的写法，因为无法实现逐帧加权。

### 10.4 总损失

```python
total_loss = (
    mask_loss
    + cfg.laplacian_weight * laplacian_loss
    + cfg.flatten_weight * flatten_loss
    + cfg.temporal_weight * temporal_loss
)
```

`temporal_loss`：

```python
reference_vertices = model.current_vertices().detach().clone()
temporal_loss = F.mse_loss(model.current_vertices(), reference_vertices)
```

需要在 `Model` 中增加：

```python
def current_vertices(self) -> torch.Tensor:
    """返回参数化后的单批次顶点，不执行 repeat。"""
```

避免复制 `forward()` 中的顶点参数化逻辑。

### 10.5 数值保护

每步检查：

- `torch.isfinite(total_loss)`。
- 所有参数梯度有限。
- 梯度裁剪后范数有限。
- 当前顶点有限。

异常时：

1. 立即停止当前轮。
2. 恢复优化前 state。
3. 返回 `accepted=false` 和稳定 reason code。
4. 不输出可供 C++ 扫描的新模型文件。

### 10.6 最佳状态

每 `validation_interval` 步在 gate 集上无梯度计算损失：

```python
with torch.no_grad():
    gate_loss = evaluate_mask_loss(...)
```

以 gate loss 为主、train loss 为辅保存 best state。MVP 为减少渲染开销，可每 10 步验证一次。

### 10.7 优化结果

候选模型只写入运行目录：

```text
runs/.../meshes/candidate_round_02.obj
```

门控通过后再原子复制或保存至 C++ 扫描目录：

```text
<cmc>/deformed_model/_accepted_02.obj.tmp
rename -> _accepted_02.obj
```

C++ 不得扫描 `runs/.../candidate`。

---

## 11. 模型门控详细设计

### 11.1 门控指标

```python
@dataclass
class MeshValidation:
    gate_loss_before: float
    gate_loss_after: float
    relative_degradation: float
    invalid_face_ratio: float
    finite_vertices: bool
    extent_ratio: float
```

### 11.2 几何检查

- 顶点全部有限。
- 三角形面积小于 `area_epsilon` 的比例不超过阈值。
- 新模型包围盒最大边与旧模型之比位于 `[0.5, 2.0]`。
- 面数量和索引拓扑不变。

### 11.3 接受规则

```python
accept = (
    finite_vertices
    and invalid_face_ratio <= max_invalid_face_ratio
    and min_extent_ratio <= extent_ratio <= max_extent_ratio
    and relative_degradation <= max_relative_degradation
)
```

若 gate loss 有提升但几何检查失败，仍必须拒绝。

### 11.4 状态机

```text
IDLE
  -> OPTIMIZING
      -> NUMERICAL_FAILURE -> ROLLED_BACK -> IDLE
      -> CANDIDATE_READY
          -> GATE_REJECTED -> ROLLED_BACK -> IDLE
          -> GATE_ACCEPTED -> PUBLISHED -> IDLE
```

任何状态异常退出时，`latest_accepted.obj` 不变。

---

## 12. 关键帧预筛选详细设计

### 12.1 C++ 质量计算

在 `eg_BIT.cpp` 增加纯函数：

```cpp
float ComputeProbabilityConfidence(const cv::Mat& probability);
float ComputeAngleNovelty(const Eigen::Vector3f& view,
                          const std::vector<Eigen::Vector3f>& accepted_views,
                          float target_angle_degrees);
```

概率置信度：

```text
mean(2 * abs(p / 255 - 0.5))
```

仅统计非零概率区域，若有效像素过少则返回 0。

### 12.2 接受逻辑

```cpp
const float tracking_confidence =
    ComputeProbabilityConfidence(data_group.prob);
const float angle_novelty =
    ComputeAngleNovelty(view, templateViews, target_angle);
const float combined =
    tracking_confidence * angle_novelty;

const bool should_send =
    data_group.valid &&
    tracking_confidence >= min_tracking_confidence &&
    angle_novelty > 0.0f &&
    combined >= min_combined_score;
```

### 12.3 保底策略

维护：

```cpp
int frames_since_last_publish{0};
FrameCandidate best_pending;
```

超过 `force_accept_after_frames` 后发布期间得分最高候选。MVP 如果缓存完整图像代价过高，可以暂不在 C++ 实现保底缓存，而是在阈值连续失败后临时降低阈值；正式版本应缓存候选数据或直接让 Python 完成最终筛选。

### 12.4 计数顺序

正确顺序：

```cpp
if (should_send && communicator->SentDatas2Py(result, index)) {
    communicator->DataNumsSent2PyAddOne();
    templateViews.push_back(view);
}
```

写入失败不得增加计数或更新已接受视角。

---

## 13. 主流程时序

### 13.1 C++ 时序

```text
load config and GT
set initial pose from GT[init_frame]
initialize tracker

for each frame:
    read image
    track from previous state
    collect DataGroup
    save estimated pose for offline evaluation
    compute internal quality
    if keyframe candidate:
        atomically publish all modalities
        publish .done
        increment sent count
    if expected model round reached:
        wait for accepted model with timeout
        load and replace model

after sequence:
    compute offline metrics using local GT vector
```

### 13.2 Python 时序

```text
load merged config
initialize SAM, renderer and initial mesh
initialize logger

while rounds remain:
    scan .done markers
    load and validate complete observations
    run SAM multi-mask selection
    calculate tracker confidence
    render current model silhouettes
    calculate render and view confidence
    filter and normalize observations

    if insufficient observations:
        continue waiting

    split train and gate sets
    optimize candidate geometry
    validate candidate

    if accepted:
        atomically publish accepted mesh
    else:
        keep previous mesh

    log full result
```

### 13.3 超时行为

- Python 等待批次超时：记录 warning，继续扫描；总超时由运行配置控制。
- C++ 等待模型超时：默认继续使用旧模型，不无限阻塞。
- 原始兼容模式保留当前无限等待行为，仅供复现实验。

---

## 14. 错误处理

### 14.1 错误分类

```python
class UBitErrorCode(str, Enum):
    CONFIG_INVALID = "config_invalid"
    BATCH_INCOMPLETE = "batch_incomplete"
    BATCH_CORRUPTED = "batch_corrupted"
    SAM_EMPTY = "sam_empty"
    NO_VALID_OBSERVATION = "no_valid_observation"
    LOSS_NOT_FINITE = "loss_not_finite"
    GRADIENT_NOT_FINITE = "gradient_not_finite"
    MESH_INVALID = "mesh_invalid"
    GATE_REJECTED = "gate_rejected"
    MODEL_PUBLISH_FAILED = "model_publish_failed"
```

### 14.2 处理原则

| 错误 | 行为 |
| --- | --- |
| 配置非法 | 启动失败，输出具体字段 |
| 单帧批次损坏 | 隔离该帧，流程继续 |
| SAM 无 Mask | 拒绝该帧 |
| 有效帧不足 | 等待更多帧，不修改模型 |
| loss/gradient NaN | 停止当前轮并回滚 |
| 网格无效 | 拒绝 candidate |
| 模型发布失败 | 保持旧模型并重试 |
| C++ 模型等待超时 | 继续旧模型或安全结束 |

---

## 15. 测试详细设计

### 15.1 Python 单元测试目录

```text
tests/
  python/
    test_config.py
    test_batch_protocol.py
    test_mask_selector.py
    test_confidence.py
    test_weighted_loss.py
    test_update_gate.py
```

### 15.2 C++ 测试建议

当前工程无明确测试框架。第一阶段可新增轻量可执行测试：

```text
tracker/test/
  communication_protocol_test.cpp
  keyframe_quality_test.cpp
```

若引入 GoogleTest 会增加依赖，可先使用返回码和 `CHECK`，后续再标准化。

### 15.3 必测用例

#### 配置

- 旧 YAML 可加载。
- 新字段覆盖默认值。
- 权重全零时报错。
- strict mode 自动禁止 GT recovery。

#### 通信

- 只有 K 文件时不可消费。
- 八类文件完整但无 `.done` 时不可消费。
- `.done` 存在且文件完整时成功消费。
- 某文件为空时隔离。
- 文件 ID 不一致时不发生跨帧拼接。
- 同一 `.done` 不重复消费。

#### Mask 选择

- SAM score 最高且 tracker 一致时被选中。
- SAM score 较低但 tracker 一致性显著更好时按权重选择。
- 候选分歧越大，最终 confidence 越低。
- 空 Mask 和异常面积被拒绝。

#### 置信度

- 所有分量为 1 时 combined 为 1。
- 任一启用分量接近 0 时 combined 明显下降。
- 禁用 render 分量时融合重新归一化。
- 批次归一化均值接近 1。

#### 优化

- 等权配置与原始逐帧 IoU 结果一致。
- 低权重错误 Mask 对梯度影响减小。
- flatten loss 实际进入 total loss。
- NaN 时恢复初始 state。
- 保存的是 best state 而不是最后 state。

#### 门控

- gate loss 改善且网格有效时接受。
- gate loss 恶化超过阈值时拒绝。
- 退化面比例过高时拒绝。
- candidate 拒绝后 accepted 文件不变。

#### 严格评测

- 只有初始化调用使用 GT。
- 跟踪失败后 pose 不等于注入的当前 GT。
- 离线指标仍可正常计算。

### 15.4 集成测试场景

#### IT-01 单批次闭环

人工构造一帧 C++ 输出，Python 加载、选择 Mask、计算置信度并写日志。

#### IT-02 三轮模型更新

使用短序列完成三次 candidate 优化，至少覆盖一次 accept 和一次 reject。

#### IT-03 进程异常

C++ 写一半后终止，Python 不崩溃；重新运行后完整帧可消费。

#### IT-04 严格序列

运行短 RBOT/MOPED 序列，验证非初始化帧无 GT 状态反馈。

---

## 16. 实施任务分解

### 16.1 变更包 A：严格基线

| 任务 | 预计时间 | 依赖 |
| --- | ---: | --- |
| 暴露 `SetInitPose` 调用链 | 0.5 天 | 无 |
| 修改 Summer 初始化和失败行为 | 0.5 天 | 上项 |
| 修改 `eg_BIT` GT 输入边界 | 0.5 天 | 上项 |
| 增加离线指标和日志检查 | 0.5 天 | 上项 |

完成标准：非初始化帧 GT 改为随机值时，跟踪输出不随随机值变化。

### 16.2 变更包 B：可靠通信

| 任务 | 预计时间 | 依赖 |
| --- | ---: | --- |
| C++ ready 目录和完成标记 | 0.5 天 | 无 |
| 写入返回值和原子 rename | 1 天 | 上项 |
| Python frame ID 扫描和校验 | 1 天 | 上项 |
| 通信集成测试 | 0.5 天 | 上述 |

完成标准：模拟乱序和残缺写入 100 次，无跨帧错配和崩溃。

### 16.3 变更包 C：置信度

| 任务 | 预计时间 | 依赖 |
| --- | ---: | --- |
| 新增数据结构和配置模块 | 0.5 天 | 无 |
| SAM 多候选评分 | 1 天 | 数据结构 |
| tracker/render/view confidence | 1.5 天 | Mask 选择 |
| JSONL 日志 | 0.5 天 | 数据结构 |
| 单元测试 | 0.5 天 | 上述 |

完成标准：每帧四类分量和最终权重可追溯，消融开关有效。

### 16.4 变更包 D：优化与门控

| 任务 | 预计时间 | 依赖 |
| --- | ---: | --- |
| 逐帧 IoU 和加权 loss | 1 天 | C |
| flatten/temporal/best state | 1 天 | 上项 |
| train/gate 划分 | 0.5 天 | 上项 |
| 网格检查和回滚 | 1 天 | 上项 |
| 原子发布 accepted mesh | 0.5 天 | B |
| 集成测试 | 1 天 | 上述 |

完成标准：错误 candidate 不会进入 C++ 模型目录，回滚结果稳定。

### 16.5 变更包 E：关键帧与实验

| 任务 | 预计时间 | 依赖 |
| --- | ---: | --- |
| C++ 关键帧预筛选 | 1 天 | B |
| 标准配置与消融配置 | 0.5 天 | A-D |
| 快速回归脚本 | 0.5 天 | A-D |
| 主实验与重复实验 | 5-7 天 | 全部 |

---

## 17. 合并与发布策略

### 17.1 推荐合并顺序

1. A：严格基线。
2. B：可靠通信。
3. C：置信度。
4. D：优化与门控。
5. E：关键帧与实验。

禁止 A-D 在一个不可拆分提交中同时落地，否则无法定位基线变化来源。

### 17.2 配置档位

```text
legacy:
    strict=false, protocol=1, confidence=false, weighted=false

strict_baseline:
    strict=true, protocol=2, confidence=false, weighted=false

u_bit_lite:
    strict=true, protocol=2, confidence=true, weighted=true, gate=true
```

### 17.3 发布检查

- Python 静态语法检查通过。
- C++ Release 构建通过。
- 单元测试通过。
- 短序列集成测试通过。
- 原始模式回归通过。
- 严格模式 GT 隔离检查通过。
- 配置和环境快照完整。

---

## 18. 验收追踪矩阵

| 上位需求 | 设计章节 | 主要文件 | 验收证据 |
| --- | --- | --- | --- |
| FR-01 严格评测 | 第 6 章 | `eg_BIT.cpp`, `tracker_summer.cc` | GT 隔离测试 |
| FR-02 通信完整性 | 第 7 章 | `communication.*`, `batch_protocol.py` | 乱序/残缺测试 |
| FR-03 Mask 选择 | 第 8 章 | `mask_selector.py`, `sam_utils.py` | 候选评分测试 |
| FR-04 观测置信度 | 第 9 章 | `confidence.py` | 分量日志和单测 |
| FR-05 加权优化 | 第 10 章 | `geometry_optimizer.py` | 梯度影响测试 |
| FR-06 更新门控 | 第 11 章 | `update_gate.py` | accept/reject 测试 |
| FR-07 关键帧 | 第 12 章 | `eg_BIT.cpp` | 筛选统计 |
| FR-08 实验输出 | 第 5.4、15 章 | `experiment_logger.py` | JSONL/指标文件 |

---

## 19. 后续研究扩展接口

第一阶段完成后，以下替换不应影响批次协议和主数据结构：

### 19.1 概率置信度

将：

```python
combine_confidence(...)
```

替换为：

```python
posterior = uncertainty_model.infer(observation)
```

`FrameObservation` 和原始分量日志保持不变。

### 19.2 姿态协方差

向 `ConfidenceComponents` 增加：

```python
pose_covariance: np.ndarray | None
```

加权优化可从标量帧权重扩展到残差协方差加权。

### 19.3 信息增益关键帧

替换 C++ 的角度预筛选器为 Python/C++ 共享的 acquisition score，但继续使用 `frame_id`、观测协议和 accepted view memory。

### 19.4 长期记忆与重定位

在 `FrameObservation` 外新增 canonical memory，不修改现有 Mask、置信度和几何优化接口。

---

## 20. 开发启动清单

开始编码前确认：

- [ ] 选定一条 RBOT 快速回归序列。
- [ ] 选定一条 MOPED 快速回归序列。
- [ ] 保存原始 BIT 输出和配置。
- [ ] 确认 CUDA、PyTorch 和 C++ 构建环境。
- [ ] 创建 `legacy/strict/u_bit_lite` 三套配置。
- [ ] 确认 GT 只用于初始化和离线评测的协议。
- [ ] 为通信协议预留 `ready/` 目录。
- [ ] 确认运行目录不会提交大模型和中间图片。
- [ ] 按 A -> B -> C -> D -> E 顺序实施。

本设计完成后，开发应从“变更包 A：严格基线”开始，不应先实现置信度加权。只有基线协议和批次一致性稳定后，后续性能差异才具有可信的实验意义。
