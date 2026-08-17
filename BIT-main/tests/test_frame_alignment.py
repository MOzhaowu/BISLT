import csv
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
sys.path.insert(0, str(ROOT / "scripts"))

from communication import DataGroups
from label_mask_uncertainty import (
    align_records, boundary_f1, mask_iou, select_first_observation_per_frame,
)


def group(frame_indices):
    count = len(frame_indices)
    return DataGroups(
        rgbs=[None] * count, probs=[None] * count, masks=[None] * count,
        Ts=np.zeros((count, 4, 4)), Ks=np.zeros((count, 9)),
        origin_rois=np.zeros((count, 4)), target_rois=np.zeros((count, 4)),
        target_sizes=np.zeros((count, 2)), frame_indices=list(frame_indices),
    )


class FrameAlignmentTest(unittest.TestCase):
    def test_data_groups_preserve_frame_indices_when_accumulated(self):
        first = group([0, 1])
        first.add(group([9, 14]))
        self.assertEqual(first.frame_indices_, [0, 1, 9, 14])

    def test_uncertainty_is_labeled_by_real_frame_index(self):
        observations = [{"observation_index": 3, "frame_index": 14}]
        diagnostics = [{
            "frame_index": "14",
            "rotation_error_deg": "5.1",
            "translation_error_mm": "2.0",
            "success_5deg_50mm": "0",
        }]
        aligned = align_records(observations, diagnostics)
        self.assertEqual(aligned[0]["failure_5deg_50mm"], 1)
        self.assertEqual(aligned[0]["rotation_error_deg"], 5.1)

    def test_first_frame_uses_projection_fallback(self):
        observations = [{
            "observation_index": 0, "frame_index": 0,
            "temporal_inconsistency": None, "projection_inconsistency": 0.4,
        }]
        diagnostics = [{
            "frame_index": "0", "rotation_error_deg": "1",
            "translation_error_mm": "1", "success_5deg_50mm": "1",
        }]
        aligned = align_records(observations, diagnostics)
        self.assertEqual(aligned[0]["temporal_missing"], 1)
        self.assertEqual(aligned[0]["temporal_or_projection_inconsistency"], 0.4)

    def test_mask_quality_metrics(self):
        target = np.array([[1, 1], [0, 0]], dtype=bool)
        self.assertEqual(mask_iou(target, target), 1.0)
        self.assertEqual(boundary_f1(target, target), 1.0)
        self.assertEqual(mask_iou(np.zeros_like(target), target), 0.0)

    def test_frame_dataset_uses_first_causal_observation(self):
        records = [
            {"frame_index": 0, "observation_index": 2, "uncertainty": 0.2},
            {"frame_index": 1, "observation_index": 3, "uncertainty": 0.3},
            {"frame_index": 0, "observation_index": 0, "uncertainty": 0.8},
        ]
        selected = select_first_observation_per_frame(records)
        self.assertEqual([row["frame_index"] for row in selected], [0, 1])
        self.assertEqual(selected[0]["observation_index"], 0)
        self.assertEqual(selected[0]["uncertainty"], 0.8)
        self.assertEqual(selected[0]["observation_count_for_frame"], 2)


if __name__ == "__main__":
    unittest.main()
