import sys
import unittest
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))

from build_stage4_p3_teacher_dataset import build_event


class Stage4P3TeacherDatasetTests(unittest.TestCase):
    def setUp(self):
        self.protocol = {
            "online_numeric_frame_features": {
                "q_mask": ["mean", "min"],
                "pose_risk_drawup": ["mean", "max"],
            },
            "missing_feature_policy": {"anchor_frame_index": 1},
        }
        self.row = {
            "object": "object_a",
            "sequence": "01",
            "seed": 59,
            "candidate_version": 2,
            "training_frame_indices": [1, 20, 30],
            "loo_marginal_utilities": [-0.2, 0.1, 0.3],
            "validation_frames": 3,
            "same_run_original_mean_iou_loss": 0.40,
            "stable_mean_iou_loss": 0.38,
            "mean_iou_loss": 0.35,
            "same_run_original_pose_inconsistency": 0.03,
            "pose_inconsistency": 0.02,
            "same_run_original_temporal_std": 0.04,
            "temporal_std": 0.01,
            "selected_source": "loo",
            "committed": True,
            "proposed_mean_iou_loss": 0.35,
            "multimetric_feasibility": {
                "stable": {"feasible": True},
                "full": {"feasible": True},
                "loo": {"feasible": True},
            },
        }

    def test_build_event_preserves_missing_anchor_without_imputation(self):
        rows = {
            ("object_a", "01", 59, 20): {
                "decision": "accept", "pose_risk_drawup": "0.2"},
            ("object_a", "01", 59, 30): {
                "decision": "downweight", "pose_risk_drawup": "0.6"},
        }
        gates = {
            ("object_a", "01", 59, 20): {"q_mask": "0.8"},
            ("object_a", "01", 59, 30): {"q_mask": "0.4"},
        }
        event = build_event(self.row, rows, gates, self.protocol)
        anchor = event["observations"][0]
        self.assertFalse(anchor["features_available"])
        self.assertTrue(anchor["is_anchor"])
        self.assertIsNone(anchor["features"])
        self.assertFalse(anchor["teacher_keep"])
        self.assertAlmostEqual(
            event["event_features"]["available_feature_fraction"], 2 / 3)
        self.assertAlmostEqual(event["event_features"]["q_mask_mean"], 0.6)
        self.assertAlmostEqual(
            event["event_features"]["pose_risk_drawup_max"], 0.6)

    def test_teacher_labels_are_separate_from_student_features(self):
        rows = {
            ("object_a", "01", 59, index): {
                "decision": "accept", "pose_risk_drawup": "0.2"}
            for index in (1, 20, 30)}
        gates = {
            ("object_a", "01", 59, index): {"q_mask": "0.8"}
            for index in (1, 20, 30)}
        event = build_event(self.row, rows, gates, self.protocol)
        features = event["event_features"]
        self.assertNotIn("selected_source", features)
        self.assertNotIn("selected_iou_gain_vs_full", features)
        self.assertEqual(event["teacher_labels"]["selected_source"], "loo")
        self.assertAlmostEqual(
            event["teacher_labels"]["selected_iou_gain_vs_full"], 0.05)
        self.assertTrue(event["teacher_labels"]["full_harmful_vs_stable"])

    def test_mismatched_observation_labels_are_rejected(self):
        self.row["loo_marginal_utilities"] = [0.1]
        with self.assertRaisesRegex(ValueError, "differ in length"):
            build_event(self.row, {}, {}, self.protocol)


if __name__ == "__main__":
    unittest.main()
