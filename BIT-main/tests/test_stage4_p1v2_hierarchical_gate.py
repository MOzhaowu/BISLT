import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SCRIPT = SCRIPTS / "evaluate_stage4_p1v2_hierarchical_gate.py"
SPEC = importlib.util.spec_from_file_location("stage4_p1v2", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class Stage4P1V2HierarchicalGateTest(unittest.TestCase):
    def test_demotion_never_promotes_or_creates_reject(self):
        rows = [
            {"baseline_decision": "accept", "unsafe_observation": 1},
            {"baseline_decision": "downweight", "unsafe_observation": 0},
            {"baseline_decision": "reject", "unsafe_observation": 1},
        ]
        output = MODULE.apply_demotion(rows, [0.9, 0.9, 0.9], 0.5)
        self.assertEqual([row["decision"] for row in output],
                         ["downweight", "downweight", "reject"])

    def test_threshold_respects_safe_nonaccept_budget(self):
        rows = [
            {"baseline_decision": "accept", "unsafe_observation": 0},
            {"baseline_decision": "accept", "unsafe_observation": 0},
            {"baseline_decision": "accept", "unsafe_observation": 1},
            {"baseline_decision": "downweight", "unsafe_observation": 1},
        ]
        threshold, metrics = MODULE.select_demotion_threshold(
            rows, [0.1, 0.2, 0.9, 0.3], 0.5)
        self.assertLessEqual(metrics["safe_nonaccept_rate"], 0.5)
        self.assertEqual(metrics["protected_unsafe_recall"], 1.0)
        self.assertLessEqual(threshold, 0.9)

    def test_decision_metrics(self):
        rows = [
            {"decision": "downweight", "unsafe_observation": 1},
            {"decision": "reject", "unsafe_observation": 1},
            {"decision": "accept", "unsafe_observation": 0},
        ]
        metrics = MODULE.decision_metrics(rows)
        self.assertEqual(metrics["protected_unsafe_recall"], 1.0)
        self.assertEqual(metrics["safe_reject_rate"], 0.0)
        self.assertEqual(metrics["safe_nonaccept_rate"], 0.0)


    def test_causal_risk_features_use_only_run_history(self):
        rows = [
            {"object": "obj", "sequence": "00", "seed": 1,
             "frame_index": 2, "base_risk_logit": 0.7},
            {"object": "obj", "sequence": "00", "seed": 1,
             "frame_index": 1, "base_risk_logit": 0.2},
            {"object": "obj", "sequence": "00", "seed": 2,
             "frame_index": 1, "base_risk_logit": 5.0},
        ]
        output = MODULE.add_causal_risk_features(rows)
        by_key = {(row["seed"], row["frame_index"]): row for row in output}
        self.assertEqual(by_key[(1, 1)]["base_risk_drawup"], 0.0)
        self.assertAlmostEqual(by_key[(1, 2)]["base_risk_drawup"], 0.5)
        self.assertAlmostEqual(
            by_key[(1, 2)]["base_risk_delta_from_first"], 0.5)
        self.assertEqual(by_key[(2, 1)]["base_risk_drawup"], 0.0)
        self.assertAlmostEqual(by_key[(1, 2)]["pose_risk_drawup"], 0.5)

    def test_dense_pose_context_uses_only_preceding_frames(self):
        feature = (
            "log_normalized_covariance_trace_relative_to_initial_median")
        rows = [
            {"object": "obj", "sequence": "00", "seed": 1,
             "frame_index": 3, feature: 1.0},
            {"object": "obj", "sequence": "00", "seed": 1,
             "frame_index": 1, feature: 0.0},
            {"object": "obj", "sequence": "00", "seed": 1,
             "frame_index": 2, feature: 2.0},
            {"object": "obj", "sequence": "00", "seed": 2,
             "frame_index": 1, feature: 9.0},
        ]
        output = MODULE.add_dense_pose_context(rows)
        by_key = {(row["seed"], row["frame_index"]): row for row in output}
        self.assertEqual(by_key[(1, 1)]["dense_pose_risk_percentile"], 0.5)
        self.assertEqual(by_key[(1, 1)]["dense_pose_risk_robust_z"], 0.0)
        self.assertEqual(by_key[(1, 1)]["dense_pose_risk_drawup"], 0.0)
        self.assertAlmostEqual(
            by_key[(1, 2)]["dense_pose_risk_percentile"], 0.75)
        self.assertEqual(by_key[(1, 2)]["dense_pose_risk_drawup"], 2.0)
        self.assertAlmostEqual(
            by_key[(1, 3)]["dense_pose_risk_percentile"], 0.5)
        self.assertEqual(by_key[(2, 1)]["dense_pose_risk_drawup"], 0.0)

if __name__ == "__main__":
    unittest.main()
