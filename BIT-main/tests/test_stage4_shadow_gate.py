import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_stage4_shadow_gate.py"
SPEC = importlib.util.spec_from_file_location("build_stage4_shadow_gate", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class Stage4ShadowGateTest(unittest.TestCase):
    def test_gate_threshold_boundaries(self):
        self.assertEqual(MODULE.gate_decision(0.8)[0], "accept")
        self.assertEqual(MODULE.gate_decision(0.3)[0], "downweight")
        self.assertEqual(MODULE.gate_decision(0.299)[0], "reject")
        self.assertAlmostEqual(MODULE.gate_decision(0.4)[1], 0.5)

    def test_failure_probabilities_become_reliabilities(self):
        mask = [{"object": "obj", "sequence": "00", "seed": "42",
                 "frame_index": "7", "failure_label": "0",
                 "failure_probability": "0.2"}]
        pose = [{"object": "obj", "sequence": "00", "seed": "42",
                 "frame_index": "7", "failure_label": "1",
                 "failure_probability": "0.5"}]
        pose_data = [{"object": "obj", "sequence": "00", "seed": 42,
                      "frame_index": 7, "failure_5deg_50mm": 1},
                     {"object": "obj", "sequence": "00", "seed": 42,
                      "frame_index": 8, "failure_5deg_50mm": 1}]
        mask_data = [{"object": "obj", "sequence": "00", "seed": 42,
                      "frame_index": 7, "min_view_angle_deg": 10}]
        rows = MODULE.build_rows(mask, pose, pose_data, mask_data, {},
                                 0.8, 0.3, 2, 2, 1)
        self.assertEqual(len(rows), 1)
        self.assertAlmostEqual(rows[0]["q_mask"], 0.8)
        self.assertAlmostEqual(rows[0]["q_pose"], 0.5)
        self.assertAlmostEqual(rows[0]["base_reliability"], 0.4)
        self.assertEqual(rows[0]["shadow_decision"], "downweight")
        self.assertEqual(rows[0]["unsafe_observation"], 1)
        self.assertEqual(rows[0]["future_pose_failure_any"], 1)

    def test_audit_proxies_do_not_change_primary_decision(self):
        row = {"visible_boundary_ratio": 0.1, "effective_contour_ratio": 0.1,
               "occlusion_ratio": 0.9, "min_view_angle_deg": 1.0}
        self.assertLess(MODULE.visibility_reliability(row), 0.11)
        self.assertAlmostEqual(MODULE.information_reliability(row), 0.05)
        self.assertEqual(MODULE.gate_decision(0.9)[0], "accept")

    def test_update_outcome_alignment(self):
        runs = {("obj", "00", 42): {
            4: {"success_5deg_50mm": "1", "model_reloaded_after_frame": "0"},
            5: {"success_5deg_50mm": "1", "model_reloaded_after_frame": "1"},
            6: {"success_5deg_50mm": "0", "model_reloaded_after_frame": "0"},
            7: {"success_5deg_50mm": "0", "model_reloaded_after_frame": "0"},
        }}
        result = MODULE.update_outcome(("obj", "00", 42, 4), runs, 2, 2)
        self.assertEqual(result["linked_update_frame"], 5)
        self.assertEqual(result["update_degraded"], 1)
        self.assertEqual(result["clearly_bad_update"], 1)


if __name__ == "__main__":
    unittest.main()
