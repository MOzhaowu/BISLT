import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SCRIPT = SCRIPTS / "evaluate_stage4_gate_ablation.py"
SPEC = importlib.util.spec_from_file_location("evaluate_stage4_gate_ablation", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class Stage4GateAblationTest(unittest.TestCase):
    def test_variant_formulas(self):
        row = {"q_mask": 0.8, "q_pose": 0.5,
               "q_visibility_audit": 0.75, "q_information_audit": 0.25}
        self.assertAlmostEqual(MODULE.VARIANTS["mask_pose"](row), 0.4)
        self.assertAlmostEqual(
            MODULE.VARIANTS["mask_pose_visibility_information"](row), 0.075)

    def test_quantile_thresholds_respect_training_constraints(self):
        rows = [
            {"gate_reliability": 0.90, "unsafe_observation": 0},
            {"gate_reliability": 0.80, "unsafe_observation": 0},
            {"gate_reliability": 0.70, "unsafe_observation": 0},
            {"gate_reliability": 0.01, "unsafe_observation": 1},
            {"gate_reliability": 0.10, "unsafe_observation": 1},
            {"gate_reliability": 0.20, "unsafe_observation": 1},
        ]
        accept, reject, rates = MODULE.select_thresholds(
            rows, [0.5, 0.6], [0.1, 0.2], 0.0, 0.34)
        self.assertLess(reject, accept)
        self.assertEqual(rates["safe_reject_rate"], 0.0)
        self.assertLessEqual(rates["safe_nonaccept_rate"], 0.34)

    def test_variant_score_does_not_mutate_source(self):
        source = [{"q_mask": 0.8, "q_pose": 0.5,
                   "q_visibility_audit": 1.0, "q_information_audit": 1.0}]
        output = MODULE.add_variant_score(source, "mask_pose")
        self.assertNotIn("gate_reliability", source[0])
        self.assertAlmostEqual(output[0]["gate_reliability"], 0.4)

    def test_causal_relative_features_use_only_running_history(self):
        rows = [
            {"object": "obj", "sequence": "00", "seed": 42,
             "frame_index": 1, "q_visibility_audit": 0.8,
             "q_information_audit": 0.2},
            {"object": "obj", "sequence": "00", "seed": 42,
             "frame_index": 2, "q_visibility_audit": 0.4,
             "q_information_audit": 0.1},
        ]
        output = MODULE.add_causal_relative_features(rows)
        self.assertAlmostEqual(output[0]["q_visibility_causal_relative"], 1.0)
        self.assertAlmostEqual(output[1]["q_visibility_causal_relative"], 0.5)
        self.assertAlmostEqual(output[1]["q_information_causal_relative"], 0.5)


if __name__ == "__main__":
    unittest.main()
