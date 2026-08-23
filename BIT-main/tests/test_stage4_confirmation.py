import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SCRIPT = SCRIPTS / "evaluate_stage4_confirmation.py"
SPEC = importlib.util.spec_from_file_location("evaluate_stage4_confirmation", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class Stage4ConfirmationTest(unittest.TestCase):
    def test_frozen_logistic_prediction(self):
        frozen = {"features": ["x"], "model": {
            "parameters": [0.0, 1.0], "medians": [0.0],
            "means": [0.0], "scales": [2.0]}}
        self.assertAlmostEqual(MODULE.predict_failure({"x": 0.0}, frozen), 0.5)
        self.assertGreater(MODULE.predict_failure({"x": 2.0}, frozen), 0.5)

    def test_method_formulas_are_fixed(self):
        row = {"q_mask": 0.8, "q_pose": 0.5,
               "q_visibility_causal_relative": 0.75,
               "q_information_causal_relative": 0.25}
        self.assertAlmostEqual(MODULE.method_reliability(row, "mask_pose"), 0.4)
        self.assertAlmostEqual(MODULE.method_reliability(
            row, "mask_pose_causal_visibility"), 0.3)
        self.assertAlmostEqual(MODULE.method_reliability(
            row, "mask_pose_causal_information"), 0.1)


if __name__ == "__main__":
    unittest.main()
