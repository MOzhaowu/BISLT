import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SCRIPT = SCRIPTS / "select_stage4_gate_thresholds_loso.py"
SPEC = importlib.util.spec_from_file_location("select_stage4_gate_thresholds_loso", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class Stage4ThresholdSelectionTest(unittest.TestCase):
    def test_constraints_control_safe_frames(self):
        rows = [
            {"base_reliability": 0.9, "unsafe_observation": 0},
            {"base_reliability": 0.8, "unsafe_observation": 0},
            {"base_reliability": 0.02, "unsafe_observation": 1},
            {"base_reliability": 0.2, "unsafe_observation": 1},
        ]
        accept, reject, result = MODULE.select_thresholds(
            rows, [0.3, 0.7], [0.01, 0.05], 0.0, 0.0)
        self.assertEqual(accept, 0.3)
        self.assertEqual(reject, 0.05)
        self.assertEqual(result["safe_reject_rate"], 0.0)
        self.assertEqual(result["safe_nonaccept_rate"], 0.0)

    def test_invalid_grid_reports_no_feasible_policy(self):
        rows = [
            {"base_reliability": 0.01, "unsafe_observation": 0},
            {"base_reliability": 0.02, "unsafe_observation": 1},
        ]
        with self.assertRaises(ValueError):
            MODULE.select_thresholds(rows, [0.5], [0.1], 0.0, 0.0)


if __name__ == "__main__":
    unittest.main()
