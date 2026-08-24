import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
SCRIPT = SCRIPTS / "replay_stage4_p2_observation_policies.py"
SPEC = importlib.util.spec_from_file_location("stage4_p2_replay", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class Stage4P2ObservationReplayTest(unittest.TestCase):
    def test_split_batches_requires_contiguous_indices(self):
        records = [
            {"batch_index": 0, "frame_index": 1},
            {"batch_index": 1, "frame_index": 2},
            {"batch_index": 0, "frame_index": 3},
        ]
        self.assertEqual(
            [[row["frame_index"] for row in batch]
             for batch in MODULE.split_batches(records)],
            [[1, 2], [3]],
        )
        with self.assertRaises(ValueError):
            MODULE.split_batches([{"batch_index": 1, "frame_index": 1}])

    def test_policy_weights_preserve_warmup_and_parent_reject(self):
        row = {
            "baseline_decision": "accept",
            "decision": "downweight",
            "parent_weight": "1.0",
            "pose_risk_drawup": "0.2",
        }
        self.assertEqual(MODULE.policy_weight("p1v2_hard_nonaccept", row, 0.1), 0.0)
        self.assertAlmostEqual(
            MODULE.policy_weight("p1v2_soft", row, 0.1), 0.5)
        self.assertAlmostEqual(
            MODULE.policy_weight("p1v2_soft_floor75", row, 0.1), 0.75)
        self.assertEqual(MODULE.policy_weight("p1v2_soft", None, 0.1), 1.0)
        rejected = dict(row, baseline_decision="reject", parent_weight="0.0")
        self.assertEqual(MODULE.policy_weight("p1v2_soft", rejected, 0.1), 0.0)
        self.assertEqual(
            MODULE.policy_weight("p1v2_soft_floor75", rejected, 0.1), 0.0)


if __name__ == "__main__":
    unittest.main()
