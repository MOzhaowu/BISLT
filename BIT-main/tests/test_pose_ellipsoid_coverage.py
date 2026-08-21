import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from evaluate_pose_ellipsoid_coverage import loso_scale, se3_error


class PoseEllipsoidCoverageTest(unittest.TestCase):
    def test_identity_pose_has_zero_se3_error(self):
        pose = (0, np.eye(3), np.zeros(3))
        np.testing.assert_allclose(se3_error(pose, pose, 2.0), np.zeros(6))

    def test_loso_scale_never_uses_held_out_sequence(self):
        records = [
            {"group": "a/00", "squared_distance": 1.0},
            {"group": "a/00", "squared_distance": 2.0},
            {"group": "b/00", "squared_distance": 10.0},
            {"group": "b/00", "squared_distance": 20.0},
            {"group": "c/00", "squared_distance": 100.0},
            {"group": "c/00", "squared_distance": 200.0},
        ]
        calibrated, folds = loso_scale(records)
        self.assertEqual(len(calibrated), len(records))
        self.assertGreater(folds["a/00"]["training_scale"],
                           folds["c/00"]["training_scale"])
        for row in calibrated:
            self.assertIn("calibrated_squared_distance", row)


if __name__ == "__main__":
    unittest.main()
