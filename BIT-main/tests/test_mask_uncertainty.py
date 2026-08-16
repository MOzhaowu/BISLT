import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
from mask_uncertainty import (
    binary_auroc, candidate_disagreement, compute_mask_uncertainty,
    expected_calibration_error,
)


class MaskUncertaintyTest(unittest.TestCase):
    def test_candidate_disagreement(self):
        first = np.array([[1, 1], [0, 0]], dtype=bool)
        second = np.array([[1, 0], [0, 0]], dtype=bool)
        self.assertAlmostEqual(candidate_disagreement([first, second]), 0.5)

    def test_combined_features_are_bounded(self):
        masks = np.array([
            [[1, 1], [0, 0]],
            [[1, 0], [0, 0]],
            [[1, 1], [0, 1]],
        ], dtype=bool)
        features, best = compute_mask_uncertainty(
            masks, [0.9, 0.7, 0.6], logits=np.zeros((3, 2, 2)),
            previous_mask=masks[0], projected_mask=masks[0],
        )
        self.assertEqual(best.shape, (2, 2))
        self.assertGreaterEqual(features["uncertainty"], 0.0)
        self.assertLessEqual(features["uncertainty"], 1.0)

    def test_auroc_and_ece(self):
        self.assertEqual(binary_auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]), 1.0)
        self.assertAlmostEqual(expected_calibration_error([0, 1], [0.0, 1.0]), 0.0)


if __name__ == "__main__":
    unittest.main()
