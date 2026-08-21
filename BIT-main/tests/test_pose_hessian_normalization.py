import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_normalized_pose_uncertainty_dataset import normalize_hessian


class PoseHessianNormalizationTest(unittest.TestCase):
    def row(self, matrix, length=2.0):
        result = {f"hessian_{r}{c}": matrix[r, c]
                  for r in range(6) for c in range(6)}
        result["object_characteristic_length"] = length
        return result

    def test_translation_block_is_scaled_by_length_squared(self):
        result = normalize_hessian(self.row(np.eye(6)), 1e-6)
        self.assertAlmostEqual(result["normalized_hessian_00"], 1.0)
        self.assertAlmostEqual(result["normalized_hessian_33"], 4.0)

    def test_covariance_is_symmetric_positive_semidefinite(self):
        matrix = np.diag([4.0, 3.0, 2.0, 1.0, 0.0, -1e-8])
        result = normalize_hessian(self.row(matrix), 1e-6)
        covariance = np.asarray([[result[f"normalized_covariance_{r}{c}"]
                                  for c in range(6)] for r in range(6)])
        self.assertTrue(np.allclose(covariance, covariance.T))
        self.assertGreaterEqual(np.linalg.eigvalsh(covariance).min(), -1e-8)
        self.assertEqual(result["psd_clipped_eigenvalues"], 2)


if __name__ == "__main__":
    unittest.main()
