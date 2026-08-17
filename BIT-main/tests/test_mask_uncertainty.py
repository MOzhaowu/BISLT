import sys
import unittest
from pathlib import Path
import json
import tempfile
from types import SimpleNamespace
from unittest import mock

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

    def test_segment_records_candidates_without_changing_mask_zero(self):
        import sam_utils

        masks = np.array([
            [[0, 0], [0, 0]],
            [[1, 1], [1, 1]],
            [[1, 0], [0, 1]],
        ], dtype=bool)

        class Predictor:
            def set_image(self, image):
                pass

            def predict(self, **kwargs):
                return masks, np.array([0.1, 0.9, 0.2]), np.zeros((3, 2, 2))

        groups = SimpleNamespace(
            probs_=[np.zeros((2, 2), dtype=np.uint8)],
            rgbs_=[np.zeros((2, 2, 3), dtype=np.uint8)],
            origin_rois_=[np.array([0, 0, 2, 2])],
            target_rois_=[np.array([0, 0, 2, 2])],
            target_sizes_=[np.array([2, 2])],
        )
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(sam_utils, "show_prompt"), \
                mock.patch.object(sam_utils, "ResizeMask", side_effect=lambda mask, *args, **kwargs: mask.astype(np.uint8) * 255):
            output = sam_utils.segment(
                Predictor(), 2, 2, groups, {}, 0, uncertainty_dir=directory
            )
            self.assertTrue(np.array_equal(output[0], masks[0].astype(np.uint8) * 255))
            archive = np.load(Path(directory) / "observation_000000.npz")
            self.assertEqual(archive["masks"].shape, (3, 2, 2))
            self.assertTrue(np.allclose(archive["scores"], [0.1, 0.9, 0.2]))
            self.assertEqual(archive["logits"].dtype, np.float32)
            record = json.loads((Path(directory) / "mask_uncertainty.jsonl").read_text())
            self.assertEqual(record["selected_index"], 0)
            self.assertEqual(record["output_candidate_index"], 0)
            self.assertEqual(record["best_index"], 1)
            for name in ("candidate_disagreement", "boundary_entropy", "temporal_inconsistency", "projection_inconsistency"):
                self.assertIn(name, record)


if __name__ == "__main__":
    unittest.main()
