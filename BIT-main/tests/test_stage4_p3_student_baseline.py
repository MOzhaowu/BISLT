import sys
import unittest
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))

from evaluate_stage4_p3_student_baseline import (
    binary_auroc,
    fit_preprocessor,
    observation_rows,
    transform,
)


class Stage4P3StudentBaselineTests(unittest.TestCase):
    def test_binary_auroc_handles_ranking_and_ties(self):
        self.assertEqual(binary_auroc([0, 1], [0.1, 0.9]), 1.0)
        self.assertEqual(binary_auroc([0, 1], [0.5, 0.5]), 0.5)

    def test_preprocessor_imputes_from_training_fold_only(self):
        train = [[1.0, np.nan], [3.0, 8.0], [5.0, 10.0]]
        preprocessor = fit_preprocessor(train)
        transformed = transform([[np.nan, np.nan]], preprocessor)
        self.assertAlmostEqual(transformed[0, 0], 0.0)
        self.assertAlmostEqual(transformed[0, 1], 0.0)

    def test_observation_features_exclude_identity_and_teacher_labels(self):
        event = {
            "identity": {
                "object": "obj", "sequence": "01", "seed": 59,
                "candidate_version": 2},
            "group": "obj/01",
            "event_features": {"candidate_frame_count": 2},
            "observations": [{
                "frame_index": 1,
                "features_available": False,
                "is_anchor": True,
                "features": None,
                "teacher_keep": False,
            }],
            "teacher_labels": {"selected_source": "stable"},
        }
        rows = observation_rows(
            [event], ["candidate_frame_count"], ["q_mask"])
        self.assertEqual(rows[0]["features"], [2.0, 0.0, 1.0, np.nan])
        self.assertNotIn("teacher_labels", rows[0])
        self.assertNotIn("object", rows[0]["features"])


if __name__ == "__main__":
    unittest.main()
