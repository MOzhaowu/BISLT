import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from stage4_influence import (  # noqa: E402
    cosine_alignments,
    gradient_matching_weights,
    influence_rescue_weights,
    monotone_validation_commit,
    select_lowest_validation_geometry,
    select_multimetric_validation_geometry,
    short_loo_filter_weights,
    short_loo_softmax_weights,
)


class Stage4InfluenceTest(unittest.TestCase):
    def test_cosine_alignment_sign_and_zero_gradient(self):
        scores = cosine_alignments(
            np.array([1.0, 0.0]),
            [np.array([2.0, 0.0]), np.array([-3.0, 0.0]), np.zeros(2)],
        )
        self.assertTrue(np.allclose(scores, [1.0, -1.0, 0.0]))

    def test_rescue_only_restores_demoted_parent_accept(self):
        rows = [
            {"baseline_decision": "accept", "decision": "downweight"},
            {"baseline_decision": "reject", "decision": "reject"},
            {"baseline_decision": "accept", "decision": "downweight"},
            None,
        ]
        weights = influence_rescue_weights(
            rows, [0.2, 0.0, 0.3, 1.0], [0.1, 1.0, -0.1, 1.0])
        self.assertEqual(weights, [1.0, 0.0, 0.3, 1.0])

    def test_gradient_matching_suppresses_opposed_gradient(self):
        weights = gradient_matching_weights(
            np.array([1.0, 0.0]),
            [np.array([1.0, 0.0]), np.array([-1.0, 0.0])],
            [0.2, 1.0],
            ridge=1.0,
        )
        self.assertGreater(weights[0], 0.2)
        self.assertLess(weights[1], 1.0)
        locked = gradient_matching_weights(
            np.ones(2), [np.ones(2)], [0.0], locked_zero=[True])
        self.assertEqual(locked, [0.0])
    def test_short_loo_weights_use_robust_relative_scale_and_lock_rejects(self):
        rows = [
            {"baseline_decision": "accept"},
            {"baseline_decision": "accept"},
            {"baseline_decision": "reject"},
        ]
        weights = short_loo_softmax_weights(rows, [1.0, 0.0, 10.0])
        self.assertAlmostEqual(weights[0], 1.0)
        self.assertAlmostEqual(weights[1], np.exp(-2.0))
        self.assertEqual(weights[2], 0.0)

    def test_short_loo_filter_removes_harmful_and_parent_rejected_rows(self):
        rows = [
            {"baseline_decision": "accept"},
            {"baseline_decision": "accept"},
            {"baseline_decision": "reject"},
            None,
        ]
        weights = short_loo_filter_weights(rows, [0.1, 0.0, 2.0, -0.1])
        self.assertEqual(weights, [1.0, 0.0, 0.0, 0.0])

    def test_monotone_validation_commit_and_secondary_guards(self):
        stable = {
            "mean_iou_loss": 0.4, "pose_inconsistency": 0.2,
            "temporal_std": 0.1,
        }
        improved = {
            "mean_iou_loss": 0.3, "pose_inconsistency": 0.1,
            "temporal_std": 0.05,
        }
        self.assertTrue(monotone_validation_commit(
            stable, improved, max_pose_regression=0.0,
            max_temporal_regression=0.0)["committed"])
        regressed = dict(improved, mean_iou_loss=0.41)
        decision = monotone_validation_commit(stable, regressed)
        self.assertFalse(decision["committed"])
        self.assertEqual(decision["reason"], "iou_monotone")

    def test_select_lowest_validation_geometry_prefers_stable_ties(self):
        stable = {"mean_iou_loss": 0.4}
        choice = select_lowest_validation_geometry(stable, {
            "full": {"mean_iou_loss": 0.3},
            "loo": {"mean_iou_loss": 0.2},
        })
        self.assertEqual(choice["selected_source"], "loo")
        self.assertTrue(choice["committed"])
        tie = select_lowest_validation_geometry(stable, {
            "full": {"mean_iou_loss": 0.4}})
        self.assertEqual(tie["selected_source"], "stable")

    def test_multimetric_selector_filters_pose_and_temporal_regressions(self):
        full = {"mean_iou_loss": 0.4, "pose_inconsistency": 0.2,
                "temporal_std": 0.1}
        stable = {"mean_iou_loss": 0.2, "pose_inconsistency": 0.3,
                  "temporal_std": 0.1}
        loo = {"mean_iou_loss": 0.3, "pose_inconsistency": 0.1,
               "temporal_std": 0.05}
        choice = select_multimetric_validation_geometry(stable, full, loo)
        self.assertEqual(choice["selected_source"], "loo")
        self.assertFalse(choice["feasibility"]["stable"]["feasible"])
        self.assertTrue(choice["feasibility"]["full"]["feasible"])


    def test_length_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            influence_rescue_weights([], [1.0], [], 0.0)


if __name__ == "__main__":
    unittest.main()
