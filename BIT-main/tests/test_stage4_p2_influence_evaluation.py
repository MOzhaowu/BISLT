import importlib.util
import sys
import unittest
from pathlib import Path


SCRIPT = (Path(__file__).resolve().parents[1] / "scripts" /
          "evaluate_stage4_p2_influence.py")
SPEC = importlib.util.spec_from_file_location("stage4_p2_influence_eval", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class Stage4P2InfluenceEvaluationTest(unittest.TestCase):
    def test_evaluate_returns_json_native_acceptance_values(self):
        common = {
            "object": "obj", "sequence": "00", "seed": 62,
            "candidate_version": 2, "status": "optimized",
            "pose_inconsistency": 0.2, "temporal_std": 0.1,
            "uncertainty": 0.3,
        }
        baseline = dict(common, policy="original", mean_iou_loss=0.4)
        candidate = dict(
            common, policy="candidate", mean_iou_loss=0.3,
            pose_inconsistency=0.1, temporal_std=0.05,
            uncertainty=0.2, influence_elapsed_ms=12.0)
        protocol = {"acceptance": {
            "mean_iou_loss_delta_vs_original_max": 0.0,
            "improved_event_fraction_min": 0.5,
            "mean_pose_inconsistency_delta_max": 0.0,
            "mean_temporal_std_delta_max": 0.0,
            "safe_reject_rate_max": 0.1,
            "safe_nonaccept_rate_max": 0.5,
        }}
        gate = {"candidate": {
            "safe_reject_rate": 0.0, "safe_nonaccept_rate": 0.25}}
        result = MODULE.evaluate(
            [baseline, candidate], gate, protocol, "candidate",
            bootstrap_replicates=20)
        self.assertTrue(result["pass"])
        self.assertEqual(result["mean_marginal_utility_elapsed_ms"], 12.0)
        self.assertIs(type(result["improved_iou_events"]), int)
        self.assertTrue(all(
            type(value) is bool
            for value in result["acceptance_checks"].values()))

    def test_evaluate_uses_loo_elapsed_time(self):
        common = {
            "object": "obj", "sequence": "00", "seed": 65,
            "candidate_version": 2, "status": "optimized",
            "mean_iou_loss": 0.3, "pose_inconsistency": 0.1,
            "temporal_std": 0.05, "uncertainty": 0.2,
        }
        baseline = dict(common, policy="original", mean_iou_loss=0.4)
        candidate = dict(
            common, policy="loo", loo_elapsed_ms=25.0,
            influence_elapsed_ms=None, status="rolled_back",
            committed=False, same_run_original_mean_iou_loss=0.35)
        protocol = {"acceptance": {
            "mean_iou_loss_delta_vs_original_max": 0.0,
            "improved_event_fraction_min": 0.5,
            "mean_pose_inconsistency_delta_max": 0.0,
            "mean_temporal_std_delta_max": 0.0,
            "safe_reject_rate_max": 0.1,
            "safe_nonaccept_rate_max": 0.5,
        }}
        gate = {"candidate": {
            "safe_reject_rate": 0.0, "safe_nonaccept_rate": 0.25}}
        result = MODULE.evaluate(
            [baseline, candidate], gate, protocol, "loo",
            bootstrap_replicates=20)
        self.assertEqual(result["mean_marginal_utility_elapsed_ms"], 25.0)
        self.assertEqual(result["rolled_back_events"], 1)
        self.assertEqual(result["paired_optimized_events"], 1)
        self.assertAlmostEqual(
            result["paired_events"][0]["deltas"]["mean_iou_loss"], -0.05)

    def test_evaluate_accepts_embedded_same_run_baseline_without_original_row(self):
        candidate = {
            "object": "obj", "sequence": "00", "seed": 68,
            "candidate_version": 2, "policy": "candidate",
            "status": "committed", "committed": True,
            "mean_iou_loss": 0.3, "pose_inconsistency": 0.1,
            "temporal_std": 0.05, "uncertainty": 0.2,
            "same_run_original_mean_iou_loss": 0.4,
            "same_run_original_pose_inconsistency": 0.2,
            "same_run_original_temporal_std": 0.1,
            "same_run_original_uncertainty": 0.3,
        }
        protocol = {"acceptance": {
            "mean_iou_loss_delta_vs_original_max": 0.0,
            "improved_event_fraction_min": 0.5,
            "mean_pose_inconsistency_delta_max": 0.0,
            "mean_temporal_std_delta_max": 0.0,
            "safe_reject_rate_max": 0.1,
            "safe_nonaccept_rate_max": 0.5,
        }}
        gate = {"candidate": {
            "safe_reject_rate": 0.0, "safe_nonaccept_rate": 0.25}}
        result = MODULE.evaluate(
            [candidate], gate, protocol, "candidate",
            bootstrap_replicates=20)
        self.assertEqual(result["paired_optimized_events"], 1)
        self.assertAlmostEqual(
            result["mean_deltas"]["mean_iou_loss"], -0.1)


if __name__ == "__main__":
    unittest.main()
