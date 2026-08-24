import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = (Path(__file__).resolve().parents[1] / "scripts" /
          "finalize_stage4_p2_projection.py")
SPEC = importlib.util.spec_from_file_location("projection_finalize", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class ProjectionFinalizationTest(unittest.TestCase):
    def test_development_requires_evaluation_projection_and_feasible_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            replay = root / "replay.jsonl"
            evaluation = root / "evaluation.json"
            protocol = root / "protocol.json"
            row = {
                "policy": MODULE.SOURCE_POLICY,
                "proposed_mean_iou_loss": 0.2,
                "proposed_pose_inconsistency": 0.1,
                "same_run_original_pose_inconsistency": 0.2,
                "proposed_temporal_std": 0.1,
                "same_run_original_temporal_std": 0.2,
                "proposed_optimization_regularization": {
                    "gradient_projection_summary": {
                        "steps": 2, "projected_steps": 1,
                        "pose_active_steps": 1, "temporal_active_steps": 1,
                        "all_steps_feasible": True,
                    }
                },
            }
            replay.write_text(json.dumps(row) + "\n")
            evaluation.write_text(json.dumps({
                "pass": True, "paired_optimized_events": 1,
                "improved_iou_events": 1, "mean_deltas": {},
                "mean_iou_delta_bootstrap_ci95": [-1.0, -0.1],
                "by_seed": {}, "acceptance_checks": {},
            }))
            protocol.write_text(json.dumps({
                "acceptance": {"loo_feasible_events_min": 1}}))
            result = MODULE.finalize(
                replay, evaluation, protocol, "development")
            self.assertTrue(result["pass"])
            self.assertEqual(result["projection"]["optimizer_steps"], 2)

    def test_independent_rejects_projection_violation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            replay = root / "replay.jsonl"
            evaluation = root / "evaluation.json"
            protocol = root / "protocol.json"
            replay.write_text(json.dumps({
                "policy": MODULE.SOURCE_POLICY,
                "proposed_mean_iou_loss": 0.2,
                "proposed_pose_inconsistency": 0.1,
                "same_run_original_pose_inconsistency": 0.2,
                "proposed_temporal_std": 0.1,
                "same_run_original_temporal_std": 0.2,
                "proposed_optimization_regularization": {
                    "gradient_projection_summary": {
                        "steps": 1, "projected_steps": 1,
                        "pose_active_steps": 1, "temporal_active_steps": 0,
                        "all_steps_feasible": False,
                    }
                },
            }) + "\n")
            evaluation.write_text(json.dumps({
                "pass": True, "paired_optimized_events": 1,
                "improved_iou_events": 1, "mean_deltas": {},
                "mean_iou_delta_bootstrap_ci95": [-1.0, -0.1],
                "by_seed": {}, "acceptance_checks": {},
            }))
            protocol.write_text(json.dumps({"acceptance": {
                "loo_feasible_events_min": 1}}))
            result = MODULE.finalize(
                replay, evaluation, protocol, "independent")
            self.assertFalse(result["pass"])


if __name__ == "__main__":
    unittest.main()
