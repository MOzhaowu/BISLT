import sys
import unittest
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from stage4_candidate_regularization import (  # noqa: E402
    candidate_regularization_terms,
    cosine_similarity,
    project_direction_to_halfspaces,
    regularized_candidate_loss,
    se3_axis_perturbations,
)


class Stage4CandidateRegularizationTest(unittest.TestCase):
    def test_terms_match_gate_metric_definitions(self):
        nominal = torch.tensor([0.2, 0.4])
        perturbed = torch.tensor([[0.1, 0.3], [0.5, 0.6]])
        pose, temporal = candidate_regularization_terms(
            nominal, perturbed, torch.ones(2))
        self.assertAlmostEqual(pose.item(), 0.05, places=6)
        self.assertAlmostEqual(temporal.item(), 0.1, places=6)

    def test_terms_backpropagate(self):
        nominal = torch.tensor([0.4, 0.2], requires_grad=True)
        perturbed = torch.tensor(
            [[0.1, 0.5], [0.3, 0.4]], requires_grad=True)
        pose, temporal = candidate_regularization_terms(
            nominal, perturbed, torch.ones(2))
        (pose + temporal).backward()
        self.assertGreater(float(nominal.grad.abs().sum()), 0.0)
        self.assertGreater(float(perturbed.grad.abs().sum()), 0.0)

    def test_zero_weights_preserve_original_objective(self):
        result = regularized_candidate_loss(
            torch.tensor(0.3), torch.tensor(0.2),
            torch.tensor(9.0), torch.tensor(8.0),
            pose_weight=0.0, temporal_weight=0.0)
        self.assertAlmostEqual(result.item(), 0.32, places=6)

    def test_pose_probes_cover_signed_six_dof(self):
        probes = se3_axis_perturbations(
            2.0, 0.005, torch.device("cpu"), torch.float32)
        self.assertEqual(tuple(probes.shape), (12, 4, 4))
        self.assertEqual(int((probes[:, :3, 3] != 0).sum()), 6)

    def test_single_frame_temporal_term_matches_gate_zero(self):
        _, temporal = candidate_regularization_terms(
            torch.tensor([0.2]), None, torch.ones(1))
        self.assertEqual(temporal.item(), 0.0)

    def test_projection_removes_conflicts_for_multiple_constraints(self):
        direction = torch.tensor([-1.0, -2.0])
        normals = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])]
        projected = project_direction_to_halfspaces(direction, normals)
        self.assertGreaterEqual(torch.dot(projected, normals[0]).item(), -1e-7)
        self.assertGreaterEqual(torch.dot(projected, normals[1]).item(), -1e-7)
        self.assertTrue(torch.allclose(projected, torch.zeros(2), atol=1e-6))

    def test_projection_preserves_already_feasible_direction(self):
        direction = torch.tensor([2.0, 1.0])
        normals = [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])]
        projected = project_direction_to_halfspaces(direction, normals)
        self.assertTrue(torch.equal(projected, direction))

    def test_cosine_similarity_handles_zero_gradient(self):
        self.assertEqual(
            cosine_similarity(torch.zeros(2), torch.ones(2)), 0.0)


if __name__ == "__main__":
    unittest.main()
