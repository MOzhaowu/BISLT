import sys
import unittest
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples'))
from validation_selection import (  # noqa: E402
    ValidationBuffer,
    rotation_distance_deg,
)


def pose(yaw_deg=0.0, x=0.0):
    angle = np.deg2rad(yaw_deg)
    c, s = np.cos(angle), np.sin(angle)
    result = np.eye(4, dtype=np.float32)
    result[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    result[0, 3] = x
    return result


class ValidationSelectionTest(unittest.TestCase):
    def test_rotation_distance(self):
        self.assertAlmostEqual(rotation_distance_deg(pose(), pose(90)), 90.0, places=4)

    def test_buffer_deduplicates_and_prefers_diversity(self):
        masks = np.ones((4, 4, 8, 8), dtype=np.float32)
        poses = np.stack([pose(), pose(), pose(60), pose(120)])
        intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None], 4, axis=0)
        buffer = ValidationBuffer(capacity=8)
        buffer.add(masks, poses, intrinsics)
        self.assertEqual(len(buffer), 3)
        selected = buffer.select(2, quality_weight=0.0)
        self.assertEqual(len(selected), 2)
        separation = rotation_distance_deg(selected[0]['pose'], selected[1]['pose'])
        self.assertGreaterEqual(separation, 60.0)

    def test_capacity_evicts_oldest(self):
        masks = np.ones((3, 4, 8, 8), dtype=np.float32)
        poses = np.stack([pose(0), pose(30), pose(60)])
        intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None], 3, axis=0)
        buffer = ValidationBuffer(capacity=2)
        buffer.add(masks, poses, intrinsics)
        self.assertEqual(len(buffer), 2)


if __name__ == '__main__':
    unittest.main()
