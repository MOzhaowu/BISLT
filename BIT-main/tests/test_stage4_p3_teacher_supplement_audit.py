import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))

from audit_stage4_p3_teacher_supplement import (
    capture_status,
    missing_feature_counts,
)


class Stage4P3TeacherSupplementAuditTests(unittest.TestCase):
    def test_capture_status_requires_identity_completeness_and_zero_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            run = Path(directory) / "obj" / "01" / "p2_capture_s80"
            run.mkdir(parents=True)
            (run / "manifest.json").write_text(json.dumps({
                "object": "obj", "sequence": "01", "seed": 80}))
            (run / "completeness.json").write_text(json.dumps({
                "valid": 1, "invalid": 0}))
            (run / "exit_code.txt").write_text("0\n")
            result = capture_status(Path(directory), ["obj/01"], [80])
            self.assertTrue(result[0]["valid"])
            (run / "exit_code.txt").write_text("1\n")
            result = capture_status(Path(directory), ["obj/01"], [80])
            self.assertFalse(result[0]["valid"])

    def test_missing_features_separates_anchor_warmup_and_unexplained(self):
        observations = [
            {"frame_index": 1, "features_available": False,
             "is_anchor": True},
            {"frame_index": 3, "features_available": False,
             "is_anchor": False},
            {"frame_index": 9, "features_available": False,
             "is_anchor": False},
            {"frame_index": 10, "features_available": True,
             "is_anchor": False},
        ]
        counts = missing_feature_counts(
            [{"observations": observations}], warmup_frames=5)
        self.assertEqual(counts, {
            "available": 1,
            "missing": 3,
            "missing_anchor": 1,
            "missing_pose_warmup_nonanchor": 1,
            "missing_unexplained": 1,
        })


if __name__ == "__main__":
    unittest.main()
