import sys
import unittest
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))

from audit_stage4_p3_teacher_dataset import event_key, label_counts


class Stage4P3TeacherAuditTests(unittest.TestCase):
    def test_event_key_includes_group_seed_and_candidate_version(self):
        event = {
            "identity": {
                "object": "obj", "sequence": "01", "seed": 71,
                "candidate_version": 3}}
        self.assertEqual(event_key(event), ("obj", "01", 71, 3))

    def test_label_counts_preserves_both_classes(self):
        rows = [{"keep": False}, {"keep": True}, {"keep": True}]
        self.assertEqual(
            label_counts(rows, lambda row: row["keep"]),
            {"negative": 1, "positive": 2})


if __name__ == "__main__":
    unittest.main()
