import hashlib
import json
import unittest
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
PROTOCOL = PROJECT / "config/stage4_p3_teacher_supplement_s80_82_protocol.json"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Stage4P3TeacherSupplementProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.protocol = json.loads(PROTOCOL.read_text())

    def test_development_and_locked_seeds_are_disjoint(self):
        development = set(self.protocol["supplement_design"]["development_seeds"])
        locked = self.protocol["locked_sets"]
        forbidden = set(locked["previous_p2_report_only_seeds"])
        forbidden.update(locked["formula_confirmation_seeds"])
        forbidden.update(locked["external_independent_seeds"])
        self.assertFalse(development & forbidden)
        self.assertEqual(development, {80, 81, 82})

    def test_full_factorial_and_no_optional_stopping(self):
        design = self.protocol["supplement_design"]
        self.assertEqual(
            design["planned_runs"],
            len(design["development_seeds"]) * len(design["groups"]))
        self.assertTrue(design["run_every_group_seed_pair"])
        self.assertTrue(design["no_early_stopping_after_intermediate_labels"])
        self.assertTrue(design[
            "include_every_completed_run_regardless_of_label_balance"])

    def test_opposite_label_targets_cover_all_single_label_groups(self):
        targets = self.protocol["label_targets_fixed_before_capture"]
        harmful = set(targets[
            "currently_harmful_only_require_at_least_one_safe_event"])
        safe = set(targets[
            "currently_safe_only_require_at_least_one_harmful_event"])
        self.assertFalse(harmful & safe)
        self.assertEqual(harmful | safe, set(
            self.protocol["supplement_design"]["groups"]))
        self.assertEqual(
            self.protocol["primary_acceptance"][
                "newly_mixed_previously_single_label_groups_min"], 2)

    def test_frozen_source_hashes_match_workspace(self):
        expected = self.protocol["frozen_teacher"]["source_hashes"]
        locations = {
            "stage4_p2_control_iou_projection_frozen.json":
                PROJECT / "config/stage4_p2_control_iou_projection_frozen.json",
            "stage4_p1v2_reconfirmation.json":
                PROJECT / "config/stage4_p1v2_reconfirmation.json",
            "run_stage4_p2_replay_capture.sh":
                PROJECT / "scripts/run_stage4_p2_replay_capture.sh",
            "replay_stage4_p2_geometry.py":
                PROJECT / "scripts/replay_stage4_p2_geometry.py",
            "apply_stage4_p2_multimetric_trust_region.py":
                PROJECT / "scripts/apply_stage4_p2_multimetric_trust_region.py",
            "build_stage4_p3_teacher_dataset.py":
                PROJECT / "scripts/build_stage4_p3_teacher_dataset.py",
            "audit_stage4_p3_teacher_dataset.py":
                PROJECT / "scripts/audit_stage4_p3_teacher_dataset.py",
            "audit_stage4_p3_teacher_supplement.py":
                PROJECT / "scripts/audit_stage4_p3_teacher_supplement.py",
            "run_stage4_p3_teacher_supplement_s80_82.sh":
                PROJECT / "scripts/run_stage4_p3_teacher_supplement_s80_82.sh",
        }
        self.assertEqual(set(expected), set(locations))
        for name, path in locations.items():
            self.assertEqual(sha256(path), expected[name], name)

    def test_locked_sets_cannot_be_opened_before_formula_freeze(self):
        locked = self.protocol["locked_sets"]
        self.assertTrue(locked[
            "seed_74_79_must_not_be_captured_read_or_evaluated_before_student_formula_freeze"])
        self.assertFalse(locked[
            "seed_74_79_files_present_at_preregistration"])


if __name__ == "__main__":
    unittest.main()
