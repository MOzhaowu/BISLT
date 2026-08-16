import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / 'run_gate_sweep.py'
SPEC = importlib.util.spec_from_file_location('run_gate_sweep', SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class GateSweepTest(unittest.TestCase):
    def test_priority_one_presets_isolate_sync_and_gate(self):
        args = SimpleNamespace(
            presets=['original', 'sync', 'iou', 'multi'],
            thresholds=[0.01],
            seeds=[42],
        )
        rows = list(MODULE.cases(args))
        self.assertEqual([row['preset'] for row in rows], args.presets)
        self.assertFalse(rows[0]['consume_once'])
        self.assertFalse(rows[0]['model_validation']['enabled'])
        self.assertTrue(rows[1]['consume_once'])
        self.assertFalse(rows[1]['model_validation']['enabled'])
        self.assertEqual(rows[2]['model_validation']['weights']['pose_consistency'], 0.0)
        self.assertEqual(rows[3]['model_validation']['weights']['pose_consistency'], 0.25)
        self.assertEqual(rows[3]['model_validation']['validation_frames'], 3)


if __name__ == '__main__':
    unittest.main()
