import importlib.util
import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1]/"scripts"
SCRIPT = SCRIPTS/"merge_stage4_development_inputs.py"
SPEC = importlib.util.spec_from_file_location("merge_stage4_inputs", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class MergeStage4DevelopmentInputsTest(unittest.TestCase):
    def test_merge_csv_rejects_duplicate_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root/"source.csv"
            row = {"object": "obj", "sequence": "00", "seed": 1,
                   "frame_index": 2, "value": 3}
            with source.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                writer.writeheader(); writer.writerow(row)
            with self.assertRaisesRegex(ValueError, "duplicate gate frame"):
                MODULE.merge_csv([source, source], root/"merged.csv")

    def test_merge_jsonl_preserves_unique_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for seed in (1, 2):
                path = root/f"{seed}.jsonl"
                path.write_text(json.dumps({
                    "object": "obj", "sequence": "00", "seed": seed,
                    "frame_index": 2})+"\n")
                paths.append(path)
            output = root/"merged.jsonl"
            self.assertEqual(MODULE.merge_jsonl(paths, output), 2)
            self.assertEqual(len(output.read_text().splitlines()), 2)


if __name__ == "__main__":
    unittest.main()
