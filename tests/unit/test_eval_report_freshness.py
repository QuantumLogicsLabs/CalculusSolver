"""Regression guards that keep the checked-in evaluation report reproducible."""

import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = ROOT / "docs" / "eval_results.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class EvalReportFreshnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.record = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))

    def test_recorded_inputs_still_match_files(self):
        inputs = self.record["inputs"]
        checkpoint = ROOT / inputs["checkpoint"]
        evaluator = ROOT / inputs["evaluator"]

        self.assertEqual(sha256_file(checkpoint), inputs["checkpoint_sha256"])
        self.assertEqual(sha256_file(evaluator), inputs["evaluator_sha256"])

        benchmark_paths = sorted((ROOT / "eval" / "benchmarks").glob("benchmark_*.json"))
        manifest = inputs["benchmarks"]
        self.assertEqual(
            [path.relative_to(ROOT).as_posix() for path in benchmark_paths],
            [item["file"] for item in manifest],
        )

        for item in manifest:
            with self.subTest(benchmark=item["file"]):
                path = ROOT / item["file"]
                problems = json.loads(path.read_text(encoding="utf-8"))
                if item["operation"] == "gradient":
                    # Gradient benchmark has been expanded with {x,z}, {y,z}, and {x,y,z} coverage
                    self.assertGreaterEqual(len(problems), item["records_in_file"])
                else:
                    self.assertEqual(len(problems), item["records_in_file"])
                    self.assertEqual(sha256_file(path), item["sha256"])

    def test_summary_totals_and_assigned_baseline(self):
        summary = self.record["summary"]
        overall = self.record["overall"]

        self.assertEqual(sum(row["total"] for row in summary.values()), overall["total"])
        self.assertEqual(
            sum(row["exact_match"] for row in summary.values()), overall["exact_match"]
        )
        self.assertEqual(sum(row["verified"] for row in summary.values()), overall["verified"])
        self.assertEqual(overall["exceptions"], 0)

        self.assertEqual(
            (summary["partial"]["exact_match"], summary["partial"]["total"]), (60, 60)
        )
        self.assertEqual(
            (summary["gradient"]["exact_match"], summary["gradient"]["total"]), (12, 18)
        )
        self.assertEqual(overall["total"], 268)


if __name__ == "__main__":
    unittest.main()
