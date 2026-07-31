from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class IntegrationTests(unittest.TestCase):
    def run_script(
        self,
        script: str,
        *arguments: str,
    ) -> dict[str, object]:
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"

        completed = subprocess.run(
            [
                sys.executable,
                str(ROOT / "src" / script),
                *arguments,
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            env=env,
            check=False,
        )

        self.assertEqual(
            completed.returncode,
            0,
            msg=(
                f"{script} failed\n"
                f"stdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            ),
        )

        try:
            return json.loads(completed.stdout)
        except json.JSONDecodeError as error:
            self.fail(
                f"{script} did not return valid JSON: {error}"
            )

    def test_validate_evaluate_compare_workflow(self) -> None:
        validation = self.run_script(
            "validate.py",
            "data/demo_mixed.csv",
            "--result-file",
        )
        self.assertTrue(validation["valid"])
        self.assertEqual(validation["rows"], 6)
        self.assertEqual(validation["missing_columns"], [])

        evaluation = self.run_script(
            "evaluate.py",
            "data/demo_mixed.csv",
            "--group-by-length",
            "--length-thresholds",
            "200",
            "400",
            "--confidence-buckets",
        )
        self.assertEqual(evaluation["task_type"], "mixed")
        self.assertEqual(evaluation["valid_rows"], 6)
        self.assertEqual(
            evaluation["overall"]["confusion_matrix"],
            {"tp": 2, "fn": 1, "fp": 1, "tn": 2},
        )
        self.assertIn("by_source", evaluation)
        self.assertIn("by_length", evaluation)
        self.assertIn("by_confidence", evaluation)
        self.assertEqual(
            evaluation["by_length"]["column"],
            "sent_length",
        )

        comparison = self.run_script(
            "compare.py",
            "data/demo_mixed.csv",
            "data/demo_run_b.csv",
        )
        self.assertEqual(comparison["common_ids"], 6)
        self.assertEqual(
            comparison["prediction_agreement"],
            5 / 6,
        )

    def test_evaluation_output_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"

            env = os.environ.copy()
            env["PYTHONUTF8"] = "1"

            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "src" / "evaluate.py"),
                    "data/demo_mixed.csv",
                    "--output",
                    str(output),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                encoding="utf-8",
                env=env,
                check=False,
            )

            self.assertEqual(
                completed.returncode,
                0,
                msg=completed.stderr,
            )
            self.assertTrue(output.exists())

            report = json.loads(
                output.read_text(encoding="utf-8")
            )
            self.assertIn("overall", report)
            self.assertIn("by_source", report)
            self.assertEqual(report["valid_rows"], 6)


if __name__ == "__main__":
    unittest.main()
