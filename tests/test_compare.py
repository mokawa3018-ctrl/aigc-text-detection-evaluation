from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from compare import compare_runs, load_rows, predicted


class CompareTests(unittest.TestCase):
    def test_compare_runs(self) -> None:
        run_a = {
            "a": {"api_label": "1"},
            "b": {"api_label": "0"},
        }
        run_b = {
            "a": {"api_label": "1"},
            "b": {"api_label": "1"},
        }

        report = compare_runs(run_a, run_b)

        self.assertEqual(report["common_ids"], 2)
        self.assertEqual(report["prediction_agreement"], 0.5)
        self.assertEqual(report["a_human_b_ai"], 1)

    def test_prediction_alias(self) -> None:
        self.assertEqual(
            predicted({"predicted_label": "AI生成文本"}),
            1,
        )

    def test_no_common_ids(self) -> None:
        report = compare_runs(
            {"a": {"api_label": "1"}},
            {"b": {"api_label": "0"}},
        )
        self.assertEqual(report["common_ids"], 0)
        self.assertIsNone(report["prediction_agreement"])

    def test_invalid_label_includes_run_context(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "run A 样本 a",
        ):
            compare_runs(
                {"a": {"api_label": "invalid"}},
                {"a": {"api_label": "1"}},
            )

    def test_load_rows_rejects_duplicate_id(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "duplicate.csv"
            path.write_text(
                "id,api_label\n"
                "a,1\n"
                "a,0\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "重复 ID"):
                load_rows(path, "id")

    def test_load_rows_rejects_missing_id_column(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing-id.csv"
            path.write_text(
                "sample,api_label\n"
                "a,1\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "缺少 ID"):
                load_rows(path, "id")

    def test_load_rows_rejects_missing_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing-prediction.csv"
            path.write_text(
                "id,true_label\n"
                "a,1\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                "api_label|predicted_label",
            ):
                load_rows(path, "id")

    def test_load_rows_rejects_header_only_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "header-only.csv"
            path.write_text(
                "id,api_label\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                "没有数据行",
            ):
                load_rows(path, "id")


if __name__ == "__main__":
    unittest.main()
