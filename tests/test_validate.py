from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from validate import read_csv, validate_rows


class ValidateTests(unittest.TestCase):
    def validate(
        self,
        headers: list[str],
        rows: list[dict[str, str | None]],
        *,
        result_file: bool = True,
    ) -> dict[str, object]:
        return validate_rows(
            headers,
            rows,
            "answer",
            "id",
            "label",
            result_file,
        )

    def test_valid_result_file(self) -> None:
        report = self.validate(
            ["id", "true_label", "api_label"],
            [
                {
                    "id": "a",
                    "true_label": "1",
                    "api_label": "1",
                }
            ],
        )
        self.assertTrue(report["valid"])
        self.assertEqual(report["errors"], [])

    def test_header_only_file_is_invalid(self) -> None:
        report = self.validate(
            ["id", "true_label", "api_label"],
            [],
        )
        self.assertFalse(report["valid"])
        self.assertIn("CSV 没有数据行", report["errors"])

    def test_missing_required_column(self) -> None:
        report = self.validate(
            ["id", "true_label"],
            [{"id": "a", "true_label": "1"}],
        )
        self.assertFalse(report["valid"])
        self.assertEqual(
            report["missing_columns"],
            ["api_label"],
        )

    def test_empty_required_value(self) -> None:
        report = self.validate(
            ["id", "true_label", "api_label"],
            [
                {
                    "id": "",
                    "true_label": "1",
                    "api_label": "1",
                }
            ],
        )
        self.assertFalse(report["valid"])
        self.assertEqual(
            report["empty_required_values"]["id"],
            1,
        )

    def test_duplicate_ids(self) -> None:
        report = self.validate(
            ["id", "true_label", "api_label"],
            [
                {
                    "id": "a",
                    "true_label": "1",
                    "api_label": "1",
                },
                {
                    "id": "a",
                    "true_label": "0",
                    "api_label": "0",
                },
            ],
        )
        self.assertEqual(report["duplicate_id_rows"], 1)

    def test_duplicate_normalized_text(self) -> None:
        report = self.validate(
            ["id", "answer", "label"],
            [
                {
                    "id": "a",
                    "answer": "A B",
                    "label": "1",
                },
                {
                    "id": "b",
                    "answer": "AB",
                    "label": "1",
                },
            ],
            result_file=False,
        )
        self.assertEqual(report["unique_nonempty_texts"], 1)
        self.assertEqual(report["duplicate_text_rows"], 1)

    def test_missing_file(self) -> None:
        with self.assertRaises(FileNotFoundError):
            read_csv(Path("definitely-not-present.csv"))

    def test_invalid_utf8(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.csv"
            path.write_bytes(b"\xff\xfe\xfa")

            with self.assertRaises(UnicodeError):
                read_csv(path)


if __name__ == "__main__":
    unittest.main()
