from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluate import (
    calculate_metrics,
    detect_optional_columns,
    evaluate_rows,
    read_result_csv,
)
from label_utils import normalize_label


class LabelNormalizationTests(unittest.TestCase):
    def test_supported_labels(self) -> None:
        self.assertEqual(normalize_label("AI生成文本"), 1)
        self.assertEqual(normalize_label(" 人类文本 "), 0)
        self.assertEqual(normalize_label("A I"), 1)
        self.assertEqual(normalize_label("0"), 0)

    def test_invalid_label(self) -> None:
        with self.assertRaisesRegex(ValueError, "无法识别"):
            normalize_label("unknown")


class MetricTests(unittest.TestCase):
    def test_mixed_metrics(self) -> None:
        result = calculate_metrics(
            [(1, 1), (1, 0), (0, 1), (0, 0)]
        )
        self.assertEqual(
            result["confusion_matrix"],
            {"tp": 1, "fn": 1, "fp": 1, "tn": 1},
        )
        self.assertEqual(result["accuracy"], 0.5)
        self.assertEqual(result["ai_detection_rate"], 0.5)
        self.assertEqual(
            result["human_false_positive_rate"],
            0.5,
        )

    def test_all_positive(self) -> None:
        result = calculate_metrics([(1, 1), (1, 0)])
        self.assertEqual(result["positive_samples"], 2)
        self.assertEqual(result["negative_samples"], 0)
        self.assertEqual(result["ai_detection_rate"], 0.5)
        self.assertIsNone(
            result["human_false_positive_rate"]
        )

    def test_all_negative(self) -> None:
        result = calculate_metrics([(0, 0), (0, 1)])
        self.assertEqual(result["positive_samples"], 0)
        self.assertEqual(result["negative_samples"], 2)
        self.assertIsNone(result["ai_detection_rate"])
        self.assertEqual(
            result["human_false_positive_rate"],
            0.5,
        )

    def test_empty_metrics(self) -> None:
        result = calculate_metrics([])
        self.assertEqual(result["samples"], 0)
        self.assertIsNone(result["accuracy"])
        self.assertIsNone(result["f1"])


class EvaluationTests(unittest.TestCase):
    def test_positive_only_interpretation(self) -> None:
        report = evaluate_rows(
            [
                {
                    "true_label": "1",
                    "api_label": "AI生成文本",
                    "source": "x",
                },
                {
                    "true_label": "1",
                    "api_label": "人类文本",
                    "source": "x",
                },
            ]
        )
        self.assertEqual(report["task_type"], "positive_only")
        self.assertEqual(
            report["overall"]["ai_detection_rate"],
            0.5,
        )

    def test_negative_only_interpretation(self) -> None:
        report = evaluate_rows(
            [
                {
                    "true_label": "0",
                    "api_label": "人类文本",
                },
                {
                    "true_label": "0",
                    "api_label": "AI生成文本",
                },
            ]
        )
        self.assertEqual(report["task_type"], "negative_only")
        self.assertEqual(
            report["overall"]["human_false_positive_rate"],
            0.5,
        )

    def test_missing_columns(self) -> None:
        with self.assertRaisesRegex(ValueError, "需要"):
            evaluate_rows([{"id": "x"}])

    def test_empty_rows(self) -> None:
        with self.assertRaisesRegex(ValueError, "没有数据行"):
            evaluate_rows([])

    def test_invalid_rows_are_recorded(self) -> None:
        report = evaluate_rows(
            [
                {
                    "true_label": "1",
                    "api_label": "1",
                },
                {
                    "true_label": "invalid",
                    "api_label": "0",
                },
            ]
        )
        self.assertEqual(report["valid_rows"], 1)
        self.assertEqual(report["skipped_rows"], [3])

    def test_all_invalid_rows_raise(self) -> None:
        with self.assertRaisesRegex(ValueError, "没有可计算"):
            evaluate_rows(
                [
                    {
                        "true_label": "invalid",
                        "api_label": "invalid",
                    }
                ]
            )

    def test_empty_csv_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.csv"
            path.write_text("", encoding="utf-8")

            with self.assertRaisesRegex(
                ValueError,
                "空文件|缺少 CSV 表头",
            ):
                read_result_csv(path)

    def test_header_only_csv_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "header.csv"
            path.write_text(
                "id,true_label,api_label\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ValueError,
                "没有数据行",
            ):
                read_result_csv(path)

    def test_optional_column_detection(self) -> None:
        detected = detect_optional_columns(
            {
                "source",
                "target_char_count",
                "generator",
                "confidence",
            }
        )
        self.assertEqual(
            detected,
            {
                "confidence": "confidence",
                "length": "target_char_count",
                "generator": "generator",
                "source": "source",
            },
        )

    def test_length_analysis(self) -> None:
        rows = [
            {
                "true_label": "1",
                "api_label": "1",
                "sent_length": "100",
            },
            {
                "true_label": "1",
                "api_label": "0",
                "sent_length": "300",
            },
            {
                "true_label": "0",
                "api_label": "0",
                "sent_length": "700",
            },
        ]

        report = evaluate_rows(
            rows,
            group_by_length=True,
            length_thresholds=(200, 500),
        )
        analysis = report["by_length"]

        self.assertEqual(analysis["column"], "sent_length")
        self.assertEqual(
            analysis["groups"]["short"]["samples"],
            1,
        )
        self.assertEqual(
            analysis["groups"]["medium"]["samples"],
            1,
        )
        self.assertEqual(
            analysis["groups"]["long"]["samples"],
            1,
        )

    def test_generator_analysis(self) -> None:
        rows = [
            {
                "true_label": "1",
                "api_label": "1",
                "generator": "model_a",
            },
            {
                "true_label": "1",
                "api_label": "0",
                "generator": "model_b",
            },
        ]

        report = evaluate_rows(
            rows,
            group_by_generator=True,
        )

        self.assertEqual(
            report["by_generator"]["groups"]["model_a"][
                "ai_detection_rate"
            ],
            1.0,
        )
        self.assertEqual(
            report["by_generator"]["groups"]["model_b"][
                "ai_detection_rate"
            ],
            0.0,
        )

    def test_missing_generator_adds_warning(self) -> None:
        report = evaluate_rows(
            [{"true_label": "1", "api_label": "1"}],
            group_by_generator=True,
        )
        self.assertNotIn("by_generator", report)
        self.assertTrue(report["analysis_warnings"])

    def test_confidence_buckets(self) -> None:
        rows = [
            {
                "true_label": "1",
                "api_label": "0",
                "confidence": "0.3",
            },
            {
                "true_label": "1",
                "api_label": "1",
                "confidence": "0.9",
            },
        ]

        report = evaluate_rows(
            rows,
            confidence_buckets=True,
            confidence_bin_width=0.5,
        )
        bins = report["by_confidence"]["bins"]

        self.assertEqual(bins[0]["metrics"]["samples"], 1)
        self.assertEqual(bins[0]["metrics"]["accuracy"], 0.0)
        self.assertEqual(bins[1]["metrics"]["samples"], 1)
        self.assertEqual(bins[1]["metrics"]["accuracy"], 1.0)

    def test_threshold_curve(self) -> None:
        rows = [
            {
                "true_label": "1",
                "api_label": "1",
                "confidence": "0.9",
            },
            {
                "true_label": "0",
                "api_label": "0",
                "confidence": "0.1",
            },
        ]

        report = evaluate_rows(
            rows,
            threshold_curve=True,
            threshold_step=0.5,
            score_positive_class="ai",
        )
        points = report["threshold_curve"]["points"]
        threshold_half = next(
            point
            for point in points
            if point["threshold"] == 0.5
        )
        self.assertEqual(threshold_half["accuracy"], 1.0)

    def test_invalid_length_thresholds(self) -> None:
        with self.assertRaisesRegex(ValueError, "SHORT_MAX"):
            evaluate_rows(
                [
                    {
                        "true_label": "1",
                        "api_label": "1",
                        "length": "100",
                    }
                ],
                group_by_length=True,
                length_thresholds=(500, 200),
            )


if __name__ == "__main__":
    unittest.main()
