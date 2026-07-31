from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluate import calculate_metrics, evaluate_rows, normalize_label


class EvaluationTests(unittest.TestCase):
    def test_label_mapping(self) -> None:
        self.assertEqual(normalize_label("AI生成文本"), 1)
        self.assertEqual(normalize_label("人类文本"), 0)
        self.assertEqual(normalize_label("1"), 1)
        self.assertEqual(normalize_label("0"), 0)

    def test_mixed_metrics(self) -> None:
        result = calculate_metrics([(1, 1), (1, 0), (0, 1), (0, 0)])
        self.assertEqual(result["confusion_matrix"], {"tp": 1, "fn": 1, "fp": 1, "tn": 1})
        self.assertEqual(result["accuracy"], 0.5)
        self.assertEqual(result["ai_detection_rate"], 0.5)
        self.assertEqual(result["human_false_positive_rate"], 0.5)

    def test_positive_only_interpretation(self) -> None:
        report = evaluate_rows([
            {"true_label": "1", "api_label": "AI生成文本", "source": "x"},
            {"true_label": "1", "api_label": "人类文本", "source": "x"},
        ])
        self.assertEqual(report["task_type"], "positive_only")
        self.assertIsNone(report["overall"]["human_false_positive_rate"])
        self.assertEqual(report["overall"]["ai_detection_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
