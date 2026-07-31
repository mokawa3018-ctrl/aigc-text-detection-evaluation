#!/usr/bin/env python3
"""Evaluate anonymized AIGC detector results with AI as the positive class."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

AI_TRUE = {"1", "ai", "aigc", "generated", "machine", "ai生成", "ai生成文本"}
HUMAN_TRUE = {"0", "human", "manual", "人工", "人类", "人类文本", "人工文本"}


def normalize_label(value: object) -> int:
    normalized = str(value).strip().lower().replace(" ", "")
    if normalized in AI_TRUE:
        return 1
    if normalized in HUMAN_TRUE:
        return 0
    raise ValueError(f"无法识别的标签：{value!r}")


def safe_divide(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def calculate_metrics(pairs: Iterable[tuple[int, int]]) -> dict[str, object]:
    rows = list(pairs)
    tp = sum(actual == 1 and predicted == 1 for actual, predicted in rows)
    fn = sum(actual == 1 and predicted == 0 for actual, predicted in rows)
    fp = sum(actual == 0 and predicted == 1 for actual, predicted in rows)
    tn = sum(actual == 0 and predicted == 0 for actual, predicted in rows)
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    return {
        "samples": len(rows),
        "positive_samples": tp + fn,
        "negative_samples": fp + tn,
        "confusion_matrix": {"tp": tp, "fn": fn, "fp": fp, "tn": tn},
        "accuracy": safe_divide(tp + tn, len(rows)),
        "ai_detection_rate": recall,
        "human_false_positive_rate": safe_divide(fp, fp + tn),
        "precision": precision,
        "f1": f1,
    }


def evaluate_rows(rows: list[dict[str, str]]) -> dict[str, object]:
    if not rows:
        raise ValueError("结果文件没有数据行")
    headers = set(rows[0])
    true_col = "true_label" if "true_label" in headers else "label"
    pred_col = "api_label" if "api_label" in headers else "predicted_label"
    if true_col not in headers or pred_col not in headers:
        raise ValueError("需要 true_label/label 和 api_label/predicted_label 字段")

    prepared: list[tuple[int, int, str]] = []
    skipped: list[int] = []
    for line_number, row in enumerate(rows, start=2):
        try:
            actual = normalize_label(row.get(true_col, ""))
            predicted = normalize_label(row.get(pred_col, ""))
        except ValueError:
            skipped.append(line_number)
            continue
        prepared.append((actual, predicted, (row.get("source") or "未分类").strip()))
    if not prepared:
        raise ValueError("没有可计算的有效标签行")

    grouped: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for actual, predicted, source in prepared:
        grouped[source].append((actual, predicted))

    overall = calculate_metrics((a, p) for a, p, _ in prepared)
    task_type = (
        "positive_only" if overall["negative_samples"] == 0
        else "negative_only" if overall["positive_samples"] == 0
        else "mixed"
    )
    return {
        "task_type": task_type,
        "valid_rows": len(prepared),
        "skipped_rows": skipped,
        "overall": overall,
        "by_source": {key: calculate_metrics(value) for key, value in sorted(grouped.items())},
        "interpretation": (
            "纯 AI 集仅解释 ai_detection_rate；FPR、Precision 和 F1 不适用。"
            if task_type == "positive_only"
            else "纯人工集仅解释 human_false_positive_rate。"
            if task_type == "negative_only"
            else "正负混合集可解释全部核心二分类指标。"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="计算 AIGC 文本检测结果指标")
    parser.add_argument("result_csv", type=Path)
    parser.add_argument("--output", type=Path, help="可选 JSON 输出路径")
    args = parser.parse_args()
    with args.result_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        report = evaluate_rows(list(csv.DictReader(handle)))
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    print(payload)
    if args.output:
        args.output.write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
