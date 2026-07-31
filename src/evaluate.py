#!/usr/bin/env python3
"""Evaluate anonymized AIGC detector results with optional grouped analyses."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

# Re-exported for backward compatibility.
from label_utils import normalize_label

CsvRow = dict[str, str | None]
MetricPair = tuple[int, int]

OPTIONAL_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "confidence": ("confidence",),
    "length": (
        "length",
        "target_char_count",
        "sent_length",
        "original_length",
    ),
    "generator": ("generator",),
    "source": ("source",),
}


@dataclass(frozen=True)
class PreparedSample:
    """One row with normalized labels and its original optional values."""

    line_number: int
    actual: int
    predicted: int
    source: str
    row: CsvRow


def safe_divide(numerator: int, denominator: int) -> float | None:
    """Divide two integers, returning ``None`` when undefined."""
    return numerator / denominator if denominator else None


def calculate_metrics(pairs: Iterable[MetricPair]) -> dict[str, object]:
    """Calculate binary metrics with AI as the positive class."""
    rows = list(pairs)

    tp = sum(actual == 1 and predicted == 1 for actual, predicted in rows)
    fn = sum(actual == 1 and predicted == 0 for actual, predicted in rows)
    fp = sum(actual == 0 and predicted == 1 for actual, predicted in rows)
    tn = sum(actual == 0 and predicted == 0 for actual, predicted in rows)

    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None
        and recall is not None
        and precision + recall
        else None
    )

    return {
        "samples": len(rows),
        "positive_samples": tp + fn,
        "negative_samples": fp + tn,
        "confusion_matrix": {
            "tp": tp,
            "fn": fn,
            "fp": fp,
            "tn": tn,
        },
        "accuracy": safe_divide(tp + tn, len(rows)),
        "ai_detection_rate": recall,
        "human_false_positive_rate": safe_divide(fp, fp + tn),
        "precision": precision,
        "f1": f1,
    }


def select_label_columns(headers: set[str]) -> tuple[str, str]:
    """Select compatible truth and prediction columns."""
    true_column = (
        "true_label"
        if "true_label" in headers
        else "label"
        if "label" in headers
        else None
    )
    predicted_column = (
        "api_label"
        if "api_label" in headers
        else "predicted_label"
        if "predicted_label" in headers
        else None
    )

    if true_column is None or predicted_column is None:
        raise ValueError(
            "需要 true_label/label 和 api_label/predicted_label 字段"
        )

    return true_column, predicted_column


def detect_optional_columns(headers: set[str]) -> dict[str, str]:
    """Map optional semantic fields to available CSV columns."""
    detected: dict[str, str] = {}

    for semantic_name, aliases in OPTIONAL_COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in headers:
                detected[semantic_name] = alias
                break

    return detected


def read_result_csv(path: Path) -> list[CsvRow]:
    """Read a UTF-8 result CSV with headers and data rows."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames is None:
            raise ValueError(f"{path.name} 是空文件或缺少 CSV 表头")

        rows: list[CsvRow] = list(reader)

    if not rows:
        raise ValueError(f"{path.name} 没有数据行")

    return rows


def prepare_samples(
    rows: list[CsvRow],
    true_column: str,
    predicted_column: str,
) -> tuple[list[PreparedSample], list[int]]:
    """Normalize labels and retain optional values for valid rows."""
    prepared: list[PreparedSample] = []
    skipped_rows: list[int] = []

    for line_number, row in enumerate(rows, start=2):
        try:
            actual = normalize_label(row.get(true_column, ""))
            predicted = normalize_label(row.get(predicted_column, ""))
        except ValueError:
            skipped_rows.append(line_number)
            continue

        source = (row.get("source") or "").strip() or "未分类"
        prepared.append(
            PreparedSample(
                line_number=line_number,
                actual=actual,
                predicted=predicted,
                source=source,
                row=row,
            )
        )

    if not prepared:
        raise ValueError("没有可计算的有效标签行")

    return prepared, skipped_rows


def metrics_for_samples(
    samples: Iterable[PreparedSample],
) -> dict[str, object]:
    """Calculate metrics for prepared samples."""
    return calculate_metrics(
        (sample.actual, sample.predicted) for sample in samples
    )


def parse_nonnegative_number(
    value: str | None,
    field_name: str,
) -> float:
    """Parse a finite, non-negative numeric field."""
    raw = (value or "").strip()

    if not raw:
        raise ValueError(f"{field_name} 为空")

    try:
        number = float(raw)
    except ValueError as error:
        raise ValueError(
            f"{field_name} 不是有效数字：{value!r}"
        ) from error

    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{field_name} 必须是有限的非负数")

    return number


def parse_unit_score(value: str | None, field_name: str) -> float:
    """Parse a finite score in the inclusive range 0 to 1."""
    score = parse_nonnegative_number(value, field_name)

    if score > 1:
        raise ValueError(f"{field_name} 必须位于 0 到 1 之间")

    return score


def analyze_by_length(
    samples: list[PreparedSample],
    column: str,
    short_max: int,
    medium_max: int,
) -> dict[str, object]:
    """Group metrics into short, medium and long text buckets."""
    if short_max < 0 or medium_max < 0:
        raise ValueError("长度阈值不能为负数")
    if short_max >= medium_max:
        raise ValueError("长度阈值必须满足 SHORT_MAX < MEDIUM_MAX")

    grouped: dict[str, list[PreparedSample]] = {
        "short": [],
        "medium": [],
        "long": [],
    }
    invalid_rows: list[int] = []

    for sample in samples:
        try:
            length = parse_nonnegative_number(
                sample.row.get(column),
                column,
            )
        except ValueError:
            invalid_rows.append(sample.line_number)
            continue

        if length <= short_max:
            bucket = "short"
        elif length <= medium_max:
            bucket = "medium"
        else:
            bucket = "long"

        grouped[bucket].append(sample)

    return {
        "column": column,
        "thresholds": {
            "short_max": short_max,
            "medium_max": medium_max,
        },
        "invalid_rows": invalid_rows,
        "groups": {
            name: metrics_for_samples(group)
            for name, group in grouped.items()
        },
    }


def analyze_by_generator(
    samples: list[PreparedSample],
    column: str,
) -> dict[str, object]:
    """Calculate metrics for each generator value."""
    grouped: dict[str, list[PreparedSample]] = defaultdict(list)

    for sample in samples:
        generator = (sample.row.get(column) or "").strip() or "未分类"
        grouped[generator].append(sample)

    return {
        "column": column,
        "groups": {
            generator: metrics_for_samples(group)
            for generator, group in sorted(grouped.items())
        },
    }


def build_unit_boundaries(width: float) -> list[float]:
    """Build stable boundaries from 0 to 1."""
    if not 0 < width <= 1:
        raise ValueError("分桶宽度必须满足 0 < WIDTH <= 1")

    boundaries = [0.0]
    current = width

    while current < 1:
        boundaries.append(round(current, 10))
        current += width

    if boundaries[-1] != 1.0:
        boundaries.append(1.0)

    return boundaries


def analyze_confidence_buckets(
    samples: list[PreparedSample],
    column: str,
    width: float,
) -> dict[str, object]:
    """Calculate accuracy and classification metrics by score bucket."""
    boundaries = build_unit_boundaries(width)
    buckets: list[list[PreparedSample]] = [
        [] for _ in range(len(boundaries) - 1)
    ]
    invalid_rows: list[int] = []

    for sample in samples:
        try:
            score = parse_unit_score(sample.row.get(column), column)
        except ValueError:
            invalid_rows.append(sample.line_number)
            continue

        for index in range(len(boundaries) - 1):
            lower = boundaries[index]
            upper = boundaries[index + 1]
            is_last = index == len(boundaries) - 2

            if lower <= score < upper or (is_last and score == upper):
                buckets[index].append(sample)
                break

    bucket_reports: list[dict[str, object]] = []

    for index, bucket_samples in enumerate(buckets):
        lower = boundaries[index]
        upper = boundaries[index + 1]

        bucket_reports.append(
            {
                "lower": lower,
                "upper": upper,
                "upper_inclusive": index == len(buckets) - 1,
                "metrics": metrics_for_samples(bucket_samples),
            }
        )

    return {
        "column": column,
        "bin_width": width,
        "valid_rows": len(samples) - len(invalid_rows),
        "invalid_rows": invalid_rows,
        "bins": bucket_reports,
    }


def build_thresholds(step: float) -> list[float]:
    """Build threshold values from 0 to 1."""
    if not 0 < step <= 1:
        raise ValueError("阈值步长必须满足 0 < STEP <= 1")

    thresholds = [0.0]
    current = step

    while current < 1:
        thresholds.append(round(current, 10))
        current += step

    if thresholds[-1] != 1.0:
        thresholds.append(1.0)

    return thresholds


def analyze_threshold_curve(
    samples: list[PreparedSample],
    column: str,
    step: float,
    score_positive_class: str,
) -> dict[str, object]:
    """Calculate classification metrics over score thresholds.

    The input score must represent a probability or comparable score for one
    fixed class. Confidence in the already-predicted label is not sufficient.
    """
    scored_samples: list[tuple[int, float]] = []
    invalid_rows: list[int] = []

    for sample in samples:
        try:
            score = parse_unit_score(sample.row.get(column), column)
        except ValueError:
            invalid_rows.append(sample.line_number)
            continue

        ai_score = score if score_positive_class == "ai" else 1 - score
        scored_samples.append((sample.actual, ai_score))

    points: list[dict[str, object]] = []

    for threshold in build_thresholds(step):
        metrics = calculate_metrics(
            (
                actual,
                1 if ai_score >= threshold else 0,
            )
            for actual, ai_score in scored_samples
        )
        points.append(
            {
                "threshold": threshold,
                **metrics,
            }
        )

    return {
        "column": column,
        "score_positive_class": score_positive_class,
        "step": step,
        "valid_rows": len(scored_samples),
        "invalid_rows": invalid_rows,
        "points": points,
    }


def evaluate_rows(
    rows: list[CsvRow],
    *,
    group_by_length: bool = False,
    length_thresholds: tuple[int, int] = (200, 500),
    group_by_generator: bool = False,
    confidence_buckets: bool = False,
    confidence_bin_width: float = 0.2,
    threshold_curve: bool = False,
    threshold_step: float = 0.05,
    score_column: str | None = None,
    score_positive_class: str = "ai",
) -> dict[str, object]:
    """Evaluate rows and optionally append grouped analyses."""
    if not rows:
        raise ValueError("结果文件没有数据行")

    headers = {header for header in rows[0] if header is not None}
    true_column, predicted_column = select_label_columns(headers)
    prepared, skipped_rows = prepare_samples(
        rows,
        true_column,
        predicted_column,
    )

    grouped_by_source: dict[str, list[PreparedSample]] = defaultdict(list)
    for sample in prepared:
        grouped_by_source[sample.source].append(sample)

    overall = metrics_for_samples(prepared)

    if overall["negative_samples"] == 0:
        task_type = "positive_only"
    elif overall["positive_samples"] == 0:
        task_type = "negative_only"
    else:
        task_type = "mixed"

    if task_type == "positive_only":
        interpretation = (
            "纯 AI 集仅解释 ai_detection_rate；"
            "FPR、Precision 和 F1 不适用。"
        )
    elif task_type == "negative_only":
        interpretation = "纯人工集仅解释 human_false_positive_rate。"
    else:
        interpretation = "正负混合集可解释全部核心二分类指标。"

    report: dict[str, object] = {
        "task_type": task_type,
        "valid_rows": len(prepared),
        "skipped_rows": skipped_rows,
        "overall": overall,
        "by_source": {
            source: metrics_for_samples(group)
            for source, group in sorted(grouped_by_source.items())
        },
        "interpretation": interpretation,
    }

    enhanced_analysis_requested = any(
        (
            group_by_length,
            group_by_generator,
            confidence_buckets,
            threshold_curve,
        )
    )

    if not enhanced_analysis_requested:
        return report

    optional_columns = detect_optional_columns(headers)
    report["optional_columns"] = optional_columns
    warnings: list[str] = []

    if group_by_length:
        length_column = optional_columns.get("length")

        if length_column is None:
            warnings.append("未找到长度字段，已跳过长度分组分析")
        else:
            report["by_length"] = analyze_by_length(
                prepared,
                length_column,
                length_thresholds[0],
                length_thresholds[1],
            )

    if group_by_generator:
        generator_column = optional_columns.get("generator")

        if generator_column is None:
            warnings.append(
                "未找到 generator 字段，已跳过生成器分组分析"
            )
        else:
            report["by_generator"] = analyze_by_generator(
                prepared,
                generator_column,
            )

    resolved_score_column = (
        score_column
        if score_column is not None
        else optional_columns.get("confidence")
    )

    if score_column is not None and score_column not in headers:
        resolved_score_column = None
        warnings.append(f"未找到指定分数字段：{score_column}")

    if confidence_buckets:
        if resolved_score_column is None:
            warnings.append("未找到置信度字段，已跳过置信度分桶分析")
        else:
            report["by_confidence"] = analyze_confidence_buckets(
                prepared,
                resolved_score_column,
                confidence_bin_width,
            )

    if threshold_curve:
        if resolved_score_column is None:
            warnings.append("未找到连续分数字段，已跳过阈值曲线分析")
        else:
            report["threshold_curve"] = analyze_threshold_curve(
                prepared,
                resolved_score_column,
                threshold_step,
                score_positive_class,
            )

    if warnings:
        report["analysis_warnings"] = warnings

    return report


def format_error(error: Exception) -> str:
    """Convert an exception into a concise CLI message."""
    if isinstance(error, FileNotFoundError):
        return f"输入文件不存在：{error.filename}"
    if isinstance(error, IsADirectoryError):
        return f"输入路径是目录而不是文件：{error.filename}"
    if isinstance(error, PermissionError):
        return f"没有权限读取或写入文件：{error.filename}"
    if isinstance(error, UnicodeError):
        return "文件编码错误：请输入 UTF-8 或 UTF-8 BOM 编码的 CSV"
    if isinstance(error, csv.Error):
        return f"CSV 格式错误：{error}"
    return str(error)


def build_parser() -> argparse.ArgumentParser:
    """Build the backward-compatible command-line parser."""
    parser = argparse.ArgumentParser(
        description="计算 AIGC 文本检测结果指标"
    )
    parser.add_argument("result_csv", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        help="可选 JSON 输出路径",
    )
    parser.add_argument(
        "--group-by-length",
        action="store_true",
        help="启用短、中、长文本分组指标",
    )
    parser.add_argument(
        "--length-thresholds",
        nargs=2,
        type=int,
        metavar=("SHORT_MAX", "MEDIUM_MAX"),
        default=(200, 500),
        help="长度分组上限，默认 200 500",
    )
    parser.add_argument(
        "--group-by-generator",
        action="store_true",
        help="启用 generator 分组指标",
    )
    parser.add_argument(
        "--confidence-buckets",
        action="store_true",
        help="启用置信度分桶分析",
    )
    parser.add_argument(
        "--confidence-bin-width",
        type=float,
        default=0.2,
        metavar="WIDTH",
        help="置信度分桶宽度，默认 0.2",
    )
    parser.add_argument(
        "--threshold-curve",
        action="store_true",
        help="输出连续分数的阈值-指标曲线",
    )
    parser.add_argument(
        "--threshold-step",
        type=float,
        default=0.05,
        metavar="STEP",
        help="阈值曲线步长，默认 0.05",
    )
    parser.add_argument(
        "--score-column",
        help="连续分数字段，默认自动使用 confidence",
    )
    parser.add_argument(
        "--score-positive-class",
        choices=("ai", "human"),
        default="ai",
        help="分数代表的正向类别，默认 ai",
    )
    return parser


def main() -> int:
    """Run the evaluation command-line interface."""
    parser = build_parser()
    args = parser.parse_args()

    try:
        rows = read_result_csv(args.result_csv)
        report = evaluate_rows(
            rows,
            group_by_length=args.group_by_length,
            length_thresholds=tuple(args.length_thresholds),
            group_by_generator=args.group_by_generator,
            confidence_buckets=args.confidence_buckets,
            confidence_bin_width=args.confidence_bin_width,
            threshold_curve=args.threshold_curve,
            threshold_step=args.threshold_step,
            score_column=args.score_column,
            score_positive_class=args.score_positive_class,
        )
        payload = json.dumps(report, ensure_ascii=False, indent=2)

        print(payload)

        if args.output:
            args.output.write_text(payload + "\n", encoding="utf-8")
    except (
        FileNotFoundError,
        IsADirectoryError,
        PermissionError,
        UnicodeError,
        csv.Error,
        OSError,
        ValueError,
    ) as error:
        print(f"错误：{format_error(error)}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
