#!/usr/bin/env python3
"""Print a readable summary of an evaluation JSON report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Mapping

JsonObject = dict[str, object]


def format_rate(value: object) -> str:
    """Format a JSON metric as a percentage or an unavailable marker."""
    if value is None:
        return "N/A"
    if isinstance(value, (int, float)):
        return f"{float(value):.2%}"
    return str(value)


def print_confusion_matrix(metrics: Mapping[str, object]) -> None:
    """Print the confusion matrix from one metrics object."""
    matrix = metrics.get("confusion_matrix")
    if not isinstance(matrix, dict):
        print("Confusion matrix: unavailable")
        return

    tp = matrix.get("tp", 0)
    fn = matrix.get("fn", 0)
    fp = matrix.get("fp", 0)
    tn = matrix.get("tn", 0)

    print("Confusion matrix")
    print("+-------------+--------------+-----------------+")
    print("| Actual      | Predicted AI | Predicted Human |")
    print("+-------------+--------------+-----------------+")
    print(f"| AI          | {str(tp):>12} | {str(fn):>15} |")
    print(f"| Human       | {str(fp):>12} | {str(tn):>15} |")
    print("+-------------+--------------+-----------------+")


def print_metrics_line(name: str, metrics: Mapping[str, object]) -> None:
    """Print one compact grouped-metrics line."""
    samples = metrics.get("samples", 0)
    accuracy = format_rate(metrics.get("accuracy"))
    detection_rate = format_rate(metrics.get("ai_detection_rate"))
    false_positive_rate = format_rate(
        metrics.get("human_false_positive_rate")
    )
    print(
        f"- {name}: samples={samples}, accuracy={accuracy}, "
        f"AI recall={detection_rate}, human FPR={false_positive_rate}"
    )


def print_group_mapping(title: str, groups: object) -> None:
    """Print a mapping of group names to metrics dictionaries."""
    if not isinstance(groups, dict) or not groups:
        return

    print(f"\n{title}")
    for name, metrics in groups.items():
        if isinstance(metrics, dict):
            print_metrics_line(str(name), metrics)


def print_confidence_bins(analysis: object) -> None:
    """Print confidence bucket summaries when present."""
    if not isinstance(analysis, dict):
        return

    bins = analysis.get("bins")
    if not isinstance(bins, list) or not bins:
        return

    print("\nConfidence buckets")
    for bucket in bins:
        if not isinstance(bucket, dict):
            continue

        lower = bucket.get("lower", "?")
        upper = bucket.get("upper", "?")
        closing = "]" if bucket.get("upper_inclusive") else ")"
        metrics = bucket.get("metrics")
        if isinstance(metrics, dict):
            print_metrics_line(
                f"[{lower}, {upper}{closing}",
                metrics,
            )


def print_report(report: Mapping[str, object]) -> None:
    """Print overall and grouped summaries from an evaluation report."""
    print(f"Task type: {report.get('task_type', 'unknown')}")
    print(f"Valid rows: {report.get('valid_rows', 'unknown')}")

    overall = report.get("overall")
    if isinstance(overall, dict):
        print()
        print_confusion_matrix(overall)
        print("\nOverall metrics")
        print_metrics_line("overall", overall)

    print_group_mapping("By source", report.get("by_source"))

    by_length = report.get("by_length")
    if isinstance(by_length, dict):
        print_group_mapping("By length", by_length.get("groups"))

    by_generator = report.get("by_generator")
    if isinstance(by_generator, dict):
        print_group_mapping("By generator", by_generator.get("groups"))

    print_confidence_bins(report.get("by_confidence"))

    warnings = report.get("analysis_warnings")
    if isinstance(warnings, list) and warnings:
        print("\nAnalysis warnings")
        for warning in warnings:
            print(f"- {warning}")


def load_report(path: Path) -> JsonObject:
    """Load and validate a JSON report object from disk."""
    with path.open("r", encoding="utf-8-sig") as handle:
        report = json.load(handle)

    if not isinstance(report, dict):
        raise ValueError("报告 JSON 顶层必须是对象")

    return report


def main() -> int:
    """Run the report visualization command-line example."""
    parser = argparse.ArgumentParser(
        description="打印 evaluate.py JSON 报告的表格摘要"
    )
    parser.add_argument("report_json", type=Path)
    args = parser.parse_args()

    try:
        report = load_report(args.report_json)
        print_report(report)
    except FileNotFoundError:
        print(f"错误：报告文件不存在：{args.report_json}", file=sys.stderr)
        return 2
    except PermissionError:
        print(f"错误：没有权限读取：{args.report_json}", file=sys.stderr)
        return 2
    except UnicodeError:
        print("错误：报告文件必须使用 UTF-8 编码", file=sys.stderr)
        return 2
    except json.JSONDecodeError as error:
        print(f"错误：无效 JSON：{error}", file=sys.stderr)
        return 2
    except (OSError, ValueError) as error:
        print(f"错误：{error}", file=sys.stderr)
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
