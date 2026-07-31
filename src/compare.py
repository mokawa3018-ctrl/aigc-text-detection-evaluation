#!/usr/bin/env python3
"""Compare two detector runs by anonymized sample ID."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from label_utils import normalize_label

CsvRow = dict[str, str | None]
IndexedRows = dict[str, CsvRow]


def select_prediction_column(headers: set[str]) -> str:
    """Select the supported prediction column from CSV headers."""
    if "api_label" in headers:
        return "api_label"
    if "predicted_label" in headers:
        return "predicted_label"

    raise ValueError("需要 api_label 或 predicted_label 预测字段")


def load_rows(path: Path, id_column: str) -> IndexedRows:
    """Load and index one detector result CSV by sample ID."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = {
            header
            for header in (reader.fieldnames or [])
            if header is not None
        }

        if not headers:
            raise ValueError(f"{path.name} 是空文件或缺少 CSV 表头")
        if id_column not in headers:
            raise ValueError(f"{path.name} 缺少 ID 字段：{id_column}")

        select_prediction_column(headers)
        rows: list[CsvRow] = list(reader)

    if not rows:
        raise ValueError(f"{path.name} 没有数据行")

    indexed: IndexedRows = {}

    for line_number, row in enumerate(rows, start=2):
        sample_id = (row.get(id_column) or "").strip()

        if not sample_id:
            raise ValueError(
                f"{path.name} 第 {line_number} 行缺少 ID"
            )
        if sample_id in indexed:
            raise ValueError(f"{path.name} 存在重复 ID：{sample_id}")

        indexed[sample_id] = row

    return indexed


def predicted(row: CsvRow) -> int:
    """Read and normalize a prediction from one CSV row."""
    prediction_column = select_prediction_column(
        {
            header
            for header in row
            if header is not None
        }
    )
    return normalize_label(row.get(prediction_column, ""))


def compare_runs(
    run_a: IndexedRows,
    run_b: IndexedRows,
) -> dict[str, object]:
    """Compare predictions from two runs over their common sample IDs."""
    common_ids = sorted(set(run_a) & set(run_b))
    run_a_only_ids = sorted(set(run_a) - set(run_b))
    run_b_only_ids = sorted(set(run_b) - set(run_a))

    agree = 0
    a_ai_b_human = 0
    a_human_b_ai = 0

    for sample_id in common_ids:
        try:
            prediction_a = predicted(run_a[sample_id])
        except ValueError as error:
            raise ValueError(
                f"run A 样本 {sample_id}：{error}"
            ) from error

        try:
            prediction_b = predicted(run_b[sample_id])
        except ValueError as error:
            raise ValueError(
                f"run B 样本 {sample_id}：{error}"
            ) from error

        agree += prediction_a == prediction_b
        a_ai_b_human += (
            prediction_a == 1 and prediction_b == 0
        )
        a_human_b_ai += (
            prediction_a == 0 and prediction_b == 1
        )

    agreement = agree / len(common_ids) if common_ids else None

    return {
        "run_a_rows": len(run_a),
        "run_b_rows": len(run_b),
        "common_ids": len(common_ids),
        "run_a_only_ids": len(run_a_only_ids),
        "run_b_only_ids": len(run_b_only_ids),
        "prediction_agreement": agreement,
        "a_ai_b_human": a_ai_b_human,
        "a_human_b_ai": a_human_b_ai,
    }


def format_error(error: Exception) -> str:
    """Convert a file or comparison exception into a concise CLI message."""
    if isinstance(error, FileNotFoundError):
        return f"输入文件不存在：{error.filename}"
    if isinstance(error, IsADirectoryError):
        return f"输入路径是目录而不是文件：{error.filename}"
    if isinstance(error, PermissionError):
        return f"没有权限读取文件：{error.filename}"
    if isinstance(error, UnicodeError):
        return "文件编码错误：请输入 UTF-8 或 UTF-8 BOM 编码的 CSV"
    if isinstance(error, csv.Error):
        return f"CSV 格式错误：{error}"
    return str(error)


def main() -> int:
    """Run the detector-run comparison command-line interface."""
    parser = argparse.ArgumentParser(
        description="按 ID 比较两个检测运行"
    )
    parser.add_argument("run_a", type=Path)
    parser.add_argument("run_b", type=Path)
    parser.add_argument("--id-column", default="id")
    args = parser.parse_args()

    try:
        run_a = load_rows(args.run_a, args.id_column)
        run_b = load_rows(args.run_b, args.id_column)
        report = compare_runs(run_a, run_b)
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

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
