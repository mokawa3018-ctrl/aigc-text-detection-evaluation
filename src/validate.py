#!/usr/bin/env python3
"""Validate input or result CSV files without printing submitted text."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

CsvRow = dict[str, str | None]


def normalize_text(text: str | None) -> str:
    """Normalize text for privacy-preserving duplicate detection."""
    return re.sub(r"\s+", "", text or "").strip().lower()


def read_csv(path: Path) -> tuple[list[str], list[CsvRow]]:
    """Read a UTF-8 CSV file and return its headers and rows."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = [
            header
            for header in (reader.fieldnames or [])
            if header is not None
        ]
        rows: list[CsvRow] = list(reader)

    return headers, rows


def validate_rows(
    headers: list[str],
    rows: list[CsvRow],
    text_column: str,
    id_column: str,
    label_column: str,
    result_file: bool,
) -> dict[str, object]:
    """Validate required fields, empty values and duplicate identifiers.

    Args:
        headers: CSV header names.
        rows: CSV rows.
        text_column: Text field used for duplicate-content checks.
        id_column: Unique sample identifier field.
        label_column: Ground-truth field for non-result datasets.
        result_file: Whether the CSV is a detector result file.

    Returns:
        A JSON-serializable validation report.
    """
    required = (
        {id_column, "true_label", "api_label"}
        if result_file
        else {id_column, text_column, label_column}
    )
    header_set = set(headers)
    missing_columns = sorted(required - header_set)

    empty_counts = {
        column: sum(
            not (row.get(column) or "").strip()
            for row in rows
        )
        for column in sorted(required & header_set)
    }

    id_counts = Counter(
        (row.get(id_column) or "").strip()
        for row in rows
    )
    duplicate_ids = sum(
        count - 1
        for sample_id, count in id_counts.items()
        if sample_id and count > 1
    )

    duplicate_text_rows = 0
    unique_texts: int | None = None

    if not result_file and text_column in header_set:
        text_hashes: Counter[str] = Counter()

        for row in rows:
            normalized = normalize_text(row.get(text_column))
            if normalized:
                digest = hashlib.sha256(
                    normalized.encode("utf-8")
                ).hexdigest()
                text_hashes[digest] += 1

        unique_texts = len(text_hashes)
        duplicate_text_rows = sum(
            count - 1
            for count in text_hashes.values()
            if count > 1
        )

    errors: list[str] = []
    if not headers:
        errors.append("CSV 为空或缺少表头")
    if not rows:
        errors.append("CSV 没有数据行")
    if missing_columns:
        errors.append("缺少必需字段：" + ", ".join(missing_columns))
    if any(empty_counts.values()):
        errors.append("一个或多个必需字段存在空值")

    valid = not errors

    return {
        "file": None,
        "rows": len(rows),
        "headers": headers,
        "missing_columns": missing_columns,
        "empty_required_values": empty_counts,
        "unique_nonempty_ids": sum(
            bool(sample_id)
            for sample_id in id_counts
        ),
        "duplicate_id_rows": duplicate_ids,
        "unique_nonempty_texts": unique_texts,
        "duplicate_text_rows": duplicate_text_rows,
        "valid": valid,
        "errors": errors,
    }


def format_error(error: Exception) -> str:
    """Convert a file-reading exception into a concise CLI message."""
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
    """Run the CSV validation command-line interface."""
    parser = argparse.ArgumentParser(
        description="检查评测 CSV 的结构与重复情况"
    )
    parser.add_argument("csv_file", type=Path)
    parser.add_argument("--text-column", default="answer")
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--result-file", action="store_true")
    args = parser.parse_args()

    try:
        headers, rows = read_csv(args.csv_file)
        report = validate_rows(
            headers,
            rows,
            args.text_column,
            args.id_column,
            args.label_column,
            args.result_file,
        )
        report["file"] = args.csv_file.name
    except (
        FileNotFoundError,
        IsADirectoryError,
        PermissionError,
        UnicodeError,
        csv.Error,
        OSError,
    ) as error:
        print(f"错误：{format_error(error)}", file=sys.stderr)
        return 2

    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
