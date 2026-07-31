#!/usr/bin/env python3
"""Validate input or result CSV files without printing submitted text."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", text or "").strip().lower()


def main() -> int:
    parser = argparse.ArgumentParser(description="检查评测 CSV 的结构与重复情况")
    parser.add_argument("csv_file", type=Path)
    parser.add_argument("--text-column", default="answer")
    parser.add_argument("--id-column", default="id")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--result-file", action="store_true")
    args = parser.parse_args()

    with args.csv_file.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        headers = reader.fieldnames or []
    required = (
        {args.id_column, "true_label", "api_label"}
        if args.result_file
        else {args.id_column, args.text_column, args.label_column}
    )
    missing_columns = sorted(required - set(headers))
    empty_counts = {
        column: sum(not (row.get(column) or "").strip() for row in rows)
        for column in required & set(headers)
    }
    id_counts = Counter((row.get(args.id_column) or "").strip() for row in rows)
    duplicate_ids = sum(count - 1 for key, count in id_counts.items() if key and count > 1)

    duplicate_text_rows = 0
    unique_texts = None
    if not args.result_file and args.text_column in headers:
        hashes = Counter()
        for row in rows:
            normalized = normalize_text(row.get(args.text_column, ""))
            if normalized:
                digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
                hashes[digest] += 1
        unique_texts = len(hashes)
        duplicate_text_rows = sum(count - 1 for count in hashes.values() if count > 1)

    report = {
        "file": args.csv_file.name,
        "rows": len(rows),
        "headers": headers,
        "missing_columns": missing_columns,
        "empty_required_values": empty_counts,
        "unique_nonempty_ids": sum(bool(key) for key in id_counts),
        "duplicate_id_rows": duplicate_ids,
        "unique_nonempty_texts": unique_texts,
        "duplicate_text_rows": duplicate_text_rows,
        "valid": not missing_columns and not any(empty_counts.values()),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
