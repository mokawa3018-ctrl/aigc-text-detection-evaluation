#!/usr/bin/env python3
"""Compare two detector runs by anonymized sample ID."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from evaluate import normalize_label, safe_divide


def load_rows(path: Path, id_column: str) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    indexed: dict[str, dict[str, str]] = {}
    for line, row in enumerate(rows, start=2):
        sample_id = (row.get(id_column) or "").strip()
        if not sample_id:
            raise ValueError(f"{path.name} 第 {line} 行缺少 ID")
        if sample_id in indexed:
            raise ValueError(f"{path.name} 存在重复 ID：{sample_id}")
        indexed[sample_id] = row
    return indexed


def predicted(row: dict[str, str]) -> int:
    return normalize_label(row.get("api_label", row.get("predicted_label", "")))


def main() -> int:
    parser = argparse.ArgumentParser(description="按 ID 比较两个检测运行")
    parser.add_argument("run_a", type=Path)
    parser.add_argument("run_b", type=Path)
    parser.add_argument("--id-column", default="id")
    args = parser.parse_args()
    a = load_rows(args.run_a, args.id_column)
    b = load_rows(args.run_b, args.id_column)
    common = sorted(set(a) & set(b))
    a_only = sorted(set(a) - set(b))
    b_only = sorted(set(b) - set(a))

    agree = a_ai_b_human = a_human_b_ai = 0
    for sample_id in common:
        pa, pb = predicted(a[sample_id]), predicted(b[sample_id])
        agree += pa == pb
        a_ai_b_human += pa == 1 and pb == 0
        a_human_b_ai += pa == 0 and pb == 1
    report = {
        "run_a_rows": len(a),
        "run_b_rows": len(b),
        "common_ids": len(common),
        "run_a_only_ids": len(a_only),
        "run_b_only_ids": len(b_only),
        "prediction_agreement": safe_divide(agree, len(common)),
        "a_ai_b_human": a_ai_b_human,
        "a_human_b_ai": a_human_b_ai,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
