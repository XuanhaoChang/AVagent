#!/usr/bin/env python3
"""Attach one prediction CSV to existing per-sample review packages."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from av_eval.review_export import (
    attach_auralis_evidence_to_review_samples,
    attach_prediction_to_review_samples,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="把一个预测 CSV 按序号附加到现有逐样本复核目录。",
    )
    parser.add_argument("--prediction-csv", type=Path, required=True)
    parser.add_argument("--samples-root", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument(
        "--replace",
        action="store_true",
        help="原子替换已存在的同名预测 JSON。",
    )
    parser.add_argument(
        "--run-log",
        type=Path,
        default=None,
        help=(
            "可选 AVAgent JSONL 日志；提供后会按序号附加 asr.json 和 ocr.json，"
            "内容来自 auralis_evidence。"
        ),
    )
    args = parser.parse_args()
    count = attach_prediction_to_review_samples(
        prediction_csv=args.prediction_csv,
        samples_root=args.samples_root,
        label=args.label,
        replace=args.replace,
    )
    print(f"已附加 {count} 条 {args.label} 预测：{args.samples_root}")
    if args.run_log is not None:
        evidence_count = attach_auralis_evidence_to_review_samples(
            run_log=args.run_log,
            samples_root=args.samples_root,
            replace=args.replace,
        )
        print(
            f"已附加 {evidence_count} 条 ASR/OCR 证据："
            f"{args.samples_root}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
