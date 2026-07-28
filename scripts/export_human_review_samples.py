#!/usr/bin/env python3
"""Export readable per-sample packages for human video review."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from av_eval.review_export import export_review_samples


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="导出逐样本人工复核目录")
    parser.add_argument("--gt-csv", type=Path, default=Path("input/gt.csv"))
    parser.add_argument("--media-root", type=Path, default=Path("input"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output/human_review_samples"),
    )
    parser.add_argument(
        "--gpt-a",
        type=Path,
        default=Path("output/benchmark/runs/gpt/baseline_a/pred.csv"),
    )
    parser.add_argument(
        "--gpt-b",
        type=Path,
        default=Path("output/benchmark/runs/gpt/harness_b/pred.csv"),
    )
    parser.add_argument(
        "--seed-a",
        type=Path,
        default=Path("output/benchmark/runs/seed_lite/baseline_a/pred.csv"),
    )
    parser.add_argument(
        "--seed-b",
        type=Path,
        default=Path("output/benchmark/runs/seed_lite/harness_b/pred.csv"),
    )
    parser.add_argument(
        "--seed-c",
        type=Path,
        default=Path("output/benchmark/runs/seed_lite/harness_c/pred.csv"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    prediction_csvs = {
        "gpt_a": args.gpt_a,
        "gpt_b": args.gpt_b,
        "seed_a": args.seed_a,
        "seed_b": args.seed_b,
        "seed_c": args.seed_c,
    }
    count = export_review_samples(
        gt_csv=args.gt_csv,
        prediction_csvs=prediction_csvs,
        media_root=args.media_root,
        output_root=args.output_root,
    )
    print(f"已导出 {count} 条样本：{args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
