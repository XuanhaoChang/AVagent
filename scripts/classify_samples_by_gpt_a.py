#!/usr/bin/env python3
"""Copy human-review samples into folders grouped by one review source."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


CATEGORY_DIRS = {
    1: "category_1_完整指出GT的问题",
    2: "category_2_指出GT的问题且指出别的问题",
    3: "category_3_没有完全指出GT的问题",
    4: "category_4_GT问题无法由框架解答",
    5: "category_5_输入材料不足",
}

REVIEW_FIELDS = (
    "prediction_source",
    "category",
    "category_name",
    "reason",
    "gt_coverage",
    "extra_prediction_indices",
    "confidence",
)
DEFAULT_OUTPUT_ROOT = Path("output/human_review_samples_by_gpt_a")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="按照指定预测来源的文本复核类别复制样本，并附上复核评语。",
    )
    parser.add_argument(
        "--samples-root",
        type=Path,
        default=Path("output/human_review_samples"),
    )
    parser.add_argument(
        "--reviews-jsonl",
        type=Path,
        default=Path("output/text_review/results.jsonl"),
    )
    parser.add_argument(
        "--prediction-source",
        default="gpt_a",
        help="要提取和分类的 prediction_source，默认 gpt_a。",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    return parser


def resolve_output_root(args: argparse.Namespace) -> Path:
    if (
        args.output_root == DEFAULT_OUTPUT_ROOT
        and args.prediction_source != "gpt_a"
    ):
        return Path(f"output/human_review_samples_by_{args.prediction_source}")
    return args.output_root


def read_source_reviews(
    path: Path,
    prediction_source: str,
) -> list[tuple[str, dict[str, Any]]]:
    if not path.is_file():
        raise FileNotFoundError(f"复核结果不存在：{path}")

    records: list[tuple[str, dict[str, Any]]] = []
    seen: set[str] = set()
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        result = json.loads(line)
        sample_id = result.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"{path}:{line_number} 缺少有效 sample_id")
        if sample_id in seen:
            raise ValueError(f"{path}:{line_number} 重复样本：{sample_id}")

        matches = [
            review
            for review in result.get("reviews", [])
            if review.get("prediction_source") == prediction_source
        ]
        if len(matches) != 1:
            raise ValueError(
                f"{path}:{line_number} 的 {sample_id} 应有且仅有一条 "
                f"{prediction_source} 复核，"
                f"实际为 {len(matches)} 条"
            )
        review = matches[0]
        category = review.get("category")
        if category not in CATEGORY_DIRS:
            raise ValueError(
                f"{path}:{line_number} 的 {sample_id} 类别无效：{category!r}"
            )

        seen.add(sample_id)
        records.append((sample_id, review))

    if not records:
        raise ValueError(f"复核结果为空：{path}")
    return records


def read_gpt_a_reviews(path: Path) -> list[tuple[str, dict[str, Any]]]:
    return read_source_reviews(path, "gpt_a")


def copy_classified_samples(
    samples_root: Path,
    output_root: Path,
    records: list[tuple[str, dict[str, Any]]],
    prediction_source: str = "gpt_a",
) -> Counter[int]:
    if output_root.exists():
        raise FileExistsError(f"输出目录已存在，为避免覆盖已停止：{output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)

    staging_root = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent)
    )
    counts: Counter[int] = Counter()
    try:
        for directory_name in CATEGORY_DIRS.values():
            (staging_root / directory_name).mkdir()

        for sample_id, review in records:
            source = samples_root / sample_id
            if not source.is_dir():
                raise FileNotFoundError(f"源样本目录不存在：{source}")

            category = review["category"]
            destination = staging_root / CATEGORY_DIRS[category] / sample_id
            shutil.copytree(source, destination)

            review_output = {"sample_id": sample_id}
            review_output.update(
                {field: review.get(field) for field in REVIEW_FIELDS}
            )
            (destination / f"{prediction_source}_review.json").write_text(
                json.dumps(review_output, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            counts[category] += 1

        summary = {
            "sample_count": sum(counts.values()),
            "prediction_source": prediction_source,
            "categories": {
                str(category): {
                    "directory": directory_name,
                    "count": counts[category],
                }
                for category, directory_name in CATEGORY_DIRS.items()
            },
        }
        (staging_root / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        staging_root.rename(output_root)
    except Exception:
        shutil.rmtree(staging_root, ignore_errors=True)
        raise
    return counts


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_root = resolve_output_root(args)
    records = read_source_reviews(args.reviews_jsonl, args.prediction_source)
    counts = copy_classified_samples(
        samples_root=args.samples_root,
        output_root=output_root,
        records=records,
        prediction_source=args.prediction_source,
    )
    details = "，".join(
        f"类别{category}={counts[category]}" for category in CATEGORY_DIRS
    )
    print(
        f"完成：{sum(counts.values())} 个样本 -> {output_root}（{details}）"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
