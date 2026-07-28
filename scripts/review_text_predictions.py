#!/usr/bin/env python3
"""Classify prediction coverage using only exported GT and prediction JSON."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from av_eval.project_env import load_project_env
from av_eval.text_review import (
    CATEGORY_NAMES,
    PREDICTION_SOURCES,
    build_missing_material_result,
    build_messages,
    chat_completion,
    missing_required_materials,
    parse_review_response,
    read_sample,
)


DEFAULT_API_URL = (
    "https://sd8fq9c4cuac30otu789g.apigateway-cn-beijing.volceapi.com/"
    "ark-router/v1/chat/completions"
)
DEFAULT_MODEL = "gpt-5.5-2026-04-24"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "读取 input、GT 与预测 JSON，用 GPT 对预测覆盖情况做 1-5 类文本复核；"
            "prompt 依赖但 input 未提供的参考素材强制归为第 5 类。"
        ),
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("output/human_review_samples"),
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path("output/text_review/results.jsonl"),
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("output/text_review/summary.json"),
    )
    parser.add_argument(
        "--api-url",
        default=os.getenv("VIDEO_EVAL_API_URL", DEFAULT_API_URL),
    )
    parser.add_argument(
        "--model",
        default=os.getenv("VIDEO_EVAL_MODEL", DEFAULT_MODEL),
    )
    parser.add_argument("--start", type=int, default=1, help="从第几个样本开始。")
    parser.add_argument("--limit", type=int, default=0, help="处理数量；0 表示全部。")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--api-retries", type=int, default=3)
    parser.add_argument(
        "--prediction-source",
        action="append",
        dest="prediction_sources",
        help=(
            "要复核的预测来源，可重复指定；默认复核 "
            + ",".join(PREDICTION_SOURCES)
        ),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只校验输入文件，不发送 API 请求。",
    )
    return parser


def _sample_dirs(input_root: Path) -> list[Path]:
    samples = sorted(path for path in input_root.glob("sample_*") if path.is_dir())
    if not samples:
        raise ValueError(f"没有找到 sample_* 目录：{input_root}")
    return samples


def _read_existing(
    path: Path,
    prediction_sources: tuple[str, ...] = PREDICTION_SOURCES,
) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    results = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict) or not value.get("sample_id"):
            raise ValueError(f"{path}:{line_number} 不是有效复核记录")
        reviews = value.get("reviews")
        sources = [
            review.get("prediction_source")
            for review in reviews
            if isinstance(review, dict)
        ] if isinstance(reviews, list) else []
        if sources != list(prediction_sources):
            raise ValueError(
                f"{path}:{line_number} 的预测来源与当前任务不一致："
                f"{sources!r} != {list(prediction_sources)!r}"
            )
        results.append(value)
    return results


def summarize_results(
    results: list[dict[str, Any]],
    prediction_sources: tuple[str, ...] = PREDICTION_SOURCES,
) -> dict[str, Any]:
    counts = {source: Counter() for source in prediction_sources}
    for result in results:
        for review in result.get("reviews", []):
            source = review.get("prediction_source")
            category = review.get("category")
            if source in counts and category in CATEGORY_NAMES:
                counts[source][str(category)] += 1
    return {
        "sample_count": len({result.get("sample_id") for result in results}),
        "category_names": {str(key): value for key, value in CATEGORY_NAMES.items()},
        "by_source": {
            source: {str(category): counts[source][str(category)] for category in CATEGORY_NAMES}
            for source in prediction_sources
        },
    }


def _write_summary(
    path: Path,
    results: list[dict[str, Any]],
    prediction_sources: tuple[str, ...] = PREDICTION_SOURCES,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            summarize_results(results, prediction_sources),
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    load_project_env(BASE_DIR / ".env.local")
    args = build_parser().parse_args(argv)
    prediction_sources = tuple(args.prediction_sources or PREDICTION_SOURCES)
    if len(set(prediction_sources)) != len(prediction_sources):
        raise ValueError("--prediction-source 不能重复")
    samples = _sample_dirs(args.input_root)
    start = max(0, args.start - 1)
    end = len(samples) if args.limit <= 0 else min(len(samples), start + args.limit)
    selected = samples[start:end]
    if not selected:
        raise ValueError("所选样本范围为空")

    for sample_dir in selected:
        read_sample(sample_dir, prediction_sources)
    if args.dry_run:
        print(f"dry-run: validated {len(selected)} samples; no API request sent")
        return 0

    api_key = os.getenv("ARK_API_KEY", "").strip()

    existing = (
        _read_existing(args.output_jsonl, prediction_sources)
        if args.resume
        else []
    )
    if args.output_jsonl.exists() and not args.resume:
        raise FileExistsError(
            f"输出已存在：{args.output_jsonl}；使用 --resume 续跑或换一个输出路径"
        )
    completed = {result["sample_id"] for result in existing}
    all_results = list(existing)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with args.output_jsonl.open("a", encoding="utf-8") as output:
        for position, sample_dir in enumerate(selected, start=1):
            sample_id = sample_dir.name
            if sample_id in completed:
                print(f"[{position:03d}/{len(selected)}] skip {sample_id}", flush=True)
                continue
            print(f"[{position:03d}/{len(selected)}] review {sample_id}", flush=True)
            input_data, gt, predictions = read_sample(
                sample_dir,
                prediction_sources,
            )
            missing_materials = missing_required_materials(input_data)
            started = time.monotonic()
            if missing_materials:
                result = build_missing_material_result(
                    sample_id,
                    prediction_sources,
                    gt,
                    missing_materials,
                )
                usage: dict[str, Any] = {}
                request_bytes = 0
            else:
                if not api_key:
                    raise ValueError(
                        "缺少 ARK_API_KEY；请检查项目根目录 .env.local"
                    )
                text, usage, request_bytes = chat_completion(
                    api_url=args.api_url,
                    api_key=api_key,
                    model=args.model,
                    messages=build_messages(
                        sample_id,
                        input_data,
                        gt,
                        predictions,
                        prediction_sources,
                    ),
                    timeout=args.timeout,
                    max_attempts=args.api_retries,
                )
                result = parse_review_response(
                    text,
                    sample_id,
                    prediction_sources,
                )
            result.update(
                {
                    "model": args.model,
                    "elapsed_sec": round(time.monotonic() - started, 3),
                    "request_bytes": request_bytes,
                    "usage": usage if isinstance(usage, dict) else {},
                }
            )
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
            output.flush()
            os.fsync(output.fileno())
            all_results.append(result)

    _write_summary(args.summary_json, all_results, prediction_sources)
    print(
        f"done: {len(all_results)} samples -> {args.output_jsonl}; "
        f"summary -> {args.summary_json}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
