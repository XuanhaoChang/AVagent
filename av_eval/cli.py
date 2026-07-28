"""Command-line entry points for reproducible offline evaluation."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

from .audit import audit_dataset
from .experiments import capacity_matrix, experiment_profiles
from .feishu_export import ISSUE_COLUMNS, flatten_prediction_rows
from .routing import route_observations
from .taxonomy import EXPLORATORY_DIMENSIONS, JING_TAXONOMY


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = columns or (list(rows[0]) if rows else [])
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _taxonomy_markdown() -> str:
    lines = [
        "# 经典问题 Taxonomy v0.1",
        "",
        "> 用户反馈是高优先级核查线索，不是单条问题的真值。Jing 频次只用于总体优先级。",
        "",
        "| Key | 中文名 | Jing 频次 | 必要证据 | 候选工具 |",
        "|---|---:|---:|---|---|",
    ]
    for item in JING_TAXONOMY:
        lines.append(
            f"| `{item.key}` | {item.name} | {item.jing_count} | {item.evidence} | "
            f"{'、'.join(item.candidate_tools)} |"
        )
    lines.extend(
        [
            "",
            "## 探索维度",
            "",
            "、".join(EXPLORATORY_DIMENSIONS)
            + "。这些维度单独人工评审，不强制映射到现有受控标签。",
            "",
            "## 判定约定",
            "",
            "- 正例：预期依据和生成结果证据均存在，并能描述明确差异。",
            "- 负例：完成相应模态核查后没有发现该类差异。",
            "- 不可判断：缺少必要模态、参考绑定不清、媒体不可用或证据强度不足。",
        ]
    )
    return "\n".join(lines) + "\n"


def _audit_markdown(summary: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# 100 条 Pilot 数据审计报告",
            "",
            f"- 样本数：{summary['sample_count']}",
            f"- 视频解析：{summary['resolved_video_count']}/{summary['sample_count']}",
            f"- 参考图解析：{summary['resolved_reference_count']}/{summary['reference_count']}",
            f"- 有效 fenced JSON 真值：{summary['valid_gold_count']}/{summary['sample_count']}",
            f"- 待人工复核真值：{summary['needs_review_gold_count']}",
            f"- 已 ffprobe 视频：{summary['probed_video_count']}",
            f"- 含音轨视频：{summary['audio_stream_video_count']}",
            "",
            "## 真值解析异常",
            "",
            "```json",
            json.dumps(summary["gold_parse_reasons"], ensure_ascii=False, indent=2),
            "```",
            "",
            "原始 CSV 未被修改；标准答案仅在离线审计中读取，不进入模型请求。",
        ]
    ) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AVagent 经典问题离线评测工具")
    commands = parser.add_subparsers(dest="command", required=True)

    capacity = commands.add_parser("capacity-plan")
    capacity.add_argument("--duration-sec", type=float, default=30.0)
    capacity.add_argument("--output", type=Path, required=True)

    taxonomy = commands.add_parser("taxonomy")
    taxonomy.add_argument("--output", type=Path, required=True)

    audit = commands.add_parser("audit")
    audit.add_argument("--gt", type=Path, default=Path("input/gt.csv"))
    audit.add_argument("--media-root", type=Path, default=Path("input"))
    audit.add_argument("--output-dir", type=Path, default=Path("output/benchmark/audit"))
    audit.add_argument("--no-probe", action="store_true")

    route = commands.add_parser("route")
    route.add_argument("--gt", type=Path, default=Path("input/gt.csv"))
    route.add_argument("--audit-manifest", type=Path, required=True)
    route.add_argument("--output", type=Path, required=True)

    review = commands.add_parser("review-queue")
    review.add_argument("--gt", type=Path, default=Path("input/gt.csv"))
    review.add_argument("--output", type=Path, required=True)
    review.add_argument("--limit", type=int, default=20)

    feishu = commands.add_parser("feishu-flat")
    feishu.add_argument("--pred", type=Path, required=True)
    feishu.add_argument("--output", type=Path, required=True)

    profiles = commands.add_parser("profiles")
    profiles.add_argument("--output", type=Path, required=True)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "capacity-plan":
        _write_json(args.output, capacity_matrix(args.duration_sec))
    elif args.command == "taxonomy":
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(_taxonomy_markdown(), encoding="utf-8")
    elif args.command == "audit":
        result = audit_dataset(args.gt, args.media_root, probe=not args.no_probe)
        _write_json(args.output_dir / "audit_summary.json", result.summary)
        _write_csv(args.output_dir / "media_manifest.csv", result.rows)
        _write_csv(
            args.output_dir / "gold_review_queue.csv",
            [
                {
                    "row_index": row["row_index"],
                    "序号": row["序号"],
                    "gold_reason": row["gold_reason"],
                    "人工复核结论": "",
                    "人工备注": "",
                }
                for row in result.rows
                if row["gold_status"] != "valid"
            ],
            ["row_index", "序号", "gold_reason", "人工复核结论", "人工备注"],
        )
        (args.output_dir / "pilot_data_audit.md").write_text(
            _audit_markdown(result.summary), encoding="utf-8"
        )
    elif args.command == "route":
        source = _read_csv(args.gt)
        manifest = {
            row["序号"]: row
            for row in _read_csv(args.audit_manifest)
            if row.get("序号")
        }
        rows = []
        for row in source:
            meta = manifest.get(row.get("序号", ""), {})
            decision = route_observations(
                row.get("user_prompt", ""),
                row.get("用户反馈", ""),
                str(meta.get("has_audio", "")).lower() == "true",
                int(meta.get("reference_count") or 0),
                float(meta.get("duration_sec") or 0),
            )
            rows.append(
                {
                    "序号": row.get("序号", ""),
                    "experts": "|".join(decision.experts),
                    "dense_sampling": decision.dense_sampling,
                    "local_crop": decision.local_crop,
                    "model_tier_candidate": decision.model_tier_candidate,
                    "reasons": "|".join(decision.reasons),
                }
            )
        _write_csv(args.output, rows)
        expert_names = ("asr", "av_sync", "ocr", "identity")
        expert_counts = {
            expert: sum(expert in row["experts"].split("|") for row in rows)
            for expert in expert_names
        }
        routed_calls = sum(expert_counts.values())
        all_tools_calls = len(rows) * len(expert_names)
        _write_json(
            args.output.with_suffix(".summary.json"),
            {
                "sample_count": len(rows),
                "model_tier_candidate_counts": {
                    tier: sum(row["model_tier_candidate"] == tier for row in rows)
                    for tier in ("seed_lite_candidate", "gpt_candidate")
                },
                "routed_expert_call_counts": expert_counts,
                "routed_expert_calls": routed_calls,
                "all_tools_reference_calls": all_tools_calls,
                "candidate_call_reduction_rate": (
                    round(1 - routed_calls / all_tools_calls, 6)
                    if all_tools_calls
                    else None
                ),
                "note": "仅为可观测信号路由统计，必须用真实准确率和延迟实验验证。",
            },
        )
    elif args.command == "review-queue":
        allowed = (
            "序号",
            "user_prompt",
            "reference_image_urls",
            "generated_video_url",
            "用户反馈",
        )
        rows = []
        for source in _read_csv(args.gt)[: max(0, args.limit)]:
            row = {key: source.get(key, "") for key in allowed}
            row.update({dimension: "" for dimension in EXPLORATORY_DIMENSIONS})
            row.update({"评审证据": "", "评审备注": ""})
            rows.append(row)
        _write_csv(args.output, rows)
    elif args.command == "feishu-flat":
        rows = flatten_prediction_rows(_read_csv(args.pred))
        _write_csv(
            args.output,
            rows,
            ["序号", "问题序号", *ISSUE_COLUMNS],
        )
    elif args.command == "profiles":
        _write_json(args.output, experiment_profiles())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
