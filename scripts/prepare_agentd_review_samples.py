#!/usr/bin/env python3
"""Prepare per-sample review packages for one Agent-D prediction CSV."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import tempfile
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from av_eval.data import extract_gold_array


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"CSV 为空：{path}")
    return rows


def _json(value: str, field: str, sample_id: str):
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{sample_id} 的 {field} 不是合法 JSON") from exc


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _copy_media(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"媒体不存在：{source}")
    shutil.copy2(source, destination)


def export(
    *,
    input_csv: Path,
    gt_csv: Path,
    prediction_csv: Path,
    output_root: Path,
) -> int:
    input_rows = _read_rows(input_csv)
    gt_rows = {row["序号"].strip(): row for row in _read_rows(gt_csv)}
    prediction_rows = {
        row["序号"].strip(): row for row in _read_rows(prediction_csv)
    }
    input_ids = [row["序号"].strip() for row in input_rows]
    if len(set(input_ids)) != len(input_ids):
        raise ValueError("输入 CSV 存在重复序号")
    if set(input_ids) != set(gt_rows) or set(input_ids) != set(prediction_rows):
        raise ValueError("输入、GT 和预测 CSV 的序号集合不一致")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"输出目录已存在且非空：{output_root}")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(
        tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent)
    )
    try:
        for index, input_row in enumerate(input_rows, start=1):
            sample_id = input_row["序号"].strip()
            gt_result = extract_gold_array(
                gt_rows[sample_id].get("思考过程及标准答案", "")
            )
            if gt_result.status != "valid":
                raise ValueError(f"{sample_id} 的 GT 无法解析：{gt_result.reason}")
            prediction = _json(
                prediction_rows[sample_id].get("GPT预测结果", ""),
                "GPT预测结果",
                sample_id,
            )
            if not isinstance(prediction, list) or not all(
                isinstance(item, dict) for item in prediction
            ):
                raise ValueError(f"{sample_id} 的 GPT预测结果 必须是对象数组")

            sample_dir = temporary_root / f"sample_{index:03d}"
            sample_dir.mkdir()
            input_data = {
                field: input_row.get(field, "")
                for field in (
                    "序号",
                    "user_prompt",
                    "用户反馈",
                    "reference_image_urls",
                    "generated_video_url",
                )
            }
            _write_json(sample_dir / "input.json", input_data)
            _write_json(sample_dir / "gt.json", gt_result.items)
            _write_json(sample_dir / "agentd.json", prediction)

            references = _json(
                input_row.get("reference_image_urls", "[]"),
                "reference_image_urls",
                sample_id,
            )
            if not isinstance(references, list):
                raise ValueError(f"{sample_id} 的 reference_image_urls 必须是数组")
            for ref_index, reference in enumerate(references, start=1):
                source = Path(str(reference))
                _copy_media(
                    source,
                    sample_dir / f"reference_{ref_index:02d}{source.suffix.lower()}",
                )
            video = Path(input_row.get("generated_video_url", ""))
            _copy_media(video, sample_dir / f"video{video.suffix.lower()}")

        if output_root.exists():
            output_root.rmdir()
        temporary_root.replace(output_root)
    except Exception:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise
    return len(input_rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="准备 Agent-D 五类复核样本包")
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--gt-csv", type=Path, required=True)
    parser.add_argument("--prediction-csv", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    count = export(
        input_csv=args.input_csv,
        gt_csv=args.gt_csv,
        prediction_csv=args.prediction_csv,
        output_root=args.output_root,
    )
    print(f"已准备 {count} 条复核样本：{args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
