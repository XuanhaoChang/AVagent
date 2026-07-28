"""Export one human-review directory per evaluation sample."""

from __future__ import annotations

import csv
import json
import shutil
import tempfile
import re
from pathlib import Path
from typing import Any

from .data import extract_gold_array, resolve_legacy_media_path


APPROVED_PREDICTIONS = ("gpt_a", "gpt_b", "seed_a", "seed_b", "seed_c")
INPUT_FIELDS = (
    "序号",
    "user_prompt",
    "用户反馈",
    "reference_image_urls",
    "generated_video_url",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"CSV 没有数据：{path}")
    sample_ids = [row.get("序号", "").strip() for row in rows]
    if any(not sample_id for sample_id in sample_ids):
        raise ValueError(f"CSV 存在空序号：{path}")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError(f"CSV 存在重复序号：{path}")
    return rows


def _parse_object_array(raw: str, source: str, sample_id: str) -> list[dict[str, Any]]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{source} 样本 {sample_id} 不是合法 JSON：{exc}") from exc
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"{source} 样本 {sample_id} 必须是对象数组")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _copy_media(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(f"媒体不存在：{source}")
    shutil.copy2(source, destination)
    if destination.stat().st_size != source.stat().st_size:
        raise RuntimeError(f"媒体复制后大小不一致：{source}")


def attach_prediction_to_review_samples(
    *,
    prediction_csv: Path,
    samples_root: Path,
    label: str,
    replace: bool = False,
) -> int:
    if re.fullmatch(r"[a-z][a-z0-9_]*", label) is None:
        raise ValueError(f"预测来源名称无效：{label!r}")
    rows = _read_csv(prediction_csv)
    sample_dirs = sorted(
        path for path in samples_root.glob("sample_*") if path.is_dir()
    )
    if len(rows) != len(sample_dirs):
        raise ValueError(
            f"预测行数与样本目录数不一致：{len(rows)} != {len(sample_dirs)}"
        )

    rows_by_id = {row["序号"].strip(): row for row in rows}
    sample_ids: set[str] = set()
    prepared: list[tuple[Path, list[dict[str, Any]]]] = []
    for sample_dir in sample_dirs:
        input_data = json.loads(
            (sample_dir / "input.json").read_text(encoding="utf-8")
        )
        sample_id = str(input_data.get("序号", "")).strip()
        if not sample_id or sample_id in sample_ids:
            raise ValueError(f"{sample_dir.name} 缺少有效唯一序号")
        sample_ids.add(sample_id)
        row = rows_by_id.get(sample_id)
        if row is None:
            raise ValueError(f"{sample_dir.name} 缺少预测：{sample_id}")
        destination = sample_dir / f"{label}.json"
        if destination.exists() and not replace:
            raise FileExistsError(f"预测文件已存在：{destination}")
        raw_prediction = row.get("GPT预测结果", "").strip()
        if not raw_prediction:
            raise ValueError(f"{label} 样本 {sample_id} 缺少预测")
        prepared.append(
            (
                destination,
                _parse_object_array(raw_prediction, label, sample_id),
            )
        )
    extra_ids = set(rows_by_id) - sample_ids
    if extra_ids:
        raise ValueError(f"预测 CSV 存在无对应样本的序号：{sorted(extra_ids)!r}")

    staged: list[tuple[Path, Path]] = []
    try:
        for destination, prediction in prepared:
            temporary = destination.with_name(f".{destination.name}.new")
            if temporary.exists():
                raise FileExistsError(f"临时预测文件已存在：{temporary}")
            _write_json(temporary, prediction)
            staged.append((temporary, destination))
        for temporary, destination in staged:
            temporary.replace(destination)
    except Exception:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)
        raise
    return len(prepared)


def export_review_samples(
    *,
    gt_csv: Path,
    prediction_csvs: dict[str, Path],
    media_root: Path,
    output_root: Path,
) -> int:
    labels = tuple(prediction_csvs)
    if labels != APPROVED_PREDICTIONS:
        raise ValueError(
            "预测来源必须按顺序为：" + ", ".join(APPROVED_PREDICTIONS)
        )
    if output_root.exists():
        if not output_root.is_dir() or any(output_root.iterdir()):
            raise FileExistsError(f"输出目录已存在且非空：{output_root}")

    gold_rows = _read_csv(gt_csv)
    gold_ids = [row["序号"].strip() for row in gold_rows]
    prediction_rows: dict[str, dict[str, dict[str, str]]] = {}
    for label, path in prediction_csvs.items():
        rows = _read_csv(path)
        ids = [row["序号"].strip() for row in rows]
        if ids != gold_ids:
            raise ValueError(f"{label} 的序号或行顺序与 GT 不一致：{path}")
        prediction_rows[label] = {row["序号"].strip(): row for row in rows}

    output_root.parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(
        tempfile.mkdtemp(
            prefix=f".{output_root.name}.",
            dir=output_root.parent,
        )
    )
    try:
        for row_index, gold_row in enumerate(gold_rows, start=1):
            sample_id = gold_row["序号"].strip()
            sample_dir = temp_root / f"sample_{row_index:03d}"
            sample_dir.mkdir()

            references_raw = gold_row.get("reference_image_urls", "")
            try:
                references = json.loads(references_raw)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"GT 样本 {sample_id} 的 reference_image_urls 不是合法 JSON"
                ) from exc
            if not isinstance(references, list) or not all(
                isinstance(item, str) for item in references
            ):
                raise ValueError(f"GT 样本 {sample_id} 的参考图必须是字符串数组")

            video_source = resolve_legacy_media_path(
                gold_row.get("generated_video_url", ""),
                media_root,
            )
            _copy_media(
                video_source,
                sample_dir / f"video{video_source.suffix.lower()}",
            )
            for reference_index, reference in enumerate(references, start=1):
                reference_source = resolve_legacy_media_path(reference, media_root)
                _copy_media(
                    reference_source,
                    sample_dir
                    / f"reference_{reference_index:02d}{reference_source.suffix.lower()}",
                )

            input_data = {field: gold_row.get(field, "") for field in INPUT_FIELDS}
            input_data["reference_image_urls"] = references
            _write_json(sample_dir / "input.json", input_data)

            gold = extract_gold_array(
                gold_row.get("思考过程及标准答案", "")
            )
            if gold.status != "valid":
                raise ValueError(
                    f"GT 样本 {sample_id} 无法解析：{gold.reason}"
                )
            _write_json(sample_dir / "gt.json", gold.items)

            for label in APPROVED_PREDICTIONS:
                raw_prediction = prediction_rows[label][sample_id].get(
                    "GPT预测结果",
                    "",
                ).strip()
                if not raw_prediction:
                    raise ValueError(f"{label} 样本 {sample_id} 缺少预测")
                prediction = _parse_object_array(
                    raw_prediction,
                    label,
                    sample_id,
                )
                _write_json(sample_dir / f"{label}.json", prediction)

        if output_root.exists():
            output_root.rmdir()
        temp_root.replace(output_root)
    except Exception:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise
    return len(gold_rows)
