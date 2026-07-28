"""Lossless, sidecar-only flattening for Feishu/Bitable import."""

from __future__ import annotations

import json
from typing import Iterable


ISSUE_COLUMNS = (
    "可定位性",
    "置信度",
    "问题说明",
    "问题类型",
    "时间区间",
    "关键帧秒",
    "BBox",
)


def flatten_prediction_rows(
    source_rows: Iterable[dict[str, str]],
    prediction_column: str = "GPT预测结果",
) -> list[dict[str, object]]:
    flattened: list[dict[str, object]] = []
    for source in source_rows:
        raw = (source.get(prediction_column) or "").strip()
        if not raw:
            continue
        value = json.loads(raw)
        if not isinstance(value, list):
            raise ValueError(f"{source.get('序号', '')} 的预测结果不是数组")
        for index, issue in enumerate(value, start=1):
            if not isinstance(issue, dict):
                raise ValueError(f"{source.get('序号', '')} 的问题点不是对象")
            row: dict[str, object] = {
                "序号": source.get("序号", ""),
                "问题序号": index,
            }
            row.update({column: issue.get(column, "") for column in ISSUE_COLUMNS})
            flattened.append(row)
    return flattened


def import_key(row: dict[str, object]) -> str:
    return f"{row.get('序号', '')}:{row.get('问题序号', '')}"


def build_bitable_records(rows: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    records = []
    for row in rows:
        fields = {"导入键": import_key(row)}
        fields.update(
            {
                key: value
                for key, value in row.items()
                if key not in {"导入键"} and value is not None
            }
        )
        records.append({"fields": fields})
    return records
