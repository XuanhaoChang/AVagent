"""Pilot dataset audit that keeps gold isolated from inference inputs."""

from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .data import extract_gold_array, probe_media, resolve_legacy_media_path
@dataclass(frozen=True)
class AuditResult:
    summary: dict[str, Any]
    rows: list[dict[str, Any]]


def _read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def audit_dataset(csv_path: Path, media_root: Path, probe: bool = True) -> AuditResult:
    rows = _read_rows(csv_path)
    details: list[dict[str, Any]] = []
    resolved_videos = resolved_references = 0
    reference_total = valid_gold = 0
    parse_reasons: Counter[str] = Counter()
    has_audio = 0

    for row_index, row in enumerate(rows, start=1):
        references_error = ""
        try:
            references = json.loads(row.get("reference_image_urls", "") or "[]")
            if not isinstance(references, list) or not all(isinstance(x, str) for x in references):
                raise ValueError("reference_image_urls_not_string_array")
        except (json.JSONDecodeError, ValueError) as exc:
            references = []
            references_error = str(exc)
        reference_total += len(references)
        missing_references = []
        for reference in references:
            try:
                resolve_legacy_media_path(reference, media_root)
                resolved_references += 1
            except (FileNotFoundError, ValueError):
                missing_references.append(reference)

        video_error = ""
        metadata: dict[str, Any] = {}
        try:
            video_path = resolve_legacy_media_path(row.get("generated_video_url", ""), media_root)
            resolved_videos += 1
            if probe:
                metadata = probe_media(video_path)
                has_audio += int(bool(metadata["has_audio"]))
        except (FileNotFoundError, ValueError, OSError) as exc:
            video_error = str(exc)

        gold = extract_gold_array(row.get("思考过程及标准答案", ""))
        if gold.status == "valid":
            valid_gold += 1
        else:
            parse_reasons[gold.reason] += 1
        details.append(
            {
                "row_index": row_index,
                "序号": row.get("序号", ""),
                "reference_count": len(references),
                "missing_reference_count": len(missing_references),
                "references_error": references_error,
                "video_error": video_error,
                "gold_status": gold.status,
                "gold_reason": gold.reason,
                "gold_issue_count": len(gold.items),
                **metadata,
            }
        )

    summary = {
        "sample_count": len(rows),
        "reference_count": reference_total,
        "resolved_reference_count": resolved_references,
        "resolved_video_count": resolved_videos,
        "valid_gold_count": valid_gold,
        "needs_review_gold_count": len(rows) - valid_gold,
        "gold_parse_reasons": dict(sorted(parse_reasons.items())),
        "probed_video_count": sum(bool(row.get("video_codec")) for row in details),
        "audio_stream_video_count": has_audio if probe else None,
    }
    return AuditResult(summary, details)
