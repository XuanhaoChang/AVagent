"""High-resolution visual gate for OCR-derived Auralis text issues.

OCR is deliberately treated as a proposal mechanism here.  A recognizer can
map stars, dashboard graphics, clothing hardware, or decorative inscriptions
to plausible characters with high confidence.  Before a non-deterministic
Auralis text issue enters Agent-D's required union, this module asks an
independent visual model to inspect the exact full-resolution frame and OCR
crop and classify what is actually visible.

The verifier never invents new issues.  It can only support, contradict, or
abstain on an existing Auralis issue, and every decision remains auditable in
the run log.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import tempfile
from typing import Any, Callable, Mapping, Sequence

import call_ffmpeg_skill as visual_agent


MAX_SEGMENTS_PER_ISSUE = 3
MAX_VERIFIED_ISSUES = 6

_TIME_RANGE = re.compile(
    r"^\s*(?P<start>\d+(?:\.\d+)?)\s*(?:s|秒)?\s*[-–—]\s*"
    r"(?P<end>\d+(?:\.\d+)?)\s*(?:s|秒)?\s*$",
    re.IGNORECASE,
)
_ALLOWED_DECISIONS = {"supported", "contradicted", "inconclusive"}
_ALLOWED_REGION_TYPES = {
    "subtitle_overlay",
    "scene_text_or_ui",
    "logo_or_brand",
    "non_text_texture",
    "no_visible_text",
    "unknown",
}


OCR_VISUAL_VERIFIER_SYSTEM_MESSAGE = """你是 Agent-D 的 OCR 高清视觉证据复核器，不是通用视频评测器。

你只裁决输入中已有的 Auralis 文字问题候选，不能添加新问题，也不能判断音频内容。OCR 字符、
OCR 置信度和 Auralis/Gemini 的问题说明都只是待核查候选，不是事实。每个候选都附有原始高清帧
和 OCR BBox 的上下文放大图；必须以像素证据为准。

对每个 candidate_id 判断：
- supported：画面确实存在候选所声称的可见文字，并且文字层级和用户要求使该具体问题成立；
- contradicted：候选框实际是星形/图标、衣纹、五官、建筑线条、配件结构、普通物体纹理，或者
  虽然存在场景文字/UI，但候选错误地把它说成了字幕；
- inconclusive：分辨率、遮挡或证据范围不足，无法可靠裁决。

分类 region_type 只能是：subtitle_overlay、scene_text_or_ui、logo_or_brand、non_text_texture、
no_visible_text、unknown。

严格区分要求范围：
1. “禁止字幕/无字幕”只约束叠加字幕；柱面铭文、仪表盘、招牌和场景 UI 不能被改写为字幕。
2. “ZERO text of any kind/任何文字都不能出现”可以覆盖真实场景文字和 logo，但几何形状恰好像
   字母不算文字。
3. 指定文字去除失败只在目标文字或目标区域仍可见时成立；星级图标不能被 OCR 乱码冒充残留字。
4. 字幕与语音不一致候选中，你不裁决语音，只判断候选区域是否真的是可与语音比较的字幕；
   仪表盘数字、商品 UI 或场景标牌不具备这一前提。
5. 真正可见但内容损坏的乱码仍然是文字，不能因为不成词就拒绝；要区分真实字形和非文字纹理。

只输出 JSON 数组，不要 Markdown。数组必须为每个 candidate_id 输出且只输出一个对象，形状为：
{"candidate_id":"ocr_001","decision":"supported|contradicted|inconclusive",
 "region_type":"subtitle_overlay|scene_text_or_ui|logo_or_brand|non_text_texture|no_visible_text|unknown",
 "reason":"一句简短的像素证据说明"}。"""


@dataclass(frozen=True)
class OcrIssueCandidate:
    candidate_id: str
    issue_index: int
    issue: Mapping[str, Any]
    segments: tuple[Mapping[str, Any], ...]


def _compact_text(value: Any) -> str:
    return "".join(
        character.casefold()
        for character in str(value or "")
        if character.isalnum()
    )


def _time_bounds(value: Any) -> tuple[float, float] | None:
    match = _TIME_RANGE.match(str(value or ""))
    if match is None:
        return None
    start = float(match.group("start"))
    end = float(match.group("end"))
    if not math.isfinite(start) or not math.isfinite(end) or end <= start:
        return None
    return start, end


def _overlaps(
    left_start: float,
    left_end: float,
    right_start: float,
    right_end: float,
) -> bool:
    return min(left_end, right_end) > max(left_start, right_start)


def _segment_bbox(segment: Mapping[str, Any]) -> tuple[float, float, float, float] | None:
    raw_bbox = segment.get("bbox")
    if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
        return None
    try:
        bbox = tuple(float(item) for item in raw_bbox)
    except (TypeError, ValueError):
        return None
    x1, y1, x2, y2 = bbox
    if not all(math.isfinite(item) for item in bbox):
        return None
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        return None
    return bbox


def _rank_segment(
    segment: Mapping[str, Any],
    description: str,
) -> tuple[int, float, float, int]:
    text = _compact_text(segment.get("text"))
    exact_description_match = bool(text) and text in description
    try:
        confidence = float(segment.get("confidence") or 0.0)
        duration = float(segment.get("end_sec") or 0.0) - float(
            segment.get("start_sec") or 0.0
        )
    except (TypeError, ValueError):
        confidence = 0.0
        duration = 0.0
    return (
        1 if exact_description_match else 0,
        confidence,
        max(0.0, duration),
        len(text),
    )


def _spatially_distinct(
    selected: Sequence[Mapping[str, Any]],
    candidate: Mapping[str, Any],
) -> bool:
    candidate_bbox = _segment_bbox(candidate)
    if candidate_bbox is None:
        return False
    candidate_center = (
        (candidate_bbox[0] + candidate_bbox[2]) / 2,
        (candidate_bbox[1] + candidate_bbox[3]) / 2,
    )
    for segment in selected:
        bbox = _segment_bbox(segment)
        if bbox is None:
            continue
        center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        same_text = _compact_text(segment.get("text")) == _compact_text(
            candidate.get("text")
        )
        if (
            same_text
            and abs(center[0] - candidate_center[0]) <= 0.08
            and abs(center[1] - candidate_center[1]) <= 0.06
        ):
            return False
    return True


def select_issue_segments(
    issue: Mapping[str, Any],
    subtitle_segments: Sequence[Mapping[str, Any]],
    *,
    maximum: int = MAX_SEGMENTS_PER_ISSUE,
) -> tuple[Mapping[str, Any], ...]:
    """Select OCR regions that most directly support one model-written issue."""

    bounds = _time_bounds(issue.get("时间区间"))
    if bounds is None:
        return ()
    overlapping: list[Mapping[str, Any]] = []
    for segment in subtitle_segments:
        if not isinstance(segment, Mapping) or _segment_bbox(segment) is None:
            continue
        try:
            start = float(segment.get("start_sec"))
            end = float(segment.get("end_sec"))
        except (TypeError, ValueError):
            continue
        if _overlaps(bounds[0], bounds[1], start, end):
            overlapping.append(segment)
    description = _compact_text(issue.get("问题说明"))
    ordered = sorted(
        overlapping,
        key=lambda segment: _rank_segment(segment, description),
        reverse=True,
    )
    exact_matches = [
        segment
        for segment in ordered
        if _rank_segment(segment, description)[0]
    ]
    if exact_matches:
        ordered = exact_matches
    selected: list[Mapping[str, Any]] = []
    for segment in ordered:
        if not _spatially_distinct(selected, segment):
            continue
        selected.append(segment)
        if len(selected) >= max(1, maximum):
            break
    return tuple(selected)


def build_ocr_issue_candidates(
    issues: Sequence[Mapping[str, Any]],
    *,
    subtitle_segments: Sequence[Mapping[str, Any]],
    deterministic_issues: Sequence[Mapping[str, Any]] = (),
) -> tuple[OcrIssueCandidate, ...]:
    """Build visual-verification candidates for non-deterministic text issues."""

    deterministic_keys = {
        json.dumps(dict(issue), ensure_ascii=False, sort_keys=True)
        for issue in deterministic_issues
        if isinstance(issue, Mapping)
    }
    candidates: list[OcrIssueCandidate] = []
    for index, issue in enumerate(issues):
        if str(issue.get("问题类型") or "") != "文字质量问题":
            continue
        issue_key = json.dumps(dict(issue), ensure_ascii=False, sort_keys=True)
        if issue_key in deterministic_keys:
            continue
        if _time_bounds(issue.get("时间区间")) is None:
            continue
        candidates.append(
            OcrIssueCandidate(
                candidate_id=f"ocr_{len(candidates) + 1:03d}",
                issue_index=index,
                issue=dict(issue),
                segments=select_issue_segments(issue, subtitle_segments),
            )
        )
    return tuple(candidates)


def _expanded_bbox(
    bbox: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    margin_x = max(0.025, min(0.12, width * 0.45))
    margin_y = max(0.025, min(0.12, height * 0.65))
    return (
        max(0.0, bbox[0] - margin_x),
        max(0.0, bbox[1] - margin_y),
        min(1.0, bbox[2] + margin_x),
        min(1.0, bbox[3] + margin_y),
    )


def _candidate_timestamp(candidate: OcrIssueCandidate) -> float:
    if candidate.segments:
        return _segment_timestamp(candidate.segments[0])
    bounds = _time_bounds(candidate.issue.get("时间区间"))
    assert bounds is not None
    return (bounds[0] + bounds[1]) / 2


def _segment_timestamp(segment: Mapping[str, Any]) -> float:
    start = float(segment.get("start_sec") or 0.0)
    end = float(segment.get("end_sec") or start)
    return start if end <= start else (start + end) / 2


def build_verification_messages(
    *,
    user_prompt: str,
    candidates: Sequence[OcrIssueCandidate],
    video_path: Path,
    reference_images: Sequence[str] = (),
    temp_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build one no-tools multimodal request and an auditable crop manifest."""

    metadata = visual_agent.probe_video(video_path)
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                "用户原始 prompt：\n"
                + str(user_prompt or "").strip()
                + "\n\n请只裁决下列候选；OCR 内容不可信。"
            ),
        }
    ]
    manifest: list[dict[str, Any]] = []
    for reference_index, reference in enumerate(reference_images[:4], start=1):
        try:
            source = visual_agent.local_media_path(
                str(reference),
                f"OCR 复核参考图 {reference_index}",
            )
            proxy = visual_agent.prepare_model_image(
                source,
                temp_dir / f"reference_{reference_index:02d}.jpg",
            )
        except (FileNotFoundError, ValueError, RuntimeError):
            continue
        content.extend(
            [
                {"type": "text", "text": f"参考图 {reference_index}"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": visual_agent.image_data_url(proxy),
                        "detail": "high",
                    },
                },
            ]
        )

    for candidate_index, candidate in enumerate(candidates, start=1):
        timestamp = _candidate_timestamp(candidate)
        frame_path = temp_dir / f"{candidate.candidate_id}_frame.jpg"
        frame = visual_agent.extract_frame(
            video_path,
            frame_path,
            metadata,
            {"timestamp_sec": timestamp},
        )
        evidence_description = {
            "candidate_id": candidate.candidate_id,
            "auralis_issue": dict(candidate.issue),
            "ocr_candidates": [dict(segment) for segment in candidate.segments],
            "evidence_note": (
                "ocr_candidates 只是定位线索；如果为空，请在候选时间中检查"
                "所声称的字幕存在或缺失。"
            ),
        }
        content.extend(
            [
                {
                    "type": "text",
                    "text": json.dumps(evidence_description, ensure_ascii=False),
                },
                {
                    "type": "text",
                    "text": f"{candidate.candidate_id} 的原始高清上下文帧：{frame['description']}",
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": visual_agent.image_data_url(frame_path),
                        "detail": "high",
                    },
                },
            ]
        )
        candidate_manifest: dict[str, Any] = {
            "candidate_id": candidate.candidate_id,
            "issue_index": candidate.issue_index,
            "frame_timestamp_sec": timestamp,
            "crops": [],
        }
        for segment_index, segment in enumerate(candidate.segments, start=1):
            bbox = _segment_bbox(segment)
            if bbox is None:
                continue
            expanded_bbox = _expanded_bbox(bbox)
            segment_timestamp = _segment_timestamp(segment)
            crop_path = temp_dir / (
                f"{candidate.candidate_id}_crop_{segment_index:02d}.jpg"
            )
            crop = visual_agent.extract_crop(
                video_path,
                crop_path,
                metadata,
                {
                    "timestamp_sec": segment_timestamp,
                    "x1": expanded_bbox[0],
                    "y1": expanded_bbox[1],
                    "x2": expanded_bbox[2],
                    "y2": expanded_bbox[3],
                },
            )
            content.extend(
                [
                    {
                        "type": "text",
                        "text": (
                            f"{candidate.candidate_id} OCR 局部 {segment_index}；"
                            f"OCR 猜测={segment.get('text', '')!s}；{crop['description']}"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": visual_agent.image_data_url(crop_path),
                            "detail": "high",
                        },
                    },
                ]
            )
            candidate_manifest["crops"].append(
                {
                    "timestamp_sec": segment_timestamp,
                    "ocr_text": str(segment.get("text") or ""),
                    "ocr_confidence": float(segment.get("confidence") or 0.0),
                    "bbox": list(bbox),
                    "expanded_bbox": list(expanded_bbox),
                }
            )
        manifest.append(candidate_manifest)

    content.append(
        {
            "type": "text",
            "text": (
                "逐项输出全部 candidate_id 的裁决。不要因为 OCR 写出了字符就假定图中有字；"
                "也不要因为真实乱码不成词就否定其文字属性。"
            ),
        }
    )
    return (
        [
            {"role": "system", "content": OCR_VISUAL_VERIFIER_SYSTEM_MESSAGE},
            {"role": "user", "content": content},
        ],
        manifest,
    )


def _extract_json_array(text: str) -> list[Any]:
    stripped = str(text or "").strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[-1]
        stripped = stripped.rsplit("```", 1)[0].strip()
    start = stripped.find("[")
    end = stripped.rfind("]")
    if start < 0 or end < start:
        raise ValueError("OCR 视觉复核器未返回 JSON 数组")
    value = json.loads(stripped[start : end + 1])
    if not isinstance(value, list):
        raise ValueError("OCR 视觉复核器结果必须是数组")
    return value


def parse_verdicts(
    text: str,
    *,
    candidate_ids: Sequence[str],
) -> dict[str, dict[str, str]]:
    """Parse one verdict per known candidate; malformed entries abstain."""

    expected = set(candidate_ids)
    parsed: dict[str, dict[str, str]] = {}
    for raw in _extract_json_array(text):
        if not isinstance(raw, Mapping):
            continue
        candidate_id = str(raw.get("candidate_id") or "")
        if candidate_id not in expected or candidate_id in parsed:
            continue
        decision = str(raw.get("decision") or "").strip().casefold()
        region_type = str(raw.get("region_type") or "").strip().casefold()
        if decision not in _ALLOWED_DECISIONS:
            decision = "inconclusive"
        if region_type not in _ALLOWED_REGION_TYPES:
            region_type = "unknown"
        parsed[candidate_id] = {
            "candidate_id": candidate_id,
            "decision": decision,
            "region_type": region_type,
            "reason": str(raw.get("reason") or "").strip()[:500],
        }
    for candidate_id in candidate_ids:
        parsed.setdefault(
            candidate_id,
            {
                "candidate_id": candidate_id,
                "decision": "inconclusive",
                "region_type": "unknown",
                "reason": "missing_or_malformed_verifier_result",
            },
        )
    return parsed


def apply_verdicts(
    issues: Sequence[Mapping[str, Any]],
    *,
    candidates: Sequence[OcrIssueCandidate],
    verdicts: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep supported candidates and audit every contradiction/abstention."""

    candidate_by_index = {candidate.issue_index: candidate for candidate in candidates}
    accepted: list[dict[str, Any]] = []
    reviews: list[dict[str, Any]] = []
    for index, issue in enumerate(issues):
        candidate = candidate_by_index.get(index)
        if candidate is None:
            accepted.append(dict(issue))
            continue
        verdict = dict(verdicts.get(candidate.candidate_id, {}))
        decision = str(verdict.get("decision") or "inconclusive")
        reviews.append(
            {
                "candidate_id": candidate.candidate_id,
                "issue": dict(issue),
                "decision": decision,
                "region_type": str(verdict.get("region_type") or "unknown"),
                "reason": str(verdict.get("reason") or ""),
                "ocr_candidates": [dict(segment) for segment in candidate.segments],
            }
        )
        if decision == "supported":
            accepted.append(dict(issue))
    return accepted, reviews


def verify_auralis_ocr_issues(
    issues: Sequence[Mapping[str, Any]],
    *,
    subtitle_segments: Sequence[Mapping[str, Any]],
    deterministic_issues: Sequence[Mapping[str, Any]],
    user_prompt: str,
    video_path: Path,
    reference_images: Sequence[str],
    complete: Callable[[list[dict[str, Any]]], str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run the visual gate once and return accepted issues plus diagnostics."""

    candidates = build_ocr_issue_candidates(
        issues,
        subtitle_segments=subtitle_segments,
        deterministic_issues=deterministic_issues,
    )
    if not candidates:
        return [dict(issue) for issue in issues], {
            "status": "not_needed",
            "candidate_count": 0,
            "candidate_reviews": [],
        }
    verdicts: dict[str, dict[str, str]] = {}
    manifest: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="ocr_visual_verifier_") as temp_text:
        root = Path(temp_text)
        for batch_index, start in enumerate(
            range(0, len(candidates), MAX_VERIFIED_ISSUES),
            start=1,
        ):
            batch = candidates[start : start + MAX_VERIFIED_ISSUES]
            batch_dir = root / f"batch_{batch_index:02d}"
            batch_dir.mkdir(parents=True, exist_ok=True)
            messages, batch_manifest = build_verification_messages(
                user_prompt=user_prompt,
                candidates=batch,
                video_path=video_path,
                reference_images=reference_images,
                temp_dir=batch_dir,
            )
            raw_response = complete(messages)
            verdicts.update(
                parse_verdicts(
                    raw_response,
                    candidate_ids=[candidate.candidate_id for candidate in batch],
                )
            )
            manifest.extend(batch_manifest)
    accepted, reviews = apply_verdicts(
        issues,
        candidates=candidates,
        verdicts=verdicts,
    )
    return accepted, {
        "status": "ok",
        "candidate_count": len(candidates),
        "input_candidates": [
            {
                "candidate_id": candidate.candidate_id,
                "issue_index": candidate.issue_index,
                "issue": dict(candidate.issue),
                "ocr_candidates": [dict(segment) for segment in candidate.segments],
            }
            for candidate in candidates
        ],
        "crop_manifest": manifest,
        "candidate_reviews": reviews,
        "accepted_issue_count": len(accepted),
        "rejected_or_abstained_count": len(issues) - len(accepted),
    }
