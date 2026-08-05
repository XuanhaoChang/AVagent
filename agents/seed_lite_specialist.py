"""Seed-Lite visual candidate agent with local evidence gates.

Seed-Lite supplies semantic, prompt-aware candidates.  Local OpenCV analysis
does not scan the full video for defects; it only verifies the time/BBox named
by each candidate and returns supported/contradicted/inconclusive evidence.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import call_ffmpeg_skill as visual_agent


DEFAULT_SEED_LITE_MODEL = "doubao-seed-2-0-lite-260428"

SEED_LITE_SPECIALIST_SYSTEM_MESSAGE = """你是 Agent-D 中职责受限的 Seed-Lite 视觉物理子智能体。

你只生成以下三类候选，其他问题一律不输出：
1. logo、商标、品牌字标、水印及服装/商品文字违反 prompt 或参考图约束；
2. 连续动作不符合基本物理规律，例如走路、转身、重心、脚步、姿态或交互轨迹发生无过渡跳变；
3. 人体或物体的畸变、穿模、穿插、粘连和实体边界冲突。

职责边界：
- 字幕、台词、音频、一般服装属性、人物身份、构图、审美和普通时序问题由其他智能体负责，不输出。
- 用户反馈只是候选线索，必须由生成视频画面证实，不能照抄反馈。
- logo 候选必须查看原分辨率关键帧或局部裁剪。若 prompt 明确要求无 logo，应报告实际出现
  的品牌式字标，而不是把参考图中的品牌误写成必须复刻的属性。
- 动作连续性必须比较时间有序的多帧，优先调用 make_contact_sheet；问题说明要指出起始状态、
  缺失的中间过渡和结束状态，不能仅凭一个关键帧判断动作。
- 穿模或畸变候选必须在多个相邻帧持续可见，并明确涉及哪些实体边界；单帧遮挡、运动模糊、
  衣物褶皱或二维重叠不足以定性。

每个候选必须是高置信度、可定位问题，包含规范化 0-1 BBox 和关键帧。“可定位性”只能写
“是”，“置信度”只能写“高”，BBox 必须写成 <bbox>x1,y1,x2,y2</bbox>。只输出 JSON 数组；
每个对象严格包含：可定位性、置信度、问题说明、问题类型、时间区间、关键帧秒、BBox。
没有满足条件的候选时输出 []。"""

_TIME_RANGE = re.compile(
    r"^\s*(?P<start>\d+(?:\.\d+)?)\s*(?:s|秒)?\s*[-–—]\s*"
    r"(?P<end>\d+(?:\.\d+)?)\s*(?:s|秒)?\s*$",
    re.IGNORECASE,
)
_BBOX = re.compile(
    r"^\s*(?:<bbox>\s*)?([+-]?\d+(?:\.\d+)?)\s*[, ]\s*"
    r"([+-]?\d+(?:\.\d+)?)\s*[, ]\s*"
    r"([+-]?\d+(?:\.\d+)?)\s*[, ]\s*"
    r"([+-]?\d+(?:\.\d+)?)(?:\s*</bbox>)?\s*$",
    re.IGNORECASE,
)
_KEYFRAME = re.compile(
    r"^\s*(?P<seconds>\d+(?:\.\d+)?)\s*(?:s|秒)?\s*$",
    re.IGNORECASE,
)
_LOGO_MARKERS = (
    "logo",
    "品牌",
    "商标",
    "字标",
    "品牌字母",
    "品牌文字",
    "印花文字",
    "水印",
)
_STRONG_LOGO_MARKERS = (
    "品牌",
    "商标",
    "字标",
    "品牌字母",
    "品牌文字",
    "印花文字",
    "水印",
)
_SUBTITLE_MARKERS = (
    "字幕",
    "台词文字",
    "底部文字",
)
_PHYSICAL_STRUCTURE_MARKERS = (
    "穿模",
    "穿插",
    "穿透",
    "粘连",
    "融合",
    "畸变",
    "变形",
    "实体边界",
    "结构崩坏",
)
_ACTION_CONTINUITY_MARKERS = (
    "无过渡",
    "缺少过渡",
    "不连贯",
    "跳变",
    "跳转",
    "突然切换",
    "直接切换",
    "脚步",
    "步态",
    "重心",
    "滑移",
    "瞬移",
    "运动轨迹",
    "动作衔接",
    "肢体运动",
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


def _video_size(video_path: Path) -> tuple[int, int]:
    import cv2

    capture = cv2.VideoCapture(str(video_path))
    try:
        return (
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        capture.release()


def _bbox_bounds(
    value: Any,
    *,
    video_width: int,
    video_height: int,
) -> tuple[float, float, float, float] | None:
    match = _BBOX.search(str(value or ""))
    if match is None:
        return None
    values = [float(item) for item in match.groups()]
    if max(values) > 1.0:
        if max(values) <= 1000.0:
            values = [item / 1000.0 for item in values]
        elif video_width > 0 and video_height > 0:
            values = [
                values[0] / video_width,
                values[1] / video_height,
                values[2] / video_width,
                values[3] / video_height,
            ]
    x1, y1, x2, y2 = values
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        return None
    return x1, y1, x2, y2


def _boxes_overlap(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> bool:
    return min(left[2], right[2]) > max(left[0], right[0]) and min(
        left[3], right[3]
    ) > max(left[1], right[1])


def _issue_category(issue: Mapping[str, Any]) -> str:
    description = str(issue.get("问题说明") or "").casefold()
    problem_type = str(issue.get("问题类型") or "").casefold()
    if any(marker in description for marker in _PHYSICAL_STRUCTURE_MARKERS):
        return "physical_structure"
    if "连续动作" in problem_type or any(
        marker in description for marker in _ACTION_CONTINUITY_MARKERS
    ):
        return "action_continuity"
    # A subtitle issue often repeats the prompt phrase "无 logo" even though
    # the observed defect is only a subtitle.  Do not let that incidental
    # requirement word escape the specialist scope filter.
    if any(marker in description for marker in _SUBTITLE_MARKERS):
        observed_description = description
        for separator in ("实际", "生成视频", "画面中", "视频中"):
            if separator in description:
                observed_description = description.split(separator, 1)[1]
                break
        if not any(
            marker.casefold() in observed_description
            for marker in _LOGO_MARKERS
        ):
            return "out_of_scope"
    if any(marker.casefold() in description for marker in _LOGO_MARKERS):
        return "logo_compliance"
    return "out_of_scope"


def _normalized_issue(
    issue: Mapping[str, Any],
    *,
    video_width: int,
    video_height: int,
) -> tuple[dict[str, Any] | None, str]:
    confidence = str(issue.get("置信度") or "").strip().casefold()
    if confidence not in {"高", "高置信度", "high"}:
        return None, "seed_candidate_not_high_confidence"
    localizable = str(issue.get("可定位性") or "").strip().casefold()
    # Treat this model-written label as advisory.  Concrete localizability is
    # enforced below by strict time/keyframe/BBox parsing, which is more robust
    # than chasing variants such as "可定位", "可精准定位" or "可明确识别".
    positive_localization = bool(localizable) and not any(
        marker in localizable
        for marker in (
            "不可",
            "不能",
            "无法",
            "否",
            "不精准",
            "未知",
            "不确定",
            "unknown",
            "false",
            "no",
        )
    )
    if not positive_localization:
        return None, "seed_candidate_not_localizable"
    bounds = _time_bounds(issue.get("时间区间"))
    if bounds is None:
        return None, "invalid_time_range"
    bbox = _bbox_bounds(
        issue.get("BBox"),
        video_width=video_width,
        video_height=video_height,
    )
    if bbox is None:
        return None, "invalid_bbox"
    keyframe_match = _KEYFRAME.match(str(issue.get("关键帧秒") or ""))
    if keyframe_match is None:
        return None, "invalid_keyframe"
    keyframe = float(keyframe_match.group("seconds"))
    if not (bounds[0] - 0.25 <= keyframe <= bounds[1] + 0.25):
        return None, "keyframe_outside_issue_range"
    category = _issue_category(issue)
    if category == "out_of_scope":
        return None, "seed_candidate_outside_specialist_scope"
    normalized = {
        key: issue.get(key, "") for key in visual_agent.OUTPUT_KEYS
    }
    normalized["可定位性"] = "是"
    normalized["置信度"] = "高"
    normalized["时间区间"] = f"{bounds[0]:g}s - {bounds[1]:g}s"
    normalized["关键帧秒"] = f"{keyframe:g}"
    normalized["BBox"] = (
        f"<bbox>{bbox[0]:.4f},{bbox[1]:.4f},"
        f"{bbox[2]:.4f},{bbox[3]:.4f}</bbox>"
    )
    if category == "logo_compliance":
        normalized["问题类型"] = "文字质量问题"
    elif category in {"action_continuity", "physical_structure"}:
        normalized["问题类型"] = "动作异常"
    return normalized, category


def _logo_has_high_resolution_evidence(
    issue: Mapping[str, Any],
    bbox: tuple[float, float, float, float],
    tool_calls: Sequence[Mapping[str, Any]],
) -> bool:
    bounds = _time_bounds(issue.get("时间区间"))
    if bounds is None:
        return False
    for call in tool_calls:
        if not call.get("ok") or call.get("name") not in {
            "extract_frame",
            "extract_crop",
        }:
            continue
        arguments = call.get("arguments")
        if not isinstance(arguments, Mapping):
            continue
        try:
            timestamp = float(arguments["timestamp_sec"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (bounds[0] - 0.5 <= timestamp <= bounds[1] + 0.5):
            continue
        if call.get("name") == "extract_crop":
            try:
                crop = tuple(
                    float(arguments[key]) for key in ("x1", "y1", "x2", "y2")
                )
            except (KeyError, TypeError, ValueError):
                continue
            if not _boxes_overlap(bbox, crop):
                continue
        return True
    return False


def analyze_local_motion_evidence(
    video_path: Path,
    *,
    start_sec: float,
    end_sec: float,
    bbox: tuple[float, float, float, float],
    target_fps: float = 12.0,
) -> dict[str, Any]:
    """Measure local temporal discontinuity without claiming scene semantics."""

    import cv2
    import numpy as np

    capture = cv2.VideoCapture(str(video_path))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if source_fps <= 0:
        source_fps = 25.0
    stride = max(1, int(round(source_fps / max(1.0, target_fps))))
    capture.set(cv2.CAP_PROP_POS_MSEC, max(0.0, start_sec) * 1000.0)
    frames: list[Any] = []
    frame_index = 0
    try:
        while len(frames) < 180:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp = float(capture.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0
            if timestamp > end_sec + 1.0 / source_fps:
                break
            if frame_index % stride == 0:
                height, width = frame.shape[:2]
                x1 = max(0, min(width - 1, int(bbox[0] * width)))
                y1 = max(0, min(height - 1, int(bbox[1] * height)))
                x2 = max(x1 + 2, min(width, int(math.ceil(bbox[2] * width))))
                y2 = max(y1 + 2, min(height, int(math.ceil(bbox[3] * height))))
                crop = frame[y1:y2, x1:x2]
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                gray = cv2.resize(gray, (320, 320), interpolation=cv2.INTER_AREA)
                frames.append(gray)
            frame_index += 1
    finally:
        capture.release()
    if len(frames) < 4:
        return {
            "status": "insufficient_frames",
            "frame_count": len(frames),
        }

    differences: list[float] = []
    histogram_correlations: list[float] = []
    lost_track_ratios: list[float] = []
    median_displacements: list[float] = []
    for previous, current in zip(frames, frames[1:]):
        differences.append(
            float(np.mean(cv2.absdiff(previous, current))) / 255.0
        )
        previous_hist = cv2.calcHist([previous], [0], None, [32], [0, 256])
        current_hist = cv2.calcHist([current], [0], None, [32], [0, 256])
        cv2.normalize(previous_hist, previous_hist)
        cv2.normalize(current_hist, current_hist)
        histogram_correlations.append(
            float(
                cv2.compareHist(
                    previous_hist,
                    current_hist,
                    cv2.HISTCMP_CORREL,
                )
            )
        )
        points = cv2.goodFeaturesToTrack(
            previous,
            maxCorners=160,
            qualityLevel=0.01,
            minDistance=5,
            blockSize=5,
        )
        if points is None or len(points) < 8:
            lost_track_ratios.append(1.0)
            median_displacements.append(0.0)
            continue
        tracked, status, _error = cv2.calcOpticalFlowPyrLK(
            previous,
            current,
            points,
            None,
            winSize=(21, 21),
            maxLevel=3,
        )
        if tracked is None or status is None:
            lost_track_ratios.append(1.0)
            median_displacements.append(0.0)
            continue
        valid = status.reshape(-1).astype(bool)
        lost_track_ratios.append(1.0 - float(np.mean(valid)))
        if not np.any(valid):
            median_displacements.append(0.0)
            continue
        displacement = np.linalg.norm(
            tracked.reshape(-1, 2)[valid] - points.reshape(-1, 2)[valid],
            axis=1,
        )
        median_displacements.append(
            float(np.median(displacement)) / math.hypot(320, 320)
        )

    diff_array = np.asarray(differences, dtype="float32")
    diff_median = float(np.median(diff_array))
    diff_mad = float(np.median(np.abs(diff_array - diff_median)))
    jump_threshold = max(0.11, diff_median + 4.0 * max(diff_mad, 0.005))
    cut_indices = [
        index
        for index, (difference, correlation) in enumerate(
            zip(differences, histogram_correlations)
        )
        if difference >= 0.18 and correlation <= 0.70
    ]
    abrupt_indices = [
        index
        for index, (difference, correlation, lost_ratio) in enumerate(
            zip(differences, histogram_correlations, lost_track_ratios)
        )
        if difference >= jump_threshold
        and (correlation <= 0.90 or lost_ratio >= 0.55)
    ]
    noncut_structure_indices = [
        index
        for index, (difference, correlation, lost_ratio) in enumerate(
            zip(differences, histogram_correlations, lost_track_ratios)
        )
        if index not in cut_indices
        and difference >= 0.09
        and correlation > 0.70
        and lost_ratio >= 0.72
    ]
    return {
        "status": "ok",
        "method": "opencv_local_histogram_lk_flow",
        "frame_count": len(frames),
        "sample_fps": round(source_fps / stride, 3),
        "pair_count": len(differences),
        "median_frame_difference": round(diff_median, 5),
        "max_frame_difference": round(max(differences), 5),
        "min_histogram_correlation": round(min(histogram_correlations), 5),
        "max_lost_track_ratio": round(max(lost_track_ratios), 5),
        "max_median_displacement": round(max(median_displacements), 5),
        "cut_like_event_count": len(cut_indices),
        "abrupt_event_count": len(abrupt_indices),
        "noncut_structure_instability_count": len(noncut_structure_indices),
        "limitations": (
            "2D point tracking cannot by itself prove static 3D penetration; "
            "inconclusive structure candidates stay out of the final union."
        ),
    }


def filter_seed_lite_candidates(
    prediction: str,
    *,
    video_path: Path,
    tool_calls: Sequence[Mapping[str, Any]] = (),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw = json.loads(visual_agent.parse_prediction(prediction))
    width, height = _video_size(video_path)
    accepted: list[dict[str, Any]] = []
    reviews: list[dict[str, Any]] = []
    for raw_issue in raw:
        issue, category_or_reason = _normalized_issue(
            raw_issue,
            video_width=width,
            video_height=height,
        )
        if issue is None:
            reviews.append(
                {
                    "candidate": dict(raw_issue),
                    "decision": "rejected",
                    "reason": category_or_reason,
                }
            )
            continue
        category = category_or_reason
        bounds = _time_bounds(issue["时间区间"])
        bbox = _bbox_bounds(
            issue["BBox"],
            video_width=width,
            video_height=height,
        )
        assert bounds is not None and bbox is not None
        if category == "logo_compliance":
            supported = _logo_has_high_resolution_evidence(
                issue,
                bbox,
                tool_calls,
            )
            evidence: dict[str, Any] = {
                "method": "seed_lite_high_resolution_visual_verification",
                "matching_high_resolution_call": supported,
            }
            decision = "supported" if supported else "inconclusive"
            reason = (
                "high_resolution_frame_or_crop_matches_logo_candidate"
                if supported
                else "missing_matching_high_resolution_logo_evidence"
            )
        else:
            evidence = analyze_local_motion_evidence(
                video_path,
                # Include a small context margin so a discontinuity exactly
                # at the candidate boundary is still visible to the tracker.
                start_sec=max(0.0, bounds[0] - 0.5),
                end_sec=bounds[1] + 0.5,
                bbox=bbox,
            )
            if evidence.get("status") != "ok":
                decision = "inconclusive"
                reason = "insufficient_local_motion_evidence"
            elif category == "action_continuity" and (
                int(evidence.get("abrupt_event_count", 0)) >= 1
                or int(evidence.get("cut_like_event_count", 0)) >= 1
            ):
                decision = "supported"
                reason = "local_tracking_supports_temporal_discontinuity"
            elif category == "action_continuity" and (
                float(evidence.get("max_frame_difference", 1.0)) < 0.10
                and float(evidence.get("max_lost_track_ratio", 1.0)) < 0.45
            ):
                decision = "contradicted"
                reason = "local_tracking_remains_continuous"
            elif category == "physical_structure" and int(
                evidence.get("noncut_structure_instability_count", 0)
            ) >= 2:
                decision = "supported"
                reason = "repeated_noncut_local_structure_instability"
            elif category == "action_continuity":
                decision = "inconclusive"
                reason = "local_tracking_does_not_confirm_action_discontinuity"
            else:
                decision = "inconclusive"
                reason = "local_2d_evidence_cannot_confirm_structure_claim"
        reviews.append(
            {
                "candidate": dict(issue),
                "category": category,
                "decision": decision,
                "reason": reason,
                "evidence": evidence,
            }
        )
        if decision == "supported":
            accepted.append(issue)
    return accepted, reviews


def run_seed_lite_specialist(
    input_data: dict[str, Any],
    *,
    api_url: str,
    api_key: str,
    model: str = DEFAULT_SEED_LITE_MODEL,
    timeout: int,
    api_retries: int,
    max_agent_steps: int = 8,
    run_stats: dict[str, Any] | None = None,
) -> str:
    stats = run_stats if run_stats is not None else {}
    raw_prediction = visual_agent.run_agent(
        input_data,
        api_url,
        api_key,
        model,
        timeout,
        api_retries,
        max_agent_steps,
        2.0,
        512,
        0,
        "none",
        None,
        None,
        True,
        stats,
        skill_text_override=SEED_LITE_SPECIALIST_SYSTEM_MESSAGE,
        runtime_instruction=(
            " Seed-Lite 专项候选将在返回后接受本地点轨迹和镜头变化门控；"
            "不得扩大职责范围来提高问题数量。"
        ),
    )
    stats["model"] = model
    stats["raw_prediction"] = json.loads(
        visual_agent.parse_prediction(raw_prediction)
    )
    video_path = visual_agent.ensure_video(
        str(input_data.get("generated_video_url") or "")
    )
    accepted, reviews = filter_seed_lite_candidates(
        raw_prediction,
        video_path=video_path,
        tool_calls=[
            item
            for item in stats.get("tool_calls", ())
            if isinstance(item, Mapping)
        ],
    )
    stats["candidate_reviews"] = reviews
    stats["accepted_issues"] = accepted
    stats["status"] = "ok"
    return json.dumps(accepted, ensure_ascii=False)
