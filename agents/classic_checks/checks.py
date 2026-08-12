"""Ten compositional classic checks built from the current specialist tools."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import re
from typing import Any, Callable

from .assignment import (
    AUDIO_QUALITY_NOISE_ARTIFACTS,
    AV_LIP_SYNC,
    CLASSIC_CHECK_NAMES,
    DIALOGUE_SPEAKER_BINDING,
    ENTITY_COUNT_SPATIAL_COMPOSITION,
    MOTION_PHYSICS_CONTINUITY,
    REFERENCE_SUBJECT_IDENTITY,
    REMAINING_HARD_INSTRUCTION_COMPLIANCE,
    SUBTITLE_TEXT_LOGO_WATERMARK,
    VISUAL_QUALITY_TEMPORAL_ARTIFACTS,
    VOICE_CHARACTERISTICS,
    assign_issue,
    candidate_issue,
    issue_fingerprint,
)
from .context import EvaluationContext
from .contracts import ClassicCheckResult, ToolResult


METADATA_TOOL = "metadata"
GPT_A_TOOL = "gpt_a"
SEED_LITE_TOOL = "seed_lite"
AURALIS_TOOL = "auralis"
AVBENCH_TOOL = "avbench"
OCR_VISUAL_VERIFIER_TOOL = "ocr_visual_verifier"


CHECK_TOOL_DEPENDENCIES: Mapping[str, tuple[str, ...]] = {
    REFERENCE_SUBJECT_IDENTITY: (GPT_A_TOOL,),
    ENTITY_COUNT_SPATIAL_COMPOSITION: (GPT_A_TOOL,),
    REMAINING_HARD_INSTRUCTION_COMPLIANCE: (METADATA_TOOL, GPT_A_TOOL),
    MOTION_PHYSICS_CONTINUITY: (SEED_LITE_TOOL, GPT_A_TOOL),
    DIALOGUE_SPEAKER_BINDING: (AURALIS_TOOL,),
    VOICE_CHARACTERISTICS: (AURALIS_TOOL,),
    SUBTITLE_TEXT_LOGO_WATERMARK: (
        AURALIS_TOOL,
        OCR_VISUAL_VERIFIER_TOOL,
        SEED_LITE_TOOL,
        GPT_A_TOOL,
    ),
    AV_LIP_SYNC: (AVBENCH_TOOL,),
    VISUAL_QUALITY_TEMPORAL_ARTIFACTS: (
        METADATA_TOOL,
        GPT_A_TOOL,
        SEED_LITE_TOOL,
    ),
    AUDIO_QUALITY_NOISE_ARTIFACTS: (AURALIS_TOOL,),
}


_TOOL_EVIDENCE_LEVEL = {
    METADATA_TOOL: "deterministic",
    GPT_A_TOOL: "candidate",
    SEED_LITE_TOOL: "supported",
    AURALIS_TOOL: "supported",
    AVBENCH_TOOL: "supported",
    OCR_VISUAL_VERIFIER_TOOL: "supported",
}
_LEVEL_ORDER = {
    "none": 0,
    "candidate": 1,
    "supported": 2,
    "deterministic": 3,
}
_ISSUE_KEYS = frozenset(
    {
        "可定位性",
        "置信度",
        "问题说明",
        "问题类型",
        "时间区间",
        "关键帧秒",
        "BBox",
    }
)
_REFERENCE_AUDIO_PATTERN = re.compile(
    r"(?:参考\s*(?:音频|声音|音色)|音色\s*参考|"
    r"[@<【]?\s*(?:音频|声音素材)\s*[0-9一二三四五六七八九十]+)",
    re.IGNORECASE,
)
_JUNK_EXACT = frozenset(
    {
        "x",
        "test",
        "testcase",
        "test case",
        "测试",
        "测试case",
        "测试 case",
        "测试用例",
        "占位",
        "占位数据",
    }
)
_JUNK_PATTERN = re.compile(
    r"^(?:test\s*case|测试\s*case|测试用例|占位数据)[\s#_\-\d]*$",
    re.IGNORECASE,
)


def _tool_status(result: ToolResult) -> str:
    return str(getattr(result, "status", "failed") or "failed")


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _tool_artifacts(result: ToolResult) -> Mapping[str, Any]:
    return _as_mapping(getattr(result, "artifacts", {}))


def _merge_provenance(tool_name: str, provenance: Any) -> Mapping[str, Any]:
    base: dict[str, Any] = {}
    if isinstance(provenance, Mapping):
        base.update(provenance)
    elif str(provenance or "").strip():
        base["detail"] = str(provenance)
    # The context invocation, rather than model-authored metadata inside an
    # artifact, is the authoritative source for ownership decisions.
    base["source"] = tool_name
    return base


def _artifact_candidates(
    tool_name: str,
    result: ToolResult,
) -> list[tuple[dict[str, Any], Mapping[str, Any]]]:
    """Read only the runner's normalized ``artifacts['issues']`` channel."""

    if tool_name == OCR_VISUAL_VERIFIER_TOOL:
        # OCR verification supports or vetoes Auralis candidates.  Treating its
        # accepted list as a second source would duplicate the same issue.
        return []
    raw = _tool_artifacts(result).get("issues", ())
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return []
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return []

    candidates: list[tuple[dict[str, Any], Mapping[str, Any]]] = []
    seen: set[str] = set()
    for item in raw:
        try:
            issue, provenance = candidate_issue(item)
        except TypeError:
            continue
        if not _ISSUE_KEYS.issubset(issue):
            continue
        normalized = {key: issue.get(key, "") for key in _ISSUE_KEYS}
        fingerprint = issue_fingerprint(normalized)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        candidates.append(
            (normalized, _merge_provenance(tool_name, provenance))
        )
    return candidates


def _level_from_provenance(
    provenance: Mapping[str, Any],
    tool_name: str,
) -> str:
    explicit = str(provenance.get("evidence_level") or "")
    if explicit in _LEVEL_ORDER:
        return explicit
    return _TOOL_EVIDENCE_LEVEL[tool_name]


def _strongest_level(levels: Sequence[str]) -> str:
    return max(levels or ("none",), key=lambda item: _LEVEL_ORDER[item])


def _deduplicate_issues(issues: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for issue in issues:
        fingerprint = issue_fingerprint(issue)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        unique.append(dict(issue))
    return tuple(unique)


def _run_dependencies(
    context: EvaluationContext,
    check_name: str,
) -> tuple[tuple[str, ToolResult], ...]:
    return tuple(
        (tool_name, context.run_tool(tool_name))
        for tool_name in CHECK_TOOL_DEPENDENCIES[check_name]
    )


def _limitations_from_tools(
    runs: Sequence[tuple[str, ToolResult]],
) -> list[str]:
    limitations: list[str] = []
    for tool_name, result in runs:
        status = _tool_status(result)
        error = str(getattr(result, "error", "") or "").strip()
        if status == "failed":
            limitations.append(
                f"工具 {tool_name} 执行失败"
                + (f"：{error}" if error else "。")
            )
        elif status == "not_applicable":
            limitations.append(f"工具 {tool_name} 对当前样本不适用。")
    return limitations


def _owned_issues(
    check_name: str,
    runs: Sequence[tuple[str, ToolResult]],
) -> tuple[tuple[dict[str, Any], ...], tuple[str, ...]]:
    issues: list[dict[str, Any]] = []
    levels: list[str] = []
    for tool_name, result in runs:
        if _tool_status(result) != "ok":
            continue
        for issue, provenance in _artifact_candidates(tool_name, result):
            if assign_issue(issue, provenance) != check_name:
                continue
            issues.append(issue)
            levels.append(_level_from_provenance(provenance, tool_name))
    return _deduplicate_issues(issues), tuple(levels)


def _build_result(
    check_name: str,
    runs: Sequence[tuple[str, ToolResult]],
    *,
    base_limitations: Sequence[str] = (),
    force_not_evaluable: bool = False,
    override_issues: Sequence[Mapping[str, Any]] | None = None,
    override_levels: Sequence[str] = (),
) -> ClassicCheckResult:
    issues, levels = _owned_issues(check_name, runs)
    if override_issues is not None:
        issues = _deduplicate_issues(override_issues)
        levels = tuple(override_levels)

    statuses = [_tool_status(result) for _, result in runs]
    if any(status == "ok" for status in statuses):
        execution_status = "ok"
    elif any(status == "failed" for status in statuses):
        execution_status = "failed"
    else:
        execution_status = "not_applicable"

    if issues:
        decision = "detected"
    elif force_not_evaluable or execution_status != "ok":
        decision = "not_evaluable"
    else:
        decision = "not_detected"

    if issues:
        evidence_level = _strongest_level(levels)
    elif decision == "not_detected":
        evidence_level = _strongest_level(
            [
                _TOOL_EVIDENCE_LEVEL[tool_name]
                for tool_name, result in runs
                if _tool_status(result) == "ok"
            ]
        )
    else:
        evidence_level = "none"

    limitations = [*base_limitations, *_limitations_from_tools(runs)]
    return ClassicCheckResult(
        check_name=check_name,
        execution_status=execution_status,
        decision=decision,
        evidence_level=evidence_level,
        issues=issues,
        tool_refs=tuple(dict.fromkeys(tool_name for tool_name, _ in runs)),
        limitations=tuple(dict.fromkeys(item for item in limitations if item)),
    )


def check_reference_subject_identity(
    context: EvaluationContext,
) -> ClassicCheckResult:
    runs = _run_dependencies(context, REFERENCE_SUBJECT_IDENTITY)
    if not tuple(context.sample.reference_images):
        return _build_result(
            REFERENCE_SUBJECT_IDENTITY,
            runs,
            force_not_evaluable=True,
            override_issues=(),
            base_limitations=("缺少参考图，无法比较参考主体身份或绑定属性。",),
        )
    return _build_result(
        REFERENCE_SUBJECT_IDENTITY,
        runs,
        base_limitations=(
            "当前 GPT-A 为多模态候选证据，尚无独立人脸表征和跨帧身份跟踪。",
        ),
    )


def check_entity_count_spatial_composition(
    context: EvaluationContext,
) -> ClassicCheckResult:
    runs = _run_dependencies(context, ENTITY_COUNT_SPATIAL_COMPOSITION)
    return _build_result(
        ENTITY_COUNT_SPATIAL_COMPOSITION,
        runs,
        base_limitations=(
            "当前依赖 GPT-A 画面核查，尚无独立实体检测、计数、姿态和轨迹证据。",
        ),
    )


def check_remaining_hard_instruction_compliance(
    context: EvaluationContext,
) -> ClassicCheckResult:
    runs = _run_dependencies(context, REMAINING_HARD_INSTRUCTION_COMPLIANCE)
    return _build_result(
        REMAINING_HARD_INSTRUCTION_COMPLIANCE,
        runs,
        base_limitations=(
            "该检查只接收其他九项未归属的硬指令候选；主观诉求不构成正例。",
        ),
    )


def check_motion_physics_continuity(
    context: EvaluationContext,
) -> ClassicCheckResult:
    runs = _run_dependencies(context, MOTION_PHYSICS_CONTINUITY)
    return _build_result(
        MOTION_PHYSICS_CONTINUITY,
        runs,
        base_limitations=(
            "Seed-Lite 只覆盖动作连续性和局部结构候选；二维时序证据不能单独证明静态三维穿透。",
        ),
    )


def check_dialogue_speaker_binding(
    context: EvaluationContext,
) -> ClassicCheckResult:
    runs = _run_dependencies(context, DIALOGUE_SPEAKER_BINDING)
    return _build_result(
        DIALOGUE_SPEAKER_BINDING,
        runs,
        base_limitations=(
            "空 ASR 或单路 ASR 未命中不能单独证明台词缺失；角色绑定须有 prompt 锚点。",
        ),
    )


def check_voice_characteristics(
    context: EvaluationContext,
) -> ClassicCheckResult:
    runs = _run_dependencies(context, VOICE_CHARACTERISTICS)
    issues, _levels = _owned_issues(VOICE_CHARACTERISTICS, runs)
    if issues:
        return _build_result(VOICE_CHARACTERISTICS, runs)

    prompt = str(context.sample.prompt or "")
    requested_reference_audio = bool(_REFERENCE_AUDIO_PATTERN.search(prompt))
    reference_note = (
        "prompt 要求具体参考音频/音色比较，但 EvaluationSample 未提供参考音频。"
        if requested_reference_audio
        else "EvaluationSample 当前不含参考音频；没有明确声线冲突时不能判定具体音色一致。"
    )
    return _build_result(
        VOICE_CHARACTERISTICS,
        runs,
        force_not_evaluable=True,
        base_limitations=(reference_note,),
    )


def check_subtitle_text_logo_watermark(
    context: EvaluationContext,
) -> ClassicCheckResult:
    runs = _run_dependencies(context, SUBTITLE_TEXT_LOGO_WATERMARK)
    return _build_result(
        SUBTITLE_TEXT_LOGO_WATERMARK,
        runs,
        base_limitations=(
            "OCR 只提供定位候选；非确定性文字问题须经过高清视觉复核，短暂文字仍可能被 2 fps 抽样遗漏。",
        ),
    )


def _avbench_result(result: ToolResult) -> Mapping[str, Any]:
    raw = _tool_artifacts(result).get("result", {})
    return raw if isinstance(raw, Mapping) else {}


def _avbench_issue(raw: Mapping[str, Any]) -> dict[str, Any]:
    offset_frames = raw.get("offset_frames")
    offset_sec = raw.get("offset_sec")
    confidence = raw.get("confidence")
    confidence_status = str(raw.get("confidence_status") or "")
    face_tracks = raw.get("face_track_count")
    result_confidence = "中" if confidence_status == "uncertain" else "高"
    return {
        "可定位性": "否",
        "置信度": result_confidence,
        "问题说明": (
            "AVBench/SyncNet 对生成视频的人脸轨迹和音轨进行同步评估，"
            f"得到偏移 {offset_frames} 帧（{offset_sec} 秒）、confidence={confidence}、"
            f"confidence_status={confidence_status or 'unknown'}、"
            f"face_track_count={face_tracks}，且 sync_decision=desync_candidate；"
            "该证据支持音画口型不同步候选。"
        ),
        "问题类型": "音频质量问题",
        "时间区间": "",
        "关键帧秒": "",
        "BBox": "",
    }


def check_av_lip_sync(context: EvaluationContext) -> ClassicCheckResult:
    runs = _run_dependencies(context, AV_LIP_SYNC)
    tool_result = runs[0][1]
    raw = _avbench_result(tool_result)
    sync_decision = str(raw.get("sync_decision") or "")
    limitations: list[str] = [
        "AVBench 需要可用音轨和足够长的人脸轨迹；低分、低 confidence 或搜索边界命中不单独构成缺陷。"
    ]

    if sync_decision == "desync_candidate":
        issues, levels = _owned_issues(AV_LIP_SYNC, runs)
        if not issues:
            issues = (_avbench_issue(raw),)
            levels = ("supported",)
        return _build_result(
            AV_LIP_SYNC,
            runs,
            base_limitations=limitations,
            override_issues=issues,
            override_levels=levels,
        )

    if sync_decision in {"uncertain", "not_evaluable"} or (
        raw and not bool(raw.get("success", True))
    ):
        limitations.append(
            "当前 AVBench 结果不可形成同步正例或可靠负例："
            + (sync_decision or str(raw.get("status") or "unknown"))
            + "。"
        )
        return _build_result(
            AV_LIP_SYNC,
            runs,
            force_not_evaluable=True,
            override_issues=(),
            base_limitations=limitations,
        )

    return _build_result(
        AV_LIP_SYNC,
        runs,
        override_issues=(),
        base_limitations=limitations,
    )


def check_visual_quality_temporal_artifacts(
    context: EvaluationContext,
) -> ClassicCheckResult:
    runs = _run_dependencies(context, VISUAL_QUALITY_TEMPORAL_ARTIFACTS)
    return _build_result(
        VISUAL_QUALITY_TEMPORAL_ARTIFACTS,
        runs,
        base_limitations=(
            "当前没有独立 IQA、掉帧或全片闪烁检测器；ffprobe 分辨率硬约束属于剩余指令检查。",
        ),
    )


def check_audio_quality_noise_artifacts(
    context: EvaluationContext,
) -> ClassicCheckResult:
    runs = _run_dependencies(context, AUDIO_QUALITY_NOISE_ARTIFACTS)
    return _build_result(
        AUDIO_QUALITY_NOISE_ARTIFACTS,
        runs,
        base_limitations=(
            "当前主要依赖 Auralis 的直接音频判断，尚无独立削波、响度、静音、SNR 或频谱噪声门槛。",
        ),
    )


CHECK_FUNCTIONS: tuple[
    Callable[[EvaluationContext], ClassicCheckResult], ...
] = (
    check_reference_subject_identity,
    check_entity_count_spatial_composition,
    check_remaining_hard_instruction_compliance,
    check_motion_physics_continuity,
    check_dialogue_speaker_binding,
    check_voice_characteristics,
    check_subtitle_text_logo_watermark,
    check_av_lip_sync,
    check_visual_quality_temporal_artifacts,
    check_audio_quality_noise_artifacts,
)


def evaluate_all_classic_checks(
    context: EvaluationContext,
) -> tuple[ClassicCheckResult, ...]:
    """Run all ten checks in the stable taxonomy order."""

    results = tuple(check(context) for check in CHECK_FUNCTIONS)
    if tuple(result.check_name for result in results) != CLASSIC_CHECK_NAMES:
        raise AssertionError("经典检查实现顺序与 taxonomy 顺序不一致")
    return results


def _record_value(record: Any, name: str) -> Any:
    if isinstance(record, Mapping):
        return record.get(name)
    return getattr(record, name, None)


def classify_system_task_failure(record: Any) -> dict[str, Any]:
    """Classify operational failure without creating a media issue."""

    status = str(_record_value(record, "status") or "").casefold()
    success = _record_value(record, "success")
    error = str(_record_value(record, "error") or "").strip()
    failed = success is False or status in {"failed", "failure", "error"} or bool(error)
    if failed:
        decision = "failed"
        reason = error or status or "任务明确标记为失败"
    elif success is True or status in {"ok", "success", "succeeded", "complete"}:
        decision = "ok"
        reason = "任务明确完成"
    else:
        decision = "unknown"
        reason = "缺少可判定的运行状态或错误信息"
    return {
        "category": "system_task_failure",
        "status": decision,
        "is_failure": failed,
        "reason": reason,
        "issues": [],
    }


def _sample_text(sample: Any) -> str:
    values: list[str] = []
    for name in ("prompt", "feedback", "sample_id"):
        value = _record_value(sample, name)
        if value is not None:
            values.append(str(value).strip())
    return "\n".join(value for value in values if value)


def is_junk_test(sample: Any) -> bool:
    """Return true only for narrow, explicit test/placeholder records."""

    values = [
        str(_record_value(sample, name) or "").strip().casefold()
        for name in ("prompt", "feedback")
    ]
    nonempty = [value for value in values if value]
    return bool(nonempty) and all(
        value in _JUNK_EXACT or bool(_JUNK_PATTERN.fullmatch(value))
        for value in nonempty
    )


def classify_junk_test(sample: Any) -> dict[str, Any]:
    """Return data disposition; junk records never become media issues."""

    excluded = is_junk_test(sample)
    return {
        "category": "junk_test",
        "disposition": "exclude" if excluded else "include",
        "exclude": excluded,
        "reason": (
            "prompt/feedback 仅包含明确测试或占位文本"
            if excluded
            else "未命中严格测试/占位规则"
        ),
        "source_text": _sample_text(sample),
        "issues": [],
    }
