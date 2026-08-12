"""Deterministic ownership for classic-check issue candidates.

The current specialists use one shared seven-field issue schema, but their
``问题类型`` values are intentionally broader than the classic-check taxonomy.
This module assigns each candidate to exactly one classic check.  Provenance is
considered before free-text semantics because an AVBench decision or an OCR
visual-gate result is more specific than a model-written description.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import json
from typing import Any


REFERENCE_SUBJECT_IDENTITY = "reference_subject_identity"
ENTITY_COUNT_SPATIAL_COMPOSITION = "entity_count_spatial_composition"
REMAINING_HARD_INSTRUCTION_COMPLIANCE = (
    "remaining_hard_instruction_compliance"
)
MOTION_PHYSICS_CONTINUITY = "motion_physics_continuity"
DIALOGUE_SPEAKER_BINDING = "dialogue_speaker_binding"
VOICE_CHARACTERISTICS = "voice_characteristics"
SUBTITLE_TEXT_LOGO_WATERMARK = "subtitle_text_logo_watermark"
AV_LIP_SYNC = "av_lip_sync"
VISUAL_QUALITY_TEMPORAL_ARTIFACTS = "visual_quality_temporal_artifacts"
AUDIO_QUALITY_NOISE_ARTIFACTS = "audio_quality_noise_artifacts"

CLASSIC_CHECK_NAMES = (
    REFERENCE_SUBJECT_IDENTITY,
    ENTITY_COUNT_SPATIAL_COMPOSITION,
    REMAINING_HARD_INSTRUCTION_COMPLIANCE,
    MOTION_PHYSICS_CONTINUITY,
    DIALOGUE_SPEAKER_BINDING,
    VOICE_CHARACTERISTICS,
    SUBTITLE_TEXT_LOGO_WATERMARK,
    AV_LIP_SYNC,
    VISUAL_QUALITY_TEMPORAL_ARTIFACTS,
    AUDIO_QUALITY_NOISE_ARTIFACTS,
)


_SYNC_MARKERS = (
    "音画同步",
    "声画同步",
    "口型同步",
    "口型延迟",
    "对嘴",
    "唇音",
    "syncnet",
    "desync",
    "offsetframes",
    "偏移帧",
)
_VOICE_MARKERS = (
    "音色",
    "音调",
    "声线",
    "男声",
    "女声",
    "童声",
    "年龄声",
    "口音",
    "情绪声音",
    "语气不符",
)
_AUDIO_ARTIFACT_MARKERS = (
    "杂音",
    "噪音",
    "电流声",
    "爆音",
    "削波",
    "断音",
    "卡顿",
    "异常静音",
    "音量突变",
    "重复声音",
    "不自然拼接",
    "底噪",
)
_DIALOGUE_MARKERS = (
    "台词",
    "读音",
    "发音",
    "念错",
    "没念完",
    "未念完",
    "没说完",
    "错读",
    "漏读",
    "多读",
    "语种",
    "语言错误",
    "自动说话",
    "多余语音",
    "不开嘴",
    "没有开嘴",
    "嘴不动",
    "说话人",
    "角色绑定",
    "台词归属",
    "配音主体",
    "匿名声纹",
    "共用声纹",
    "cam++",
    "spk",
    "asr",
    "ctc",
)
_TEXT_MARKERS = (
    "字幕",
    "文字",
    "错别字",
    "漏字",
    "多字",
    "乱码",
    "水印",
    "logo",
    "商标",
    "品牌字标",
    "ocr",
)
_IDENTITY_MARKERS = (
    "主体id",
    "身份不一致",
    "身份错误",
    "身份漂移",
    "人物不一致",
    "角色混淆",
    "换脸",
    "变脸",
    "人不对名",
)
_REFERENCE_BINDING_MARKERS = (
    "与参考图",
    "参考图中",
    "参考人物",
    "参考角色",
    "参考主体",
    "参考商品",
    "参考物体",
    "参考场景",
    "参考风格",
    "参考污染",
    "场景还原",
    "与参考不一致",
    "不符合参考",
)
_ENTITY_SPATIAL_MARKERS = (
    "实体数量",
    "人物数量",
    "重复人物",
    "双胞胎",
    "多胞胎",
    "多生出",
    "数量错误",
    "空间关系",
    "空间布局",
    "左右颠倒",
    "位置错误",
    "站位",
    "朝向",
    "构图",
    "透视",
    "比例异常",
)
_MOTION_MARKERS = (
    "动作异常",
    "动作错误",
    "动作顺序",
    "动作连续",
    "动作衔接",
    "物理规律",
    "姿势错乱",
    "无过渡",
    "镜头跳切",
    "跳切",
    "跳变",
    "瞬移",
    "凭空出现",
    "凭空消失",
    "步态",
    "重心",
    "滑移",
    "穿模",
    "穿插",
    "穿透",
    "粘连",
    "融合",
    "畸变",
    "结构崩坏",
    "动态结构保持",
)
_VISUAL_QUALITY_MARKERS = (
    "清晰度",
    "分辨率",
    "模糊",
    "色块",
    "闪烁",
    "掉帧",
    "噪点",
    "画质",
    "视觉伪影",
    "稳定性异常",
)


def _compact(value: Any) -> str:
    return "".join(
        character.casefold()
        for character in str(value or "")
        if character.isalnum() or "\u4e00" <= character <= "\u9fff"
    )


def _provenance_text(provenance: Any) -> str:
    if isinstance(provenance, Mapping):
        preferred = (
            "source",
            "source_name",
            "tool",
            "tool_name",
            "agent",
            "specialist",
            "method",
            "decision",
            "category",
        )
        parts = [str(provenance.get(key) or "") for key in preferred]
        return _compact(" ".join(parts))
    return _compact(provenance)


def _issue_text(issue: Mapping[str, Any]) -> str:
    return _compact(
        " ".join(
            str(issue.get(key) or "")
            for key in ("问题类型", "问题说明")
        )
    )


def _contains_any(text: str, markers: Iterable[str]) -> bool:
    return any(_compact(marker) in text for marker in markers)


def assign_issue(
    issue: Mapping[str, Any],
    provenance: Any = "",
) -> str:
    """Return the one most-specific classic check that owns ``issue``.

    The order below is intentional.  Specialized acoustic/text provenance and
    semantics are resolved before broad visual or instruction-compliance types;
    the latter is a true residual bucket rather than a competing classifier.
    """

    source = _provenance_text(provenance)
    text = _issue_text(issue)
    problem_type = _compact(issue.get("问题类型"))

    if _contains_any(source, ("avbench", "syncnet")):
        return AV_LIP_SYNC
    if _contains_any(text, _SYNC_MARKERS):
        return AV_LIP_SYNC

    if _contains_any(source, ("ocr", "subtitle", "textvisualverifier")):
        return SUBTITLE_TEXT_LOGO_WATERMARK
    if problem_type == _compact("文字质量问题") or _contains_any(
        text,
        _TEXT_MARKERS,
    ):
        return SUBTITLE_TEXT_LOGO_WATERMARK

    if _contains_any(text, _VOICE_MARKERS):
        return VOICE_CHARACTERISTICS
    if _contains_any(text, _AUDIO_ARTIFACT_MARKERS):
        return AUDIO_QUALITY_NOISE_ARTIFACTS
    if _contains_any(text, _DIALOGUE_MARKERS):
        return DIALOGUE_SPEAKER_BINDING

    if _contains_any(source, ("seedlite", "visualphysics")):
        # Seed-Lite has only logo/text and motion/structure responsibilities.
        # Text was resolved above, so its remaining issues are motion/physics.
        return MOTION_PHYSICS_CONTINUITY
    if _contains_any(text, (*_IDENTITY_MARKERS, *_REFERENCE_BINDING_MARKERS)):
        return REFERENCE_SUBJECT_IDENTITY
    if _contains_any(text, _ENTITY_SPATIAL_MARKERS):
        return ENTITY_COUNT_SPATIAL_COMPOSITION
    if _contains_any(text, _MOTION_MARKERS):
        return MOTION_PHYSICS_CONTINUITY

    if _contains_any(source, ("metadata", "ffprobe")):
        # Objective resolution and duration mismatches are prompt hard-constraint
        # violations.  They are not evidence of a perceptual quality artifact.
        return REMAINING_HARD_INSTRUCTION_COMPLIANCE
    if _contains_any(text, _VISUAL_QUALITY_MARKERS):
        return VISUAL_QUALITY_TEMPORAL_ARTIFACTS

    if problem_type == _compact("音频质量问题") or _contains_any(
        source,
        ("auralis", "geminiaudio"),
    ):
        # Any source-specific acoustic issue that lacks dialogue, voice, or sync
        # semantics is most safely owned by the generic audio-artifact check.
        return AUDIO_QUALITY_NOISE_ARTIFACTS

    return REMAINING_HARD_INSTRUCTION_COMPLIANCE


def candidate_issue(candidate: Any) -> tuple[Mapping[str, Any], Any]:
    """Normalize a raw issue or ``{"issue": ..., "provenance": ...}`` wrapper."""

    if isinstance(candidate, Mapping) and isinstance(
        candidate.get("issue"), Mapping
    ):
        return candidate["issue"], candidate.get("provenance", "")
    if isinstance(candidate, Mapping):
        return candidate, candidate.get("provenance", "")
    if (
        isinstance(candidate, tuple)
        and len(candidate) == 2
        and isinstance(candidate[0], Mapping)
    ):
        return candidate[0], candidate[1]
    raise TypeError("候选必须是七字段 issue、issue/provenance 包装对象或二元组")


def partition_candidates(
    candidates: Iterable[Any],
) -> dict[str, list[dict[str, Any]]]:
    """Partition candidates without duplicating any input candidate."""

    partitions = {name: [] for name in CLASSIC_CHECK_NAMES}
    for index, candidate in enumerate(candidates):
        issue, provenance = candidate_issue(candidate)
        owner = assign_issue(issue, provenance)
        partitions[owner].append(
            {
                "candidate_index": index,
                "issue": dict(issue),
                "provenance": provenance,
                "assigned_check": owner,
            }
        )
    return partitions


def issue_fingerprint(issue: Mapping[str, Any]) -> str:
    """Return a stable identity used only for exact candidate deduplication."""

    return json.dumps(dict(issue), ensure_ascii=False, sort_keys=True)
