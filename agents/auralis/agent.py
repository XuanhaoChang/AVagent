"""Orchestration for the Auralis audio-visual forensic agent."""

from __future__ import annotations

from dataclasses import replace
import re
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from agents.auralis.constrained_asr import (
    constrained_asr_issues,
    filter_contradicted_judge_issues,
)
from agents.auralis.schemas import (
    AuralisEvidence,
    AuralisInput,
    AuralisResult,
)
from agents.auralis.speaker_binding_resolver import resolve_speaker_binding
from agents.auralis.speaker_voiceprint import (
    SpeakerVoiceprintScorer,
    evaluate_prompt_voiceprints,
)
from tools.media.ffmpeg import extract_audio_wav, probe_video
from tools.speech_subtitle_alignment.tool import check_speech_subtitle_alignment
from tools.speech_subtitle_alignment.schemas import AlignmentResult
from tools.speech_transcription.schemas import SpeechTranscript
from tools.subtitle_extraction.schemas import SubtitleSegment, SubtitleTrack
from tools.subtitle_extraction.tool import (
    extract_subtitles as run_subtitle_extraction,
    subtitle_evidence_for_judge,
)


Judge = Callable[[AuralisInput, AuralisEvidence], Sequence[Mapping[str, Any]]]
PromptCandidateScorer = Callable[
    [Path, str, SpeechTranscript],
    Mapping[str, Any],
]
PromptAwareTranscriber = Callable[[Path, str], SpeechTranscript]


_ISSUE_TIME_RANGE = re.compile(
    r"^\s*(?P<start>\d+(?:\.\d+)?)s?\s*-\s*"
    r"(?P<end>\d+(?:\.\d+)?)s?\s*$"
)
_DIRECT_ACOUSTIC_GENDER_MARKERS = (
    "实际听到",
    "听到明显",
    "听感明显",
    "明显为",
    "声音明显",
    "音色明显",
)
_SAME_VOICE_MARKERS = (
    "同一声纹",
    "同一匿名声纹",
    "同一个匿名声纹",
    "同一个声纹",
    "单一声纹",
    "相同声纹",
    "共用单一声纹",
    "共用声纹",
    "共用声音",
    "共用声线",
    "声线完全一致",
    "声纹标签未发生变化",
    "speaker标签未发生变化",
    "spk标签未发生变化",
    "由同一人发出",
)
_SPEAKER_BINDING_MARKERS = (
    "spk",
    "speaker",
    "声纹",
    "说话人绑定",
    "说话人归属",
    "台词归属",
    "配音主体",
    "声音绑定",
    "声线绑定",
    "角色绑定",
    "绑定错误",
    "错误绑定",
    "共用声音",
    "共用声线",
)


def _no_judge(_agent_input: AuralisInput, _evidence: AuralisEvidence):
    return ()


def deterministic_alignment_issues(
    alignment: AlignmentResult,
) -> tuple[Mapping[str, Any], ...]:
    """Convert only high-precision deterministic ASR/OCR diffs into issues."""

    issues: list[Mapping[str, Any]] = []
    for item in alignment.issues:
        if (
            item.method
            not in {"localized_asr_ocr", "numeric_timeline_alignment"}
            or item.confidence != "high"
        ):
            continue
        issues.append(
            {
                "可定位性": "否",
                "置信度": "高",
                "问题说明": (
                    "预期烧录字幕应与实际语音一致；"
                    f"ASR 实际语音为“{item.speech_text}”，"
                    f"OCR 实际字幕为“{item.subtitle_text}”，"
                    f"{item.difference}；发生在 "
                    f"{item.start_sec:.2f}s - {item.end_sec:.2f}s。"
                ),
                "问题类型": "文字质量问题",
                "时间区间": f"{item.start_sec:.2f}s - {item.end_sec:.2f}s",
                "关键帧秒": "",
                "BBox": "",
            }
        )
    return tuple(issues)


def _time_range(issue: Mapping[str, Any]) -> tuple[float, float] | None:
    match = _ISSUE_TIME_RANGE.fullmatch(str(issue.get("时间区间") or ""))
    if match is None:
        return None
    start = float(match.group("start"))
    end = float(match.group("end"))
    if end <= start:
        return None
    return start, end


def _overlaps(start: float, end: float, other_start: float, other_end: float) -> bool:
    return min(end, other_end) > max(start, other_start)


def filter_unverified_ocr_judge_issues(
    trusted_subtitles: SubtitleTrack,
    alignment: AlignmentResult,
    rejected_singletons: Sequence[SubtitleSegment],
    issues: Sequence[Mapping[str, Any]],
) -> tuple[tuple[Mapping[str, Any], ...], tuple[Mapping[str, Any], ...]]:
    """Veto text defects whose only local support is singleton-frame OCR."""

    if not rejected_singletons:
        return tuple(issues), ()
    kept: list[Mapping[str, Any]] = []
    vetoed: list[Mapping[str, Any]] = []
    for issue in issues:
        interval = _time_range(issue)
        if issue.get("问题类型") != "文字质量问题" or interval is None:
            kept.append(issue)
            continue
        start, end = interval
        rejected = [
            segment
            for segment in rejected_singletons
            if _overlaps(start, end, segment.start_sec, segment.end_sec)
        ]
        if not rejected:
            kept.append(issue)
            continue
        has_trusted_ocr = any(
            _overlaps(start, end, segment.start_sec, segment.end_sec)
            for segment in trusted_subtitles.segments
        )
        has_alignment_support = any(
            _overlaps(start, end, item.start_sec, item.end_sec)
            for item in alignment.issues
        )
        if has_trusted_ocr or has_alignment_support:
            kept.append(issue)
            continue
        vetoed.append(
            {
                "issue": dict(issue),
                "reason": "only_unverified_single_frame_single_character_ocr",
                "ocr_candidates": [
                    {
                        "start_sec": segment.start_sec,
                        "end_sec": segment.end_sec,
                        "text": segment.text,
                        "bbox": list(segment.bbox),
                        "confidence": segment.confidence,
                        "source": segment.source,
                    }
                    for segment in rejected
                ],
            }
        )
    return tuple(kept), tuple(vetoed)


def _gender_mismatch_issues_from_combined_binding_claim(
    transcript: SpeechTranscript,
    issue: Mapping[str, Any],
    alignments: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    """Preserve independently supported gender claims from a mixed CAM++ claim."""

    description = str(issue.get("问题说明") or "")
    if not any(marker in description for marker in ("男声", "女声", "童声")):
        return ()
    if not any(marker in description for marker in _DIRECT_ACOUSTIC_GENDER_MARKERS):
        # A CAM++ cluster assignment cannot by itself establish acoustic sex.
        # In particular, phrases such as "同一 speaker，因此绑定为男声"
        # describe an inference from clustering rather than an audible trait.
        return ()
    plan = transcript.metadata.get("prompt_speech_plan", {})
    role_references = (
        plan.get("role_reference_images", {}) if isinstance(plan, Mapping) else {}
    )
    if not isinstance(role_references, Mapping):
        return ()

    global_actual = ""
    if "均明显" in description or "都明显" in description:
        if "男声" in description:
            global_actual = "成年男声"
        elif "女声" in description:
            global_actual = "成年女声"
        elif "童声" in description:
            global_actual = "童声"

    preserved: list[Mapping[str, Any]] = []
    for alignment in alignments:
        role = str(alignment.get("role") or "")
        references = role_references.get(role, ())
        if role not in description or not isinstance(references, (list, tuple)) or not references:
            continue
        role_index = description.find(role)
        role_context = description[max(0, role_index - 24) : role_index + 120]
        if "女性" in role_context or "女角色" in role_context:
            expected = "女性角色"
            conflicting_actual = ("成年男声", "男声")
        elif "男性" in role_context or "男角色" in role_context:
            expected = "男性角色"
            conflicting_actual = ("成年女声", "女声")
        elif "儿童" in role_context or "孩子" in role_context:
            expected = "儿童角色"
            conflicting_actual = ("成年男声", "成年女声")
        else:
            continue

        actual = global_actual
        if not actual:
            if "男声" in role_context:
                actual = "成年男声"
            elif "女声" in role_context:
                actual = "成年女声"
            elif "童声" in role_context:
                actual = "童声"
        if actual not in conflicting_actual:
            continue

        matched_segments = [
            segment
            for segment in alignment.get("matched_segments", ())
            if isinstance(segment, Mapping)
            and segment.get("start_sec") is not None
            and segment.get("end_sec") is not None
        ]
        if not matched_segments:
            continue
        start_sec = min(float(segment["start_sec"]) for segment in matched_segments)
        end_sec = max(float(segment["end_sec"]) for segment in matched_segments)
        # Short exclamations are too fragile for stable acoustic-sex judgments.
        if end_sec - start_sec < 0.80:
            continue
        dialogue = str(alignment.get("dialogue_text") or "")
        reference_label = ",".join(f"图{int(index)}" for index in references)
        preserved.append(
            {
                "可定位性": "否",
                "置信度": str(issue.get("置信度") or "中"),
                "问题说明": (
                    f"预期{role}依据角色参考{reference_label}呈现为{expected}，"
                    "其配音应具有相符的明显性别声学特征；实际在 "
                    f"{start_sec:.2f}s - {end_sec:.2f}s 的已锚定台词"
                    f"“{dialogue}”中听到明显{actual}，存在角色视觉性别与配音性别冲突。"
                ),
                "问题类型": "音频质量问题",
                "时间区间": f"{start_sec:.2f}s - {end_sec:.2f}s",
                "关键帧秒": "",
                "BBox": "",
            }
        )
    return tuple(preserved)


def filter_acoustically_contradicted_binding_issues(
    transcript: SpeechTranscript,
    issues: Sequence[Mapping[str, Any]],
) -> tuple[tuple[Mapping[str, Any], ...], tuple[Mapping[str, Any], ...]]:
    """Admit speaker-binding claims only when structured evidence supports them.

    Gemini proposes candidates; it does not decide whether a CAM++ claim is a
    fact.  A claim must have prompt-anchored role turns and actionable acoustic
    evidence.  In the production path, directional claims must match the local
    deterministic resolver and shared-voice claims must match direct role-clip
    CAM++ verification, rather than only a common anonymous label.
    """

    metadata = transcript.metadata
    binding = metadata.get("speaker_binding_evidence", {})
    if not isinstance(binding, Mapping):
        binding = {}
    plan = metadata.get("prompt_speech_plan", {})
    if not isinstance(plan, Mapping):
        plan = {}
    clustering = metadata.get("clustering", {})
    if not isinstance(clustering, Mapping):
        clustering = {}
    resolution = metadata.get("speaker_binding_resolution", {})
    if not isinstance(resolution, Mapping):
        resolution = {}
    alignments = [
        item
        for item in binding.get("prompt_turn_alignment", ())
        if isinstance(item, Mapping)
        and item.get("status") == "anchored"
        and str(item.get("role") or "")
        and item.get("actual_speakers")
    ]
    role_to_speakers: dict[str, set[Any]] = {}
    for alignment in alignments:
        role = str(alignment.get("role") or "")
        role_to_speakers.setdefault(role, set()).update(
            alignment.get("actual_speakers", ())
        )
    speaker_to_roles: dict[Any, set[str]] = {}
    for role, speakers in role_to_speakers.items():
        for speaker in speakers:
            speaker_to_roles.setdefault(speaker, set()).add(role)

    speaker_turn_count = int(clustering.get("speaker_turn_count", 0) or 0)
    speech_status = str(metadata.get("speech_evidence_status") or "")
    prompt_scope = str(
        binding.get("prompt_scope") or plan.get("scope") or ""
    )

    def preserved_gender_issues(
        issue: Mapping[str, Any],
        relevant: Sequence[Mapping[str, Any]] = alignments,
    ) -> tuple[Mapping[str, Any], ...]:
        return _gender_mismatch_issues_from_combined_binding_claim(
            transcript,
            issue,
            relevant,
        )

    def same_speaker_confirmation(speaker: str) -> tuple[bool, Mapping[str, Any]]:
        algorithm_status = str(clustering.get("cluster_algorithm_status") or "")
        quality_by_speaker = clustering.get("cluster_similarity", {})
        quality = (
            quality_by_speaker.get(str(speaker), {})
            if isinstance(quality_by_speaker, Mapping)
            else {}
        )
        if not isinstance(quality, Mapping):
            quality = {}
        threshold = float(clustering.get("similarity_threshold", 0.78) or 0.78)
        try:
            window_count = int(quality.get("window_count", 0) or 0)
            pair_count = int(quality.get("within_pair_count", 0) or 0)
            minimum = float(quality.get("within_similarity_min"))
            mean = float(quality.get("within_similarity_mean"))
        except (TypeError, ValueError):
            window_count = 0
            pair_count = 0
            minimum = -1.0
            mean = -1.0
        confirmed = (
            algorithm_status == "spectral_clustered"
            and window_count >= 2
            and pair_count >= 1
            and mean >= threshold
            and minimum >= threshold - 0.12
        )
        return confirmed, {
            "speaker": speaker,
            "cluster_algorithm_status": algorithm_status or "missing",
            "similarity_threshold": threshold,
            "cluster_quality": dict(quality),
        }

    kept: list[Mapping[str, Any]] = []
    vetoed: list[Mapping[str, Any]] = []
    for issue in issues:
        description = str(issue.get("问题说明") or "")
        if not (
            issue.get("问题类型") == "音频质量问题"
            and any(marker in description.casefold() for marker in _SPEAKER_BINDING_MARKERS)
        ):
            kept.append(issue)
            continue
        same_voice_claim = any(
            marker in description for marker in _SAME_VOICE_MARKERS
        )
        if (
            any(marker in description for marker in ("男声", "女声", "童声"))
            and not any(
                marker in description
                for marker in _DIRECT_ACOUSTIC_GENDER_MARKERS
            )
        ):
            vetoed.append(
                {
                    "issue": dict(issue),
                    "reason": "speaker_cluster_cannot_prove_acoustic_gender",
                    "preserved_gender_issues": [],
                }
            )
            continue
        mentioned = [
            item
            for item in alignments
            if str(item.get("role") or "")
            and str(item.get("role")) in description
        ]
        mentioned_roles = {str(item.get("role")) for item in mentioned}
        mentioned_single_speakers = {
            item["actual_speakers"][0]
            for item in mentioned
            if len(item.get("actual_speakers", ())) == 1
        }

        def reject(reason: str, **details: Any) -> None:
            preserved = preserved_gender_issues(issue, mentioned or alignments)
            kept.extend(preserved)
            vetoed.append(
                {
                    "issue": dict(issue),
                    "reason": reason,
                    **details,
                    "preserved_gender_issues": [dict(item) for item in preserved],
                }
            )

        # Current Agent-D computes a deterministic resolution before invoking
        # Gemini.  The model may phrase a supported fact, but it cannot create
        # one.  Keep the legacy checks below only for callers using older
        # transcripts that do not yet carry resolver output.
        if resolution.get("version") == 2:
            if same_voice_claim:
                supported_pairs = {
                    tuple(sorted(str(role) for role in conflict.get("roles", ())))
                    for conflict in resolution.get("shared_voice_conflicts", ())
                    if isinstance(conflict, Mapping)
                    and len(conflict.get("roles", ())) == 2
                }
                claimed_roles = sorted(mentioned_roles)
                required_pairs = {
                    (claimed_roles[left], claimed_roles[right])
                    for left in range(len(claimed_roles))
                    for right in range(left + 1, len(claimed_roles))
                }
                if not required_pairs or not required_pairs.issubset(
                    supported_pairs
                ):
                    reject(
                        "same_voice_claim_lacks_direct_role_pair_confirmation",
                        claimed_role_pairs=sorted(required_pairs),
                        supported_role_pairs=sorted(supported_pairs),
                    )
                    continue
                kept.append(issue)
                continue
            directional_conflicts = []
            for conflict in resolution.get("directional_conflicts", ()):
                if not isinstance(conflict, Mapping):
                    continue
                conflict_roles = {
                    str(conflict.get("expected_role") or ""),
                    str(conflict.get("actual_role") or ""),
                } - {""}
                # A one-role phrase such as "李莲的台词绑定错误" may omit
                # the actual speaker and is still compatible with the local
                # result.  Once Gemini names two or more roles, every named
                # role must belong to the exact resolved pair; otherwise a
                # correct expected role could accidentally admit a fabricated
                # actual role.
                if (
                    len(mentioned_roles) == 1
                    and mentioned_roles & conflict_roles
                ) or (
                    len(mentioned_roles) >= 2
                    and mentioned_roles.issubset(conflict_roles)
                ):
                    directional_conflicts.append(conflict)
            if not directional_conflicts:
                reject(
                    "directional_binding_claim_lacks_resolver_confirmation",
                    resolver_decision=str(
                        resolution.get("decision") or "underdetermined"
                    ),
                )
                continue
            kept.append(issue)
            continue

        if not transcript.segments:
            reject("speaker_binding_claim_has_no_valid_speech_segments")
            continue
        if speech_status == "bgm_only":
            reject("speaker_binding_claim_uses_bgm_only_asr")
            continue
        if binding.get("status") != "fine_grained_turns":
            reject(
                "speaker_binding_evidence_not_actionable",
                binding_status=str(binding.get("status") or "missing"),
            )
            continue
        if prompt_scope == "none":
            reject("speaker_binding_claim_has_no_prompt_role_scope")
            continue
        if not alignments:
            reject("speaker_binding_claim_has_no_anchored_prompt_turns")
            continue
        if not mentioned_roles:
            reject(
                "speaker_binding_claim_mentions_no_anchored_role",
                anchored_roles=sorted(role_to_speakers),
            )
            continue

        split_roles = {
            role for role, speakers in role_to_speakers.items() if len(speakers) > 1
        }
        split_support = sorted(mentioned_roles & split_roles)
        shared_support = [
            {"speaker": speaker, "roles": sorted(roles & mentioned_roles)}
            for speaker, roles in speaker_to_roles.items()
            if len(roles & mentioned_roles) >= 2
        ]

        if same_voice_claim and len(mentioned_single_speakers) >= 2:
            reject(
                "fine_grained_campp_turns_contradict_same_voice_claim",
                role_speakers={
                    role: next(iter(role_to_speakers[role]))
                    for role in sorted(mentioned_roles)
                    if len(role_to_speakers.get(role, ())) == 1
                },
            )
            continue
        if not split_support and not shared_support:
            reject(
                "speaker_binding_claim_has_no_structured_role_conflict",
                mentioned_anchored_roles=sorted(mentioned_roles),
            )
            continue

        requires_same_speaker_confirmation = same_voice_claim or not split_support
        if requires_same_speaker_confirmation:
            if speaker_turn_count < 2:
                reject(
                    "same_voice_claim_lacks_independent_anchored_campp_turns",
                    mentioned_anchored_roles=sorted(mentioned_roles),
                    speaker_turn_count=speaker_turn_count,
                )
                continue
            if speech_status == "speech_with_bgm":
                reject(
                    "same_voice_claim_requires_bgm_robust_acoustic_confirmation",
                    mentioned_anchored_roles=sorted(mentioned_roles),
                )
                continue
            confirmations = []
            for candidate in shared_support:
                confirmed, detail = same_speaker_confirmation(candidate["speaker"])
                confirmations.append({**candidate, **detail, "confirmed": confirmed})
            if not any(item["confirmed"] for item in confirmations):
                reject(
                    "same_voice_claim_lacks_pairwise_campp_confirmation",
                    campp_confirmations=confirmations,
                )
                continue
        kept.append(issue)
    return tuple(kept), tuple(vetoed)


def filter_unanchored_gender_voice_issues(
    transcript: SpeechTranscript,
    issues: Sequence[Mapping[str, Any]],
) -> tuple[tuple[Mapping[str, Any], ...], tuple[Mapping[str, Any], ...]]:
    """Require every acoustic-gender claim to have a role-labelled ASR clip."""

    metadata = transcript.metadata
    plan = metadata.get("prompt_speech_plan", {})
    binding = metadata.get("speaker_binding_evidence", {})
    role_references = (
        plan.get("role_reference_images", {}) if isinstance(plan, Mapping) else {}
    )
    alignments = (
        binding.get("prompt_turn_alignment", ())
        if isinstance(binding, Mapping)
        else ()
    )
    eligible_checks: list[dict[str, Any]] = []
    if isinstance(role_references, Mapping):
        for alignment in alignments:
            if not isinstance(alignment, Mapping) or alignment.get("status") != "anchored":
                continue
            role = str(alignment.get("role") or "")
            references = role_references.get(role, ())
            if not role or not isinstance(references, (list, tuple)) or not references:
                continue
            matched_segments = [
                segment
                for segment in alignment.get("matched_segments", ())
                if isinstance(segment, Mapping)
                and segment.get("start_sec") is not None
                and segment.get("end_sec") is not None
            ]
            if not matched_segments:
                continue
            start_sec = min(float(segment["start_sec"]) for segment in matched_segments)
            end_sec = max(float(segment["end_sec"]) for segment in matched_segments)
            if end_sec - start_sec < 0.80:
                continue
            eligible_checks.append(
                {
                    "role": role,
                    "reference_image_indices": [int(index) for index in references],
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "dialogue_text": str(alignment.get("dialogue_text") or ""),
                }
            )

    gender_markers = ("男声", "女声", "童声", "性别声线", "性别音色")
    kept: list[Mapping[str, Any]] = []
    vetoed: list[Mapping[str, Any]] = []
    for issue in issues:
        description = str(issue.get("问题说明") or "")
        if not (
            issue.get("问题类型") == "音频质量问题"
            and any(marker in description for marker in gender_markers)
        ):
            kept.append(issue)
            continue
        interval = _time_range(issue)
        supporting_checks = [
            check
            for check in eligible_checks
            if check["role"] in description
            and (
                interval is None
                or _overlaps(
                    interval[0],
                    interval[1],
                    float(check["start_sec"]),
                    float(check["end_sec"]),
                )
            )
        ]
        if supporting_checks:
            kept.append(issue)
            continue
        vetoed.append(
            {
                "issue": dict(issue),
                "reason": "gender_claim_lacks_role_labelled_acoustic_check",
                "eligible_role_checks": eligible_checks,
            }
        )
    return tuple(kept), tuple(vetoed)


def filter_single_asr_negative_claims(
    transcript: SpeechTranscript,
    issues: Sequence[Mapping[str, Any]],
) -> tuple[tuple[Mapping[str, Any], ...], tuple[Mapping[str, Any], ...]]:
    """Veto absence claims and short unanchored speech without corroboration.

    A positive ASR match can establish what was spoken. A single ASR miss cannot
    establish that an expected line was absent, and one short unassigned token
    is too fragile to establish an extra line in a closed script.
    """

    metadata = transcript.metadata
    binding = metadata.get("speaker_binding_evidence", {})
    unassigned = (
        binding.get("unassigned_segments", ())
        if isinstance(binding, Mapping)
        else ()
    )
    short_unanchored: list[Mapping[str, Any]] = []
    for segment in unassigned:
        if not isinstance(segment, Mapping) or not segment.get(
            "closed_script_candidate"
        ):
            continue
        try:
            duration = float(segment["end_sec"]) - float(segment["start_sec"])
        except (KeyError, TypeError, ValueError):
            continue
        text = str(segment.get("text") or "").strip()
        latin_words = re.findall(r"[A-Za-z]+(?:['’][A-Za-z]+)*", text)
        compact = "".join(character for character in text if character.isalnum())
        short_lexical_content = (
            bool(latin_words) and len(latin_words) <= 1
        ) or (not latin_words and len(compact) <= 2)
        if duration <= 1.0 and short_lexical_content:
            short_unanchored.append(segment)

    missing_markers = (
        "未检测到",
        "并未检测到",
        "没有检测到",
        "未说出",
        "没有说出",
        "未出现",
        "台词缺失",
        "对白缺失",
    )
    extra_speech_markers = (
        "多余",
        "额外",
        "未经授权",
        "无台词",
        "不应有",
        "不应出现",
    )
    kept: list[Mapping[str, Any]] = []
    vetoed: list[Mapping[str, Any]] = []
    for issue in issues:
        description = str(issue.get("问题说明") or "")
        if (
            issue.get("问题类型") == "音频质量问题"
            and any(marker in description for marker in missing_markers)
            and any(marker in description for marker in ("台词", "对白", "语音"))
        ):
            vetoed.append(
                {
                    "issue": dict(issue),
                    "reason": "single_asr_miss_cannot_prove_dialogue_absence",
                }
            )
            continue

        interval = _time_range(issue)
        matched_short = []
        if interval is not None:
            start, end = interval
            matched_short = [
                segment
                for segment in short_unanchored
                if _overlaps(
                    start,
                    end,
                    float(segment["start_sec"]),
                    float(segment["end_sec"]),
                )
                and str(segment.get("text") or "").strip().casefold()
                in description.casefold()
            ]
        if matched_short and (
            issue.get("问题类型") == "文字质量问题"
            or any(marker in description for marker in extra_speech_markers)
        ):
            vetoed.append(
                {
                    "issue": dict(issue),
                    "reason": "short_unanchored_asr_token_is_not_confirmed_speech",
                    "asr_candidates": [dict(segment) for segment in matched_short],
                }
            )
            continue
        kept.append(issue)
    return tuple(kept), tuple(vetoed)


class AuralisAgent:
    """Run every audio evidence tool and then ask one judge to verify findings."""

    def __init__(
        self,
        *,
        probe_video: Callable[[Path], Mapping[str, Any]] = probe_video,
        extract_audio: Callable[[Path, Path], Any] = extract_audio_wav,
        transcribe_speech: Callable[[Path], Any] | None = None,
        transcribe_speech_with_prompt: PromptAwareTranscriber | None = None,
        extract_subtitles: Callable[[Path], Any] | None = None,
        align_speech_subtitles: Callable[[Any, Any], Any] = (
            check_speech_subtitle_alignment
        ),
        score_prompt_candidates: PromptCandidateScorer | None = None,
        score_speaker_voiceprints: SpeakerVoiceprintScorer | None = None,
        judge: Judge | None = None,
        local_only: bool = False,
    ) -> None:
        if judge is None:
            if not local_only:
                raise ValueError(
                    "AuralisAgent 必须提供 judge；仅提取本地证据时显式设置 "
                    "local_only=True。"
                )
            judge = _no_judge
        if transcribe_speech is None and transcribe_speech_with_prompt is None:
            from agents.auralis.constrained_asr import (
                evaluate_prompt_constrained_asr,
            )
            from tools.speech_transcription.backends.sensevoice import (
                SenseVoiceBackend,
            )

            asr_backend = SenseVoiceBackend()
            transcribe_speech_with_prompt = lambda path, prompt: (
                asr_backend.transcribe(path, user_prompt=prompt)
            )
            if score_prompt_candidates is None:
                score_prompt_candidates = lambda path, prompt, transcript: (
                    evaluate_prompt_constrained_asr(
                        path,
                        prompt,
                        transcript,
                        scorer=asr_backend.score_candidates,
                    )
                )
            if score_speaker_voiceprints is None:
                score_speaker_voiceprints = asr_backend.score_speaker_segments
        if extract_subtitles is None:
            from tools.subtitle_extraction.backends.rapidocr import (
                RapidOCRBackend,
            )

            ocr_backend = RapidOCRBackend()
            extract_subtitles = lambda path: run_subtitle_extraction(
                path,
                backend=ocr_backend,
            )
        self._probe_video = probe_video
        self._extract_audio = extract_audio
        self._transcribe_speech = transcribe_speech
        self._transcribe_speech_with_prompt = transcribe_speech_with_prompt
        self._extract_subtitles = extract_subtitles
        self._align_speech_subtitles = align_speech_subtitles
        self._score_prompt_candidates = score_prompt_candidates
        self._score_speaker_voiceprints = score_speaker_voiceprints
        self._judge = judge

    def analyze(self, agent_input: AuralisInput) -> AuralisResult:
        metadata = self._probe_video(agent_input.video_path)
        if not bool(metadata.get("has_audio")):
            return AuralisResult(
                status="no_audio",
                diagnostics={"reason": "ffprobe did not detect an audio stream"},
            )

        with tempfile.TemporaryDirectory(prefix="auralis_") as temp_dir:
            audio_path = Path(temp_dir) / "audio.wav"
            extracted = self._extract_audio(agent_input.video_path, audio_path)
            if isinstance(extracted, Path):
                audio_path = extracted
            if self._transcribe_speech_with_prompt is not None:
                transcript = self._transcribe_speech_with_prompt(
                    audio_path,
                    agent_input.user_prompt,
                )
            elif self._transcribe_speech is not None:
                transcript = self._transcribe_speech(audio_path)
            else:  # pragma: no cover - constructor always configures one path.
                raise RuntimeError("Auralis ASR transcriber 未配置")
            source_transcript = transcript
            voiceprint_evidence = evaluate_prompt_voiceprints(
                audio_path,
                transcript,
                scorer=self._score_speaker_voiceprints,
            )
            transcript_metadata = dict(transcript.metadata)
            binding = transcript_metadata.get("speaker_binding_evidence", {})
            binding = dict(binding) if isinstance(binding, Mapping) else {}
            binding["voiceprint_verification"] = voiceprint_evidence
            transcript_metadata["speaker_binding_evidence"] = binding
            prompt_plan = transcript_metadata.get("prompt_speech_plan", {})
            prompt_plan = (
                prompt_plan if isinstance(prompt_plan, Mapping) else {}
            )
            clustering = transcript_metadata.get("clustering", {})
            clustering = (
                clustering if isinstance(clustering, Mapping) else {}
            )
            speaker_binding_resolution = resolve_speaker_binding(
                prompt_plan,
                tuple(
                    item
                    for item in binding.get("prompt_turn_alignment", ())
                    if isinstance(item, Mapping)
                ),
                binding_status=str(binding.get("status") or ""),
                speech_evidence_status=str(
                    transcript_metadata.get("speech_evidence_status") or ""
                ),
                clustering=clustering,
                voiceprint_evidence=voiceprint_evidence,
            )
            transcript_metadata["speaker_binding_resolution"] = (
                speaker_binding_resolution
            )
            transcript = replace(transcript, metadata=transcript_metadata)
            constrained_asr: Mapping[str, Any] = {}
            if self._score_prompt_candidates is not None:
                try:
                    constrained_asr = self._score_prompt_candidates(
                        audio_path,
                        agent_input.user_prompt,
                        source_transcript,
                    )
                except Exception as exc:
                    # Preserve the existing ASR/OCR/Gemini path and expose a
                    # machine-readable failure instead of losing the row.
                    constrained_asr = {
                        "status": "scoring_failed",
                        "reason": "prompt_candidate_pipeline_exception",
                        "scoring_error": f"{type(exc).__name__}: {exc}",
                        "candidate_scores": [],
                    }
            subtitles = self._extract_subtitles(agent_input.video_path)
            judge_subtitles, rejected_ocr_singletons = (
                subtitle_evidence_for_judge(subtitles)
            )
            alignment = self._align_speech_subtitles(
                transcript,
                judge_subtitles,
            )
            evidence = AuralisEvidence(
                media_metadata=metadata,
                transcript=transcript,
                subtitles=subtitles,
                alignment=alignment,
                constrained_asr=constrained_asr,
            )
            judge_evidence = AuralisEvidence(
                media_metadata=metadata,
                transcript=transcript,
                subtitles=judge_subtitles,
                alignment=alignment,
                constrained_asr=constrained_asr,
            )
            judged_issues, vetoed_pronunciation_issues = filter_contradicted_judge_issues(
                constrained_asr,
                tuple(self._judge(agent_input, judge_evidence)),
            )
            judged_issues, vetoed_binding_issues = (
                filter_acoustically_contradicted_binding_issues(
                    transcript,
                    judged_issues,
                )
            )
            judged_issues, vetoed_gender_voice_issues = (
                filter_unanchored_gender_voice_issues(
                    transcript,
                    judged_issues,
                )
            )
            judged_issues, vetoed_negative_asr_issues = (
                filter_single_asr_negative_claims(
                    transcript,
                    judged_issues,
                )
            )
            judged_issues, vetoed_ocr_issues = filter_unverified_ocr_judge_issues(
                judge_subtitles,
                alignment,
                rejected_ocr_singletons,
                judged_issues,
            )
            local_asr_issues = constrained_asr_issues(constrained_asr)
            local_alignment_issues = deterministic_alignment_issues(alignment)
            local_binding_issues = tuple(
                item
                for item in speaker_binding_resolution.get("issues", ())
                if isinstance(item, Mapping)
            )
            deterministic_issues = (
                local_asr_issues
                + local_alignment_issues
                + local_binding_issues
            )
            issues = deterministic_issues + judged_issues
        diagnostics: Mapping[str, Any] = {}
        diagnostics_payload: dict[str, Any] = {}
        diagnostics_payload["speaker_binding_resolution"] = (
            speaker_binding_resolution
        )
        if vetoed_pronunciation_issues:
            diagnostics_payload["constrained_asr_vetoed_judge_issues"] = (
                vetoed_pronunciation_issues
            )
        if vetoed_binding_issues:
            diagnostics_payload["campp_vetoed_binding_issues"] = (
                vetoed_binding_issues
            )
        if vetoed_gender_voice_issues:
            diagnostics_payload["acoustic_gender_vetoed_issues"] = (
                vetoed_gender_voice_issues
            )
        if vetoed_negative_asr_issues:
            diagnostics_payload["asr_vetoed_negative_claims"] = (
                vetoed_negative_asr_issues
            )
        if rejected_ocr_singletons:
            diagnostics_payload["ocr_unverified_singletons"] = [
                {
                    "start_sec": segment.start_sec,
                    "end_sec": segment.end_sec,
                    "text": segment.text,
                    "bbox": list(segment.bbox),
                    "confidence": segment.confidence,
                    "source": segment.source,
                }
                for segment in rejected_ocr_singletons
            ]
        if vetoed_ocr_issues:
            diagnostics_payload["ocr_vetoed_judge_issues"] = vetoed_ocr_issues
        if diagnostics_payload:
            diagnostics = diagnostics_payload
        return AuralisResult(
            status="ok",
            issues=issues,
            deterministic_issues=deterministic_issues,
            evidence=evidence,
            diagnostics=diagnostics,
        )
