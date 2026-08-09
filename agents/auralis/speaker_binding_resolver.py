"""Deterministic prompt-role to anonymous-speaker conflict resolution.

The resolver handles directional binding mistakes identifiable from repeated
prompt-anchored turns. A hard CAM++ label changing inside one role, or being
shared by several roles, is only a candidate. It becomes directional evidence
when the expected role has a stable speaker prototype from at least two
independent turns and the anomalous label is itself the stable prototype of
another role.

Sparse single-turn role pairs can also be supported, but only when direct
role-clip CAM++ verification independently exceeds a strict same-speaker
threshold. A shared cluster alone, low-quality clips, partial pair coverage,
or role-free prompts remain ``underdetermined``.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence


MIN_SEGMENT_DURATION_SEC = 0.20
MIN_CONFLICT_DURATION_SEC = 0.35
MIN_DIALOGUE_MATCH_SCORE = 0.70
MIN_DIALOGUE_OBSERVED_PRECISION = 0.75
MIN_CANONICAL_TURNS = 2
MIN_CANONICAL_DURATION_SEC = 1.00
MIN_CANONICAL_SHARE = 0.68
MIN_CANONICAL_MARGIN = 0.70
MAX_MERGE_GAP_SEC = 0.35


def _speaker_key(value: Any) -> str:
    return str(value)


def _source_turn(
    prompt_speech_plan: Mapping[str, Any],
    role: str,
    dialogue_text: str,
) -> Mapping[str, Any]:
    candidates = [
        item
        for item in prompt_speech_plan.get("turns", ())
        if isinstance(item, Mapping)
        and str(item.get("role") or "") == role
        and str(item.get("dialogue_text") or "") == dialogue_text
    ]
    return candidates[0] if candidates else {}


def _anchored_segment_records(
    prompt_speech_plan: Mapping[str, Any],
    prompt_turn_alignment: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for turn_index, alignment in enumerate(prompt_turn_alignment):
        if not isinstance(alignment, Mapping):
            continue
        if (
            alignment.get("status") != "anchored"
            or alignment.get("anchor_method") != "dialogue_text_similarity"
        ):
            continue
        role = str(alignment.get("role") or "")
        dialogue_text = str(alignment.get("dialogue_text") or "")
        if not role or not dialogue_text:
            continue
        source = _source_turn(prompt_speech_plan, role, dialogue_text)
        for segment in alignment.get("matched_segments", ()):
            if not isinstance(segment, Mapping) or segment.get("speaker") is None:
                continue
            try:
                start_sec = float(segment["start_sec"])
                end_sec = float(segment["end_sec"])
                match_score = float(segment.get("dialogue_match_score", 1.0))
                observed_precision = float(
                    segment.get("dialogue_observed_precision", 1.0)
                )
            except (KeyError, TypeError, ValueError):
                continue
            duration_sec = end_sec - start_sec
            if (
                duration_sec < MIN_SEGMENT_DURATION_SEC
                or match_score < MIN_DIALOGUE_MATCH_SCORE
                or observed_precision < MIN_DIALOGUE_OBSERVED_PRECISION
            ):
                continue
            records.append(
                {
                    "turn_index": turn_index,
                    "role": role,
                    "dialogue_text": dialogue_text,
                    "prompt_start": source.get("prompt_start"),
                    "prompt_end": source.get("prompt_end"),
                    "prompt_source_text": str(
                        source.get("prompt_source_text") or dialogue_text
                    ),
                    "speaker": segment.get("speaker"),
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "duration_sec": duration_sec,
                    "text": str(segment.get("text") or ""),
                    "dialogue_match_score": match_score,
                    "dialogue_observed_precision": observed_precision,
                    "weight": duration_sec * match_score,
                }
            )
    return records


def _role_speaker_support(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, list[Mapping[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in records:
        grouped[str(record["role"])][_speaker_key(record["speaker"])].append(
            record
        )

    support: dict[str, dict[str, dict[str, Any]]] = {}
    for role, speakers in grouped.items():
        support[role] = {}
        for speaker, items in speakers.items():
            support[role][speaker] = {
                "speaker": items[0]["speaker"],
                "weighted_duration": round(
                    sum(float(item["weight"]) for item in items), 6
                ),
                "duration_sec": round(
                    sum(float(item["duration_sec"]) for item in items), 6
                ),
                "turn_count": len({int(item["turn_index"]) for item in items}),
                "segment_count": len(items),
            }
    return support


def _canonical_role_candidates(
    support: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    provisional: dict[str, dict[str, Any]] = {}
    rejected: dict[str, dict[str, Any]] = {}
    for role, speaker_support in support.items():
        ranked = sorted(
            speaker_support.values(),
            key=lambda item: (
                -float(item.get("weighted_duration", 0.0)),
                _speaker_key(item.get("speaker")),
            ),
        )
        if not ranked:
            continue
        primary = dict(ranked[0])
        total = sum(float(item.get("weighted_duration", 0.0)) for item in ranked)
        runner_up = (
            float(ranked[1].get("weighted_duration", 0.0))
            if len(ranked) > 1
            else 0.0
        )
        primary_weight = float(primary.get("weighted_duration", 0.0))
        primary["share"] = round(primary_weight / max(total, 1e-9), 6)
        primary["margin"] = round(primary_weight - runner_up, 6)
        reasons = []
        if int(primary.get("turn_count", 0)) < MIN_CANONICAL_TURNS:
            reasons.append("fewer_than_two_independent_prompt_turns")
        if float(primary.get("duration_sec", 0.0)) < MIN_CANONICAL_DURATION_SEC:
            reasons.append("insufficient_anchored_duration")
        if float(primary["share"]) < MIN_CANONICAL_SHARE:
            reasons.append("speaker_share_not_dominant")
        if float(primary["margin"]) < MIN_CANONICAL_MARGIN:
            reasons.append("speaker_margin_too_small")
        if reasons:
            rejected[role] = {**primary, "reasons": reasons}
        else:
            provisional[role] = primary

    roles_by_speaker: dict[str, list[str]] = defaultdict(list)
    for role, item in provisional.items():
        roles_by_speaker[_speaker_key(item["speaker"])].append(role)
    canonical: dict[str, dict[str, Any]] = {}
    for role, item in provisional.items():
        competing_roles = roles_by_speaker[_speaker_key(item["speaker"])]
        if len(competing_roles) > 1:
            rejected[role] = {
                **item,
                "reasons": ["speaker_is_provisional_for_multiple_roles"],
                "competing_roles": sorted(competing_roles),
            }
            continue
        canonical[role] = item
    return canonical, rejected


def _observed_candidates(
    records: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    role_speakers: dict[str, set[str]] = defaultdict(set)
    speaker_roles: dict[str, set[str]] = defaultdict(set)
    speaker_values: dict[str, Any] = {}
    for record in records:
        role = str(record["role"])
        key = _speaker_key(record["speaker"])
        speaker_values.setdefault(key, record["speaker"])
        role_speakers[role].add(key)
        speaker_roles[key].add(role)
    split = [
        {
            "role": role,
            "speakers": [speaker_values[key] for key in sorted(speakers)],
        }
        for role, speakers in sorted(role_speakers.items())
        if len(speakers) > 1
    ]
    shared = [
        {"speaker": speaker_values[key], "roles": sorted(roles)}
        for key, roles in sorted(speaker_roles.items())
        if len(roles) > 1
    ]
    return split, shared


def _merge_conflicts(conflicts: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for conflict in sorted(
        conflicts,
        key=lambda item: (float(item["start_sec"]), float(item["end_sec"])),
    ):
        if (
            merged
            and merged[-1]["expected_role"] == conflict["expected_role"]
            and merged[-1]["actual_role"] == conflict["actual_role"]
            and _speaker_key(merged[-1]["actual_speaker"])
            == _speaker_key(conflict["actual_speaker"])
            and float(conflict["start_sec"]) - float(merged[-1]["end_sec"])
            <= MAX_MERGE_GAP_SEC
        ):
            merged[-1]["end_sec"] = max(
                float(merged[-1]["end_sec"]), float(conflict["end_sec"])
            )
            merged[-1]["observed_text"] += str(conflict.get("observed_text") or "")
            merged[-1]["segment_count"] += 1
            continue
        merged.append({**conflict, "segment_count": 1})
    return merged


def _issue_from_conflict(conflict: Mapping[str, Any]) -> dict[str, str]:
    expected = str(conflict["expected_role"])
    actual = str(conflict["actual_role"])
    expected_speaker = conflict["expected_speaker"]
    actual_speaker = conflict["actual_speaker"]
    start_sec = float(conflict["start_sec"])
    end_sec = float(conflict["end_sec"])
    observed_text = str(conflict.get("observed_text") or "")
    expected_turns = int(conflict["expected_prototype"]["turn_count"])
    actual_turns = int(conflict["actual_prototype"]["turn_count"])
    return {
        "可定位性": "否",
        "置信度": "高",
        "问题说明": (
            f"prompt 明确将台词“{observed_text}”归给{expected}。根据全片多个独立台词轮次，"
            f"{expected}在{expected_turns}个锚定轮次稳定对应匿名声纹 spk{expected_speaker}，"
            f"{actual}在{actual_turns}个锚定轮次稳定对应 spk{actual_speaker}；实际 "
            f"{start_sec:.2f}s - {end_sec:.2f}s 的该段台词却由 spk{actual_speaker} 发出，"
            f"与{actual}的声纹映射一致而非{expected}的声纹映射，存在角色台词绑定错误。"
        ),
        "问题类型": "音频质量问题",
        "时间区间": f"{start_sec:.2f}s - {end_sec:.2f}s",
        "关键帧秒": "",
        "BBox": "",
    }


def _shared_voice_conflicts(
    records: Sequence[Mapping[str, Any]],
    voiceprint_evidence: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Admit only clean single-turn role pairs with direct voiceprint support."""

    if voiceprint_evidence.get("status") != "scored":
        return []
    turn_ids_by_role: dict[str, set[int]] = defaultdict(set)
    for record in records:
        turn_ids_by_role[str(record["role"])].add(int(record["turn_index"]))
    eligible_clips_by_role: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    clips_by_id: dict[str, Mapping[str, Any]] = {}
    for clip in voiceprint_evidence.get("clips", ()):
        if not isinstance(clip, Mapping) or not clip.get("clip_id"):
            continue
        clips_by_id[str(clip["clip_id"])] = clip
        if clip.get("eligible") and clip.get("quality_valid"):
            eligible_clips_by_role[str(clip.get("role") or "")].append(clip)

    conflicts: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for pair in voiceprint_evidence.get("pairs", ()):
        if not isinstance(pair, Mapping):
            continue
        if (
            pair.get("decision") != "same_speaker_supported"
            or not pair.get("same_anonymous_speaker")
        ):
            continue
        left_role = str(pair.get("left_role") or "")
        right_role = str(pair.get("right_role") or "")
        if not left_role or not right_role or left_role == right_role:
            continue
        role_pair = tuple(sorted((left_role, right_role)))
        if role_pair in seen_pairs:
            continue
        # This branch closes only the sparse single-turn gap.  Repeated roles
        # use the stronger canonical-prototype path above.
        if any(len(turn_ids_by_role.get(role, ())) != 1 for role in role_pair):
            continue
        if any(len(eligible_clips_by_role.get(role, ())) != 1 for role in role_pair):
            continue
        left_clip = clips_by_id.get(str(pair.get("left_clip_id") or ""))
        right_clip = clips_by_id.get(str(pair.get("right_clip_id") or ""))
        if left_clip is None or right_clip is None:
            continue
        seen_pairs.add(role_pair)
        conflicts.append(
            {
                "roles": list(role_pair),
                "speaker": pair.get("left_speaker"),
                "cosine_similarity": float(pair["cosine_similarity"]),
                "same_speaker_threshold": float(
                    voiceprint_evidence.get("same_speaker_threshold", 0.55)
                ),
                "role_clips": {
                    left_role: dict(left_clip),
                    right_role: dict(right_clip),
                },
                "start_sec": min(
                    float(left_clip["start_sec"]),
                    float(right_clip["start_sec"]),
                ),
                "end_sec": max(
                    float(left_clip["end_sec"]),
                    float(right_clip["end_sec"]),
                ),
            }
        )
    return conflicts


def _issue_from_shared_voice(conflict: Mapping[str, Any]) -> dict[str, str]:
    left_role, right_role = [str(item) for item in conflict["roles"]]
    role_clips = conflict["role_clips"]
    left_clip = role_clips[left_role]
    right_clip = role_clips[right_role]
    similarity = float(conflict["cosine_similarity"])
    threshold = float(conflict["same_speaker_threshold"])
    speaker = conflict.get("speaker")
    start_sec = float(conflict["start_sec"])
    end_sec = float(conflict["end_sec"])
    return {
        "可定位性": "否",
        "置信度": "高",
        "问题说明": (
            f"prompt 明确由{left_role}说出台词“{left_clip.get('dialogue_text', '')}”，"
            f"并由{right_role}说出“{right_clip.get('dialogue_text', '')}”。实际对两段独立"
            f"台词片段进行 CAM++ 直接声纹验证，余弦相似度为 {similarity:.3f}，"
            f"超过严格同声纹阈值 {threshold:.2f}，且两段均对应匿名声纹 spk{speaker}；"
            "两名角色的配音声纹未被区分，存在多角色共用高度一致声纹的问题。"
        ),
        "问题类型": "音频质量问题",
        "时间区间": f"{start_sec:.2f}s - {end_sec:.2f}s",
        "关键帧秒": "",
        "BBox": "",
    }


def _underdetermined(reason: str) -> dict[str, Any]:
    return {
        "version": 2,
        "decision": "underdetermined",
        "reason": reason,
        "canonical_role_speakers": {},
        "rejected_role_prototypes": {},
        "role_speaker_support": {},
        "split_role_candidates": [],
        "shared_speaker_candidates": [],
        "directional_conflicts": [],
        "shared_voice_conflicts": [],
        "issues": [],
    }


def resolve_speaker_binding(
    prompt_speech_plan: Mapping[str, Any],
    prompt_turn_alignment: Sequence[Mapping[str, Any]],
    *,
    binding_status: str = "fine_grained_turns",
    speech_evidence_status: str = "speech_present",
    clustering: Mapping[str, Any] | None = None,
    voiceprint_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve only evidence-identifiable role binding or shared-voice defects."""

    clustering = clustering if isinstance(clustering, Mapping) else {}
    voiceprint_evidence = (
        voiceprint_evidence if isinstance(voiceprint_evidence, Mapping) else {}
    )
    if speech_evidence_status == "bgm_only":
        return _underdetermined("bgm_only_has_no_actionable_speech")
    if binding_status != "fine_grained_turns":
        return _underdetermined("fine_grained_speaker_turns_unavailable")
    if str(prompt_speech_plan.get("scope") or "") == "none":
        return _underdetermined("prompt_has_no_explicit_role_dialogue_scope")
    algorithm_status = str(clustering.get("cluster_algorithm_status") or "")
    if algorithm_status.startswith("fallback_single_cluster"):
        return _underdetermined("campp_clustering_fallback_is_not_actionable")

    records = _anchored_segment_records(
        prompt_speech_plan,
        prompt_turn_alignment,
    )
    if not records:
        return _underdetermined("no_prompt_anchored_speaker_segments")
    support = _role_speaker_support(records)
    canonical, rejected = _canonical_role_candidates(support)
    canonical_role_by_speaker = {
        _speaker_key(item["speaker"]): role for role, item in canonical.items()
    }
    raw_conflicts: list[dict[str, Any]] = []
    for record in records:
        expected_role = str(record["role"])
        expected_prototype = canonical.get(expected_role)
        if expected_prototype is None:
            continue
        if _speaker_key(record["speaker"]) == _speaker_key(
            expected_prototype["speaker"]
        ):
            continue
        actual_role = canonical_role_by_speaker.get(_speaker_key(record["speaker"]))
        if actual_role is None or actual_role == expected_role:
            continue
        if float(record["duration_sec"]) < MIN_CONFLICT_DURATION_SEC:
            continue
        raw_conflicts.append(
            {
                "expected_role": expected_role,
                "actual_role": actual_role,
                "expected_speaker": expected_prototype["speaker"],
                "actual_speaker": record["speaker"],
                "start_sec": record["start_sec"],
                "end_sec": record["end_sec"],
                "observed_text": record["text"],
                "dialogue_text": record["dialogue_text"],
                "prompt_start": record.get("prompt_start"),
                "prompt_end": record.get("prompt_end"),
                "dialogue_match_score": record["dialogue_match_score"],
                "expected_prototype": expected_prototype,
                "actual_prototype": canonical[actual_role],
            }
        )
    conflicts = _merge_conflicts(raw_conflicts)
    split_candidates, shared_candidates = _observed_candidates(records)
    shared_voice_conflicts = _shared_voice_conflicts(records, voiceprint_evidence)
    issues = [
        *(_issue_from_conflict(conflict) for conflict in conflicts),
        *(
            _issue_from_shared_voice(conflict)
            for conflict in shared_voice_conflicts
        ),
    ]
    if issues:
        decision = "supported"
        if conflicts and shared_voice_conflicts:
            reason = "multiple_supported_speaker_binding_defects"
        elif conflicts:
            reason = "cross_role_canonical_speaker_conflict"
        else:
            reason = "direct_voiceprint_shared_role_voice"
    elif split_candidates or shared_candidates:
        decision = "underdetermined"
        reason = "speaker_candidates_lack_symmetric_role_prototypes"
    elif len(canonical) >= 2:
        decision = "contradicted"
        reason = "anchored_turns_are_consistent_with_distinct_role_speakers"
    else:
        decision = "underdetermined"
        reason = "insufficient_repeated_role_turns_for_global_mapping"
    return {
        "version": 2,
        "decision": decision,
        "reason": reason,
        "canonical_role_speakers": canonical,
        "rejected_role_prototypes": rejected,
        "role_speaker_support": support,
        "split_role_candidates": split_candidates,
        "shared_speaker_candidates": shared_candidates,
        "directional_conflicts": conflicts,
        "shared_voice_conflicts": shared_voice_conflicts,
        "issues": issues,
    }
