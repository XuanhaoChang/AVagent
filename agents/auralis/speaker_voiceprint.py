"""Direct CAM++ verification for prompt-anchored dialogue clips.

Anonymous diarization labels are useful candidates, but a shared label does
not establish that two role turns contain the same voice.  This module builds
independent, source-anchored clips and asks a local speaker-verification scorer
for role-pair cosine evidence.  It never infers role identity from a cluster
number and explicitly abstains when clips are short or acoustically invalid.
"""

from __future__ import annotations

from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from tools.speech_transcription.schemas import SpeechTranscript


SpeakerVoiceprintScorer = Callable[
    [Path, Sequence[Mapping[str, Any]]],
    Mapping[str, Any],
]


MIN_CLIP_SPEECH_DURATION_SEC = 0.75
MIN_DIALOGUE_MATCH_SCORE = 0.70
MIN_OBSERVED_DIALOGUE_PURITY = 0.80
MAX_INTERNAL_GAP_SEC = 0.45
MAX_CLIP_SPAN_SEC = 8.00


def _candidate_groups(
    alignment: Mapping[str, Any],
) -> list[list[dict[str, Any]]]:
    expected = "".join(
        character.casefold()
        for character in str(alignment.get("dialogue_text") or "")
        if character.isalnum()
    )
    records: list[dict[str, Any]] = []
    for segment in alignment.get("matched_segments", ()):
        if not isinstance(segment, Mapping) or segment.get("speaker") is None:
            continue
        try:
            start_sec = float(segment["start_sec"])
            end_sec = float(segment["end_sec"])
            match_score = float(segment.get("dialogue_match_score", 1.0))
        except (KeyError, TypeError, ValueError):
            continue
        if end_sec <= start_sec or match_score < MIN_DIALOGUE_MATCH_SCORE:
            continue
        observed = "".join(
            character.casefold()
            for character in str(segment.get("text") or "")
            if character.isalnum()
        )
        try:
            observed_precision = float(segment["dialogue_observed_precision"])
        except (KeyError, TypeError, ValueError):
            matcher = SequenceMatcher(None, expected, observed)
            matched = sum(block.size for block in matcher.get_matching_blocks())
            observed_precision = matched / max(1, len(observed))
        records.append(
            {
                "start_sec": start_sec,
                "end_sec": end_sec,
                "speaker": segment.get("speaker"),
                "text": str(segment.get("text") or ""),
                "dialogue_match_score": match_score,
                "dialogue_observed_precision": observed_precision,
            }
        )
    records.sort(key=lambda item: (item["start_sec"], item["end_sec"]))
    groups: list[list[dict[str, Any]]] = []
    for record in records:
        if (
            not groups
            or groups[-1][-1]["speaker"] != record["speaker"]
            or record["start_sec"] - groups[-1][-1]["end_sec"]
            > MAX_INTERNAL_GAP_SEC
            or record["end_sec"] - groups[-1][0]["start_sec"]
            > MAX_CLIP_SPAN_SEC
        ):
            groups.append([record])
        else:
            groups[-1].append(record)
    return groups


def build_role_voiceprint_clips(
    transcript: SpeechTranscript,
) -> list[dict[str, Any]]:
    """Build auditable role-turn clips without using prompt count as clusters."""

    binding = transcript.metadata.get("speaker_binding_evidence", {})
    if not isinstance(binding, Mapping):
        return []
    clips: list[dict[str, Any]] = []
    for turn_index, alignment in enumerate(
        binding.get("prompt_turn_alignment", ())
    ):
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
        for group_index, group in enumerate(_candidate_groups(alignment)):
            start_sec = float(group[0]["start_sec"])
            end_sec = float(group[-1]["end_sec"])
            speech_duration_sec = sum(
                float(item["end_sec"]) - float(item["start_sec"])
                for item in group
            )
            reasons: list[str] = []
            if speech_duration_sec < MIN_CLIP_SPEECH_DURATION_SEC:
                reasons.append("insufficient_anchored_speech_duration")
            if end_sec - start_sec > MAX_CLIP_SPAN_SEC:
                reasons.append("clip_span_too_long")
            observed_precision_min = min(
                float(item["dialogue_observed_precision"])
                for item in group
            )
            if observed_precision_min < MIN_OBSERVED_DIALOGUE_PURITY:
                reasons.append("observed_text_contains_other_dialogue")
            clips.append(
                {
                    "clip_id": (
                        f"prompt-turn-{turn_index:03d}-part-{group_index:02d}"
                    ),
                    "prompt_turn_index": turn_index,
                    "role": role,
                    "dialogue_text": dialogue_text,
                    "speaker": group[0]["speaker"],
                    "start_sec": round(start_sec, 6),
                    "end_sec": round(end_sec, 6),
                    "speech_duration_sec": round(speech_duration_sec, 6),
                    "dialogue_match_score_min": round(
                        min(float(item["dialogue_match_score"]) for item in group),
                        6,
                    ),
                    "dialogue_observed_precision_min": round(
                        observed_precision_min,
                        6,
                    ),
                    "observed_text": "".join(str(item["text"]) for item in group),
                    "eligible": not reasons,
                    "rejection_reasons": reasons,
                }
            )
    return clips


def _not_evaluable(
    reason: str,
    clips: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any]:
    return {
        "version": 1,
        "status": "not_evaluable",
        "reason": reason,
        "clips": [dict(item) for item in clips],
        "pairs": [],
    }


def evaluate_prompt_voiceprints(
    audio_path: Path,
    transcript: SpeechTranscript,
    *,
    scorer: SpeakerVoiceprintScorer | None,
) -> dict[str, Any]:
    """Score independent role clips and expose strict pair decisions."""

    if scorer is None:
        return _not_evaluable("speaker_voiceprint_scorer_disabled")
    metadata = transcript.metadata
    if str(metadata.get("speech_evidence_status") or "") == "bgm_only":
        return _not_evaluable("bgm_only_has_no_actionable_speech")
    if str(metadata.get("speech_evidence_status") or "") == "speech_with_bgm":
        return _not_evaluable("speech_with_bgm_requires_robust_voice_separation")
    binding = metadata.get("speaker_binding_evidence", {})
    if not isinstance(binding, Mapping) or binding.get("status") != "fine_grained_turns":
        return _not_evaluable("fine_grained_speaker_turns_unavailable")

    clips = build_role_voiceprint_clips(transcript)
    eligible = [item for item in clips if item["eligible"]]
    if len({str(item["role"]) for item in eligible}) < 2:
        return _not_evaluable("fewer_than_two_eligible_prompt_roles", clips)
    try:
        raw = scorer(audio_path, eligible)
    except Exception as exc:
        return {
            "version": 1,
            "status": "scoring_failed",
            "reason": "speaker_voiceprint_scorer_exception",
            "error": f"{type(exc).__name__}: {exc}",
            "clips": clips,
            "pairs": [],
        }
    if not isinstance(raw, Mapping):
        return {
            "version": 1,
            "status": "scoring_failed",
            "reason": "speaker_voiceprint_scorer_returned_non_mapping",
            "clips": clips,
            "pairs": [],
        }

    worker_clips = {
        str(item.get("clip_id") or ""): item
        for item in raw.get("clips", ())
        if isinstance(item, Mapping) and item.get("clip_id")
    }
    joined_clips: list[dict[str, Any]] = []
    by_id: dict[str, dict[str, Any]] = {}
    for clip in clips:
        acoustic = worker_clips.get(str(clip["clip_id"]), {})
        joined = {
            **clip,
            "acoustic_quality": dict(acoustic),
            "quality_valid": bool(acoustic.get("quality_valid")),
        }
        joined_clips.append(joined)
        by_id[str(clip["clip_id"])] = joined

    try:
        same_threshold = float(raw.get("same_speaker_threshold", 0.55))
        different_threshold = float(raw.get("different_speaker_threshold", 0.30))
    except (TypeError, ValueError):
        same_threshold = 0.55
        different_threshold = 0.30
    pairs: list[dict[str, Any]] = []
    for item in raw.get("pairs", ()):
        if not isinstance(item, Mapping):
            continue
        left = by_id.get(str(item.get("left_clip_id") or ""))
        right = by_id.get(str(item.get("right_clip_id") or ""))
        if left is None or right is None:
            continue
        try:
            similarity = float(item["cosine_similarity"])
        except (KeyError, TypeError, ValueError):
            continue
        same_anonymous_speaker = str(left["speaker"]) == str(right["speaker"])
        if not left["quality_valid"] or not right["quality_valid"]:
            decision = "insufficient_clip_quality"
        elif same_anonymous_speaker and similarity >= same_threshold:
            decision = "same_speaker_supported"
        elif similarity <= different_threshold:
            decision = "different_speaker_supported"
        elif not same_anonymous_speaker and similarity >= same_threshold:
            decision = "clustering_and_voiceprint_disagree"
        else:
            decision = "ambiguous"
        pairs.append(
            {
                "left_clip_id": left["clip_id"],
                "right_clip_id": right["clip_id"],
                "left_role": left["role"],
                "right_role": right["role"],
                "left_prompt_turn_index": left["prompt_turn_index"],
                "right_prompt_turn_index": right["prompt_turn_index"],
                "left_speaker": left["speaker"],
                "right_speaker": right["speaker"],
                "same_anonymous_speaker": same_anonymous_speaker,
                "cosine_similarity": round(similarity, 6),
                "decision": decision,
            }
        )
    return {
        "version": 1,
        "status": "scored",
        "reason": "direct_campp_role_clip_verification",
        "backend": str(raw.get("backend") or "campp_direct_voiceprint"),
        "model": str(raw.get("model") or ""),
        "device": str(raw.get("device") or ""),
        "same_speaker_threshold": same_threshold,
        "different_speaker_threshold": different_threshold,
        "clips": joined_clips,
        "pairs": pairs,
    }
