"""Prompt-anchored constrained ASR candidate scoring.

The user prompt is deliberately treated as free-form text.  Reference speech
is found by aligning each complete ASR segment (or a small adjacent context
window) against every possible prompt substring after format-insensitive text
normalization.  Quotes, role labels, Markdown, JSON keys, and line structure
are therefore optional context rather than required syntax.

Every accepted reference keeps an exact character span in the original
prompt.  If no sufficiently strong span exists, the caller gets
``no_reference_dialogue`` and no acoustic candidate is manufactured.
"""

from __future__ import annotations

import math
import os
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from tools.speech_transcription.schemas import SpeechSegment, SpeechTranscript


MIN_REFERENCE_CHARACTERS = 5
MIN_ALIGNMENT_SIMILARITY = 0.80
MAX_ALIGNMENT_EDITS = 4
MAX_CONTEXT_SEGMENTS = 3
MAX_CONTEXT_CHARACTERS = 72
MAX_CONTEXT_GAP_SEC = 1.25
MAX_SCORE_CANDIDATES = 12
DEFAULT_DECISION_DELTA = 0.70
_ISSUE_TIME_RANGE = re.compile(
    r"^\s*(?P<start>\d+(?:\.\d+)?)s\s*-\s*"
    r"(?P<end>\d+(?:\.\d+)?)s\s*$"
)

CandidateScorer = Callable[
    [Path, Sequence[Mapping[str, Any]]],
    Mapping[str, Any],
]


def normalize_reference_text(value: str) -> tuple[str, tuple[int, ...]]:
    """Return format-insensitive text plus source indexes for every character."""

    normalized: list[str] = []
    source_indexes: list[int] = []
    for source_index, source_character in enumerate(str(value or "")):
        # Normalize character-by-character so every retained output character
        # still maps to an auditable location in the original prompt.
        for character in unicodedata.normalize("NFKC", source_character).casefold():
            if unicodedata.category(character)[:1] not in {"L", "N"}:
                continue
            normalized.append(character)
            source_indexes.append(source_index)
    return "".join(normalized), tuple(source_indexes)


def _semi_global_match(query: str, target: str) -> tuple[int, int, int] | None:
    """Align all of ``query`` to the best contiguous substring of ``target``.

    Prefixes and suffixes in ``target`` are free.  Insertions, deletions, and
    substitutions inside the selected substring each cost one.
    """

    if not query or not target:
        return None
    target_length = len(target)
    previous_cost = [0] * (target_length + 1)
    previous_start = list(range(target_length + 1))

    for query_index, query_character in enumerate(query, start=1):
        current_cost = [query_index] + [0] * target_length
        current_start = [0] * (target_length + 1)
        for target_index, target_character in enumerate(target, start=1):
            candidates = (
                (
                    previous_cost[target_index - 1]
                    + int(query_character != target_character),
                    0,
                    previous_start[target_index - 1],
                ),
                (
                    previous_cost[target_index] + 1,
                    1,
                    previous_start[target_index],
                ),
                (
                    current_cost[target_index - 1] + 1,
                    2,
                    current_start[target_index - 1],
                ),
            )
            cost, _priority, start = min(candidates)
            current_cost[target_index] = cost
            current_start[target_index] = start
        previous_cost = current_cost
        previous_start = current_start

    end = min(
        range(1, target_length + 1),
        key=lambda index: (
            previous_cost[index],
            abs((index - previous_start[index]) - len(query)),
            index,
        ),
    )
    start = previous_start[end]
    if start >= end:
        return None
    return start, end, previous_cost[end]


def _differences(observed: str, expected: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for operation, observed_start, observed_end, expected_start, expected_end in (
        SequenceMatcher(None, observed, expected, autojunk=False).get_opcodes()
    ):
        if operation == "equal":
            continue
        records.append(
            {
                "operation": operation,
                "observed": observed[observed_start:observed_end],
                "expected": expected[expected_start:expected_end],
                "observed_start": observed_start,
                "observed_end": observed_end,
                "expected_start": expected_start,
                "expected_end": expected_end,
            }
        )
    return records


def _difference_signatures(anchor: Mapping[str, Any]) -> set[tuple[Any, ...]]:
    prompt_start = int(anchor["prompt_normalized_start"])
    signatures: set[tuple[Any, ...]] = set()
    for item in anchor.get("differences", ()):
        if not isinstance(item, Mapping):
            continue
        signatures.add(
            (
                str(item.get("operation") or ""),
                prompt_start + int(item.get("expected_start") or 0),
                prompt_start + int(item.get("expected_end") or 0),
                str(item.get("observed") or ""),
                str(item.get("expected") or ""),
            )
        )
    return signatures


def _segments_can_share_context(
    left: SpeechSegment,
    right: SpeechSegment,
) -> bool:
    if float(right.start_sec) - float(left.end_sec) > MAX_CONTEXT_GAP_SEC:
        return False
    return not (
        left.speaker is not None
        and right.speaker is not None
        and left.speaker != right.speaker
    )


def _candidate_window(
    segments: Sequence[SpeechSegment],
    start_index: int,
    end_index: int,
) -> dict[str, Any] | None:
    selected = segments[start_index:end_index]
    if not selected:
        return None
    for left, right in zip(selected, selected[1:]):
        if not _segments_can_share_context(left, right):
            return None
    observed = "".join(normalize_reference_text(item.text)[0] for item in selected)
    if not observed or len(observed) > MAX_CONTEXT_CHARACTERS:
        return None
    return {
        "segment_start_index": start_index,
        "segment_end_index": end_index,
        "segment_indices": list(range(start_index, end_index)),
        "start_sec": float(selected[0].start_sec),
        "end_sec": float(selected[-1].end_sec),
        "observed_text": observed,
    }


def _align_window_to_prompt(
    window: Mapping[str, Any],
    *,
    prompt: str,
    normalized_prompt: str,
    prompt_source_indexes: Sequence[int],
) -> dict[str, Any] | None:
    observed = str(window["observed_text"])
    if len(observed) < MIN_REFERENCE_CHARACTERS:
        return None
    match = _semi_global_match(observed, normalized_prompt)
    if match is None:
        return None
    normalized_start, normalized_end, edit_distance = match
    expected = normalized_prompt[normalized_start:normalized_end]
    denominator = max(len(observed), len(expected), 1)
    similarity = 1.0 - (edit_distance / denominator)
    allowed_edits = min(
        MAX_ALIGNMENT_EDITS,
        max(1, int(math.floor(len(observed) * (1.0 - MIN_ALIGNMENT_SIMILARITY)))),
    )
    if (
        similarity < MIN_ALIGNMENT_SIMILARITY
        or edit_distance > allowed_edits
        or len(expected) < MIN_REFERENCE_CHARACTERS
        or not (0.70 <= len(expected) / len(observed) <= 1.30)
    ):
        return None

    prompt_start = int(prompt_source_indexes[normalized_start])
    prompt_end = int(prompt_source_indexes[normalized_end - 1]) + 1
    return {
        **dict(window),
        "expected_text": expected,
        "prompt_start": prompt_start,
        "prompt_end": prompt_end,
        "prompt_source_text": prompt[prompt_start:prompt_end],
        "prompt_normalized_start": normalized_start,
        "prompt_normalized_end": normalized_end,
        "alignment_similarity": round(similarity, 6),
        "alignment_edit_distance": int(edit_distance),
        "differences": _differences(observed, expected),
    }


def _context_windows_for_segment(
    segments: Sequence[SpeechSegment],
    segment_index: int,
) -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []
    segment_count = len(segments)
    for width in range(2, MAX_CONTEXT_SEGMENTS + 1):
        first_start = max(0, segment_index - width + 1)
        last_start = min(segment_index, segment_count - width)
        for start_index in range(first_start, last_start + 1):
            window = _candidate_window(
                segments,
                start_index,
                start_index + width,
            )
            if window is not None:
                windows.append(window)
    return windows


def extract_prompt_reference_candidates(
    user_prompt: str,
    transcript: SpeechTranscript,
) -> dict[str, Any]:
    """Mine source-anchored expected strings from an arbitrary user prompt."""

    prompt = str(user_prompt or "")
    normalized_prompt, prompt_source_indexes = normalize_reference_text(prompt)
    base: dict[str, Any] = {
        "version": 1,
        "method": "format_agnostic_prompt_asr_semiglobal_character_alignment",
        "normalization": "NFKC casefold; retain Unicode letters and numbers",
        "minimum_reference_characters": MIN_REFERENCE_CHARACTERS,
        "minimum_alignment_similarity": MIN_ALIGNMENT_SIMILARITY,
        "anchors": [],
        "candidates": [],
    }
    if not normalized_prompt:
        return {
            **base,
            "status": "no_reference_dialogue",
            "reason": "prompt_has_no_reference_text",
            "anchor_count": 0,
            "candidate_count": 0,
        }
    segments = tuple(transcript.segments)
    if not segments:
        return {
            **base,
            "status": "no_reference_dialogue",
            "reason": "asr_has_no_speech_segments",
            "anchor_count": 0,
            "candidate_count": 0,
        }

    anchors: list[dict[str, Any]] = []
    anchored_segments: set[int] = set()
    for index in range(len(segments)):
        window = _candidate_window(segments, index, index + 1)
        if window is None:
            continue
        anchor = _align_window_to_prompt(
            window,
            prompt=prompt,
            normalized_prompt=normalized_prompt,
            prompt_source_indexes=prompt_source_indexes,
        )
        if anchor is not None:
            anchors.append(anchor)
            anchored_segments.add(index)

    # A short or noisier VAD fragment may not carry enough context by itself.
    # Try a bounded adjacent window, but only for segments that failed the
    # strict full-segment test above.  This avoids flooding the scorer with
    # overlapping copies of already accepted references.
    selected_context_windows: set[tuple[int, int]] = set()
    seen_difference_signatures: set[tuple[Any, ...]] = set()
    for anchor in anchors:
        seen_difference_signatures.update(_difference_signatures(anchor))
    for index in range(len(segments)):
        if index in anchored_segments:
            continue
        aligned_windows = []
        for window in _context_windows_for_segment(segments, index):
            key = (
                int(window["segment_start_index"]),
                int(window["segment_end_index"]),
            )
            if key in selected_context_windows:
                continue
            anchor = _align_window_to_prompt(
                window,
                prompt=prompt,
                normalized_prompt=normalized_prompt,
                prompt_source_indexes=prompt_source_indexes,
            )
            if anchor is not None:
                signatures = _difference_signatures(anchor)
                if signatures and signatures.issubset(seen_difference_signatures):
                    continue
                aligned_windows.append(anchor)
        if not aligned_windows:
            continue
        best = max(
            aligned_windows,
            key=lambda item: (
                float(item["alignment_similarity"]),
                len(str(item["observed_text"])),
                -int(item["alignment_edit_distance"]),
            ),
        )
        key = (
            int(best["segment_start_index"]),
            int(best["segment_end_index"]),
        )
        selected_context_windows.add(key)
        anchors.append(best)
        seen_difference_signatures.update(_difference_signatures(best))
        anchored_segments.update(int(item) for item in best["segment_indices"])

    anchors.sort(
        key=lambda item: (
            float(item["start_sec"]),
            float(item["end_sec"]),
            int(item["prompt_start"]),
        )
    )
    for anchor_index, anchor in enumerate(anchors, start=1):
        anchor["candidate_id"] = f"prompt-asr-{anchor_index:03d}"

    candidates = [item for item in anchors if item["differences"]]
    candidates.sort(
        key=lambda item: (
            -float(item["alignment_similarity"]),
            -len(str(item["observed_text"])),
            float(item["start_sec"]),
        )
    )
    candidates = candidates[:MAX_SCORE_CANDIDATES]
    candidates.sort(key=lambda item: float(item["start_sec"]))
    status = (
        "no_reference_dialogue"
        if not anchors
        else "candidates_ready"
        if candidates
        else "matched_reference"
    )
    reason = (
        "no_full_asr_segment_matched_a_prompt_span"
        if not anchors
        else "prompt_references_match_asr"
        if not candidates
        else "source_anchored_asr_differences_found"
    )
    return {
        **base,
        "status": status,
        "reason": reason,
        "anchor_count": len(anchors),
        "candidate_count": len(candidates),
        "anchors": anchors,
        "candidates": candidates,
    }


def _decision_delta() -> float:
    try:
        value = float(
            os.getenv(
                "AURALIS_CONSTRAINED_CTC_MIN_DELTA",
                str(DEFAULT_DECISION_DELTA),
            )
        )
    except (TypeError, ValueError):
        return DEFAULT_DECISION_DELTA
    if not math.isfinite(value) or value <= 0:
        return DEFAULT_DECISION_DELTA
    return value


def _score_value(item: Mapping[str, Any], label: str) -> float | None:
    score = item.get(label)
    if not isinstance(score, Mapping):
        return None
    value = score.get("ctc_log_likelihood")
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def evaluate_prompt_constrained_asr(
    audio_path: Path,
    user_prompt: str,
    transcript: SpeechTranscript,
    *,
    scorer: CandidateScorer,
) -> dict[str, Any]:
    """Extract prompt candidates, score them, and assign pairwise decisions."""

    extraction = extract_prompt_reference_candidates(user_prompt, transcript)
    candidates = extraction["candidates"]
    if not candidates:
        return {
            **extraction,
            "candidate_scores": [],
        }
    threshold = _decision_delta()
    try:
        response = scorer(audio_path, candidates)
        raw_scores = response.get("scores", ())
        if not isinstance(raw_scores, (list, tuple)):
            raise ValueError("constrained scorer response has no score list")
        score_by_id = {
            str(item.get("candidate_id")): item
            for item in raw_scores
            if isinstance(item, Mapping) and item.get("candidate_id")
        }
        merged_scores: list[dict[str, Any]] = []
        for candidate in candidates:
            candidate_id = str(candidate["candidate_id"])
            raw_score = score_by_id.get(candidate_id, {})
            observed_score = _score_value(raw_score, "observed")
            expected_score = _score_value(raw_score, "expected")
            decision = "scoring_failed"
            pronunciation_relation = str(
                raw_score.get("pronunciation_relation") or "unverified"
            )
            delta = None
            ratio = None
            if observed_score is not None and expected_score is not None:
                delta = observed_score - expected_score
                ratio = math.exp(max(-50.0, min(50.0, delta)))
                if pronunciation_relation == "same_pronunciation":
                    # Character CTC scores also contain orthographic/language
                    # priors. They cannot turn two identical pronunciations
                    # (for example 棠棠/糖糖) into an audio error.
                    decision = "orthographic_homophone"
                elif pronunciation_relation != "different_pronunciation":
                    decision = "pronunciation_unverified"
                elif delta >= threshold:
                    decision = "observed_preferred"
                elif delta <= -threshold:
                    decision = "expected_preferred"
                else:
                    decision = "ambiguous"
            merged_scores.append(
                {
                    **dict(candidate),
                    "scoring_backend": str(
                        response.get("backend") or "sensevoice_constrained_ctc"
                    ),
                    "observed_score": dict(raw_score.get("observed", {}))
                    if isinstance(raw_score.get("observed"), Mapping)
                    else {},
                    "expected_score": dict(raw_score.get("expected", {}))
                    if isinstance(raw_score.get("expected"), Mapping)
                    else {},
                    "decision": decision,
                    "pronunciation_relation": pronunciation_relation,
                    "observed_pronunciation": list(
                        raw_score.get("observed_pronunciation", ())
                    ),
                    "expected_pronunciation": list(
                        raw_score.get("expected_pronunciation", ())
                    ),
                    "decision_threshold_log_likelihood": threshold,
                    "delta_log_likelihood_observed_minus_expected": (
                        round(delta, 6) if delta is not None else None
                    ),
                    "likelihood_ratio_observed_to_expected": (
                        round(ratio, 6) if ratio is not None else None
                    ),
                    "scoring_error": str(raw_score.get("error") or ""),
                }
            )
    except Exception as exc:
        return {
            **extraction,
            "status": "scoring_failed",
            "scoring_error": f"{type(exc).__name__}: {exc}",
            "candidate_scores": [],
        }
    return {
        **extraction,
        "status": "scored",
        "scoring_backend": str(
            response.get("backend") or "sensevoice_constrained_ctc"
        ),
        "decision_threshold_log_likelihood": threshold,
        "candidate_scores": merged_scores,
    }


def constrained_asr_issues(
    evidence: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    """Convert decisive observed-preferred comparisons into output issues."""

    issues: list[Mapping[str, Any]] = []
    for score in evidence.get("candidate_scores", ()):
        if not isinstance(score, Mapping):
            continue
        if score.get("decision") != "observed_preferred":
            continue
        expected = str(score.get("expected_text") or "")
        observed = str(score.get("observed_text") or "")
        if not expected or not observed or expected == observed:
            continue
        start = float(score["start_sec"])
        end = float(score["end_sec"])
        delta = float(score["delta_log_likelihood_observed_minus_expected"])
        ratio = float(score["likelihood_ratio_observed_to_expected"])
        similarity = float(score.get("alignment_similarity") or 0.0)
        confidence = "高" if delta >= 1.10 and similarity >= 0.90 else "中"
        issues.append(
            {
                "可定位性": "否",
                "置信度": confidence,
                "问题说明": (
                    f"prompt 原文字符区间 [{int(score['prompt_start'])}, "
                    f"{int(score['prompt_end'])}) 锚定的预期台词为“{expected}”，"
                    f"ASR 实际识别为“{observed}”。对同一局部音频进行受约束 "
                    f"SenseVoice CTC 候选评分后，实际候选相对预期候选的 "
                    f"ΔlogL={delta:.3f}（似然比约 {ratio:.2f}:1），支持该台词/读音差异；"
                    f"发生在 {start:.2f}s - {end:.2f}s。"
                ),
                "问题类型": "音频质量问题",
                "时间区间": f"{start:.2f}s - {end:.2f}s",
                "关键帧秒": "",
                "BBox": "",
            }
        )
    return tuple(issues)


def _issue_time_range(issue: Mapping[str, Any]) -> tuple[float, float] | None:
    match = _ISSUE_TIME_RANGE.match(str(issue.get("时间区间") or ""))
    if match is None:
        return None
    start = float(match["start"])
    end = float(match["end"])
    return (start, end) if end > start else None


def filter_contradicted_judge_issues(
    evidence: Mapping[str, Any],
    issues: Sequence[Mapping[str, Any]],
) -> tuple[tuple[Mapping[str, Any], ...], tuple[Mapping[str, Any], ...]]:
    """Veto Gemini pronunciation claims contradicted by local ASR decisions.

    The veto is intentionally narrow: it requires an audio issue, overlapping
    time, explicit candidate text, and pronunciation/dialogue wording.  Role
    binding findings at the same time remain untouched.
    """

    blocked_decisions = {
        "expected_preferred",
        "orthographic_homophone",
        "ambiguous",
        "pronunciation_unverified",
        "scoring_failed",
    }
    blocked_scores = [
        item
        for item in evidence.get("candidate_scores", ())
        if isinstance(item, Mapping) and item.get("decision") in blocked_decisions
    ]
    kept: list[Mapping[str, Any]] = []
    vetoed: list[Mapping[str, Any]] = []
    pronunciation_markers = ("读音", "发音", "念成", "说成", "台词内容", "台词错误")
    role_markers = ("角色", "说话人", "声纹", "归属", "绑定", "配音主体")
    for issue in issues:
        description = str(issue.get("问题说明") or "")
        time_range = _issue_time_range(issue)
        matched_score: Mapping[str, Any] | None = None
        if (
            issue.get("问题类型") == "音频质量问题"
            and time_range is not None
            and any(marker in description for marker in pronunciation_markers)
            and not any(marker in description for marker in role_markers)
        ):
            issue_start, issue_end = time_range
            for score in blocked_scores:
                score_start = float(score["start_sec"])
                score_end = float(score["end_sec"])
                overlap = min(issue_end, score_end) - max(issue_start, score_start)
                if overlap <= 0:
                    continue
                shorter_duration = min(
                    issue_end - issue_start,
                    score_end - score_start,
                )
                if overlap / max(shorter_duration, 1e-6) < 0.50:
                    continue
                observed = str(score.get("observed_text") or "")
                expected = str(score.get("expected_text") or "")
                differences = [
                    item
                    for item in score.get("differences", ())
                    if isinstance(item, Mapping)
                ]
                full_text_match = (
                    (observed and observed in description)
                    or (expected and expected in description)
                )
                changed_text_match = any(
                    str(item.get("observed") or "") in description
                    and str(item.get("expected") or "") in description
                    and bool(item.get("observed"))
                    and bool(item.get("expected"))
                    for item in differences
                )
                if full_text_match or changed_text_match:
                    matched_score = score
                    break
        if matched_score is None:
            kept.append(issue)
            continue
        vetoed.append(
            {
                "issue": dict(issue),
                "candidate_id": str(matched_score.get("candidate_id") or ""),
                "decision": str(matched_score.get("decision") or ""),
                "reason": "local_constrained_asr_contradicts_pronunciation_claim",
            }
        )
    return tuple(kept), tuple(vetoed)
