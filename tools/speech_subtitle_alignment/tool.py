"""Deterministic edit-distance comparison of overlapping speech and subtitles."""

from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Sequence

from tools.speech_transcription.schemas import SpeechTranscript
from tools.subtitle_extraction.schemas import SubtitleTrack

from .schemas import AlignmentIssue, AlignmentResult


LOCAL_MIN_CHARACTERS = 4
# Localized issues are injected into the final result without model discretion,
# so require stronger OCR evidence than the broader candidate-only path.
LOCAL_MIN_OCR_CONFIDENCE = 0.90
LOCAL_MIN_SIMILARITY = 0.78
NUMERIC_MIN_DIGITS = 4
NUMERIC_MIN_OCR_WIDTH = 0.35
NUMERIC_MAX_TIME_GAP_SEC = 1.75
NUMERIC_MIN_MATCHED_EVENTS = 3
NUMERIC_MIN_MISMATCHES = 2

_NUMERIC_TOKEN = re.compile(r"(?<!\d)(\d[\d,，]*)(?!\d)")
_STANDALONE_OCR_NUMBER = re.compile(
    r"^\s*([+¥￥$]?)\s*(\d[\d,，]*)\s*(?:元|块|块钱)?\s*$"
)


@dataclass(frozen=True)
class _NumericEvent:
    value: str
    text: str
    time_sec: float
    start_sec: float
    end_sec: float
    confidence: float
    explicit_amount_marker: bool = False


def _normalize(text: str) -> str:
    value = unicodedata.normalize("NFKC", text)
    return "".join(
        character
        for character in value
        if not character.isspace()
        and not unicodedata.category(character).startswith(("P", "S"))
    )


def _edit_operations(expected: str, actual: str):
    rows, columns = len(expected) + 1, len(actual) + 1
    cost = [[0] * columns for _ in range(rows)]
    for row in range(rows):
        cost[row][0] = row
    for column in range(columns):
        cost[0][column] = column
    for row in range(1, rows):
        for column in range(1, columns):
            cost[row][column] = min(
                cost[row - 1][column] + 1,
                cost[row][column - 1] + 1,
                cost[row - 1][column - 1]
                + (expected[row - 1] != actual[column - 1]),
            )
    operations = []
    row, column = len(expected), len(actual)
    while row or column:
        if (
            row
            and column
            and cost[row][column]
            == cost[row - 1][column - 1]
            + (expected[row - 1] != actual[column - 1])
        ):
            if expected[row - 1] != actual[column - 1]:
                operations.append(
                    ("substitute", expected[row - 1], actual[column - 1])
                )
            row -= 1
            column -= 1
        elif row and cost[row][column] == cost[row - 1][column] + 1:
            operations.append(("delete", expected[row - 1], ""))
            row -= 1
        else:
            operations.append(("insert", "", actual[column - 1]))
            column -= 1
    operations.reverse()
    return operations


def _best_local_match(
    speech_text: str,
    subtitle_text: str,
) -> tuple[str, list[tuple[str, str, str]], float] | None:
    """Find the subtitle's best bounded character-level span in one ASR window."""

    if not speech_text or not subtitle_text:
        return None
    maximum_edits = max(2, int(math.ceil(len(subtitle_text) * 0.35)))
    minimum_length = max(1, len(subtitle_text) - maximum_edits)
    maximum_length = min(len(speech_text), len(subtitle_text) + maximum_edits)
    best: tuple[
        tuple[float, int, int, int],
        str,
        list[tuple[str, str, str]],
        float,
    ] | None = None
    for start in range(len(speech_text)):
        for length in range(minimum_length, maximum_length + 1):
            end = start + length
            if end > len(speech_text):
                break
            candidate = speech_text[start:end]
            operations = _edit_operations(candidate, subtitle_text)
            distance = len(operations)
            denominator = max(len(candidate), len(subtitle_text), 1)
            similarity = 1.0 - (distance / denominator)
            rank = (
                similarity,
                -distance,
                -abs(len(candidate) - len(subtitle_text)),
                -start,
            )
            if best is None or rank > best[0]:
                best = (rank, candidate, operations, similarity)
    if best is None:
        return None
    return best[1], best[2], best[3]


def _collect(operations: Sequence[tuple[str, str, str]], kind: str, index: int) -> str:
    return "".join(operation[index] for operation in operations if operation[0] == kind)


def _language_family(text: str) -> str:
    han = sum("\u3400" <= character <= "\u9fff" for character in text)
    latin = sum(
        "LATIN" in unicodedata.name(character, "")
        for character in text
        if character.isalpha()
    )
    if han >= 2 and han > latin:
        return "zh"
    if latin >= 2 and latin > han:
        return "latin"
    return "unknown"


def _gap(
    left_start: float,
    left_end: float,
    right_start: float,
    right_end: float,
) -> float:
    if min(left_end, right_end) > max(left_start, right_start):
        return 0.0
    return max(left_start, right_start) - min(left_end, right_end)


def _likely_subtitle(bbox: tuple[float, float, float, float]) -> bool:
    center_x = (bbox[0] + bbox[2]) / 2
    center_y = (bbox[1] + bbox[3]) / 2
    width = bbox[2] - bbox[0]
    return 0.15 <= center_x <= 0.85 and (center_y >= 0.45 or width >= 0.35)


def _canonical_number(raw: str) -> str | None:
    digits = "".join(character for character in raw if character.isdigit())
    if len(digits) < NUMERIC_MIN_DIGITS or not digits or set(digits) == {"0"}:
        return None
    return str(int(digits))


def _speech_numeric_events(speech_segments: Sequence) -> list[_NumericEvent]:
    events: list[_NumericEvent] = []
    for segment in speech_segments:
        word_events: list[_NumericEvent] = []
        for word in getattr(segment, "words", ()):
            normalized_word = unicodedata.normalize("NFKC", str(word.text))
            for match in _NUMERIC_TOKEN.finditer(normalized_word):
                value = _canonical_number(match.group(1))
                if value is None:
                    continue
                midpoint = (float(word.start_sec) + float(word.end_sec)) / 2
                word_events.append(
                    _NumericEvent(
                        value=value,
                        text=match.group(1),
                        time_sec=midpoint,
                        start_sec=float(word.start_sec),
                        end_sec=float(word.end_sec),
                        confidence=1.0,
                    )
                )
        if word_events:
            events.extend(word_events)
            continue

        text = unicodedata.normalize("NFKC", str(segment.text))
        if not text:
            continue
        duration = max(0.0, float(segment.end_sec) - float(segment.start_sec))
        for match in _NUMERIC_TOKEN.finditer(text):
            value = _canonical_number(match.group(1))
            if value is None:
                continue
            character_midpoint = match.start() + len(match.group(1)) / 2
            estimated_time = float(segment.start_sec) + (
                character_midpoint / max(len(text), 1)
            ) * duration
            events.append(
                _NumericEvent(
                    value=value,
                    text=match.group(1),
                    time_sec=estimated_time,
                    start_sec=float(segment.start_sec),
                    end_sec=float(segment.end_sec),
                    confidence=1.0,
                )
            )
    return sorted(events, key=lambda event: event.time_sec)


def _ocr_numeric_events(subtitle_segments: Sequence) -> list[_NumericEvent]:
    candidates: list[_NumericEvent] = []
    for subtitle in subtitle_segments:
        text = unicodedata.normalize("NFKC", str(subtitle.text))
        match = _STANDALONE_OCR_NUMBER.fullmatch(text)
        if match is None:
            continue
        value = _canonical_number(match.group(2))
        if value is None:
            continue
        width = float(subtitle.bbox[2]) - float(subtitle.bbox[0])
        confidence = float(subtitle.confidence)
        duration = float(subtitle.end_sec) - float(subtitle.start_sec)
        explicit_marker = bool(match.group(1))
        if confidence < LOCAL_MIN_OCR_CONFIDENCE or width < NUMERIC_MIN_OCR_WIDTH:
            continue
        if not explicit_marker and not (
            confidence >= 0.98 and width >= 0.50 and duration >= 0.75
        ):
            continue
        candidates.append(
            _NumericEvent(
                value=value,
                text=text.strip(),
                time_sec=float(subtitle.start_sec),
                start_sec=float(subtitle.start_sec),
                end_sec=float(subtitle.end_sec),
                confidence=confidence,
                explicit_amount_marker=explicit_marker,
            )
        )

    deduplicated: list[_NumericEvent] = []
    for event in sorted(candidates, key=lambda item: (item.time_sec, item.value)):
        duplicate_index = next(
            (
                index
                for index, previous in enumerate(deduplicated)
                if previous.value == event.value
                and abs(previous.time_sec - event.time_sec) <= 0.25
            ),
            None,
        )
        if duplicate_index is None:
            deduplicated.append(event)
            continue
        previous = deduplicated[duplicate_index]
        if (event.confidence, event.end_sec - event.start_sec) > (
            previous.confidence,
            previous.end_sec - previous.start_sec,
        ):
            deduplicated[duplicate_index] = event
    return sorted(deduplicated, key=lambda event: event.time_sec)


def _numeric_timeline_issues(
    speech_segments: Sequence,
    subtitle_segments: Sequence,
) -> list[AlignmentIssue]:
    """Compare high-confidence OCR amount onsets with ASR amount order.

    SenseVoice may return one coarse segment containing several amounts.  A
    plain substring lookup therefore accepts an amount that was spoken later
    in the segment.  This check estimates each ASR amount's position inside
    that segment (or uses word timestamps when available) and matches OCR
    amount-onset events monotonically on the same timeline.
    """

    speech_events = _speech_numeric_events(speech_segments)
    ocr_events = _ocr_numeric_events(subtitle_segments)
    if (
        len(speech_events) < NUMERIC_MIN_MATCHED_EVENTS
        or len(ocr_events) < NUMERIC_MIN_MATCHED_EVENTS
    ):
        return []

    pairs: list[tuple[_NumericEvent, _NumericEvent]] = []
    last_speech_index = -1
    for ocr_event in ocr_events:
        candidates = [
            (abs(speech_event.time_sec - ocr_event.time_sec), index, speech_event)
            for index, speech_event in enumerate(speech_events)
            if index > last_speech_index
            and abs(speech_event.time_sec - ocr_event.time_sec)
            <= NUMERIC_MAX_TIME_GAP_SEC
        ]
        if not candidates:
            continue
        _, speech_index, speech_event = min(candidates, key=lambda item: item[:2])
        last_speech_index = speech_index
        pairs.append((speech_event, ocr_event))

    mismatches = [
        (speech_event, ocr_event)
        for speech_event, ocr_event in pairs
        if speech_event.value != ocr_event.value
    ]
    if (
        len(pairs) < NUMERIC_MIN_MATCHED_EVENTS
        or len(mismatches) < NUMERIC_MIN_MISMATCHES
        or len(mismatches) / len(pairs) < 0.5
        or sum(event.explicit_amount_marker for _, event in mismatches) < 2
    ):
        return []

    speech_summary = " → ".join(
        f"{speech.value}@约{speech.time_sec:.2f}s" for speech, _ in mismatches
    )
    ocr_summary = " → ".join(
        f"{ocr.text}@{ocr.time_sec:.2f}s" for _, ocr in mismatches
    )
    detail = "；".join(
        (
            f"{ocr.time_sec:.2f}s OCR 显示“{ocr.value}”，"
            f"对应 ASR 金额候选为“{speech.value}”"
        )
        for speech, ocr in mismatches
    )
    return [
        AlignmentIssue(
            issue_type="numeric_timeline_mismatch",
            speech_text=speech_summary,
            subtitle_text=ocr_summary,
            difference="金额时序不一致：" + detail,
            start_sec=min(ocr.start_sec for _, ocr in mismatches),
            end_sec=max(
                max(ocr.end_sec, speech.time_sec)
                for speech, ocr in mismatches
            ),
            confidence="high",
            method="numeric_timeline_alignment",
        )
    ]


def _difference_from_operations(
    operations: Sequence[tuple[str, str, str]],
) -> tuple[str, str]:
    missing = _collect(operations, "delete", 1)
    extra = _collect(operations, "insert", 2)
    substitutions = [
        f"{left}→{right}"
        for kind, left, right in operations
        if kind == "substitute"
    ]
    if missing and not extra and not substitutions:
        return "missing_text", f"字幕缺少“{missing}”"
    if extra and not missing and not substitutions:
        return "extra_text", f"字幕多出“{extra}”"
    if substitutions and not missing and not extra:
        return "wrong_text", "字幕存在错字：" + "、".join(substitutions)
    details = []
    if missing:
        details.append(f"缺少“{missing}”")
    if extra:
        details.append(f"多出“{extra}”")
    if substitutions:
        details.append("错字" + "、".join(substitutions))
    return "text_mismatch", "字幕与语音不一致：" + "；".join(details)


def _localized_text_issues(
    transcript: SpeechTranscript,
    speech_segments: Sequence,
    subtitle_segments: Sequence,
) -> list[AlignmentIssue]:
    """Create high-precision ASR/OCR mismatches before noisy OCR aggregation."""

    issues: list[AlignmentIssue] = []
    transcript_language = (
        "zh"
        if transcript.language.lower().startswith(("zh", "yue"))
        else ""
    )
    for subtitle in subtitle_segments:
        actual = _normalize(subtitle.text)
        if (
            len(actual) < LOCAL_MIN_CHARACTERS
            or float(subtitle.confidence) < LOCAL_MIN_OCR_CONFIDENCE
        ):
            continue
        overlapping_speech = sorted(
            (
                speech
                for speech in speech_segments
                if min(speech.end_sec, subtitle.end_sec)
                > max(speech.start_sec, subtitle.start_sec)
            ),
            key=lambda item: (item.start_sec, item.end_sec),
        )
        if not overlapping_speech:
            continue
        speech_window = _normalize("".join(item.text for item in overlapping_speech))
        if len(speech_window) < LOCAL_MIN_CHARACTERS:
            continue
        speech_language = transcript_language or _language_family(speech_window)
        subtitle_language = _language_family(actual)
        if (
            speech_language != "unknown"
            and subtitle_language != "unknown"
            and speech_language != subtitle_language
        ):
            continue
        match = _best_local_match(speech_window, actual)
        if match is None:
            continue
        expected, operations, similarity = match
        if similarity < LOCAL_MIN_SIMILARITY or not operations:
            continue
        issue_type, difference = _difference_from_operations(operations)
        issues.append(
            AlignmentIssue(
                issue_type=issue_type,
                speech_text=expected,
                subtitle_text=actual,
                difference=difference,
                start_sec=float(subtitle.start_sec),
                end_sec=float(subtitle.end_sec),
                confidence="high",
                method="localized_asr_ocr",
            )
        )
    return issues


def check_speech_subtitle_alignment(
    transcript: SpeechTranscript,
    subtitles: SubtitleTrack,
) -> AlignmentResult:
    speech_segments = [
        segment for segment in transcript.segments if _normalize(segment.text)
    ]
    subtitle_segments = [
        segment
        for segment in subtitles.segments
        if _normalize(segment.text) and _likely_subtitle(segment.bbox)
    ]
    issues = _localized_text_issues(
        transcript,
        speech_segments,
        subtitle_segments,
    )
    issues.extend(_numeric_timeline_issues(speech_segments, subtitle_segments))
    speech_edges: list[set[int]] = [set() for _ in speech_segments]
    subtitle_edges: list[set[int]] = [set() for _ in subtitle_segments]
    for speech_index, speech in enumerate(speech_segments):
        for subtitle_index, subtitle in enumerate(subtitle_segments):
            if min(speech.end_sec, subtitle.end_sec) > max(
                speech.start_sec,
                subtitle.start_sec,
            ):
                speech_edges[speech_index].add(subtitle_index)
                subtitle_edges[subtitle_index].add(speech_index)

    components: list[tuple[list[int], list[int]]] = []
    visited_speech: set[int] = set()
    for initial_speech in range(len(speech_segments)):
        if initial_speech in visited_speech or not speech_edges[initial_speech]:
            continue
        pending_speech = [initial_speech]
        component_speech: set[int] = set()
        component_subtitles: set[int] = set()
        while pending_speech:
            speech_index = pending_speech.pop()
            if speech_index in component_speech:
                continue
            component_speech.add(speech_index)
            for subtitle_index in speech_edges[speech_index]:
                if subtitle_index in component_subtitles:
                    continue
                component_subtitles.add(subtitle_index)
                pending_speech.extend(subtitle_edges[subtitle_index])
        visited_speech.update(component_speech)
        components.append(
            (sorted(component_speech), sorted(component_subtitles))
        )

    for speech_indices, subtitle_indices in components:
        speech_group = sorted(
            (speech_segments[index] for index in speech_indices),
            key=lambda item: (item.start_sec, item.end_sec),
        )
        subtitle_group = sorted(
            (subtitle_segments[index] for index in subtitle_indices),
            key=lambda item: (item.start_sec, item.end_sec, item.bbox[0]),
        )
        speech_text = "".join(item.text for item in speech_group)
        expected = _normalize(speech_text)
        expected_language = (
            "zh"
            if transcript.language.lower().startswith(("zh", "yue"))
            else _language_family(expected)
        )
        same_script_subtitles = [
            item
            for item in subtitle_group
            if _language_family(_normalize(item.text))
            in {expected_language, "unknown"}
        ]
        if same_script_subtitles:
            subtitle_group = same_script_subtitles
        subtitle_text = "".join(item.text for item in subtitle_group)
        actual = _normalize(subtitle_text)
        if not expected or not actual or expected == actual:
            continue
        actual_language = _language_family(actual)
        if (
            expected_language == "zh"
            and actual_language == "latin"
        ) or (
            expected_language == "latin"
            and actual_language == "zh"
        ):
            issues.append(
                AlignmentIssue(
                    issue_type="language_mismatch",
                    speech_text=speech_text,
                    subtitle_text=subtitle_text,
                    difference=(
                        f"语音语言为 {expected_language}，"
                        f"字幕文字脚本为 {actual_language}"
                    ),
                    start_sec=max(
                        min(item.start_sec for item in speech_group),
                        min(item.start_sec for item in subtitle_group),
                    ),
                    end_sec=min(
                        max(item.end_sec for item in speech_group),
                        max(item.end_sec for item in subtitle_group),
                    ),
                    confidence="high",
                )
            )
            continue
        operations = _edit_operations(expected, actual)
        issue_type, difference = _difference_from_operations(operations)
        error_count = len(operations)
        denominator = max(len(expected), len(actual), 1)
        confidence = "high" if error_count / denominator >= 0.15 else "medium"
        issues.append(
            AlignmentIssue(
                issue_type=issue_type,
                speech_text=speech_text,
                subtitle_text=subtitle_text,
                difference=difference,
                start_sec=max(
                    min(item.start_sec for item in speech_group),
                    min(item.start_sec for item in subtitle_group),
                ),
                end_sec=min(
                    max(item.end_sec for item in speech_group),
                    max(item.end_sec for item in subtitle_group),
                ),
                confidence=confidence,
            )
        )

    for speech_index, speech in enumerate(speech_segments):
        if speech_edges[speech_index]:
            continue
        expected = _normalize(speech.text)
        same_text = [
            subtitle
            for subtitle in subtitle_segments
            if _normalize(subtitle.text) == expected
            and _gap(
                speech.start_sec,
                speech.end_sec,
                subtitle.start_sec,
                subtitle.end_sec,
            )
            <= 3.0
        ]
        if not same_text:
            continue
        closest = min(
            same_text,
            key=lambda subtitle: _gap(
                speech.start_sec,
                speech.end_sec,
                subtitle.start_sec,
                subtitle.end_sec,
            ),
        )
        issues.append(
            AlignmentIssue(
                issue_type="timing_mismatch",
                speech_text=speech.text,
                subtitle_text=closest.text,
                difference=(
                    "字幕文字与语音一致，但出现时间没有重叠："
                    f"语音 {speech.start_sec:g}s-{speech.end_sec:g}s，"
                    f"字幕 {closest.start_sec:g}s-{closest.end_sec:g}s"
                ),
                start_sec=min(speech.start_sec, closest.start_sec),
                end_sec=max(speech.end_sec, closest.end_sec),
                confidence="medium",
            )
        )
    return AlignmentResult(issues=tuple(issues))
