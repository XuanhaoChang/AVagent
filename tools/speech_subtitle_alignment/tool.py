"""Deterministic edit-distance comparison of overlapping speech and subtitles."""

from __future__ import annotations

import unicodedata
from typing import Sequence

from tools.speech_transcription.schemas import SpeechTranscript
from tools.subtitle_extraction.schemas import SubtitleTrack

from .schemas import AlignmentIssue, AlignmentResult


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

    issues = []
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
        subtitle_text = "".join(item.text for item in subtitle_group)
        expected = _normalize(speech_text)
        actual = _normalize(subtitle_text)
        if not expected or not actual or expected == actual:
            continue
        expected_language = (
            "zh"
            if transcript.language.lower().startswith(("zh", "yue"))
            else _language_family(expected)
        )
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
        missing = _collect(operations, "delete", 1)
        extra = _collect(operations, "insert", 2)
        substitutions = [
            f"{left}→{right}"
            for kind, left, right in operations
            if kind == "substitute"
        ]
        if missing and not extra and not substitutions:
            issue_type = "missing_text"
            difference = f"字幕缺少“{missing}”"
        elif extra and not missing and not substitutions:
            issue_type = "extra_text"
            difference = f"字幕多出“{extra}”"
        elif substitutions and not missing and not extra:
            issue_type = "wrong_text"
            difference = "字幕存在错字：" + "、".join(substitutions)
        else:
            issue_type = "text_mismatch"
            details = []
            if missing:
                details.append(f"缺少“{missing}”")
            if extra:
                details.append(f"多出“{extra}”")
            if substitutions:
                details.append("错字" + "、".join(substitutions))
            difference = "字幕与语音不一致：" + "；".join(details)
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
