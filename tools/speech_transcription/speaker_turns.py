"""Pure helpers for preserving CAM++ turns below ASR sentence granularity."""

from __future__ import annotations

import re
from typing import Any, Mapping, Sequence


def _speaker_key(value: Any) -> int | str:
    try:
        return int(value)
    except (TypeError, ValueError):
        return str(value)


def normalize_speaker_turns(
    raw_turns: Sequence[Sequence[Any]],
) -> tuple[list[dict[str, Any]], dict[int | str, int]]:
    label_map: dict[int | str, int] = {}
    turns: list[dict[str, Any]] = []
    for item in raw_turns:
        if len(item) < 3:
            continue
        start = float(item[0])
        end = float(item[1])
        if end <= start:
            continue
        key = _speaker_key(item[2])
        speaker = label_map.setdefault(key, len(label_map))
        turns.append(
            {
                "start_sec": start,
                "end_sec": end,
                "speaker": speaker,
            }
        )
    return turns, label_map


def _flatten_tokens(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        tokens: list[str] = []
        for item in value:
            tokens.extend(_flatten_tokens(item))
        return tokens
    return [item for item in str(value).split() if item]


_SURFACE_TOKEN_RE = re.compile(
    r"[A-Za-z]+(?:['’][A-Za-z]+)*|\d+(?:\.\d+)*|[^\s]"
)
_SENTENCE_TERMINATORS = {"。", "！", "？", ".", "!", "?"}
_CLAUSE_SILENCE_GAP_SEC = 0.24


def _surface_tokens(value: Any) -> list[str]:
    """Recover FunASR timestamp units from sentence ``text`` fields.

    FunASR sometimes returns one string per VAD result while concatenating all
    word/character timestamps.  For Chinese, those timestamp units include
    individual Han characters and punctuation; Latin words remain grouped.
    Only callers that observe an exact timestamp-count match use this fallback.
    """

    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        tokens: list[str] = []
        for item in value:
            tokens.extend(_surface_tokens(item))
        return tokens
    return _SURFACE_TOKEN_RE.findall(str(value))


def _timestamp_tokens(item: Mapping[str, Any], count: int) -> list[str]:
    whitespace_tokens = _flatten_tokens(item.get("raw_text"))
    if len(whitespace_tokens) == count:
        return whitespace_tokens
    for value in (item.get("raw_text"), item.get("text")):
        surface_tokens = _surface_tokens(value)
        if len(surface_tokens) == count:
            return surface_tokens
    return []


def _join_tokens(tokens: Sequence[str]) -> str:
    text = ""
    for token in tokens:
        if (
            text
            and token
            and text[-1].isascii()
            and text[-1].isalnum()
            and token[0].isascii()
            and token[0].isalnum()
        ):
            text += " "
        text += token
    return text.strip()


def _timestamped_groups(
    timestamped: Sequence[tuple[str, float, float, int | str | None]],
) -> list[list[tuple[str, float, float, int | str | None]]]:
    """Split one FunASR sentence at acoustic turns and explicit sentence ends.

    Punctuation inference can occasionally return several dialogue sentences in
    one ``sentence_info`` item even though character timestamps are available.
    Keeping that whole item intact hides single-turn role boundaries (for
    example, male-female-male dialogue collapsed under one anonymous speaker).
    Terminal punctuation is source-local evidence for a clause boundary, so it
    is safe to retain the speaker label while exposing independent text spans.
    """

    groups: list[list[tuple[str, float, float, int | str | None]]] = []
    current: list[tuple[str, float, float, int | str | None]] = []
    for token in timestamped:
        if current and (
            current[-1][3] != token[3]
            or (
                token[0] not in _SENTENCE_TERMINATORS
                and token[1] - current[-1][2] > _CLAUSE_SILENCE_GAP_SEC
            )
        ):
            groups.append(current)
            current = []
        current.append(token)
        if token[0] in _SENTENCE_TERMINATORS:
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups


def _speaker_for_interval(
    start_sec: float,
    end_sec: float,
    turns: Sequence[Mapping[str, Any]],
    fallback: int | str | None,
) -> int | str | None:
    best_speaker = fallback
    best_overlap = 0.0
    for turn in turns:
        overlap = min(end_sec, float(turn["end_sec"])) - max(
            start_sec,
            float(turn["start_sec"]),
        )
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = turn.get("speaker")
    return best_speaker


def sentence_info_to_segments(
    sentence_info: Sequence[Mapping[str, Any]],
    speaker_turns: Sequence[Mapping[str, Any]],
    raw_label_map: Mapping[int | str, int],
) -> list[dict[str, Any]]:
    """Split punctuation sentences when token timestamps cross speaker turns."""

    segments: list[dict[str, Any]] = []
    for item in sentence_info:
        raw_speaker = item.get("spk")
        fallback = (
            None
            if raw_speaker is None
            else raw_label_map.get(_speaker_key(raw_speaker))
        )
        timestamps = list(item.get("timestamp") or ())
        tokens = _timestamp_tokens(item, len(timestamps))
        timestamped: list[tuple[str, float, float, int | str | None]] = []
        if tokens and len(tokens) == len(timestamps):
            for token, pair in zip(tokens, timestamps):
                if not isinstance(pair, (list, tuple)) or len(pair) < 2:
                    continue
                start_sec = float(pair[0]) / 1000.0
                end_sec = float(pair[1]) / 1000.0
                if end_sec <= start_sec:
                    continue
                timestamped.append(
                    (
                        token,
                        start_sec,
                        end_sec,
                        _speaker_for_interval(
                            start_sec,
                            end_sec,
                            speaker_turns,
                            fallback,
                        ),
                    )
                )
        groups = _timestamped_groups(timestamped) if timestamped else []
        # Even one timestamped group is more informative than the sentence
        # fallback: its speaker was resolved from the captured CAM++ turn.
        # This matters on the CPU ``vad_segment`` path, where FunASR omits a
        # sentence-level ``spk`` value but still returns fine-grained turns.
        if groups:
            for group in groups:
                if not any(token[0].isalnum() for token in group):
                    continue
                text = _join_tokens([token[0] for token in group])
                if text:
                    segments.append(
                        {
                            "start_sec": group[0][1],
                            "end_sec": group[-1][2],
                            "text": text,
                            "speaker": group[0][3],
                        }
                    )
            continue

        text_value = item.get("raw_text") or item.get("text") or ""
        text = _join_tokens(_flatten_tokens(text_value))
        start = item.get("start")
        end = item.get("end")
        valid_timestamps = [
            pair
            for pair in timestamps
            if isinstance(pair, (list, tuple)) and len(pair) >= 2
        ]
        if valid_timestamps:
            start = valid_timestamps[0][0]
            end = valid_timestamps[-1][1]
        if text and start is not None and end is not None and float(end) > float(start):
            segments.append(
                {
                    "start_sec": float(start) / 1000.0,
                    "end_sec": float(end) / 1000.0,
                    "text": text,
                    "speaker": fallback,
                }
            )
    return segments
