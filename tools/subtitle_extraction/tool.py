"""Extract and temporally merge burned-in subtitle observations."""

from __future__ import annotations

import re
import tempfile
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable, Protocol

from tools.media.ffmpeg import extract_video_frames

from .schemas import SubtitleObservation, SubtitleSegment, SubtitleTrack


UNVERIFIED_SINGLETON_SOURCE = "burned_in_unverified_singleton"


class SubtitleExtractionBackend(Protocol):
    name: str

    def recognize(
        self,
        timestamp_sec: float,
        image_path: Path,
    ) -> Iterable[SubtitleObservation]:
        ...


def _normalized_text(text: str) -> str:
    return re.sub(r"\s+", "", text).strip()


def _same_caption(left: str, right: str) -> bool:
    left_normalized = _normalized_text(left)
    right_normalized = _normalized_text(right)
    if not left_normalized or not right_normalized:
        return False
    if left_normalized == right_normalized:
        return True
    return SequenceMatcher(None, left_normalized, right_normalized).ratio() >= 0.9


def _textual_characters(text: str) -> str:
    return "".join(
        character
        for character in _normalized_text(text)
        if character.isalnum()
    )


def is_unverified_singleton(segment: SubtitleSegment) -> bool:
    """Return whether OCR saw one character in only one sampled frame."""

    return segment.source == UNVERIFIED_SINGLETON_SOURCE


def subtitle_evidence_for_judge(
    track: SubtitleTrack,
) -> tuple[SubtitleTrack, tuple[SubtitleSegment, ...]]:
    """Keep raw singleton OCR auditable while withholding it from the judge."""

    rejected = tuple(
        segment for segment in track.segments if is_unverified_singleton(segment)
    )
    accepted = tuple(
        segment for segment in track.segments if not is_unverified_singleton(segment)
    )
    return SubtitleTrack(segments=accepted, backend=track.backend), rejected


def _nearby_bbox(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> bool:
    left_center = ((left[0] + left[2]) / 2, (left[1] + left[3]) / 2)
    right_center = ((right[0] + right[2]) / 2, (right[1] + right[3]) / 2)
    return (
        abs(left_center[0] - right_center[0]) <= 0.15
        and abs(left_center[1] - right_center[1]) <= 0.12
    )


def merge_subtitle_observations(
    observations: Iterable[SubtitleObservation],
    *,
    frame_interval: float,
    backend: str = "unknown",
) -> SubtitleTrack:
    ordered = sorted(
        (item for item in observations if _normalized_text(item.text)),
        key=lambda item: (item.timestamp_sec, item.bbox[1], item.bbox[0]),
    )
    if not ordered:
        return SubtitleTrack(segments=(), backend=backend)
    groups: list[list[SubtitleObservation]] = []
    for observation in ordered:
        candidates = [
            group
            for group in groups
            if observation.timestamp_sec - group[-1].timestamp_sec
            <= frame_interval * 1.5
            and _same_caption(group[-1].text, observation.text)
            and _nearby_bbox(group[-1].bbox, observation.bbox)
        ]
        if not candidates:
            groups.append([observation])
            continue
        best_group = min(
            candidates,
            key=lambda group: (
                observation.timestamp_sec - group[-1].timestamp_sec,
                abs(
                    (observation.bbox[1] + observation.bbox[3])
                    - (group[-1].bbox[1] + group[-1].bbox[3])
                ),
            ),
        )
        best_group.append(observation)
    segments = []
    for group in groups:
        text = max((item.text for item in group), key=len)
        source = "burned_in"
        if len(group) == 1 and len(_textual_characters(text)) <= 1:
            source = UNVERIFIED_SINGLETON_SOURCE
        segments.append(
            SubtitleSegment(
                start_sec=group[0].timestamp_sec,
                end_sec=group[-1].timestamp_sec + frame_interval,
                text=text,
                bbox=group[len(group) // 2].bbox,
                confidence=sum(item.confidence for item in group) / len(group),
                source=source,
            )
        )
    return SubtitleTrack(segments=tuple(segments), backend=backend)


def extract_subtitles(
    video_path: Path,
    *,
    backend: SubtitleExtractionBackend | None = None,
    fps: float = 2.0,
) -> SubtitleTrack:
    if backend is None:
        from .backends.rapidocr import RapidOCRBackend

        backend = RapidOCRBackend()
    with tempfile.TemporaryDirectory(prefix="auralis_subtitles_") as temp_dir:
        frames = extract_video_frames(video_path, Path(temp_dir), fps=fps)
        observations = [
            observation
            for timestamp_sec, image_path in frames
            for observation in backend.recognize(timestamp_sec, image_path)
        ]
    return merge_subtitle_observations(
        observations,
        frame_interval=1.0 / fps,
        backend=backend.name,
    )
