"""Typed contracts for the Auralis specialist agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Tuple

from tools.speech_subtitle_alignment.schemas import AlignmentResult
from tools.speech_transcription.schemas import SpeechTranscript
from tools.subtitle_extraction.schemas import SubtitleTrack


@dataclass(frozen=True)
class AuralisInput:
    video_path: Path
    user_prompt: str
    reference_images: Tuple[str, ...] = ()
    sample_id: str = ""


@dataclass(frozen=True)
class AuralisEvidence:
    media_metadata: Mapping[str, Any]
    transcript: SpeechTranscript
    subtitles: SubtitleTrack
    alignment: AlignmentResult


@dataclass(frozen=True)
class AuralisResult:
    status: str
    issues: Tuple[Mapping[str, Any], ...] = ()
    evidence: AuralisEvidence | None = None
    error: str = ""
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
