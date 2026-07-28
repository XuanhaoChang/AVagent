"""ASR result contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class SpeechWord:
    start_sec: float
    end_sec: float
    text: str
    probability: float | None = None


@dataclass(frozen=True)
class SpeechSegment:
    start_sec: float
    end_sec: float
    text: str
    confidence: str
    words: Tuple[SpeechWord, ...] = ()


@dataclass(frozen=True)
class SpeechTranscript:
    language: str
    segments: Tuple[SpeechSegment, ...]
    backend: str
    model: str
    device: str
