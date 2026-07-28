"""Public speech transcription tool."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .schemas import SpeechTranscript


class SpeechTranscriptionBackend(Protocol):
    def transcribe(self, audio_path: Path) -> SpeechTranscript:
        ...


def transcribe_speech(
    audio_path: Path,
    *,
    backend: SpeechTranscriptionBackend | None = None,
) -> SpeechTranscript:
    if backend is None:
        from .backends.faster_whisper import FasterWhisperBackend

        backend = FasterWhisperBackend()
    return backend.transcribe(audio_path)
