"""Orchestration for the Auralis audio-visual forensic agent."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from agents.auralis.schemas import (
    AuralisEvidence,
    AuralisInput,
    AuralisResult,
)
from tools.media.ffmpeg import extract_audio_wav, probe_video
from tools.speech_subtitle_alignment.tool import check_speech_subtitle_alignment
from tools.speech_transcription.tool import (
    transcribe_speech as run_speech_transcription,
)
from tools.subtitle_extraction.tool import (
    extract_subtitles as run_subtitle_extraction,
)


Judge = Callable[[AuralisInput, AuralisEvidence], Sequence[Mapping[str, Any]]]


def _no_judge(_agent_input: AuralisInput, _evidence: AuralisEvidence):
    return ()


class AuralisAgent:
    """Run every audio evidence tool and then ask one judge to verify findings."""

    def __init__(
        self,
        *,
        probe_video: Callable[[Path], Mapping[str, Any]] = probe_video,
        extract_audio: Callable[[Path, Path], Any] = extract_audio_wav,
        transcribe_speech: Callable[[Path], Any] | None = None,
        extract_subtitles: Callable[[Path], Any] | None = None,
        align_speech_subtitles: Callable[[Any, Any], Any] = (
            check_speech_subtitle_alignment
        ),
        judge: Judge | None = None,
        local_only: bool = False,
    ) -> None:
        if judge is None:
            if not local_only:
                raise ValueError(
                    "AuralisAgent 必须提供 judge；仅提取本地证据时显式设置 "
                    "local_only=True。"
                )
            judge = _no_judge
        if transcribe_speech is None:
            from tools.speech_transcription.backends.faster_whisper import (
                FasterWhisperBackend,
            )

            asr_backend = FasterWhisperBackend()
            transcribe_speech = lambda path: run_speech_transcription(
                path,
                backend=asr_backend,
            )
        if extract_subtitles is None:
            from tools.subtitle_extraction.backends.rapidocr import (
                RapidOCRBackend,
            )

            ocr_backend = RapidOCRBackend()
            extract_subtitles = lambda path: run_subtitle_extraction(
                path,
                backend=ocr_backend,
            )
        self._probe_video = probe_video
        self._extract_audio = extract_audio
        self._transcribe_speech = transcribe_speech
        self._extract_subtitles = extract_subtitles
        self._align_speech_subtitles = align_speech_subtitles
        self._judge = judge

    def analyze(self, agent_input: AuralisInput) -> AuralisResult:
        metadata = self._probe_video(agent_input.video_path)
        if not bool(metadata.get("has_audio")):
            return AuralisResult(
                status="no_audio",
                diagnostics={"reason": "ffprobe did not detect an audio stream"},
            )

        with tempfile.TemporaryDirectory(prefix="auralis_") as temp_dir:
            audio_path = Path(temp_dir) / "audio.wav"
            extracted = self._extract_audio(agent_input.video_path, audio_path)
            if isinstance(extracted, Path):
                audio_path = extracted
            transcript = self._transcribe_speech(audio_path)
            subtitles = self._extract_subtitles(agent_input.video_path)
            alignment = self._align_speech_subtitles(transcript, subtitles)
            evidence = AuralisEvidence(
                media_metadata=metadata,
                transcript=transcript,
                subtitles=subtitles,
                alignment=alignment,
            )
            issues = tuple(self._judge(agent_input, evidence))
        return AuralisResult(status="ok", issues=issues, evidence=evidence)
