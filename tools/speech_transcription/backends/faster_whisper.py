"""Local Faster-Whisper backend with CUDA-to-CPU fallback."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from ..schemas import SpeechSegment, SpeechTranscript, SpeechWord


_CUDA_INITIALIZATION_ERROR_MARKERS = (
    "unable to load libcublas",
    "unable to load libcudnn",
    "cannot load libcublas",
    "cannot load libcudnn",
    "libcublas.so",
    "libcudnn.so",
    "libcudnn_",
    "cuda driver version is insufficient",
    "cudaerrorinsufficientdriver",
    "no cuda-capable device is detected",
    "cudaerrornodevice",
    "no cuda device",
    "cuda initialization error",
    "cudaerrorinitializationerror",
)


class FasterWhisperBackend:
    def __init__(
        self,
        *,
        model_name: str | None = None,
        device: str | None = None,
        compute_type: str | None = None,
        allow_cpu_fallback: bool | None = None,
    ) -> None:
        self.model_name = model_name or os.getenv(
            "AURALIS_ASR_MODEL",
            "large-v3",
        )
        self.requested_device = device or os.getenv("AURALIS_ASR_DEVICE", "cuda")
        self.requested_compute_type = compute_type or os.getenv(
            "AURALIS_ASR_COMPUTE_TYPE",
            "int8_float16",
        )
        self.allow_cpu_fallback = (
            allow_cpu_fallback
            if allow_cpu_fallback is not None
            else os.getenv("AURALIS_ASR_ALLOW_CPU_FALLBACK", "0")
            .strip()
            .lower()
            in {"1", "true", "yes"}
        )
        self._model: Any = None
        self.device = self.requested_device
        self.compute_type = self.requested_compute_type
        self.fallback_reason = ""

    def _load_model(self):
        if self._model is not None:
            return self._model
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "缺少 faster-whisper；请安装 requirements-audio-agent.txt"
            ) from exc
        try:
            self._model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
            )
        except Exception as exc:
            detail = str(exc).lower()
            cuda_initialization_error = any(
                marker in detail
                for marker in _CUDA_INITIALIZATION_ERROR_MARKERS
            ) and "out of memory" not in detail
            if (
                self.device != "cuda"
                or not self.allow_cpu_fallback
                or not cuda_initialization_error
            ):
                raise
            self.fallback_reason = str(exc)
            self.device = "cpu"
            self.compute_type = "int8"
            self._model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type,
            )
        return self._model

    def transcribe(self, audio_path: Path) -> SpeechTranscript:
        if not audio_path.is_file():
            raise FileNotFoundError(f"待转写音频不存在：{audio_path}")
        model = self._load_model()
        segments, info = model.transcribe(
            str(audio_path),
            beam_size=5,
            vad_filter=True,
            word_timestamps=True,
        )
        normalized = []
        for segment in segments:
            words = tuple(
                SpeechWord(
                    start_sec=float(word.start),
                    end_sec=float(word.end),
                    text=str(word.word),
                    probability=(
                        float(word.probability)
                        if word.probability is not None
                        else None
                    ),
                )
                for word in (segment.words or ())
                if word.start is not None and word.end is not None
            )
            average_probability = (
                sum(
                    word.probability
                    for word in words
                    if word.probability is not None
                )
                / max(
                    1,
                    sum(word.probability is not None for word in words),
                )
                if words
                else None
            )
            confidence = (
                "high"
                if average_probability is not None and average_probability >= 0.8
                else "medium"
            )
            normalized.append(
                SpeechSegment(
                    start_sec=float(segment.start),
                    end_sec=float(segment.end),
                    text=str(segment.text).strip(),
                    confidence=confidence,
                    words=words,
                )
            )
        return SpeechTranscript(
            language=str(getattr(info, "language", "") or ""),
            segments=tuple(normalized),
            backend="faster-whisper",
            model=self.model_name,
            device=f"{self.device}/{self.compute_type}",
        )
