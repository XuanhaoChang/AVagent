#!/usr/bin/env python3
"""Run Auralis local evidence tools without any paid API."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from tools.speech_transcription.cuda import (
    cuda_library_dirs,
    cuda_process_environment,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="验证 Auralis 的 ffmpeg、Faster-Whisper、RapidOCR 和对齐工具。"
    )
    parser.add_argument("video", type=Path)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--asr-model", default="large-v3")
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda",
    )
    parser.add_argument(
        "--compute-type",
        default=None,
        help="默认 CUDA=int8_float16，CPU=int8。",
    )
    return parser.parse_args()


def _restart_for_cuda_if_needed(device: str) -> None:
    if device != "cuda" or os.getenv("AURALIS_CUDA_BOOTSTRAPPED") == "1":
        return
    if not cuda_library_dirs():
        return
    os.execve(
        sys.executable,
        [sys.executable, *sys.argv],
        cuda_process_environment(),
    )


def main() -> int:
    args = parse_args()
    _restart_for_cuda_if_needed(args.device)

    from agents.auralis.agent import AuralisAgent
    from agents.auralis.schemas import AuralisInput
    from tools.speech_transcription.backends.faster_whisper import (
        FasterWhisperBackend,
    )
    from tools.speech_transcription.tool import transcribe_speech
    from tools.subtitle_extraction.backends.rapidocr import RapidOCRBackend
    from tools.subtitle_extraction.tool import extract_subtitles

    compute_type = args.compute_type or (
        "int8_float16" if args.device == "cuda" else "int8"
    )
    asr_backend = FasterWhisperBackend(
        model_name=args.asr_model,
        device=args.device,
        compute_type=compute_type,
    )
    ocr_backend = RapidOCRBackend()
    agent = AuralisAgent(
        transcribe_speech=lambda path: transcribe_speech(
            path,
            backend=asr_backend,
        ),
        extract_subtitles=lambda path: extract_subtitles(
            path,
            backend=ocr_backend,
        ),
        local_only=True,
    )
    result = agent.analyze(
        AuralisInput(
            video_path=args.video.resolve(),
            user_prompt=args.prompt,
        )
    )
    print(json.dumps(asdict(result), ensure_ascii=False, indent=2, default=str))
    if asr_backend.fallback_reason:
        print(
            "CUDA fallback reason: " + asr_backend.fallback_reason,
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
