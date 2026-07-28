#!/usr/bin/env python3
"""Compatibility entry point for the Auralis audio specialist agent."""

import base64
import json
import time
import urllib
from typing import Any, Dict

import call_ffmpeg_skill as gpt_a
from agents.auralis.gemini_backend import (
    AUDIO_SEGMENT_SECONDS,
    DEFAULT_MODEL,
    OUTPUT_KEYS,
    SYSTEM_MESSAGE,
    VIDEO_FRAME_FPS,
    VIDEO_FRAME_WIDTH,
    build_chat_payload,
    build_prompt,
    build_user_content,
    chat_completion,
    parse_prediction,
    split_wav_bytes,
)
from agents.auralis.runner import (
    API_KEY_ENV,
    BASE_DIR,
    DEFAULT_API_URL,
    DEFAULT_GPT_A_MODEL,
    INFERENCE_COLUMNS,
    INPUT_CSV,
    OUTPUT_CSV,
    PREDICTION_COLUMN,
    SOURCE_COLUMNS,
    inference_input,
    main,
    merge_predictions,
    parse_args,
    read_matching_predictions,
    run_audio_row,
)

__all__ = [
    "API_KEY_ENV",
    "AUDIO_SEGMENT_SECONDS",
    "BASE_DIR",
    "DEFAULT_API_URL",
    "DEFAULT_GPT_A_MODEL",
    "DEFAULT_MODEL",
    "INFERENCE_COLUMNS",
    "INPUT_CSV",
    "OUTPUT_CSV",
    "OUTPUT_KEYS",
    "PREDICTION_COLUMN",
    "SOURCE_COLUMNS",
    "SYSTEM_MESSAGE",
    "VIDEO_FRAME_FPS",
    "VIDEO_FRAME_WIDTH",
    "build_chat_payload",
    "build_prompt",
    "build_user_content",
    "base64",
    "chat_completion",
    "gpt_a",
    "inference_input",
    "main",
    "merge_predictions",
    "parse_args",
    "parse_prediction",
    "read_matching_predictions",
    "run_audio_row",
    "run_combined_row",
    "split_wav_bytes",
    "time",
    "urllib",
]


def run_combined_row(
    gpt_a_input: Dict[str, Any],
    audio_input: Dict[str, Any],
    *,
    api_url: str,
    api_key: str,
    gpt_a_model: str,
    gemini_model: str,
    timeout: int,
    api_retries: int,
    max_gpt_a_agent_steps: int,
    run_stats: Dict[str, Any] | None = None,
) -> str:
    """Backward-compatible wrapper whose two calls remain independently patchable."""
    gpt_a_stats: Dict[str, Any] = {}
    gpt_a_prediction = ""
    gpt_a_error: Exception | None = None
    try:
        gpt_a_prediction = gpt_a.run_agent(
            gpt_a_input,
            api_url,
            api_key,
            gpt_a_model,
            timeout,
            api_retries,
            max_gpt_a_agent_steps,
            VIDEO_FRAME_FPS,
            VIDEO_FRAME_WIDTH,
            0,
            "none",
            None,
            None,
            False,
            gpt_a_stats,
        )
    except Exception as exc:
        gpt_a_error = exc
    if run_stats is not None:
        run_stats["gpt_a"] = gpt_a_stats
    auralis_stats: Dict[str, Any] = {}
    if run_stats is not None:
        run_stats["gemini_audio"] = auralis_stats
        run_stats["auralis_audio"] = auralis_stats
    audio_prediction = ""
    auralis_error: Exception | None = None
    try:
        audio_prediction = run_audio_row(
            audio_input,
            api_url=api_url,
            api_key=api_key,
            model=gemini_model,
            timeout=timeout,
            api_retries=api_retries,
            run_stats=auralis_stats,
        )
    except Exception as exc:
        auralis_error = exc
    if run_stats is not None:
        run_stats["api_calls"] = int(gpt_a_stats.get("api_calls", 0)) + int(
            auralis_stats.get("api_calls", 0)
        )
        run_stats["request_bytes"] = int(
            gpt_a_stats.get("request_bytes", 0)
        ) + int(auralis_stats.get("request_bytes", 0))
    if gpt_a_error is not None and auralis_error is not None:
        raise RuntimeError(
            f"GPT-A 失败：{gpt_a_error}；Auralis 失败：{auralis_error}"
        ) from gpt_a_error
    if gpt_a_error is not None:
        raise gpt_a_error
    if auralis_error is not None:
        raise auralis_error
    return merge_predictions(gpt_a_prediction, audio_prediction)


if __name__ == "__main__":
    raise SystemExit(main())
