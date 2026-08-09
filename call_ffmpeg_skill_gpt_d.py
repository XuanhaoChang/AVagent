#!/usr/bin/env python3
"""Compatibility entry point for the Auralis audio specialist agent."""

import base64
import json
import time
import urllib
from typing import Any, Dict

import call_ffmpeg_skill as gpt_a
from agents.auralis.gemini_backend import (
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
)
from agents.auralis.runner import (
    API_KEY_ENV,
    BASE_DIR,
    DEFAULT_API_URL,
    DEFAULT_GPT_A_MODEL,
    FINAL_SYNTHESIS_SYSTEM_MESSAGE,
    INFERENCE_COLUMNS,
    INPUT_CSV,
    OUTPUT_CSV,
    PREDICTION_COLUMN,
    SOURCE_COLUMNS,
    inference_input,
    main,
    merge_predictions,
    parse_args,
    preserve_deterministic_issues,
    read_matching_predictions,
    run_audio_row,
    build_synthesis_fact_registry,
    build_synthesis_prompt,
    build_avbench_runner,
    deduplicate_prediction_issues,
    evaluate_visual_metadata_constraints,
    extract_visual_metadata_constraints,
    final_chat_completion,
    gate_auralis_ocr_prediction,
    metadata_constraint_issues,
    preserve_synthesis_fact_coverage,
    run_avbench_row,
    select_evidence_backed_gpt_a_issues,
    synthesize_predictions,
)
from agents.seed_lite_specialist import (
    DEFAULT_SEED_LITE_MODEL,
    filter_seed_lite_candidates,
    run_seed_lite_specialist,
)

__all__ = [
    "API_KEY_ENV",
    "BASE_DIR",
    "DEFAULT_API_URL",
    "DEFAULT_GPT_A_MODEL",
    "DEFAULT_MODEL",
    "DEFAULT_SEED_LITE_MODEL",
    "FINAL_SYNTHESIS_SYSTEM_MESSAGE",
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
    "build_avbench_runner",
    "build_prompt",
    "build_synthesis_fact_registry",
    "build_synthesis_prompt",
    "build_user_content",
    "deduplicate_prediction_issues",
    "evaluate_visual_metadata_constraints",
    "extract_visual_metadata_constraints",
    "base64",
    "chat_completion",
    "final_chat_completion",
    "gate_auralis_ocr_prediction",
    "filter_seed_lite_candidates",
    "gpt_a",
    "inference_input",
    "main",
    "merge_predictions",
    "metadata_constraint_issues",
    "parse_args",
    "parse_prediction",
    "preserve_deterministic_issues",
    "preserve_synthesis_fact_coverage",
    "read_matching_predictions",
    "run_audio_row",
    "run_avbench_row",
    "run_seed_lite_specialist",
    "select_evidence_backed_gpt_a_issues",
    "synthesize_predictions",
    "run_combined_row",
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
    avbench_runner: Any | None = None,
) -> str:
    """Backward-compatible wrapper whose two calls remain independently patchable."""
    metadata_issues = evaluate_visual_metadata_constraints(
        gpt_a_input,
        run_stats=run_stats,
    )
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
            True,
            gpt_a_stats,
        )
    except Exception as exc:
        gpt_a_error = exc
    if gpt_a_error is None:
        raw_gpt_a_issues = json.loads(gpt_a_prediction)
        gpt_a_stats["raw_prediction"] = raw_gpt_a_issues
        gpt_a_stats["evidence_backed_visual_issues"] = (
            select_evidence_backed_gpt_a_issues(gpt_a_prediction, gpt_a_stats)
        )
    if run_stats is not None:
        run_stats["gpt_a"] = gpt_a_stats
    auralis_stats: Dict[str, Any] = {}
    avbench_stats: Dict[str, Any] = {}
    if run_stats is not None:
        run_stats["gemini_audio"] = auralis_stats
        run_stats["auralis_audio"] = auralis_stats
        run_stats["avbench"] = avbench_stats
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
    avbench_result: Dict[str, Any] = {}
    avbench_error: Exception | None = None
    try:
        avbench_result = run_avbench_row(
            audio_input,
            avbench_runner=avbench_runner,
            run_stats=avbench_stats,
        )
    except Exception as exc:
        avbench_error = exc
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
    if avbench_error is not None:
        raise avbench_error
    audio_prediction = gate_auralis_ocr_prediction(
        audio_prediction,
        input_data=audio_input,
        auralis_stats=auralis_stats,
        api_url=api_url,
        api_key=api_key,
        model=gpt_a_model,
        timeout=timeout,
        api_retries=api_retries,
        run_stats=run_stats,
    )
    ocr_visual_stats = (
        run_stats.get("ocr_visual_verifier", {})
        if isinstance(run_stats, dict)
        else {}
    )
    deduplicated_audio_prediction = deduplicate_prediction_issues(
        audio_prediction,
        "Auralis 音频",
    )
    auralis_stats["deduplicated_issues"] = json.loads(
        deduplicated_audio_prediction
    )
    # The final GPT only organizes the specialist outputs.  Preserve their
    # complete union in code so it cannot silently drop an independent issue.
    required_issues = tuple(
        issue
        for issue in (
            *metadata_issues,
            *gpt_a_stats.get("raw_prediction", ()),
            *auralis_stats.get("deduplicated_issues", ()),
        )
        if isinstance(issue, dict)
    )
    final_stats: Dict[str, Any] = {}
    if run_stats is not None:
        run_stats["final_synthesis"] = final_stats
    final_stats["required_issues"] = [dict(issue) for issue in required_issues]
    final_prediction = synthesize_predictions(
        user_prompt=str(gpt_a_input.get("user_prompt", "")),
        gpt_a_prediction=gpt_a_prediction,
        auralis_prediction=deduplicated_audio_prediction,
        avbench_result=avbench_result,
        metadata_prediction=json.dumps(metadata_issues, ensure_ascii=False),
        api_url=api_url,
        api_key=api_key,
        model=gpt_a_model,
        timeout=timeout,
        api_retries=api_retries,
        run_stats=final_stats,
        deterministic_issues=required_issues,
    )
    if run_stats is not None:
        run_stats["final_prediction"] = json.loads(final_prediction)
        run_stats["api_calls"] = sum(
            int(stats.get("api_calls", 0))
            for stats in (
                gpt_a_stats,
                auralis_stats,
                ocr_visual_stats,
                final_stats,
            )
        )
        run_stats["request_bytes"] = sum(
            int(stats.get("request_bytes", 0))
            for stats in (
                gpt_a_stats,
                auralis_stats,
                ocr_visual_stats,
                final_stats,
            )
        )
    return final_prediction


if __name__ == "__main__":
    raise SystemExit(main())
