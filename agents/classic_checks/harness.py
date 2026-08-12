"""Internal harness for behavior-preserving AVAgent classic checks.

The production runner still executes its established specialist stages in the
established order.  This module turns those already-computed stage results into
the shared context consumed by the ten issue-oriented check functions.  A
standalone caller may instead register lazy tools directly on
``EvaluationContext`` and call an individual check.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from agents.classic_checks.checks import evaluate_all_classic_checks
from agents.classic_checks.context import EvaluationContext
from agents.classic_checks.contracts import (
    EvaluationResult,
    EvaluationSample,
    ToolResult,
)


TOOL_ORDER = (
    "metadata",
    "gpt_a",
    "seed_lite",
    "auralis",
    "avbench",
    "ocr_visual_verifier",
)


def _reference_images(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            decoded = ()
        value = decoded
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(str(item) for item in value if str(item).strip())


def build_evaluation_sample(
    gpt_a_input: Mapping[str, Any],
    audio_input: Mapping[str, Any],
) -> EvaluationSample:
    """Build the only sample payload visible to the functionized check layer.

    Both inputs are already inference-safe dictionaries.  Fields outside this
    explicit allow-list are deliberately ignored, so a caller cannot leak a
    standard answer by attaching it to either source mapping.
    """

    sample_id = str(
        gpt_a_input.get("序号") or audio_input.get("序号") or ""
    )
    prompt = str(
        gpt_a_input.get("user_prompt")
        or audio_input.get("user_prompt")
        or ""
    )
    references = _reference_images(
        gpt_a_input.get("reference_image_urls")
        or audio_input.get("reference_image_urls")
        or ()
    )
    video_value = str(
        gpt_a_input.get("generated_video_url")
        or audio_input.get("generated_video_url")
        or ""
    )
    return EvaluationSample(
        sample_id=sample_id,
        prompt=prompt,
        reference_images=references,
        video_path=Path(video_value),
        feedback=str(gpt_a_input.get("用户反馈") or ""),
    )


def evaluate_precomputed_tools(
    sample: EvaluationSample,
    stage_results: Mapping[str, ToolResult],
    *,
    final_issues: Sequence[Mapping[str, Any]] = (),
    compatibility_log: Mapping[str, Any] | None = None,
) -> tuple[EvaluationResult, list[dict[str, Any]]]:
    """Run all ten checks over one set of precomputed production tools.

    Full AVAgent mode passes every stage, including explicit
    ``not_applicable`` or ``failed`` results.  Enforcing that invariant keeps a
    missing integration hook distinguishable from an intentionally skipped
    tool.
    """

    missing = [name for name in TOOL_ORDER if name not in stage_results]
    unexpected = [name for name in stage_results if name not in TOOL_ORDER]
    if missing or unexpected:
        details = []
        if missing:
            details.append("missing=" + ",".join(missing))
        if unexpected:
            details.append("unexpected=" + ",".join(unexpected))
        raise ValueError("invalid classic-check stage results: " + "; ".join(details))

    context = EvaluationContext(sample)
    for tool_name in TOOL_ORDER:
        tool_result = stage_results[tool_name]
        if not isinstance(tool_result, ToolResult):
            raise TypeError(f"{tool_name} must be a ToolResult")
        context.preload_tool_result(tool_name, tool_result)

    checks = evaluate_all_classic_checks(context)
    call_stats = context.call_stats
    serialized_tools = [
        {
            "tool_name": tool_name,
            **stage_results[tool_name].to_dict(),
            "cache": call_stats.get(tool_name, {}),
        }
        for tool_name in TOOL_ORDER
    ]
    result = EvaluationResult(
        checks=checks,
        final_issues=tuple(dict(issue) for issue in final_issues),
        tool_trace=context.tool_trace,
        compatibility_log={
            **dict(compatibility_log or {}),
            "classic_check_tool_calls": call_stats,
        },
    )
    return result, serialized_tools
