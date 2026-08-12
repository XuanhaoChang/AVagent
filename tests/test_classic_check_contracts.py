import json
import unittest
from dataclasses import fields
from pathlib import Path

from agents.classic_checks import (
    ClassicCheckResult,
    EvaluationContext,
    EvaluationResult,
    EvaluationSample,
    ToolResult,
)


def sample() -> EvaluationSample:
    return EvaluationSample(
        sample_id="sample-007",
        prompt="人物说你好",
        reference_images=("reference_01.jpg",),
        video_path=Path("video.mp4"),
        feedback="请检查台词",
    )


def check(index: int) -> ClassicCheckResult:
    return ClassicCheckResult(
        check_name=f"classic_check_{index:02d}",
        execution_status="ok",
        decision="not_detected",
        evidence_level="supported",
        issues=(),
        tool_refs=("probe_media",),
        limitations=("no issue observed",),
    )


class ClassicCheckContractTest(unittest.TestCase):
    def test_evaluation_sample_has_only_inference_safe_fields(self):
        contract = sample()

        self.assertEqual(
            [field.name for field in fields(contract)],
            [
                "sample_id",
                "prompt",
                "reference_images",
                "video_path",
                "feedback",
            ],
        )
        payload = contract.to_dict()
        self.assertEqual(payload["video_path"], "video.mp4")
        self.assertEqual(payload["reference_images"], ["reference_01.jpg"])
        self.assertNotIn("思考过程及标准答案", payload)
        self.assertNotIn("gold", " ".join(payload).casefold())

        repeated_references = EvaluationSample(
            sample_id="sample-008",
            prompt="图1和图2",
            reference_images=("same.jpg", "same.jpg"),
            video_path=Path("video.mp4"),
            feedback="",
        )
        self.assertEqual(
            repeated_references.reference_images,
            ("same.jpg", "same.jpg"),
        )

    def test_contract_json_is_stable_and_rejects_non_finite_values(self):
        first = ToolResult.ok(
            evidence={"z": 1, "a": {"right": 2, "left": 1}},
            diagnostics={"path": Path("clip.wav")},
        )
        second = ToolResult.ok(
            diagnostics={"path": "clip.wav"},
            evidence={"a": {"left": 1, "right": 2}, "z": 1},
        )

        self.assertEqual(first.to_json(), second.to_json())
        self.assertEqual(json.loads(first.to_json()), first.to_dict())
        with self.assertRaisesRegex(ValueError, "non-finite"):
            ToolResult.ok(evidence={"score": float("nan")})

    def test_status_decision_and_evidence_level_are_controlled(self):
        for status in ("ok", "not_applicable", "failed"):
            result = ToolResult(
                status=status,
                evidence={},
                artifacts={},
                diagnostics={},
                error="failure" if status == "failed" else "",
                usage={},
            )
            self.assertEqual(result.status, status)
        with self.assertRaisesRegex(ValueError, "ToolResult.status"):
            ToolResult(
                status="skipped",
                evidence={},
                artifacts={},
                diagnostics={},
                error="",
                usage={},
            )
        with self.assertRaisesRegex(ValueError, "decision"):
            ClassicCheckResult(
                check_name="bad",
                execution_status="ok",
                decision="maybe",
                evidence_level="none",
                issues=(),
                tool_refs=(),
                limitations=(),
            )
        with self.assertRaisesRegex(ValueError, "evidence_level"):
            ClassicCheckResult(
                check_name="bad",
                execution_status="ok",
                decision="not_evaluable",
                evidence_level="high",
                issues=(),
                tool_refs=(),
                limitations=(),
            )

    def test_evaluation_result_requires_ten_unique_checks(self):
        checks = tuple(check(index) for index in range(10))
        result = EvaluationResult(
            checks=checks,
            final_issues=(),
            tool_trace=(),
            compatibility_log={"adapter": "legacy_json_string"},
        )

        self.assertEqual(len(result.to_dict()["checks"]), 10)
        self.assertEqual(
            result.to_dict()["compatibility_log"]["adapter"],
            "legacy_json_string",
        )
        with self.assertRaisesRegex(ValueError, "exactly 10"):
            EvaluationResult(
                checks=checks[:-1],
                final_issues=(),
                tool_trace=(),
                compatibility_log={},
            )


class EvaluationContextTest(unittest.TestCase):
    def test_equivalent_parameters_share_one_lazy_execution(self):
        calls = []

        def tool(**parameters):
            calls.append(parameters)
            return ToolResult.ok(evidence={"frame_count": 12})

        context = EvaluationContext(sample(), {"dense_frames": tool})
        first = context.run_tool(
            "dense_frames",
            video_path=Path("video.mp4"),
            timestamps=[1, 2],
            options={"width": 384, "fps": 2.0},
        )
        second = context.run_tool(
            "dense_frames",
            options={"fps": 2.0, "width": 384},
            timestamps=(1, 2),
            video_path="video.mp4",
        )

        self.assertIs(first, second)
        self.assertEqual(len(calls), 1)
        self.assertEqual(
            [entry.cache_status for entry in context.tool_trace],
            ["miss", "hit"],
        )
        self.assertEqual(
            context.call_stats["dense_frames"],
            {
                "requests": 2,
                "executions": 1,
                "cache_hits": 1,
                "cache_misses": 1,
                "preloaded": 0,
                "ok_returns": 2,
                "not_applicable_returns": 0,
                "failed_returns": 0,
            },
        )

    def test_different_parameters_execute_independently(self):
        execution_count = 0

        def tool(**_parameters):
            nonlocal execution_count
            execution_count += 1
            return ToolResult.ok(evidence={"execution": execution_count})

        context = EvaluationContext(sample(), {"extract_frame": tool})
        context.run_tool("extract_frame", timestamp_sec=1.0)
        context.run_tool("extract_frame", timestamp_sec=2.0)

        self.assertEqual(execution_count, 2)
        self.assertEqual(context.cached_result_count, 2)

    def test_preloaded_result_is_a_cache_hit_without_registered_tool(self):
        context = EvaluationContext(sample())
        preloaded = ToolResult.ok(
            evidence={"duration_sec": 4.2},
            artifacts={"source": "existing_run_stats"},
        )
        cache_key = context.preload_tool_result(
            "probe_media",
            preloaded,
            video_path=Path("video.mp4"),
        )

        result = context.run_tool("probe_media", video_path="video.mp4")

        self.assertIs(result, preloaded)
        self.assertEqual(context.tool_trace[0].cache_key, cache_key)
        self.assertEqual(context.tool_trace[0].cache_status, "hit")
        self.assertEqual(context.tool_trace[0].source, "preloaded")
        self.assertEqual(context.call_stats["probe_media"]["executions"], 0)
        self.assertEqual(context.call_stats["probe_media"]["preloaded"], 1)

    def test_tool_failure_is_converted_and_cached(self):
        executions = 0

        def failing_tool(**_parameters):
            nonlocal executions
            executions += 1
            raise RuntimeError("decoder failed")

        context = EvaluationContext(sample(), {"decode": failing_tool})
        first = context.run_tool("decode", stream="audio")
        second = context.run_tool("decode", stream="audio")

        self.assertEqual(executions, 1)
        self.assertIs(first, second)
        self.assertEqual(first.status, "failed")
        self.assertIn("decoder failed", first.error)
        self.assertEqual(
            [entry.cache_status for entry in context.tool_trace],
            ["miss", "hit"],
        )
        self.assertEqual(context.call_stats["decode"]["failed_returns"], 2)

    def test_unregistered_tool_returns_auditable_failed_result(self):
        context = EvaluationContext(sample())

        result = context.run_tool("missing_tool", value=1)

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.diagnostics["reason"], "tool_not_registered")
        self.assertEqual(context.call_stats["missing_tool"]["executions"], 0)
        self.assertEqual(context.tool_trace[0].cache_status, "miss")


if __name__ == "__main__":
    unittest.main()
