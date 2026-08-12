import unittest
from pathlib import Path

from agents.classic_checks.assignment import (
    AUDIO_QUALITY_NOISE_ARTIFACTS,
    AV_LIP_SYNC,
    CLASSIC_CHECK_NAMES,
    DIALOGUE_SPEAKER_BINDING,
    ENTITY_COUNT_SPATIAL_COMPOSITION,
    MOTION_PHYSICS_CONTINUITY,
    REFERENCE_SUBJECT_IDENTITY,
    REMAINING_HARD_INSTRUCTION_COMPLIANCE,
    SUBTITLE_TEXT_LOGO_WATERMARK,
    VISUAL_QUALITY_TEMPORAL_ARTIFACTS,
    VOICE_CHARACTERISTICS,
    assign_issue,
    issue_fingerprint,
    partition_candidates,
)
from agents.classic_checks.checks import (
    AURALIS_TOOL,
    AVBENCH_TOOL,
    CHECK_TOOL_DEPENDENCIES,
    GPT_A_TOOL,
    METADATA_TOOL,
    OCR_VISUAL_VERIFIER_TOOL,
    SEED_LITE_TOOL,
    check_av_lip_sync,
    check_reference_subject_identity,
    check_voice_characteristics,
    classify_junk_test,
    classify_system_task_failure,
    evaluate_all_classic_checks,
    is_junk_test,
)
from agents.classic_checks.context import EvaluationContext
from agents.classic_checks.contracts import EvaluationSample, ToolResult


def make_sample(
    *,
    prompt="人物面对镜头说你好",
    reference_images=("reference.png",),
):
    return EvaluationSample(
        sample_id="classic-001",
        prompt=prompt,
        reference_images=reference_images,
        video_path=Path("generated.mp4"),
        feedback="",
    )


def make_issue(problem_type, description):
    return {
        "可定位性": "是",
        "置信度": "高",
        "问题说明": description,
        "问题类型": problem_type,
        "时间区间": "1.0-2.0",
        "关键帧秒": "1.5",
        "BBox": "[0.1, 0.1, 0.5, 0.5]",
    }


def tool_returning(result):
    def tool(**_parameters):
        return result

    return tool


class IssueAssignmentTest(unittest.TestCase):
    def test_assignment_uses_specific_semantics_and_provenance(self):
        cases = (
            (
                make_issue("清晰度问题", "ffprobe 检测到分辨率不满足提示词"),
                {"source": "metadata"},
                REMAINING_HARD_INSTRUCTION_COMPLIANCE,
            ),
            (
                make_issue("时长错误", "ffprobe 检测到视频时长不满足提示词"),
                {"source": "metadata"},
                REMAINING_HARD_INSTRUCTION_COMPLIANCE,
            ),
            (
                make_issue("角色属性错误", "角色服装与参考图中的人物不一致"),
                {"source": "gpt_a"},
                REFERENCE_SUBJECT_IDENTITY,
            ),
            (
                make_issue("场景错误", "生成画面的场景还原度与参考场景很差"),
                {"source": "gpt_a"},
                REFERENCE_SUBJECT_IDENTITY,
            ),
            (
                make_issue("角色属性错误", "提示词要求红衣，实际角色穿蓝衣"),
                {"source": "gpt_a"},
                REMAINING_HARD_INSTRUCTION_COMPLIANCE,
            ),
            (
                make_issue("动态结构异常", "手臂在移动中发生结构崩坏"),
                {"source": "seed_lite"},
                MOTION_PHYSICS_CONTINUITY,
            ),
            (
                make_issue("文字质量问题", "画面出现错误字幕和 logo"),
                {"source": "auralis"},
                SUBTITLE_TEXT_LOGO_WATERMARK,
            ),
            (
                make_issue("台词错误", "ASR/CTC 显示说话人漏读台词"),
                {"source": "auralis"},
                DIALOGUE_SPEAKER_BINDING,
            ),
            (
                make_issue("音色错误", "音色参考男声，实际输出女声"),
                {"source": "auralis"},
                VOICE_CHARACTERISTICS,
            ),
            (
                make_issue("音频质量问题", "音轨有持续电流声和底噪"),
                {"source": "auralis"},
                AUDIO_QUALITY_NOISE_ARTIFACTS,
            ),
            (
                make_issue("实体数量错误", "画面多出一个重复人物"),
                {"source": "gpt_a"},
                ENTITY_COUNT_SPATIAL_COMPOSITION,
            ),
            (
                make_issue("稳定性异常", "画面连续闪烁并出现色块"),
                {"source": "gpt_a"},
                VISUAL_QUALITY_TEMPORAL_ARTIFACTS,
            ),
            (
                make_issue("音频质量问题", "口型同步偏移超过阈值"),
                {"source": "avbench"},
                AV_LIP_SYNC,
            ),
        )

        for issue, provenance, expected in cases:
            with self.subTest(expected=expected, description=issue["问题说明"]):
                self.assertEqual(assign_issue(issue, provenance), expected)

    def test_partition_assigns_every_candidate_exactly_once(self):
        candidates = [
            {
                "issue": make_issue("人物身份错误", "参考人物发生身份漂移"),
                "provenance": {"source": "gpt_a"},
            },
            {
                "issue": make_issue("动作异常", "角色动作发生瞬移"),
                "provenance": {"source": "seed_lite"},
            },
            {
                "issue": make_issue("音频质量问题", "结尾存在爆音"),
                "provenance": {"source": "auralis"},
            },
        ]

        partitions = partition_candidates(candidates)
        assigned = [
            item
            for check_name in CLASSIC_CHECK_NAMES
            for item in partitions[check_name]
        ]

        self.assertEqual(len(assigned), len(candidates))
        self.assertEqual(
            sorted(item["candidate_index"] for item in assigned),
            list(range(len(candidates))),
        )
        self.assertEqual(
            len({item["candidate_index"] for item in assigned}),
            len(candidates),
        )


class ClassicCheckCompositionTest(unittest.TestCase):
    def test_each_check_has_explicit_tool_dependencies(self):
        self.assertEqual(
            CHECK_TOOL_DEPENDENCIES,
            {
                REFERENCE_SUBJECT_IDENTITY: (GPT_A_TOOL,),
                ENTITY_COUNT_SPATIAL_COMPOSITION: (GPT_A_TOOL,),
                REMAINING_HARD_INSTRUCTION_COMPLIANCE: (
                    METADATA_TOOL,
                    GPT_A_TOOL,
                ),
                MOTION_PHYSICS_CONTINUITY: (SEED_LITE_TOOL, GPT_A_TOOL),
                DIALOGUE_SPEAKER_BINDING: (AURALIS_TOOL,),
                VOICE_CHARACTERISTICS: (AURALIS_TOOL,),
                SUBTITLE_TEXT_LOGO_WATERMARK: (
                    AURALIS_TOOL,
                    OCR_VISUAL_VERIFIER_TOOL,
                    SEED_LITE_TOOL,
                    GPT_A_TOOL,
                ),
                AV_LIP_SYNC: (AVBENCH_TOOL,),
                VISUAL_QUALITY_TEMPORAL_ARTIFACTS: (
                    METADATA_TOOL,
                    GPT_A_TOOL,
                    SEED_LITE_TOOL,
                ),
                AUDIO_QUALITY_NOISE_ARTIFACTS: (AURALIS_TOOL,),
            },
        )

    def test_all_ten_checks_compose_tools_in_stable_order_without_duplication(self):
        identity = make_issue("人物身份错误", "人物身份与参考图中的角色不一致")
        entity = make_issue("实体数量错误", "画面多出一个重复人物")
        remaining = make_issue("风格错误", "提示词要求赛博朋克，实际为写实风格")
        visual = make_issue("稳定性异常", "画面连续闪烁并出现色块")
        resolution = make_issue("清晰度问题", "ffprobe 检测到分辨率不满足提示词")
        motion = make_issue("动态结构异常", "手臂运动时结构崩坏并穿模")
        logo = make_issue("文字质量问题", "画面右上角生成了多余 logo")
        dialogue = make_issue("台词错误", "ASR/CTC 显示说话人漏读台词")
        voice = make_issue("音色错误", "音色参考男声，实际输出女声")
        subtitle = make_issue("文字质量问题", "人物说话时生成了错误字幕")
        noise = make_issue("音频质量问题", "结尾存在明显电流声和底噪")

        results_by_tool = {
            GPT_A_TOOL: ToolResult.ok(
                artifacts={"issues": [identity, entity, remaining, visual]}
            ),
            METADATA_TOOL: ToolResult.ok(artifacts={"issues": [resolution]}),
            SEED_LITE_TOOL: ToolResult.ok(artifacts={"issues": [motion, logo]}),
            AURALIS_TOOL: ToolResult.ok(
                artifacts={"issues": [dialogue, voice, subtitle, noise]}
            ),
            # This repeats an Auralis issue deliberately. OCR is supporting
            # evidence only and must not introduce a duplicate candidate.
            OCR_VISUAL_VERIFIER_TOOL: ToolResult.ok(
                artifacts={"issues": [subtitle]}
            ),
            AVBENCH_TOOL: ToolResult.ok(
                artifacts={
                    "result": {
                        "success": True,
                        "sync_decision": "desync_candidate",
                        "offset_frames": 8,
                        "offset_sec": 0.32,
                        "confidence": 8.5,
                        "confidence_status": "confident",
                        "face_track_count": 1,
                    }
                }
            ),
        }
        context = EvaluationContext(
            make_sample(),
            {
                name: tool_returning(result)
                for name, result in results_by_tool.items()
            },
        )

        results = evaluate_all_classic_checks(context)

        self.assertEqual(
            tuple(result.check_name for result in results),
            CLASSIC_CHECK_NAMES,
        )
        self.assertEqual(len(results), 10)
        self.assertTrue(all(result.decision == "detected" for result in results))

        emitted = [issue for result in results for issue in result.issues]
        fingerprints = [issue_fingerprint(issue) for issue in emitted]
        self.assertEqual(len(fingerprints), len(set(fingerprints)))
        self.assertEqual(len(emitted), 12)

        for tool_name in results_by_tool:
            with self.subTest(tool=tool_name):
                self.assertEqual(
                    context.call_stats[tool_name]["executions"],
                    1,
                )
        self.assertGreater(context.call_stats[GPT_A_TOOL]["cache_hits"], 0)
        self.assertGreater(context.call_stats[AURALIS_TOOL]["cache_hits"], 0)

    def test_reference_check_calls_tool_but_is_not_evaluable_without_reference(self):
        impossible_reference_claim = make_issue(
            "人物身份错误",
            "人物身份与参考图中的角色不一致",
        )
        context = EvaluationContext(
            make_sample(reference_images=()),
            {
                GPT_A_TOOL: tool_returning(
                    ToolResult.ok(artifacts={"issues": [impossible_reference_claim]})
                )
            },
        )

        result = check_reference_subject_identity(context)

        self.assertEqual(result.execution_status, "ok")
        self.assertEqual(result.decision, "not_evaluable")
        self.assertEqual(result.evidence_level, "none")
        self.assertEqual(result.issues, ())
        self.assertIn(GPT_A_TOOL, result.tool_refs)
        self.assertEqual(context.call_stats[GPT_A_TOOL]["executions"], 1)

    def test_failed_dependency_is_auditable_not_evaluable(self):
        context = EvaluationContext(
            make_sample(),
            {GPT_A_TOOL: tool_returning(ToolResult.failed("upstream timeout"))},
        )

        result = check_reference_subject_identity(context)

        self.assertEqual(result.execution_status, "failed")
        self.assertEqual(result.decision, "not_evaluable")
        self.assertEqual(result.evidence_level, "none")
        self.assertTrue(any("upstream timeout" in item for item in result.limitations))

    def test_voice_is_not_evaluable_without_explicit_voice_issue_or_reference_audio(self):
        context = EvaluationContext(
            make_sample(prompt="人物说你好"),
            {AURALIS_TOOL: tool_returning(ToolResult.ok(artifacts={"issues": []}))},
        )

        result = check_voice_characteristics(context)

        self.assertEqual(result.execution_status, "ok")
        self.assertEqual(result.decision, "not_evaluable")
        self.assertEqual(result.evidence_level, "none")
        self.assertEqual(result.issues, ())
        self.assertTrue(any("参考音频" in item for item in result.limitations))
        self.assertEqual(context.call_stats[AURALIS_TOOL]["executions"], 1)

    def test_avbench_decision_gate(self):
        cases = (
            (
                {
                    "success": True,
                    "sync_decision": "desync_candidate",
                    "offset_frames": -7,
                    "offset_sec": -0.28,
                    "confidence": 9.0,
                    "confidence_status": "confident",
                    "face_track_count": 1,
                },
                "detected",
                1,
            ),
            (
                {
                    "success": True,
                    "sync_decision": "aligned",
                    "offset_frames": 0,
                    "confidence": 12.0,
                },
                "not_detected",
                0,
            ),
            (
                {
                    "success": True,
                    "sync_decision": "uncertain",
                    "offset_frames": 5,
                    "confidence": 1.2,
                    "confidence_status": "uncertain",
                },
                "not_evaluable",
                0,
            ),
        )

        for raw, expected_decision, issue_count in cases:
            with self.subTest(sync_decision=raw["sync_decision"]):
                context = EvaluationContext(
                    make_sample(),
                    {
                        AVBENCH_TOOL: tool_returning(
                            ToolResult.ok(artifacts={"result": raw})
                        )
                    },
                )

                result = check_av_lip_sync(context)

                self.assertEqual(result.execution_status, "ok")
                self.assertEqual(result.decision, expected_decision)
                self.assertEqual(len(result.issues), issue_count)
                self.assertEqual(context.call_stats[AVBENCH_TOOL]["executions"], 1)
                if expected_decision == "detected":
                    self.assertEqual(result.evidence_level, "supported")
                    self.assertEqual(set(result.issues[0]), set(make_issue("", "")))


class OperationalCategoryTest(unittest.TestCase):
    def test_system_task_failure_is_run_state_not_media_issue(self):
        failed = classify_system_task_failure(
            {"status": "failed", "success": False, "error": "任务创建失败"}
        )
        succeeded = classify_system_task_failure({"status": "success"})

        self.assertTrue(failed["is_failure"])
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["issues"], [])
        self.assertFalse(succeeded["is_failure"])
        self.assertEqual(succeeded["status"], "ok")
        self.assertEqual(succeeded["issues"], [])

    def test_junk_filter_is_narrow_and_never_emits_media_issue(self):
        placeholder = {"prompt": "测试 case 12", "feedback": "占位数据"}
        genuine = {"prompt": "这是一个测试镜头，人物挥手", "feedback": ""}

        self.assertTrue(is_junk_test(placeholder))
        self.assertFalse(is_junk_test(genuine))
        self.assertEqual(classify_junk_test(placeholder)["disposition"], "exclude")
        self.assertEqual(classify_junk_test(placeholder)["issues"], [])
        self.assertEqual(classify_junk_test(genuine)["disposition"], "include")


if __name__ == "__main__":
    unittest.main()
