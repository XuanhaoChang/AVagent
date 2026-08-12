import json
import unittest
from unittest import mock

from agents.auralis import runner
from agents.classic_checks.contracts import ToolResult
from agents.classic_checks.harness import (
    TOOL_ORDER,
    build_evaluation_sample,
    evaluate_precomputed_tools,
)


def issue(problem_type: str, description: str) -> dict[str, str]:
    return {
        "可定位性": "否",
        "置信度": "高",
        "问题说明": description,
        "问题类型": problem_type,
        "时间区间": "0.00s - 1.00s",
        "关键帧秒": "",
        "BBox": "",
    }


class ClassicHarnessTest(unittest.TestCase):
    def test_sample_builder_cannot_copy_standard_answer(self):
        secret = "STANDARD_ANSWER_MUST_NOT_LEAK"
        sample = build_evaluation_sample(
            {
                "序号": "#9",
                "user_prompt": "人物说你好",
                "reference_image_urls": ["ref.jpg"],
                "generated_video_url": "video.mp4",
                "用户反馈": "检查说话人",
                "思考过程及标准答案": secret,
            },
            {
                "序号": "#9",
                "user_prompt": "人物说你好",
                "reference_image_urls": ["ref.jpg"],
                "generated_video_url": "video.mp4",
                "思考过程及标准答案": secret,
            },
        )

        payload = sample.to_dict()
        self.assertEqual(
            set(payload),
            {
                "sample_id",
                "prompt",
                "reference_images",
                "video_path",
                "feedback",
            },
        )
        self.assertNotIn(secret, json.dumps(payload, ensure_ascii=False))

    def test_precomputed_harness_returns_ten_checks_and_six_tools(self):
        sample = build_evaluation_sample(
            {
                "序号": "#1",
                "user_prompt": "人物站立",
                "reference_image_urls": ["ref.jpg"],
                "generated_video_url": "video.mp4",
                "用户反馈": "",
            },
            {},
        )
        stage_results = {
            name: ToolResult.ok(
                artifacts={
                    "issues": [],
                    **(
                        {
                            "result": {
                                "success": True,
                                "status": "ok",
                                "sync_decision": "aligned_or_no_large_offset",
                            }
                        }
                        if name == "avbench"
                        else {}
                    ),
                }
            )
            for name in TOOL_ORDER
        }

        evaluation, tool_results = evaluate_precomputed_tools(
            sample,
            stage_results,
            final_issues=(),
        )

        self.assertEqual(len(evaluation.checks), 10)
        self.assertEqual(
            [record["tool_name"] for record in tool_results],
            list(TOOL_ORDER),
        )
        self.assertTrue(
            all(record["cache"]["preloaded"] == 1 for record in tool_results)
        )
        call_stats = evaluation.compatibility_log["classic_check_tool_calls"]
        self.assertTrue(call_stats)
        self.assertTrue(
            all(stats["executions"] == 0 for stats in call_stats.values())
        )
        self.assertTrue(
            all(stats["preloaded"] == 1 for stats in call_stats.values())
        )

    def test_production_runner_preserves_stage_order_and_final_json(self):
        calls: list[str] = []
        visual_issue = issue("动作异常", "人物动作连续性出现跳变")
        audio_issue = issue("音频质量问题", "人物把台词“你好”错读为“你早”")
        seed_issue = issue("文字质量问题", "画面出现禁止的品牌 logo")
        final_prediction = json.dumps(
            [visual_issue, audio_issue, seed_issue], ensure_ascii=False
        )

        def metadata(*_args, **_kwargs):
            calls.append("metadata")
            return []

        def gpt_a(*_args, **_kwargs):
            calls.append("gpt_a")
            return json.dumps([visual_issue], ensure_ascii=False)

        def seed(*_args, **_kwargs):
            calls.append("seed_lite")
            return json.dumps([seed_issue], ensure_ascii=False)

        def auralis(*_args, **_kwargs):
            calls.append("auralis")
            return json.dumps([audio_issue], ensure_ascii=False)

        def avbench(*_args, **_kwargs):
            calls.append("avbench")
            return {
                "success": True,
                "status": "ok",
                "sync_decision": "aligned_or_no_large_offset",
            }

        def ocr_gate(prediction, **_kwargs):
            calls.append("ocr_visual_verifier")
            return prediction

        def synthesis(**_kwargs):
            calls.append("final_synthesis")
            return final_prediction

        stats: dict[str, object] = {}
        with (
            mock.patch.object(
                runner,
                "evaluate_visual_metadata_constraints",
                side_effect=metadata,
            ),
            mock.patch.object(runner.gpt_a, "run_agent", side_effect=gpt_a),
            mock.patch.object(
                runner, "run_seed_lite_specialist", side_effect=seed
            ),
            mock.patch.object(runner, "run_audio_row", side_effect=auralis),
            mock.patch.object(runner, "run_avbench_row", side_effect=avbench),
            mock.patch.object(
                runner, "gate_auralis_ocr_prediction", side_effect=ocr_gate
            ),
            mock.patch.object(
                runner, "synthesize_predictions", side_effect=synthesis
            ),
        ):
            actual = runner.run_combined_row(
                {
                    "序号": "#1",
                    "user_prompt": "人物说你好并保持动作连续，无 logo",
                    "reference_image_urls": ["ref.jpg"],
                    "generated_video_url": "video.mp4",
                    "用户反馈": "检查动作",
                },
                {
                    "序号": "#1",
                    "user_prompt": "人物说你好并保持动作连续，无 logo",
                    "reference_image_urls": ["ref.jpg"],
                    "generated_video_url": "video.mp4",
                },
                api_url="https://example.test/chat/completions",
                api_key="token",
                gpt_a_model="gpt",
                gemini_model="gemini",
                timeout=1,
                api_retries=1,
                max_gpt_a_agent_steps=2,
                run_stats=stats,
                seed_lite_model="seed",
            )

        self.assertEqual(actual, final_prediction)
        self.assertEqual(
            calls,
            [
                "metadata",
                "gpt_a",
                "seed_lite",
                "auralis",
                "avbench",
                "ocr_visual_verifier",
                "final_synthesis",
            ],
        )
        self.assertEqual(len(stats["classic_checks"]), 10)
        self.assertEqual(
            [record["tool_name"] for record in stats["tool_results"]],
            list(TOOL_ORDER),
        )
        self.assertTrue(
            all(
                record["cache"]["executions"] == 0
                and record["cache"]["preloaded"] == 1
                for record in stats["tool_results"]
            )
        )
        self.assertEqual(stats["final_prediction"], json.loads(final_prediction))
        json.dumps(stats, ensure_ascii=False)


if __name__ == "__main__":
    unittest.main()
