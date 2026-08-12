import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import scripts.review_text_predictions as review_script
from av_eval.project_env import load_project_env
from av_eval.text_review import (
    PREDICTION_SOURCES,
    build_messages,
    missing_required_materials,
    parse_review_response,
    read_sample,
)
from scripts.review_text_predictions import (
    _read_existing,
    build_parser,
    summarize_results,
)


class ProjectEnvTest(unittest.TestCase):
    def test_loads_project_env_without_overwriting_existing_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            env_path = Path(temporary) / ".env.local"
            env_path.write_text(
                "AVAGENT_API_KEY='from-file'\n"
                "AVAGENT_API_URL=https://example.test/chat/completions\n",
                encoding="utf-8",
            )
            environ = {"AVAGENT_API_KEY": "already-set"}

            load_project_env(env_path, environ=environ)

            self.assertEqual(environ["AVAGENT_API_KEY"], "already-set")
            self.assertEqual(
                environ["AVAGENT_API_URL"],
                "https://example.test/chat/completions",
            )


class TextReviewTest(unittest.TestCase):
    def test_cli_defaults_use_review_package_and_project_output(self):
        args = build_parser().parse_args([])
        self.assertEqual(args.input_root, Path("output/human_review_samples"))
        self.assertEqual(args.output_jsonl, Path("output/text_review/results.jsonl"))
        self.assertEqual(args.model, "")
        self.assertFalse(args.resume)

    def test_messages_include_input_gt_predictions_and_material_audit(self):
        input_data = {
            "序号": "#1",
            "user_prompt": "参考视频1的动作，音频1作为音色参考",
            "reference_image_urls": ["reference.jpg"],
            "generated_video_url": "video.mp4",
            "用户反馈": "动作不一致",
        }
        gt = [{"问题类型": "动作异常", "问题说明": "人物没有抬手"}]
        predictions = {
            source: [{"问题类型": "动作异常", "问题说明": f"{source} 检出"}]
            for source in PREDICTION_SOURCES
        }

        messages = build_messages("#1", input_data, gt, predictions)
        payload = json.loads(messages[1]["content"])
        serialized = json.dumps(payload, ensure_ascii=False)

        self.assertEqual(payload["sample_id"], "#1")
        self.assertEqual(
            set(payload),
            {"sample_id", "input", "material_audit", "gt", "predictions"},
        )
        self.assertEqual(
            payload["input"],
            {
                "序号": "#1",
                "user_prompt": input_data["user_prompt"],
                "provided_materials": {
                    "reference_image_count": 1,
                    "reference_video_provided": False,
                    "reference_audio_provided": False,
                    "generated_video_provided": True,
                },
            },
        )
        self.assertEqual(
            payload["material_audit"]["missing_required_materials"],
            ["参考视频", "参考音频"],
        )
        self.assertIsNone(payload["material_audit"]["force_category"])
        self.assertTrue(
            payload["material_audit"]["missing_materials_are_not_auto_category_5"]
        )
        self.assertIn("user_prompt", serialized)
        self.assertNotIn("video.mp4", serialized)
        self.assertNotIn("动作不一致", serialized)
        self.assertIn("不能\n  自动把任何 prediction_source 归为类别 5", messages[0]["content"])
        self.assertIn("缺失素材与某个 GT 问题无关", messages[0]["content"])

    def test_read_sample_requires_and_returns_input_json(self):
        with tempfile.TemporaryDirectory() as temporary:
            sample_dir = Path(temporary) / "sample_001"
            sample_dir.mkdir()
            input_data = {
                "序号": "#1",
                "user_prompt": "生成视频",
                "reference_image_urls": [],
                "generated_video_url": "video.mp4",
                "用户反馈": "",
            }
            (sample_dir / "input.json").write_text(
                json.dumps(input_data, ensure_ascii=False),
                encoding="utf-8",
            )
            (sample_dir / "gt.json").write_text("[]", encoding="utf-8")
            (sample_dir / "avagent.json").write_text("[]", encoding="utf-8")

            actual_input, gt, predictions = read_sample(
                sample_dir,
                ("avagent",),
            )

        self.assertEqual(actual_input, input_data)
        self.assertEqual(gt, [])
        self.assertEqual(predictions, {"avagent": []})

    def test_missing_required_materials_distinguishes_references_from_output_media(self):
        missing = missing_required_materials(
            {
                "user_prompt": (
                    "参考视频1的动作，使用音频1的音色；"
                    "生成的视频需要保留环境音频。"
                ),
                "reference_image_urls": ["image.jpg"],
                "generated_video_url": "generated.mp4",
            }
        )
        self.assertEqual(missing, ("参考视频", "参考音频"))
        self.assertEqual(
            missing_required_materials(
                {
                    "user_prompt": (
                        "按照视频1中的动作生成，根据影片2进行修改，"
                        "按照音频1的音色配音，根据声音素材2来配音"
                    ),
                    "generated_video_url": "generated.mp4",
                }
            ),
            ("参考视频", "参考音频"),
        )

        complete = missing_required_materials(
            {
                "user_prompt": "参考视频1的动作，使用参考音频的音色",
                "reference_video_urls": ["reference.mp4"],
                "reference_audio_urls": ["reference.wav"],
                "generated_video_url": "generated.mp4",
            }
        )
        self.assertEqual(complete, ())

        output_only = missing_required_materials(
            {
                "user_prompt": "生成的视频不要有背景音频",
                "generated_video_url": "generated.mp4",
            }
        )
        self.assertEqual(output_only, ())
        self.assertEqual(
            missing_required_materials(
                {
                    "user_prompt": (
                        "生成视频1应保持稳定，让@小明的音色低沉自然；"
                        "输出视频1不能切镜，导出的音频1不要有杂音。"
                    ),
                    "generated_video_url": "generated.mp4",
                }
            ),
            (),
        )

    def test_missing_required_materials_does_not_force_every_source_to_category_five(self):
        result = {
            "sample_id": "#1",
            "reviews": [
                {
                    "prediction_source": source,
                    "category": 1,
                    "category_name": "完整指出了GT的问题",
                    "reason": "原分类",
                    "gt_coverage": [],
                    "extra_prediction_indices": [1],
                    "confidence": "中",
                }
                for source in ("gpt_a", "avagent")
            ],
        }
        self.assertEqual(
            [review["category"] for review in result["reviews"]],
            [1, 1],
        )
        self.assertIn("missing_materials_are_not_auto_category_5", build_messages(
            "#1",
            {
                "user_prompt": "使用参考音频1的音色",
                "generated_video_url": "generated.mp4",
            },
            [{"问题说明": "动作不一致"}],
            {source: [] for source in PREDICTION_SOURCES},
        )[1]["content"])

    def test_cli_reviews_missing_material_sample_with_api_context(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sample_dir = root / "samples" / "sample_001"
            sample_dir.mkdir(parents=True)
            (sample_dir / "input.json").write_text(
                json.dumps(
                    {
                        "序号": "#1",
                        "user_prompt": "使用参考音频1的音色",
                        "reference_image_urls": [],
                        "generated_video_url": "generated.mp4",
                        "用户反馈": "",
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            (sample_dir / "gt.json").write_text(
                '[{"问题说明":"音色错误"}]',
                encoding="utf-8",
            )
            (sample_dir / "avagent.json").write_text("[]", encoding="utf-8")
            output = root / "results.jsonl"
            summary = root / "summary.json"

            response = {
                "sample_id": "sample_001",
                "reviews": [
                    {
                        "prediction_source": "avagent",
                        "category": 3,
                        "reason": "缺少参考音频，但GT动作问题仍可判断。",
                        "gt_coverage": [
                            {
                                "gt_index": 1,
                                "status": "covered",
                                "matched_prediction_indices": [],
                                "reason": "动作问题不依赖参考音频。",
                            }
                        ],
                        "extra_prediction_indices": [],
                        "confidence": "高",
                    }
                ],
            }
            with (
                mock.patch.dict(
                    os.environ,
                    {"AVAGENT_API_KEY": "token"},
                    clear=True,
                ),
                mock.patch.object(review_script, "load_project_env"),
                mock.patch.object(
                    review_script,
                    "chat_completion",
                    return_value=(json.dumps(response, ensure_ascii=False), {}, 123),
                ),
            ):
                exit_code = review_script.main(
                    [
                        "--input-root",
                        str(root / "samples"),
                        "--prediction-source",
                        "avagent",
                        "--output-jsonl",
                        str(output),
                        "--summary-json",
                        str(summary),
                    ]
                )

            result = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(result["reviews"][0]["category"], 3)
        self.assertEqual(result["request_bytes"], 123)
        self.assertEqual(result["usage"], {})

    def test_parse_review_response_requires_all_sources_and_valid_categories(self):
        response = {
            "sample_id": "#1",
            "reviews": [
                {
                    "prediction_source": source,
                    "category": 1,
                    "reason": "完整覆盖 GT，且没有额外问题。",
                    "gt_coverage": [
                        {
                            "gt_index": 1,
                            "status": "covered",
                            "matched_prediction_indices": [1],
                            "reason": "语义一致",
                        }
                    ],
                    "extra_prediction_indices": [],
                    "confidence": "高",
                }
                for source in PREDICTION_SOURCES
            ],
        }

        parsed = parse_review_response(json.dumps(response, ensure_ascii=False), "#1")

        self.assertEqual(
            [item["prediction_source"] for item in parsed["reviews"]],
            list(PREDICTION_SOURCES),
        )
        self.assertTrue(all(item["category"] == 1 for item in parsed["reviews"]))

    def test_parse_review_response_rejects_missing_source(self):
        response = {
            "sample_id": "#1",
            "reviews": [
                {
                    "prediction_source": source,
                    "category": 3,
                    "reason": "遗漏问题。",
                    "gt_coverage": [],
                    "extra_prediction_indices": [],
                    "confidence": "中",
                }
                for source in PREDICTION_SOURCES[:-1]
            ],
        }

        with self.assertRaisesRegex(ValueError, "预测来源"):
            parse_review_response(json.dumps(response, ensure_ascii=False), "#1")

    def test_supports_single_avagent_prediction_source(self):
        gt = [{"问题类型": "音频质量问题", "问题说明": "台词错误"}]
        predictions = {
            "avagent": [{"问题类型": "音频质量问题", "问题说明": "台词错误"}]
        }
        messages = build_messages(
            "#1",
            {"user_prompt": "检查台词", "generated_video_url": "video.mp4"},
            gt,
            predictions,
            prediction_sources=("avagent",),
        )
        self.assertIn("reviews 必须按 avagent 顺序", messages[0]["content"])
        self.assertNotIn("五个 prediction_source", messages[0]["content"])
        payload = json.loads(messages[1]["content"])
        self.assertEqual(tuple(payload["predictions"]), ("avagent",))

        response = {
            "sample_id": "#1",
            "reviews": [
                {
                    "prediction_source": "avagent",
                    "category": 1,
                    "reason": "完整覆盖。",
                    "gt_coverage": [
                        {
                            "gt_index": 1,
                            "status": "covered",
                            "matched_prediction_indices": [1],
                            "reason": "语义一致",
                        }
                    ],
                    "extra_prediction_indices": [],
                    "confidence": "高",
                }
            ],
        }
        parsed = parse_review_response(
            json.dumps(response, ensure_ascii=False),
            "#1",
            prediction_sources=("avagent",),
        )
        self.assertEqual(parsed["reviews"][0]["prediction_source"], "avagent")

    def test_multi_source_prompt_enforces_superset_coverage_monotonicity(self):
        messages = build_messages(
            "#1",
            {"user_prompt": "检查台词", "generated_video_url": "video.mp4"},
            [{"问题说明": "台词错误"}],
            {
                "gpt_a": [{"问题说明": "台词错误"}],
                "avagent": [
                    {"问题说明": "台词错误"},
                    {"问题说明": "额外杂音"},
                ],
            },
            prediction_sources=("gpt_a", "avagent"),
        )
        self.assertIn("GT 覆盖不能更差", messages[0]["content"])
        self.assertIn("类别 1 变为类别 2", messages[0]["content"])

    def test_resume_rejects_existing_results_from_other_sources(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "results.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "sample_id": "sample_001",
                        "reviews": [
                            {
                                "prediction_source": "gpt_a",
                                "category": 1,
                            }
                        ],
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "预测来源"):
                _read_existing(path, ("avagent",))

    def test_summary_counts_categories_per_prediction_source(self):
        results = [
            {
                "sample_id": "sample_001",
                "reviews": [
                    {
                        "prediction_source": source,
                        "category": 1 if source == "gpt_a" else 3,
                    }
                    for source in PREDICTION_SOURCES
                ],
            }
        ]

        summary = summarize_results(results)

        self.assertEqual(summary["sample_count"], 1)
        self.assertEqual(summary["by_source"]["gpt_a"]["1"], 1)
        self.assertEqual(summary["by_source"]["gpt_b"]["3"], 1)

    def test_summary_supports_single_avagent_source(self):
        results = [
            {
                "sample_id": "sample_001",
                "reviews": [{"prediction_source": "avagent", "category": 2}],
            }
        ]
        summary = summarize_results(results, prediction_sources=("avagent",))
        self.assertEqual(tuple(summary["by_source"]), ("avagent",))
        self.assertEqual(summary["by_source"]["avagent"]["2"], 1)


if __name__ == "__main__":
    unittest.main()
