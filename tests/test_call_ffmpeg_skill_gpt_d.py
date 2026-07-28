import csv
import io
import json
import tempfile
import unittest
import urllib.error
import wave
from pathlib import Path
from unittest import mock

import call_ffmpeg_skill_gpt_d as runner


class CallFfmpegSkillGptDTest(unittest.TestCase):
    def test_uses_gemini_35_flash_through_existing_ark_gateway(self):
        self.assertEqual(runner.DEFAULT_MODEL, "gemini-3.5-flash")
        self.assertEqual(runner.API_KEY_ENV, "ARK_API_KEY")
        self.assertIn("/ark-router/v1/chat/completions", runner.DEFAULT_API_URL)

    def test_inference_input_includes_references_but_excludes_feedback_and_gold(self):
        header = [
            "序号",
            "user_prompt",
            "reference_image_urls",
            "generated_video_url",
            "用户反馈",
            "思考过程及标准答案",
        ]
        row = [
            "7",
            "孩子说你好，背景为轻快钢琴声",
            '["secret-reference.jpg"]',
            "sample.mp4",
            "反馈称声音错误",
            "gold answer",
        ]
        value = runner.inference_input(header, row, 1)
        self.assertEqual(
            value,
            {
                "序号": "7",
                "user_prompt": "孩子说你好，背景为轻快钢琴声",
                "reference_image_urls": ["secret-reference.jpg"],
                "generated_video_url": "sample.mp4",
            },
        )

    def test_prompt_directly_requests_clear_audio_and_av_conflict_issues(self):
        prompt = runner.build_prompt("孩子说你好，背景为轻快钢琴声")
        self.assertIn("参考图", runner.SYSTEM_MESSAGE)
        self.assertIn("孩子说你好，背景为轻快钢琴声", prompt)
        self.assertIn("联合分析", prompt)
        self.assertIn("只输出明确错误", prompt)
        self.assertIn("嘴部运动", prompt)
        self.assertIn("不能判断具体人物的声纹", prompt)
        self.assertIn("参考图", prompt)
        self.assertIn("音色、音调", prompt)
        self.assertIn("语言、台词、声音与主体的绑定关系", prompt)
        self.assertIn("角色 A 的台词由角色 B 发出", prompt)
        self.assertIn("旁白或画外音错误绑定", prompt)
        self.assertIn("即使 prompt 明确禁止字幕", prompt)
        self.assertIn("字幕与实际语音", prompt)
        self.assertIn("错别字", prompt)
        self.assertIn("音频片段", prompt)
        self.assertIn("不得自行估算比分片边界更精细的音频时间", prompt)
        self.assertIn("音频质量问题", prompt)
        self.assertNotIn("思考过程及标准答案", prompt)

    def test_split_wav_bytes_produces_complete_ordered_one_second_segments(self):
        source = io.BytesIO()
        samples = b"".join(
            value.to_bytes(2, "little", signed=True)
            for value in range(9)
        )
        with wave.open(source, "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(4)
            output.writeframes(samples)

        segments = runner.split_wav_bytes(source.getvalue())

        self.assertEqual(
            [
                (segment["start_sec"], segment["end_sec"])
                for segment in segments
            ],
            [(0.0, 1.0), (1.0, 2.0), (2.0, 2.25)],
        )
        reconstructed = bytearray()
        for segment in segments:
            with wave.open(io.BytesIO(segment["wav_bytes"]), "rb") as chunk:
                self.assertEqual(chunk.getframerate(), 4)
                reconstructed.extend(chunk.readframes(chunk.getnframes()))
        self.assertEqual(bytes(reconstructed), samples)

    def test_user_content_uses_inline_data_for_references_frames_and_wav_segments(self):
        parts = runner.build_user_content(
            reference_images=[
                "data:image/jpeg;base64,cmVmMQ==",
                "data:image/jpeg;base64,cmVmMg==",
            ],
            video_frames=[
                {
                    "timestamp_sec": 0.0,
                    "data_url": "data:image/jpeg;base64,ZmFrZQ==",
                }
            ],
            audio_segments=[
                {
                    "start_sec": 0.0,
                    "end_sec": 1.0,
                    "wav_bytes": b"wav-1",
                },
                {
                    "start_sec": 1.0,
                    "end_sec": 1.25,
                    "wav_bytes": b"wav-2",
                },
            ],
            user_prompt="检查声音",
        )
        self.assertEqual(
            [set(part) for part in parts],
            [
                {"text"},
                {"text"},
                {"text"},
                {"inline_data"},
                {"text"},
                {"inline_data"},
                {"text"},
                {"text"},
                {"inline_data"},
                {"text"},
                {"inline_data"},
                {"text"},
                {"inline_data"},
            ],
        )
        self.assertIn("检查声音", parts[0]["text"])
        self.assertEqual(
            parts[3]["inline_data"],
            {"mime_type": "image/jpeg", "data": "cmVmMQ=="},
        )
        self.assertEqual(
            parts[5]["inline_data"],
            {"mime_type": "image/jpeg", "data": "cmVmMg=="},
        )
        self.assertEqual(
            parts[8]["inline_data"],
            {"mime_type": "image/jpeg", "data": "ZmFrZQ=="},
        )
        self.assertEqual(parts[10]["inline_data"]["mime_type"], "audio/wav")
        self.assertNotIn("data:", parts[10]["inline_data"]["data"])
        self.assertIn("time_range=0.00s - 1.00s", parts[9]["text"])
        self.assertIn("time_range=1.00s - 1.25s", parts[11]["text"])
        self.assertEqual(
            parts[12]["inline_data"]["data"],
            runner.base64.b64encode(b"wav-2").decode("ascii"),
        )

    def test_user_content_marks_missing_audio_without_fabricating_evidence(self):
        parts = runner.build_user_content(
            reference_images=[],
            video_frames=[],
            audio_segments=[],
            user_prompt="检查声音",
        )
        self.assertIn("未检测到音轨", parts[-1]["text"])
        self.assertFalse(any("inline_data" in part for part in parts))

    def test_chat_payload_uses_gemini_contents_on_existing_endpoint(self):
        parts = [{"text": "test"}]
        payload = runner.build_chat_payload("gemini-3.5-flash", parts)
        self.assertEqual(
            payload,
            {
                "model": "gemini-3.5-flash",
                "contents": [
                    {
                        "role": "user",
                        "parts": [
                            {"text": runner.SYSTEM_MESSAGE},
                            {"text": "test"},
                        ],
                    }
                ],
            },
        )
        self.assertNotIn("messages", payload)
        self.assertNotIn("tools", payload)

    def test_prediction_accepts_empty_or_populated_issue_arrays(self):
        self.assertEqual(runner.parse_prediction("[]"), "[]")
        issue = {
            "可定位性": "否",
            "置信度": "高",
            "问题说明": "预期为童声，实际为低沉成年男声，音色与可见儿童明显冲突。",
            "问题类型": "音频质量问题",
            "时间区间": "0s - 2s",
            "关键帧秒": "",
            "BBox": "",
        }
        parsed = json.loads(runner.parse_prediction(json.dumps([issue], ensure_ascii=False)))
        self.assertEqual(parsed, [issue])

    def test_prediction_preserves_subtitle_audio_mismatch_as_text_issue(self):
        issue = {
            "可定位性": "否",
            "置信度": "高",
            "问题说明": "字幕写成你好，实际语音说再见。",
            "问题类型": "文字质量问题",
            "时间区间": "1s - 2s",
            "关键帧秒": "",
            "BBox": "",
        }
        parsed = json.loads(
            runner.parse_prediction(json.dumps([issue], ensure_ascii=False))
        )
        self.assertEqual(parsed[0]["问题类型"], "文字质量问题")

    def test_merge_preserves_all_gpt_a_issues_and_appends_gemini_audio(self):
        gpt_a_issues = [
            {
                "可定位性": "是",
                "置信度": "高",
                "问题说明": "人物没有按要求抬手。",
                "问题类型": "动作异常",
                "时间区间": "1s - 2s",
                "关键帧秒": "1.5",
                "BBox": "",
            }
        ]
        audio_issues = [
            {
                "可定位性": "否",
                "置信度": "高",
                "问题说明": "预期无台词，实际出现男声台词。",
                "问题类型": "音频质量问题",
                "时间区间": "3s - 4s",
                "关键帧秒": "",
                "BBox": "",
            }
        ]

        merged = json.loads(
            runner.merge_predictions(
                json.dumps(gpt_a_issues, ensure_ascii=False),
                json.dumps(audio_issues, ensure_ascii=False),
            )
        )

        self.assertEqual(merged, gpt_a_issues + audio_issues)

    def test_merge_with_no_audio_issue_equals_gpt_a_prediction(self):
        gpt_a_issues = [
            {
                "可定位性": "是",
                "置信度": "中",
                "问题说明": "镜头发生了额外切换。",
                "问题类型": "镜头变化问题",
                "时间区间": "2s - 3s",
                "关键帧秒": "2.5",
                "BBox": "",
            }
        ]
        merged = json.loads(
            runner.merge_predictions(
                json.dumps(gpt_a_issues, ensure_ascii=False),
                "[]",
            )
        )
        self.assertEqual(merged, gpt_a_issues)

    def test_combined_row_calls_live_gpt_a_before_gemini_and_merges(self):
        calls = []
        gpt_a_prediction = json.dumps(
            [{"问题类型": "动作异常", "问题说明": "动作错误"}],
            ensure_ascii=False,
        )
        audio_prediction = json.dumps(
            [{"问题类型": "音频质量问题", "问题说明": "台词错误"}],
            ensure_ascii=False,
        )

        def fake_gpt_a(*args, **kwargs):
            calls.append(("gpt_a", args[0]))
            return gpt_a_prediction

        def fake_gemini(*args, **kwargs):
            calls.append(("gemini", args[0]))
            return audio_prediction

        with (
            mock.patch.object(runner.gpt_a, "run_agent", side_effect=fake_gpt_a),
            mock.patch.object(runner, "run_audio_row", side_effect=fake_gemini),
        ):
            merged = runner.run_combined_row(
                {
                    "序号": "#1",
                    "user_prompt": "人物抬手并说你好",
                    "reference_image_urls": ["ref.jpg"],
                    "generated_video_url": "video.mp4",
                    "用户反馈": "动作不自然",
                },
                {
                    "序号": "#1",
                    "user_prompt": "人物抬手并说你好",
                    "reference_image_urls": ["ref.jpg"],
                    "generated_video_url": "video.mp4",
                },
                api_url="https://example.test/chat/completions",
                api_key="token",
                gpt_a_model="gpt-model",
                gemini_model="gemini-model",
                timeout=30,
                api_retries=2,
                max_gpt_a_agent_steps=10,
            )

        self.assertEqual([name for name, _ in calls], ["gpt_a", "gemini"])
        self.assertEqual(calls[0][1]["reference_image_urls"], ["ref.jpg"])
        self.assertEqual(calls[0][1]["用户反馈"], "动作不自然")
        self.assertEqual(calls[1][1]["reference_image_urls"], ["ref.jpg"])
        self.assertEqual(
            json.loads(merged),
            json.loads(gpt_a_prediction) + json.loads(audio_prediction),
        )

    def test_combined_row_keeps_gpt_a_stats_when_gemini_fails(self):
        run_stats = {}

        def fake_gpt_a(*args, **kwargs):
            args[-1].update({"api_calls": 1, "request_bytes": 123})
            return "[]"

        def fake_gemini(*args, **kwargs):
            raise RuntimeError("gemini failed")

        with (
            mock.patch.object(runner.gpt_a, "run_agent", side_effect=fake_gpt_a),
            mock.patch.object(runner, "run_audio_row", side_effect=fake_gemini),
            self.assertRaisesRegex(RuntimeError, "gemini failed"),
        ):
            runner.run_combined_row(
                {"序号": "#1"},
                {"序号": "#1"},
                api_url="https://example.test/chat/completions",
                api_key="token",
                gpt_a_model="gpt-model",
                gemini_model="gemini-model",
                timeout=30,
                api_retries=2,
                max_gpt_a_agent_steps=10,
                run_stats=run_stats,
            )

        self.assertEqual(
            run_stats["gpt_a"],
            {"api_calls": 1, "request_bytes": 123},
        )
        self.assertEqual(run_stats["gemini_audio"], {})

    def test_prediction_rejects_issue_without_required_evidence(self):
        with self.assertRaisesRegex(ValueError, "问题说明"):
            runner.parse_prediction(
                json.dumps(
                    [
                        {
                            "置信度": "高",
                            "问题说明": "",
                            "时间区间": "0s - 2s",
                        }
                    ],
                    ensure_ascii=False,
                )
            )

    def test_prediction_rejects_invalid_confidence(self):
        with self.assertRaisesRegex(ValueError, "置信度"):
            runner.parse_prediction(
                json.dumps(
                    [
                        {
                            "置信度": "低",
                            "问题说明": "存在明确断音。",
                            "时间区间": "0s - 2s",
                        }
                    ],
                    ensure_ascii=False,
                )
            )

    def test_prediction_normalizes_minute_style_time_range_to_seconds(self):
        issue = {
            "置信度": "高",
            "问题说明": "预期只有喘息声，实际出现了清晰台词。",
            "时间区间": "00:03s - 01:04.5s",
        }
        parsed = json.loads(
            runner.parse_prediction(json.dumps([issue], ensure_ascii=False))
        )
        self.assertEqual(parsed[0]["时间区间"], "3.00s - 64.50s")

    def test_resume_rejects_output_from_different_source_rows(self):
        header = runner.SOURCE_COLUMNS
        current = ["1", "new prompt", "[]", "new.mp4", "", "new gold"]
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "pred.csv"
            with output.open("w", encoding="utf-8-sig", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(header + [runner.PREDICTION_COLUMN])
                writer.writerow(
                    ["1", "old prompt", "[]", "old.mp4", "", "old gold", "[]"]
                )
            with self.assertRaisesRegex(ValueError, "源字段不一致"):
                runner.read_matching_predictions(output, header, [current])

    def test_retry_metadata_counts_all_chat_completion_attempts(self):
        response = {
            "choices": [{"message": {"role": "assistant", "content": "[]"}}],
            "usage": {"total_tokens": 5},
        }

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def read(self):
                return json.dumps(response).encode("utf-8")

        parts = [{"text": "test"}]
        expected_bytes = len(
            json.dumps(
                runner.build_chat_payload(runner.DEFAULT_MODEL, parts),
                ensure_ascii=False,
            ).encode("utf-8")
        )
        with (
            mock.patch.object(
                runner.urllib.request,
                "urlopen",
                side_effect=[urllib.error.URLError("temporary"), FakeResponse()],
            ),
            mock.patch.object(runner.time, "sleep"),
        ):
            message = runner.chat_completion(
                runner.DEFAULT_API_URL,
                "token",
                runner.DEFAULT_MODEL,
                parts,
                timeout=1,
                max_attempts=2,
            )
        self.assertEqual(message["_api_attempts"], 2)
        self.assertEqual(message["_request_bytes"], expected_bytes * 2)

if __name__ == "__main__":
    unittest.main()
