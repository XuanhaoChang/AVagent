import csv
import io
import json
import tempfile
import unittest
import urllib.error
import wave
from pathlib import Path
from unittest import mock

import run_avagent as runner


class AVAgentEntryPointTest(unittest.TestCase):
    def test_extracts_free_format_resolution_and_single_shot_duration(self):
        constraints = runner.extract_visual_metadata_constraints(
            {
                "user_prompt": (
                    "继续上一段剧情，不要降低画质。\n"
                    "🎬 镜头 1：持续15秒，人物完成对话。"
                ),
                "用户反馈": (
                    "客户是直出1080P的视频，反馈画面模糊，观感达不到1080P。"
                ),
            }
        )

        self.assertEqual(
            [item["kind"] for item in constraints],
            ["duration", "minimum_resolution"],
        )
        self.assertEqual(constraints[0]["required_sec"], 15.0)
        self.assertEqual(constraints[1]["minimum_short_side"], 1080)
        self.assertEqual(constraints[1]["minimum_long_side"], 1920)
        self.assertEqual(constraints[1]["source_field"], "用户反馈")

    def test_reference_resolution_is_not_mistaken_for_output_requirement(self):
        constraints = runner.extract_visual_metadata_constraints(
            {
                "user_prompt": (
                    "参考视频为4K，仅参考剧情和人物站位，不需要参考画质。"
                    "输出视频保持正常清晰度即可。"
                )
            }
        )

        self.assertEqual(constraints, [])

    def test_metadata_mismatch_generates_required_resolution_and_duration_issues(self):
        stats = {}
        with (
            mock.patch.object(
                runner.gpt_a,
                "ensure_video",
                return_value=Path("video.mp4"),
            ),
            mock.patch.object(
                runner.gpt_a,
                "probe_video",
                return_value={
                    "duration_sec": 13.042,
                    "width": 720,
                    "height": 1280,
                    "has_audio": True,
                },
            ),
        ):
            issues = runner.evaluate_visual_metadata_constraints(
                {
                    "user_prompt": "🎬 镜头 1：持续15秒，不要降低画质。",
                    "用户反馈": "客户要求直出1080P。",
                    "generated_video_url": "video.mp4",
                },
                run_stats=stats,
            )

        self.assertEqual(
            [issue["问题类型"] for issue in issues],
            ["时序错误", "清晰度异常"],
        )
        self.assertIn("720×1280", issues[1]["问题说明"])
        self.assertIn("最低1080×1920", issues[1]["问题说明"])
        self.assertNotIn("人物脸部", issues[1]["问题说明"])
        self.assertEqual(
            stats["visual_metadata_constraints"]["media_metadata"]["width"],
            720,
        )

    def test_matching_metadata_does_not_generate_an_issue(self):
        constraints = runner.extract_visual_metadata_constraints(
            {
                "user_prompt": "输出1080P视频，总时长15秒。",
            }
        )

        self.assertEqual(
            runner.metadata_constraint_issues(
                constraints,
                {"duration_sec": 15.2, "width": 1080, "height": 1920},
            ),
            [],
        )

    def test_uses_public_configuration_names(self):
        self.assertEqual(runner.DEFAULT_MODEL, "gemini-3.5-flash")
        self.assertEqual(runner.API_KEY_ENV, "AVAGENT_API_KEY")
        self.assertEqual(runner.DEFAULT_API_URL, "")

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

    def test_prompt_declares_output_contract_and_delegates_sync_to_avbench(self):
        prompt = runner.build_prompt("孩子说你好，背景为轻快钢琴声")
        self.assertIn("参考图", runner.SYSTEM_MESSAGE)
        self.assertIn("孩子说你好，背景为轻快钢琴声", prompt)
        self.assertIn("ASR 是台词内容和读音问题的判定依据", prompt)
        self.assertIn("只输出明确错误", prompt)
        self.assertIn("AVBench", prompt)
        self.assertIn("必须包含以下 7 个键", prompt)
        self.assertIn("问题说明", prompt)
        self.assertIn("ASR 是台词内容和读音问题的判定依据", prompt)
        self.assertNotIn("粗粒度声画冲突", prompt)
        self.assertNotIn("音频片段", prompt)
        self.assertNotIn("分片边界", prompt)
        self.assertIn("不能判断具体人物的声纹", prompt)
        self.assertIn("OCR/字幕画面", prompt)
        self.assertIn("音色、音调", prompt)
        self.assertIn("语言、台词、声音与主体的绑定关系", prompt)
        self.assertIn("角色 A 的台词由角色 B 发出", prompt)
        self.assertIn("旁白或画外音错误绑定", prompt)
        self.assertIn("即使 prompt 明确禁止字幕", prompt)
        self.assertIn("字幕与实际语音", prompt)
        self.assertIn("错别字", prompt)
        self.assertIn("音频质量问题", prompt)
        self.assertNotIn("思考过程及标准答案", prompt)

    def test_synthesis_prompt_requires_semantic_merge_and_separate_caption_violations(self):
        prompt = runner.build_synthesis_prompt(
            "要求无字幕，并让女孩说：你都修仨小时了",
            '[{"问题说明":"GPT-A发现视频有字幕"}]',
            '[{"问题说明":"OCR写成三小时，ASR为三小时"}]',
        )
        self.assertIn(
            "prompt 要求无字幕，但视频有字幕",
            runner.FINAL_SYNTHESIS_SYSTEM_MESSAGE,
        )
        self.assertIn("字幕内容存在错字", runner.FINAL_SYNTHESIS_SYSTEM_MESSAGE)
        self.assertIn("ASR 是台词内容和读音差异的判定依据", runner.FINAL_SYNTHESIS_SYSTEM_MESSAGE)
        self.assertIn("不要求最终 GPT 或", runner.FINAL_SYNTHESIS_SYSTEM_MESSAGE)
        self.assertIn("Auralis 与 Seed-Lite 输入中的每个独立问题事实", runner.FINAL_SYNTHESIS_SYSTEM_MESSAGE)
        self.assertIn("decision=observed_preferred", runner.FINAL_SYNTHESIS_SYSTEM_MESSAGE)
        self.assertIn("expected_preferred", runner.FINAL_SYNTHESIS_SYSTEM_MESSAGE)
        self.assertIn("自由格式 prompt", runner.FINAL_SYNTHESIS_SYSTEM_MESSAGE)
        self.assertIn("映射支持的", runner.FINAL_SYNTHESIS_SYSTEM_MESSAGE)
        self.assertIn("无法把 spk0 绝对命名为某角色", runner.FINAL_SYNTHESIS_SYSTEM_MESSAGE)
        self.assertIn("人物 -> 匿名 spk", runner.FINAL_SYNTHESIS_SYSTEM_MESSAGE)
        self.assertIn("不得改写成“整句应由同一 spk 完整发出”", runner.FINAL_SYNTHESIS_SYSTEM_MESSAGE)
        self.assertIn("每个独立问题事实都必须", prompt)
        self.assertIn("独立问题事实并集为下限", runner.FINAL_SYNTHESIS_SYSTEM_MESSAGE)
        self.assertIn("复合对象拆分后不要重复输出", prompt)
        self.assertIn("GPT-A 候选结果", prompt)
        self.assertIn("Auralis（ASR/OCR/Gemini，以本地受约束 ASR 为台词判定依据）专家结果", prompt)
        self.assertIn("Seed-Lite 视觉物理专家结果", prompt)
        self.assertIn("本地视频元数据约束检查结果", prompt)
        self.assertIn("要求无字幕", prompt)

    def test_avbench_is_mandatory_and_added_to_synthesis_prompt(self):
        class FakeAvbench:
            def evaluate(self, video_path):
                self.video_path = video_path
                return {
                    "source": "AVBench evaluate_syncnet.py",
                    "success": True,
                    "status": "ok",
                    "sync_quality": "Poor",
                    "sync_score": 12.5,
                }

        fake = FakeAvbench()
        stats = {}
        with mock.patch.object(
            runner.gpt_a,
            "ensure_video",
            return_value=Path("video.mp4"),
        ):
            result = runner.run_avbench_row(
                {"generated_video_url": "video.mp4"},
                avbench_runner=fake,
                run_stats=stats,
            )

        self.assertEqual(fake.video_path, Path("video.mp4"))
        self.assertEqual(result["sync_quality"], "Poor")
        self.assertEqual(stats["status"], "ok")
        prompt = runner.build_synthesis_prompt(
            "检查音画同步",
            "[]",
            "[]",
            result,
        )
        self.assertIn("AVBench 音画同步结果", prompt)
        self.assertIn("sync_score", prompt)

    def test_user_content_uses_one_inline_data_for_the_complete_wav(self):
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
            audio_wav=b"complete-wav",
            audio_duration_sec=1.25,
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
        self.assertIn("可选的原始 WAV", parts[9]["text"])
        self.assertIn("总时长约 1.25s", parts[9]["text"])
        self.assertEqual(
            parts[10]["inline_data"]["data"],
            runner.base64.b64encode(b"complete-wav").decode("ascii"),
        )

    def test_user_content_marks_missing_audio_without_fabricating_evidence(self):
        parts = runner.build_user_content(
            reference_images=[],
            video_frames=[],
            audio_wav=None,
            user_prompt="检查声音",
        )
        self.assertIn("未检测到音轨", parts[-1]["text"])
        self.assertFalse(any("inline_data" in part for part in parts))

    def test_user_content_adds_role_labelled_voice_check_clip(self):
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as target:
            target.setnchannels(1)
            target.setsampwidth(2)
            target.setframerate(16000)
            target.writeframes(b"\x00\x00" * 32000)
        evidence = {
            "asr": {
                "metadata": {
                    "prompt_speech_plan": {
                        "role_reference_images": {"林止": [2]},
                    },
                    "speaker_binding_evidence": {
                        "prompt_turn_alignment": [
                            {
                                "role": "林止",
                                "dialogue_text": "别碰我！放开！",
                                "observed_text": "别碰我放开",
                                "status": "anchored",
                                "anchor_method": "dialogue_text_similarity",
                                "actual_speakers": [0],
                                "matched_segments": [
                                    {
                                        "start_sec": 0.25,
                                        "end_sec": 1.25,
                                        "speaker": 0,
                                        "text": "别碰我放开",
                                    }
                                ],
                            }
                        ]
                    },
                }
            }
        }
        prompt = (
            '林止急喊：“别碰我！放开！”\n'
            '视频角色对照表: 林止=【图2】'
        )

        parts = runner.build_user_content(
            reference_images=[
                "data:image/jpeg;base64,cmVmMQ==",
                "data:image/jpeg;base64,cmVmMg==",
            ],
            video_frames=[],
            audio_wav=wav_buffer.getvalue(),
            audio_duration_sec=2.0,
            user_prompt=prompt,
            local_evidence_json=runner.json.dumps(evidence, ensure_ascii=False),
        )

        text_parts = [part["text"] for part in parts if "text" in part]
        self.assertTrue(any("对应角色=林止" in text for text in text_parts))
        self.assertTrue(any("角色声线核查" in text and "林止" in text for text in text_parts))
        wav_parts = [
            part["inline_data"]
            for part in parts
            if part.get("inline_data", {}).get("mime_type") == "audio/wav"
        ]
        self.assertEqual(len(wav_parts), 2)
        clip = runner.base64.b64decode(wav_parts[0]["data"])
        with wave.open(io.BytesIO(clip), "rb") as source:
            duration_sec = source.getnframes() / source.getframerate()
        self.assertAlmostEqual(duration_sec, 1.0, places=2)

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

    def test_final_synthesis_uses_text_only_chat_payload(self):
        response = {
            "choices": [
                {"message": {"role": "assistant", "content": "[]"}}
            ],
            "usage": {"total_tokens": 7},
        }

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def read(self):
                return json.dumps(response).encode("utf-8")

        stats = {}
        with mock.patch.object(
            runner.urllib.request,
            "urlopen",
            return_value=FakeResponse(),
        ) as urlopen:
            result = runner.final_chat_completion(
                "https://example.test/chat/completions",
                "token",
                "gpt-model",
                [{"role": "user", "content": "整理结果"}],
                timeout=1,
                max_attempts=1,
                run_stats=stats,
            )

        self.assertEqual(result, "[]")
        request = urlopen.call_args.args[0]
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(payload["model"], "gpt-model")
        self.assertIn("messages", payload)
        self.assertNotIn("tools", payload)
        self.assertEqual(stats["api_calls"], 1)
        self.assertGreater(stats["request_bytes"], 0)

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

    def test_preserve_deterministic_issue_when_synthesizer_omits_it(self):
        required = {
            "可定位性": "否",
            "置信度": "高",
            "问题说明": (
                "预期烧录字幕应与实际语音一致；ASR 实际语音为“你都修仨小时了”，"
                "OCR 实际字幕为“你都修三小时了”，字幕存在错字：仨→三；"
                "发生在 10.50s - 12.00s。"
            ),
            "问题类型": "文字质量问题",
            "时间区间": "10.50s - 12.00s",
            "关键帧秒": "",
            "BBox": "",
        }

        preserved = json.loads(
            runner.preserve_deterministic_issues("[]", [required])
        )

        self.assertEqual(preserved, [required])

    def test_preserve_deterministic_issue_does_not_duplicate_rephrased_match(self):
        required = {
            "可定位性": "否",
            "置信度": "高",
            "问题说明": (
                "预期烧录字幕应与实际语音一致；ASR 实际语音为“你都修仨小时了”，"
                "OCR 实际字幕为“你都修三小时了”，字幕存在错字：仨→三；"
                "发生在 10.50s - 12.00s。"
            ),
            "问题类型": "文字质量问题",
            "时间区间": "10.50s - 12.00s",
            "关键帧秒": "",
            "BBox": "",
        }
        synthesized = {
            **required,
            "问题说明": "10.5秒处字幕把语音中的“仨”错误写成了“三”。",
        }

        preserved = json.loads(
            runner.preserve_deterministic_issues(
                json.dumps([synthesized], ensure_ascii=False),
                [required],
            )
        )

        self.assertEqual(preserved, [synthesized])

    def test_selects_only_high_resolution_verified_gpt_a_text_issue(self):
        issue = {
            "可定位性": "是",
            "置信度": "高",
            "问题说明": "预期显示“据实结算”，实际显示为“据实纤算”。",
            "问题类型": "文字质量问题",
            "时间区间": "0s - 5s",
            "关键帧秒": "0.5",
            "BBox": "<bbox>0.2,0.3,0.7,0.6</bbox>",
        }
        prediction = json.dumps([issue], ensure_ascii=False)
        verified = runner.select_evidence_backed_gpt_a_issues(
            prediction,
            {
                "tool_calls": [
                    {
                        "name": "extract_crop",
                        "ok": True,
                        "arguments": {
                            "timestamp_sec": 0.5,
                            "x1": 0.1,
                            "y1": 0.2,
                            "x2": 0.8,
                            "y2": 0.7,
                        },
                    }
                ]
            },
        )
        unverified = runner.select_evidence_backed_gpt_a_issues(
            prediction,
            {
                "tool_calls": [
                    {
                        "name": "extract_frame",
                        "ok": True,
                        "arguments": {"timestamp_sec": 10.0},
                    }
                ]
            },
        )

        self.assertEqual(verified, [issue])
        self.assertEqual(unverified, [])

    def test_deduplicates_auralis_rephrases_before_synthesis(self):
        first = {
            "问题类型": "文字质量问题",
            "问题说明": "字幕把“倍”错误写成“碚”。",
            "时间区间": "13.50s - 15.00s",
        }
        duplicate = {
            "问题类型": "文字质量问题",
            "问题说明": "预期字幕为“倍”，实际字幕为“碚”，存在错别字。",
            "时间区间": "13.50s - 15.00s",
        }
        deduplicated = json.loads(
            runner.deduplicate_prediction_issues(
                json.dumps([first, duplicate], ensure_ascii=False),
                "Auralis",
            )
        )
        self.assertEqual(deduplicated, [first])

    def test_keeps_pronunciation_and_speaker_binding_at_the_same_time(self):
        pronunciation = {
            "可定位性": "否",
            "置信度": "高",
            "问题说明": (
                "ASR 实际识别为“雪明”，prompt 预期为“雪芳”，"
                "受约束 CTC 候选评分支持读音差异。"
            ),
            "问题类型": "音频质量问题",
            "时间区间": "2.36s - 4.64s",
            "关键帧秒": "",
            "BBox": "",
        }
        speaker_binding = {
            "可定位性": "否",
            "置信度": "高",
            "问题说明": (
                "预期李莲的台词“雪芳终于想通了”由李莲发出，实际由已锚定为"
                "贺雨棠的声纹 spk0 发出，属于角色绑定错误。"
            ),
            "问题类型": "音频质量问题",
            "时间区间": "2.36s - 4.64s",
            "关键帧秒": "",
            "BBox": "",
        }

        deduplicated = json.loads(
            runner.deduplicate_prediction_issues(
                json.dumps(
                    [pronunciation, speaker_binding],
                    ensure_ascii=False,
                ),
                "Auralis",
            )
        )
        preserved = json.loads(
            runner.preserve_deterministic_issues(
                json.dumps([pronunciation], ensure_ascii=False),
                [pronunciation, speaker_binding],
            )
        )

        self.assertEqual(deduplicated, [pronunciation, speaker_binding])
        self.assertEqual(preserved, [pronunciation, speaker_binding])

    def test_split_subtitle_facts_collectively_cover_composite_issue(self):
        presence = {
            "可定位性": "否",
            "置信度": "高",
            "问题说明": (
                "prompt 明确要求“视频中不要出现任何字幕”，"
                "实际视频中违规出现了零碎字幕。"
            ),
            "问题类型": "文字质量问题",
            "时间区间": "1.50s - 13.50s",
            "关键帧秒": "",
            "BBox": "",
        }
        content = {
            **presence,
            "问题说明": (
                "实际字幕内容与语音不符：语音为“谢谢妈”时字幕显示为“香”；"
                "另一处字幕显示为“之”。"
            ),
        }
        composite = {
            **presence,
            "问题说明": (
                "预期为视频中不要出现任何字幕，实际违规出现了零碎字幕，"
                "且字幕文字严重错误：语音为“谢谢妈”时字幕显示为“香”，"
                "另一处字幕显示为“之”。"
            ),
        }

        preserved = json.loads(
            runner.preserve_deterministic_issues(
                json.dumps([presence, content], ensure_ascii=False),
                [composite],
            )
        )
        pruned = json.loads(
            runner.preserve_deterministic_issues(
                json.dumps(
                    [presence, content, composite],
                    ensure_ascii=False,
                ),
                [composite],
            )
        )

        self.assertEqual(preserved, [presence, content])
        self.assertEqual(pruned, [presence, content])

    def test_unrelated_subtitle_content_does_not_cover_composite_issue(self):
        presence = {
            "可定位性": "否",
            "置信度": "高",
            "问题说明": "prompt 要求不要出现字幕，实际出现了字幕。",
            "问题类型": "文字质量问题",
            "时间区间": "1.50s - 13.50s",
            "关键帧秒": "",
            "BBox": "",
        }
        unrelated_content = {
            **presence,
            "问题说明": "另一处字幕把“报价单”错误写成“合同”。",
        }
        composite = {
            **presence,
            "问题说明": (
                "预期不要出现字幕，实际出现字幕且字幕内容错误："
                "语音为“谢谢妈”时字幕显示为“香”。"
            ),
        }

        preserved = json.loads(
            runner.preserve_deterministic_issues(
                json.dumps(
                    [presence, unrelated_content],
                    ensure_ascii=False,
                ),
                [composite],
            )
        )

        self.assertEqual(preserved, [presence, unrelated_content, composite])

    def test_combined_row_calls_live_gpt_a_gemini_and_final_synthesis(self):
        calls = []
        gpt_a_prediction = json.dumps(
            [{"问题类型": "动作异常", "问题说明": "动作错误"}],
            ensure_ascii=False,
        )
        audio_prediction = json.dumps(
            [{"问题类型": "音频质量问题", "问题说明": "台词错误"}],
            ensure_ascii=False,
        )
        final_prediction = json.dumps(
            [{"问题类型": "文字质量问题", "问题说明": "最终整理后的问题"}],
            ensure_ascii=False,
        )

        def fake_gpt_a(*args, **kwargs):
            calls.append(("gpt_a", args[0]))
            return gpt_a_prediction

        def fake_gemini(*args, **kwargs):
            calls.append(("gemini", args[0]))
            return audio_prediction

        avbench_result = {
            "source": "AVBench evaluate_syncnet.py",
            "success": True,
            "status": "ok",
            "sync_quality": "Poor",
            "sync_score": 12.5,
            "offset_sec": 0.8,
        }

        def fake_avbench(*args, **kwargs):
            calls.append(("avbench", args[0]))
            return avbench_result

        def fake_synthesis(**kwargs):
            calls.append(("synthesis", kwargs))
            return final_prediction

        with (
            mock.patch.object(runner.gpt_a, "run_agent", side_effect=fake_gpt_a),
            mock.patch.object(runner, "run_audio_row", side_effect=fake_gemini),
            mock.patch.object(runner, "run_avbench_row", side_effect=fake_avbench),
            mock.patch.object(
                runner,
                "synthesize_predictions",
                side_effect=fake_synthesis,
            ),
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

        self.assertEqual(
            [name for name, _ in calls],
            ["gpt_a", "gemini", "avbench", "synthesis"],
        )
        self.assertEqual(calls[0][1]["reference_image_urls"], ["ref.jpg"])
        self.assertEqual(calls[0][1]["用户反馈"], "动作不自然")
        self.assertEqual(calls[1][1]["reference_image_urls"], ["ref.jpg"])
        self.assertEqual(json.loads(merged), json.loads(final_prediction))
        self.assertEqual(calls[3][1]["model"], "gpt-model")
        self.assertEqual(calls[3][1]["gpt_a_prediction"], gpt_a_prediction)
        self.assertEqual(calls[3][1]["auralis_prediction"], audio_prediction)
        self.assertEqual(calls[3][1]["avbench_result"]["sync_quality"], "Poor")
        self.assertEqual(
            list(calls[3][1]["deterministic_issues"]),
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

        self.assertEqual(run_stats["gpt_a"]["api_calls"], 1)
        self.assertEqual(run_stats["gpt_a"]["request_bytes"], 123)
        self.assertEqual(run_stats["gpt_a"]["raw_prediction"], [])
        self.assertEqual(
            run_stats["gpt_a"]["evidence_backed_visual_issues"],
            [],
        )
        self.assertEqual(run_stats["gemini_audio"], {})

    def test_combined_row_preserves_deterministic_metadata_issue(self):
        metadata_issue = {
            "可定位性": "否",
            "置信度": "高",
            "问题说明": "用户反馈要求1080P，实际分辨率为720×1280。",
            "问题类型": "清晰度异常",
            "时间区间": "0.00s - 13.04s",
            "关键帧秒": "",
            "BBox": "",
        }
        synthesis_call = {}

        def fake_synthesis(**kwargs):
            synthesis_call.update(kwargs)
            return json.dumps([metadata_issue], ensure_ascii=False)

        with (
            mock.patch.object(
                runner,
                "evaluate_visual_metadata_constraints",
                return_value=[metadata_issue],
            ),
            mock.patch.object(runner.gpt_a, "run_agent", return_value="[]"),
            mock.patch.object(runner, "run_audio_row", return_value="[]"),
            mock.patch.object(
                runner,
                "run_avbench_row",
                return_value={"success": True, "sync_decision": "aligned_or_no_large_offset"},
            ),
            mock.patch.object(
                runner,
                "synthesize_predictions",
                side_effect=fake_synthesis,
            ),
        ):
            result = runner.run_combined_row(
                {
                    "user_prompt": "不要降低画质",
                    "用户反馈": "客户要求直出1080P",
                    "generated_video_url": "video.mp4",
                },
                {
                    "user_prompt": "不要降低画质",
                    "generated_video_url": "video.mp4",
                },
                api_url="https://example.test/chat/completions",
                api_key="token",
                gpt_a_model="gpt-model",
                gemini_model="gemini-model",
                timeout=30,
                api_retries=1,
                max_gpt_a_agent_steps=2,
            )

        self.assertEqual(json.loads(result), [metadata_issue])
        self.assertEqual(
            list(synthesis_call["deterministic_issues"]),
            [metadata_issue],
        )
        self.assertEqual(
            json.loads(synthesis_call["metadata_prediction"]),
            [metadata_issue],
        )

    def test_preserver_deduplicates_reworded_metadata_facts(self):
        organized = [
            {
                "可定位性": "是",
                "置信度": "高",
                "问题说明": (
                    "用户反馈提出1080P目标；ffprobe显示实际分辨率为720×1280，"
                    "低于最低1080×1920规格，画面细节也偏软。"
                ),
                "问题类型": "清晰度异常",
                "时间区间": "0.00s - 13.04s",
                "关键帧秒": "3.00",
                "BBox": "<bbox>0,0,1,1</bbox>",
            },
            {
                "可定位性": "否",
                "置信度": "高",
                "问题说明": (
                    "prompt要求持续15秒，ffprobe显示实际时长为13.04秒，"
                    "短于目标并超过0.75秒容差。"
                ),
                "问题类型": "时序错误",
                "时间区间": "0.00s - 13.04s（目标至15.00s）",
                "关键帧秒": "",
                "BBox": "",
            },
        ]
        required = [
            {
                **organized[0],
                "可定位性": "否",
                "问题说明": (
                    "用户反馈明确提出1080P输出目标；实际分辨率为720×1280，"
                    "低于最低1080×1920规格。"
                ),
                "关键帧秒": "",
                "BBox": "",
            },
            {
                **organized[1],
                "问题说明": (
                    "用户prompt要求视频时长为15秒；实际时长为13.04秒，"
                    "差值超过0.75秒容差。"
                ),
                "时间区间": "0.00s - 13.04s",
            },
        ]

        preserved = json.loads(
            runner.preserve_deterministic_issues(
                json.dumps(organized, ensure_ascii=False),
                required,
            )
        )

        self.assertEqual(preserved, organized)

    def test_synthesis_runs_safe_dedup_after_required_issue_preservation(self):
        organized = {
            "可定位性": "是",
            "置信度": "高",
            "问题说明": (
                "用户反馈要求1080P；ffprobe显示实际分辨率为720×1280，"
                "低于最低1080×1920规格。"
            ),
            "问题类型": "清晰度异常",
            "时间区间": "0.00s - 13.04s",
            "关键帧秒": "3.00",
            "BBox": "<bbox>0,0,1,1</bbox>",
        }
        required = {
            **organized,
            "可定位性": "否",
            "问题说明": (
                "用户反馈明确提出1080P目标；实际分辨率为720×1280，"
                "低于最低1080×1920规格。"
            ),
            "关键帧秒": "",
            "BBox": "",
        }
        fact_id = runner.build_synthesis_fact_registry([required])[0]["fact_id"]
        organized_with_coverage = {
            **organized,
            "covered_fact_ids": [fact_id],
        }
        with mock.patch(
            "agents.auralis.runner.final_chat_completion",
            return_value=json.dumps([organized_with_coverage], ensure_ascii=False),
        ):
            result = runner.synthesize_predictions(
                user_prompt="不要降低画质",
                gpt_a_prediction="[]",
                auralis_prediction="[]",
                metadata_prediction=json.dumps([required], ensure_ascii=False),
                api_url="https://example.test/chat/completions",
                api_key="token",
                model="gpt-model",
                timeout=30,
                api_retries=1,
                deterministic_issues=[required],
            )

        self.assertEqual(json.loads(result), [organized])

    def test_fact_coverage_does_not_reappend_rephrased_missing_caption(self):
        required = {
            "可定位性": "否",
            "置信度": "高",
            "问题说明": (
                "预期在 4.18s - 5.14s 处显示与实际语音“5点到头了”相吻合的"
                "台词字幕，实际视频画面中并未提供该台词字幕，导致字幕缺失且"
                "与实际语音不一致。"
            ),
            "问题类型": "文字质量问题",
            "时间区间": "4.18s - 5.14s",
            "关键帧秒": "",
            "BBox": "",
        }
        organized = {
            **required,
            "问题说明": (
                "4.18s - 5.14s 处实际语音为“5点到头了”，但视频画面中未提供"
                "与该语音相吻合的台词字幕，存在字幕缺失且与实际语音不一致的问题。"
            ),
        }
        fact_id = runner.build_synthesis_fact_registry([required])[0]["fact_id"]
        response_issue = {
            **organized,
            "covered_fact_ids": [fact_id],
        }
        stats = {}

        with mock.patch(
            "agents.auralis.runner.final_chat_completion",
            return_value=json.dumps([response_issue], ensure_ascii=False),
        ):
            result = runner.synthesize_predictions(
                user_prompt="检查视频",
                gpt_a_prediction="[]",
                auralis_prediction=json.dumps([required], ensure_ascii=False),
                api_url="https://example.test/chat/completions",
                api_key="token",
                model="gpt-model",
                timeout=30,
                api_retries=1,
                run_stats=stats,
                deterministic_issues=[required],
            )

        self.assertEqual(json.loads(result), [organized])
        self.assertEqual(stats["covered_fact_ids"], [fact_id])
        self.assertEqual(stats["missing_fact_ids"], [])

    def test_fact_coverage_appends_only_a_truly_missing_fact(self):
        first = {
            "可定位性": "否",
            "置信度": "高",
            "问题说明": "实际存在字幕缺失。",
            "问题类型": "文字质量问题",
            "时间区间": "1.00s - 2.00s",
            "关键帧秒": "",
            "BBox": "",
        }
        second = {
            **first,
            "问题说明": "实际存在角色声音绑定错误。",
            "问题类型": "音频质量问题",
            "时间区间": "3.00s - 4.00s",
        }
        registry = runner.build_synthesis_fact_registry([first, second])
        first_id = next(
            record["fact_id"] for record in registry if record["issue"] == first
        )
        organized = {**first, "covered_fact_ids": [first_id]}
        stats = {}

        preserved = json.loads(
            runner.preserve_synthesis_fact_coverage(
                [{key: organized[key] for key in runner.OUTPUT_KEYS}],
                [organized["covered_fact_ids"]],
                registry,
                run_stats=stats,
            )
        )

        self.assertEqual(preserved, [first, second])
        self.assertEqual(len(stats["missing_fact_ids"]), 1)

    def test_prediction_rejects_issue_without_required_evidence(self):
        with self.assertRaisesRegex(ValueError, "问题说明"):
            runner.parse_prediction(
                json.dumps(
                    [
                        {
                            "可定位性": "否",
                            "置信度": "高",
                            "问题说明": "",
                            "问题类型": "音频质量问题",
                            "时间区间": "0s - 2s",
                            "关键帧秒": "",
                            "BBox": "",
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
                            "可定位性": "否",
                            "置信度": "低",
                            "问题说明": "存在明确断音。",
                            "问题类型": "音频质量问题",
                            "时间区间": "0s - 2s",
                            "关键帧秒": "",
                            "BBox": "",
                        }
                    ],
                    ensure_ascii=False,
                )
            )

    def test_prediction_normalizes_minute_style_time_range_to_seconds(self):
        issue = {
            "可定位性": "否",
            "置信度": "高",
            "问题说明": "预期只有喘息声，实际出现了清晰台词。",
            "问题类型": "音频质量问题",
            "时间区间": "00:03s - 01:04.5s",
            "关键帧秒": "",
            "BBox": "",
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
        model = "test-model"
        api_url = "https://example.test/chat/completions"
        expected_bytes = len(
            json.dumps(
                runner.build_chat_payload(model, parts),
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
                api_url,
                "token",
                model,
                parts,
                timeout=1,
                max_attempts=2,
            )
        self.assertEqual(message["_api_attempts"], 2)
        self.assertEqual(message["_request_bytes"], expected_bytes * 2)

if __name__ == "__main__":
    unittest.main()
