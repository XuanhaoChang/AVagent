import json
import unittest

from agents.auralis.schemas import AuralisEvidence
from agents.auralis.gemini_backend import (
    SYSTEM_MESSAGE,
    build_chat_payload,
    build_prompt,
    evidence_json,
    parse_prediction,
)
from tools.speech_subtitle_alignment.schemas import AlignmentResult
from tools.speech_transcription.schemas import SpeechSegment, SpeechTranscript
from tools.subtitle_extraction.schemas import SubtitleTrack


class AuralisGeminiBackendTest(unittest.TestCase):
    def test_prompt_contains_local_tool_evidence_and_evidence_boundary(self):
        evidence_json = json.dumps(
            {
                "asr": [{"start_sec": 0.0, "end_sec": 1.0, "text": "你好"}],
                "subtitles": [{"start_sec": 0.0, "end_sec": 1.0, "text": "你号"}],
                "alignment": [{"issue_type": "wrong_text"}],
            },
            ensure_ascii=False,
        )

        prompt = build_prompt("人物说你好", evidence_json)

        self.assertIn("人物说你好", prompt)
        self.assertIn("本地专家工具候选证据", prompt)
        self.assertIn("wrong_text", prompt)
        self.assertIn("ASR 是台词内容和读音问题的判定依据", prompt)
        self.assertIn("不要求回听完整 WAV", prompt)
        self.assertIn("ASR 是本流程对实际台词内容和读音差异的检测与判定依据", SYSTEM_MESSAGE)
        self.assertIn("原文锚点预期“发霉”、ASR 为“发膜", SYSTEM_MESSAGE)
        self.assertIn("不得自行依赖引号", SYSTEM_MESSAGE)
        self.assertIn("no_reference_dialogue", SYSTEM_MESSAGE)
        self.assertIn("observed_preferred", SYSTEM_MESSAGE)
        self.assertIn("expected_preferred", SYSTEM_MESSAGE)
        self.assertIn("人物 -> 匿名 spk", SYSTEM_MESSAGE)
        self.assertIn("台词内部连续" , SYSTEM_MESSAGE)
        self.assertIn("不能笼统写成“整句应由 spk1 完整发出”", SYSTEM_MESSAGE)
        self.assertIn("同一个 spk 是否跨越了 prompt 中明确属于", SYSTEM_MESSAGE)
        self.assertIn("把“匿名”误解成“不可用于绑定检查”", SYSTEM_MESSAGE)
        self.assertIn("prompt_speech_plan", SYSTEM_MESSAGE)
        self.assertIn("expected_speaker_count", SYSTEM_MESSAGE)
        self.assertIn("speaker_diarization.speaker_turns", SYSTEM_MESSAGE)
        self.assertIn("granularity_conflict=true", SYSTEM_MESSAGE)
        self.assertIn("单个采样帧中孤立出现的单字符 OCR", SYSTEM_MESSAGE)
        self.assertNotIn("思考过程及标准答案", prompt)

    def test_local_evidence_exposes_prompt_anchored_candidate_decision(self):
        evidence = AuralisEvidence(
            media_metadata={"has_audio": True, "duration_sec": 2.0},
            transcript=SpeechTranscript(
                language="zh",
                segments=(SpeechSegment(0.0, 1.0, "发膜", "medium"),),
                backend="fake",
                model="fake",
                device="cpu",
            ),
            subtitles=SubtitleTrack(segments=(), backend="fake"),
            alignment=AlignmentResult(issues=()),
            constrained_asr={
                "status": "scored",
                "candidate_scores": [
                    {
                        "prompt_start": 12,
                        "prompt_end": 14,
                        "prompt_source_text": "发霉",
                        "observed_text": "发膜",
                        "expected_text": "发霉",
                        "decision": "observed_preferred",
                    }
                ],
            },
        )

        payload = json.loads(evidence_json(evidence))

        self.assertEqual(payload["constrained_asr"]["status"], "scored")
        self.assertEqual(
            payload["constrained_asr"]["candidate_scores"][0]["decision"],
            "observed_preferred",
        )
        self.assertTrue(
            payload["subtitle_evidence_policy"][
                "isolated_single_frame_single_character_is_unverified"
            ]
        )

    def test_local_evidence_preserves_anonymous_speaker_labels(self):
        evidence = AuralisEvidence(
            media_metadata={"has_audio": True, "duration_sec": 2.0},
            transcript=SpeechTranscript(
                language="zh",
                segments=(
                    SpeechSegment(0.0, 1.0, "谢谢妈", "medium", speaker=0),
                    SpeechSegment(1.1, 2.0, "建军你快看", "medium", speaker=1),
                ),
                backend="fake",
                model="fake",
                device="cpu",
            ),
            subtitles=SubtitleTrack(segments=(), backend="fake"),
            alignment=AlignmentResult(issues=()),
        )

        payload = json.loads(evidence_json(evidence))

        self.assertEqual(payload["asr"]["segments"][0]["speaker"], 0)
        self.assertEqual(payload["asr"]["segments"][1]["speaker"], 1)

    def test_payload_uses_gateway_compatible_gemini_contents(self):
        payload = build_chat_payload("gemini-3.5-flash", [{"text": "test"}])

        self.assertEqual(payload["model"], "gemini-3.5-flash")
        self.assertEqual(payload["contents"][0]["role"], "user")
        self.assertEqual(payload["contents"][0]["parts"][0], {"text": SYSTEM_MESSAGE})
        self.assertNotIn("messages", payload)

    def test_prediction_rejects_non_audio_problem_types(self):
        issue = {
            "可定位性": "否",
            "置信度": "高",
            "问题说明": "动作错误",
            "问题类型": "动作异常",
            "时间区间": "0s - 1s",
            "关键帧秒": "",
            "BBox": "",
        }

        with self.assertRaisesRegex(ValueError, "音频质量问题或文字质量问题"):
            parse_prediction(json.dumps([issue], ensure_ascii=False))

    def test_prediction_rejects_time_outside_media_duration(self):
        issue = {
            "可定位性": "否",
            "置信度": "高",
            "问题说明": "存在明确断音。",
            "问题类型": "音频质量问题",
            "时间区间": "0s - 9999s",
            "关键帧秒": "",
            "BBox": "",
        }

        with self.assertRaisesRegex(ValueError, "视频时长"):
            parse_prediction(
                json.dumps([issue], ensure_ascii=False),
                duration_sec=4.04,
            )

    def test_prediction_accepts_evidence_supported_subsecond_time(self):
        issue = {
            "可定位性": "否",
            "置信度": "高",
            "问题说明": "存在明确断音。",
            "问题类型": "音频质量问题",
            "时间区间": "0.13s - 1.37s",
            "关键帧秒": "",
            "BBox": "",
        }

        parsed = parse_prediction(
            json.dumps([issue], ensure_ascii=False),
            duration_sec=4.04,
        )
        self.assertEqual(json.loads(parsed)[0]["时间区间"], "0.13s - 1.37s")

    def test_prediction_rejects_missing_output_fields(self):
        issue = {
            "置信度": "高",
            "问题说明": "存在明确断音。",
            "问题类型": "音频质量问题",
            "时间区间": "0s - 1s",
        }

        with self.assertRaisesRegex(ValueError, "必填字段"):
            parse_prediction(json.dumps([issue], ensure_ascii=False))

    def test_prediction_rejects_zero_length_time_range(self):
        issue = {
            "可定位性": "否",
            "置信度": "高",
            "问题说明": "存在明确断音。",
            "问题类型": "音频质量问题",
            "时间区间": "1s - 1s",
            "关键帧秒": "",
            "BBox": "",
        }

        with self.assertRaisesRegex(ValueError, "时间区间格式无效"):
            parse_prediction(json.dumps([issue], ensure_ascii=False))


if __name__ == "__main__":
    unittest.main()
