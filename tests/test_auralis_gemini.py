import io
import json
import unittest
import wave

from agents.auralis.gemini_backend import (
    SYSTEM_MESSAGE,
    build_chat_payload,
    build_prompt,
    parse_prediction,
    split_wav_bytes,
)


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
        self.assertIn("不得把 ASR、OCR 或对齐结果直接当作问题真值", prompt)
        self.assertNotIn("思考过程及标准答案", prompt)

    def test_split_wav_bytes_preserves_all_audio_frames(self):
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

        segments = split_wav_bytes(source.getvalue(), segment_seconds=1.0)

        self.assertEqual(
            [
                (segment["start_sec"], segment["end_sec"])
                for segment in segments
            ],
            [(0.0, 1.0), (1.0, 2.0), (2.0, 2.25)],
        )

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
            "置信度": "高",
            "问题说明": "存在明确断音。",
            "问题类型": "音频质量问题",
            "时间区间": "0s - 9999s",
        }

        with self.assertRaisesRegex(ValueError, "视频时长"):
            parse_prediction(
                json.dumps([issue], ensure_ascii=False),
                duration_sec=4.04,
            )

    def test_prediction_rejects_finer_time_than_audio_segments(self):
        issue = {
            "置信度": "高",
            "问题说明": "存在明确断音。",
            "问题类型": "音频质量问题",
            "时间区间": "0.13s - 1s",
        }

        with self.assertRaisesRegex(ValueError, "音频分片边界"):
            parse_prediction(
                json.dumps([issue], ensure_ascii=False),
                duration_sec=4.04,
                segment_seconds=1.0,
            )

    def test_prediction_accepts_actual_final_audio_boundary(self):
        issue = {
            "置信度": "高",
            "问题说明": "视频要求无声，实际存在持续音调。",
            "问题类型": "音频质量问题",
            "时间区间": "0s - 2.05s",
        }

        parsed = parse_prediction(
            json.dumps([issue], ensure_ascii=False),
            duration_sec=2.05,
            allowed_boundaries=(0.0, 1.0, 2.0, 2.048),
        )

        self.assertEqual(json.loads(parsed)[0]["时间区间"], "0s - 2.05s")

    def test_prediction_uses_explicit_boundaries_without_duration(self):
        issue = {
            "置信度": "高",
            "问题说明": "存在明确断音。",
            "问题类型": "音频质量问题",
            "时间区间": "0.13s - 1s",
        }

        with self.assertRaisesRegex(ValueError, "音频分片边界"):
            parse_prediction(
                json.dumps([issue], ensure_ascii=False),
                allowed_boundaries=(0.0, 1.0, 2.0),
            )

    def test_prediction_rejects_zero_length_time_range(self):
        issue = {
            "置信度": "高",
            "问题说明": "存在明确断音。",
            "问题类型": "音频质量问题",
            "时间区间": "1s - 1s",
        }

        with self.assertRaisesRegex(ValueError, "时间区间格式无效"):
            parse_prediction(json.dumps([issue], ensure_ascii=False))


if __name__ == "__main__":
    unittest.main()
