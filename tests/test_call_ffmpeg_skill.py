import argparse
import unittest

import call_ffmpeg_skill as runner


class CallFfmpegSkillTest(unittest.TestCase):
    def test_profile_settings_lock_ablation_variables(self):
        args = argparse.Namespace(
            profile="harness_c",
            video_frame_fps=None,
            max_video_frames=None,
            audio_mode=None,
            enable_local_crop=None,
        )
        runner.apply_profile_defaults(args)
        self.assertEqual(args.video_frame_fps, 1.0)
        self.assertEqual(args.max_video_frames, 48)
        self.assertEqual(args.audio_mode, "direct")
        self.assertTrue(args.enable_local_crop)

    def test_direct_audio_part_uses_input_audio_without_data_url_prefix(self):
        parts = runner.build_audio_parts("direct", b"wav-bytes", "")
        self.assertEqual(parts[0]["type"], "text")
        self.assertEqual(parts[1]["type"], "input_audio")
        self.assertEqual(parts[1]["input_audio"]["format"], "wav")
        self.assertNotIn("data:", parts[1]["input_audio"]["data"])

    def test_transcript_part_labels_asr_as_tool_evidence(self):
        parts = runner.build_audio_parts("transcript", b"", "0.0-1.0 你好")
        text = parts[0]["text"]
        self.assertIn("ASR 工具证据", text)
        self.assertIn("0.0-1.0 你好", text)

    def test_accumulates_numeric_usage_across_agent_steps(self):
        stats = {}
        runner.accumulate_usage(
            stats,
            {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
            request_bytes=100,
        )
        runner.accumulate_usage(
            stats,
            {"prompt_tokens": 20, "completion_tokens": 4, "total_tokens": 24},
            request_bytes=200,
        )
        self.assertEqual(stats["api_calls"], 2)
        self.assertEqual(stats["prompt_tokens"], 30)
        self.assertEqual(stats["total_tokens"], 37)
        self.assertEqual(stats["request_bytes"], 300)

    def test_expert_evidence_is_labeled_and_not_treated_as_truth(self):
        parts = runner.build_expert_evidence_parts('{"ocr":[{"text":"示例"}]}')
        self.assertIn("专家工具候选证据", parts[0]["text"])
        self.assertIn("不是问题必然存在的证明", parts[0]["text"])

    def test_no_audio_mode_forbids_audio_issues_and_placeholder_findings(self):
        instruction = runner.audio_evidence_instruction("none")
        self.assertIn("不得输出任何音频相关问题", instruction)
        self.assertIn("不得输出“缺少音频证据”", instruction)
        self.assertIn("台词内容", instruction)
        self.assertIn("音色", instruction)
        self.assertIn("背景音乐", instruction)
        self.assertIn("声画同步", instruction)

    def test_audio_enabled_modes_do_not_forbid_audio_findings(self):
        for mode in ("direct", "transcript"):
            instruction = runner.audio_evidence_instruction(mode)
            self.assertNotIn("不得输出任何音频相关问题", instruction)
            self.assertIn(f"本次音频证据模式为 {mode}", instruction)

    def test_prediction_accepts_empty_array(self):
        self.assertEqual(runner.parse_prediction("[]"), "[]")

    def test_none_audio_mode_filters_audio_and_missing_evidence_issues(self):
        prediction = runner.filter_prediction_for_audio_mode(
            '[{"问题类型":"其他","问题说明":"缺少音频证据，无法核实女声是否变成男声"},'
            '{"问题类型":"音频质量问题","问题说明":"存在背景音乐"},'
            '{"问题类型":"其他","问题说明":"角色说错了台词"},'
            '{"问题类型":"其他","问题说明":"语言与要求不一致"},'
            '{"问题类型":"其他","问题说明":"缺少可靠证据，无法确认台词是否正确"},'
            '{"问题类型":"动作异常","问题说明":"人物没有抬手"}]',
            "none",
        )
        self.assertEqual(
            runner.json.loads(prediction),
            [{"问题类型": "动作异常", "问题说明": "人物没有抬手"}],
        )

    def test_none_audio_mode_preserves_visual_language_and_music_scene_issues(self):
        prediction = runner.filter_prediction_for_audio_mode(
            '[{"问题类型":"镜头语言问题","问题说明":"景别组合不符合要求"},'
            '{"问题类型":"场景问题","问题说明":"音乐节场景中的舞台位置错误"}]',
            "none",
        )
        self.assertEqual(
            runner.json.loads(prediction),
            [
                {"问题类型": "镜头语言问题", "问题说明": "景别组合不符合要求"},
                {"问题类型": "场景问题", "问题说明": "音乐节场景中的舞台位置错误"},
            ],
        )

if __name__ == "__main__":
    unittest.main()
