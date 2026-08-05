import json
import unittest
from pathlib import Path
from unittest import mock

from agents import seed_lite_specialist as specialist


def issue(
    description: str,
    *,
    problem_type: str = "动作异常",
    confidence: str = "高",
    time_range: str = "1s - 3s",
    keyframe: str = "2",
    bbox: str = "<bbox>100 100 900 900</bbox>",
):
    return {
        "可定位性": "是",
        "置信度": confidence,
        "问题说明": description,
        "问题类型": problem_type,
        "时间区间": time_range,
        "关键帧秒": keyframe,
        "BBox": bbox,
    }


class SeedLiteSpecialistTest(unittest.TestCase):
    def filter(self, candidates, *, evidence=None, tool_calls=()):
        patches = [
            mock.patch.object(specialist, "_video_size", return_value=(1920, 1080)),
        ]
        if evidence is not None:
            patches.append(
                mock.patch.object(
                    specialist,
                    "analyze_local_motion_evidence",
                    return_value=evidence,
                )
            )
        with patches[0]:
            if len(patches) == 2:
                with patches[1]:
                    return specialist.filter_seed_lite_candidates(
                        json.dumps(candidates, ensure_ascii=False),
                        video_path=Path("video.mp4"),
                        tool_calls=tool_calls,
                    )
            return specialist.filter_seed_lite_candidates(
                json.dumps(candidates, ensure_ascii=False),
                video_path=Path("video.mp4"),
                tool_calls=tool_calls,
            )

    def test_logo_requires_matching_high_resolution_visual_call(self):
        candidate = issue(
            "prompt要求无logo，实际T恤出现品牌字标BALENCIEN。",
            problem_type="文字质量问题",
            time_range="10s - 14s",
            keyframe="11.5",
            bbox="<bbox>177 652 802 727</bbox>",
        )
        accepted, reviews = self.filter(
            [candidate],
            tool_calls=[
                {
                    "name": "extract_frame",
                    "arguments": {"timestamp_sec": 11.5},
                    "ok": True,
                }
            ],
        )
        self.assertEqual(len(accepted), 1)
        self.assertEqual(reviews[0]["decision"], "supported")
        self.assertEqual(
            accepted[0]["BBox"],
            "<bbox>0.1770,0.6520,0.8020,0.7270</bbox>",
        )

    def test_seed_lite_equivalent_localization_and_bbox_formats_are_normalized(self):
        candidate = issue(
            "模特转身无过渡并发生跳变。",
            problem_type="连续动作不符合基本物理规律",
            time_range="5.0-7.5秒",
            keyframe="7.0秒",
            bbox="0.17 0.0 0.78 0.99",
        )
        candidate["可定位性"] = "可明确识别"
        evidence = {
            "status": "ok",
            "abrupt_event_count": 1,
            "cut_like_event_count": 0,
            "max_frame_difference": 0.17,
            "max_lost_track_ratio": 0.2,
        }

        accepted, _reviews = self.filter([candidate], evidence=evidence)

        self.assertEqual(accepted[0]["可定位性"], "是")
        self.assertEqual(accepted[0]["关键帧秒"], "7")
        self.assertEqual(
            accepted[0]["BBox"],
            "<bbox>0.1700,0.0000,0.7800,0.9900</bbox>",
        )

    def test_negative_localization_word_is_not_treated_as_positive(self):
        candidate = issue("模特转身无过渡并发生跳变。")
        candidate["可定位性"] = "无法精准定位"

        accepted, reviews = self.filter([candidate])

        self.assertEqual(accepted, [])
        self.assertEqual(reviews[0]["reason"], "seed_candidate_not_localizable")

    def test_subtitle_that_only_repeats_no_logo_requirement_is_out_of_scope(self):
        candidate = issue(
            "prompt要求无水印、无字幕、无logo，实际底部持续叠加字幕。",
            problem_type="文字质量问题",
        )
        accepted, reviews = self.filter([candidate])
        self.assertEqual(accepted, [])
        self.assertEqual(
            reviews[0]["reason"],
            "seed_candidate_outside_specialist_scope",
        )

    def test_action_jump_is_supported_by_local_temporal_evidence(self):
        evidence = {
            "status": "ok",
            "abrupt_event_count": 1,
            "cut_like_event_count": 0,
            "max_frame_difference": 0.17,
            "max_lost_track_ratio": 0.2,
        }
        accepted, reviews = self.filter(
            [issue("人物转身无过渡并发生姿态跳变。")],
            evidence=evidence,
        )
        self.assertEqual(len(accepted), 1)
        self.assertEqual(reviews[0]["decision"], "supported")

    def test_smooth_action_candidate_is_contradicted(self):
        evidence = {
            "status": "ok",
            "abrupt_event_count": 0,
            "cut_like_event_count": 0,
            "max_frame_difference": 0.06,
            "max_lost_track_ratio": 0.2,
        }
        accepted, reviews = self.filter(
            [issue("人物转身缺少过渡，动作不连贯。")],
            evidence=evidence,
        )
        self.assertEqual(accepted, [])
        self.assertEqual(reviews[0]["decision"], "contradicted")

    def test_unverified_penetration_stays_inconclusive(self):
        evidence = {
            "status": "ok",
            "noncut_structure_instability_count": 0,
            "max_frame_difference": 0.24,
            "max_lost_track_ratio": 0.2,
        }
        accepted, reviews = self.filter(
            [issue("腕表与手腕发生穿模和实体边界穿插。")],
            evidence=evidence,
        )
        self.assertEqual(accepted, [])
        self.assertEqual(reviews[0]["decision"], "inconclusive")

    def test_medium_confidence_candidate_is_rejected_before_local_analysis(self):
        accepted, reviews = self.filter(
            [issue("走路时腿部畸变。", confidence="中")]
        )
        self.assertEqual(accepted, [])
        self.assertEqual(
            reviews[0]["reason"],
            "seed_candidate_not_high_confidence",
        )

    def test_runner_uses_scoped_prompt_and_returns_only_accepted_candidates(self):
        accepted_issue = issue("人物转身无过渡并发生跳变。")
        stats = {}
        with (
            mock.patch.object(
                specialist.visual_agent,
                "run_agent",
                return_value=json.dumps([accepted_issue], ensure_ascii=False),
            ) as run_agent,
            mock.patch.object(
                specialist.visual_agent,
                "ensure_video",
                return_value=Path("video.mp4"),
            ),
            mock.patch.object(
                specialist,
                "filter_seed_lite_candidates",
                return_value=([accepted_issue], [{"decision": "supported"}]),
            ),
        ):
            result = specialist.run_seed_lite_specialist(
                {"generated_video_url": "video.mp4"},
                api_url="https://example.test",
                api_key="token",
                timeout=30,
                api_retries=1,
                run_stats=stats,
            )

        self.assertEqual(json.loads(result), [accepted_issue])
        self.assertIn(
            "logo、商标、品牌字标",
            run_agent.call_args.kwargs["skill_text_override"],
        )
        self.assertEqual(stats["candidate_reviews"], [{"decision": "supported"}])


if __name__ == "__main__":
    unittest.main()
