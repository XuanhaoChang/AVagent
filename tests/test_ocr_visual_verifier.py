import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from agents.auralis import runner as auralis_runner
from agents import ocr_visual_verifier
from agents.ocr_visual_verifier import (
    apply_verdicts,
    build_ocr_issue_candidates,
    build_verification_messages,
    parse_verdicts,
)


def _issue(description: str, *, issue_type: str = "文字质量问题"):
    return {
        "可定位性": "否",
        "置信度": "高",
        "问题说明": description,
        "问题类型": issue_type,
        "时间区间": "2.50s - 4.50s",
        "关键帧秒": "",
        "BBox": "",
    }


def _segment(
    text: str,
    confidence: float,
    bbox=(0.35, 0.80, 0.86, 0.84),
    start=2.5,
    end=3.0,
):
    return {
        "start_sec": start,
        "end_sec": end,
        "text": text,
        "bbox": list(bbox),
        "confidence": confidence,
        "source": "burned_in",
    }


class OcrVisualVerifierTest(unittest.TestCase):
    def test_issue_quoted_gibberish_outranks_unrelated_high_confidence_ui(self):
        issue = _issue(
            "实际残留乱码“古古古由古古古古”和“會會會會會古會會”。"
        )
        segments = [
            _segment("US", 0.999, bbox=(0.05, 0.68, 0.31, 0.79)),
            _segment("BASE SIZE", 0.999, bbox=(0.07, 0.81, 0.32, 0.84)),
            _segment("古古古由古古古古", 0.58),
            _segment(
                "會會會會會古會會",
                0.76,
                bbox=(0.35, 0.84, 0.86, 0.88),
            ),
        ]

        candidates = build_ocr_issue_candidates(
            [issue],
            subtitle_segments=segments,
        )

        self.assertEqual(len(candidates), 1)
        self.assertEqual(
            [item["text"] for item in candidates[0].segments[:2]],
            ["會會會會會古會會", "古古古由古古古古"],
        )

    def test_non_text_verdict_removes_only_the_ocr_candidate(self):
        text_issue = _issue("OCR 将星级图标识别成乱码。")
        audio_issue = _issue("实际音频存在爆音。", issue_type="音频质量问题")
        candidates = build_ocr_issue_candidates(
            [text_issue, audio_issue],
            subtitle_segments=[_segment("古古古古", 0.55)],
        )

        accepted, reviews = apply_verdicts(
            [text_issue, audio_issue],
            candidates=candidates,
            verdicts={
                "ocr_001": {
                    "decision": "contradicted",
                    "region_type": "non_text_texture",
                    "reason": "局部实际为星级图标。",
                }
            },
        )

        self.assertEqual(accepted, [audio_issue])
        self.assertEqual(reviews[0]["decision"], "contradicted")

    def test_supported_real_gibberish_remains_an_issue(self):
        issue = _issue("报价单中确实出现无法辨认的损坏字形。")
        candidates = build_ocr_issue_candidates(
            [issue],
            subtitle_segments=[_segment("古會結寡", 0.57)],
        )

        accepted, reviews = apply_verdicts(
            [issue],
            candidates=candidates,
            verdicts={
                "ocr_001": {
                    "decision": "supported",
                    "region_type": "scene_text_or_ui",
                    "reason": "单据区域可见真实但损坏的连续字形。",
                }
            },
        )

        self.assertEqual(accepted, [issue])
        self.assertEqual(reviews[0]["region_type"], "scene_text_or_ui")

    def test_deterministic_alignment_issue_bypasses_model_written_ocr_gate(self):
        issue = _issue("ASR 为“仨”，OCR 字幕为“三”。")

        candidates = build_ocr_issue_candidates(
            [issue],
            subtitle_segments=[_segment("你都修三小时了", 0.999)],
            deterministic_issues=[issue],
        )

        self.assertEqual(candidates, ())

    def test_missing_or_malformed_verdict_becomes_inconclusive(self):
        verdicts = parse_verdicts(
            '[{"candidate_id":"ocr_001","decision":"maybe",'
            '"region_type":"text"}]',
            candidate_ids=["ocr_001", "ocr_002"],
        )

        self.assertEqual(verdicts["ocr_001"]["decision"], "inconclusive")
        self.assertEqual(verdicts["ocr_001"]["region_type"], "unknown")
        self.assertEqual(verdicts["ocr_002"]["decision"], "inconclusive")

    def test_visual_evidence_uses_segment_midpoint_not_transition_boundary(self):
        issue = _issue("字幕显示为“你的预算直接翻一碚”。")
        issue["时间区间"] = "13.50s - 15.00s"
        candidates = build_ocr_issue_candidates(
            [issue],
            subtitle_segments=[
                _segment(
                    "你的预算直接翻一碚",
                    0.996,
                    start=13.5,
                    end=15.0,
                )
            ],
        )
        with TemporaryDirectory() as temp_dir, mock.patch.object(
            ocr_visual_verifier.visual_agent,
            "probe_video",
            return_value={"duration_sec": 15.0, "width": 1080, "height": 1920},
        ), mock.patch.object(
            ocr_visual_verifier.visual_agent,
            "extract_frame",
            return_value={"description": "frame"},
        ) as extract_frame, mock.patch.object(
            ocr_visual_verifier.visual_agent,
            "extract_crop",
            return_value={"description": "crop"},
        ) as extract_crop, mock.patch.object(
            ocr_visual_verifier.visual_agent,
            "image_data_url",
            return_value="data:image/jpeg;base64,AA==",
        ):
            build_verification_messages(
                user_prompt="",
                candidates=candidates,
                video_path=Path("/tmp/sample.mp4"),
                temp_dir=Path(temp_dir),
            )

        self.assertEqual(
            extract_frame.call_args.args[3]["timestamp_sec"],
            14.25,
        )
        self.assertEqual(
            extract_crop.call_args.args[3]["timestamp_sec"],
            14.25,
        )

    def test_runner_gate_uses_accepted_visual_verifier_result(self):
        issue = _issue("仪表盘数字被错误当成字幕。")
        stats = {
            "auralis_evidence": {
                "subtitles": {"segments": [_segment("22", 0.92)]}
            },
            "deterministic_issues": [],
        }
        run_stats = {}
        with mock.patch.object(
            auralis_runner.gpt_a,
            "ensure_video",
            return_value=Path("/tmp/sample.mp4"),
        ), mock.patch.object(
            auralis_runner,
            "verify_auralis_ocr_issues",
            return_value=(
                [],
                {
                    "status": "ok",
                    "candidate_count": 1,
                    "candidate_reviews": [
                        {"candidate_id": "ocr_001", "decision": "contradicted"}
                    ],
                },
            ),
        ):
            result = auralis_runner.gate_auralis_ocr_prediction(
                json.dumps([issue], ensure_ascii=False),
                input_data={
                    "generated_video_url": "sample.mp4",
                    "user_prompt": "无字幕",
                    "reference_image_urls": [],
                },
                auralis_stats=stats,
                api_url="https://example.test",
                api_key="token",
                model="gpt",
                timeout=1,
                api_retries=1,
                run_stats=run_stats,
            )

        self.assertEqual(json.loads(result), [])
        self.assertEqual(run_stats["ocr_visual_verifier"]["status"], "ok")

    def test_verifier_failure_abstains_without_dropping_deterministic_or_audio(self):
        deterministic = _issue("ASR 为“仨”，OCR 字幕为“三”。")
        unverified = _issue("OCR 将仪表盘数字写成字幕。")
        audio = _issue("实际音频存在爆音。", issue_type="音频质量问题")
        stats = {
            "auralis_evidence": {
                "subtitles": {"segments": [_segment("22", 0.92)]}
            },
            "deterministic_issues": [deterministic],
        }
        run_stats = {}
        with mock.patch.object(
            auralis_runner.gpt_a,
            "ensure_video",
            return_value=Path("/tmp/sample.mp4"),
        ), mock.patch.object(
            auralis_runner,
            "verify_auralis_ocr_issues",
            side_effect=RuntimeError("temporary verifier failure"),
        ):
            result = auralis_runner.gate_auralis_ocr_prediction(
                json.dumps(
                    [deterministic, unverified, audio],
                    ensure_ascii=False,
                ),
                input_data={
                    "generated_video_url": "sample.mp4",
                    "user_prompt": "",
                    "reference_image_urls": [],
                },
                auralis_stats=stats,
                api_url="https://example.test",
                api_key="token",
                model="gpt",
                timeout=1,
                api_retries=1,
                run_stats=run_stats,
            )

        self.assertEqual(json.loads(result), [deterministic, audio])
        self.assertEqual(run_stats["ocr_visual_verifier"]["status"], "failed")


if __name__ == "__main__":
    unittest.main()
