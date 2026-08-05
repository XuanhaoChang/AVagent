import unittest
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

import call_ffmpeg_skill_gpt_d as legacy_runner
import numpy as np
from agents.auralis.agent import AuralisAgent, deterministic_alignment_issues
from agents.auralis.schemas import AuralisEvidence, AuralisInput, AuralisResult
from agents.auralis import runner as auralis_runner
from agents.auralis.gemini_backend import GeminiGateway
from tools.speech_subtitle_alignment.tool import check_speech_subtitle_alignment
from tools.speech_transcription.schemas import SpeechSegment, SpeechTranscript
from tools.speech_transcription.backends.faster_whisper import (
    FasterWhisperBackend,
)
from tools.speech_transcription.backends.sensevoice import SenseVoiceBackend
from tools.speech_transcription.tool import transcribe_speech
from tools.subtitle_extraction.schemas import SubtitleObservation
from tools.subtitle_extraction.backends.rapidocr import RapidOCRBackend
from tools.subtitle_extraction.tool import (
    UNVERIFIED_SINGLETON_SOURCE,
    merge_subtitle_observations,
    subtitle_evidence_for_judge,
)


class FakeTranscriber:
    def transcribe(self, audio_path: Path) -> SpeechTranscript:
        self.audio_path = audio_path
        return SpeechTranscript(
            language="zh",
            segments=(
                SpeechSegment(0.0, 1.5, "你好世界", "high"),
            ),
            backend="fake-asr",
            model="fake-model",
            device="cpu",
        )


class AudioAgentArchitectureTest(unittest.TestCase):
    def test_sensevoice_segments_preserve_anonymous_speaker_labels(self):
        segments = SenseVoiceBackend._segments(
            {
                "segments": [
                    {
                        "start_sec": 1.0,
                        "end_sec": 2.0,
                        "text": "<|zh|>你好。",
                        "speaker": 1,
                    }
                ]
            }
        )

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].text, "你好")
        self.assertEqual(segments[0].speaker, 1)

    def test_sensevoice_segments_flatten_list_valued_text(self):
        segments = SenseVoiceBackend._segments(
            {
                "segments": [
                    {
                        "start_sec": 6.92,
                        "end_sec": 15.02,
                        "text": "['Yeah.', '你都修仨小时了。']",
                        "speaker": 0,
                    }
                ]
            }
        )

        self.assertEqual(segments[0].text, "Yeah. 你都修仨小时了")
        self.assertNotIn("['", segments[0].text)

    def test_transcribe_speech_uses_injected_local_backend(self):
        backend = FakeTranscriber()

        result = transcribe_speech(Path("sample.wav"), backend=backend)

        self.assertEqual(result.segments[0].text, "你好世界")
        self.assertEqual(backend.audio_path, Path("sample.wav"))

    def test_adjacent_equal_ocr_observations_become_one_subtitle_segment(self):
        observations = [
            SubtitleObservation(0.0, "你好世界", (0.1, 0.8, 0.9, 0.95), 0.98),
            SubtitleObservation(0.5, "你好世界", (0.1, 0.8, 0.9, 0.95), 0.97),
            SubtitleObservation(1.0, "你好世界", (0.1, 0.8, 0.9, 0.95), 0.96),
        ]

        track = merge_subtitle_observations(observations, frame_interval=0.5)

        self.assertEqual(len(track.segments), 1)
        self.assertEqual(track.segments[0].text, "你好世界")
        self.assertEqual(track.segments[0].start_sec, 0.0)
        self.assertEqual(track.segments[0].end_sec, 1.5)

    def test_single_frame_single_character_ocr_is_withheld_from_judge(self):
        track = merge_subtitle_observations(
            [
                SubtitleObservation(
                    9.0,
                    "二",
                    (0.4, 0.30, 0.59, 0.44),
                    0.99,
                )
            ],
            frame_interval=0.5,
        )

        trusted, rejected = subtitle_evidence_for_judge(track)

        self.assertEqual(trusted.segments, ())
        self.assertEqual(len(rejected), 1)
        self.assertEqual(rejected[0].source, UNVERIFIED_SINGLETON_SOURCE)

    def test_repeated_single_character_ocr_remains_judge_evidence(self):
        track = merge_subtitle_observations(
            [
                SubtitleObservation(
                    9.0,
                    "二",
                    (0.4, 0.80, 0.45, 0.86),
                    0.98,
                ),
                SubtitleObservation(
                    9.5,
                    "二",
                    (0.4, 0.80, 0.45, 0.86),
                    0.99,
                ),
            ],
            frame_interval=0.5,
        )

        trusted, rejected = subtitle_evidence_for_judge(track)

        self.assertEqual(len(trusted.segments), 1)
        self.assertEqual(trusted.segments[0].source, "burned_in")
        self.assertEqual(rejected, ())

    def test_interleaved_logo_and_caption_form_two_temporal_tracks(self):
        observations = [
            SubtitleObservation(0.0, "LOGO", (0.8, 0.05, 0.98, 0.12), 0.99),
            SubtitleObservation(0.0, "你好", (0.3, 0.8, 0.7, 0.92), 0.99),
            SubtitleObservation(0.5, "LOGO", (0.8, 0.05, 0.98, 0.12), 0.99),
            SubtitleObservation(0.5, "你好", (0.3, 0.8, 0.7, 0.92), 0.99),
        ]

        track = merge_subtitle_observations(observations, frame_interval=0.5)

        self.assertEqual(len(track.segments), 2)
        self.assertEqual(
            sorted((segment.text, segment.end_sec) for segment in track.segments),
            [("LOGO", 1.0), ("你好", 1.0)],
        )

    def test_rapidocr_entries_accept_numpy_result_arrays(self):
        class Result:
            boxes = np.array(
                [[[10, 10], [100, 10], [100, 30], [10, 30]]],
                dtype=np.float32,
            )
            txts = ["你好"]
            scores = np.array([0.99], dtype=np.float32)

        entries = list(RapidOCRBackend._entries(Result()))

        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0][1], "你好")

    def test_alignment_classifies_missing_subtitle_characters(self):
        transcript = SpeechTranscript(
            language="zh",
            segments=(SpeechSegment(0.0, 2.0, "欢迎来到杭州", "high"),),
            backend="fake",
            model="fake",
            device="cpu",
        )
        subtitles = merge_subtitle_observations(
            [
                SubtitleObservation(
                    0.0,
                    "欢迎杭州",
                    (0.1, 0.8, 0.9, 0.95),
                    0.99,
                )
            ],
            frame_interval=2.0,
        )

        result = check_speech_subtitle_alignment(transcript, subtitles)

        self.assertEqual(len(result.issues), 1)
        self.assertEqual(result.issues[0].issue_type, "missing_text")
        self.assertIn("来到", result.issues[0].difference)

    def test_alignment_ignores_unrelated_logo_text_when_subtitle_matches(self):
        transcript = SpeechTranscript(
            language="zh",
            segments=(SpeechSegment(0.0, 1.0, "你好", "high"),),
            backend="fake",
            model="fake",
            device="cpu",
        )
        subtitles = merge_subtitle_observations(
            [
                SubtitleObservation(0.0, "抖音", (0.8, 0.05, 0.98, 0.12), 0.99),
                SubtitleObservation(0.0, "你好", (0.3, 0.8, 0.7, 0.92), 0.99),
            ],
            frame_interval=1.0,
        )

        result = check_speech_subtitle_alignment(transcript, subtitles)

        self.assertEqual(result.issues, ())

    def test_alignment_does_not_treat_corner_logo_as_subtitle(self):
        transcript = SpeechTranscript(
            language="en",
            segments=(SpeechSegment(0.0, 1.0, "hello world", "high"),),
            backend="fake",
            model="fake",
            device="cpu",
        )
        subtitles = merge_subtitle_observations(
            [
                SubtitleObservation(
                    0.0,
                    "TikTok",
                    (0.82, 0.05, 0.98, 0.12),
                    0.99,
                ),
            ],
            frame_interval=1.0,
        )

        result = check_speech_subtitle_alignment(transcript, subtitles)

        self.assertEqual(result.issues, ())

    def test_alignment_reports_same_text_at_wrong_time(self):
        transcript = SpeechTranscript(
            language="zh",
            segments=(SpeechSegment(0.0, 1.0, "你好", "high"),),
            backend="fake",
            model="fake",
            device="cpu",
        )
        subtitles = merge_subtitle_observations(
            [
                SubtitleObservation(2.0, "你好", (0.3, 0.8, 0.7, 0.92), 0.99),
            ],
            frame_interval=1.0,
        )

        result = check_speech_subtitle_alignment(transcript, subtitles)

        self.assertEqual(result.issues[0].issue_type, "timing_mismatch")

    def test_alignment_reports_language_mismatch(self):
        transcript = SpeechTranscript(
            language="zh",
            segments=(SpeechSegment(0.0, 1.0, "欢迎回来", "high"),),
            backend="fake",
            model="fake",
            device="cpu",
        )
        subtitles = merge_subtitle_observations(
            [
                SubtitleObservation(
                    0.0,
                    "welcome back",
                    (0.2, 0.8, 0.8, 0.92),
                    0.99,
                ),
            ],
            frame_interval=1.0,
        )

        result = check_speech_subtitle_alignment(transcript, subtitles)

        self.assertEqual(result.issues[0].issue_type, "language_mismatch")

    def test_alignment_localizes_chinese_subtitle_diff_despite_latin_logo(self):
        transcript = SpeechTranscript(
            language="zh",
            segments=(
                SpeechSegment(
                    6.92,
                    15.02,
                    "Yeah. 我这手机林师傅以前5分钟搞定，"
                    "你都修仨小时了，急什么，专业维修，讲究精细。",
                    "medium",
                    speaker=0,
                ),
            ),
            backend="fake",
            model="fake",
            device="cpu",
        )
        observations = []
        for timestamp in (10.5, 11.0, 11.5):
            observations.extend(
                [
                    SubtitleObservation(
                        timestamp,
                        "你都修三小时了",
                        (0.18, 0.67, 0.81, 0.72),
                        0.999,
                    ),
                    SubtitleObservation(
                        timestamp,
                        "BALENCIEN",
                        (0.19, 0.62, 0.80, 0.70),
                        0.95,
                    ),
                ]
            )
        subtitles = merge_subtitle_observations(
            observations,
            frame_interval=0.5,
        )

        result = check_speech_subtitle_alignment(transcript, subtitles)

        localized = [
            issue
            for issue in result.issues
            if issue.method == "localized_asr_ocr"
        ]
        self.assertEqual(len(localized), 1)
        self.assertEqual(localized[0].issue_type, "wrong_text")
        self.assertEqual(localized[0].speech_text, "你都修仨小时了")
        self.assertEqual(localized[0].subtitle_text, "你都修三小时了")
        self.assertIn("仨→三", localized[0].difference)
        self.assertEqual((localized[0].start_sec, localized[0].end_sec), (10.5, 12.0))
        self.assertFalse(
            any(issue.issue_type == "language_mismatch" for issue in result.issues)
        )

        deterministic = deterministic_alignment_issues(result)
        self.assertEqual(len(deterministic), 1)
        self.assertIn("ASR 实际语音为“你都修仨小时了”", deterministic[0]["问题说明"])
        self.assertIn("OCR 实际字幕为“你都修三小时了”", deterministic[0]["问题说明"])

    def test_alignment_handles_two_asr_segments_against_one_subtitle(self):
        transcript = SpeechTranscript(
            language="zh",
            segments=(
                SpeechSegment(0.0, 1.0, "欢迎来到", "high"),
                SpeechSegment(1.0, 2.0, "杭州", "high"),
            ),
            backend="fake",
            model="fake",
            device="cpu",
        )
        subtitles = merge_subtitle_observations(
            [
                SubtitleObservation(
                    0.0,
                    "欢迎来到杭州",
                    (0.2, 0.8, 0.8, 0.92),
                    0.99,
                ),
                SubtitleObservation(
                    1.0,
                    "欢迎来到杭州",
                    (0.2, 0.8, 0.8, 0.92),
                    0.99,
                ),
            ],
            frame_interval=1.0,
        )

        result = check_speech_subtitle_alignment(transcript, subtitles)

        self.assertEqual(result.issues, ())

    def test_alignment_handles_one_asr_segment_against_two_subtitles(self):
        transcript = SpeechTranscript(
            language="zh",
            segments=(SpeechSegment(0.0, 2.0, "欢迎来到杭州", "high"),),
            backend="fake",
            model="fake",
            device="cpu",
        )
        subtitles = merge_subtitle_observations(
            [
                SubtitleObservation(
                    0.0,
                    "欢迎来到",
                    (0.2, 0.8, 0.8, 0.92),
                    0.99,
                ),
                SubtitleObservation(
                    1.0,
                    "杭州",
                    (0.2, 0.8, 0.8, 0.92),
                    0.99,
                ),
            ],
            frame_interval=1.0,
        )

        result = check_speech_subtitle_alignment(transcript, subtitles)

        self.assertEqual(result.issues, ())

    def test_alignment_detects_amount_order_mismatch_in_coarse_asr_segment(self):
        transcript = SpeechTranscript(
            language="zh",
            segments=(
                SpeechSegment(
                    1.89,
                    10.11,
                    "装修合同上的这四字就是个无底洞，水电预收3000，"
                    "结算时变8000，防水又加3000",
                    "medium",
                ),
                SpeechSegment(
                    10.29,
                    12.33,
                    "美缝还要单独收2000",
                    "medium",
                ),
            ),
            backend="fake",
            model="fake",
            device="cpu",
        )
        observations = []
        for timestamp, text, bbox in (
            (5.5, "+8000", (0.05, 0.40, 0.89, 0.59)),
            (6.0, "+8000", (0.05, 0.40, 0.89, 0.59)),
            (7.5, "+3000", (0.10, 0.11, 0.85, 0.28)),
            (8.0, "+3000", (0.10, 0.11, 0.85, 0.28)),
            (9.0, "+2000", (0.10, 0.32, 0.85, 0.48)),
            (9.5, "+2000", (0.10, 0.32, 0.85, 0.48)),
            (10.5, "+2000", (0.10, 0.33, 0.85, 0.47)),
            (11.0, "+2000", (0.10, 0.33, 0.85, 0.47)),
        ):
            observations.append(
                SubtitleObservation(timestamp, text, bbox, 0.999)
            )
        subtitles = merge_subtitle_observations(
            observations,
            frame_interval=0.5,
        )

        result = check_speech_subtitle_alignment(transcript, subtitles)

        numeric = [
            issue
            for issue in result.issues
            if issue.method == "numeric_timeline_alignment"
        ]
        self.assertEqual(len(numeric), 1)
        self.assertEqual(numeric[0].issue_type, "numeric_timeline_mismatch")
        self.assertIn("OCR 显示“8000”", numeric[0].difference)
        self.assertIn("ASR 金额候选为“3000”", numeric[0].difference)
        self.assertEqual((numeric[0].start_sec, numeric[0].end_sec), (5.5, 10.0))

        deterministic = deterministic_alignment_issues(result)
        self.assertTrue(
            any("金额时序不一致" in issue["问题说明"] for issue in deterministic)
        )

    def test_alignment_accepts_matching_amount_timeline(self):
        transcript = SpeechTranscript(
            language="zh",
            segments=(
                SpeechSegment(
                    1.0,
                    7.0,
                    "水电预收3000，结算时变8000，防水又加3000",
                    "high",
                ),
            ),
            backend="fake",
            model="fake",
            device="cpu",
        )
        observations = [
            SubtitleObservation(3.0, "+3000", (0.1, 0.4, 0.9, 0.6), 0.999),
            SubtitleObservation(4.5, "+8000", (0.1, 0.3, 0.9, 0.5), 0.999),
            SubtitleObservation(6.0, "+3000", (0.1, 0.2, 0.9, 0.4), 0.999),
        ]
        subtitles = merge_subtitle_observations(
            observations,
            frame_interval=0.5,
        )

        result = check_speech_subtitle_alignment(transcript, subtitles)

        self.assertFalse(
            any(
                issue.method == "numeric_timeline_alignment"
                for issue in result.issues
            )
        )

    def test_alignment_ignores_small_document_numbers(self):
        transcript = SpeechTranscript(
            language="zh",
            segments=(
                SpeechSegment(
                    0.0,
                    6.0,
                    "项目依次是3000、8000和2000",
                    "high",
                ),
            ),
            backend="fake",
            model="fake",
            device="cpu",
        )
        observations = [
            SubtitleObservation(1.0, "+8000", (0.02, 0.2, 0.12, 0.25), 0.999),
            SubtitleObservation(3.0, "+3000", (0.20, 0.3, 0.30, 0.35), 0.999),
            SubtitleObservation(5.0, "+2000", (0.40, 0.4, 0.50, 0.45), 0.999),
        ]
        subtitles = merge_subtitle_observations(
            observations,
            frame_interval=0.5,
        )

        result = check_speech_subtitle_alignment(transcript, subtitles)

        self.assertFalse(
            any(
                issue.method == "numeric_timeline_alignment"
                for issue in result.issues
            )
        )

    def test_auralis_calls_audio_tools_for_every_audio_sample(self):
        calls = []

        def probe_video(_path):
            calls.append("probe")
            return {"has_audio": True}

        def extract_audio(_video_path, output_path):
            calls.append("extract_audio")
            output_path.write_bytes(b"RIFF-local-test")
            return output_path

        def transcribe(_audio_path):
            calls.append("asr")
            return SpeechTranscript(
                language="zh",
                segments=(SpeechSegment(0.0, 1.0, "你好", "high"),),
                backend="fake",
                model="fake",
                device="cpu",
            )

        def extract_subtitles(_video_path):
            calls.append("subtitles")
            return merge_subtitle_observations([], frame_interval=0.5)

        def align(transcript, subtitles):
            calls.append("alignment")
            return check_speech_subtitle_alignment(transcript, subtitles)

        def judge(_agent_input, evidence):
            calls.append("judge")
            self.assertEqual(evidence.transcript.segments[0].text, "你好")
            return []

        agent = AuralisAgent(
            probe_video=probe_video,
            extract_audio=extract_audio,
            transcribe_speech=transcribe,
            extract_subtitles=extract_subtitles,
            align_speech_subtitles=align,
            judge=judge,
        )

        with TemporaryDirectory() as temp_dir:
            result = agent.analyze(
                AuralisInput(
                    video_path=Path(temp_dir) / "sample.mp4",
                    user_prompt="检查音频",
                )
            )

        self.assertEqual(result.status, "ok")
        self.assertEqual(
            calls,
            ["probe", "extract_audio", "asr", "subtitles", "alignment", "judge"],
        )

    def test_auralis_vetoes_judge_issue_based_only_on_singleton_ocr(self):
        raw_subtitles = merge_subtitle_observations(
            [
                SubtitleObservation(
                    9.0,
                    "二",
                    (0.4, 0.30, 0.59, 0.44),
                    0.795,
                ),
                SubtitleObservation(
                    13.5,
                    "G",
                    (0.41, 0.24, 0.65, 0.38),
                    0.975,
                ),
            ],
            frame_interval=0.5,
            backend="fake-ocr",
        )
        transcript = SpeechTranscript(
            language="en",
            segments=(
                SpeechSegment(7.44, 8.34, "Where are you taking me", "medium"),
                SpeechSegment(10.59, 12.63, "A reunion only I remember", "medium"),
            ),
            backend="fake-asr",
            model="fake-model",
            device="cpu",
        )

        def extract_audio(_video_path, output_path):
            output_path.write_bytes(b"RIFF-local-test")
            return output_path

        def align(local_transcript, subtitles):
            self.assertEqual(subtitles.segments, ())
            return check_speech_subtitle_alignment(local_transcript, subtitles)

        def judge(_agent_input, evidence):
            self.assertEqual(evidence.subtitles.segments, ())
            return [
                {
                    "可定位性": "否",
                    "置信度": "高",
                    "问题说明": "实际在该时段出现字幕单字二。",
                    "问题类型": "文字质量问题",
                    "时间区间": "9.00s - 9.50s",
                    "关键帧秒": "",
                    "BBox": "",
                },
                {
                    "可定位性": "否",
                    "置信度": "高",
                    "问题说明": "实际在该时段出现字幕字符G。",
                    "问题类型": "文字质量问题",
                    "时间区间": "13.50s - 14.00s",
                    "关键帧秒": "",
                    "BBox": "",
                },
            ]

        agent = AuralisAgent(
            probe_video=lambda _path: {"has_audio": True, "duration_sec": 15.0},
            extract_audio=extract_audio,
            transcribe_speech=lambda _path: transcript,
            extract_subtitles=lambda _path: raw_subtitles,
            align_speech_subtitles=align,
            judge=judge,
        )

        with TemporaryDirectory() as temp_dir:
            result = agent.analyze(
                AuralisInput(
                    video_path=Path(temp_dir) / "sample.mp4",
                    user_prompt="不要生成字幕",
                )
            )

        self.assertEqual(result.issues, ())
        self.assertIsNotNone(result.evidence)
        self.assertEqual(len(result.evidence.subtitles.segments), 2)
        self.assertEqual(len(result.diagnostics["ocr_unverified_singletons"]), 2)
        self.assertEqual(len(result.diagnostics["ocr_vetoed_judge_issues"]), 2)

    def test_auralis_no_audio_is_a_successful_unconditional_call(self):
        calls = []
        agent = AuralisAgent(
            probe_video=lambda _path: {"has_audio": False},
            extract_audio=lambda *_args: calls.append("extract_audio"),
            transcribe_speech=lambda *_args: calls.append("asr"),
            extract_subtitles=lambda *_args: calls.append("subtitles"),
            align_speech_subtitles=lambda *_args: calls.append("alignment"),
            judge=lambda *_args: calls.append("judge"),
        )

        result = agent.analyze(
            AuralisInput(video_path=Path("silent.mp4"), user_prompt="")
        )

        self.assertEqual(result.status, "no_audio")
        self.assertEqual(result.issues, ())
        self.assertEqual(calls, [])

    def test_auralis_requires_explicit_judge_unless_local_only(self):
        with self.assertRaisesRegex(ValueError, "judge"):
            AuralisAgent()

        agent = AuralisAgent(local_only=True)
        self.assertIsInstance(agent, AuralisAgent)

    def test_combined_runner_calls_auralis_even_when_visual_agent_fails(self):
        calls = []

        class FakeAuralis:
            def analyze(self, _agent_input):
                calls.append("auralis")
                from agents.auralis.schemas import AuralisResult

                return AuralisResult(status="ok")

        with (
            mock.patch.object(
                auralis_runner.gpt_a,
                "run_agent",
                side_effect=RuntimeError("visual failed"),
            ),
            mock.patch.object(
                auralis_runner.gpt_a,
                "ensure_video",
                return_value=Path("/tmp/fake.mp4"),
            ),
            mock.patch.object(
                auralis_runner,
                "run_avbench_row",
                side_effect=lambda *_args, **_kwargs: calls.append("avbench")
                or {"success": True, "status": "ok"},
            ),
            self.assertRaisesRegex(RuntimeError, "visual failed"),
        ):
            auralis_runner.run_combined_row(
                {"序号": "#1"},
                {
                    "序号": "#1",
                    "user_prompt": "",
                    "reference_image_urls": [],
                    "generated_video_url": "fake.mp4",
                },
                api_url="https://example.test",
                api_key="token",
                gpt_a_model="gpt",
                gemini_model="gemini",
                timeout=1,
                api_retries=1,
                max_gpt_a_agent_steps=1,
                auralis_agent=FakeAuralis(),
            )

        self.assertEqual(calls, ["auralis", "avbench"])

    def test_combined_runner_adds_supported_seed_lite_issue_to_required_union(self):
        visual_issue = {
            "可定位性": "是",
            "置信度": "高",
            "问题说明": "GPT-A动作问题",
            "问题类型": "动作异常",
            "时间区间": "1s - 2s",
            "关键帧秒": "1.5",
            "BBox": "<bbox>0.1,0.1,0.9,0.9</bbox>",
        }
        audio_issue = {
            **visual_issue,
            "问题说明": "Auralis台词问题",
            "问题类型": "音频质量问题",
            "关键帧秒": "",
            "BBox": "",
        }
        seed_issue = {
            **visual_issue,
            "问题说明": "Seed-Lite发现禁用logo",
            "问题类型": "文字质量问题",
        }
        captured = {}

        def fake_synthesis(**kwargs):
            captured.update(kwargs)
            return "[]"

        with (
            mock.patch.object(
                auralis_runner.gpt_a,
                "run_agent",
                return_value=json.dumps([visual_issue], ensure_ascii=False),
            ),
            mock.patch.object(
                auralis_runner,
                "run_seed_lite_specialist",
                return_value=json.dumps([seed_issue], ensure_ascii=False),
            ) as seed_agent,
            mock.patch.object(
                auralis_runner,
                "run_audio_row",
                return_value=json.dumps([audio_issue], ensure_ascii=False),
            ),
            mock.patch.object(
                auralis_runner,
                "run_avbench_row",
                return_value={"success": True, "status": "ok"},
            ),
            mock.patch.object(
                auralis_runner,
                "synthesize_predictions",
                side_effect=fake_synthesis,
            ),
        ):
            auralis_runner.run_combined_row(
                {"序号": "#1", "user_prompt": "无logo"},
                {"序号": "#1"},
                api_url="https://example.test",
                api_key="token",
                gpt_a_model="gpt",
                gemini_model="gemini",
                seed_lite_model="seed-lite",
                timeout=1,
                api_retries=1,
                max_gpt_a_agent_steps=1,
            )

        seed_agent.assert_called_once()
        self.assertEqual(json.loads(captured["seed_lite_prediction"]), [seed_issue])
        self.assertEqual(
            list(captured["deterministic_issues"]),
            [visual_issue, audio_issue, seed_issue],
        )

    def test_review_sample_media_accepts_png_and_jpeg_references(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sample = root / "sample_001"
            sample.mkdir()
            (sample / "video.mp4").write_bytes(b"video")
            (sample / "reference_01.png").write_bytes(b"png")
            (sample / "reference_02.jpeg").write_bytes(b"jpeg")
            header = ["generated_video_url", "reference_image_urls"]

            row = auralis_runner._row_with_review_sample_media(
                header,
                ["old-video", "[]"],
                1,
                root,
            )

        self.assertTrue(row[0].endswith("sample_001/video.mp4"))
        self.assertEqual(
            [Path(value).suffix for value in json.loads(row[1])],
            [".png", ".jpeg"],
        )

    def test_review_sample_media_accepts_unique_decorated_sample_directory(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sample = root / "sample_014_未提供需要的音频"
            sample.mkdir()
            (sample / "video.mp4").write_bytes(b"video")
            (sample / "reference_01.jpg").write_bytes(b"jpg")
            header = ["generated_video_url", "reference_image_urls"]

            row = auralis_runner._row_with_review_sample_media(
                header,
                ["old-video", "[]"],
                14,
                root,
            )

        self.assertTrue(row[0].endswith("sample_014_未提供需要的音频/video.mp4"))

    def test_review_sample_media_allows_missing_reference_images_as_empty_input(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            sample = root / "sample_008"
            sample.mkdir()
            (sample / "video.mp4").write_bytes(b"video")
            header = ["generated_video_url", "reference_image_urls"]

            row = auralis_runner._row_with_review_sample_media(
                header,
                ["old-video", "[\"unavailable.bin\"]"],
                8,
                root,
            )

        self.assertEqual(json.loads(row[1]), [])

    def test_legacy_combined_entry_also_calls_auralis_when_visual_fails(self):
        calls = []

        with (
            mock.patch.object(
                legacy_runner.gpt_a,
                "run_agent",
                side_effect=RuntimeError("visual failed"),
            ),
            mock.patch.object(
                legacy_runner,
                "run_audio_row",
                side_effect=lambda *_args, **_kwargs: calls.append("auralis")
                or "[]",
            ),
            mock.patch.object(
                legacy_runner,
                "run_avbench_row",
                side_effect=lambda *_args, **_kwargs: calls.append("avbench")
                or {"success": True, "status": "ok"},
            ),
            self.assertRaisesRegex(RuntimeError, "visual failed"),
        ):
            legacy_runner.run_combined_row(
                {"序号": "#1"},
                {"序号": "#1"},
                api_url="https://example.test",
                api_key="token",
                gpt_a_model="gpt",
                gemini_model="gemini",
                timeout=1,
                api_retries=1,
                max_gpt_a_agent_steps=1,
            )

        self.assertEqual(calls, ["auralis", "avbench"])

    def test_no_audio_row_clears_reused_gateway_stats(self):
        gateway = GeminiGateway(
            api_url="https://example.test",
            api_key="token",
        )
        gateway.last_attempts = 2
        gateway.last_usage = {"total_tokens": 9}
        gateway.last_request_bytes = 123

        class NoAudioAgent:
            def analyze(self, _agent_input):
                from agents.auralis.schemas import AuralisResult

                return AuralisResult(status="no_audio")

        stats = {}
        with mock.patch.object(
            auralis_runner.gpt_a,
            "ensure_video",
            return_value=Path("/tmp/silent.mp4"),
        ):
            result = auralis_runner.run_audio_row(
                {
                    "序号": "#1",
                    "user_prompt": "",
                    "reference_image_urls": [],
                    "generated_video_url": "silent.mp4",
                },
                api_url="https://example.test",
                api_key="token",
                model="gemini",
                timeout=1,
                api_retries=1,
                run_stats=stats,
                auralis_agent=NoAudioAgent(),
                gateway=gateway,
            )

        self.assertEqual(result, "[]")
        self.assertEqual(stats["status"], "no_audio")
        self.assertNotIn("api_calls", stats)
        self.assertEqual(gateway.last_attempts, 0)

    def test_run_audio_row_preserves_raw_asr_ocr_and_alignment_evidence(self):
        transcript = SpeechTranscript(
            language="zh",
            segments=(SpeechSegment(4.0, 5.0, "没事", "high"),),
            backend="fake-asr",
            model="fake-model",
            device="cuda/int8_float16",
        )
        subtitles = merge_subtitle_observations(
            [
                SubtitleObservation(
                    9.0,
                    "没事",
                    (0.3, 0.8, 0.7, 0.92),
                    0.99,
                )
            ],
            frame_interval=0.5,
            backend="fake-ocr",
        )
        evidence = AuralisEvidence(
            media_metadata={"duration_sec": 12.0, "has_audio": True},
            transcript=transcript,
            subtitles=subtitles,
            alignment=check_speech_subtitle_alignment(transcript, subtitles),
        )

        class FakeAuralis:
            def analyze(self, _agent_input):
                return AuralisResult(
                    status="ok",
                    issues=(),
                    evidence=evidence,
                )

        stats = {}
        with mock.patch.object(
            auralis_runner.gpt_a,
            "ensure_video",
            return_value=Path("/tmp/sample18.mp4"),
        ):
            prediction = auralis_runner.run_audio_row(
                {
                    "序号": "#89",
                    "user_prompt": "",
                    "reference_image_urls": [],
                    "generated_video_url": "sample18.mp4",
                },
                api_url="https://example.test",
                api_key="token",
                model="gemini",
                timeout=1,
                api_retries=1,
                run_stats=stats,
                auralis_agent=FakeAuralis(),
            )

        self.assertEqual(prediction, "[]")
        self.assertEqual(stats["auralis_issues"], [])
        self.assertEqual(
            stats["auralis_evidence"]["transcript"]["segments"][0]["text"],
            "没事",
        )
        self.assertEqual(
            stats["auralis_evidence"]["subtitles"]["segments"][0]["start_sec"],
            9.0,
        )
        self.assertIn("alignment", stats["auralis_evidence"])
        json.dumps(stats, ensure_ascii=False)

    def test_cuda_backend_is_fail_fast_by_default(self):
        class BrokenWhisperModel:
            def __init__(self, *_args, **_kwargs):
                raise RuntimeError("Unable to load libcudnn.so.9")

        backend = FasterWhisperBackend(
            model_name="large-v3",
            device="cuda",
            compute_type="int8_float16",
        )
        with (
            mock.patch.dict(
                sys.modules,
                {
                    "faster_whisper": SimpleNamespace(
                        WhisperModel=BrokenWhisperModel
                    )
                },
            ),
            self.assertRaisesRegex(RuntimeError, "libcudnn"),
        ):
            backend._load_model()

        self.assertEqual(backend.device, "cuda")

    def test_cuda_backend_can_fallback_for_missing_cudnn_when_opted_in(self):
        calls = []

        class ConditionalWhisperModel:
            def __init__(self, *_args, **kwargs):
                calls.append(kwargs["device"])
                if kwargs["device"] == "cuda":
                    raise RuntimeError(
                        "Unable to load any of "
                        "{libcudnn_ops.so.9.1.0, libcudnn_ops.so.9}"
                    )

        backend = FasterWhisperBackend(
            model_name="large-v3",
            device="cuda",
            compute_type="int8_float16",
            allow_cpu_fallback=True,
        )
        with mock.patch.dict(
            sys.modules,
            {
                "faster_whisper": SimpleNamespace(
                    WhisperModel=ConditionalWhisperModel
                )
            },
        ):
            backend._load_model()

        self.assertEqual(calls, ["cuda", "cpu"])
        self.assertEqual(backend.device, "cpu")

    def test_cuda_backend_does_not_fallback_for_corrupt_model(self):
        class BrokenWhisperModel:
            def __init__(self, *_args, **_kwargs):
                raise RuntimeError("CUDA model file is corrupted")

        backend = FasterWhisperBackend(
            model_name="large-v3",
            device="cuda",
            allow_cpu_fallback=True,
        )
        with (
            mock.patch.dict(
                sys.modules,
                {
                    "faster_whisper": SimpleNamespace(
                        WhisperModel=BrokenWhisperModel
                    )
                },
            ),
            self.assertRaisesRegex(RuntimeError, "corrupted"),
        ):
            backend._load_model()

        self.assertEqual(backend.device, "cuda")

    def test_cuda_backend_does_not_fallback_for_out_of_memory(self):
        class BrokenWhisperModel:
            def __init__(self, *_args, **_kwargs):
                raise RuntimeError("CUDA out of memory")

        backend = FasterWhisperBackend(
            model_name="large-v3",
            device="cuda",
            allow_cpu_fallback=True,
        )
        with (
            mock.patch.dict(
                sys.modules,
                {
                    "faster_whisper": SimpleNamespace(
                        WhisperModel=BrokenWhisperModel
                    )
                },
            ),
            self.assertRaisesRegex(RuntimeError, "out of memory"),
        ):
            backend._load_model()

        self.assertEqual(backend.device, "cuda")


if __name__ == "__main__":
    unittest.main()
