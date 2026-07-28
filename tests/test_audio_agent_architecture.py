import unittest
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import mock

import call_ffmpeg_skill_gpt_d as legacy_runner
import numpy as np
from agents.auralis.agent import AuralisAgent
from agents.auralis.schemas import AuralisInput
from agents.auralis import runner as auralis_runner
from agents.auralis.gemini_backend import GeminiGateway
from tools.speech_subtitle_alignment.tool import check_speech_subtitle_alignment
from tools.speech_transcription.schemas import SpeechSegment, SpeechTranscript
from tools.speech_transcription.backends.faster_whisper import (
    FasterWhisperBackend,
)
from tools.speech_transcription.tool import transcribe_speech
from tools.subtitle_extraction.schemas import SubtitleObservation
from tools.subtitle_extraction.backends.rapidocr import RapidOCRBackend
from tools.subtitle_extraction.tool import merge_subtitle_observations


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

        self.assertEqual(calls, ["auralis"])

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

        self.assertEqual(calls, ["auralis"])

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
