import unittest
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from agents.auralis.agent import AuralisAgent
from agents.auralis.constrained_asr import (
    constrained_asr_issues,
    evaluate_prompt_constrained_asr,
    extract_prompt_reference_candidates,
    filter_contradicted_judge_issues,
)
from agents.auralis.schemas import AuralisInput
from tools.speech_subtitle_alignment.tool import check_speech_subtitle_alignment
from tools.speech_transcription.backends.sensevoice import SenseVoiceBackend
from tools.speech_transcription.schemas import SpeechSegment, SpeechTranscript
from tools.subtitle_extraction.tool import merge_subtitle_observations


def _transcript(*segments: SpeechSegment) -> SpeechTranscript:
    return SpeechTranscript(
        language="zh",
        segments=tuple(segments),
        backend="fake-sensevoice",
        model="fake-model",
        device="cpu",
    )


class PromptReferenceExtractionTest(unittest.TestCase):
    def test_extracts_sample13_reference_with_exact_source_span(self):
        prompt = (
            "【镜头1】画面描述随意。\n"
            "主体动作: 贺雨棠失笑地说：“景林，你还想让我在这里住多久呀？"
            "到处都是消毒水的味道，我都觉得自己快要发霉了。”\n"
            "视频中不要出现字幕。"
        )
        transcript = _transcript(
            SpeechSegment(
                4.35,
                5.55,
                "我都觉得自己快要发膜了",
                "medium",
                speaker=1,
            )
        )

        result = extract_prompt_reference_candidates(prompt, transcript)

        self.assertEqual(result["status"], "candidates_ready")
        self.assertEqual(result["candidate_count"], 1)
        candidate = result["candidates"][0]
        self.assertEqual(candidate["observed_text"], "我都觉得自己快要发膜了")
        self.assertEqual(candidate["expected_text"], "我都觉得自己快要发霉了")
        self.assertEqual(
            prompt[candidate["prompt_start"] : candidate["prompt_end"]],
            candidate["prompt_source_text"],
        )
        self.assertEqual(candidate["prompt_source_text"], "我都觉得自己快要发霉了")
        self.assertEqual(candidate["differences"][0]["observed"], "膜")
        self.assertEqual(candidate["differences"][0]["expected"], "霉")

    def test_extracts_unquoted_dialogue_without_role_or_field_syntax(self):
        prompt = (
            "vertical video 9:16\n"
            "hospital room / daylight\n"
            "景林 你还想让我在这里住多久呀 到处都是消毒水的味道 "
            "我都觉得自己快要发霉了\n"
            "end on a close-up"
        )
        transcript = _transcript(
            SpeechSegment(4.35, 5.55, "我都觉得自己快要发膜了", "medium")
        )

        result = extract_prompt_reference_candidates(prompt, transcript)

        self.assertEqual(result["status"], "candidates_ready")
        self.assertEqual(result["candidates"][0]["expected_text"], "我都觉得自己快要发霉了")
        self.assertEqual(result["candidates"][0]["prompt_source_text"], "我都觉得自己快要发霉了")

    def test_extracts_reference_inside_arbitrary_json_like_format(self):
        prompt = (
            '{"shots":[{"payload":{"anything":"我都觉得自己快要发霉了"}}],'
            '"captions":false}'
        )
        transcript = _transcript(
            SpeechSegment(4.35, 5.55, "我都觉得自己快要发膜了", "medium")
        )

        result = extract_prompt_reference_candidates(prompt, transcript)

        self.assertEqual(result["candidate_count"], 1)
        self.assertEqual(result["candidates"][0]["prompt_source_text"], "我都觉得自己快要发霉了")

    def test_five_character_dialogue_candidate_is_not_dropped(self):
        prompt = "完全自由的输入格式\n肯叫我妈了"
        transcript = _transcript(
            SpeechSegment(2.0, 2.8, "肯见我妈了", "medium", speaker=1)
        )

        result = extract_prompt_reference_candidates(prompt, transcript)

        self.assertEqual(result["candidate_count"], 1)
        self.assertEqual(result["candidates"][0]["observed_text"], "肯见我妈了")
        self.assertEqual(result["candidates"][0]["expected_text"], "肯叫我妈了")
        self.assertEqual(result["candidates"][0]["alignment_edit_distance"], 1)

    def test_prompt_without_reference_dialogue_does_not_invent_candidate(self):
        prompt = "医院病房日内近景。女人想回家，男人温柔安慰她。不要出现字幕。"
        transcript = _transcript(
            SpeechSegment(4.35, 5.55, "我都觉得自己快要发膜了", "medium")
        )

        result = extract_prompt_reference_candidates(prompt, transcript)

        self.assertEqual(result["status"], "no_reference_dialogue")
        self.assertEqual(result["anchor_count"], 0)
        self.assertEqual(result["candidate_count"], 0)

    def test_short_vad_fragments_use_bounded_same_speaker_context(self):
        prompt = "台词可以写成任何形式 -> 自己快要发霉了"
        transcript = _transcript(
            SpeechSegment(1.0, 1.3, "自己快要", "medium", speaker=0),
            SpeechSegment(1.35, 1.8, "发膜了", "medium", speaker=0),
        )

        result = extract_prompt_reference_candidates(prompt, transcript)

        self.assertEqual(result["candidate_count"], 1)
        self.assertEqual(result["candidates"][0]["observed_text"], "自己快要发膜了")
        self.assertEqual(result["candidates"][0]["expected_text"], "自己快要发霉了")
        self.assertEqual(result["candidates"][0]["segment_indices"], [0, 1])

    def test_short_fragments_from_different_speakers_are_not_combined(self):
        prompt = "自己快要发霉了"
        transcript = _transcript(
            SpeechSegment(1.0, 1.3, "自己快要", "medium", speaker=0),
            SpeechSegment(1.35, 1.8, "发膜了", "medium", speaker=1),
        )

        result = extract_prompt_reference_candidates(prompt, transcript)

        self.assertEqual(result["status"], "no_reference_dialogue")
        self.assertEqual(result["candidates"], [])

    def test_suppresses_narration_speech_cue_at_dialogue_boundary(self):
        prompt = "戏谑说道：他没事，很舒服。"
        transcript = _transcript(
            SpeechSegment(9.82, 9.88, "了", "medium", speaker=2),
            SpeechSegment(10.0, 10.66, "他没事", "medium", speaker=2),
            SpeechSegment(10.84, 11.68, "很舒服", "medium", speaker=2),
        )

        result = extract_prompt_reference_candidates(prompt, transcript)

        self.assertEqual(result["candidates"], [])
        self.assertEqual(len(result["suppressed_candidates"]), 1)
        self.assertEqual(
            result["suppressed_candidates"][0]["decision"],
            "prompt_boundary_artifact",
        )
        self.assertEqual(
            result["suppressed_candidates"][0]["prompt_source_text"],
            "道：他没事，很舒服",
        )


class ConstrainedScoringDecisionTest(unittest.TestCase):
    def setUp(self):
        self.transcript = _transcript(
            SpeechSegment(4.35, 5.55, "我都觉得自己快要发膜了", "medium")
        )
        self.prompt = "无固定结构\n我都觉得自己快要发霉了\nEOF"

    @staticmethod
    def _scorer_with_scores(observed: float, expected: float):
        def scorer(_audio_path, candidates):
            return {
                "backend": "fake-ctc",
                "scores": [
                    {
                        "candidate_id": candidates[0]["candidate_id"],
                        "pronunciation_relation": "different_pronunciation",
                        "observed": {
                            "text": candidates[0]["observed_text"],
                            "ctc_log_likelihood": observed,
                        },
                        "expected": {
                            "text": candidates[0]["expected_text"],
                            "ctc_log_likelihood": expected,
                        },
                    }
                ],
            }

        return scorer

    def test_observed_preferred_creates_deterministic_audio_issue(self):
        evidence = evaluate_prompt_constrained_asr(
            Path("sample.wav"),
            self.prompt,
            self.transcript,
            scorer=self._scorer_with_scores(-7.30, -8.42),
        )

        self.assertEqual(evidence["status"], "scored")
        self.assertEqual(evidence["candidate_scores"][0]["decision"], "observed_preferred")
        issues = constrained_asr_issues(evidence)
        self.assertEqual(len(issues), 1)
        self.assertIn("发霉", issues[0]["问题说明"])
        self.assertIn("发膜", issues[0]["问题说明"])
        self.assertEqual(issues[0]["时间区间"], "4.35s - 5.55s")

    def test_expected_preferred_suppresses_raw_asr_false_positive(self):
        evidence = evaluate_prompt_constrained_asr(
            Path("sample.wav"),
            self.prompt,
            self.transcript,
            scorer=self._scorer_with_scores(-9.2, -7.1),
        )

        self.assertEqual(evidence["candidate_scores"][0]["decision"], "expected_preferred")
        self.assertEqual(constrained_asr_issues(evidence), ())

    def test_ambiguous_score_is_not_promoted_to_error(self):
        evidence = evaluate_prompt_constrained_asr(
            Path("sample.wav"),
            self.prompt,
            self.transcript,
            scorer=self._scorer_with_scores(-7.3, -7.8),
        )

        self.assertEqual(evidence["candidate_scores"][0]["decision"], "ambiguous")
        self.assertEqual(constrained_asr_issues(evidence), ())

    def test_same_pronunciation_is_kept_as_orthographic_evidence_only(self):
        def scorer(_audio_path, candidates):
            return {
                "backend": "fake-ctc",
                "scores": [
                    {
                        "candidate_id": candidates[0]["candidate_id"],
                        "pronunciation_relation": "same_pronunciation",
                        "observed_pronunciation": ["tang2", "tang2"],
                        "expected_pronunciation": ["tang2", "tang2"],
                        "observed": {"ctc_log_likelihood": -3.0},
                        "expected": {"ctc_log_likelihood": -20.0},
                    }
                ],
            }

        evidence = evaluate_prompt_constrained_asr(
            Path("sample.wav"),
            "棠棠肯定都不认得妈妈了",
            _transcript(
                SpeechSegment(
                    1.0,
                    2.0,
                    "糖糖肯定都不认得妈妈了",
                    "medium",
                )
            ),
            scorer=scorer,
        )

        score = evidence["candidate_scores"][0]
        self.assertEqual(score["decision"], "orthographic_homophone")
        self.assertGreater(
            score["delta_log_likelihood_observed_minus_expected"],
            10.0,
        )
        self.assertEqual(constrained_asr_issues(evidence), ())

    def test_homophone_decision_vetoes_only_matching_gemini_pronunciation_issue(self):
        evidence = {
            "candidate_scores": [
                {
                    "candidate_id": "prompt-asr-001",
                    "decision": "orthographic_homophone",
                    "start_sec": 1.0,
                    "end_sec": 2.0,
                    "observed_text": "糖糖肯定都不认得妈妈了",
                    "expected_text": "棠棠肯定都不认得妈妈了",
                    "differences": [
                        {"observed": "糖糖", "expected": "棠棠"},
                    ],
                }
            ]
        }
        pronunciation_issue = {
            "问题类型": "音频质量问题",
            "问题说明": "预期棠棠，实际糖糖，台词发音错误。",
            "时间区间": "1.00s - 2.00s",
        }
        role_issue = {
            "问题类型": "音频质量问题",
            "问题说明": "角色甲的棠棠台词由错误说话人说出。",
            "时间区间": "1.00s - 2.00s",
        }

        kept, vetoed = filter_contradicted_judge_issues(
            evidence,
            [pronunciation_issue, role_issue],
        )

        self.assertEqual(kept, (role_issue,))
        self.assertEqual(len(vetoed), 1)
        self.assertEqual(vetoed[0]["decision"], "orthographic_homophone")

    def test_no_reference_does_not_call_acoustic_scorer(self):
        scorer = mock.Mock()

        evidence = evaluate_prompt_constrained_asr(
            Path("sample.wav"),
            "医院病房日内近景，不要字幕。",
            self.transcript,
            scorer=scorer,
        )

        self.assertEqual(evidence["status"], "no_reference_dialogue")
        scorer.assert_not_called()


class ConstrainedScoringIntegrationTest(unittest.TestCase):
    def test_legacy_whisper_model_variable_does_not_override_sensevoice(self):
        with mock.patch.dict(
            os.environ,
            {"AURALIS_ASR_MODEL": "/models/faster-whisper-large-v3"},
            clear=True,
        ):
            backend = SenseVoiceBackend(device="cpu", use_campp=False)

        self.assertEqual(backend.model_name, "iic/SenseVoiceSmall")

    def test_sensevoice_backend_sends_only_source_derived_score_fields(self):
        backend = SenseVoiceBackend(device="cpu", use_campp=False)
        captured = {}

        def fake_request(request):
            captured.update(request)
            return {"ok": True, "backend": "fake", "scores": []}

        with TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "audio.wav"
            audio_path.write_bytes(b"RIFF")
            with mock.patch.object(backend, "_worker_request", side_effect=fake_request):
                backend.score_candidates(
                    audio_path,
                    [
                        {
                            "candidate_id": "prompt-asr-001",
                            "start_sec": 4.35,
                            "end_sec": 5.55,
                            "observed_text": "发膜",
                            "expected_text": "发霉",
                            "prompt_source_text": "secret surrounding prompt text",
                        }
                    ],
                )

        self.assertEqual(captured["action"], "score_candidates")
        self.assertEqual(
            set(captured["candidates"][0]),
            {"candidate_id", "start_sec", "end_sec", "observed_text", "expected_text"},
        )

    def test_sensevoice_voiceprint_request_excludes_role_and_prompt_text(self):
        backend = SenseVoiceBackend(device="cpu", use_campp=True)
        captured = {}

        def fake_request(request):
            captured.update(request)
            return {"ok": True, "clips": [], "pairs": []}

        with TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "audio.wav"
            audio_path.write_bytes(b"RIFF")
            with mock.patch.object(backend, "_worker_request", side_effect=fake_request):
                backend.score_speaker_segments(
                    audio_path,
                    [
                        {
                            "clip_id": "prompt-turn-000-part-00",
                            "start_sec": 1.0,
                            "end_sec": 2.0,
                            "role": "不得发送给声学模型",
                            "dialogue_text": "秘密台词",
                        }
                    ],
                )

        self.assertEqual(captured["action"], "score_speaker_segments")
        self.assertEqual(
            set(captured["clips"][0]),
            {"clip_id", "start_sec", "end_sec"},
        )

    def test_auralis_scores_before_gemini_and_appends_local_issue(self):
        calls = []

        def probe_video(_path):
            calls.append("probe")
            return {"has_audio": True, "duration_sec": 6.0}

        def extract_audio(_video_path, output_path):
            calls.append("extract_audio")
            output_path.write_bytes(b"RIFF-local-test")
            return output_path

        transcript = _transcript(
            SpeechSegment(4.35, 5.55, "我都觉得自己快要发膜了", "medium")
        )

        def transcribe(_audio_path):
            calls.append("asr")
            return transcript

        def score(audio_path, prompt, scored_transcript):
            calls.append("constrained_score")
            self.assertTrue(audio_path.is_file())
            self.assertIs(scored_transcript, transcript)
            return evaluate_prompt_constrained_asr(
                audio_path,
                prompt,
                scored_transcript,
                scorer=ConstrainedScoringDecisionTest._scorer_with_scores(-7.3, -8.42),
            )

        def extract_subtitles(_video_path):
            calls.append("subtitles")
            return merge_subtitle_observations([], frame_interval=0.5)

        def align(scored_transcript, subtitles):
            calls.append("alignment")
            return check_speech_subtitle_alignment(scored_transcript, subtitles)

        def judge(_agent_input, evidence):
            calls.append("judge")
            self.assertEqual(evidence.constrained_asr["status"], "scored")
            return []

        agent = AuralisAgent(
            probe_video=probe_video,
            extract_audio=extract_audio,
            transcribe_speech=transcribe,
            score_prompt_candidates=score,
            extract_subtitles=extract_subtitles,
            align_speech_subtitles=align,
            judge=judge,
        )
        with TemporaryDirectory() as temp_dir:
            result = agent.analyze(
                AuralisInput(
                    video_path=Path(temp_dir) / "sample.mp4",
                    user_prompt="任意格式\n我都觉得自己快要发霉了",
                )
            )

        self.assertEqual(
            calls,
            [
                "probe",
                "extract_audio",
                "asr",
                "constrained_score",
                "subtitles",
                "alignment",
                "judge",
            ],
        )
        self.assertEqual(len(result.issues), 1)
        self.assertIn("受约束 SenseVoice CTC", result.issues[0]["问题说明"])


if __name__ == "__main__":
    unittest.main()
