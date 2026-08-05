import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from agents.auralis.agent import (
    AuralisAgent,
    filter_acoustically_contradicted_binding_issues,
)
from agents.auralis.schemas import AuralisInput
from agents.auralis.speaker_plan import extract_prompt_speech_plan
from tools.speech_transcription.backends.sensevoice import SenseVoiceBackend
from tools.speech_transcription.schemas import SpeechSegment, SpeechTranscript
from tools.speech_transcription.speaker_turns import (
    normalize_speaker_turns,
    sentence_info_to_segments,
)
from tools.subtitle_extraction.tool import merge_subtitle_observations


class SpeakerPlanTest(unittest.TestCase):
    def test_extracts_closed_two_role_dialogue_plan_with_times(self):
        prompt = """
镜号：1 | 0-5s | 台词 & 音效：Dialogue：无
镜号：2 | 5-10s | 台词 & 音效：Dialogue：@王守明 (抱怨)："你都修仨小时了！"
镜号：3 | 10-14s | 台词 & 音效：Dialogue：@程星 (喊)："急什么！讲究精细！"
"""

        plan = extract_prompt_speech_plan(prompt)

        self.assertEqual(plan["scope"], "closed")
        self.assertEqual(plan["expected_speaking_roles"], ["王守明", "程星"])
        self.assertEqual(plan["expected_speaker_count"], 2)
        self.assertEqual(
            [
                (turn["expected_start_sec"], turn["expected_end_sec"])
                for turn in plan["turns"]
            ],
            [(5.0, 10.0), (10.0, 14.0)],
        )
        self.assertFalse(plan["allows_unassigned_speech"])

    def test_unlabelled_quotes_do_not_create_speaker_ground_truth(self):
        plan = extract_prompt_speech_plan('画面要求“无字幕”，人物可自由发挥。')

        self.assertEqual(plan["scope"], "none")
        self.assertEqual(plan["expected_speaker_count"], 0)
        self.assertEqual(plan["turns"], [])

    def test_partial_prompt_keeps_unassigned_speech_open(self):
        plan = extract_prompt_speech_plan('甲说：“回来。”其余对白可自由发挥。')

        self.assertEqual(plan["scope"], "partial")
        self.assertEqual(plan["expected_speaking_roles"], ["甲"])
        self.assertTrue(plan["allows_unassigned_speech"])


class SpeakerTurnTest(unittest.TestCase):
    def test_splits_one_punctuation_sentence_at_campp_turn_boundary(self):
        turns, label_map = normalize_speaker_turns(
            [[5.0, 10.0, 1], [10.0, 14.0, 2]]
        )
        sentence_info = [
            {
                "raw_text": "你 都 修 仨 小 时 了 急 什 么",
                "text": "你都修仨小时了，急什么！",
                "start": 5000,
                "end": 14000,
                "timestamp": [
                    [5000, 5600],
                    [5600, 6200],
                    [6200, 6800],
                    [6800, 7400],
                    [7400, 8000],
                    [8000, 8600],
                    [8600, 9200],
                    [10200, 11000],
                    [11000, 11800],
                    [11800, 12600],
                ],
                "spk": 1,
            }
        ]

        segments = sentence_info_to_segments(sentence_info, turns, label_map)

        self.assertEqual(len(segments), 2)
        self.assertEqual([item["speaker"] for item in segments], [0, 1])
        self.assertEqual(segments[0]["text"], "你都修仨小时了")
        self.assertEqual(segments[1]["text"], "急什么")

    def test_splits_funasr_list_text_with_character_timestamps(self):
        turns, label_map = normalize_speaker_turns(
            [[2.82, 3.84, 0], [6.65, 11.53, 1], [11.53, 15.05, 2]]
        )
        text = [
            "Yeah.",
            "我这手机林师傅以前5分钟搞定，你都修仨小时了，"
            "急什么专业维修，讲究精细。",
        ]
        units = ["Yeah", "."] + list(text[1])
        timestamps = []
        for index in range(len(units)):
            if index < 2:
                start = 3150 + index * 600
            else:
                start = 6920 + (index - 2) * 220
            timestamps.append([start, start + 60])
        sentence_info = [
            {
                "text": text,
                "start": timestamps[0][0],
                "end": timestamps[-1][1],
                "timestamp": timestamps,
                "spk": 1,
            }
        ]

        segments = sentence_info_to_segments(sentence_info, turns, label_map)

        self.assertEqual([item["speaker"] for item in segments], [0, 1, 2])
        self.assertEqual(segments[0]["text"], "Yeah.")
        self.assertTrue(segments[1]["text"].startswith("我这手机"))
        self.assertIn("急什么", segments[2]["text"])

    def test_sensevoice_metadata_keeps_prompt_plan_and_fine_turns(self):
        response = {
            "language": "zh",
            "segments": [
                {
                    "start_sec": 5.0,
                    "end_sec": 10.0,
                    "text": "你都修仨小时了",
                    "speaker": 0,
                },
                {
                    "start_sec": 10.0,
                    "end_sec": 14.0,
                    "text": "急什么讲究精细",
                    "speaker": 1,
                },
            ],
            "speaker_turns": [
                {"start_sec": 5.0, "end_sec": 10.0, "speaker": 0},
                {"start_sec": 10.0, "end_sec": 14.0, "speaker": 1},
            ],
            "speaker_clustering": {"granularity_conflict": True},
            "raw_sentence_info": [{"text": "原始句子"}],
        }
        backend = SenseVoiceBackend(device="cpu")
        prompt = (
            '镜号：2 | 5-10s | Dialogue：@王守明："你都修仨小时了"\n'
            '镜号：3 | 10-14s | Dialogue：@程星："急什么讲究精细"'
        )
        with TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "audio.wav"
            audio_path.write_bytes(b"RIFF")
            with mock.patch.object(backend, "_request", return_value=response):
                transcript = backend.transcribe(audio_path, user_prompt=prompt)

        metadata = transcript.metadata
        self.assertEqual(
            metadata["prompt_speech_plan"]["expected_speaking_roles"],
            ["王守明", "程星"],
        )
        self.assertEqual(
            metadata["speaker_binding_evidence"]["status"],
            "fine_grained_turns",
        )
        alignment = metadata["speaker_binding_evidence"]["prompt_turn_alignment"]
        self.assertEqual(
            [(item["role"], item["actual_speakers"]) for item in alignment],
            [("王守明", [0]), ("程星", [1])],
        )
        self.assertEqual(
            [item["anchor_method"] for item in alignment],
            ["dialogue_text_similarity", "dialogue_text_similarity"],
        )
        self.assertEqual(
            metadata["raw_sentence_info"],
            [{"text": "原始句子"}],
        )
        binding = metadata["speaker_binding_evidence"]
        self.assertEqual(
            binding["role_to_speakers"],
            {"王守明": [0], "程星": [1]},
        )
        self.assertEqual(
            binding["speaker_to_roles"],
            {"0": ["王守明"], "1": ["程星"]},
        )
        self.assertEqual(binding["split_role_candidates"], [])
        self.assertEqual(binding["shared_speaker_candidates"], [])

    def test_closed_prompt_marks_unassigned_speech_without_forcing_cluster_count(self):
        response = {
            "language": "zh",
            "segments": [
                {
                    "start_sec": 3.15,
                    "end_sec": 3.81,
                    "text": "Yeah",
                    "speaker": 0,
                },
                {
                    "start_sec": 6.92,
                    "end_sec": 11.48,
                    "text": "你都修仨小时了",
                    "speaker": 1,
                },
            ],
            "speaker_turns": [
                {"start_sec": 3.15, "end_sec": 3.81, "speaker": 0},
                {"start_sec": 6.92, "end_sec": 11.48, "speaker": 1},
            ],
            "speaker_clustering": {"embedding_cluster_count": 2},
        }
        backend = SenseVoiceBackend(device="cpu")
        prompt = '镜号：3 | 5-10s | Dialogue：@王守明："你都修仨小时了"'
        with TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "audio.wav"
            audio_path.write_bytes(b"RIFF")
            with mock.patch.object(backend, "_request", return_value=response):
                transcript = backend.transcribe(audio_path, user_prompt=prompt)

        binding = transcript.metadata["speaker_binding_evidence"]
        self.assertEqual(binding["expected_speaker_count"], 1)
        self.assertEqual(binding["actual_acoustic_speaker_count"], 2)
        self.assertTrue(binding["prompt_count_is_not_cluster_ground_truth"])
        self.assertEqual(
            binding["unassigned_segments"],
            [
                {
                    "start_sec": 3.15,
                    "end_sec": 3.81,
                    "speaker": 0,
                    "text": "Yeah",
                    "closed_script_candidate": True,
                }
            ],
        )

    def test_repeated_role_turns_are_unioned_for_binding_conflicts(self):
        response = {
            "language": "zh",
            "segments": [
                {"start_sec": 0.0, "end_sec": 2.0, "text": "第一句", "speaker": 0},
                {"start_sec": 2.0, "end_sec": 4.0, "text": "第二句", "speaker": 0},
                {"start_sec": 4.0, "end_sec": 6.0, "text": "第三句", "speaker": 2},
            ],
            "speaker_turns": [
                {"start_sec": 0.0, "end_sec": 4.0, "speaker": 0},
                {"start_sec": 4.0, "end_sec": 6.0, "speaker": 2},
            ],
            "speaker_clustering": {},
        }
        backend = SenseVoiceBackend(device="cpu")
        prompt = (
            '0-2s 甲说：“第一句”\n'
            '2-4s 乙说：“第二句”\n'
            '4-6s 甲说：“第三句”'
        )
        with TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "audio.wav"
            audio_path.write_bytes(b"RIFF")
            with mock.patch.object(backend, "_request", return_value=response):
                transcript = backend.transcribe(audio_path, user_prompt=prompt)

        binding = transcript.metadata["speaker_binding_evidence"]
        self.assertEqual(binding["role_to_speakers"], {"甲": [0, 2], "乙": [0]})
        self.assertEqual(
            binding["split_role_candidates"],
            [{"role": "甲", "speakers": [0, 2]}],
        )
        self.assertEqual(
            binding["shared_speaker_candidates"],
            [{"speaker": "0", "roles": ["甲", "乙"]}],
        )


class SpeakerBindingGateTest(unittest.TestCase):
    @staticmethod
    def _transcript() -> SpeechTranscript:
        return SpeechTranscript(
            language="zh",
            segments=(
                SpeechSegment(5.0, 10.0, "你都修仨小时了", "medium", speaker=0),
                SpeechSegment(10.0, 14.0, "急什么讲究精细", "medium", speaker=1),
            ),
            backend="fake",
            model="fake",
            device="cpu",
            metadata={
                "speaker_binding_evidence": {
                    "status": "fine_grained_turns",
                    "prompt_turn_alignment": [
                        {
                            "role": "王守明",
                            "actual_speakers": [0],
                            "status": "anchored",
                        },
                        {
                            "role": "程星",
                            "actual_speakers": [1],
                            "status": "anchored",
                        },
                    ],
                }
            },
        )

    def test_vetoes_same_voice_claim_contradicted_by_fine_turns(self):
        issue = {
            "问题类型": "音频质量问题",
            "问题说明": (
                "程星声线与王守明完全一致，CAM++声纹标签未发生变化，"
                "属于角色绑定错误。"
            ),
        }

        kept, vetoed = filter_acoustically_contradicted_binding_issues(
            self._transcript(),
            [issue],
        )

        self.assertEqual(kept, ())
        self.assertEqual(len(vetoed), 1)
        self.assertEqual(vetoed[0]["role_speakers"], {"王守明": 0, "程星": 1})

    def test_keeps_binding_claim_that_is_not_acoustically_contradicted(self):
        issue = {
            "问题类型": "音频质量问题",
            "问题说明": "程星台词由王守明的spk0发出，存在角色绑定错误。",
        }

        kept, vetoed = filter_acoustically_contradicted_binding_issues(
            self._transcript(),
            [issue],
        )

        self.assertEqual(kept, (issue,))
        self.assertEqual(vetoed, ())

    def test_agent_passes_prompt_to_prompt_aware_transcriber(self):
        seen = []
        transcript = self._transcript()

        def transcribe(_path, prompt):
            seen.append(prompt)
            return transcript

        agent = AuralisAgent(
            probe_video=lambda _path: {"has_audio": True},
            extract_audio=lambda _video, output: output.write_bytes(b"RIFF") or output,
            transcribe_speech_with_prompt=transcribe,
            extract_subtitles=lambda _path: merge_subtitle_observations(
                [], frame_interval=0.5
            ),
            judge=lambda *_args: (),
        )
        with TemporaryDirectory() as temp_dir:
            result = agent.analyze(
                AuralisInput(
                    video_path=Path(temp_dir) / "video.mp4",
                    user_prompt="王守明说台词",
                )
            )

        self.assertEqual(result.status, "ok")
        self.assertEqual(seen, ["王守明说台词"])


if __name__ == "__main__":
    unittest.main()
