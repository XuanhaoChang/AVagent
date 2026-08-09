import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from agents.auralis.agent import (
    AuralisAgent,
    filter_acoustically_contradicted_binding_issues,
    filter_single_asr_negative_claims,
    filter_unanchored_gender_voice_issues,
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

    def test_extracts_action_subjects_instead_of_speech_manner_labels(self):
        prompt = """
【林止处于反抗；伙计甲处于控制；薛琴处于焦急】
主体动作:伙计甲一把拽住林止的胳膊，厉声道：“不是奸细能穿成这样？”
主体动作:伙计乙抬起粗绳，林止拼命挣扎，急喊：“别碰我！放开！”
主体动作:众人停住，画外传来薛琴急促高喊：“住手！”
视频角色对照表: 林止=【图2】；伙计甲=【@伙计甲/粗布短打】；
伙计乙=【@伙计乙/粗布短打】；薛琴=【图3】
"""

        plan = extract_prompt_speech_plan(prompt)

        self.assertEqual(
            plan["expected_speaking_roles"],
            ["伙计甲", "林止", "薛琴"],
        )
        self.assertEqual(
            plan["role_reference_images"],
            {"林止": [2], "薛琴": [3]},
        )
        self.assertNotIn("reference_image_indices", plan["turns"][0])
        self.assertEqual(plan["turns"][1]["reference_image_indices"], [2])
        self.assertEqual(plan["turns"][2]["reference_image_indices"], [3])

    def test_role_reference_extraction_ignores_scene_and_prop_tables(self):
        prompt = """
视频场景对照表: 河岸=【图1】
视频角色对照表: 林止=【图2】；薛琴=【图3】
视频道具对照表: 粗绳=【图4】；扁担=【图5】
林止说：“放开！”
"""

        plan = extract_prompt_speech_plan(prompt)

        self.assertEqual(plan["role_reference_images"], {"林止": [2], "薛琴": [3]})

    def test_extracts_at_image_role_aliases_without_action_phrase_roles(self):
        prompt = (
            "@图片2 为女主,@图片1 为男主。"
            "女主平静地低头说到：“是你唤醒了吾？”"
            "男主非常虚弱地说：“救我。”"
        )

        plan = extract_prompt_speech_plan(prompt)

        self.assertEqual(plan["expected_speaking_roles"], ["女主", "男主"])
        self.assertEqual(plan["role_reference_images"], {"女主": [2], "男主": [1]})
        self.assertEqual(
            [turn["role"] for turn in plan["turns"]],
            ["女主", "男主"],
        )

    def test_declared_roles_replace_speech_manner_phrases(self):
        prompt = """
主体动作:伙计甲上前半步，厉声喝道：“什么人！”
主体动作:伙计甲指着林止，大声喝道：“抓起来！”
主体动作:沈淮川看向贺雨棠，颤着声问：“真的是他？”
视频角色对照表: 伙计甲=【@伙计甲】；林止=【图2】；沈淮川=【图3】；贺雨棠=【图4】
"""

        plan = extract_prompt_speech_plan(prompt)

        self.assertEqual(
            [turn["role"] for turn in plan["turns"]],
            ["伙计甲", "伙计甲", "沈淮川"],
        )
        self.assertNotIn("厉声喝道", plan["expected_speaking_roles"])
        self.assertNotIn("颤着声", plan["expected_speaking_roles"])

    def test_inline_image_aliases_canonicalize_action_descriptions(self):
        prompt = (
            "女主图片2用指尖揉搓鼻翼：“好舒服。”"
            "毛豆图片1傲娇道：“那当然。”"
            "图片2惊喜道：“真的好了。”"
            "拿着产品图片3试探道：“这支归我？”"
        )

        plan = extract_prompt_speech_plan(prompt)

        self.assertEqual(
            [turn["role"] for turn in plan["turns"]],
            ["女主", "毛豆", "女主"],
        )
        self.assertEqual(
            plan["role_reference_images"],
            {"女主": [2], "毛豆": [1]},
        )

    def test_role_table_prevents_sound_effect_from_becoming_a_role(self):
        prompt = (
            '主体动作: “砰”的一声后，萧彻进入。\n'
            '萧彻厉声大喝：“大胆！”\n'
            "视频角色对照表: 萧彻=【图3】；白凌薇=【图4】"
        )

        plan = extract_prompt_speech_plan(prompt)

        self.assertEqual(
            [(turn["role"], turn["dialogue_text"]) for turn in plan["turns"]],
            [("萧彻", "大胆！")],
        )

    def test_extracts_explicit_unquoted_dialogue(self):
        prompt = "视频角色对照表: 李莲=【图3】\n李莲高兴喊道：建军，你快看！"

        plan = extract_prompt_speech_plan(prompt)

        self.assertEqual(plan["expected_speaking_roles"], ["李莲"])
        self.assertEqual(plan["turns"][0]["dialogue_text"], "建军，你快看！")

    def test_unquoted_action_phrase_without_named_role_stays_unassigned(self):
        prompt = (
            "古代寺庙内，突然激动跑进来大喊：前辈！隔壁薇薇安说找不到魔法球了。"
            "见状直接问：零前辈，你怎么了？"
        )

        plan = extract_prompt_speech_plan(prompt)

        self.assertEqual(plan["scope"], "none")
        self.assertEqual(plan["turns"], [])


class SpeakerTurnTest(unittest.TestCase):
    def test_single_timestamp_group_uses_overlapping_campp_speaker(self):
        turns, label_map = normalize_speaker_turns([[0.39, 6.11, 7]])
        segments = sentence_info_to_segments(
            [
                {
                    "raw_text": ["你", "好"],
                    "text": ["你", "好"],
                    "start": 600,
                    "end": 1060,
                    "timestamp": [[600, 820], [840, 1060]],
                    "spk": None,
                }
            ],
            turns,
            label_map,
        )

        self.assertEqual(
            segments,
            [
                {
                    "start_sec": 0.6,
                    "end_sec": 1.06,
                    "text": "你好",
                    "speaker": 0,
                }
            ],
        )

    def test_splits_same_speaker_at_clear_inter_utterance_pause(self):
        turns, label_map = normalize_speaker_turns([[1.4, 3.3, 0]])
        segments = sentence_info_to_segments(
            [
                {
                    "raw_text": ["谢", "谢", "妈", "建", "军"],
                    "text": ["谢", "谢", "妈", "建", "军"],
                    "start": 1640,
                    "end": 2700,
                    "timestamp": [
                        [1640, 1760],
                        [1780, 1900],
                        [1940, 2060],
                        [2360, 2500],
                        [2540, 2700],
                    ],
                    "spk": None,
                }
            ],
            turns,
            label_map,
        )

        self.assertEqual(
            [(item["text"], item["speaker"]) for item in segments],
            [("谢谢妈", 0), ("建军", 0)],
        )

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

    def test_splits_multiple_punctuated_roles_even_when_campp_label_is_shared(self):
        turns, label_map = normalize_speaker_turns([[1.17, 9.96, 0]])
        text = "求你。 是你唤醒了我。 救我。"
        units = list("求你。是你唤醒了我。救我。")
        timestamps = [
            [1170, 1230],
            [1290, 1350],
            [1890, 1950],
            [4750, 4810],
            [4870, 4930],
            [4990, 5050],
            [5110, 5170],
            [5230, 5290],
            [5350, 5410],
            [6670, 6730],
            [8640, 8700],
            [8760, 8820],
            [9900, 9960],
        ]
        self.assertEqual(len(units), len(timestamps))
        sentence_info = [
            {
                "raw_text": text,
                "text": text,
                "start": 1170,
                "end": 9960,
                "timestamp": timestamps,
                "spk": 0,
            }
        ]

        segments = sentence_info_to_segments(sentence_info, turns, label_map)

        self.assertEqual(
            [item["text"] for item in segments],
            ["求你。", "是你唤醒了我。", "救我。"],
        )
        self.assertEqual([item["speaker"] for item in segments], [0, 0, 0])
        self.assertEqual(
            [(item["start_sec"], item["end_sec"]) for item in segments],
            [(1.17, 1.95), (4.75, 6.73), (8.64, 9.96)],
        )

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

    def test_sensevoice_drops_lexical_placeholders_tagged_only_as_bgm(self):
        response = {
            "language": "auto",
            "raw_text": (
                "<|ja|><|EMO_UNKNOWN|><|BGM|><|withitn|>The. "
                "<|ja|><|EMO_UNKNOWN|><|BGM|><|withitn|>The."
            ),
            "segments": [
                {
                    "start_sec": 2.13,
                    "end_sec": 2.97,
                    "text": "The.",
                    "speaker": 0,
                }
            ],
            "speaker_turns": [
                {"start_sec": 2.0, "end_sec": 3.0, "speaker": 0}
            ],
            "speaker_clustering": {"embedding_cluster_count": 1},
        }
        backend = SenseVoiceBackend(device="cpu")
        with TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "audio.wav"
            audio_path.write_bytes(b"RIFF")
            with mock.patch.object(backend, "_request", return_value=response):
                transcript = backend.transcribe(audio_path)

        self.assertEqual(transcript.segments, ())
        self.assertEqual(
            transcript.metadata["speech_evidence_status"],
            "bgm_only",
        )
        self.assertEqual(transcript.metadata["audio_event_types"], ["bgm"])
        self.assertEqual(
            transcript.metadata["clustering"]["speaker_turn_count"],
            0,
        )

    def test_sensevoice_keeps_substantial_dialogue_when_bgm_is_also_tagged(self):
        response = {
            "language": "auto",
            "raw_text": (
                "<|zh|><|HAPPY|><|BGM|><|withitn|>小心别掉了。"
                "零前辈，你怎么了？他没事，很舒服。"
            ),
            "segments": [
                {
                    "start_sec": 4.29,
                    "end_sec": 8.38,
                    "text": "小心别掉了",
                    "speaker": 2,
                },
                {
                    "start_sec": 8.44,
                    "end_sec": 9.88,
                    "text": "零前辈你怎么了",
                    "speaker": 2,
                },
            ],
            "speaker_turns": [
                {"start_sec": 4.29, "end_sec": 8.38, "speaker": 2},
                {"start_sec": 8.44, "end_sec": 9.88, "speaker": 2},
            ],
            "speaker_clustering": {
                "embedding_count": 2,
                "embedding_cluster_count": 1,
                "speaker_turn_count": 2,
            },
        }
        backend = SenseVoiceBackend(device="cpu")
        with TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "audio.wav"
            audio_path.write_bytes(b"RIFF")
            with mock.patch.object(backend, "_request", return_value=response):
                transcript = backend.transcribe(audio_path)

        self.assertEqual(
            transcript.metadata["speech_evidence_status"],
            "speech_with_bgm",
        )
        self.assertEqual(len(transcript.segments), 2)
        self.assertNotIn(
            "suppressed_bgm_embedding_count",
            transcript.metadata["clustering"],
        )

    def test_text_only_dialogue_turns_can_anchor_speakers_without_timestamps(self):
        response = {
            "language": "zh",
            "segments": [
                {"start_sec": 1.0, "end_sec": 2.0, "text": "你好朋友", "speaker": 0},
                {"start_sec": 2.2, "end_sec": 3.2, "text": "再见朋友", "speaker": 1},
            ],
            "speaker_turns": [
                {"start_sec": 1.0, "end_sec": 2.0, "speaker": 0},
                {"start_sec": 2.2, "end_sec": 3.2, "speaker": 1},
            ],
            "speaker_clustering": {"embedding_cluster_count": 2},
        }
        backend = SenseVoiceBackend(device="cpu")
        prompt = '甲说：“你好朋友。”\n乙说：“再见朋友。”'
        with TemporaryDirectory() as temp_dir:
            audio_path = Path(temp_dir) / "audio.wav"
            audio_path.write_bytes(b"RIFF")
            with mock.patch.object(backend, "_request", return_value=response):
                transcript = backend.transcribe(audio_path, user_prompt=prompt)

        alignments = transcript.metadata["speaker_binding_evidence"][
            "prompt_turn_alignment"
        ]
        self.assertEqual(
            [(item["role"], item["actual_speakers"]) for item in alignments],
            [("甲", [0]), ("乙", [1])],
        )
        self.assertTrue(all(item["status"] == "anchored" for item in alignments))

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

    def test_vetoes_directional_binding_claim_without_structured_conflict(self):
        issue = {
            "问题类型": "音频质量问题",
            "问题说明": "程星台词由王守明的spk0发出，存在角色绑定错误。",
        }

        kept, vetoed = filter_acoustically_contradicted_binding_issues(
            self._transcript(),
            [issue],
        )

        self.assertEqual(kept, ())
        self.assertEqual(
            vetoed[0]["reason"],
            "speaker_binding_claim_has_no_structured_role_conflict",
        )

    def test_vetoes_sample18_binding_claim_without_prompt_role_scope(self):
        transcript = SpeechTranscript(
            language="zh",
            segments=(
                SpeechSegment(4.29, 8.38, "小心别掉了", "medium", speaker=2),
                SpeechSegment(8.44, 9.88, "零前辈你怎么了", "medium", speaker=2),
                SpeechSegment(10.0, 11.68, "他没事很舒服", "medium", speaker=2),
            ),
            backend="fake",
            model="fake",
            device="cpu",
            metadata={
                "speech_evidence_status": "speech_with_bgm",
                "prompt_speech_plan": {"scope": "none", "turns": []},
                "clustering": {"speaker_turn_count": 3},
                "speaker_binding_evidence": {
                    "status": "fine_grained_turns",
                    "prompt_scope": "none",
                    "prompt_turn_alignment": [],
                },
            },
        )
        issue = {
            "问题类型": "音频质量问题",
            "问题说明": (
                "图1与图2角色的台词均由同一个匿名声纹spk2发出，"
                "存在说话人绑定和共用声纹错误。"
            ),
        }

        kept, vetoed = filter_acoustically_contradicted_binding_issues(
            transcript,
            [issue],
        )

        self.assertEqual(kept, ())
        self.assertEqual(
            vetoed[0]["reason"],
            "speaker_binding_claim_has_no_prompt_role_scope",
        )

    def test_keeps_split_role_binding_supported_by_prompt_turns(self):
        transcript = SpeechTranscript(
            language="zh",
            segments=(
                SpeechSegment(1.64, 2.06, "谢谢妈", "medium", speaker=0),
                SpeechSegment(2.36, 4.64, "建军你快看", "medium", speaker=0),
                SpeechSegment(4.88, 5.60, "肯叫我妈了", "medium", speaker=1),
                SpeechSegment(12.56, 14.84, "行了你先歇着", "medium", speaker=1),
            ),
            backend="fake",
            model="fake",
            device="cpu",
            metadata={
                "speech_evidence_status": "speech_present",
                "prompt_speech_plan": {"scope": "closed"},
                "clustering": {"speaker_turn_count": 3},
                "speaker_binding_evidence": {
                    "status": "fine_grained_turns",
                    "prompt_scope": "closed",
                    "prompt_turn_alignment": [
                        {"role": "贺雨棠", "actual_speakers": [0], "status": "anchored"},
                        {"role": "李莲", "actual_speakers": [0, 1], "status": "anchored"},
                        {"role": "李莲", "actual_speakers": [1], "status": "anchored"},
                    ],
                },
            },
        )
        issue = {
            "问题类型": "音频质量问题",
            "问题说明": (
                "预期李莲台词由李莲声纹spk1发出，实际前半句由贺雨棠的spk0发出，"
                "存在角色绑定错误。"
            ),
        }

        kept, vetoed = filter_acoustically_contradicted_binding_issues(
            transcript,
            [issue],
        )

        self.assertEqual(kept, (issue,))
        self.assertEqual(vetoed, ())

    def test_keeps_shared_voice_claim_with_pairwise_campp_confirmation(self):
        transcript = SpeechTranscript(
            language="zh",
            segments=(
                SpeechSegment(1.0, 2.0, "第一句", "medium", speaker=0),
                SpeechSegment(3.0, 4.0, "第二句", "medium", speaker=0),
            ),
            backend="fake",
            model="fake",
            device="cpu",
            metadata={
                "speech_evidence_status": "speech_present",
                "prompt_speech_plan": {"scope": "closed"},
                "clustering": {
                    "speaker_turn_count": 2,
                    "similarity_threshold": 0.78,
                    "cluster_algorithm_status": "spectral_clustered",
                    "cluster_similarity": {
                        "0": {
                            "window_count": 2,
                            "within_pair_count": 1,
                            "within_similarity_min": 0.88,
                            "within_similarity_mean": 0.88,
                        }
                    },
                },
                "speaker_binding_evidence": {
                    "status": "fine_grained_turns",
                    "prompt_scope": "closed",
                    "prompt_turn_alignment": [
                        {"role": "甲", "actual_speakers": [0], "status": "anchored"},
                        {"role": "乙", "actual_speakers": [0], "status": "anchored"},
                    ],
                },
            },
        )
        issue = {
            "问题类型": "音频质量问题",
            "问题说明": "甲和乙的台词由同一声纹spk0发出，存在共用声纹错误。",
        }

        kept, vetoed = filter_acoustically_contradicted_binding_issues(
            transcript,
            [issue],
        )

        self.assertEqual(kept, (issue,))
        self.assertEqual(vetoed, ())

    def test_vetoes_shared_voice_claim_without_pairwise_campp_confirmation(self):
        transcript = SpeechTranscript(
            language="zh",
            segments=(
                SpeechSegment(1.0, 2.0, "是你唤醒了我", "medium", speaker=0),
                SpeechSegment(3.0, 4.0, "救我", "medium", speaker=0),
            ),
            backend="fake",
            model="fake",
            device="cpu",
            metadata={
                "speech_evidence_status": "speech_present",
                "prompt_speech_plan": {"scope": "partial"},
                "clustering": {"speaker_turn_count": 3},
                "speaker_binding_evidence": {
                    "status": "fine_grained_turns",
                    "prompt_scope": "partial",
                    "prompt_turn_alignment": [
                        {"role": "女主", "actual_speakers": [0], "status": "anchored"},
                        {"role": "男主", "actual_speakers": [0], "status": "anchored"},
                    ],
                },
            },
        )
        issue = {
            "问题类型": "音频质量问题",
            "问题说明": "女主与男主使用同一个单一声纹speaker 0，存在角色绑定错误。",
        }

        kept, vetoed = filter_acoustically_contradicted_binding_issues(
            transcript,
            [issue],
        )

        self.assertEqual(kept, ())
        self.assertEqual(
            vetoed[0]["reason"],
            "same_voice_claim_lacks_pairwise_campp_confirmation",
        )

    def test_vetoes_same_voice_claim_from_only_one_campp_turn(self):
        transcript = SpeechTranscript(
            language="zh",
            segments=(
                SpeechSegment(0.6, 1.7, "不是奸细", "medium", speaker=0),
                SpeechSegment(1.8, 3.1, "别碰我放开", "medium", speaker=0),
                SpeechSegment(5.5, 6.1, "住手", "medium", speaker=0),
            ),
            backend="fake",
            model="fake",
            device="cpu",
            metadata={
                "clustering": {"speaker_turn_count": 1},
                "speaker_binding_evidence": {
                    "status": "fine_grained_turns",
                    "prompt_turn_alignment": [
                        {
                            "role": role,
                            "actual_speakers": [0],
                            "status": "anchored",
                        }
                        for role in ("伙计甲", "林止", "薛琴")
                    ],
                },
            },
        )
        issue = {
            "问题类型": "音频质量问题",
            "问题说明": (
                "伙计甲、林止、薛琴的台词均由同一匿名声纹 speaker 0 发出，"
                "存在共用单一声纹问题。"
            ),
        }

        kept, vetoed = filter_acoustically_contradicted_binding_issues(
            transcript,
            [issue],
        )

        self.assertEqual(kept, ())
        self.assertEqual(
            vetoed[0]["reason"],
            "same_voice_claim_lacks_independent_anchored_campp_turns",
        )

    def test_resolver_does_not_expand_one_confirmed_pair_to_three_roles(self):
        transcript = SpeechTranscript(
            language="zh",
            segments=(
                SpeechSegment(1.0, 2.0, "第一句", "medium", speaker=0),
                SpeechSegment(3.0, 4.0, "第二句", "medium", speaker=0),
                SpeechSegment(5.0, 6.0, "第三句", "medium", speaker=0),
            ),
            backend="fake",
            model="fake",
            device="cpu",
            metadata={
                "speaker_binding_evidence": {
                    "status": "fine_grained_turns",
                    "prompt_scope": "partial",
                    "prompt_turn_alignment": [
                        {
                            "role": role,
                            "actual_speakers": [0],
                            "status": "anchored",
                        }
                        for role in ("甲", "乙", "丙")
                    ],
                },
                "speaker_binding_resolution": {
                    "version": 2,
                    "decision": "supported",
                    "directional_conflicts": [],
                    "shared_voice_conflicts": [{"roles": ["甲", "乙"]}],
                },
            },
        )
        issue = {
            "问题类型": "音频质量问题",
            "问题说明": "甲、乙、丙全部共用同一声纹spk0。",
        }

        kept, vetoed = filter_acoustically_contradicted_binding_issues(
            transcript,
            [issue],
        )

        self.assertEqual(kept, ())
        self.assertEqual(
            vetoed[0]["reason"],
            "same_voice_claim_lacks_direct_role_pair_confirmation",
        )

    def test_resolver_directional_claim_must_name_the_resolved_role_pair(self):
        transcript = SpeechTranscript(
            language="zh",
            segments=(
                SpeechSegment(2.36, 3.20, "建军你快看", "medium", speaker=0),
            ),
            backend="fake",
            model="fake",
            device="cpu",
            metadata={
                "speaker_binding_evidence": {
                    "status": "fine_grained_turns",
                    "prompt_scope": "partial",
                    "prompt_turn_alignment": [
                        {"role": role, "actual_speakers": [speaker], "status": "anchored"}
                        for role, speaker in (("李莲", 0), ("贺雨棠", 0), ("林建军", 2))
                    ],
                },
                "speaker_binding_resolution": {
                    "version": 2,
                    "decision": "supported",
                    "directional_conflicts": [
                        {"expected_role": "李莲", "actual_role": "贺雨棠"}
                    ],
                    "shared_voice_conflicts": [],
                },
            },
        )
        supported = {
            "问题类型": "音频质量问题",
            "问题说明": "李莲的台词由贺雨棠声纹发出，存在角色绑定错误。",
        }
        wrong_actual_role = {
            "问题类型": "音频质量问题",
            "问题说明": "李莲的台词由林建军声纹发出，存在角色绑定错误。",
        }

        kept, vetoed = filter_acoustically_contradicted_binding_issues(
            transcript,
            [supported, wrong_actual_role],
        )

        self.assertEqual(kept, (supported,))
        self.assertEqual(len(vetoed), 1)
        self.assertEqual(
            vetoed[0]["reason"],
            "directional_binding_claim_lacks_resolver_confirmation",
        )

    def test_preserves_gender_mismatch_when_mixed_same_voice_claim_is_vetoed(self):
        transcript = SpeechTranscript(
            language="zh",
            segments=(
                SpeechSegment(0.6, 1.7, "不是奸细", "medium", speaker=0),
                SpeechSegment(1.8, 3.06, "别碰我放开", "medium", speaker=0),
                SpeechSegment(5.52, 6.06, "住手", "medium", speaker=0),
            ),
            backend="fake",
            model="fake",
            device="cpu",
            metadata={
                "prompt_speech_plan": {
                    "role_reference_images": {"林止": [2], "薛琴": [3]},
                },
                "clustering": {"speaker_turn_count": 1},
                "speaker_binding_evidence": {
                    "status": "fine_grained_turns",
                    "prompt_turn_alignment": [
                        {
                            "role": "林止",
                            "dialogue_text": "别碰我！放开！",
                            "observed_text": "别碰我放开",
                            "actual_speakers": [0],
                            "status": "anchored",
                            "matched_segments": [
                                {
                                    "start_sec": 1.8,
                                    "end_sec": 3.06,
                                    "speaker": 0,
                                    "text": "别碰我放开",
                                }
                            ],
                        },
                        {
                            "role": "薛琴",
                            "dialogue_text": "住手！",
                            "observed_text": "住手",
                            "actual_speakers": [0],
                            "status": "anchored",
                            "matched_segments": [
                                {
                                    "start_sec": 5.52,
                                    "end_sec": 6.06,
                                    "speaker": 0,
                                    "text": "住手",
                                }
                            ],
                        },
                    ],
                },
            },
        )
        issue = {
            "可定位性": "否",
            "置信度": "高",
            "问题说明": (
                "预期林止（参考图2，年轻女性）、薛琴（参考图3，中年女性）应独立配音，"
                "实际所有角色由同一声纹spk0发出，导致林止和薛琴均明显为粗鲁成年男声，"
                "存在人声绑定与性别音色冲突。"
            ),
            "问题类型": "音频质量问题",
            "时间区间": "0.60s - 6.06s",
            "关键帧秒": "",
            "BBox": "",
        }

        kept, vetoed = filter_acoustically_contradicted_binding_issues(
            transcript,
            [issue],
        )

        self.assertEqual(len(kept), 1)
        self.assertIn("林止", kept[0]["问题说明"])
        self.assertIn("图2", kept[0]["问题说明"])
        self.assertIn("明显成年男声", kept[0]["问题说明"])
        self.assertNotIn("同一声纹", kept[0]["问题说明"])
        self.assertNotIn("薛琴", kept[0]["问题说明"])
        self.assertEqual(kept[0]["时间区间"], "1.80s - 3.06s")
        self.assertEqual(len(vetoed[0]["preserved_gender_issues"]), 1)

    def test_vetoes_gender_inference_based_only_on_single_campp_cluster(self):
        transcript = SpeechTranscript(
            language="zh",
            segments=(
                SpeechSegment(1.17, 1.99, "求你", "medium", speaker=0),
                SpeechSegment(4.54, 6.74, "是你唤醒了我", "medium", speaker=0),
                SpeechSegment(8.25, 9.96, "救我", "medium", speaker=0),
            ),
            backend="fake",
            model="fake",
            device="cpu",
            metadata={
                "clustering": {"speaker_turn_count": 3},
                "speaker_binding_evidence": {
                    "status": "fine_grained_turns",
                    "prompt_turn_alignment": [],
                },
            },
        )
        issue = {
            "可定位性": "否",
            "置信度": "高",
            "问题说明": (
                "预期女主台词‘是你唤醒了吾？’应由具有女性特征的音色发出，"
                "实际却与男主台词‘求你’‘救我’使用同一个单一声纹speaker 0发出，"
                "导致女主角色被错误地绑定为男声。"
            ),
            "问题类型": "音频质量问题",
            "时间区间": "4.54s - 6.74s",
            "关键帧秒": "",
            "BBox": "",
        }

        kept, vetoed = filter_acoustically_contradicted_binding_issues(
            transcript,
            [issue],
        )

        self.assertEqual(kept, ())
        self.assertEqual(len(vetoed), 1)
        self.assertEqual(
            vetoed[0]["reason"],
            "speaker_cluster_cannot_prove_acoustic_gender",
        )
        self.assertEqual(vetoed[0]["preserved_gender_issues"], [])

    def test_does_not_preserve_bound_as_male_without_direct_acoustic_claim(self):
        transcript = SpeechTranscript(
            language="zh",
            segments=(
                SpeechSegment(4.54, 6.74, "是你唤醒了我", "medium", speaker=0),
                SpeechSegment(8.25, 9.96, "救我", "medium", speaker=0),
            ),
            backend="fake",
            model="fake",
            device="cpu",
            metadata={
                "prompt_speech_plan": {
                    "role_reference_images": {"女主": [2], "男主": [1]},
                },
                "clustering": {"speaker_turn_count": 2},
                "speaker_binding_evidence": {
                    "status": "fine_grained_turns",
                    "prompt_turn_alignment": [
                        {
                            "role": "女主",
                            "dialogue_text": "是你唤醒了吾？",
                            "actual_speakers": [0],
                            "status": "anchored",
                            "matched_segments": [
                                {
                                    "start_sec": 4.54,
                                    "end_sec": 6.74,
                                    "speaker": 0,
                                    "text": "是你唤醒了我",
                                }
                            ],
                        },
                        {
                            "role": "男主",
                            "dialogue_text": "救我",
                            "actual_speakers": [0],
                            "status": "anchored",
                            "matched_segments": [
                                {
                                    "start_sec": 8.25,
                                    "end_sec": 9.96,
                                    "speaker": 0,
                                    "text": "救我",
                                }
                            ],
                        },
                    ],
                },
            },
        )
        issue = {
            "可定位性": "否",
            "置信度": "高",
            "问题说明": (
                "预期女主（参考图2，女性）使用女声，实际女主与男主使用单一声纹，"
                "因此女主被绑定为男声。"
            ),
            "问题类型": "音频质量问题",
            "时间区间": "4.54s - 6.74s",
            "关键帧秒": "",
            "BBox": "",
        }

        kept, vetoed = filter_acoustically_contradicted_binding_issues(
            transcript,
            [issue],
        )

        self.assertEqual(kept, ())
        self.assertEqual(
            vetoed[0]["reason"],
            "speaker_cluster_cannot_prove_acoustic_gender",
        )
        self.assertEqual(vetoed[0]["preserved_gender_issues"], [])

    def test_vetoes_standalone_gender_claim_without_role_labelled_clip(self):
        transcript = SpeechTranscript(
            language="zh",
            segments=(
                SpeechSegment(4.54, 6.74, "是你唤醒了我", "medium", speaker=0),
            ),
            backend="fake",
            model="fake",
            device="cpu",
            metadata={
                "prompt_speech_plan": {"role_reference_images": {}},
                "speaker_binding_evidence": {
                    "status": "fine_grained_turns",
                    "prompt_turn_alignment": [],
                },
            },
        )
        issue = {
            "可定位性": "否",
            "置信度": "高",
            "问题说明": (
                "预期女主（参考图02）使用女性声音，实际该段女主台词明显为男声，"
                "存在性别声线冲突。"
            ),
            "问题类型": "音频质量问题",
            "时间区间": "4.54s - 6.74s",
            "关键帧秒": "",
            "BBox": "",
        }

        kept, vetoed = filter_unanchored_gender_voice_issues(transcript, [issue])

        self.assertEqual(kept, ())
        self.assertEqual(
            vetoed[0]["reason"],
            "gender_claim_lacks_role_labelled_acoustic_check",
        )

    def test_keeps_gender_claim_with_reference_and_role_labelled_clip(self):
        transcript = SpeechTranscript(
            language="zh",
            segments=(
                SpeechSegment(1.8, 3.06, "别碰我放开", "medium", speaker=0),
            ),
            backend="fake",
            model="fake",
            device="cpu",
            metadata={
                "prompt_speech_plan": {
                    "role_reference_images": {"林止": [2]},
                },
                "speaker_binding_evidence": {
                    "status": "fine_grained_turns",
                    "prompt_turn_alignment": [
                        {
                            "role": "林止",
                            "dialogue_text": "别碰我！放开！",
                            "actual_speakers": [0],
                            "status": "anchored",
                            "matched_segments": [
                                {
                                    "start_sec": 1.8,
                                    "end_sec": 3.06,
                                    "speaker": 0,
                                    "text": "别碰我放开",
                                }
                            ],
                        }
                    ],
                },
            },
        )
        issue = {
            "可定位性": "否",
            "置信度": "高",
            "问题说明": "参考图展示林止为年轻女性，实际台词中听到明显成年男声。",
            "问题类型": "音频质量问题",
            "时间区间": "1.80s - 3.06s",
            "关键帧秒": "",
            "BBox": "",
        }

        kept, vetoed = filter_unanchored_gender_voice_issues(transcript, [issue])

        self.assertEqual(kept, (issue,))
        self.assertEqual(vetoed, ())

    def test_vetoes_short_unanchored_extra_word_and_dependent_ocr_claim(self):
        transcript = SpeechTranscript(
            language="auto",
            segments=(SpeechSegment(3.15, 3.81, "Yeah.", "medium", speaker=0),),
            backend="fake",
            model="fake",
            device="cpu",
            metadata={
                "speaker_binding_evidence": {
                    "unassigned_segments": [
                        {
                            "start_sec": 3.15,
                            "end_sec": 3.81,
                            "speaker": 0,
                            "text": "Yeah.",
                            "closed_script_candidate": True,
                        }
                    ]
                }
            },
        )
        issues = [
            {
                "问题类型": "音频质量问题",
                "问题说明": "无台词时段出现多余英文人声“Yeah.”。",
                "时间区间": "3.15s - 3.81s",
            },
            {
                "问题类型": "文字质量问题",
                "问题说明": "实际语音为“Yeah.”，字幕为“H”。",
                "时间区间": "3.15s - 3.81s",
            },
            {
                "问题类型": "文字质量问题",
                "问题说明": "屏幕 UI 将“4K/60/RS/W”显示为“4K60/RS/W”。",
                "时间区间": "11.00s - 11.50s",
            },
        ]

        kept, vetoed = filter_single_asr_negative_claims(transcript, issues)

        self.assertEqual(kept, (issues[2],))
        self.assertEqual(len(vetoed), 2)

    def test_vetoes_dialogue_absence_claim_based_on_one_asr_miss(self):
        transcript = SpeechTranscript(
            language="auto",
            segments=(SpeechSegment(3.89, 4.55, "おお", "medium", speaker=0),),
            backend="fake",
            model="fake",
            device="cpu",
        )
        issue = {
            "问题类型": "音频质量问题",
            "问题说明": "实际未检测到任何中文台词，发生严重台词缺失。",
            "时间区间": "4.00s - 9.00s",
        }

        kept, vetoed = filter_single_asr_negative_claims(transcript, [issue])

        self.assertEqual(kept, ())
        self.assertEqual(
            vetoed[0]["reason"],
            "single_asr_miss_cannot_prove_dialogue_absence",
        )

    def test_agent_generates_sample64_binding_issue_without_gemini_candidate(self):
        plan = {
            "scope": "partial",
            "turns": [
                {"role": "贺雨棠", "dialogue_text": "谢谢妈"},
                {
                    "role": "李莲",
                    "dialogue_text": "建军你快看雪芳终于想通了肯叫我妈了",
                },
                {"role": "贺雨棠", "dialogue_text": "现在只能指望你们了"},
                {"role": "李莲", "dialogue_text": "行了你先歇着"},
            ],
        }

        def alignment(role, dialogue, *segments):
            return {
                "role": role,
                "dialogue_text": dialogue,
                "status": "anchored",
                "anchor_method": "dialogue_text_similarity",
                "actual_speakers": list(
                    dict.fromkeys(item[2] for item in segments)
                ),
                "matched_segments": [
                    {
                        "start_sec": start,
                        "end_sec": end,
                        "speaker": speaker,
                        "text": text,
                        "dialogue_match_score": score,
                    }
                    for start, end, speaker, text, score in segments
                ],
            }

        alignments = [
            alignment("贺雨棠", "谢谢妈", (1.64, 2.06, 0, "谢谢妈", 1.0)),
            alignment(
                "李莲",
                "建军你快看雪芳终于想通了肯叫我妈了",
                (2.36, 3.20, 0, "建军你快看", 1.0),
                (3.62, 4.64, 1, "雪芳终于想通了", 0.86),
                (4.70, 5.48, 1, "肯叫我妈", 0.75),
            ),
            alignment(
                "贺雨棠",
                "现在只能指望你们了",
                (9.80, 11.06, 0, "以前的事记不起来了", 0.91),
                (11.12, 12.26, 0, "现在只能指望你们了", 1.0),
            ),
            alignment(
                "李莲",
                "行了你先歇着",
                (12.32, 12.86, 1, "行了", 1.0),
                (12.92, 13.52, 1, "你先歇着", 1.0),
                (13.76, 14.84, 1, "去后面菜地转转", 1.0),
            ),
        ]
        transcript = SpeechTranscript(
            language="zh",
            segments=tuple(
                SpeechSegment(start, end, text, "medium", speaker=speaker)
                for item in alignments
                for start, end, speaker, text, _score in [
                    (
                        segment["start_sec"],
                        segment["end_sec"],
                        segment["speaker"],
                        segment["text"],
                        segment["dialogue_match_score"],
                    )
                    for segment in item["matched_segments"]
                ]
            ),
            backend="fake",
            model="fake",
            device="cpu",
            metadata={
                "speech_evidence_status": "speech_present",
                "prompt_speech_plan": plan,
                "clustering": {"cluster_algorithm_status": "spectral_clustered"},
                "speaker_binding_evidence": {
                    "status": "fine_grained_turns",
                    "prompt_scope": "partial",
                    "prompt_turn_alignment": alignments,
                },
            },
        )
        agent = AuralisAgent(
            probe_video=lambda _path: {"has_audio": True},
            extract_audio=lambda _video, output: output.write_bytes(b"RIFF") or output,
            transcribe_speech=lambda _path: transcript,
            extract_subtitles=lambda _path: merge_subtitle_observations(
                [], frame_interval=0.5
            ),
            judge=lambda *_args: (),
        )

        with TemporaryDirectory() as temp_dir:
            result = agent.analyze(
                AuralisInput(
                    video_path=Path(temp_dir) / "video.mp4",
                    user_prompt="unused because transcript already carries the plan",
                )
            )

        binding_issues = [
            issue
            for issue in result.deterministic_issues
            if "角色台词绑定错误" in str(issue.get("问题说明") or "")
        ]
        self.assertEqual(len(binding_issues), 1)
        self.assertIn("建军你快看", binding_issues[0]["问题说明"])
        self.assertEqual(binding_issues[0]["时间区间"], "2.36s - 3.20s")
        self.assertEqual(
            result.diagnostics["speaker_binding_resolution"]["decision"],
            "supported",
        )

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
