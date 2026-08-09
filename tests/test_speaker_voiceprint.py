import unittest
from pathlib import Path

from agents.auralis.speaker_voiceprint import (
    build_role_voiceprint_clips,
    evaluate_prompt_voiceprints,
)
from tools.speech_transcription.schemas import SpeechSegment, SpeechTranscript


def _alignment(role, dialogue, start, end, *, speaker=0, score=1.0):
    return {
        "role": role,
        "dialogue_text": dialogue,
        "status": "anchored",
        "anchor_method": "dialogue_text_similarity",
        "actual_speakers": [speaker],
        "matched_segments": [
            {
                "start_sec": start,
                "end_sec": end,
                "speaker": speaker,
                "text": dialogue,
                "dialogue_match_score": score,
            }
        ],
    }


class SpeakerVoiceprintTest(unittest.TestCase):
    @staticmethod
    def _transcript() -> SpeechTranscript:
        return SpeechTranscript(
            language="zh",
            segments=(
                SpeechSegment(1.0, 2.1, "第一句", "medium", speaker=0),
                SpeechSegment(3.0, 4.2, "第二句", "medium", speaker=0),
                SpeechSegment(5.0, 5.5, "太短", "medium", speaker=0),
            ),
            backend="fake",
            model="fake",
            device="cpu",
            metadata={
                "speech_evidence_status": "speech_present",
                "speaker_binding_evidence": {
                    "status": "fine_grained_turns",
                    "prompt_turn_alignment": [
                        _alignment("甲", "第一句", 1.0, 2.1),
                        _alignment("乙", "第二句", 3.0, 4.2),
                        _alignment("丙", "太短", 5.0, 5.5),
                    ],
                },
            },
        )

    def test_builds_only_source_anchored_role_clip_candidates(self):
        clips = build_role_voiceprint_clips(self._transcript())

        self.assertEqual([item["role"] for item in clips], ["甲", "乙", "丙"])
        self.assertTrue(clips[0]["eligible"])
        self.assertTrue(clips[1]["eligible"])
        self.assertFalse(clips[2]["eligible"])
        self.assertEqual(
            clips[2]["rejection_reasons"],
            ["insufficient_anchored_speech_duration"],
        )

    def test_scores_role_pair_and_applies_strict_same_speaker_threshold(self):
        captured = []

        def scorer(_audio_path, clips):
            captured.extend(dict(item) for item in clips)
            return {
                "backend": "fake-campp",
                "model": "fake",
                "device": "cpu",
                "same_speaker_threshold": 0.55,
                "different_speaker_threshold": 0.30,
                "clips": [
                    {"clip_id": item["clip_id"], "quality_valid": True}
                    for item in clips
                ],
                "pairs": [
                    {
                        "left_clip_id": clips[0]["clip_id"],
                        "right_clip_id": clips[1]["clip_id"],
                        "cosine_similarity": 0.62,
                    }
                ],
            }

        evidence = evaluate_prompt_voiceprints(
            Path("audio.wav"),
            self._transcript(),
            scorer=scorer,
        )

        self.assertEqual([item["role"] for item in captured], ["甲", "乙"])
        self.assertEqual(evidence["status"], "scored")
        self.assertEqual(evidence["pairs"][0]["decision"], "same_speaker_supported")
        self.assertTrue(evidence["pairs"][0]["same_anonymous_speaker"])

    def test_shared_cluster_with_only_ambiguous_similarity_abstains(self):
        def scorer(_audio_path, clips):
            return {
                "same_speaker_threshold": 0.55,
                "different_speaker_threshold": 0.30,
                "clips": [
                    {"clip_id": item["clip_id"], "quality_valid": True}
                    for item in clips
                ],
                "pairs": [
                    {
                        "left_clip_id": clips[0]["clip_id"],
                        "right_clip_id": clips[1]["clip_id"],
                        "cosine_similarity": 0.415,
                    }
                ],
            }

        evidence = evaluate_prompt_voiceprints(
            Path("audio.wav"),
            self._transcript(),
            scorer=scorer,
        )

        self.assertEqual(evidence["pairs"][0]["decision"], "ambiguous")

    def test_rejects_clip_contaminated_by_previous_roles_dialogue(self):
        contaminated = _alignment(
            "赵舒悦",
            "快了嫂子再等会儿",
            12.03,
            14.97,
        )
        contaminated["matched_segments"][0]["text"] = (
            "我饿得心慌快了嫂子再等会儿"
        )
        transcript = SpeechTranscript(
            language="zh",
            segments=(
                SpeechSegment(
                    12.03,
                    14.97,
                    "我饿得心慌快了嫂子再等会儿",
                    "medium",
                    speaker=0,
                ),
            ),
            backend="fake",
            model="fake",
            device="cpu",
            metadata={
                "speaker_binding_evidence": {
                    "status": "fine_grained_turns",
                    "prompt_turn_alignment": [contaminated],
                }
            },
        )

        clips = build_role_voiceprint_clips(transcript)

        self.assertFalse(clips[0]["eligible"])
        self.assertIn(
            "observed_text_contains_other_dialogue",
            clips[0]["rejection_reasons"],
        )
        self.assertLess(clips[0]["dialogue_observed_precision_min"], 0.8)

    def test_music_contaminated_speech_does_not_call_voiceprint_scorer(self):
        transcript = self._transcript()
        transcript.metadata["speech_evidence_status"] = "speech_with_bgm"
        calls = []

        evidence = evaluate_prompt_voiceprints(
            Path("audio.wav"),
            transcript,
            scorer=lambda *_args: calls.append(True) or {},
        )

        self.assertEqual(evidence["status"], "not_evaluable")
        self.assertEqual(
            evidence["reason"],
            "speech_with_bgm_requires_robust_voice_separation",
        )
        self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
