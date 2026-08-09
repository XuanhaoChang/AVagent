import unittest

from agents.auralis.speaker_binding_resolver import resolve_speaker_binding


def _turn(role, text):
    return {
        "role": role,
        "dialogue_text": text,
        "prompt_start": 0,
        "prompt_end": len(text),
        "prompt_source_text": text,
    }


def _alignment(role, text, *segments):
    return {
        "role": role,
        "dialogue_text": text,
        "status": "anchored",
        "anchor_method": "dialogue_text_similarity",
        "actual_speakers": list(dict.fromkeys(item[2] for item in segments)),
        "matched_segments": [
            {
                "start_sec": start,
                "end_sec": end,
                "speaker": speaker,
                "text": observed,
                "dialogue_match_score": score,
            }
            for start, end, speaker, observed, score in segments
        ],
    }


def _voiceprint_pair(
    left_role,
    right_role,
    *,
    similarity=0.72,
    decision="same_speaker_supported",
):
    clips = [
        {
            "clip_id": "left",
            "prompt_turn_index": 0,
            "role": left_role,
            "dialogue_text": "第一句",
            "speaker": 0,
            "start_sec": 1.0,
            "end_sec": 2.0,
            "eligible": True,
            "quality_valid": True,
        },
        {
            "clip_id": "right",
            "prompt_turn_index": 1,
            "role": right_role,
            "dialogue_text": "第二句",
            "speaker": 0,
            "start_sec": 3.0,
            "end_sec": 4.0,
            "eligible": True,
            "quality_valid": True,
        },
    ]
    return {
        "status": "scored",
        "same_speaker_threshold": 0.55,
        "clips": clips,
        "pairs": [
            {
                "left_clip_id": "left",
                "right_clip_id": "right",
                "left_role": left_role,
                "right_role": right_role,
                "left_speaker": 0,
                "right_speaker": 0,
                "same_anonymous_speaker": True,
                "cosine_similarity": similarity,
                "decision": decision,
            }
        ],
    }


class SpeakerBindingResolverTest(unittest.TestCase):
    def test_supports_sample64_style_cross_role_conflict(self):
        plan = {
            "scope": "partial",
            "turns": [
                _turn("贺雨棠", "谢谢妈"),
                _turn("李莲", "建军你快看雪芳终于想通了肯叫我妈了"),
                _turn("贺雨棠", "现在只能指望你们了"),
                _turn("李莲", "行了你先歇着"),
            ],
        }
        alignments = [
            _alignment("贺雨棠", "谢谢妈", (1.64, 2.06, 0, "谢谢妈", 1.0)),
            _alignment(
                "李莲",
                "建军你快看雪芳终于想通了肯叫我妈了",
                (2.36, 3.20, 0, "建军你快看", 1.0),
                (3.62, 4.64, 1, "雪芳终于想通了", 0.86),
                (4.70, 5.48, 1, "肯叫我妈", 0.75),
            ),
            _alignment(
                "贺雨棠",
                "现在只能指望你们了",
                (9.80, 11.06, 0, "我以前的事记不起来了", 0.91),
                (11.12, 12.26, 0, "现在只能指望你们了", 1.0),
            ),
            _alignment(
                "李莲",
                "行了你先歇着",
                (12.32, 12.86, 1, "行了", 1.0),
                (12.92, 13.52, 1, "你先歇着", 1.0),
                (13.76, 14.84, 1, "我得去菜地转转", 1.0),
            ),
        ]

        result = resolve_speaker_binding(plan, alignments)

        self.assertEqual(result["decision"], "supported")
        self.assertEqual(len(result["issues"]), 1)
        self.assertIn("建军你快看", result["issues"][0]["问题说明"])
        self.assertIn("贺雨棠", result["issues"][0]["问题说明"])
        self.assertEqual(result["issues"][0]["时间区间"], "2.36s - 3.20s")

    def test_split_role_without_another_role_prototype_is_underdetermined(self):
        plan = {
            "scope": "closed",
            "turns": [_turn("新郎", "第一句"), _turn("新郎", "第二句")],
        }
        alignments = [
            _alignment("新郎", "第一句", (1.0, 2.3, 0, "第一句", 1.0)),
            _alignment("新郎", "第二句", (4.0, 7.4, 1, "第二句", 1.0)),
        ]

        result = resolve_speaker_binding(plan, alignments)

        self.assertEqual(result["decision"], "underdetermined")
        self.assertEqual(result["issues"], [])

    def test_single_shared_cluster_is_underdetermined(self):
        plan = {
            "scope": "partial",
            "turns": [
                _turn("伙计甲", "不是奸细"),
                _turn("林止", "别碰我"),
                _turn("薛琴", "住手"),
            ],
        }
        alignments = [
            _alignment("伙计甲", "不是奸细", (0.6, 1.6, 0, "不是奸细", 0.9)),
            _alignment("林止", "别碰我", (1.8, 3.1, 0, "别碰我", 1.0)),
            _alignment("薛琴", "住手", (5.5, 6.1, 0, "住手", 1.0)),
        ]

        result = resolve_speaker_binding(plan, alignments)

        self.assertEqual(result["decision"], "underdetermined")
        self.assertEqual(result["issues"], [])

    def test_direct_voiceprint_supports_clean_single_turn_role_pair(self):
        plan = {
            "scope": "partial",
            "turns": [_turn("甲", "第一句"), _turn("乙", "第二句")],
        }
        alignments = [
            _alignment("甲", "第一句", (1.0, 2.0, 0, "第一句", 1.0)),
            _alignment("乙", "第二句", (3.0, 4.0, 0, "第二句", 1.0)),
        ]

        result = resolve_speaker_binding(
            plan,
            alignments,
            voiceprint_evidence=_voiceprint_pair("甲", "乙"),
        )

        self.assertEqual(result["decision"], "supported")
        self.assertEqual(result["reason"], "direct_voiceprint_shared_role_voice")
        self.assertEqual(len(result["shared_voice_conflicts"]), 1)
        self.assertEqual(len(result["issues"]), 1)
        self.assertIn("余弦相似度为 0.720", result["issues"][0]["问题说明"])
        self.assertIn("甲", result["issues"][0]["问题说明"])
        self.assertIn("乙", result["issues"][0]["问题说明"])

    def test_sample15_style_ambiguous_pair_does_not_create_shared_voice_issue(self):
        plan = {
            "scope": "partial",
            "turns": [_turn("伙计甲", "第一句"), _turn("林止", "第二句")],
        }
        alignments = [
            _alignment("伙计甲", "第一句", (0.6, 1.68, 0, "第一句", 1.0)),
            _alignment("林止", "第二句", (1.8, 3.06, 0, "第二句", 1.0)),
        ]

        result = resolve_speaker_binding(
            plan,
            alignments,
            voiceprint_evidence=_voiceprint_pair(
                "伙计甲",
                "林止",
                similarity=0.415,
                decision="ambiguous",
            ),
        )

        self.assertEqual(result["decision"], "underdetermined")
        self.assertEqual(result["issues"], [])

    def test_supported_pair_does_not_expand_to_unverified_third_role(self):
        plan = {
            "scope": "partial",
            "turns": [
                _turn("甲", "第一句"),
                _turn("乙", "第二句"),
                _turn("丙", "第三句"),
            ],
        }
        alignments = [
            _alignment("甲", "第一句", (1.0, 2.0, 0, "第一句", 1.0)),
            _alignment("乙", "第二句", (3.0, 4.0, 0, "第二句", 1.0)),
            _alignment("丙", "第三句", (5.0, 6.0, 0, "第三句", 1.0)),
        ]

        result = resolve_speaker_binding(
            plan,
            alignments,
            voiceprint_evidence=_voiceprint_pair("甲", "乙"),
        )

        self.assertEqual(len(result["issues"]), 1)
        self.assertNotIn("丙", result["issues"][0]["问题说明"])

    def test_prompt_without_role_dialogue_scope_is_underdetermined(self):
        result = resolve_speaker_binding(
            {"scope": "none", "turns": []},
            [],
            binding_status="fine_grained_turns",
        )

        self.assertEqual(result["decision"], "underdetermined")
        self.assertEqual(
            result["reason"], "prompt_has_no_explicit_role_dialogue_scope"
        )

    def test_one_anchored_role_cannot_prove_cross_role_binding(self):
        plan = {
            "scope": "partial",
            "turns": [_turn("女主", "是你唤醒了我"), _turn("男主", "救我")],
        }
        alignments = [
            _alignment(
                "男主",
                "救我",
                (1.2, 9.9, 0, "求你是你唤醒了我救我", 1.0),
            )
        ]

        result = resolve_speaker_binding(plan, alignments)

        self.assertEqual(result["decision"], "underdetermined")
        self.assertEqual(result["issues"], [])

    def test_consistent_repeated_role_speakers_contradict_binding_error(self):
        plan = {
            "scope": "closed",
            "turns": [
                _turn("甲", "甲一"),
                _turn("乙", "乙一"),
                _turn("甲", "甲二"),
                _turn("乙", "乙二"),
            ],
        }
        alignments = [
            _alignment("甲", "甲一", (0.0, 1.0, 0, "甲一", 1.0)),
            _alignment("乙", "乙一", (1.2, 2.2, 1, "乙一", 1.0)),
            _alignment("甲", "甲二", (2.4, 3.4, 0, "甲二", 1.0)),
            _alignment("乙", "乙二", (3.6, 4.6, 1, "乙二", 1.0)),
        ]

        result = resolve_speaker_binding(plan, alignments)

        self.assertEqual(result["decision"], "contradicted")
        self.assertEqual(result["issues"], [])


if __name__ == "__main__":
    unittest.main()
