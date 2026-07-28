import unittest

from av_eval.routing import route_observations


class RoutingTest(unittest.TestCase):
    def test_routes_audio_text_and_dense_motion_from_observable_inputs(self):
        result = route_observations(
            prompt="人物快速转身并说：你好。画面无字幕。",
            feedback="口型没有对上，手部一闪而过",
            has_audio=True,
            reference_count=3,
            duration_sec=20,
        )
        self.assertIn("asr", result.experts)
        self.assertIn("av_sync", result.experts)
        self.assertIn("ocr", result.experts)
        self.assertTrue(result.dense_sampling)
        self.assertTrue(result.local_crop)
        self.assertEqual(result.model_tier_candidate, "gpt_candidate")

    def test_jing_frequency_is_not_per_sample_evidence(self):
        result = route_observations(prompt="一片安静的湖面", feedback="", has_audio=False)
        self.assertEqual(result.experts, ())
        self.assertFalse(result.dense_sampling)
        self.assertEqual(result.model_tier_candidate, "seed_lite_candidate")


if __name__ == "__main__":
    unittest.main()
