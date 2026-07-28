import unittest

from av_eval.experiments import capacity_matrix, experiment_profiles


class ExperimentsTest(unittest.TestCase):
    def test_capacity_matrix_includes_required_counts_and_long_video_rates(self):
        matrix = capacity_matrix(duration_sec=30)
        self.assertEqual(matrix["image_counts"], [8, 16, 32, 48, 60, 80])
        self.assertEqual(matrix["long_video_frames"]["0.5"], 15)
        self.assertEqual(matrix["long_video_frames"]["2.0"], 60)

    def test_profiles_lock_baseline_differences(self):
        profiles = experiment_profiles()
        self.assertEqual(list(profiles), ["baseline_a", "harness_b", "harness_c"])
        self.assertEqual(profiles["baseline_a"]["audio_mode"], "none")
        self.assertEqual(profiles["harness_c"]["audio_mode"], "direct")


if __name__ == "__main__":
    unittest.main()
