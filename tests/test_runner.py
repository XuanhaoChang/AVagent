import unittest
from pathlib import Path

from av_eval.runner import build_capacity_commands, build_experiment_commands


class RunnerTest(unittest.TestCase):
    def test_commands_keep_credentials_in_environment_and_separate_outputs(self):
        commands = build_experiment_commands(
            python="python3",
            script=Path("run_visual_baseline.py"),
            models={"gpt": "gpt-model", "seed_lite": "seed-model"},
            profiles=("baseline_a", "harness_b"),
            limit=2,
            start=1,
            output_root=Path("output/runs"),
        )
        self.assertEqual(len(commands), 4)
        flat = [" ".join(command) for command in commands]
        self.assertTrue(all("ARK_API_KEY" not in command for command in flat))
        self.assertIn("output/runs/gpt/baseline_a/pred.csv", flat[0])
        self.assertIn("--model gpt-model", flat[0])

    def test_capacity_commands_use_fixed_sample_and_required_image_counts(self):
        commands = build_capacity_commands(
            python="python3",
            script=Path("run_visual_baseline.py"),
            model="gpt-model",
            sample_index=39,
            image_counts=(8, 16, 32, 48, 60, 80),
            output_root=Path("output/capacity"),
        )
        self.assertEqual(len(commands), 6)
        self.assertIn("--start", commands[-1])
        self.assertIn("39", commands[-1])
        self.assertIn("--max-video-frames", commands[-1])
        self.assertIn("80", commands[-1])


if __name__ == "__main__":
    unittest.main()
