import json
import tempfile
import unittest
from pathlib import Path

from av_eval.cli import main


class CliTest(unittest.TestCase):
    def test_writes_capacity_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "capacity.json"
            self.assertEqual(main(["capacity-plan", "--output", str(output)]), 0)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload["image_counts"], [8, 16, 32, 48, 60, 80])


if __name__ == "__main__":
    unittest.main()
