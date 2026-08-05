import sys
import unittest
from pathlib import Path


EVALUATION_DIR = Path(__file__).resolve().parents[1] / "third_party" / "AVBench" / "evaluation"
sys.path.insert(0, str(EVALUATION_DIR))

from evaluate_syncnet import classify_sync_result  # noqa: E402


class SyncDecisionTests(unittest.TestCase):
    def test_small_offset_is_not_a_sync_error_even_with_low_confidence(self):
        result = classify_sync_result(0.178, 2)
        self.assertEqual(result["sync_decision"], "aligned_or_no_large_offset")
        self.assertEqual(result["confidence_status"], "uncertain")

    def test_known_delay_is_a_candidate_even_when_confidence_is_low(self):
        result = classify_sync_result(1.83, -11)
        self.assertEqual(result["sync_decision"], "desync_candidate")
        self.assertFalse(result["offset_boundary_hit"])

    def test_search_boundary_is_inconclusive(self):
        result = classify_sync_result(8.0, 15)
        self.assertEqual(result["sync_decision"], "uncertain")
        self.assertTrue(result["offset_boundary_hit"])


if __name__ == "__main__":
    unittest.main()
