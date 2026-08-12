import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from av_eval.syncnet import classify_sync_result, evaluate_lip_sync


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

    def test_project_adapter_evaluates_detected_face_tracks(self):
        class Capture:
            def get(self, _field):
                return 30.0

            def release(self):
                return None

        fake_cv2 = SimpleNamespace(
            CAP_PROP_FPS=5,
            VideoCapture=lambda _path: Capture(),
        )

        class Detector:
            detect_results_dir = ""

            def __call__(self, **_kwargs):
                crop = Path(self.detect_results_dir) / "crop"
                crop.mkdir(parents=True)
                (crop / "track.mp4").touch()

        class Evaluator:
            def evaluate(self, *_args, **_kwargs):
                return -11, 0.25, 1.83

        with tempfile.TemporaryDirectory() as temporary, mock.patch.dict(
            sys.modules,
            {"cv2": fake_cv2},
        ):
            result = evaluate_lip_sync(
                "video.mp4",
                Evaluator(),
                temporary,
                syncnet_detector=Detector(),
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["sync_decision"], "desync_candidate")
        self.assertEqual(result["face_track_count"], 1)


if __name__ == "__main__":
    unittest.main()
