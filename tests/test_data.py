import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from av_eval.data import (
    extract_gold_array,
    resolve_legacy_media_path,
    safe_extract_tar,
)


class DataTest(unittest.TestCase):
    def test_extracts_last_fenced_json_array(self):
        text = '<thinking>ignored</thinking>\n```json\n[{"问题类型":"动作异常"}]\n```'
        result = extract_gold_array(text)
        self.assertEqual(result.status, "valid")
        self.assertEqual(result.items[0]["问题类型"], "动作异常")

    def test_reports_missing_fence_without_reading_thinking(self):
        result = extract_gold_array("<thinking>[{\"问题类型\":\"动作异常\"}]</thinking>")
        self.assertEqual(result.status, "needs_review")
        self.assertEqual(result.items, [])

    def test_maps_legacy_absolute_path_under_media_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "data02/a/video.mp4"
            target.parent.mkdir(parents=True)
            target.write_bytes(b"x")
            self.assertEqual(
                resolve_legacy_media_path("/data02/a/video.mp4", root),
                target.resolve(),
            )

    def test_rejects_tar_path_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / "bad.tar"
            with tarfile.open(archive, "w") as tar:
                info = tarfile.TarInfo("../escape.txt")
                payload = b"bad"
                info.size = len(payload)
                tar.addfile(info, io.BytesIO(payload))
            with self.assertRaises(ValueError):
                safe_extract_tar(archive, Path(tmp) / "out")


if __name__ == "__main__":
    unittest.main()
