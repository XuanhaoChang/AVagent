import csv
import json
import tempfile
import unittest
from pathlib import Path

from av_eval.audit import audit_dataset


class AuditTest(unittest.TestCase):
    def test_audits_media_and_gold_without_modifying_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            media = root / "data02/x"
            media.mkdir(parents=True)
            (media / "v.mp4").write_bytes(b"video")
            (media / "r.jpg").write_bytes(b"image")
            source = root / "gt.csv"
            header = [
                "序号",
                "user_prompt",
                "reference_image_urls",
                "generated_video_url",
                "用户反馈",
                "思考过程及标准答案",
            ]
            with source.open("w", encoding="utf-8-sig", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(header)
                writer.writerow(
                    [
                        "#1",
                        "prompt",
                        json.dumps(["/data02/x/r.jpg"]),
                        "/data02/x/v.mp4",
                        "feedback",
                        '```json\n[{"问题类型":"动作异常"}]\n```',
                    ]
                )
            before = source.read_bytes()
            result = audit_dataset(source, root, probe=False)
            self.assertEqual(result.summary["sample_count"], 1)
            self.assertEqual(result.summary["resolved_video_count"], 1)
            self.assertEqual(result.summary["resolved_reference_count"], 1)
            self.assertEqual(result.summary["valid_gold_count"], 1)
            self.assertEqual(source.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
