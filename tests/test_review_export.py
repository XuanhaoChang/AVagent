import csv
import json
import tempfile
import unittest
from pathlib import Path

from av_eval.review_export import export_review_samples
from scripts.export_human_review_samples import build_parser


SOURCE_COLUMNS = [
    "序号",
    "user_prompt",
    "reference_image_urls",
    "generated_video_url",
    "用户反馈",
    "思考过程及标准答案",
]


def write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
        writer.writerows(rows)


class ReviewExportTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.media_root = self.root / "media"
        media_dir = self.media_root / "data02" / "demo"
        media_dir.mkdir(parents=True)
        (media_dir / "video.mp4").write_bytes(b"video-bytes")
        (media_dir / "ref-a.jpg").write_bytes(b"ref-a")
        (media_dir / "ref-b.png").write_bytes(b"ref-b")

        gold = [
            {
                "问题类型": "人物动作异常",
                "问题说明": "人物走路时重心异常",
                "时间区间": "0-2",
            }
        ]
        self.source_row = [
            "#1",
            "一个人向前行走",
            json.dumps(
                ["/data02/demo/ref-a.jpg", "/data02/demo/ref-b.png"],
                ensure_ascii=False,
            ),
            "/data02/demo/video.mp4",
            "走路不自然",
            "<thinking>不应导出</thinking>\n```json\n"
            + json.dumps(gold, ensure_ascii=False)
            + "\n```",
        ]
        self.gt_csv = self.root / "gt.csv"
        write_csv(self.gt_csv, SOURCE_COLUMNS, [self.source_row])

        self.prediction_csvs = {}
        for label in ("gpt_a", "gpt_b", "seed_a", "seed_b", "seed_c"):
            prediction = [
                {
                    "可定位性": "是",
                    "置信度": "高",
                    "问题说明": f"{label}发现动作异常",
                    "问题类型": "人物动作异常",
                    "时间区间": "0-2",
                    "关键帧秒": "1.0",
                    "BBox": "<bbox>0.1,0.1,0.9,0.9</bbox>",
                }
            ]
            path = self.root / f"{label}.csv"
            write_csv(
                path,
                SOURCE_COLUMNS + ["GPT预测结果"],
                [self.source_row + [json.dumps(prediction, ensure_ascii=False)]],
            )
            self.prediction_csvs[label] = path

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_exports_copied_media_and_multiline_json(self):
        output_root = self.root / "review"

        count = export_review_samples(
            gt_csv=self.gt_csv,
            prediction_csvs=self.prediction_csvs,
            media_root=self.media_root,
            output_root=output_root,
        )

        self.assertEqual(count, 1)
        sample = output_root / "sample_001"
        self.assertEqual((sample / "video.mp4").read_bytes(), b"video-bytes")
        self.assertEqual((sample / "reference_01.jpg").read_bytes(), b"ref-a")
        self.assertEqual((sample / "reference_02.png").read_bytes(), b"ref-b")
        json_names = (
            "input.json",
            "gt.json",
            "gpt_a.json",
            "gpt_b.json",
            "seed_a.json",
            "seed_b.json",
            "seed_c.json",
        )
        for name in json_names:
            text = (sample / name).read_text(encoding="utf-8")
            self.assertTrue(text.endswith("\n"), name)
            self.assertIn("\n  ", text, name)
            json.loads(text)
        self.assertNotIn("thinking", (sample / "gt.json").read_text(encoding="utf-8"))
        input_data = json.loads((sample / "input.json").read_text(encoding="utf-8"))
        self.assertNotIn("思考过程及标准答案", input_data)

    def test_rejects_nonempty_output_directory(self):
        output_root = self.root / "review"
        output_root.mkdir()
        (output_root / "notes.txt").write_text("人工备注", encoding="utf-8")

        with self.assertRaisesRegex(FileExistsError, "输出目录已存在且非空"):
            export_review_samples(
                gt_csv=self.gt_csv,
                prediction_csvs=self.prediction_csvs,
                media_root=self.media_root,
                output_root=output_root,
            )

    def test_cli_defaults_include_five_approved_predictions(self):
        args = build_parser().parse_args([])

        self.assertEqual(args.gt_csv, Path("input/gt.csv"))
        self.assertEqual(args.media_root, Path("input"))
        self.assertEqual(args.output_root, Path("output/human_review_samples"))
        self.assertEqual(args.gpt_a, Path("output/benchmark/runs/gpt/baseline_a/pred.csv"))
        self.assertEqual(args.gpt_b, Path("output/benchmark/runs/gpt/harness_b/pred.csv"))
        self.assertEqual(
            args.seed_c,
            Path("output/benchmark/runs/seed_lite/harness_c/pred.csv"),
        )


if __name__ == "__main__":
    unittest.main()
