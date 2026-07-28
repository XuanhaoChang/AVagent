import csv
import json
import tempfile
import unittest
from pathlib import Path

from av_eval.review_export import attach_prediction_to_review_samples


class AttachPredictionTest(unittest.TestCase):
    def test_attaches_gpt_d_prediction_by_sample_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            samples = root / "samples"
            for index in (1, 2):
                sample = samples / f"sample_{index:03d}"
                sample.mkdir(parents=True)
                (sample / "input.json").write_text(
                    json.dumps({"序号": f"#{index}"}, ensure_ascii=False),
                    encoding="utf-8",
                )
            prediction = [{"问题类型": "音频质量问题", "问题说明": "台词错误"}]
            csv_path = root / "pred.csv"
            with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=["序号", "GPT预测结果"],
                )
                writer.writeheader()
                for index in (2, 1):
                    writer.writerow(
                        {
                            "序号": f"#{index}",
                            "GPT预测结果": json.dumps(
                                prediction,
                                ensure_ascii=False,
                            ),
                        }
                    )

            count = attach_prediction_to_review_samples(
                prediction_csv=csv_path,
                samples_root=samples,
                label="gpt_d",
            )

            self.assertEqual(count, 2)
            for index in (1, 2):
                value = json.loads(
                    (
                        samples
                        / f"sample_{index:03d}"
                        / "gpt_d.json"
                    ).read_text(encoding="utf-8")
                )
                self.assertEqual(value, prediction)

            replacement = [
                {"问题类型": "音频质量问题", "问题说明": "替换后的预测"}
            ]
            with csv_path.open("w", encoding="utf-8-sig", newline="") as file:
                writer = csv.DictWriter(
                    file,
                    fieldnames=["序号", "GPT预测结果"],
                )
                writer.writeheader()
                for index in (1, 2):
                    writer.writerow(
                        {
                            "序号": f"#{index}",
                            "GPT预测结果": json.dumps(
                                replacement,
                                ensure_ascii=False,
                            ),
                        }
                    )
            attach_prediction_to_review_samples(
                prediction_csv=csv_path,
                samples_root=samples,
                label="gpt_d",
                replace=True,
            )
            for index in (1, 2):
                value = json.loads(
                    (
                        samples
                        / f"sample_{index:03d}"
                        / "gpt_d.json"
                    ).read_text(encoding="utf-8")
                )
                self.assertEqual(value, replacement)


if __name__ == "__main__":
    unittest.main()
