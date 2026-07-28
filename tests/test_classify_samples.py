import json
import tempfile
import unittest
from pathlib import Path

from scripts.classify_samples_by_gpt_a import (
    build_parser,
    copy_classified_samples,
    read_source_reviews,
    resolve_output_root,
)


class ClassifySamplesTest(unittest.TestCase):
    def test_default_output_root_follows_prediction_source(self):
        args = build_parser().parse_args(["--prediction-source", "gpt_d"])
        self.assertEqual(
            resolve_output_root(args),
            Path("output/human_review_samples_by_gpt_d"),
        )

    def test_classifies_gpt_d_and_writes_gpt_d_review(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            samples = root / "samples"
            sample = samples / "sample_001"
            sample.mkdir(parents=True)
            (sample / "gt.json").write_text("[]\n", encoding="utf-8")
            (sample / "gpt_d.json").write_text("[]\n", encoding="utf-8")
            reviews = root / "reviews.jsonl"
            reviews.write_text(
                json.dumps(
                    {
                        "sample_id": "sample_001",
                        "reviews": [
                            {
                                "prediction_source": "gpt_d",
                                "category": 3,
                                "category_name": "没有完全指出GT的问题",
                                "reason": "遗漏问题。",
                                "gt_coverage": [],
                                "extra_prediction_indices": [],
                                "confidence": "高",
                            }
                        ],
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )

            records = read_source_reviews(reviews, "gpt_d")
            output = root / "classified"
            counts = copy_classified_samples(
                samples,
                output,
                records,
                prediction_source="gpt_d",
            )

            destination = (
                output
                / "category_3_没有完全指出GT的问题"
                / "sample_001"
            )
            self.assertEqual(counts[3], 1)
            self.assertTrue((destination / "gpt_d_review.json").is_file())
            summary = json.loads(
                (output / "summary.json").read_text(encoding="utf-8")
            )
            self.assertEqual(summary["prediction_source"], "gpt_d")


if __name__ == "__main__":
    unittest.main()
