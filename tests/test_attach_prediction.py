import csv
import json
import tempfile
import unittest
from pathlib import Path

from av_eval.review_export import (
    attach_auralis_evidence_to_review_samples,
    attach_prediction_to_review_samples,
)


class AttachPredictionTest(unittest.TestCase):
    def test_attaches_avagent_prediction_by_sample_order(self):
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
                label="avagent",
            )

            self.assertEqual(count, 2)
            for index in (1, 2):
                value = json.loads(
                    (
                        samples
                        / f"sample_{index:03d}"
                        / "avagent.json"
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
                label="avagent",
                replace=True,
            )
            for index in (1, 2):
                value = json.loads(
                    (
                        samples
                        / f"sample_{index:03d}"
                        / "avagent.json"
                    ).read_text(encoding="utf-8")
                )
                self.assertEqual(value, replacement)

    def test_attaches_auralis_asr_and_ocr_evidence_from_run_log(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            samples = root / "samples"
            sample = samples / "sample_019"
            sample.mkdir(parents=True)
            (sample / "input.json").write_text(
                json.dumps({"序号": "#91"}, ensure_ascii=False),
                encoding="utf-8",
            )
            run_log = root / "run.jsonl"
            record = {
                "row_index": 19,
                "序号": "#91",
                "auralis_audio": {
                    "status": "ok",
                    "asr_backend": "sensevoice",
                    "asr_model": "SenseVoiceSmall",
                    "asr_device": "cpu/funasr",
                    "subtitle_backend": "rapidocr",
                    "auralis_evidence": {
                        "media_metadata": {"duration_sec": 15.0},
                        "transcript": {"segments": [{"text": "测试"}]},
                        "constrained_asr": {"status": "no_reference_dialogue"},
                        "subtitles": {"segments": [{"text": "字幕"}]},
                        "alignment": {"issues": []},
                    },
                },
            }
            run_log.write_text(
                json.dumps(record, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            count = attach_auralis_evidence_to_review_samples(
                run_log=run_log,
                samples_root=samples,
            )

            self.assertEqual(count, 1)
            asr = json.loads((sample / "asr.json").read_text(encoding="utf-8"))
            ocr = json.loads((sample / "ocr.json").read_text(encoding="utf-8"))
            self.assertEqual(asr["sample_id"], "#91")
            self.assertEqual(asr["transcript"]["segments"][0]["text"], "测试")
            self.assertEqual(asr["constrained_asr"]["status"], "no_reference_dialogue")
            self.assertEqual(ocr["subtitles"]["segments"][0]["text"], "字幕")
            self.assertEqual(ocr["alignment"]["issues"], [])

    def test_attaches_empty_asr_and_ocr_evidence_for_no_audio(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            samples = root / "samples"
            sample = samples / "sample_005"
            sample.mkdir(parents=True)
            (sample / "input.json").write_text(
                json.dumps({"序号": "#32"}, ensure_ascii=False),
                encoding="utf-8",
            )
            run_log = root / "run.jsonl"
            record = {
                "row_index": 5,
                "序号": "#32",
                "auralis_audio": {
                    "status": "no_audio",
                    "auralis_diagnostics": {
                        "reason": "ffprobe did not detect an audio stream"
                    },
                },
            }
            run_log.write_text(
                json.dumps(record, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            count = attach_auralis_evidence_to_review_samples(
                run_log=run_log,
                samples_root=samples,
            )

            self.assertEqual(count, 1)
            asr = json.loads((sample / "asr.json").read_text(encoding="utf-8"))
            ocr = json.loads((sample / "ocr.json").read_text(encoding="utf-8"))
            self.assertEqual(asr["status"], "no_audio")
            self.assertEqual(asr["transcript"]["segments"], [])
            self.assertEqual(
                asr["diagnostics"]["reason"],
                "ffprobe did not detect an audio stream",
            )
            self.assertEqual(ocr["status"], "no_audio")
            self.assertEqual(ocr["subtitles"]["segments"], [])


if __name__ == "__main__":
    unittest.main()
