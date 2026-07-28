import unittest

from av_eval.feishu_export import (
    build_bitable_records,
    flatten_prediction_rows,
    import_key,
)


class FeishuExportTest(unittest.TestCase):
    def test_flattens_one_issue_per_row_and_preserves_sample_id(self):
        source = [
            {
                "序号": "#1",
                "GPT预测结果": (
                    '[{"可定位性":"否","置信度":"高","问题说明":"x",'
                    '"问题类型":"音频质量问题","时间区间":"1s - 2s",'
                    '"关键帧秒":"","BBox":""}]'
                ),
            }
        ]
        rows = flatten_prediction_rows(source)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["序号"], "#1")
        self.assertEqual(rows[0]["问题序号"], 1)
        self.assertEqual(rows[0]["问题类型"], "音频质量问题")

    def test_bitable_records_have_stable_import_key_for_checkpointing(self):
        row = {"序号": "#1", "问题序号": 2, "问题说明": "x"}
        self.assertEqual(import_key(row), "#1:2")
        records = build_bitable_records([row])
        self.assertEqual(records[0]["fields"]["导入键"], "#1:2")


if __name__ == "__main__":
    unittest.main()
