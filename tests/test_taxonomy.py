import unittest

from av_eval.taxonomy import (
    EXPLORATORY_DIMENSIONS,
    JING_TAXONOMY,
)


class TaxonomyTest(unittest.TestCase):
    def test_contains_all_jing_categories_and_exploratory_dimensions(self):
        self.assertEqual(len(JING_TAXONOMY), 13)
        self.assertEqual(
            EXPLORATORY_DIMENSIONS,
            ("美感", "动作协调性", "动作连续性"),
        )

if __name__ == "__main__":
    unittest.main()
