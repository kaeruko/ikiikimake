from __future__ import annotations

import unittest

from analysis.analyze_region_rank_set import _normalize_ranks


class AnalyzeRegionRankSetTests(unittest.TestCase):
    def test_normalize_ranks_preserves_explicit_order(self):
        self.assertEqual(_normalize_ranks([3, 1, 5]), (3, 1, 5))

    def test_normalize_ranks_rejects_empty(self):
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            _normalize_ranks([])

    def test_normalize_ranks_rejects_duplicates(self):
        with self.assertRaisesRegex(ValueError, "duplicates"):
            _normalize_ranks([1, 2, 2])

    def test_normalize_ranks_rejects_bool_non_integer_and_zero(self):
        for ranks, error in (
            ([True], TypeError),
            ([1.0], TypeError),
            ([0], ValueError),
            ([-1], ValueError),
        ):
            with self.subTest(ranks=ranks), self.assertRaises(error):
                _normalize_ranks(ranks)


if __name__ == "__main__":
    unittest.main()
