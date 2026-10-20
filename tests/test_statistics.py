import unittest

import numpy as np

from sardiag import statistics as st


class StatisticsTest(unittest.TestCase):
    def test_auc(self):
        label = np.array([0, 0, 1, 1], dtype=bool)
        self.assertEqual(st.auc([1, 2, 3, 4], label), 1.0)
        self.assertEqual(st.auc([4, 3, 2, 1], label), 0.0)
        self.assertEqual(st.auc([1, 1, 1, 1], label), 0.5)
        self.assertTrue(np.isnan(st.auc([1, 2], np.array([1, 1], dtype=bool))))

    def test_holm(self):
        adjusted = st.holm({"a": 0.01, "b": 0.04, "c": 0.03})
        self.assertAlmostEqual(adjusted["a"], 0.03)
        self.assertAlmostEqual(adjusted["c"], 0.06)
        self.assertAlmostEqual(adjusted["b"], 0.06)

    def test_rank_correlation_and_concordance(self):
        x = np.arange(10.0)
        self.assertAlmostEqual(st.spearman(x, x ** 3), 1.0)
        self.assertAlmostEqual(st.rank_correlation(x, -x), -1.0)
        self.assertAlmostEqual(st.lin_ccc(x, x), 1.0)
        self.assertLess(st.lin_ccc(x, x + 5.0), 1.0)

    def test_one_sided_permutation(self):
        x = np.arange(12.0)
        self.assertLess(st.one_sided_permutation_p(x, x, +1, permutations=999), 0.01)
        self.assertGreater(st.one_sided_permutation_p(x, x, -1, permutations=999), 0.99)


if __name__ == "__main__":
    unittest.main()
