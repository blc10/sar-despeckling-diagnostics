"""The package must reproduce the rows of the released result tables."""
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter

from sardiag import degradations, evaluation, simulation as sim

RESULTS = Path(__file__).resolve().parents[1] / "results"
COLUMNS = ["true_log_rmse", "edge_error", "peak_error", "psnr", "ssim", "r", "gamma_jsd", "m_index", "tdm",
           "tdm_deviation"]


def assert_rows_match(test, record, row):
    for c in COLUMNS:
        if np.isfinite(row[c]):
            test.assertTrue(np.isclose(record[c], row[c], rtol=1e-6, atol=1e-7), f"{c}: {record[c]} != {row[c]}")


class ReleasedResultsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.series = pd.read_csv(RESULTS / "simulation" / "controlled_degradations.csv.gz")
        cls.patches = pd.read_csv(RESULTS / "simulation" / "patch_metrics.csv.gz", low_memory=False)

    def test_controlled_degradations(self):
        for response in ("s1", "umbra", "sarlo"):
            item = next(iter(sim.phantom_set("dv", response, n=1)))
            stored = self.series[(self.series.response == response) & (self.series.phantom == item["index"])]
            outputs = degradations.controlled_degradations(item["clean"].astype(np.float32), item["noisy"], item["coords"])
            self.assertEqual(len(outputs), len(stored))
            for name, _, _, estimate in outputs:
                record = evaluation.evaluate(item["noisy"], estimate, item["clean"], item["coords"], item["mask"],
                                             seed_key=degradations.seed_key(name, item["index"]))
                assert_rows_match(self, record, stored[stored.degradation == name].iloc[0])

    def test_method_outputs(self):
        for response in ("s1", "umbra", "sarlo"):
            item = next(iter(sim.phantom_set("eval", response, n=1)))
            noisy = item["noisy"]
            unit = noisy.astype(np.float64)
            mean7 = uniform_filter(unit / unit.mean(), size=7, mode="reflect")
            estimates = {"identity": noisy, "mean_filter_7x7": (mean7 / mean7.mean()).astype(np.float32)}
            for name, estimate in estimates.items():
                stored = self.patches[(self.patches.response == response) & (self.patches.phantom == item["pid"])
                                      & (self.patches.method == name)].iloc[0]
                record = evaluation.evaluate(noisy, estimate, item["clean"], item["coords"], item["mask"],
                                             seed_key=f"x|{item['pid']}_{name}")
                assert_rows_match(self, record, stored)


if __name__ == "__main__":
    unittest.main()
