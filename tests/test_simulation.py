import unittest

import numpy as np

from sardiag import simulation as sim


class SimulationTest(unittest.TestCase):
    def test_median_bias(self):
        self.assertAlmostEqual(sim.median_bias(11), 0.7365, places=4)
        self.assertAlmostEqual(sim.median_bias(1), 1.0)

    def test_white_response_has_no_lag_correlation(self):
        self.assertAlmostEqual(sim.intensity_correlation(1.0, 1.0, 1), 0.0, places=10)
        self.assertGreater(sim.intensity_correlation(1.488, 0.70, 1), 0.4)

    def test_sentinel1_sidelobes(self):
        for resp in sim.SENTINEL1_IW.values():
            for os_, a in ((resp["os_az"], resp["a_az"]), (resp["os_rg"], resp["a_rg"])):
                self.assertLess(sim.pslr_irw(os_, a)["pslr_db"], -20.0)

    def test_phantoms_are_deterministic_and_shared_across_responses(self):
        a = list(sim.phantom_set("eval", "s1", n=1))
        b = list(sim.phantom_set("eval", "s1", n=1))
        c = list(sim.phantom_set("eval", "umbra", n=1))
        self.assertEqual(len(a), 2)
        for x, y, z in zip(a, b, c):
            np.testing.assert_array_equal(x["z"], y["z"])
            np.testing.assert_array_equal(x["phantom"]["background"], z["phantom"]["background"])
            self.assertEqual(x["coords"], z["coords"])

    def test_speckle_is_unit_mean(self):
        for sensor in ("s1", "umbra", "sarlo"):
            item = next(iter(sim.phantom_set("eval", sensor, n=1)))
            ratio = item["noisy"][item["mask"]] / item["clean"][item["mask"]]
            self.assertAlmostEqual(float(ratio.mean()), 1.0, delta=0.1)


if __name__ == "__main__":
    unittest.main()
