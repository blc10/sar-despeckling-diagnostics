"""Checks of the revision measures (sardiag.revision)."""
import math
import unittest

import numpy as np

from sardiag import degradations, diagnostics, metrics, real_data, revision as rv, simulation as sim


class RevisionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.item = next(iter(sim.phantom_set("dv", "sarlo", n=1)))
        cls.rng = np.random.default_rng(7)

    def test_normalized_jsd_is_a_bounded_symmetric_divergence(self):
        a = self.rng.gamma(1.0, 1.0, 20000)
        b = self.rng.gamma(1.0, 1.0, 20000)
        c = self.rng.gamma(4.0, 0.25, 20000)
        self.assertLess(rv.fitted_gamma_jsd_normalized(a, b), 1e-3)
        j_ac, j_ca = rv.fitted_gamma_jsd_normalized(a, c), rv.fitted_gamma_jsd_normalized(c, a)
        self.assertAlmostEqual(j_ac, j_ca, places=6)
        self.assertTrue(0.0 < j_ac <= rv.LN2)
        self.assertLessEqual(rv.fitted_gamma_jsd_normalized(a, 1e4 * b), rv.LN2)
        self.assertGreater(rv.fitted_gamma_jsd_normalized(a, 1e4 * b), 0.99 * rv.LN2)

    def test_normalized_jsd_does_not_depend_on_the_grid(self):
        a, c = self.rng.gamma(1.0, 1.0, 20000), self.rng.gamma(2.0, 0.5, 20000)
        coarse = rv.fitted_gamma_jsd_normalized(a, c)
        old = rv.JSD_GRID
        try:
            rv.JSD_GRID = 4 * old
            fine = rv.fitted_gamma_jsd_normalized(a, c)
        finally:
            rv.JSD_GRID = old
        self.assertAlmostEqual(coarse, fine, places=4)

    def test_released_convention_scales_with_grid_spacing(self):
        a, c = self.rng.gamma(1.0, 1.0, 20000), self.rng.gamma(2.0, 0.5, 20000)
        released = diagnostics._fitted_gamma_jsd(a, c)
        approx = released * rv.released_jsd_step(a, c)
        self.assertAlmostEqual(approx, rv.fitted_gamma_jsd_normalized(a, c), delta=0.15 * approx)

    def test_level_aware_error_sees_radiometric_scaling(self):
        clean = self.item["clean"]
        self.assertLess(metrics.true_log_rmse(1.2 * clean, clean), 1e-6)
        self.assertAlmostEqual(rv.total_log_rmse(1.2 * clean, clean), math.log(1.2), places=5)
        self.assertAlmostEqual(rv.radiometric_log_bias(0.8 * clean, clean), math.log(0.8), places=5)

    def test_targetwise_error_matches_released_error_for_uniform_attenuation(self):
        item = self.item
        outputs = degradations.controlled_degradations(item["clean"].astype(np.float32), item["noisy"], item["coords"])
        estimate = next(o[3] for o in outputs if o[0] == "peak_keep_0.5")
        logs = rv.target_log_ratios(estimate, item["clean"], item["coords"])
        self.assertAlmostEqual(abs(float(np.log(np.median(np.exp(logs))))), metrics.peak_error(estimate, item["clean"], item["coords"]), places=5)
        self.assertGreaterEqual(rv.peak_summary(logs)["peak_abs_mean"], 0.0)

    def test_targetwise_error_does_not_cancel_opposite_signs(self):
        logs = np.array([-1.0, -1.0, 1.0, 1.0])
        self.assertAlmostEqual(rv.peak_summary(logs)["peak_abs_mean"], 1.0)
        self.assertAlmostEqual(abs(float(np.log(np.median(np.exp(logs))))), abs(math.log(math.cosh(1.0))), places=6)

    def test_proxy_table_reproduces_released_real_data_measure(self):
        item = self.item
        estimate = diagnostics.np.asarray(item["clean"]) * 0.0 + np.maximum(item["clean"], 1e-6)
        table = rv.proxy_target_table(item["noisy"], estimate, item["clean"], item["coords"])
        released, count = real_data.strong_scatterer_peak_error(item["noisy"], estimate)
        self.assertEqual(count, len(table))
        self.assertAlmostEqual(released, float(np.mean([abs(t["proxy"]) for t in table])), places=9)
        for t in table:
            self.assertAlmostEqual(t["proxy"], t["truth"] - t["select"], places=9)


if __name__ == "__main__":
    unittest.main()
