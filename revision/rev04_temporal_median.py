"""R3-3: what the temporal-median reference implies under the error of Equation (7), and how far its bias
correction depends on its assumptions.

    python revision/rev04_temporal_median.py [--phantoms 40]

Part A: on evaluation phantoms with their 11 additional dates, the difference between the exact reference X and the
median of the dates is reported (i) as a plain root-mean-square log difference (the 0.586 of the paper), (ii) after
dividing the median by its theoretical ratio (0.442) and (iii) under the mean-normalized error of Equation (7), which
is unchanged by any constant factor. The median-to-reference ratio is also given inside and outside the footprints of
the deterministic targets, where the fully-developed-speckle assumption does not hold.
Part B: Monte Carlo of the median ratio and the log-difference floor as a function of the number of looks and of the
correlation between dates, to delimit the assumptions (independent dates, single look, stable reflectivity).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT  # noqa: E402
from sardiag import metrics, simulation as sim  # noqa: E402


def part_a(n: int) -> dict:
    out = {}
    ratio11 = sim.median_bias(sim.EXTRA_DATES)
    for sensor in ("s1", "umbra", "sarlo"):
        rows = []
        for item in sim.phantom_set("eval", sensor, n=n, with_stack=True):
            clean = item["clean"].astype(np.float64)
            med = np.median(item["stack"].astype(np.float64), axis=0)
            log = np.log(np.clip(med, 1e-8, None) / clean)
            foot = np.zeros(clean.shape, dtype=bool)
            for py, px in item["coords"]:
                foot[max(py - 1, 0):py + 2, max(px - 1, 0):px + 2] = True
            rows.append({"ratio_of_means": med.mean() / clean.mean(), "pixel_ratio_elsewhere": float(np.mean((med / clean)[~foot])),
                         "pixel_ratio_targets": float(np.mean((med / clean)[foot])),
                         "plain": float(np.sqrt(np.mean(log ** 2))),
                         "theory_corrected": float(np.sqrt(np.mean((log - np.log(ratio11)) ** 2))),
                         "equation7": metrics.true_log_rmse(med, clean),
                         "equation7_after_correction": metrics.true_log_rmse(med / ratio11, clean),
                         "log_sd": float(np.std(log)),
                         "theory_corrected_targets": float(np.sqrt(np.mean((log - np.log(ratio11))[foot] ** 2)))})
        out[sensor] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]} | {"phantoms": len(rows)}
    return out


def part_b(samples: int = 400_000, n: int = 11) -> dict:
    rng = np.random.default_rng(20261112)
    out = {"looks": {}, "date_correlation": {}}
    for looks in (1, 2, 4, 8):
        z = rng.gamma(looks, 1.0 / looks, size=(samples, n))
        med = np.median(z, axis=1)
        out["looks"][str(looks)] = {"median_ratio": float(med.mean()), "plain": float(np.sqrt(np.mean(np.log(med) ** 2))),
                                    "after_ratio_correction": float(np.sqrt(np.mean((np.log(med / med.mean())) ** 2)))}
    for rho in (0.0, 0.2, 0.4, 0.6, 0.8):        # complex correlation between any two dates (common component)
        common = (rng.standard_normal(samples) + 1j * rng.standard_normal(samples)) / np.sqrt(2.0)
        own = (rng.standard_normal((samples, n)) + 1j * rng.standard_normal((samples, n))) / np.sqrt(2.0)
        z = np.abs(np.sqrt(rho) * common[:, None] + np.sqrt(1.0 - rho) * own) ** 2
        med = np.median(z, axis=1)
        out["date_correlation"][f"{rho:.1f}"] = {"median_ratio": float(med.mean()),
                                                 "residual_bias_db_after_iid_correction": float(10 * np.log10(med.mean() / sim.median_bias(n))),
                                                 "after_iid_correction": float(np.sqrt(np.mean(np.log(med / sim.median_bias(n)) ** 2)))}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phantoms", type=int, default=40, help="evaluation phantoms per class")
    args = ap.parse_args()
    a, b = part_a(args.phantoms), part_b()
    for sensor, v in a.items():
        print(f"[A] {sensor}: ratio of means {v['ratio_of_means']:.4f}; pixel ratio elsewhere {v['pixel_ratio_elsewhere']:.4f}, at targets {v['pixel_ratio_targets']:.4f}"
              f" | plain {v['plain']:.3f}, corrected {v['theory_corrected']:.3f}, Eq.7 {v['equation7']:.3f} (after correction {v['equation7_after_correction']:.3f})"
              f" | corrected, target pixels only {v['theory_corrected_targets']:.3f}")
    for k, v in b["looks"].items():
        print(f"[B] looks={k}: median ratio {v['median_ratio']:.4f}, plain floor {v['plain']:.3f}, corrected {v['after_ratio_correction']:.3f}")
    for k, v in b["date_correlation"].items():
        print(f"[B] date correlation {k}: median ratio {v['median_ratio']:.4f}, residual bias {v['residual_bias_db_after_iid_correction']:+.2f} dB, floor {v['after_iid_correction']:.3f}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "rev04_temporal_median.json").write_text(json.dumps({"phantoms": a, "monte_carlo": b}, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
