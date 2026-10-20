"""Evaluate a despeckling method on the simulated phantoms under the three system responses.

    python scripts/evaluate_despeckler.py [--phantoms 20] [--output my_method.csv]

Replace despeckle() with the method to be evaluated. It receives a single-look intensity image with unit mean and must
return an intensity estimate of the same shape. The script reports the true errors, which are available because the
reflectivity is known, next to the no-reference diagnostics, so that the agreement between both can be inspected for
the method and the response of interest.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sardiag import evaluation, simulation as sim  # noqa: E402

COLUMNS = ["true_log_rmse", "edge_error", "peak_error", "psnr", "ssim", "r", "gamma_jsd", "m_index", "tdm_deviation"]


def despeckle(intensity: np.ndarray) -> np.ndarray:
    """Example method: 7 x 7 mean filter. Replace with the method to be evaluated."""
    return uniform_filter(intensity, size=7, mode="reflect")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phantoms", type=int, default=20, help="evaluation phantoms per class (at most 200)")
    parser.add_argument("--method", default="my_method", help="name stored with the results")
    parser.add_argument("--output", type=Path, help="optional CSV file for the per-phantom results")
    args = parser.parse_args()

    rows = []
    for sensor in ("s1", "umbra", "sarlo"):
        for item in sim.phantom_set("eval", sensor, n=args.phantoms):
            for name, estimate in (("observation", item["noisy"]), (args.method, despeckle(item["noisy"]))):
                rec = evaluation.evaluate(item["noisy"], estimate, item["clean"], item["coords"], item["mask"],
                                          seed_key=f"x|{item['pid']}_{name}")
                rows.append({"response": sensor, "phantom": item["pid"], "method": name, **rec})
    frame = pd.DataFrame(rows)
    frame.loc[frame.method == "observation", "m_index"] = np.nan   # constant ratio image: M-index not meaningful
    if args.output:
        frame.to_csv(args.output, index=False)
    with pd.option_context("display.width", 200, "display.max_columns", None, "display.float_format", "{:.3f}".format):
        print(frame.groupby(["response", "method"], sort=False)[COLUMNS].mean())


if __name__ == "__main__":
    main()
