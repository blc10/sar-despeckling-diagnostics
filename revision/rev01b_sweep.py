"""R3-1 on the response sweep: does the direction of the published TDM depend on the degradation mixture?

    python revision/rev01b_sweep.py

For each of the 25 grid points (released table), the AUC of r, Gamma JSD and the published TDM is computed on the
pooled series of the paper and on the structural series (without radiometric scaling).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, auc, load, spearman  # noqa: E402
from sardiag import simulation as sim  # noqa: E402


def main() -> None:
    frame = load("response_sweep/controlled_degradations.csv.gz")
    rows = []
    for (os_, a), block in frame.groupby(["oversampling", "window_a"]):
        row = {"oversampling": os_, "window_a": a, "rho2_lag1": sim.intensity_correlation(os_, a, 1), "phantoms": block.phantom.nunique()}
        for name, b in (("pooled", block), ("structural", block[block.family != "radio"])):
            label = (b.true_log_rmse > b.true_log_rmse.median()).to_numpy()
            for d, c in (("r", "r"), ("jsd", "gamma_jsd"), ("tdm_pub", "tdm_negated"), ("tdm_dev", "tdm_deviation")):
                row[f"{d}_{name}"] = auc(b[c].to_numpy(float), label)
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "rev01b_sweep.csv", index=False, float_format="%.4f")
    with pd.option_context("display.width", 220, "display.float_format", "{:.3f}".format):
        print(out.to_string(index=False))
    for c in [c for c in out.columns if c.endswith(("pooled", "structural"))]:
        print(f"{c:<20} range {out[c].min():.3f}-{out[c].max():.3f}; grid points below 0.5: {(out[c] < 0.5).sum():>2}; rho with lag-1 correlation {spearman(out[c].to_numpy(), out.rho2_lag1.to_numpy()):+.2f}")


if __name__ == "__main__":
    main()
