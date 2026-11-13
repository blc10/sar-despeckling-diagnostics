"""Regenerate the controlled-degradation series with the revision columns (R3-1, R3-4, R3-5, R1-2, R4).

    python revision/rev02_regenerate.py --response umbra [--phantoms 80]

Writes revision/results/series_<response>.csv.gz (one row per degraded output; released columns plus the columns of
sardiag.revision.evaluate_revision and CFAR detection rates) and revision/results/proxy_targets_<response>.csv.gz
(one row per scatterer selected by the real-data rule). The unfiltered observation is added as family 'identity';
it is excluded from every AUC so that the 29-output series of the paper is unchanged.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sardiag import degradations, evaluation, revision as rv, simulation as sim  # noqa: E402

OUT = ROOT / "revision" / "results"
PFA = (1e-3, 1e-4)


def detection(image, coords) -> dict:
    target, background = rv.detection_scores(image, coords)
    return {f"det_pd_{p:.0e}": float(np.mean(target > np.quantile(background, 1.0 - p))) for p in PFA}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--response", choices=("s1", "umbra", "sarlo"), required=True)
    ap.add_argument("--phantoms", type=int, default=80)
    args = ap.parse_args()
    rows, targets = [], []
    for item in sim.phantom_set("dv", args.response, n=args.phantoms):
        clean, noisy, coords, mask = item["clean"].astype(np.float32), item["noisy"], item["coords"], item["mask"]
        outputs = degradations.controlled_degradations(clean, noisy, coords) + [("identity", "identity", 0.0, noisy)]
        contrast = rv.target_contrast_db(clean, coords)
        for name, family, level, estimate in outputs:
            base = {"response": args.response, "phantom": item["index"], "degradation": name, "family": family, "level": level}
            rec = evaluation.evaluate(noisy, estimate, clean, coords, mask, seed_key=degradations.seed_key(name, item["index"]))
            rec.update(rv.evaluate_revision(noisy, estimate, clean, coords, mask))
            rec.update(detection(estimate, coords))
            rec["n_targets"], rec["target_contrast_db_median"] = len(coords), float(np.median(contrast))
            rows.append({**base, **rec})
            for t in rv.proxy_target_table(noisy, estimate, clean, coords):
                targets.append({**base, **t})
    OUT.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame["tdm_negated"], frame["tdm_norm_negated"] = -frame["tdm"], -frame["tdm_norm"]
    frame.to_csv(OUT / f"series_{args.response}.csv.gz", index=False, float_format="%.10g")
    pd.DataFrame(targets).to_csv(OUT / f"proxy_targets_{args.response}.csv.gz", index=False, float_format="%.8g")
    print(args.response, len(frame), "outputs;", len(targets), "selected scatterers")


if __name__ == "__main__":
    main()
