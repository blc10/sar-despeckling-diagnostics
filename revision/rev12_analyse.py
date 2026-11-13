"""Analysis of rev12_reflectivity_map.py: Table 9 AUCs with the MuLoG+BM3D map and with the temporal-mean map.

    python revision/rev12_analyse.py

For each response, scene series (pooled, structural) and map: AUC of each diagnostic for the mean-normalized log error,
with a tile bootstrap of the difference temporal_mean - mulog_bm3d (paired on the same tiles), and the ranking.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DIAG, OUT, auc, ci, cluster_draws  # noqa: E402


def aucs(b: pd.DataFrame) -> dict:
    lab = (b.true_log_rmse > b.true_log_rmse.median()).to_numpy()
    return {d: auc(b[c].to_numpy(float), lab) for d, c in DIAG.items()}


def main(draws: int = 1000) -> None:
    f = pd.read_csv(OUT / "rev12_degradations_by_map.csv.gz")
    f["tdm_negated"] = -f["tdm"]
    out = {}
    for resp in ("s1", "umbra", "sarlo"):
        for series in ("pooled", "structural"):
            b = f[(f.response == resp) & ((f.family != "radio") if series == "structural" else True)]
            parts = {mp: b[b["map"] == mp].reset_index(drop=True) for mp in ("mulog_bm3d", "temporal_mean")}
            a = {mp: aucs(v) for mp, v in parts.items()}
            tiles = np.sort(b.phantom.unique())
            rows = {mp: {t: np.flatnonzero(v.phantom.to_numpy() == t) for t in tiles} for mp, v in parts.items()}
            rng = np.random.default_rng(20261112)
            diffs = []
            for _ in range(draws):
                pick = rng.choice(tiles, len(tiles))
                res = {mp: aucs(v.iloc[np.concatenate([rows[mp][t] for t in pick])]) for mp, v in parts.items()}
                diffs.append({d: res["temporal_mean"][d] - res["mulog_bm3d"][d] for d in DIAG})
            key = f"{resp}|{series}"
            out[key] = {"mulog_bm3d": a["mulog_bm3d"], "temporal_mean": a["temporal_mean"],
                        "difference_ci": {d: ci([x[d] for x in diffs]) for d in DIAG},
                        "ranking": {mp: sorted(DIAG, key=lambda d: -a[mp][d]) for mp in a}}
            print(f"{key:<17}" + "  ".join(f"{d}: {a['mulog_bm3d'][d]:.3f}->{a['temporal_mean'][d]:.3f}" for d in DIAG))
    (OUT / "rev12_reflectivity_map.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 300)
