"""R3-1: sensitivity of the controlled-degradation AUCs to the error definition and the degradation mixture.

    python revision/rev01_auc_sensitivity.py [--draws 2000]

Uses only the released tables. For every response and both scene types it reports, with phantom-bootstrap 95%
intervals, the AUC of each diagnostic for the mean-normalized log error of Equation (7) under

  pooled            the series of the paper (5 families, 29 outputs)
  pooled_dedup      one undegraded output instead of four
  structural        without the radiometric-scaling family, median label recomputed
  structural_dedup  structural, one undegraded output
  drop_<family>     pooled series without one family (mixture sensitivity)

and, as threshold-free companions, the Spearman correlation between diagnostic and error on the structural series
and the detectability of each family against the undegraded outputs (AUC of degraded vs exact reference).
The paired difference AUC(jsd) - AUC(r) is resampled on the same phantom draws.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DIAG, OUT, UNDEGRADED, auc, boot, ci, load, spearman

TARGETS = {"log_rmse": "true_log_rmse", "edge": "edge_error"}
FAMILIES = ("gauss", "mean", "edge", "peak", "radio")


def scenarios(frame: pd.DataFrame) -> dict:
    dup = frame.degradation.isin(UNDEGRADED[1:])
    out = {"pooled": frame, "pooled_dedup": frame[~dup], "structural": frame[frame.family != "radio"],
           "structural_dedup": frame[(frame.family != "radio") & ~dup]}
    out.update({f"drop_{f}": frame[frame.family != f] for f in FAMILIES})
    return out


def analyse(frame: pd.DataFrame, draws: int) -> dict:
    res = {"n_scenes": int(frame.phantom.nunique()), "n_outputs": int(len(frame)), "auc": {}, "spearman_structural": {},
           "detectability": {}, "radiometric_axis": {}}
    for t, tcol in TARGETS.items():
        for name, block in scenarios(frame).items():
            label = (block[tcol] > block[tcol].median()).to_numpy()
            scores = {d: block[c].to_numpy(float) for d, c in DIAG.items()}
            point = {d: auc(s, label) for d, s in scores.items()}
            stats = [lambda i, s=s: auc(s[i], label[i]) for s in scores.values()]
            stats.append(lambda i: auc(scores["jsd"][i], label[i]) - auc(scores["r"][i], label[i]))
            b = boot(block.phantom.to_numpy(), draws, *stats)
            entry = {d: {"auc": point[d], "ci": ci(b[:, k])} for k, d in enumerate(DIAG)}
            entry["jsd_minus_r"] = {"diff": point["jsd"] - point["r"], "ci": ci(b[:, -1])}
            entry["n_outputs"], entry["n_positive"] = int(len(block)), int(label.sum())
            entry["positive_share_by_family"] = {f: float(label[(block.family == f).to_numpy()].mean())
                                                 for f in FAMILIES if (block.family == f).any()}
            res["auc"][f"{t}|{name}"] = entry
        block = frame[frame.family != "radio"]
        err = block[tcol].to_numpy(float)
        for d, c in DIAG.items():
            s = block[c].to_numpy(float)
            b = boot(block.phantom.to_numpy(), draws, lambda i: spearman(s[i], err[i]))
            res["spearman_structural"][f"{d}|{t}"] = {"rho": spearman(s, err), "ci": ci(b[:, 0])}
    reference = frame[frame.degradation == "gauss_sigma_0.0"]
    for f in FAMILIES:
        degraded = frame[(frame.family == f) & ~frame.degradation.isin(UNDEGRADED)]
        block = pd.concat([degraded.assign(_l=True), reference.assign(_l=False)], ignore_index=True)
        label = block._l.to_numpy()
        for d, c in DIAG.items():
            s = block[c].to_numpy(float)
            b = boot(block.phantom.to_numpy(), draws, lambda i: auc(s[i], label[i]))
            res["detectability"][f"{d}|{f}"] = {"auc": auc(s, label), "ci": ci(b[:, 0])}
    radio = frame[frame.family == "radio"]
    label = (radio.radiometric_error > 0.05).to_numpy()
    peak = frame[frame.family == "peak"]
    plabel = (peak.peak_error > np.log(2.0)).to_numpy()
    for d, c in DIAG.items():
        s, sp = radio[c].to_numpy(float), peak[c].to_numpy(float)
        b = boot(radio.phantom.to_numpy(), draws, lambda i: auc(s[i], label[i]))
        bp = boot(peak.phantom.to_numpy(), draws, lambda i: auc(sp[i], plabel[i]))
        res["radiometric_axis"][d] = {"auc": auc(s, label), "ci": ci(b[:, 0])}
        res.setdefault("peak_within_family", {})[d] = {"auc": auc(sp, plabel), "ci": ci(bp[:, 0])}
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=2000)
    args = ap.parse_args()
    out = {}
    sim = load("simulation/controlled_degradations.csv.gz")
    semi = load("semi_synthetic/controlled_degradations.csv.gz")
    for key, frame in [(s, sim[sim.response == s]) for s in ("s1", "umbra", "sarlo", "s1_fitted")] + \
                      [(f"semi_{s}", semi[semi.response == s]) for s in ("s1", "umbra", "sarlo")]:
        out[key] = analyse(frame, args.draws)
        a = out[key]["auc"]
        print(f"{key:<11} scenes={out[key]['n_scenes']:>3}  pooled r={a['log_rmse|pooled']['r']['auc']:.3f} jsd={a['log_rmse|pooled']['jsd']['auc']:.3f}"
              f" | structural r={a['log_rmse|structural']['r']['auc']:.3f} jsd={a['log_rmse|structural']['jsd']['auc']:.3f}"
              f" diff={a['log_rmse|structural']['jsd_minus_r']['diff']:+.3f} {np.round(a['log_rmse|structural']['jsd_minus_r']['ci'], 3)}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "rev01_auc_sensitivity.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    rows = [{"scene_set": k, "target": key.split("|")[0], "series": key.split("|")[1], "diagnostic": d, "auc": e[d]["auc"],
             "ci_low": e[d]["ci"][0], "ci_high": e[d]["ci"][1], "n_outputs": e["n_outputs"]}
            for k, v in out.items() for key, e in v["auc"].items() for d in DIAG]
    pd.DataFrame(rows).to_csv(OUT / "rev01_auc_table.csv", index=False, float_format="%.4f")


if __name__ == "__main__":
    main()
