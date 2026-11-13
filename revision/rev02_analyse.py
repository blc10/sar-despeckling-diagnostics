"""R3-1 and R3-4 on the regenerated series: normalized TDM divergence and a level-aware error target.

    python revision/rev02_analyse.py          (after rev02_regenerate.py for s1, umbra and sarlo)

1. Checks that the regenerated released columns equal the released table.
2. Compares the released density-sum convention of the TDM divergence term with the normalized divergence:
   saturation share, rank agreement, and the AUCs of D and of the published TDM under both conventions.
3. Repeats the pooled AUC with the level-aware error (no mean normalization), for which radiometric scaling is an
   error rather than a correct output.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, auc, boot, ci, load, spearman  # noqa: E402

DIAG = {"r": "r", "jsd": "gamma_jsd", "m_index": "m_index", "tdm_dev": "tdm_deviation", "tdm_dev_norm": "tdm_deviation_norm",
        "tdm_pub": "tdm_negated", "tdm_pub_norm": "tdm_norm_negated"}
CHECK = ["true_log_rmse", "edge_error", "peak_error", "r", "gamma_jsd", "m_index", "tdm", "tdm_deviation"]


def table(block: pd.DataFrame, target: str, draws: int) -> dict:
    label = (block[target] > block[target].median()).to_numpy()
    out = {}
    for d, c in DIAG.items():
        s = block[c].to_numpy(float)
        b = boot(block.phantom.to_numpy(), draws, lambda i: auc(s[i], label[i]))
        out[d] = {"auc": auc(s, label), "ci": ci(b[:, 0])}
    return out


def main(draws: int = 1000) -> None:
    released = load("simulation/controlled_degradations.csv.gz")
    out = {}
    for sensor in ("s1", "umbra", "sarlo"):
        full = pd.read_csv(OUT / f"series_{sensor}.csv.gz")
        frame = full[full.family != "identity"].reset_index(drop=True)
        ref = released[released.response == sensor].set_index(["phantom", "degradation"]).loc[
            list(zip(frame.phantom, frame.degradation))].reset_index()
        close = np.concatenate([np.isclose(frame[c].to_numpy(), ref[c].to_numpy(), rtol=1e-6, atol=1e-7, equal_nan=True) for c in CHECK])
        worst = float(1.0 - close.mean())
        res = {"outputs": int(len(frame)), "share_of_released_values_not_reproduced": worst}
        jr, jn = frame.tdm_jsd.to_numpy(), frame.tdm_jsd_norm.to_numpy()
        res["divergence_term"] = {"released_saturated_share": float(frame.tdm_jsd_released_saturated.mean()),
                                  "spearman_released_vs_normalized": spearman(jr, jn),
                                  "median_released": float(np.median(jr)), "median_normalized": float(np.median(jn)),
                                  "median_ratio_released_over_normalized": float(np.median(jr / np.maximum(jn, 1e-12))),
                                  "share_of_D_from_divergence_released": float(np.median(0.5 * jr / np.maximum(frame.tdm_deviation, 1e-12))),
                                  "share_of_D_from_divergence_normalized": float(np.median(0.5 * jn / np.maximum(frame.tdm_deviation_norm, 1e-12)))}
        structural = frame[frame.family != "radio"]
        res["auc_equation7_pooled"] = table(frame, "true_log_rmse", draws)
        res["auc_equation7_structural"] = table(structural, "true_log_rmse", draws)
        res["auc_level_aware_pooled"] = table(frame, "total_log_rmse", draws)
        radio = frame[frame.family == "radio"]
        lab = (radio.radiometric_log_bias.abs() > 0.05).to_numpy()
        res["auc_radiometric_within_family"] = {d: auc(radio[c].to_numpy(float), lab) for d, c in DIAG.items()}
        peak = frame[frame.family == "peak"]
        lab = (peak.peak_error > np.log(2.0)).to_numpy()
        res["auc_peak_within_family"] = {d: auc(peak[c].to_numpy(float), lab) for d, c in DIAG.items()}
        out[sensor] = res
        print(f"{sensor}: released values not reproduced (rtol 1e-6): {100 * worst:.3f}%; divergence term saturated in "
              f"{100 * res['divergence_term']['released_saturated_share']:.1f}% of outputs, rank agreement "
              f"{res['divergence_term']['spearman_released_vs_normalized']:.3f}, released/normalized ~ "
              f"{res['divergence_term']['median_ratio_released_over_normalized']:.0f}x")
        for key in ("auc_equation7_pooled", "auc_equation7_structural", "auc_level_aware_pooled"):
            print(f"   {key:<26}" + "  ".join(f"{d}={v['auc']:.3f}" for d, v in res[key].items()))
        print("   radiometric within family " + "  ".join(f"{d}={v:.3f}" for d, v in res["auc_radiometric_within_family"].items()))
    (OUT / "rev02_normalized_tdm.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
