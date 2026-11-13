"""R3-7 (five urban sites) and R1-4: sensitivity of the semi-synthetic results to the site the tiles come from.

    python revision/rev08_site_sensitivity.py [--draws 1000]

revision/semi_scenes.json is the scene list of the authors' package (a160_extension/semi_scenes/scenes.json): site,
tile position and split for each of the 140 tiles. Reported, for the controlled degradations (Table 9) and for the
method outputs on the evaluation tiles:
  * AUC of each diagnostic with all sites, leaving one site out, and with a site-level bootstrap;
  * method-level Spearman correlation of each diagnostic with the true log error, per site and leaving one site out.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DIAG, OUT, auc, ci, cluster_draws, load, spearman  # noqa: E402

HERE = Path(__file__).resolve().parent
SHORT = {"ankara_continental_urban_confirmatory": "Ankara", "milan_temperate_urban_holdout": "Milan",
         "montreal_urban_confirmatory": "Montreal", "seoul_urban_confirmatory": "Seoul",
         "warsaw_temperate_urban_holdout": "Warsaw"}
METHODS = ["lee_classical_7x7", "kuan_7x7", "frost_7x7", "enhanced_lee_7x7", "gamma_map_7x7", "mean_filter_7x7",
           "lee_sigma_improved", "mulog_bm3d", "sar2sar_official_direct", "sar2sar_official_sub2", "merlin_s1tops_official",
           "merlin_tsx_spotlight_official", "cl_sar_default_official", "cl_sar_t2_official", "temporal_unet_indomain",
           "merlin_indomain", "guarded_anchor", "sds_sar_s1_reimpl", "point_passthrough"]


def aucs(block: pd.DataFrame) -> dict:
    label = (block.true_log_rmse > block.true_log_rmse.median()).to_numpy()
    return {d: auc(block[c].to_numpy(float), label) for d, c in DIAG.items()}


def method_rho(block: pd.DataFrame) -> dict:
    m = block[block.method.isin(METHODS)].groupby("method").mean(numeric_only=True)
    return {d: spearman(m[c].to_numpy(float), m.true_log_rmse.to_numpy(float)) for d, c in DIAG.items()} | {"methods": int(len(m))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=1000)
    args = ap.parse_args()
    scenes = pd.DataFrame(json.loads((HERE / "semi_scenes.json").read_text(encoding="utf-8"))["scenes"])
    site_of = dict(zip(scenes.scene, scenes.site.map(SHORT)))
    deg = load("semi_synthetic/controlled_degradations.csv.gz")
    deg["site"] = deg.phantom.map(site_of)
    pm = load("semi_synthetic/patch_metrics.csv.gz")
    pm["site"] = pm.phantom.str.replace("semi_", "").astype(int).map(site_of)
    out = {"tiles": {"degradation": deg.drop_duplicates("phantom").site.value_counts().to_dict(),
                     "evaluation": pm.drop_duplicates("phantom").site.value_counts().to_dict()}}
    for resp in ("s1", "umbra", "sarlo"):
        b = deg[deg.response == resp].reset_index(drop=True)
        res = {"all": aucs(b), "leave_one_site_out": {s: aucs(b[b.site != s]) for s in SHORT.values()}}
        label = (b.true_log_rmse > b.true_log_rmse.median()).to_numpy()
        draws = [{d: auc(b[c].to_numpy(float)[i], label[i]) for d, c in DIAG.items()} for i in cluster_draws(b.site.to_numpy(), args.draws)]
        res["site_bootstrap_ci"] = {d: ci([x[d] for x in draws]) for d in DIAG}
        e = pm[pm.response == resp]
        res["method_level_all"] = method_rho(e)
        res["method_level_leave_one_site_out"] = {s: method_rho(e[e.site != s]) for s in SHORT.values()}
        res["method_level_per_site"] = {s: method_rho(e[e.site == s]) for s in SHORT.values()}
        out[resp] = res
        loo = res["leave_one_site_out"]
        print(f"== {resp}: degradation AUC (all) " + "  ".join(f"{d}={v:.3f}" for d, v in res["all"].items()))
        for d in ("r", "jsd", "tdm_dev"):
            vals = [loo[s][d] for s in SHORT.values()]
            print(f"   {d:<8} leave-one-site-out {min(vals):.3f}-{max(vals):.3f}; site bootstrap {np.round(res['site_bootstrap_ci'][d], 3)}")
        for d in ("r", "jsd", "m_index", "tdm_dev"):
            per = [res["method_level_per_site"][s][d] for s in SHORT.values()]
            print(f"   method-level rho({d}, log error): all {res['method_level_all'][d]:+.3f}; per site {min(per):+.3f} to {max(per):+.3f}")
    (OUT / "rev08_site_sensitivity.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print("tiles:", out["tiles"])


if __name__ == "__main__":
    main()
