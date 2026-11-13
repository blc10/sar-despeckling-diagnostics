"""Analysis of rev10_sim_methods.py: the despeckling methods on the SARLO-response simulation (R3-4, R3-5, R1-2, R4).

    python revision/rev10_analyse.py

1. Reproduction: released columns of every row against a160_extension/sarlo/all_patch_metrics.csv.
2. Point targets per method: released |log median ratio|, target-wise absolute and signed error, shares of
   attenuated / amplified targets, phantoms with both signs.
3. Divergence term of the TDM: D and TDM with the released convention and with the normalized divergence, and the
   method ranking under each.
4. Observation-referenced proxy against the target-wise truth, with the released selection and with an observed-
   contrast condition; validity on true targets at the target level.
5. Detection: CFAR detection probability per method and its relation to the diagnostics.
6. Method-level relations of the diagnostics with the true errors (Spearman over the methods, identity excluded),
   including the JSD-peak relation given the smoothing strength (log ENL).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, spearman  # noqa: E402

RELEASED = {"true_log_rmse": "true_log_rmse", "peak_error": "abs_log_peak", "r": "r", "gamma_jsd": "jsd",
            "m_index": "m_index", "tdm": "tdm", "obs_peak_abs_mean": "rf_peak_log_err"}
EXCLUDE = ("identity",)
THRESHOLDS = (12.0, 14.0)


def partial_rank(x, y, z) -> float:
    rx, ry, rz = (pd.Series(v).rank().to_numpy() for v in (x, y, z))
    res = [v - np.polyval(np.polyfit(rz, v, 1), rz) for v in (rx, ry)]
    return float(np.corrcoef(*res)[0, 1])


def main(released_csv: str) -> None:
    folder = OUT / "sim_methods"
    rows = pd.concat([pd.read_csv(p) for p in sorted(folder.glob("rows_*.csv.gz"))], ignore_index=True)
    tg = pd.concat([pd.read_csv(p) for p in sorted(folder.glob("targets_*.csv.gz"))], ignore_index=True)
    rows["tdm_negated"], rows["tdm_norm_negated"] = -rows["tdm"], -rows["tdm_norm"]
    rows["log_enl"] = np.log(np.clip(rows["enl_homogeneous"], 1e-8, None))
    rows["det_pd_loss"] = rows["det_pd_observation"] - rows["det_pd"]
    out = {"phantoms": int(rows.phantom.nunique()), "methods": int(rows.method.nunique()),
           "classes": rows.drop_duplicates("phantom").phantom_class.value_counts().to_dict()}

    ref = pd.read_csv(released_csv).set_index(["phantom", "method"])
    joined = rows.set_index(["phantom", "method"]).join(ref[list(RELEASED.values())], how="inner", rsuffix="_rel")
    rep = {}
    for mine, theirs in RELEASED.items():
        a, b = joined[mine].to_numpy(float), joined[theirs if theirs not in joined or theirs == mine else theirs].to_numpy(float)
        if theirs == mine:
            b = joined[f"{mine}_rel"].to_numpy(float)
        rel = np.abs(a - b) / (np.abs(b) + 1e-6)
        rep[mine] = {"median_rel_dev": float(np.nanmedian(rel)), "share_within_1pct": float(np.nanmean(rel < 0.01)),
                     "spearman": spearman(a, b)}
    out["reproduction"] = {"rows_matched": int(len(joined)), "rows": int(len(rows)), "by_column": rep}
    worst = joined.assign(d=np.abs(joined.peak_error - joined.abs_log_peak)).groupby(level="method").d.max().sort_values().tail(3)
    out["reproduction"]["largest_peak_deviation_by_method"] = worst.round(4).to_dict()

    t = tg[tg.kind == "target"].dropna(subset=["truth"])
    per = rows.groupby("method").agg(true_log_rmse=("true_log_rmse", "mean"), peak_released=("peak_error", "mean"),
                                     peak_targetwise=("peak_abs_mean", "mean"), peak_signed=("peak_signed_mean", "mean"),
                                     tdm_dev=("tdm_deviation", "mean"), tdm_dev_norm=("tdm_deviation_norm", "mean"),
                                     tdm=("tdm", "mean"), tdm_norm=("tdm_norm", "mean"), jsd=("gamma_jsd", "mean"),
                                     m_index=("m_index", "mean"), r=("r", "mean"), log_enl=("log_enl", "mean"),
                                     proxy_released=("obs_peak_abs_mean", "mean"), det_pd=("det_pd", "mean"),
                                     det_pd_loss=("det_pd_loss", "mean"), saturated=("tdm_jsd_released_saturated", "mean"),
                                     both_signs=("peak_frac_attenuated", lambda s: 0.0))
    both = rows.assign(b=(rows.peak_frac_attenuated > 0) & (rows.peak_frac_amplified > 0)).groupby("method").b.mean()
    per["phantoms_with_both_signs"] = both
    per = per.drop(columns="both_signs")
    per["targets_attenuated"] = t.groupby("method").truth.apply(lambda v: float((v < -np.log(2)).mean()))
    per["targets_amplified"] = t.groupby("method").truth.apply(lambda v: float((v > np.log(2)).mean()))
    sel = tg[tg.kind == "selected"]
    per["proxy_true_target_share"] = sel.groupby("method").is_target.mean()
    for th in THRESHOLDS:
        g = sel[sel.obs_contrast_db >= th]
        per[f"proxy_ge{th:g}"] = g.assign(a=g.proxy.abs()).groupby(["method", "phantom"]).a.mean().groupby("method").mean()
        per[f"proxy_ge{th:g}_true_share"] = g.groupby("method").is_target.mean()
    out["per_method"] = per.round(5).to_dict(orient="index")

    m = per.drop(index=[e for e in EXCLUDE if e in per.index])
    rel = {}
    pairs = [("jsd", "peak_released"), ("jsd", "peak_targetwise"), ("m_index", "peak_targetwise"), ("tdm_dev", "peak_targetwise"),
             ("tdm_dev_norm", "peak_targetwise"), ("r", "true_log_rmse"), ("jsd", "true_log_rmse"), ("m_index", "true_log_rmse"),
             ("tdm_dev", "true_log_rmse"), ("tdm_dev_norm", "true_log_rmse"), ("tdm", "true_log_rmse"), ("tdm_norm", "true_log_rmse"),
             ("proxy_released", "peak_targetwise"), ("proxy_ge12", "peak_targetwise"), ("proxy_ge14", "peak_targetwise"),
             ("proxy_released", "log_enl"), ("proxy_ge14", "log_enl"), ("peak_targetwise", "log_enl"), ("peak_released", "peak_targetwise"),
             ("jsd", "det_pd_loss"), ("r", "det_pd_loss"), ("m_index", "det_pd_loss"), ("tdm_dev", "det_pd_loss"),
             ("peak_targetwise", "det_pd_loss"), ("tdm_dev", "tdm_dev_norm"), ("jsd", "log_enl")]
    for a, b in pairs:
        rel[f"{a}|{b}"] = spearman(m[a].to_numpy(float), m[b].to_numpy(float))
    rel["jsd|peak_targetwise|given_log_enl"] = partial_rank(m.jsd, m.peak_targetwise, m.log_enl)
    rel["proxy_released|peak_targetwise|given_log_enl"] = partial_rank(m.proxy_released, m.peak_targetwise, m.log_enl)
    rel["proxy_ge14|peak_targetwise|given_log_enl"] = partial_rank(m.proxy_ge14, m.peak_targetwise, m.log_enl)
    out["method_level"] = rel

    v = sel[sel.is_target]
    val = {"all": spearman(v.proxy.abs().to_numpy(), v.truth.abs().to_numpy())}
    for name, b in v.groupby("method"):
        val[name] = spearman(b.proxy.abs().to_numpy(), b.truth.abs().to_numpy())
    out["proxy_validity_true_targets"] = val
    patch = rows[~rows.method.isin(EXCLUDE)]
    within = [spearman(g.obs_peak_abs_mean.to_numpy(), g.peak_abs_mean.to_numpy()) for _, g in patch.groupby("method")]
    out["proxy_within_method_patch_level"] = {"median": float(np.nanmedian(within)), "range": [float(np.nanmin(within)), float(np.nanmax(within))]}
    (OUT / "rev10_sim_methods.json").write_text(json.dumps(out, indent=1, default=float) + "\n", encoding="utf-8")
    per.round(4).to_csv(OUT / "rev10_sim_methods_table.csv")

    print(f"{out['phantoms']} phantoms {out['classes']}, {out['methods']} methods; matched {out['reproduction']['rows_matched']}/{out['reproduction']['rows']}")
    for k, v in rep.items():
        print(f"  reproduction {k:<18} median rel dev {v['median_rel_dev']:.1e}, within 1% {100 * v['share_within_1pct']:.1f}%, rho {v['spearman']:.4f}")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(per[["true_log_rmse", "peak_released", "peak_targetwise", "peak_signed", "targets_attenuated", "targets_amplified",
                   "phantoms_with_both_signs", "tdm_dev", "tdm_dev_norm", "det_pd", "proxy_released", "proxy_ge14",
                   "proxy_true_target_share", "proxy_ge14_true_share"]].sort_values("peak_targetwise").round(3).to_string())
    print({k: round(x, 3) for k, x in rel.items()})
    print("proxy validity, true targets:", round(val["all"], 3), "; within-method patch level", out["proxy_within_method_patch_level"])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         "/mnt/user-data/uploads/02_Deney_Sonuclari/SAR/kodlar/outputs/a160_extension/sarlo/all_patch_metrics.csv")
