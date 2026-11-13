"""R3-5, R1-2 and R4: target-wise point-target error, validity of the observation-referenced proxy, detection.

    python revision/rev05_peak_proxy.py       (after rev02_regenerate.py for s1, umbra and sarlo)

A. Target-wise absolute error against the released |log median ratio|: agreement, share of outputs with targets of
   both signs, direction of the errors.
B. Composition of the scatterers selected by the real-data rule in simulation: share that are simulated targets,
   speckle excursion of the selected peaks, value of the proxy for the exact reference, and the same quantities
   when the selection is restricted by observed contrast (post hoc candidate rule; not part of the protocol).
C. Proxy against the truth at the target level, by target strength.
D. Released method outputs: proxy against the true peak error between method means and between patches within
   a method.
E. Cell-averaging CFAR detection probability of the simulated targets before and after each controlled output.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, auc, ci, load, spearman  # noqa: E402

LN2 = np.log(2.0)
THRESHOLDS = (None, 8.0, 10.0, 12.0, 14.0)
SHOW = ["gauss_sigma_0.0", "gauss_sigma_1.0", "gauss_sigma_2.0", "mean_3x3", "mean_7x7", "edge_keep_0.4",
        "peak_keep_0.75", "peak_keep_0.5", "peak_keep_0.25", "peak_keep_0.1", "radio_1.2", "identity"]


def section_a(series: pd.DataFrame) -> dict:
    s = series[series.family != "identity"]
    mixed = (s.peak_frac_attenuated > 0) & (s.peak_frac_amplified > 0)
    gap = s.peak_abs_mean - s.peak_error
    return {"spearman_released_vs_targetwise": spearman(s.peak_error.to_numpy(), s.peak_abs_mean.to_numpy()),
            "mean_gap_targetwise_minus_released": float(gap.mean()), "max_gap": float(gap.max()),
            "share_outputs_gap_above_0.1": float((gap > 0.1).mean()),
            "share_outputs_with_attenuated_and_amplified_targets": float(mixed.mean()),
            "signed_mean_by_family": s.groupby("family").peak_signed_mean.mean().round(4).to_dict(),
            "abs_mean_by_family": s.groupby("family").peak_abs_mean.mean().round(4).to_dict(),
            "released_by_family": s.groupby("family").peak_error.mean().round(4).to_dict(),
            "missing_targets_total": int(s.peak_missing.sum())}


def section_b(targets: pd.DataFrame, patches: int) -> dict:
    out = {}
    ident = targets[targets.degradation == "identity"]
    exact = targets[targets.degradation == "gauss_sigma_0.0"]
    for t in THRESHOLDS:
        i = ident if t is None else ident[ident.obs_contrast_db >= t]
        e = exact if t is None else exact[exact.obs_contrast_db >= t]
        counts = i.groupby("phantom").size()
        out["released_rule" if t is None else f"obs_contrast>={t:g}dB"] = {
            "scatterers_per_patch": float(len(i) / patches), "patches_without_scatterer": int(patches - counts.size),
            "share_true_targets": float(i.is_target.mean()) if len(i) else float("nan"),
            "speckle_excursion_log_selected_nontargets": float(i[~i.is_target].select.mean()) if (~i.is_target).any() else float("nan"),
            "speckle_excursion_log_selected_targets": float(i[i.is_target].select.mean()) if i.is_target.any() else float("nan"),
            "proxy_of_exact_reference_mean_abs": float(e.proxy.abs().mean()) if len(e) else float("nan")}
    return out


def section_c(targets: pd.DataFrame) -> dict:
    t = targets[~targets.family.isin(["identity"])].copy()
    t["err"] = t.proxy - t.truth      # = -select: independent of the output
    out = {"all_selected": {"spearman_abs_proxy_vs_abs_truth": spearman(t.proxy.abs().to_numpy(), t.truth.abs().to_numpy()),
                            "rms_proxy_minus_truth": float(np.sqrt(np.mean(t.err ** 2)))}}
    for flag, name in ((True, "true_targets"), (False, "speckle_maxima")):
        b = t[t.is_target == flag]
        out[name] = {"n": int(len(b)), "spearman_abs_proxy_vs_abs_truth": spearman(b.proxy.abs().to_numpy(), b.truth.abs().to_numpy()),
                     "rms_proxy_minus_truth": float(np.sqrt(np.mean(b.err ** 2))), "mean_proxy_minus_truth": float(b.err.mean())}
    true = t[t.is_target]
    bins = pd.cut(true.contrast_db, [-np.inf, 10, 13, 16, np.inf], labels=["<10 dB", "10-13 dB", "13-16 dB", ">16 dB"])
    out["true_targets_by_reference_contrast"] = {
        str(k): {"n": int(len(b)), "mean_proxy_minus_truth": float(b.err.mean()), "rms_proxy_minus_truth": float(np.sqrt(np.mean(b.err ** 2))),
                 "spearman_abs_proxy_vs_abs_truth": spearman(b.proxy.abs().to_numpy(), b.truth.abs().to_numpy())}
        for k, b in true.groupby(bins, observed=True)}
    return out


def output_level(series: pd.DataFrame, targets: pd.DataFrame) -> dict:
    s = series[series.family != "identity"].set_index(["phantom", "degradation"])
    out = {"released_rule": spearman(s.obs_peak_abs_mean.to_numpy(), s.peak_abs_mean.to_numpy())}
    for t in THRESHOLDS[1:]:
        g = targets[(targets.obs_contrast_db >= t) & (targets.family != "identity")]
        proxy = g.assign(a=g.proxy.abs()).groupby(["phantom", "degradation"]).a.mean()
        joined = s.join(proxy.rename("restricted"), how="inner")
        out[f"obs_contrast>={t:g}dB"] = spearman(joined.restricted.to_numpy(), joined.peak_abs_mean.to_numpy())
    return out


def section_d() -> dict:
    out = {}
    for name, rel in (("phantoms", "simulation/patch_metrics.csv.gz"), ("semi_synthetic", "semi_synthetic/patch_metrics.csv.gz")):
        p = load(rel)
        p = p[p.obs_peak_error.notna() & (p.method != "identity") & ~p.method.str.startswith("sweep_")]
        for sensor, b in p.groupby("response"):
            m = b.assign(log_enl=np.log(np.clip(b.enl_homogeneous, 1e-8, None))).groupby("method")[
                ["obs_peak_error", "peak_error", "log_enl", "gamma_jsd"]].mean()
            within = [spearman(g.obs_peak_error.to_numpy(), g.peak_error.to_numpy()) for _, g in b.groupby("method")]
            res = lambda y, x, z: spearman(*(np.asarray(v) - np.polyval(np.polyfit(z, v, 1), z) for v in (
                pd.Series(y).rank().to_numpy(), pd.Series(x).rank().to_numpy())))  # partial rank correlation given z
            zr = m.log_enl.rank().to_numpy()
            out[f"{name}|{sensor}"] = {
                "methods": int(len(m)), "between_methods_proxy_vs_true": spearman(m.obs_peak_error.to_numpy(), m.peak_error.to_numpy()),
                "between_methods_proxy_vs_log_enl": spearman(m.obs_peak_error.to_numpy(), m.log_enl.to_numpy()),
                "between_methods_true_vs_log_enl": spearman(m.peak_error.to_numpy(), m.log_enl.to_numpy()),
                "between_methods_jsd_vs_true": spearman(m.gamma_jsd.to_numpy(), m.peak_error.to_numpy()),
                "between_methods_jsd_vs_true_given_log_enl": res(m.gamma_jsd, m.peak_error, zr),
                "within_method_patch_level_median": float(np.nanmedian(within)),
                "within_method_patch_level_range": [float(np.nanmin(within)), float(np.nanmax(within))]}
    return out


def section_e(series: pd.DataFrame) -> dict:
    g = series.groupby("degradation")[["det_pd_1e-03", "det_pd_1e-04", "gamma_jsd", "r", "peak_abs_mean"]].mean()
    out = {"by_output": {k: {c: float(v) for c, v in row.items()} for k, row in g.loc[[d for d in SHOW if d in g.index]].iterrows()}}
    peak = series[series.family == "peak"]
    ref = peak[peak.degradation == "peak_keep_1.0"].set_index("phantom")["det_pd_1e-03"]
    lost = (peak.phantom.map(ref) - peak["det_pd_1e-03"]).to_numpy() > 0
    out["auc_for_lost_detections_within_peak_family"] = {d: auc(peak[c].to_numpy(float), lost) for d, c in
                                                         (("r", "r"), ("jsd", "gamma_jsd"), ("m_index", "m_index"), ("tdm_dev", "tdm_deviation"))}
    out["share_peak_outputs_with_lost_detections"] = float(lost.mean())
    return out


def main() -> None:
    out = {"released_method_outputs": section_d()}
    for sensor in ("s1", "umbra", "sarlo"):
        series = pd.read_csv(OUT / f"series_{sensor}.csv.gz")
        targets = pd.read_csv(OUT / f"proxy_targets_{sensor}.csv.gz")
        res = {"targetwise": section_a(series), "selection": section_b(targets, series.phantom.nunique()),
               "target_level": section_c(targets), "output_level_proxy_vs_targetwise_truth": output_level(series, targets),
               "detection": section_e(series)}
        out[sensor] = res
        a, b = res["targetwise"], res["selection"]
        print(f"== {sensor}")
        print(f" A released vs target-wise peak error: rho {a['spearman_released_vs_targetwise']:.3f}; mixed-sign outputs {100 * a['share_outputs_with_attenuated_and_amplified_targets']:.1f}%;"
              f" gap>0.1 in {100 * a['share_outputs_gap_above_0.1']:.1f}% (max {a['max_gap']:.2f})")
        for k, v in b.items():
            print(f" B {k:<22} {v['scatterers_per_patch']:.2f}/patch, true targets {100 * v['share_true_targets']:.0f}%, proxy of exact reference {v['proxy_of_exact_reference_mean_abs']:.3f},"
                  f" patches without scatterer {v['patches_without_scatterer']}")
        c = res["target_level"]
        print(f" C target level rho |proxy| vs |truth|: all {c['all_selected']['spearman_abs_proxy_vs_abs_truth']:.3f}, true targets {c['true_targets']['spearman_abs_proxy_vs_abs_truth']:.3f},"
              f" speckle maxima {c['speckle_maxima']['spearman_abs_proxy_vs_abs_truth']:.3f}")
        print("   output level:", {k: round(v, 3) for k, v in res["output_level_proxy_vs_targetwise_truth"].items()})
        print(" E Pd(1e-3):", {k: round(v["det_pd_1e-03"], 2) for k, v in res["detection"]["by_output"].items()})
        print("   AUC for lost detections:", {k: round(v, 3) for k, v in res["detection"]["auc_for_lost_detections_within_peak_family"].items()})
    for k, v in out["released_method_outputs"].items():
        print(f" D {k:<26} between methods {v['between_methods_proxy_vs_true']:.3f}; within method (patches) median {v['within_method_patch_level_median']:.3f};"
              f" JSD vs true peak {v['between_methods_jsd_vs_true']:.3f}, given log ENL {v['between_methods_jsd_vs_true_given_log_enl']:.3f}")
    (OUT / "rev05_peak_proxy.json").write_text(json.dumps(out, indent=1, default=float) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
