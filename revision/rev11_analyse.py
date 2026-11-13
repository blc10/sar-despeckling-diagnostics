"""Analysis of rev11_real_methods.py: H2 and H3 of the sealed SARLO-80 test with the released proxy and with
contrast-conditioned versions of it, on the same subsample (post hoc).

    python revision/rev11_analyse.py

1. Reproduction of the released proxy on the subsample (against all_patch_rf.csv).
2. Released full-data statistics, the same statistics on the subsample (released proxy), and with the proxy
   restricted to scatterers of observed contrast >= 12 dB and >= 14 dB. H2 for the restricted proxies uses
   predictions computed the same way on the SARLO-response simulation (rev10 results). H3 pairs use the JSD and D
   of the same patches from all_patch_rf.csv. Intervals: bootstrap over the methods (as in rev06).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, ROOT, ci  # noqa: E402
from sardiag import statistics as st  # noqa: E402

RF = "/mnt/user-data/uploads/02_Deney_Sonuclari/SAR/kodlar/outputs/a161_sarlo_real/all_patch_rf.csv"
THRESHOLDS = (12.0, 14.0)


def restricted(scat: pd.DataFrame, th: float) -> pd.Series:
    g = scat[scat.obs_contrast_db >= th]
    return g.assign(a=g.proxy.abs()).groupby(["patch", "method"]).a.mean()


def stats(means: pd.DataFrame, pred: dict, proxies: list[str], methods: list[str], draws: int = 2000) -> dict:
    out = {}
    rng = np.random.default_rng(20261113)
    picks = rng.integers(0, len(methods), size=(draws, len(methods)))
    m = means.reindex(methods)
    for p in proxies:
        x = pred[p].reindex(methods).to_numpy(float)
        y = m[p].to_numpy(float)
        out[f"H2|{p}"] = {"rho": st.rank_correlation(x, y), "ci_method": ci([st.rank_correlation(x[i], y[i]) for i in picks])}
        for d in ("rf_jsd", "rf_tdm_d"):
            a, b = m[d].to_numpy(float), y
            out[f"H3|{d}|{p}"] = {"rho": st.rank_correlation(a, b), "ci_method": ci([st.rank_correlation(a[i], b[i]) for i in picks])}
    return out


def main() -> None:
    folder = OUT / "real_methods"
    rows = pd.concat([pd.read_csv(p) for p in sorted(folder.glob("rows_*.csv.gz"))], ignore_index=True)
    scat = pd.concat([pd.read_csv(p) for p in sorted(folder.glob("scatterers_*.csv.gz"))], ignore_index=True)
    rf = pd.read_csv(RF, low_memory=False)
    protocol = json.loads((ROOT / "protocol" / "PROTOKOL.json").read_text(encoding="utf-8"))
    pred = {k: pd.Series(v) for k, v in protocol["predictions"]["method_means"].items()}
    methods = sorted(m for m in pred["rf_peak_log_err"].index if m != "identity")

    j = rows.merge(rf[["patch", "method", "rf_peak_log_err", "rf_jsd", "rf_tdm_d", "rf_scatterers", "city"]], on=["patch", "method"], how="left")
    d = (j.proxy_released - j.rf_peak_log_err).abs()
    out = {"patches": int(rows.patch.nunique()), "cities": int(j.city.nunique()),
           "reproduction": {"max_abs_diff": float(d.max()), "share_within_0.01": float((d < 0.01).mean()),
                            "scatterer_count_equal": float((j.scatterers == j.rf_scatterers).mean())}}
    for th in THRESHOLDS:
        j = j.merge(restricted(scat, th).rename(f"proxy_ge{th:g}").reset_index(), on=["patch", "method"], how="left")
    have = {f"proxy_ge{th:g}": float(j.drop_duplicates("patch")[f"proxy_ge{th:g}"].notna().mean()) for th in THRESHOLDS}
    out["patches_with_scatterer_above_threshold"] = have

    # simulation predictions for the restricted proxies (same computation on the SARLO-response simulation)
    sim = pd.concat([pd.read_csv(p) for p in sorted((OUT / "sim_methods").glob("targets_*.csv.gz"))], ignore_index=True)
    sim = sim[sim.kind == "selected"].rename(columns={"phantom": "patch"})
    for th in THRESHOLDS:
        pred[f"proxy_ge{th:g}"] = restricted(sim, th).groupby("method").mean()
    simrows = pd.concat([pd.read_csv(p) for p in sorted((OUT / "sim_methods").glob("rows_*.csv.gz"))], ignore_index=True)
    pred["proxy_released_sim_rerun"] = simrows.groupby("method").obs_peak_abs_mean.mean()
    out["prediction_check_rho_released_vs_rerun"] = st.rank_correlation(pred["rf_peak_log_err"].reindex(methods).to_numpy(float),
                                                                         pred["proxy_released_sim_rerun"].reindex(methods).to_numpy(float))

    full = rf[rf.primary].groupby(["record", "method"])[["rf_peak_log_err", "rf_jsd", "rf_tdm_d"]].mean().groupby("method").mean()
    out["full_data_released"] = stats(full, pred, ["rf_peak_log_err"], methods)
    means = j.groupby("method")[["proxy_released", "rf_peak_log_err", "rf_jsd", "rf_tdm_d", "proxy_ge12", "proxy_ge14"]].mean()
    out["subsample"] = stats(means, pred | {"proxy_released": pred["rf_peak_log_err"]},
                             ["rf_peak_log_err", "proxy_released", "proxy_ge12", "proxy_ge14"], methods)
    out["method_means"] = means.reindex(methods).round(4).to_dict(orient="index")
    (OUT / "rev11_real_proxy.json").write_text(json.dumps(out, indent=1, default=float) + "\n", encoding="utf-8")
    print(f"{out['patches']} patches, {out['cities']} cities; reproduction {out['reproduction']}; with scatterer above threshold {have}")
    print("prediction check (released vs rerun simulation means):", round(out["prediction_check_rho_released_vs_rerun"], 3))
    for block in ("full_data_released", "subsample"):
        for k, v in out[block].items():
            print(f"  {block:<19} {k:<28} rho {v['rho']:+.3f}  method CI {np.round(v['ci_method'], 3)}")


if __name__ == "__main__":
    main()
