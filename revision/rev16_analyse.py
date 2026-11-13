"""Summary of rev16_detection.py (R4-2): retention of CFAR-detected strong scatterers per method, real and simulated.

Retention = share of the pseudo-targets (observed contrast >= 14 dB, 12 dB as sensitivity) whose CFAR statistic in the
estimate exceeds the background quantile for the false-alarm probability (1e-3; 1e-2 and 1e-4 as sensitivity), pooled
over the pseudo-targets of all patches. Loss = retention of the observation (1 by construction) minus that of the method.
Method-level Spearman correlations of the loss with the released real-data diagnostics of the same patches
(all_patch_rf.csv) and with the simulated detection loss, with 95% intervals from resampling methods (as in rev06/rev11).
Simulation: the same rule on the SARLO-80 phantoms, compared with the detection probability of the known targets.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import OUT, ci  # noqa: E402
from sardiag import statistics as st  # noqa: E402

RF = "/mnt/user-data/uploads/02_Deney_Sonuclari/SAR/kodlar/outputs/a161_sarlo_real/all_patch_rf.csv"
PFA = {"1e-2": "q0.99", "1e-3": "q0.999", "1e-4": "q0.9999"}


def load(branch):
    f = OUT / f"detection_{branch}"
    s = pd.concat([pd.read_csv(p) for p in sorted(f.glob("scat_*.csv.gz"))], ignore_index=True)
    b = pd.concat([pd.read_csv(p) for p in sorted(f.glob("back_*.csv.gz"))], ignore_index=True)
    key = "patch" if "patch" in s else "phantom"
    return s.merge(b, on=[key, "method"]), b, key


def retention(m, th, q, extra=None):
    g = m[m.obs_contrast_db >= th]
    if extra is not None:
        g = g[extra(g)]
    return (g.stat > g[q]).groupby(g.method).mean()


def rho_ci(a, b, draws=2000, seed=20261114):
    a, b = np.asarray(a, float), np.asarray(b, float)
    picks = np.random.default_rng(seed).integers(0, len(a), size=(draws, len(a)))
    return {"rho": st.rank_correlation(a, b), "ci_method": ci([st.rank_correlation(a[i], b[i]) for i in picks])}


def main():
    out = {}
    m, _, key = load("real")
    out["patches"] = int(m.patch.nunique())
    obs = m[m.method == "identity"]
    for th in (12.0, 14.0):
        o = obs[obs.obs_contrast_db >= th]
        out[f"pseudo_targets|{th:g}dB"] = {"n": int(len(o)), "patches_with": int(o.patch.nunique()),
                                           "detected_in_observation_1e-3": float((o.stat > o["q0.999"]).mean())}
    rf = pd.read_csv(RF, low_memory=False)
    rf = rf[rf.patch.isin(m.patch.unique())]
    diag = rf.groupby("method")[["rf_jsd", "rf_m", "rf_tdm_d", "rf_peak_log_err", "rf_enl_gain", "r"]].mean()
    sim = json.loads((OUT / "rev10_sim_methods.json").read_text())["per_method"]
    sim_loss = pd.Series({k: v["det_pd_loss"] for k, v in sim.items()})
    methods = sorted(x for x in diag.index if x != "identity")
    for th in (12.0, 14.0):
        for p, q in PFA.items():
            ret = retention(m, th, q)
            loss = (ret.get("identity", 1.0) - ret).reindex(methods)
            blk = {"retention": ret.reindex(["identity"] + methods).round(4).to_dict()}
            for d in diag.columns:
                blk[f"loss_vs_{d}"] = rho_ci(loss, diag[d].reindex(methods))
            common = [x for x in methods if x in sim_loss.index]
            blk["loss_vs_sim_loss"] = rho_ci(loss.reindex(common), sim_loss.reindex(common))
            out[f"real|{th:g}dB|pfa{p}"] = blk
    # simulation validation
    try:
        ms, bs, _ = load("sim")
        out["sim_phantoms"] = int(ms.phantom.nunique())
        true_pd = bs.groupby("method").det_pd_true.mean()
        for th in (12.0, 14.0):
            ret = retention(ms, th, "q0.999")
            mm = sorted(x for x in ret.index if x != "identity")
            out[f"sim|{th:g}dB"] = {
                "share_true_targets": float(ms[(ms.method == "identity") & (ms.obs_contrast_db >= th)].is_target.mean()),
                "pseudo_retention_vs_true_pd": rho_ci(ret.reindex(mm), true_pd.reindex(mm)),
                "pseudo_retention_true_only_vs_true_pd": rho_ci(retention(ms, th, "q0.999", lambda g: g.is_target).reindex(mm), true_pd.reindex(mm)),
                "retention": ret.round(4).to_dict(), "true_pd": true_pd.round(4).to_dict()}
    except ValueError:
        pass
    (OUT / "rev16_detection.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if not isinstance(v, dict) or "retention" not in v}, indent=1)[:3000])


if __name__ == "__main__":
    main()
