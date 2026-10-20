"""Recompute the main quantitative results of the paper from the released result tables.

    python scripts/reproduce_results.py            # writes results/summary/*.json and prints the key values
    python scripts/reproduce_results.py --quick    # fewer draws; prints the point estimates without writing files

The point estimates do not depend on the number of draws. With the default settings, the confidence intervals and
p-values are those of the paper because the resampling uses the same seeds and the same order of draws.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sardiag import simulation, statistics as st  # noqa: E402

RESULTS = ROOT / "results"
SUMMARY = RESULTS / "summary"

DIAGNOSTICS = {"r": "r", "jsd": "gamma_jsd", "m_index": "m_index", "tdm": "tdm_deviation", "tdm_published": "tdm_negated"}
TARGETS = {"log_rmse": "true_log_rmse", "edge": "edge_error"}
CORE = ["lee_classical_7x7", "kuan_7x7", "frost_7x7", "enhanced_lee_7x7", "gamma_map_7x7", "mean_filter_7x7",
        "point_passthrough", "no_point_passthrough", "guarded_anchor", "merlin_indomain", "temporal_unet_indomain",
        "sds_sar_s1_reimpl", "mulog_bm3d", "lee_sigma_improved"]
PUBLISHED = ["merlin_s1tops_official", "merlin_tsx_spotlight_official", "sar2sar_official_direct",
             "cl_sar_default_official", "cl_sar_t2_official"]
MAIN = ["lee_classical_7x7", "kuan_7x7", "frost_7x7", "enhanced_lee_7x7", "gamma_map_7x7", "mean_filter_7x7",
        "lee_sigma_improved", "mulog_bm3d", "sar2sar_official_direct", "sar2sar_official_sub2", "merlin_s1tops_official",
        "merlin_tsx_spotlight_official", "cl_sar_default_official", "cl_sar_t2_official", "temporal_unet_indomain",
        "merlin_indomain", "guarded_anchor", "sds_sar_s1_reimpl", "point_passthrough", "identity"]
PAIRS = [("r", "true_log_rmse"), ("jsd", "true_log_rmse"), ("m_index", "true_log_rmse"), ("tdm_dev", "true_log_rmse"),
         ("tdm_pub", "true_log_rmse"), ("r", "peak_error"), ("jsd", "peak_error"), ("m_index", "peak_error"),
         ("tdm_dev", "peak_error"), ("r", "edge_error"), ("jsd", "edge_error"), ("tdm_dev", "edge_error")]
PAIR_COLUMNS = {"jsd": "gamma_jsd", "tdm_dev": "tdm_deviation", "tdm_pub": "tdm_negated"}
# Names of the real-data measures in the protocol file and in the released table.
PROTOCOL_MEASURES = {"r": "r", "rf_jsd": "gamma_jsd", "rf_m": "m_index", "rf_tdm_d": "tdm_deviation",
                     "rf_enl_gain": "enl_gain", "rf_peak_log_err": "peak_error"}


def load(rel: str) -> pd.DataFrame:
    frame = pd.read_csv(RESULTS / rel, low_memory=False)
    if "tdm" in frame:
        frame["tdm_negated"] = -frame["tdm"]
    return frame


# ------------------------------------------------------------------------------------------------ controlled degradations
def degradation_aucs(frame: pd.DataFrame, draws: int) -> dict:
    """AUC of each diagnostic for outputs whose true error is above the median of the series (Tables 6 and 9)."""
    out = {}
    for t, tcol in TARGETS.items():
        label = (frame[tcol] > frame[tcol].median()).to_numpy()
        for d, dcol in DIAGNOSTICS.items():
            block = frame.assign(_label=label)[[dcol, "_label", "phantom"]].dropna()
            out[f"{d}|{t}"] = {"auc": st.auc(block[dcol], block["_label"]),
                               "ci": st.cluster_bootstrap(block, lambda b: st.auc(b[dcol], b["_label"]), draws=draws)}
    peak = frame[frame.family == "peak"]
    label = (peak.peak_error > np.log(2.0)).to_numpy()
    for d, dcol in DIAGNOSTICS.items():
        sub = peak.assign(_label=label)[[dcol, "_label"]].dropna()
        out[f"{d}|peak_within_family"] = {"auc": st.auc(sub[dcol], sub["_label"])}
    return out


# ------------------------------------------------------------------------------------------------ method level
def method_level(frame: pd.DataFrame, methods: list[str], draws: int, permutations: int) -> dict:
    """Spearman correlations between method means of diagnostics and true errors (Table 7)."""
    frame = frame[frame.method.isin(methods)]
    phantoms = sorted(set.intersection(*[set(frame[frame.method == m].phantom) for m in methods]))
    names = sorted({c for p in PAIRS for c in p})
    cube = np.stack([frame.pivot_table(index="phantom", columns="method", values=PAIR_COLUMNS.get(c, c))
                     .loc[phantoms, methods].to_numpy(float) for c in names], axis=-1)
    out = st.method_level_correlations(cube, names, PAIRS, draws=draws, permutations=permutations)
    return {"methods": methods, "phantoms": len(phantoms), "pairs": out}


# ------------------------------------------------------------------------------------------------ response sweep
def sweep_table(patches: pd.DataFrame, series: pd.DataFrame, draws: int) -> list[dict]:
    rows = []
    for (o, a), block in patches.groupby(["oversampling", "window_a"]):
        means = block[block.method.isin(MAIN)].groupby("method")[["true_log_rmse", "gamma_jsd", "peak_error"]].mean()
        others = means.drop(index="identity")
        aucs = degradation_aucs(series[(series.oversampling == o) & (series.window_a == a)], draws)
        rows.append({"oversampling": o, "window_a": a, "rho2_lag1": simulation.intensity_correlation(o, a, 1),
                     "auc_r": aucs["r|log_rmse"]["auc"], "auc_tdm_published": aucs["tdm_published|log_rmse"]["auc"],
                     "cl_sar_default_true_log_rmse": float(means.loc["cl_sar_default_official", "true_log_rmse"]),
                     "jsd_peak_rho": st.rank_correlation(others.gamma_jsd, others.peak_error)})
    return rows


# ------------------------------------------------------------------------------------------------ real data
def record_level(frame: pd.DataFrame, measures: list[str]) -> pd.DataFrame:
    g = frame.copy()
    g["enl_gain"] = np.log(np.clip(g["enl_gain"], 1e-12, None))   # averaged on a logarithmic scale
    return g.groupby(["record", "method"])[measures].mean().reset_index()


def record_bootstrap(rec: pd.DataFrame, a: str, b: str | None, sim: pd.Series | None, methods: list[str],
                     draws: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    records = rec.record.unique()
    groups = {r: g for r, g in rec.groupby("record")}
    values = []
    for _ in range(draws):
        sample = pd.concat([groups[r] for r in rng.choice(records, size=len(records), replace=True)], ignore_index=True)
        means = sample.groupby("method")[[c for c in (a, b) if c]].mean().reindex(methods)
        v = st.rank_correlation(sim.reindex(methods), means[a]) if sim is not None else st.rank_correlation(means[a], means[b])
        if np.isfinite(v):
            values.append(v)
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def real_data_tests(frame: pd.DataFrame, protocol: dict, draws: int, permutations: int) -> dict:
    """Hypotheses H1-H3 of the pre-specified protocol (Table 10)."""
    hyp = protocol["hypotheses"]
    seed, alpha = int(hyp["seed"]), float(hyp["alpha"])
    measures = list(PROTOCOL_MEASURES.values())
    predictions = {PROTOCOL_MEASURES[k]: pd.Series(v) for k, v in protocol["predictions"]["method_means"].items()
                   if k in PROTOCOL_MEASURES}
    out = {}
    for name, block in (("primary", frame[frame.primary]), ("all_records", frame)):
        rec = record_level(block, measures)
        means = rec.groupby("method")[measures].mean()
        methods = [m for m in means.index if m != "identity"]
        res = {"records": int(rec.record.nunique()), "patches": int(block.patch.nunique()), "methods": len(methods)}
        res["H1"] = {}
        for m in hyp["H1"]["methods"]:
            gain = rec[rec.method == m].enl_gain
            res["H1"][m] = {"median_enl_gain": float(np.exp(np.median(gain))),
                            "passed": bool(np.exp(np.median(gain)) >= hyp["H1"]["threshold"])}
        h2, p2 = {}, {}
        sub = rec[rec.method.isin(methods)]
        for k in hyp["H2"]["measures"]:
            d = PROTOCOL_MEASURES[k]
            x, y = predictions[d].reindex(methods), means[d].reindex(methods)
            p2[d] = st.one_sided_permutation_p(x, y, +1, permutations, seed)
            h2[d] = {"rho": st.rank_correlation(x, y), "p": p2[d],
                     "ci": record_bootstrap(sub, d, None, predictions[d], methods, draws, seed)}
        for d, p in st.holm(p2).items():
            h2[d].update(p_holm=p, supported=bool(p < alpha))
        h3, p3 = {}, {}
        for a, b in hyp["H3"]["pairs"]:
            key = f"{PROTOCOL_MEASURES[a]}|{PROTOCOL_MEASURES[b]}"
            sign = int(np.sign(protocol["predictions"]["sim_relations"][f"{a}|{b}"]))
            x, y = means[PROTOCOL_MEASURES[a]].reindex(methods), means[PROTOCOL_MEASURES[b]].reindex(methods)
            p3[key] = st.one_sided_permutation_p(x, y, sign, permutations, seed)
            h3[key] = {"rho": st.rank_correlation(x, y), "predicted_sign": sign, "p": p3[key],
                       "ci": record_bootstrap(sub, PROTOCOL_MEASURES[a], PROTOCOL_MEASURES[b], None, methods, draws, seed)}
        for k, p in st.holm(p3).items():
            h3[k].update(p_holm=p, supported=bool(p < alpha))
        res["H2"], res["H3"] = h2, h3
        out[name] = res
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="200 resampling draws and 1000 permutations")
    args = parser.parse_args()
    draws, perms = (200, 1000) if args.quick else (2000, 10000)

    def write(name: str, obj) -> None:
        if not args.quick:   # the released summaries hold the full-precision results
            SUMMARY.mkdir(exist_ok=True)
            (SUMMARY / name).write_text(json.dumps(obj, indent=1, default=float) + "\n", encoding="utf-8")

    series = load("simulation/controlled_degradations.csv.gz")
    aucs = {s: degradation_aucs(series[series.response == s], draws) for s in ("s1", "umbra", "sarlo", "s1_fitted")}
    semi = load("semi_synthetic/controlled_degradations.csv.gz")
    aucs.update({f"semi_{s}": degradation_aucs(semi[semi.response == s], draws) for s in ("s1", "umbra", "sarlo")})
    write("controlled_degradation_auc.json", aucs)
    print("AUC of r for the true log error:", {k: round(v["r|log_rmse"]["auc"], 3) for k, v in aucs.items()})

    patches = load("simulation/patch_metrics.csv.gz")
    ml = {}
    for s in ("s1", "umbra", "sarlo"):
        block = patches[patches.response == s]
        ml[f"{s}_core"] = method_level(block, CORE, draws, perms)
        ml[f"{s}_extended"] = method_level(block, CORE + PUBLISHED, draws, perms)
    write("method_level_correlations.json", ml)
    print("Method-level rho, JSD vs peak error:", {k: round(v["pairs"]["jsd|peak_error"]["rho"], 2) for k, v in ml.items()})

    sweep = sweep_table(load("response_sweep/patch_metrics.csv.gz"), load("response_sweep/controlled_degradations.csv.gz"),
                        min(draws, 200))
    write("response_sweep.json", sweep)

    protocol = json.loads((ROOT / "protocol" / "PROTOKOL.json").read_text(encoding="utf-8"))
    real = real_data_tests(load("real_data/patch_measures.csv.gz"), protocol, draws, perms)
    write("real_data_tests.json", real)
    print("H2 rho (primary set):", {k: round(v["rho"], 2) for k, v in real["primary"]["H2"].items()})
    print("H3 rho (primary set):", {k: round(v["rho"], 2) for k, v in real["primary"]["H3"].items()})


if __name__ == "__main__":
    main()
