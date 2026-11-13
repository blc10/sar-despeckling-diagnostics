"""R3-7: uncertainty of the real-data results under source-level resampling (released table only).

    python revision/rev06_real_data_uncertainty.py [--draws 2000]

The pre-specified analysis (protocol/PROTOKOL.json) resamples records. The 340 records come from two shards and 29
named scene locations, so records share scene content. This script documents that structure and recomputes the H2
and H3 statistics with (i) the pre-specified record bootstrap, (ii) a bootstrap that resamples scene locations
(column 'city'), (iii) a bootstrap that resamples the methods, because the pre-specified intervals condition on
the fixed method set, and (iv) leave-one-location-out and per-shard estimates. It also gives a location-bootstrap interval
for the H1 medians. Everything here is post hoc and labelled as such; the pre-specified tests are not changed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, ROOT, ci, cluster_draws, load  # noqa: E402
from sardiag import statistics as st  # noqa: E402

MEASURES = {"r": "r", "rf_jsd": "gamma_jsd", "rf_m": "m_index", "rf_tdm_d": "tdm_deviation", "rf_enl_gain": "enl_gain",
            "rf_peak_log_err": "peak_error"}


def record_table(frame: pd.DataFrame) -> pd.DataFrame:
    g = frame.copy()
    g["enl_gain"] = np.log(np.clip(g["enl_gain"], 1e-12, None))
    rec = g.groupby(["record", "method"])[list(MEASURES.values())].mean().reset_index()
    meta = frame.drop_duplicates("record").set_index("record")
    rec["city"] = rec.record.map(meta.city)
    rec["shard"] = rec.record.str.split("#").str[0]
    return rec


def statistics(rec: pd.DataFrame, predictions: dict, pairs: list, methods: list) -> dict:
    means = rec.groupby("method")[list(MEASURES.values())].mean().reindex(methods)
    out = {f"H2|{c}": st.rank_correlation(predictions[c].reindex(methods), means[c]) for c in MEASURES.values()}
    out.update({f"H3|{a}|{b}": st.rank_correlation(means[a], means[b]) for a, b in pairs})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=2000)
    args = ap.parse_args()
    protocol = json.loads((ROOT / "protocol" / "PROTOKOL.json").read_text(encoding="utf-8"))
    predictions = {MEASURES[k]: pd.Series(v) for k, v in protocol["predictions"]["method_means"].items() if k in MEASURES}
    pairs = [(MEASURES[a], MEASURES[b]) for a, b in protocol["hypotheses"]["H3"]["pairs"]]
    h1_methods, h1_threshold = protocol["hypotheses"]["H1"]["methods"], protocol["hypotheses"]["H1"]["threshold"]
    frame = load("real_data/patch_measures.csv.gz")
    out = {}
    for name, block in (("primary", frame[frame.primary]), ("all_records", frame)):
        rec = record_table(block)
        methods = sorted(m for m in rec.method.unique() if m != "identity")
        rec_m = rec[rec.method.isin(methods)]
        per_record = rec.drop_duplicates("record")
        res = {"structure": {"records": int(per_record.shape[0]), "shards": per_record.shard.value_counts().to_dict(),
                             "locations": int(per_record.city.nunique()),
                             "records_per_location": per_record.city.value_counts().describe()[["min", "50%", "max"]].to_dict(),
                             "largest_locations": per_record.city.value_counts().head(5).to_dict()}}
        point = statistics(rec_m, predictions, pairs, methods)
        index = rec_m.reset_index(drop=True)
        boots = {}
        for unit in ("record", "city"):
            draws = [statistics(index.iloc[i], predictions, pairs, methods) for i in cluster_draws(index[unit].to_numpy(), args.draws)]
            boots[unit] = {k: ci([d[k] for d in draws]) for k in point}
        means = rec_m.groupby("method")[list(MEASURES.values())].mean().reindex(methods)
        rng = np.random.default_rng(20261113)
        picks = rng.integers(0, len(methods), size=(args.draws, len(methods)))

        def method_ci(key: str) -> list[float]:
            parts = key.split("|")
            x = predictions[parts[1]].reindex(methods).to_numpy() if parts[0] == "H2" else means[parts[1]].to_numpy()
            y = means[parts[1]].to_numpy() if parts[0] == "H2" else means[parts[2]].to_numpy()
            return ci([st.rank_correlation(x[i], y[i]) for i in picks])

        loo = {c: statistics(rec_m[rec_m.city != c], predictions, pairs, methods) for c in per_record.city.unique()}
        shard = {s: statistics(rec_m[rec_m.shard == s], predictions, pairs, methods) for s in per_record.shard.unique()}
        res["tests"] = {k: {"rho": point[k], "ci_record": boots["record"][k], "ci_location": boots["city"][k],
                            "ci_method": method_ci(k),
                            "leave_one_location_out": [float(min(v[k] for v in loo.values())), float(max(v[k] for v in loo.values()))],
                            "per_shard": {s: v[k] for s, v in shard.items()}} for k in point}
        res["H1"] = {}
        for m in h1_methods:
            g = rec[rec.method == m].reset_index(drop=True)
            vals = g.enl_gain.to_numpy()
            b = [np.exp(np.median(vals[i])) for i in cluster_draws(g.city.to_numpy(), args.draws)]
            by_city = g.groupby("city").enl_gain.median().pipe(np.exp)
            res["H1"][m] = {"median_enl_gain": float(np.exp(np.median(vals))), "ci_location": ci(b), "threshold": h1_threshold,
                            "locations_below_threshold": int((by_city < h1_threshold).sum()), "locations": int(by_city.size),
                            "location_median_range": [float(by_city.min()), float(by_city.max())]}
        out[name] = res
        print(name, json.dumps(res["structure"]["shards"]), "locations:", res["structure"]["locations"])
        for k, v in res["tests"].items():
            print(f"  {k:<32} rho={v['rho']:+.3f}  record CI {np.round(v['ci_record'], 3)}  location CI {np.round(v['ci_location'], 3)}  method CI {np.round(v['ci_method'], 3)}"
                  f"  LOO {np.round(v['leave_one_location_out'], 3)}")
        for m, v in res["H1"].items():
            print(f"  H1 {m}: median {v['median_enl_gain']:.1f}, location CI {np.round(v['ci_location'], 1)}, "
                  f"{v['locations_below_threshold']}/{v['locations']} locations below {h1_threshold}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "rev06_real_data_uncertainty.json").write_text(json.dumps(out, indent=1, default=float) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
