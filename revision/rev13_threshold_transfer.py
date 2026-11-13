"""R3 minor 3: uncertainty of the threshold transfer of Table 8.

    python revision/rev13_threshold_transfer.py [--draws 2000]

Same rule as a162_statistics.threshold_transfer_all, on the released tables:
  diagnostics  Youden threshold on the development phantoms (even index, 40 of the 80 degradation phantoms) of the
               source response, label = true log error above the median of those outputs; balanced accuracy on the
               test phantoms (odd index, 40) of the target response, labelled with the median of the target
               development outputs;
  point rule   tau = 95th percentile of |log point ratio| of the unfiltered observation on the even evaluation
               phantoms of the source response; pass rate on the odd evaluation phantoms of the target response.
The point estimates are checked against Table 8. 95% intervals resample development and test phantoms
independently (threshold, labels and accuracy are all recomputed in each draw), so they include the uncertainty of
the threshold itself.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, ci, load  # noqa: E402
from sardiag import statistics as st  # noqa: E402

DIAG = {"r": "r", "jsd": "gamma_jsd", "m_index": "m_index", "tdm_dev": "tdm_deviation"}
RESP = ("s1", "umbra", "sarlo")
TABLE8 = {("r", "s1", "s1"): 0.752, ("r", "s1", "umbra"): 0.712, ("r", "umbra", "umbra"): 0.703,
          ("m_index", "s1", "s1"): 0.534, ("jsd", "sarlo", "sarlo"): 0.674, ("tdm_dev", "umbra", "sarlo"): 0.634}


def split(frame: pd.DataFrame):
    return frame[frame.phantom % 2 == 0], frame[frame.phantom % 2 == 1]


def transfer(dev_src, dev_dst, test_dst, col) -> tuple[float, float]:
    thr = st.youden_threshold(dev_src[col].to_numpy(float), (dev_src.true_log_rmse > dev_src.true_log_rmse.median()).to_numpy())
    lab = (test_dst.true_log_rmse > dev_dst.true_log_rmse.median()).to_numpy()
    return thr, st.balanced_accuracy(test_dst[col].to_numpy(float), lab, thr)


def resample(frame: pd.DataFrame, ids: np.ndarray, groups: dict) -> pd.DataFrame:
    return frame.iloc[np.concatenate([groups[i] for i in ids])]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=2000)
    args = ap.parse_args()
    rng = np.random.default_rng(20261112)
    dv = load("simulation/controlled_degradations.csv.gz")
    parts = {}
    for r in RESP:
        f = dv[dv.response == r].reset_index(drop=True)
        dev, test = split(f)
        dev, test = dev.reset_index(drop=True), test.reset_index(drop=True)
        parts[r] = {"dev": dev, "test": test,
                    "gdev": {i: np.flatnonzero(dev.phantom.to_numpy() == i) for i in dev.phantom.unique()},
                    "gtest": {i: np.flatnonzero(test.phantom.to_numpy() == i) for i in test.phantom.unique()}}
    dev_ids = {r: np.array(sorted(p["gdev"])) for r, p in parts.items()}
    test_ids = {r: np.array(sorted(p["gtest"])) for r, p in parts.items()}
    point = {}
    for d, col in DIAG.items():
        for s in RESP:
            for t in RESP:
                thr, ba = transfer(parts[s]["dev"], parts[t]["dev"], parts[t]["test"], col)
                point[(d, s, t)] = {"threshold": thr, "balanced_accuracy": ba}
    check = {f"{k[0]}|{k[1]}->{k[2]}": [round(point[k]["balanced_accuracy"], 3), v] for k, v in TABLE8.items()}
    draws = {k: [] for k in point}
    for _ in range(args.draws):
        # the same phantom draw is used for every diagnostic and pair (phantom ids are shared across responses)
        dpick = rng.choice(dev_ids["s1"], len(dev_ids["s1"]))
        tpick = rng.choice(test_ids["s1"], len(test_ids["s1"]))
        rs = {r: (resample(parts[r]["dev"], dpick, parts[r]["gdev"]), resample(parts[r]["test"], tpick, parts[r]["gtest"])) for r in RESP}
        for (d, s, t) in point:
            thr, ba = transfer(rs[s][0], rs[t][0], rs[t][1], DIAG[d])
            draws[(d, s, t)].append((thr, ba))
    out = {"check_against_table8": check, "diagnostics": {}}
    for k, v in point.items():
        arr = np.array(draws[k])
        out["diagnostics"][f"{k[0]}|{k[1]}->{k[2]}"] = {**v, "ci_balanced_accuracy": ci(arr[:, 1]), "ci_threshold": ci(arr[:, 0])}
    # loss of accuracy from transfer: target-own threshold minus transferred threshold, paired on the same draws
    out["transfer_loss"] = {}
    for d in DIAG:
        for s in RESP:
            for t in RESP:
                if s == t:
                    continue
                own, tr = np.array(draws[(d, t, t)])[:, 1], np.array(draws[(d, s, t)])[:, 1]
                out["transfer_loss"][f"{d}|{s}->{t}"] = {"loss": point[(d, t, t)]["balanced_accuracy"] - point[(d, s, t)]["balanced_accuracy"],
                                                        "ci": ci(own - tr)}
    pm = load("simulation/patch_metrics.csv.gz")
    pm = pm[pm.method == "identity"].copy()
    pm["index"] = pm.phantom.str.split("_").str[-1].astype(int)
    pm["alp"] = np.abs(np.log(np.clip(pm.point_ratio, 1e-8, None)))
    out["point_rule"] = {}
    for s in RESP:
        src = pm[(pm.response == s) & (pm["index"] % 2 == 0)].alp.to_numpy()
        tau = float(np.percentile(src, 95))
        taus = [np.percentile(rng.choice(src, src.size), 95) for _ in range(args.draws)]
        for t in RESP:
            tgt = pm[(pm.response == t) & (pm["index"] % 2 == 1)].alp.to_numpy()
            rates = [float((rng.choice(tgt, tgt.size) <= taus[i]).mean()) for i in range(args.draws)]
            out["point_rule"][f"{s}->{t}"] = {"tau": tau, "ci_tau": ci(taus), "pass_rate": float((tgt <= tau).mean()),
                                               "ci_pass_rate": ci(rates), "n_test": int(tgt.size)}
    (OUT / "rev13_threshold_transfer.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print("check against Table 8 (computed, published):", check)
    for k, v in out["diagnostics"].items():
        print(f"  {k:<22} BA {v['balanced_accuracy']:.3f} {np.round(v['ci_balanced_accuracy'], 3)}  thr {v['threshold']:.4g} {np.round(v['ci_threshold'], 4)}")
    for k, v in out["transfer_loss"].items():
        print(f"  loss {k:<20} {v['loss']:+.3f} {np.round(v['ci'], 3)}")
    for k, v in out["point_rule"].items():
        print(f"  tau {k:<14} tau {v['tau']:.3f} {np.round(v['ci_tau'], 3)} pass {v['pass_rate']:.3f} {np.round(v['ci_pass_rate'], 3)} n={v['n_test']}")


if __name__ == "__main__":
    main()
