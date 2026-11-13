"""Revision columns for despeckling METHODS (R3-4, R3-5, R3-6, R1-2, R4). Needs the method adapters, which are
not part of the public release, so this script has to be run on the authors' machine.

    python revision/rev07_methods.py --response sarlo [--phantoms 200] [--adapters my_adapters.py]

--adapters points to a Python file that defines  METHODS = {"name": callable}, where each callable receives the
unit-mean single-look intensity (float32, 256 x 256) and the complex observation and returns an intensity estimate:

    def lee(intensity, complex_z): ...
    METHODS = {"lee_classical_7x7": lee, ...}          # use the identifiers of results/README.md

Without --adapters, the unfiltered observation and a 7 x 7 mean filter are run as a self-test. The released columns
are recomputed as well, so the rows can be checked against results/simulation/patch_metrics.csv.gz (the script
prints the share of reproduced values per method). Outputs in revision/results/:
  methods_<response>.csv.gz          one row per phantom and method
  methods_targets_<response>.csv.gz  one row per simulated target (true log ratio) and per selected scatterer (proxy)
  methods_<response>_summary.json    method means, method-level relations and directions of the peak errors
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, load, spearman  # noqa: E402
from sardiag import evaluation, revision as rv, simulation as sim  # noqa: E402

CHECK = ["true_log_rmse", "edge_error", "peak_error", "r", "gamma_jsd", "m_index", "tdm", "tdm_deviation"]
PAIRS = [("gamma_jsd", "peak_error"), ("gamma_jsd", "peak_abs_mean"), ("m_index", "peak_abs_mean"), ("tdm_deviation", "peak_abs_mean"),
         ("tdm_deviation_norm", "peak_abs_mean"), ("tdm_deviation", "true_log_rmse"), ("tdm_deviation_norm", "true_log_rmse"),
         ("tdm_negated", "true_log_rmse"), ("tdm_norm_negated", "true_log_rmse"), ("obs_peak_abs_mean", "peak_abs_mean"),
         ("gamma_jsd", "det_pd_loss"), ("r", "det_pd_loss")]


def builtin() -> dict:
    def mean7(intensity, _):
        unit = intensity.astype(np.float64)
        out = uniform_filter(unit / unit.mean(), size=7, mode="reflect")
        return (out / out.mean()).astype(np.float32)
    return {"identity": lambda intensity, _: intensity, "mean_filter_7x7": mean7}


def load_adapters(path: Path | None) -> dict:
    if path is None:
        return builtin()
    spec = importlib.util.spec_from_file_location("adapters", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return {"identity": builtin()["identity"], **module.METHODS}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--response", choices=("s1", "umbra", "sarlo"), required=True)
    ap.add_argument("--phantoms", type=int, default=200, help="evaluation phantoms per class (200 in the study)")
    ap.add_argument("--adapters", type=Path)
    args = ap.parse_args()
    methods = load_adapters(args.adapters)
    rows, targets = [], []
    for item in sim.phantom_set("eval", args.response, n=args.phantoms):
        noisy, clean, coords, mask = item["noisy"], item["clean"], item["coords"], item["mask"]
        contrast = rv.target_contrast_db(clean, coords)
        t_obs, bg_obs = rv.detection_scores(noisy, coords)
        pd_obs = float(np.mean(t_obs > np.quantile(bg_obs, 1.0 - 1e-3)))
        for name, method in methods.items():
            estimate = np.clip(np.asarray(method(noisy, item["z"]), dtype=np.float32), 1e-8, None)
            base = {"response": args.response, "phantom": item["pid"], "phantom_class": item["klass"], "method": name}
            rec = evaluation.evaluate(noisy, estimate, clean, coords, mask, seed_key=f"x|{item['pid']}_{name}")
            rec.update(rv.evaluate_revision(noisy, estimate, clean, coords, mask))
            t_est, bg_est = rv.detection_scores(estimate, coords)
            rec["det_pd"] = float(np.mean(t_est > np.quantile(bg_est, 1.0 - 1e-3)))
            rec["det_pd_observation"], rec["det_pd_loss"] = pd_obs, pd_obs - rec["det_pd"]
            rows.append({**base, **rec})
            for k, v in enumerate(rv.target_log_ratios(estimate, clean, coords)):
                targets.append({**base, "kind": "target", "index": k, "contrast_db": float(contrast[k]), "truth": float(v)})
            for k, t in enumerate(rv.proxy_target_table(noisy, estimate, clean, coords)):
                targets.append({**base, "kind": "selected", "index": k, **{c: t[c] for c in
                                ("is_target", "contrast_db", "obs_contrast_db", "proxy", "truth", "select")}})
    frame = pd.DataFrame(rows)
    frame["tdm_negated"], frame["tdm_norm_negated"] = -frame["tdm"], -frame["tdm_norm"]
    tt = pd.DataFrame(targets)
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / f"methods_{args.response}.csv.gz", index=False, float_format="%.10g")
    tt.to_csv(OUT / f"methods_targets_{args.response}.csv.gz", index=False, float_format="%.8g")

    released = load("simulation/patch_metrics.csv.gz")
    released = released[released.response == args.response].set_index(["phantom", "method"])
    summary = {"methods": {}, "relations": {}}
    for name, block in frame.groupby("method", sort=False):
        key = list(zip(block.phantom, block.method))
        if all(k in released.index for k in key):
            ref = released.loc[key]
            ok = np.concatenate([np.isclose(block[c].to_numpy(), ref[c].to_numpy(), rtol=1e-5, atol=1e-6, equal_nan=True) for c in CHECK])
            reproduced = float(ok.mean())
        else:
            reproduced = float("nan")
        true = tt[(tt.method == name) & (tt.kind == "target")].truth.dropna()
        sel = tt[(tt.method == name) & (tt.kind == "selected")]
        summary["methods"][name] = {
            "released_values_reproduced": reproduced,
            **{c: float(block[c].mean()) for c in ("true_log_rmse", "peak_error", "peak_abs_mean", "peak_signed_mean", "tdm_deviation",
                                                  "tdm_deviation_norm", "tdm", "tdm_norm", "obs_peak_abs_mean", "det_pd", "det_pd_loss")},
            "targets": int(true.size), "targets_attenuated_share": float((true < -np.log(2)).mean()),
            "targets_amplified_share": float((true > np.log(2)).mean()),
            "phantoms_with_both_signs": float(((block.peak_frac_attenuated > 0) & (block.peak_frac_amplified > 0)).mean()),
            "proxy_true_target_share": float(sel.is_target.mean()),
            "proxy_vs_truth_true_targets": spearman(sel[sel.is_target].proxy.abs().to_numpy(), sel[sel.is_target].truth.abs().to_numpy()),
            "divergence_term_saturated_share": float(block.tdm_jsd_released_saturated.mean())}
        print(f"{name:<32} reproduced {reproduced:.3f}  peak released {block.peak_error.mean():.3f}  target-wise {block.peak_abs_mean.mean():.3f}"
              f"  signed {block.peak_signed_mean.mean():+.3f}  D {block.tdm_deviation.mean():.3f} -> {block.tdm_deviation_norm.mean():.3f}"
              f"  Pd {block.det_pd.mean():.2f} (obs {pd.Series(block.det_pd_observation).mean():.2f})")
    frame.loc[frame.method == "identity", ["m_index"]] = np.nan
    means = frame[frame.method != "identity"].groupby("method").mean(numeric_only=True)
    if len(means) >= 4:
        for a, b in PAIRS:
            summary["relations"][f"{a}|{b}"] = spearman(means[a].to_numpy(), means[b].to_numpy())
        print("method-level Spearman:", {k: round(v, 3) for k, v in summary["relations"].items()})
    else:
        print("fewer than four methods: method-level relations not computed")
    (OUT / f"methods_{args.response}_summary.json").write_text(json.dumps(summary, indent=1, default=float) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
