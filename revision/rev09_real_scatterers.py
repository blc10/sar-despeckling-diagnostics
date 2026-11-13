"""R3-5, R1-2 on real data: what the strong-scatterer rule selects in the 3060 SARLO-80 patches of the sealed test.

    python revision/rev09_real_scatterers.py --patches <dir with real_0.npy ... real_4.npy> --manifest <patch_manifest.csv>

real_k.npy are consecutive slices of a161_sarlo_real/patches.npy (complex, 3060 x 256 x 256), in manifest order. Each
patch is normalized as in the protocol (divided by the root of its mean power). Only the observation is used, so the
sealed test is not touched. For every selected scatterer the observed contrast (pixel over the mean of the 5-11 ring)
is recorded and compared with the same quantity in the SARLO-response simulation, where the share of selections that
are simulated targets is known for each contrast level (revision/results/proxy_targets_sarlo.csv.gz, unfiltered
observation). The resulting estimate of the true-target share on real data is a model-based extrapolation and is
reported as such.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT  # noqa: E402
from sardiag import revision as rv  # noqa: E402

BINS = [-np.inf, 6, 8, 10, 12, 14, 16, 20, np.inf]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patches", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    args = ap.parse_args()
    manifest = pd.read_csv(args.manifest)
    rows, k0 = [], 0
    for part in sorted(args.patches.glob("real_*.npy"), key=lambda p: int(p.stem.split("_")[1])):
        block = np.load(part, mmap_mode="r")
        for j in range(block.shape[0]):
            z = np.asarray(block[j], dtype=np.complex128)
            z = z / np.sqrt(max(float(np.mean(np.abs(z) ** 2)), 1e-20))
            intensity = np.abs(z) ** 2
            sel = rv.strong_scatterers(intensity)
            contrast = rv.target_contrast_db(intensity, sel) if sel else []
            m = manifest.iloc[k0 + j]
            for (r, c), cdb in zip(sel, contrast):
                rows.append({"patch": m.patch, "record": m.record, "city": m.city, "primary": bool(m.primary),
                             "row": r, "col": c, "obs_contrast_db": float(cdb)})
            if not sel:
                rows.append({"patch": m.patch, "record": m.record, "city": m.city, "primary": bool(m.primary),
                             "row": -1, "col": -1, "obs_contrast_db": np.nan})
        k0 += block.shape[0]
        print(part.name, k0, flush=True)
    assert k0 == len(manifest), (k0, len(manifest))
    real = pd.DataFrame(rows)
    real.to_csv(OUT / "rev09_real_scatterers.csv.gz", index=False, float_format="%.6g")

    sim = pd.read_csv(OUT / "proxy_targets_sarlo.csv.gz")
    sim = sim[sim.degradation == "identity"]
    sim["bin"] = pd.cut(sim.obs_contrast_db, BINS)
    share = sim.groupby("bin", observed=False).is_target.agg(["mean", "size"])
    sel = real[real.row >= 0].copy()
    sel["bin"] = pd.cut(sel.obs_contrast_db, BINS)
    out = {}
    for name, b in (("primary", sel[sel.primary]), ("all_records", sel)):
        dist = b.bin.value_counts(normalize=True, sort=False)
        est = float(np.nansum(dist.reindex(share.index).to_numpy() * share["mean"].to_numpy()))
        out[name] = {"patches": int(b.patch.nunique()), "selected": int(len(b)),
                     "per_patch": float(len(b) / (real[real.primary].patch.nunique() if name == "primary" else real.patch.nunique())),
                     "obs_contrast_quantiles_db": {q: float(b.obs_contrast_db.quantile(q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
                     "share_ge_12db": float((b.obs_contrast_db >= 12).mean()), "share_ge_14db": float((b.obs_contrast_db >= 14).mean()),
                     "patches_with_any_ge_14db": float(b.groupby("patch").obs_contrast_db.max().ge(14).mean()),
                     "estimated_true_target_share_from_simulation": est,
                     "by_city_median_contrast": b.groupby("city").obs_contrast_db.median().round(1).to_dict()}
    s_all = sim.obs_contrast_db
    out["simulation_sarlo"] = {"obs_contrast_quantiles_db": {q: float(s_all.quantile(q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)},
                               "share_ge_12db": float((s_all >= 12).mean()), "share_ge_14db": float((s_all >= 14).mean()),
                               "true_target_share": float(sim.is_target.mean()),
                               "true_target_share_by_bin": {str(k): [float(v["mean"]), int(v["size"])] for k, v in share.iterrows()}}
    (OUT / "rev09_real_scatterers.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    for k, v in out.items():
        print(k, json.dumps({a: b for a, b in v.items() if a != "by_city_median_contrast"}))


if __name__ == "__main__":
    main()
