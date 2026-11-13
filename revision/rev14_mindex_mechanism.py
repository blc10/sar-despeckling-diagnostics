"""R1-4: why does the M-index invert on the semi-synthetic urban scenes but not on the phantoms?

    python revision/rev14_mindex_mechanism.py --scenes <a160_extension/semi_scenes> [--phantoms 80] [--workers 2]

For every controlled-degradation output (the series of Tables 6 and 9) the M-index is split into its two terms
(first-order term on homogeneous blocks, co-occurrence structure term) and the first-order term is recomputed with
other block selections: 5%, 50% and 100% of the blocks selected by the coefficient of variation of the observation
(the released rule uses 20%), and an oracle that selects the 20% of blocks with the lowest coefficient of variation of
the reference X (truly homogeneous blocks). The coefficient of variation of X inside the released selection measures
how much scene structure the 'homogeneous' blocks contain. Everything else follows sardiag.diagnostics.m_index.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from sardiag import degradations, diagnostics as dg, metrics, simulation as sim  # noqa: E402

OUT = ROOT / "revision" / "results"
FRACTIONS = (0.05, 0.2, 0.5, 1.0)
G: dict = {}


def blocks_by(score_image, valid, fraction):
    cand = []
    h, w = score_image.shape
    B = dg.BLOCK_SIZE
    for r in range(0, h - B + 1, B):
        for c in range(0, w - B + 1, B):
            rs, cs = slice(r, r + B), slice(c, c + B)
            m = valid[rs, cs]
            if float(m.mean()) < dg.BLOCK_MIN_VALID:
                continue
            v = score_image[rs, cs][m]
            cand.append((float(v.std() / max(v.mean(), dg.EPS)), rs, cs, m))
    cand.sort(key=lambda x: (x[0], x[1].start, x[2].start))
    n = min(max(dg.BLOCK_MIN_COUNT, int(math.ceil(fraction * len(cand)))), len(cand))
    return [(rs, cs, m) for _, rs, cs, m in cand[:n]]


def first_order(noisy, ratio, blocks):
    ne = np.asarray([dg._enl(noisy[rs, cs][m]) for rs, cs, m in blocks])
    re_ = np.asarray([dg._enl(ratio[rs, cs][m]) for rs, cs, m in blocks])
    rm = np.asarray([dg._nanmean(ratio[rs, cs][m]) for rs, cs, m in blocks])
    enl_term = dg._nanmean(np.abs((re_ - ne) / np.maximum(ne, dg.EPS)))
    mean_term = dg._nanmean(np.abs(rm - 1.0))
    return 50.0 * (enl_term + mean_term), 50.0 * enl_term, 50.0 * mean_term


def measure(noisy, est, clean, seed_key):
    noisy = np.clip(np.asarray(noisy, np.float64), dg.EPS, np.inf)
    est = np.clip(np.asarray(est, np.float64), dg.EPS, np.inf)
    valid = np.ones(noisy.shape, bool)
    ratio = noisy / est
    rec = {"true_log_rmse": metrics.true_log_rmse(est.astype(np.float32), clean)}
    sel = dg.homogeneous_blocks(noisy, valid)
    fo, fo_enl, fo_mean = first_order(noisy, ratio, sel)
    rec.update(m_first_order=fo, m_first_enl=fo_enl, m_first_mean=fo_mean,
               m_structure=dg._structure_term(ratio, valid, seed_key))
    rec["m_index"] = 0.5 * (rec["m_first_order"] + rec["m_structure"])
    rec["clean_cv_in_selected"] = float(np.mean([clean[rs, cs][m].std() / max(clean[rs, cs][m].mean(), dg.EPS) for rs, cs, m in sel]))
    for f in FRACTIONS:
        rec[f"m_first_f{int(f * 100):03d}"] = first_order(noisy, ratio, blocks_by(noisy, valid, f))[0]
    oracle = blocks_by(np.asarray(clean, np.float64) + dg.EPS, valid, dg.BLOCK_FRACTION)
    rec["m_first_oracle"] = first_order(noisy, ratio, oracle)[0]
    rec["m_index_oracle"] = 0.5 * (rec["m_first_oracle"] + rec["m_structure"])
    return rec


def init(scenes):
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"
    if scenes:
        G.update(meta=json.loads((Path(scenes) / "scenes.json").read_text())["scenes"],
                 B=np.load(Path(scenes) / "B.npy", mmap_mode="r"))


def run_tile(args):
    response, scene = args
    m = G["meta"][scene]
    resp = sim.response(response, m["subswath"])
    obs = sim.simulate({"background": np.array(G["B"][scene], np.float32), "points": []}, resp,
                       np.random.default_rng([20261007, scene]))
    noisy, clean = obs.intensity, obs.clean
    coords = [tuple(int(v) for v in c) for c in m["coords"]]
    rows = []
    for name, family, level, est in degradations.controlled_degradations(clean.astype(np.float32), noisy, coords):
        rows.append({"scene_type": "urban", "response": response, "unit": scene, "family": family, "degradation": name,
                     **measure(noisy, est, clean, degradations.seed_key(name, scene))})
    return rows


def run_phantoms(response, n):
    rows = []
    for item in sim.phantom_set("dv", response, n=n):
        clean, noisy, coords = item["clean"].astype(np.float32), item["noisy"], item["coords"]
        for name, family, level, est in degradations.controlled_degradations(clean, noisy, coords):
            rows.append({"scene_type": "phantom", "response": response, "unit": item["index"], "family": family,
                         "degradation": name, **measure(noisy, est, clean, degradations.seed_key(name, item["index"]))})
    return rows


def block_statistics(scenes, n=37):
    """Observation ENL and reference coefficient of variation in the selected blocks (rev14_block_statistics.csv)."""
    init(scenes)
    tiles = sorted(pd.read_csv(ROOT / "results" / "semi_synthetic" / "controlled_degradations.csv.gz").phantom.unique())
    out = []
    for r in ("s1", "sarlo"):
        items = []
        for t in tiles:
            m = G["meta"][int(t)]
            o = sim.simulate({"background": np.array(G["B"][int(t)], np.float32), "points": []},
                             sim.response(r, m["subswath"]), np.random.default_rng([20261007, int(t)]))
            items.append(("urban", o.intensity.astype(float), o.clean.astype(float)))
        items += [("phantom", it["noisy"].astype(float), it["clean"].astype(float)) for it in sim.phantom_set("dv", r, n=n)]
        for k, noisy, clean in items:
            v = np.ones(noisy.shape, bool)
            sel, orc = dg.homogeneous_blocks(noisy, v), blocks_by(clean + dg.EPS, v, dg.BLOCK_FRACTION)
            ratio = noisy / np.clip(clean, dg.EPS, None)
            cv = lambda bl: float(np.mean([clean[rs, cs][m].std() / clean[rs, cs][m].mean() for rs, cs, m in bl]))
            out.append(dict(kind=k, resp=r, obs_enl=float(np.mean([dg._enl(noisy[rs, cs][m]) for rs, cs, m in sel])),
                            exact_ratio_enl=float(np.mean([dg._enl(ratio[rs, cs][m]) for rs, cs, m in sel])),
                            clean_cv_sel=cv(sel), clean_cv_oracle=cv(orc)))
    pd.DataFrame(out).to_csv(OUT / "rev14_block_statistics.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--phantoms", type=int, default=80)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    import pandas as _pd
    tiles = sorted(_pd.read_csv(ROOT / "results" / "semi_synthetic" / "controlled_degradations.csv.gz").phantom.unique())
    tasks = [(r, int(t)) for r in ("s1", "umbra", "sarlo") for t in tiles]
    with get_context("fork").Pool(a.workers, initializer=init, initargs=(a.scenes,)) as pool:
        rows = [r for rr in pool.imap(run_tile, tasks) for r in rr]
        rows += [r for rr in pool.starmap(run_phantoms, [(r, a.phantoms) for r in ("s1", "umbra", "sarlo")]) for r in rr]
    pd.DataFrame(rows).to_csv(OUT / "rev14_mindex_components.csv.gz", index=False, float_format="%.8g")
    print(len(rows), "rows")
    block_statistics(a.scenes)


if __name__ == "__main__":
    main()
