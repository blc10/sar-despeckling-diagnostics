"""R1-1: sensitivity of the semi-synthetic results (Table 9) to the MuLoG+BM3D reflectivity proxy.

    python revision/rev12_reflectivity_map.py --package <folder holding SAR/> --scenes <a160_extension/semi_scenes>
           [--tmean <B_tmean.npy>] [--workers 2]   (without --tmean only the pipeline check with the original map runs)

Second reflectivity map, without any despeckling algorithm: the arithmetic temporal mean of the same 10-12 dates of
the Sentinel-1 panel (a152_stage3_panel/evalset, amplitude_dn squared, same 256 x 256 tile positions), scaled to unit
mean. This is exactly the multi-date super image that a160_scenes.py passed to MuLoG+BM3D. It keeps residual speckle
(about 10-12 looks), so the two maps bracket the reference from opposite sides: MuLoG+BM3D carries algorithm bias and
smoothing, the temporal mean carries residual speckle texture and no algorithm bias.

The degradation tiles are simulated as in a160_scenes.semi_stream (sardiag.simulation.simulate, speckle seed
[20261007, tile], same responses and subswaths, same homogeneous masks and target coordinates) once with each map,
then degraded and measured with sardiag. The run with the original map is compared with the released
results/semi_synthetic table, which checks the pipeline; the run with the temporal mean gives the sensitivity.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
OUT = HERE / "results"
G: dict = {}


def init(scenes, tmean) -> None:
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = "1"
    maps = {"mulog_bm3d": np.load(Path(scenes) / "B.npy", mmap_mode="r")}
    if tmean:
        maps["temporal_mean"] = np.load(tmean, mmap_mode="r")
    G.update(meta=json.loads((Path(scenes) / "scenes.json").read_text(encoding="utf-8"))["scenes"], maps=maps,
             homog=np.load(Path(scenes) / "homog.npy", mmap_mode="r"))


def run_tile(args):
    from sardiag import degradations, evaluation, simulation as sim
    mapname, response, scene = args
    m = G["meta"][scene]
    resp = sim.response(response, m["subswath"])
    phantom = {"background": np.array(G["maps"][mapname][scene], dtype=np.float32), "points": []}
    obs = sim.simulate(phantom, resp, np.random.default_rng([20261007, scene]))
    noisy, clean = obs.intensity, obs.clean
    coords = [tuple(int(v) for v in c) for c in m["coords"]]
    mask = np.array(G["homog"][scene])
    rows = []
    for name, family, level, est in degradations.controlled_degradations(clean.astype(np.float32), noisy, coords):
        rec = evaluation.evaluate(noisy, est, clean, coords, mask, seed_key=degradations.seed_key(name, scene))
        rows.append({"map": mapname, "response": response, "phantom": scene, "site": m["site"], "degradation": name,
                     "family": family, "level": level, **rec})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--tmean")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    released = pd.read_csv(HERE.parent / "results" / "semi_synthetic" / "controlled_degradations.csv.gz")
    tiles = sorted(released.phantom.unique())
    tasks = [(mp, r, int(t)) for mp in (("mulog_bm3d", "temporal_mean") if args.tmean else ("mulog_bm3d",)) for r in ("s1", "umbra", "sarlo") for t in tiles]
    with get_context("fork").Pool(args.workers, initializer=init, initargs=(args.scenes, args.tmean)) as pool:
        rows = [r for rr in pool.imap(run_tile, tasks, chunksize=2) for r in rr]
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / ("rev12_degradations_by_map.csv.gz" if args.tmean else "rev12_pipeline_check.csv.gz"),
                 index=False, float_format="%.10g")
    print(len(frame), "rows")


if __name__ == "__main__":
    main()
