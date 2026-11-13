"""Revision columns for the despeckling methods on the SARLO-response simulation (R3-4, R3-5, R3-6, R1-2, R4).

    python revision/rev10_sim_methods.py --package <folder holding SAR/> --sim <a160_extension/sarlo/sim>
           --sar2sar <a160_extension/sarlo/sar2sar> --calibration <a160_extension/sarlo/calibration.json>
           [--workers 2] [--start 0 --stop 400]

The methods are produced by the authors' own adapters (SAR/kodlar/a159_methods.py: cpu_estimates and
GpuMethods.estimates, CPU device) on the stored 400 evaluation phantoms; SAR2SAR is taken from its stored outputs,
as in run_a160_arm.py. Each estimate is then measured with sardiag (released columns) and sardiag.revision (new
columns). The released columns are later compared with a160_extension/sarlo/all_patch_metrics.csv, so every row
carries its own reproduction check. Results are written in chunks to revision/results/sim_methods/ and a rerun
continues where it stopped.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
OUT = HERE / "results" / "sim_methods"
CPU_GROUPS = ("identity", "classical7", "lee_sigma", "mulog")
GPU_GROUPS = ("merlin_s1tops", "merlin_spotlight", "cl_sar", "unet_s1", "merlin_s1", "anchor", "sds", "pcgt")
G: dict = {}


def init(package: str, sim: str, sar2sar: str, calibration: str) -> None:
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = "1"
    kod = str(Path(package) / "SAR" / "kodlar")
    os.environ["SAR_LOCAL_ROOT"] = str(Path(package) / "SAR")
    os.chdir(kod)
    sys.path.insert(0, kod)
    sys.path.insert(0, str(HERE))
    import np2_compat  # noqa: F401
    import winpath_shim  # noqa: F401
    import torch
    torch.set_num_threads(1)
    import a159_methods as M
    M.ctx()
    cal = json.loads(Path(calibration).read_text(encoding="utf-8"))
    G.update(M=M, gpu=M.GpuMethods("cpu"), ss=M.simset(sim, "eval"),
             scales={"merlin_s1tops": cal["merlin_s1tops"]["scale"], "merlin_spotlight": cal["merlin_spotlight"]["scale"]},
             s2s={c: np.load(Path(sar2sar) / f"{c}_out.npz") for c in ("direct", "sub2")})


def detection(image, coords, pfa=1e-3) -> float:
    from sardiag import revision as rv
    target, background = rv.detection_scores(image, coords)
    return float(np.mean(target > np.quantile(background, 1.0 - pfa)))


def run_item(i: int):
    from sardiag import evaluation, revision as rv
    M, ss = G["M"], G["ss"]
    item = ss.get(i)
    noisy, clean, mask = item["noisy"], item["clean"], item["mask"]
    coords = [tuple(int(v) for v in c) for c in item["coords"]]
    pairs, timing = [], {}
    for g in CPU_GROUPS:
        t = time.time()
        pairs += M.cpu_estimates(g, item)
        timing[g] = time.time() - t
    noisy64 = noisy.astype(np.float64)
    for c in ("direct", "sub2"):
        assert str(G["s2s"][c]["ids"][i]) == item["pid"]
        est = M.ctx()["s3"].match_mean(np.square(G["s2s"][c][f"{c}_amplitude"][i].astype(np.float64)), noisy64)
        pairs.append((f"sar2sar_official_{c}", est, None))
    for g in GPU_GROUPS:
        t = time.time()
        pairs += G["gpu"].estimates(g, item, G["scales"])
        timing[g] = time.time() - t
    contrast = rv.target_contrast_db(clean, coords)
    pd_obs = detection(noisy, coords)
    rows, targets = [], []
    for name, est, _ in pairs:
        est = np.clip(np.asarray(est, dtype=np.float32), 1e-8, None)
        base = {"phantom": item["pid"], "phantom_class": item["klass"], "method": name}
        rec = evaluation.evaluate(noisy, est, clean, coords, mask, seed_key=f"x|{item['pid']}_{name}")
        rec.update(rv.evaluate_revision(noisy, est, clean, coords, mask))
        rec["det_pd"], rec["det_pd_observation"] = detection(est, coords), pd_obs
        rows.append({**base, **rec})
        for k, v in enumerate(rv.target_log_ratios(est, clean, coords)):
            targets.append({**base, "kind": "target", "index": k, "contrast_db": float(contrast[k]), "truth": float(v)})
        for k, t in enumerate(rv.proxy_target_table(noisy, est, clean, coords)):
            targets.append({**base, "kind": "selected", "index": k, **{c: t[c] for c in
                            ("is_target", "contrast_db", "obs_contrast_db", "proxy", "truth", "select")}})
    return i, rows, targets, timing


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--package", required=True)
    ap.add_argument("--sim", required=True)
    ap.add_argument("--sar2sar", required=True)
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--stop", type=int, default=400)
    ap.add_argument("--chunk", type=int, default=20)
    ap.add_argument("--step", type=int, default=1, help="use every step-th phantom (2 = 200 of the 400, both classes)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    items = list(range(args.start, args.stop, args.step))
    chunks = [items[k:k + args.chunk] for k in range(0, len(items), args.chunk)]
    todo = [c for c in chunks if not (OUT / f"rows_{c[0]:04d}.csv.gz").is_file()]
    print(f"{len(todo)} chunks to run", flush=True)
    with get_context("fork").Pool(args.workers, initializer=init,
                                  initargs=(args.package, args.sim, args.sar2sar, args.calibration)) as pool:
        for c in todo:
            t0 = time.time()
            res = pool.map(run_item, c, chunksize=1)
            rows = [r for _, rr, _, _ in res for r in rr]
            targets = [t for _, _, tt, _ in res for t in tt]
            pd.DataFrame(targets).to_csv(OUT / f"targets_{c[0]:04d}.csv.gz", index=False, float_format="%.7g")
            pd.DataFrame(rows).to_csv(OUT / f"rows_{c[0]:04d}.csv.gz", index=False, float_format="%.10g")
            timing = pd.DataFrame([t for *_, t in res]).mean().round(2).to_dict()
            print(f"chunk {c[0]}: {len(res)} phantoms in {time.time() - t0:.0f} s; mean s per group {timing}", flush=True)


if __name__ == "__main__":
    main()
