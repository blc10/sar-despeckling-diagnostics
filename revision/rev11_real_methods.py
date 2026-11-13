"""R3-5, R1-2 on real data: the observation-referenced peak proxy with and without a contrast condition.

    python revision/rev11_real_methods.py --package <folder holding SAR/> --patches <dir with real_0..4.npy>
           --sar2sar <sar2sar_real_sub.npz> --protocol <a161_sarlo_real/PROTOKOL.json> [--workers 2]

Subsample: the first primary patch of each of the 284 primary records (revision/real_subsample_index.json), in the
order of a161_sarlo_real/patch_manifest.csv. All 24 methods are rerun with the authors' adapters and the settings of
the sealed protocol (scales, PCGT oversampling, SAR2SAR from its stored outputs). Per patch and method the script
records the released proxy (recomputed; it is checked against all_patch_rf.csv) and, per selected scatterer, the
observed contrast and the proxy term, so that any contrast condition can be applied afterwards. This is a post hoc
analysis; the sealed test and its results are unchanged. Patches are processed in a fixed random order (seed
20261112), so the chunks finished at any time form a random subsample of the 284.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
OUT = HERE / "results" / "real_methods"
CPU_GROUPS = ("identity", "classical7", "lee_sigma", "mulog")
GPU_GROUPS = ("merlin_s1tops", "merlin_spotlight", "cl_sar", "unet_s1", "merlin_s1", "anchor", "sds", "pcgt")
G: dict = {}


def init(package, patches, sar2sar, protocol) -> None:
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
    parts = sorted(Path(patches).glob("real_*.npy"), key=lambda p: int(p.stem.split("_")[1]))
    proto = json.loads(Path(protocol).read_text(encoding="utf-8"))["methods"]
    G.update(M=M, gpu=M.GpuMethods("cpu"), parts=[np.load(p, mmap_mode="r") for p in parts],
             scales=proto["scales"], os=proto["pcgt_oversampling"], s2s=np.load(sar2sar))
    G["sizes"] = np.cumsum([0] + [p.shape[0] for p in G["parts"]])


def patch(i: int) -> np.ndarray:
    k = int(np.searchsorted(G["sizes"], i, side="right") - 1)
    return np.array(G["parts"][k][i - G["sizes"][k]])


def scatterer_rows(noisy, est):
    from sardiag import revision as rv
    noisy64 = np.clip(noisy.astype(np.float64), 1e-8, None)
    est64 = np.clip(np.asarray(est, dtype=np.float64), 1e-8, None)
    sel = rv.strong_scatterers(noisy64)
    contrast = rv.target_contrast_db(noisy64, sel) if sel else []
    return [{"k": k, "row": r, "col": c, "obs_contrast_db": float(cdb),
             "proxy": math.log(rv._peak3(est64, r, c) / rv._peak3(noisy64, r, c))}
            for k, ((r, c), cdb) in enumerate(zip(sel, contrast))]


def run_item(j: int):
    from sardiag import real_data
    M = G["M"]
    i = int(G["s2s"]["index"][j])
    pid = str(G["s2s"]["ids"][j])
    z = patch(i)
    noisy = (np.abs(z) ** 2).astype(np.float32)
    item = {"z": z, "noisy": noisy, "clean": noisy, "os_az": G["os"][0], "os_rg": G["os"][1], "pid": pid}
    pairs, timing = [], {}
    for g in CPU_GROUPS:
        t = time.time()
        pairs += M.cpu_estimates(g, item)
        timing[g] = time.time() - t
    for c in ("direct", "sub2"):
        est = M.ctx()["s3"].match_mean(np.square(G["s2s"][f"{c}_amplitude"][j].astype(np.float64)), noisy.astype(np.float64))
        pairs.append((f"sar2sar_official_{c}", est, None))
    for g in GPU_GROUPS:
        t = time.time()
        pairs += G["gpu"].estimates(g, item, G["scales"])
        timing[g] = time.time() - t
    rows, scat = [], []
    for name, est, _ in pairs:
        released, count = real_data.strong_scatterer_peak_error(noisy.astype(np.float64), np.asarray(est, dtype=np.float64))
        rows.append({"patch": pid, "index": i, "method": name, "proxy_released": released, "scatterers": count})
        scat += [{"patch": pid, "method": name, **s} for s in scatterer_rows(noisy, est)]
    return j, rows, scat, timing


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    for a in ("--package", "--patches", "--sar2sar", "--protocol"):
        ap.add_argument(a, required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--chunk", type=int, default=12)
    ap.add_argument("--stop", type=int, default=284)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    order = np.random.default_rng(20261112).permutation(284)[:args.stop].tolist()   # any prefix is a random subsample
    chunks = [order[k:k + args.chunk] for k in range(0, len(order), args.chunk)]
    todo = [(n, c) for n, c in enumerate(chunks) if not (OUT / f"rows_{n:04d}.csv.gz").is_file()]
    print(f"{len(todo)} chunks to run", flush=True)
    with get_context("fork").Pool(args.workers, initializer=init, initargs=(args.package, args.patches, args.sar2sar, args.protocol)) as pool:
        for n, c in todo:
            t0 = time.time()
            res = pool.map(run_item, c, chunksize=1)
            pd.DataFrame([s for _, _, ss, _ in res for s in ss]).to_csv(OUT / f"scatterers_{n:04d}.csv.gz", index=False, float_format="%.7g")
            pd.DataFrame([r for _, rr, _, _ in res for r in rr]).to_csv(OUT / f"rows_{n:04d}.csv.gz", index=False, float_format="%.10g")
            print(f"chunk {n}: {len(res)} patches in {time.time() - t0:.0f} s; "
                  f"{pd.DataFrame([t for *_, t in res]).mean().round(1).to_dict()}", flush=True)


if __name__ == "__main__":
    main()
