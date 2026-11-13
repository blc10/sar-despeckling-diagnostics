"""R4-2: preservation of strong scatterers under CFAR detection, on real SARLO-80 data and, for validation, in simulation.

    python revision/rev16_detection.py real --package <folder holding SAR/> --patches <dir with real_0..4.npy>
           --sar2sar <sar2sar_real_sub.npz> --protocol <a161_sarlo_real/PROTOKOL.json> [--workers 2]
    python revision/rev16_detection.py sim  --package <...> --sim <a160_extension/sarlo/sim>
           --sar2sar <a160_extension/sarlo/sar2sar> --calibration <a160_extension/sarlo/calibration.json> [--step 4]

Real data have no annotated targets. Pseudo-targets are the strong scatterers selected in the observation by the rule
of the observation-referenced peak error (sardiag.revision.strong_scatterers) whose observed contrast over the 5-11 pixel
ring is at least 14 dB (12 dB as sensitivity). For every method, the cell-averaging CFAR statistic of the estimate
(pixel over ring mean; the detector of the simulated experiment) is recorded at each selected scatterer (3 x 3 maximum),
together with quantiles of the statistic over all pixels farther than 5 pixels from every selected scatterer. A
pseudo-target counts as detected when its statistic exceeds the background quantile for the false-alarm probability.
Because the pseudo-targets are selected by their contrast in the observation, they are detected in the observation by
construction; the measure is the share that a method keeps.

The simulation branch applies the same rule on the SARLO-80 phantoms and additionally records the detection of the
known targets, so that the pseudo-target measure can be validated against the true one.
Methods and settings are those of rev11 (real) and rev10 (simulation).
"""
from __future__ import annotations

import argparse
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
QUANTILES = (0.99, 0.999, 0.9999)
GUARD = 5


def scatterer_stats(noisy, estimates, coords_true=None):
    """Per selected scatterer and method: observed contrast, CFAR statistic; per method: background quantiles."""
    from sardiag import revision as rv
    noisy64 = np.clip(np.asarray(noisy, np.float64), 1e-8, None)
    sel = rv.strong_scatterers(noisy64)
    if not sel:
        return [], []
    contrast = rv.target_contrast_db(noisy64, sel)
    is_target = None
    if coords_true is not None:
        is_target = [min((np.hypot(r - py, c - px) for py, px in coords_true), default=np.inf) <= 1.5 for r, c in sel]
    scat, back = [], []
    for name, est in estimates:
        stat = rv.cfar_statistic(np.clip(np.asarray(est, np.float64), 1e-8, None))
        keep = np.ones(stat.shape, bool)
        for r, c in sel:
            keep[max(r - GUARD, 0):r + GUARD + 1, max(c - GUARD, 0):c + GUARD + 1] = False
        q = np.quantile(stat[keep], QUANTILES)
        back.append({"method": name, **{f"q{qq}": float(v) for qq, v in zip(QUANTILES, q)}})
        for k, (r, c) in enumerate(sel):
            row = {"method": name, "k": k, "obs_contrast_db": float(contrast[k]), "stat": rv._peak3(stat, r, c)}
            if is_target is not None:
                row["is_target"] = bool(is_target[k])
            scat.append(row)
    return scat, back


# ------------------------------------------------------------------------------------------------ real data
def run_real(j):
    import rev11_real_methods as R
    G, M = R.G, R.G["M"]
    i = int(G["s2s"]["index"][j]); pid = str(G["s2s"]["ids"][j])
    z = R.patch(i)
    noisy = (np.abs(z) ** 2).astype(np.float32)
    item = {"z": z, "noisy": noisy, "clean": noisy, "os_az": G["os"][0], "os_rg": G["os"][1], "pid": pid}
    pairs = []
    for g in R.CPU_GROUPS:
        pairs += M.cpu_estimates(g, item)
    for c in ("direct", "sub2"):
        est = M.ctx()["s3"].match_mean(np.square(G["s2s"][f"{c}_amplitude"][j].astype(np.float64)), noisy.astype(np.float64))
        pairs.append((f"sar2sar_official_{c}", est, None))
    for g in R.GPU_GROUPS:
        pairs += G["gpu"].estimates(g, item, G["scales"])
    scat, back = scatterer_stats(noisy, [(n, e) for n, e, _ in pairs])
    return [{"patch": pid, **s} for s in scat], [{"patch": pid, **b} for b in back]


def init_real(*a):
    import rev11_real_methods as R
    R.init(*a)


# ------------------------------------------------------------------------------------------------ simulation
def run_sim(i):
    import rev10_sim_methods as S
    G, M = S.G, S.G["M"]
    item = G["ss"].get(i)
    noisy = item["noisy"]
    coords = [tuple(int(v) for v in c) for c in item["coords"]]
    pairs = []
    for g in S.CPU_GROUPS:
        pairs += M.cpu_estimates(g, item)
    for c in ("direct", "sub2"):
        est = M.ctx()["s3"].match_mean(np.square(G["s2s"][c][f"{c}_amplitude"][i].astype(np.float64)), noisy.astype(np.float64))
        pairs.append((f"sar2sar_official_{c}", est, None))
    for g in S.GPU_GROUPS:
        pairs += G["gpu"].estimates(g, item, G["scales"])
    pairs = [(n, np.clip(np.asarray(e, np.float32), 1e-8, None)) for n, e, _ in pairs]
    scat, back = scatterer_stats(noisy, pairs, coords)
    for b, (n, e) in zip(back, pairs):
        b["det_pd_true"] = S.detection(e, coords)
    return [{"phantom": item["pid"], **s} for s in scat], [{"phantom": item["pid"], **b} for b in back]


def init_sim(*a):
    import rev10_sim_methods as S
    S.init(*a)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("branch", choices=("real", "sim"))
    for a in ("--package", "--sar2sar"):
        ap.add_argument(a, required=True)
    ap.add_argument("--patches"); ap.add_argument("--protocol"); ap.add_argument("--sim"); ap.add_argument("--calibration")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--chunk", type=int, default=12)
    ap.add_argument("--step", type=int, default=4)
    ap.add_argument("--stop", type=int, default=None)
    a = ap.parse_args()
    out = HERE / "results" / f"detection_{a.branch}"
    out.mkdir(parents=True, exist_ok=True)
    if a.branch == "real":
        items = np.random.default_rng(20261112).permutation(284).tolist()
        init, initargs, fn = init_real, (a.package, a.patches, a.sar2sar, a.protocol), run_real
    else:
        items = list(range(0, 400, a.step))
        init, initargs, fn = init_sim, (a.package, a.sim, a.sar2sar, a.calibration), run_sim
    if a.stop:
        items = items[:a.stop]
    chunks = [items[k:k + a.chunk] for k in range(0, len(items), a.chunk)]
    todo = [(n, c) for n, c in enumerate(chunks) if not (out / f"back_{n:04d}.csv.gz").is_file()]
    print(f"{len(todo)} chunks to run", flush=True)
    with get_context("fork").Pool(a.workers, initializer=init, initargs=initargs) as pool:
        for n, c in todo:
            t0 = time.time()
            res = pool.map(fn, c, chunksize=1)
            pd.DataFrame([s for ss, _ in res for s in ss]).to_csv(out / f"scat_{n:04d}.csv.gz", index=False, float_format="%.7g")
            pd.DataFrame([b for _, bb in res for b in bb]).to_csv(out / f"back_{n:04d}.csv.gz", index=False, float_format="%.7g")
            print(f"chunk {n}: {len(c)} items in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
