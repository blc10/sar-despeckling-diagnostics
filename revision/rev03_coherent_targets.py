"""R3-2 and R1-3: speckle statistics at coherent targets and the footprint of point targets in the diagnostics.

    python revision/rev03_coherent_targets.py [--phantoms 80]

Part A (R3-2). For the exact reference X-hat = X, the ratio Z / X is unit-mean but not Gamma distributed where a
deterministic target adds a coherent component (Rician amplitude, non-central intensity). Each degradation phantom is
simulated twice with the same speckle draw, with and without its point targets, and the diagnostics of the exact
reference are compared; the Gamma JSD is also evaluated after masking the coherent footprints.
Part B (R1-3). Peak attenuation (factor 0.1) is applied to phantoms whose number of point targets is raised from the
study's 4-8 to 20, 80 and 320, and the AUC of each diagnostic for separating attenuated from exact outputs is reported
against the fraction of pixels in target footprints.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, auc, boot, ci  # noqa: E402
from sardiag import degradations, diagnostics as dg, evaluation, simulation as sim  # noqa: E402

SEED = 20261111
COHERENT_SHARE = 0.1     # footprint: coherent power above 10% of the reference
DIAG = ("r", "gamma_jsd", "m_index", "tdm_deviation", "tdm")


def phantoms(n: int):
    rng = np.random.default_rng(sim.SEEDS["dv"])
    for index in range(n):
        subswath = str(rng.choice(sorted(sim.SENTINEL1_IW)))
        phantom = sim.make_phantom(rng, sim.SIZE, "structural")
        rng.standard_normal((sim.SIZE, sim.SIZE, 2))            # keep the stream aligned with phantom_set
        mask = sim.homogeneous_mask(phantom)
        if mask.sum() >= sim.MIN_MASK_PIXELS and phantom["points"]:
            yield index, subswath, phantom, mask


def observe(phantom: dict, resp: dict, index: int):
    return sim.simulate(phantom, resp, np.random.default_rng([SEED, index]))


def part_a(n: int) -> tuple[pd.DataFrame, dict]:
    rows = []
    for sensor in ("s1", "umbra", "sarlo"):
        for index, subswath, phantom, mask in phantoms(n):
            resp = sim.response(sensor, subswath)
            for label, ph in (("with_targets", phantom), ("without_targets", {**phantom, "points": []})):
                obs = observe(ph, resp, index)
                noisy, clean = obs.intensity, obs.clean
                coherent = (np.abs(obs.point_component) ** 2 / obs.scale ** 2) / np.maximum(clean, 1e-12)
                foot = coherent > COHERENT_SHARE
                ratio = dg.ratio_image(noisy, clean)
                rec = evaluation.evaluate(noisy, clean, clean, obs.coords or [(0, 0)], mask, seed_key=f"x|rev03_{index}")
                masked = ratio.copy()
                masked[foot] = np.nan
                k = coherent[foot] / np.maximum(1.0 - coherent[foot], 1e-6)       # coherent-to-diffuse power ratio
                rows.append({"response": sensor, "phantom": index, "condition": label, **{d: rec[d] for d in DIAG},
                             "gamma_jsd_masked": dg.gamma_jsd(masked), "footprint_fraction": float(foot.mean()),
                             "ratio_var_footprint": float(ratio[foot].var()) if foot.any() else np.nan,
                             "ratio_var_theory": float(np.mean((1 + 2 * k) / (1 + k) ** 2)) if foot.any() else np.nan,
                             "ratio_mean_footprint": float(ratio[foot].mean()) if foot.any() else np.nan,
                             "ratio_var_elsewhere": float(ratio[~foot].var()), "ratio_mean_elsewhere": float(ratio[~foot].mean())})
    frame = pd.DataFrame(rows)
    summary = {}
    for sensor, block in frame.groupby("response"):
        wide = block.pivot(index="phantom", columns="condition")
        s = {"footprint_fraction": float(wide["footprint_fraction"]["with_targets"].mean()),
             "ratio_var_footprint": float(wide["ratio_var_footprint"]["with_targets"].mean()),
             "ratio_var_theory": float(wide["ratio_var_theory"]["with_targets"].mean()),
             "ratio_mean_footprint": float(wide["ratio_mean_footprint"]["with_targets"].mean()),
             "ratio_var_elsewhere": float(wide["ratio_var_elsewhere"]["with_targets"].mean())}
        for d in DIAG + ("gamma_jsd_masked",):
            a, b = wide[d]["with_targets"].to_numpy(), wide[d]["without_targets"].to_numpy()
            diff = a - b
            rng = np.random.default_rng(SEED)
            bs = [np.mean(diff[rng.integers(0, len(diff), len(diff))]) for _ in range(2000)]
            s[d] = {"with_targets": float(a.mean()), "without_targets": float(b.mean()), "mean_difference": float(diff.mean()),
                    "ci": ci(bs), "relative_difference": float(diff.mean() / max(abs(b.mean()), 1e-12))}
        summary[sensor] = s
    return frame, summary


def part_b(n: int, counts=(None, 20, 80, 320)) -> tuple[pd.DataFrame, dict]:
    rows = []
    for sensor in ("s1", "sarlo"):
        for index, subswath, phantom, mask in phantoms(n):
            box = phantom["homog_mask"]
            for count in counts:
                ph = phantom
                if count is not None:
                    rng = np.random.default_rng([SEED, 7, index, count])
                    points = list(phantom["points"])
                    while len(points) < count:
                        py, px = int(rng.integers(6, sim.SIZE - 6)), int(rng.integers(6, sim.SIZE - 6))
                        if not box[max(py - 4, 0):py + 5, max(px - 4, 0):px + 5].any():
                            points.append((py, px, float(10.0 ** rng.uniform(1.0, 2.0)), float(rng.uniform(-math.pi, math.pi))))
                    ph = {**phantom, "points": points}
                obs = observe(ph, sim.response(sensor, subswath), index)
                noisy, clean = obs.intensity, obs.clean.astype(np.float32)
                outputs = [o for o in degradations.controlled_degradations(clean, noisy, obs.coords)
                           if o[0] in ("peak_keep_1.0", "peak_keep_0.1")]
                for name, _, level, estimate in outputs:
                    rec = evaluation.evaluate(noisy, estimate, clean, obs.coords, mask, seed_key=f"x|rev03b_{index}_{name}")
                    rows.append({"response": sensor, "phantom": index, "targets": len(obs.coords), "requested": count or 0,
                                 "footprint_fraction": 9.0 * len(obs.coords) / clean.size, "attenuated": level < 1.0,
                                 **{d: rec[d] for d in DIAG}, "true_log_rmse": rec["true_log_rmse"], "peak_error": rec["peak_error"]})
    frame = pd.DataFrame(rows)
    frame["tdm_negated"] = -frame["tdm"]
    summary = {}
    for (sensor, requested), block in frame.groupby(["response", "requested"]):
        label = block.attenuated.to_numpy()
        entry = {"targets_mean": float(block.targets.mean()), "footprint_fraction": float(block.footprint_fraction.mean()),
                 "true_log_rmse_attenuated": float(block[block.attenuated].true_log_rmse.mean())}
        for d in ("r", "gamma_jsd", "m_index", "tdm_deviation", "tdm_negated"):
            s = block[d].to_numpy(float)
            b = boot(block.phantom.to_numpy(), 1000, lambda i: auc(s[i], label[i]))
            entry[d] = {"auc": auc(s, label), "ci": ci(b[:, 0])}
        summary[f"{sensor}|{requested or 'study'}"] = entry
    return frame, summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phantoms", type=int, default=80)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    fa, sa = part_a(args.phantoms)
    fa.to_csv(OUT / "rev03_exact_reference.csv.gz", index=False, float_format="%.8g")
    for sensor, s in sa.items():
        print(f"[A] {sensor}: footprint {100 * s['footprint_fraction']:.2f}% of pixels; var(Z/X) footprint {s['ratio_var_footprint']:.3f}"
              f" (theory {s['ratio_var_theory']:.3f}) vs elsewhere {s['ratio_var_elsewhere']:.3f}")
        for d in DIAG + ("gamma_jsd_masked",):
            print(f"     {d:<17} with {s[d]['with_targets']:.5f}  without {s[d]['without_targets']:.5f}  diff {s[d]['mean_difference']:+.5f} {np.round(s[d]['ci'], 5)}")
    fb, sb = part_b(min(args.phantoms, 40))
    fb.to_csv(OUT / "rev03_target_density.csv.gz", index=False, float_format="%.8g")
    for key, e in sb.items():
        print(f"[B] {key:<12} targets {e['targets_mean']:.0f} ({100 * e['footprint_fraction']:.2f}% px)  "
              + "  ".join(f"{d}={e[d]['auc']:.2f}" for d in ("r", "gamma_jsd", "m_index", "tdm_deviation")))
    (OUT / "rev03_coherent_targets.json").write_text(json.dumps({"exact_reference": sa, "target_density": sb}, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
