"""Check the simulated system responses and the temporal-median bias.

    python scripts/validate_simulator.py [--phantoms 20]

Prints the peak sidelobe ratio, the -3 dB impulse response width and the lag-1 intensity correlation of each response,
compares the theoretical lag-1 correlation with the value measured on simulated speckle, compares the theoretical
bias of the 11-date temporal median with the bias measured on simulated stacks and reports the error floor of the
temporal-median reference.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sardiag import simulation as sim  # noqa: E402

RESPONSES = {**{f"Sentinel-1 {k}": v for k, v in sim.SENTINEL1_IW.items()},
             "Umbra spotlight": sim.UMBRA_SPOTLIGHT, "SARLO-80": sim.SARLO80}


def measured_lag1(resp: dict, size: int = 512, seed: int = 0) -> tuple[float, float]:
    """Lag-1 intensity correlation of simulated speckle on a flat scene, azimuth and range."""
    rng = np.random.default_rng(seed)
    s = (rng.standard_normal((size, size)) + 1j * rng.standard_normal((size, size))) / np.sqrt(2.0)
    z = np.fft.ifft2(np.fft.fft2(s) * sim.transfer(size, size, resp))
    intensity = np.abs(z) ** 2
    return sim.lag_correlation(intensity, 1, 0), sim.lag_correlation(intensity, 0, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--phantoms", type=int, default=20, help="evaluation phantoms per class for the median bias")
    args = parser.parse_args()

    print(f"{'response':<20}{'axis':<9}{'OS':>7}{'a':>6}{'PSLR dB':>10}{'IRW':>7}{'rho2(1)':>10}{'measured':>10}")
    for name, resp in RESPONSES.items():
        measured = measured_lag1(resp)
        for k, axis in enumerate(("azimuth", "range")):
            os_, a = resp["os_az" if k == 0 else "os_rg"], resp["a_az" if k == 0 else "a_rg"]
            q = sim.pslr_irw(os_, a)
            pslr = f"{q['pslr_db']:10.2f}" if a < 1.0 or os_ > 1.0 else f"{'':>10}"
            print(f"{name:<20}{axis:<9}{os_:7.3f}{a:6.2f}{pslr}{q['irw_samples']:7.3f}"
                  f"{sim.intensity_correlation(os_, a, 1):10.4f}{measured[k]:10.4f}")

    n = sim.EXTRA_DATES
    print(f"\nTemporal median of {n} single-look dates: theoretical E[median]/X = {sim.median_bias(n):.4f}")
    bias = sim.median_bias(n)
    for sensor in ("s1", "umbra"):
        ratios, floor_raw, floor_corrected = [], [], []
        for item in sim.phantom_set("eval", sensor, n=args.phantoms, with_stack=True):
            median = np.clip(np.median(item["stack"], axis=0), 1e-8, None)
            clean, mask = item["clean"], item["mask"]
            ratios.append(float(np.mean(median[mask] / clean[mask])))
            floor_raw.append(float(np.sqrt(np.mean((np.log(clean) - np.log(median)) ** 2))))
            floor_corrected.append(float(np.sqrt(np.mean((np.log(clean) - np.log(median / bias)) ** 2))))
        print(f"  {sensor}, {len(ratios)} phantoms: measured E[median]/X = {np.mean(ratios):.4f}; root-mean-square log "
              f"difference of the reference X from the raw median {np.mean(floor_raw):.3f}, "
              f"from the corrected median {np.mean(floor_corrected):.3f}")


if __name__ == "__main__":
    main()
