"""Controlled degradations with known amounts of error (29 outputs per phantom)."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter, uniform_filter

GAUSSIAN_SIGMAS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
MEAN_SIZES = (3, 5, 7, 9, 11)
EDGE_ALPHAS = (1.0, 0.8, 0.6, 0.4, 0.2, 0.0)
PEAK_BETAS = (1.0, 0.75, 0.5, 0.25, 0.1)
RADIOMETRIC_GAMMAS = (1.0, 0.9, 0.8, 1.1, 1.2)


def seed_key(name: str, phantom_index: int) -> str:
    """Key of the M-index permutations for a degraded output, as used in the study."""
    return f"{name}|a147_{phantom_index}"


def controlled_degradations(clean: np.ndarray, noisy: np.ndarray, coords) -> list[tuple[str, str, float, np.ndarray]]:
    """Return (name, family, level, output) for the five families.

    Gaussian blur, edge suppression, peak attenuation and radiometric scaling modify the reference; mean filtering is
    applied to the observation and therefore represents a denoising series.
    """
    out = []
    for sigma in GAUSSIAN_SIGMAS:
        out.append((f"gauss_sigma_{sigma}", "gauss", sigma, clean if sigma == 0 else gaussian_filter(clean, sigma)))
    for size in MEAN_SIZES:
        out.append((f"mean_{size}x{size}", "mean", float(size), uniform_filter(noisy, size)))
    for alpha in EDGE_ALPHAS:
        low = gaussian_filter(clean, 2.0)
        out.append((f"edge_keep_{alpha}", "edge", alpha, low + alpha * (clean - low)))
    for beta in PEAK_BETAS:
        attenuated = clean.copy()
        for r, c in coords:
            r, c = int(r), int(c)
            patch = attenuated[max(r - 1, 0):r + 2, max(c - 1, 0):c + 2]
            local = gaussian_filter(clean, 3.0)[max(r - 1, 0):r + 2, max(c - 1, 0):c + 2]
            attenuated[max(r - 1, 0):r + 2, max(c - 1, 0):c + 2] = local + beta * (patch - local)
        out.append((f"peak_keep_{beta}", "peak", beta, attenuated))
    for g in RADIOMETRIC_GAMMAS:
        out.append((f"radio_{g}", "radio", g, clean * g))
    return out
