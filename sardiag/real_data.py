"""Measures for real data, where the reflectivity is unknown and every reference is taken from the observation."""
from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import maximum_filter

from . import diagnostics

EPS = 1e-8

# Strong-scatterer selection: local maxima of the observation above the 99.5th percentile and ten times the median.
SCATTERER_QUANTILE = 0.995
SCATTERER_MEDIAN_FACTOR = 10.0
SCATTERER_MAX_COUNT = 8
SCATTERER_WINDOW = 3
CLUTTER_RING = (5, 11)


def enl_gain(noisy: np.ndarray, estimate: np.ndarray) -> float:
    """Mean ENL of the estimate divided by the mean ENL of the observation over the homogeneous blocks."""
    noisy = np.clip(np.asarray(noisy, dtype=np.float64), EPS, None)
    estimate = np.clip(np.asarray(estimate, dtype=np.float64), EPS, None)
    blocks = diagnostics.homogeneous_blocks(noisy, np.ones(noisy.shape, dtype=bool))
    noisy_enl = diagnostics._nanmean([diagnostics._enl(noisy[rs, cs][m]) for rs, cs, m in blocks])
    estimate_enl = diagnostics._nanmean([diagnostics._enl(estimate[rs, cs][m]) for rs, cs, m in blocks])
    return float(estimate_enl / max(noisy_enl, EPS))


def strong_scatterer_peak_error(noisy: np.ndarray, estimate: np.ndarray) -> tuple[float, int]:
    """Mean |log| ratio of the 3 x 3 peak of the estimate to that of the observation at strong scatterers.

    Returns the error and the number of scatterers used. Scatterers are detected in the observation only, so the
    measure does not depend on the method output.
    """
    noisy = np.clip(np.asarray(noisy, dtype=np.float64), EPS, None)
    estimate = np.clip(np.asarray(estimate, dtype=np.float64), EPS, None)
    inner, outer = CLUTTER_RING
    radius = SCATTERER_WINDOW // 2
    threshold = max(float(np.quantile(noisy, SCATTERER_QUANTILE)), SCATTERER_MEDIAN_FACTOR * float(np.median(noisy)))
    maxima = maximum_filter(noisy, size=outer, mode="reflect")
    candidates = np.argwhere((noisy >= threshold) & (noisy >= maxima))
    if candidates.size == 0:
        return float("nan"), 0
    order = np.argsort(noisy[candidates[:, 0], candidates[:, 1]])[::-1]
    height, width = noisy.shape
    selected = []
    for index in order:
        row, col = map(int, candidates[index])
        if row < outer or col < outer or row + outer >= height or col + outer >= width:
            continue
        if any((row - pr) ** 2 + (col - pc) ** 2 < outer ** 2 for pr, pc in selected):
            continue
        selected.append((row, col))
        if len(selected) >= SCATTERER_MAX_COUNT:
            break
    errors = []
    for row, col in selected:
        center = (slice(row - radius, row + radius + 1), slice(col - radius, col + radius + 1))
        ratio = float(np.max(estimate[center])) / max(float(np.max(noisy[center])), EPS)
        errors.append(abs(math.log(max(ratio, EPS))))
    return (float(np.mean(errors)), len(errors)) if errors else (float("nan"), 0)


def real_data_measures(noisy: np.ndarray, estimate: np.ndarray) -> dict:
    """No-reference diagnostics and observation-referenced measures of one estimate on a real patch."""
    noisy64 = np.clip(np.asarray(noisy, dtype=np.float64), EPS, None)
    estimate64 = np.clip(np.asarray(estimate, dtype=np.float64), EPS, None)
    blocks = diagnostics.homogeneous_blocks(noisy64, np.ones(noisy64.shape, dtype=bool))
    homogeneous = np.zeros(noisy64.shape, dtype=bool)
    for rs, cs, mask in blocks:
        homogeneous[rs, cs] |= mask
    t = diagnostics.tdm(noisy64, estimate64, homogeneous)
    peak, count = strong_scatterer_peak_error(noisy64, estimate64)
    return {"r": diagnostics.ratio_edge(noisy64.astype(np.float32), estimate64.astype(np.float32)),
            "gamma_jsd": diagnostics.gamma_jsd(diagnostics.ratio_image(noisy64, estimate64)),
            "m_index": diagnostics.m_index(noisy64, estimate64, seed_key="rf|rf"),
            "tdm": t["tdm"], "tdm_deviation": t["tdm_deviation"],
            "enl_gain": enl_gain(noisy64, estimate64), "peak_error": peak, "scatterers": count}
