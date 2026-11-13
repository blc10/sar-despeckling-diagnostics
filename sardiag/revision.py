"""Additional measures for the first revision (reviewer comments on electronics-4674775).

Nothing in this module changes the measures of the submitted study; the functions in ``diagnostics`` and ``metrics``
remain as released so that the stored tables stay reproducible. The functions here add

* a Jensen-Shannon divergence between the two fitted Gamma laws of the TDM that is a properly normalized divergence
  (probability masses that sum to one; bounded by log 2), next to the released density-sum convention,
* target-wise point-target errors (absolute and signed) next to the released median-ratio error,
* an error that keeps the global level (no mean normalization) and the radiometric bias as a separate axis,
* a target-level table of the observation-referenced strong-scatterer proxy, and
* a simple cell-averaging CFAR detection score for point targets.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import maximum_filter, uniform_filter
from scipy.stats import gamma

from . import diagnostics as dg
from . import real_data as rd

EPS = 1e-8
LN2 = math.log(2.0)
JSD_GRID = 4096
JSD_TAIL = 1e-9


# ------------------------------------------------------------------------------------------------ normalized TDM
def _gamma_fits(ratio_roi: np.ndarray, reference_roi: np.ndarray):
    fits = []
    for values in (ratio_roi, reference_roi):
        values = values[np.isfinite(values) & (values > 0)]
        fits.append(gamma.fit(values, floc=0))
    return fits


def fitted_gamma_jsd_normalized(ratio_roi: np.ndarray, reference_roi: np.ndarray) -> float:
    """Jensen-Shannon divergence (nats, in [0, log 2]) between the Gamma laws fitted to the two samples.

    The fits are those of ``diagnostics._fitted_gamma_jsd``. The divergence is evaluated on probability masses of a
    common partition that covers both laws up to a tail probability of 1e-9 at each end, so the two mass vectors sum
    to one and the result does not depend on the grid spacing. Degenerate ratio samples return log 2, which mirrors
    the cap of the released convention.
    """
    finite = ratio_roi[np.isfinite(ratio_roi) & (ratio_roi > 0)]
    if finite.size < 16 or float(np.std(finite)) < 1e-9:
        return LN2
    (s1, l1, c1), (s2, l2, c2) = _gamma_fits(ratio_roi, reference_roi)
    lo = min(gamma.ppf(JSD_TAIL, s1, l1, c1), gamma.ppf(JSD_TAIL, s2, l2, c2))
    hi = max(gamma.ppf(1.0 - JSD_TAIL, s1, l1, c1), gamma.ppf(1.0 - JSD_TAIL, s2, l2, c2))
    if not np.isfinite(lo + hi) or hi <= lo:
        return LN2
    edges = np.concatenate(([0.0], np.geomspace(max(lo, 1e-300), hi, JSD_GRID), [np.inf]))
    p = np.diff(gamma.cdf(edges, s1, l1, c1))
    q = np.diff(gamma.cdf(edges, s2, l2, c2))
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)
    with np.errstate(divide="ignore", invalid="ignore"):
        kl_p = np.where(p > 0, p * np.log(p / m), 0.0).sum()
        kl_q = np.where(q > 0, q * np.log(q / m), 0.0).sum()
    return float(min(max(0.5 * (kl_p + kl_q), 0.0), LN2))


def released_jsd_step(ratio_roi: np.ndarray, reference_roi: np.ndarray) -> float:
    """Grid spacing of the released density-sum convention; released value ~ (truncated JSD) / spacing."""
    (s1, _, c1), (s2, _, c2) = _gamma_fits(ratio_roi, reference_roi)
    mean = max(s1 * c1, s2 * c2)
    std = max(np.sqrt(s1) * c1, np.sqrt(s2) * c2)
    return float(((mean + 4.0 * std) - max(1e-3 * mean, mean - 4.0 * std)) / 999.0)


def tdm_normalized(noisy: np.ndarray, estimate: np.ndarray, homogeneous: np.ndarray) -> dict:
    """TDM and deviation D with the divergence term J = JSD / log 2 in [0, 1] (no cap, no density sum)."""
    ratio = np.asarray(noisy, dtype=np.float64) / np.maximum(np.asarray(estimate, dtype=np.float64), 1e-12)
    released = dg.tdm(noisy, estimate, homogeneous)          # also fills the pure-speckle reference cache
    ref, _ = dg._TDM_REFERENCE[ratio.shape]
    roi = np.asarray(homogeneous, dtype=bool)
    j = fitted_gamma_jsd_normalized(ratio[roi], ref[roi]) / LN2
    c_hat, v_hat = released["contrast"], released["variance"]
    return {"tdm_norm": float(0.5 * dg.TDM_ALPHA * (c_hat + v_hat) + (1.0 - dg.TDM_ALPHA) * (1.0 - j)),
            "tdm_deviation_norm": float(0.25 * (abs(c_hat - 1.0) + abs(v_hat - 1.0)) + 0.5 * j),
            "tdm_jsd_norm": float(j), "tdm_jsd_released_saturated": bool(released["jsd"] >= 1.0 - 1e-12)}


# ------------------------------------------------------------------------------------------------ level-aware error
def _unit(x: np.ndarray) -> np.ndarray:
    return (x / (float(np.mean(x)) + 1e-8)).astype(np.float32)


def total_log_rmse(estimate: np.ndarray, clean: np.ndarray) -> float:
    """Root-mean-square log difference without mean normalization; includes a global level error."""
    a = np.log(np.clip(np.asarray(estimate, dtype=np.float64), EPS, None))
    b = np.log(np.clip(np.asarray(clean, dtype=np.float64), EPS, None))
    return float(np.sqrt(np.mean((a - b) ** 2)))


def radiometric_log_bias(estimate: np.ndarray, clean: np.ndarray) -> float:
    """Signed log ratio of the image means of estimate and reference."""
    return float(np.log(max(float(np.mean(estimate)), EPS) / max(float(np.mean(clean)), EPS)))


# ------------------------------------------------------------------------------------------------ target-wise peaks
def target_log_ratios(estimate: np.ndarray, clean: np.ndarray, coords) -> np.ndarray:
    """Log of (3 x 3 peak of the estimate / reference peak) per target, on mean-normalized images.

    Same normalization and neighborhood as ``metrics.point_target_ratio``. Targets whose reference peak is not above
    1e-6 are returned as NaN (they are skipped by the released measure).
    """
    est, tgt = _unit(estimate), _unit(clean)
    out = np.full(len(coords), np.nan)
    for k, (py, px) in enumerate(coords):
        y0, y1 = max(0, py - 1), min(est.shape[0], py + 2)
        x0, x1 = max(0, px - 1), min(est.shape[1], px + 2)
        peak = float(tgt[py, px])
        if peak > 1e-6:
            out[k] = math.log(max(float(est[y0:y1, x0:x1].max()), EPS) / peak)
    return out


def target_contrast_db(clean: np.ndarray, coords, inner: int = 5, outer: int = 11) -> np.ndarray:
    """Target strength: reference peak over the mean reference in the surrounding ring, in dB."""
    clean = np.asarray(clean, dtype=np.float64)
    ring = (uniform_filter(clean, outer, mode="reflect") * outer ** 2
            - uniform_filter(clean, inner, mode="reflect") * inner ** 2) / (outer ** 2 - inner ** 2)
    return np.asarray([10.0 * math.log10(max(clean[py, px], EPS) / max(ring[py, px], EPS)) for py, px in coords])


def peak_summary(log_ratios: np.ndarray, tolerance: float = LN2) -> dict:
    """Target-wise summaries. tolerance: |log ratio| above which a target counts as attenuated or amplified."""
    v = log_ratios[np.isfinite(log_ratios)]
    if v.size == 0:
        return {k: float("nan") for k in ("peak_abs_mean", "peak_abs_median", "peak_signed_mean", "peak_signed_median",
                                          "peak_frac_attenuated", "peak_frac_amplified")} | {"peak_targets": 0,
                                                                                             "peak_missing": int(log_ratios.size)}
    return {"peak_abs_mean": float(np.mean(np.abs(v))), "peak_abs_median": float(np.median(np.abs(v))),
            "peak_signed_mean": float(np.mean(v)), "peak_signed_median": float(np.median(v)),
            "peak_frac_attenuated": float(np.mean(v < -tolerance)), "peak_frac_amplified": float(np.mean(v > tolerance)),
            "peak_targets": int(v.size), "peak_missing": int(log_ratios.size - v.size)}


# ------------------------------------------------------------------------------------------------ proxy, target level
def strong_scatterers(noisy: np.ndarray) -> list[tuple[int, int]]:
    """Positions selected by ``real_data.strong_scatterer_peak_error`` (identical rule; observation only)."""
    noisy = np.clip(np.asarray(noisy, dtype=np.float64), EPS, None)
    _, outer = rd.CLUTTER_RING
    threshold = max(float(np.quantile(noisy, rd.SCATTERER_QUANTILE)), rd.SCATTERER_MEDIAN_FACTOR * float(np.median(noisy)))
    maxima = maximum_filter(noisy, size=outer, mode="reflect")
    candidates = np.argwhere((noisy >= threshold) & (noisy >= maxima))
    if candidates.size == 0:
        return []
    order = np.argsort(noisy[candidates[:, 0], candidates[:, 1]])[::-1]
    height, width = noisy.shape
    selected: list[tuple[int, int]] = []
    for index in order:
        row, col = map(int, candidates[index])
        if row < outer or col < outer or row + outer >= height or col + outer >= width:
            continue
        if any((row - pr) ** 2 + (col - pc) ** 2 < outer ** 2 for pr, pc in selected):
            continue
        selected.append((row, col))
        if len(selected) >= rd.SCATTERER_MAX_COUNT:
            break
    return selected


def _peak3(image: np.ndarray, row: int, col: int) -> float:
    return float(np.max(image[max(row - 1, 0):row + 2, max(col - 1, 0):col + 2]))


def proxy_target_table(noisy: np.ndarray, estimate: np.ndarray, clean: np.ndarray, coords, match_radius: float = 1.5) -> list[dict]:
    """One row per scatterer selected in the observation, with the proxy term and the corresponding true term.

    proxy  = log(peak3(estimate) / peak3(observation))     -- what the real-data measure averages in absolute value
    truth  = log(peak3(estimate) / peak3(reference))       -- same neighborhood, exact reference X
    select = log(peak3(observation) / peak3(reference))    -- speckle excursion of the selected observation peak
    All images are used at the common unit-mean-observation scale of the simulator (no per-image renormalization),
    as on real data. is_target: a simulated deterministic target lies within match_radius pixels. contrast_db and
    obs_contrast_db: selected pixel over the mean of the surrounding ring in the reference and in the observation.
    """
    noisy = np.clip(np.asarray(noisy, dtype=np.float64), EPS, None)
    estimate = np.clip(np.asarray(estimate, dtype=np.float64), EPS, None)
    clean = np.clip(np.asarray(clean, dtype=np.float64), EPS, None)
    selected = strong_scatterers(noisy)
    contrast = target_contrast_db(clean, selected)
    observed = target_contrast_db(noisy, selected)
    rows = []
    for k, (row, col) in enumerate(selected):
        dist = min((math.hypot(row - py, col - px) for py, px in coords), default=float("inf"))
        pe, pz, px_ = _peak3(estimate, row, col), _peak3(noisy, row, col), _peak3(clean, row, col)
        rows.append({"row": row, "col": col, "is_target": bool(dist <= match_radius), "contrast_db": float(contrast[k]), "obs_contrast_db": float(observed[k]),
                     "proxy": math.log(pe / pz), "truth": math.log(pe / px_), "select": math.log(pz / px_)})
    return rows


# ------------------------------------------------------------------------------------------------ detection
def cfar_statistic(image: np.ndarray, inner: int = 5, outer: int = 11) -> np.ndarray:
    """Cell-averaging CFAR statistic: pixel value over the mean of the surrounding ring."""
    image = np.clip(np.asarray(image, dtype=np.float64), EPS, None)
    ring = (uniform_filter(image, outer, mode="reflect") * outer ** 2
            - uniform_filter(image, inner, mode="reflect") * inner ** 2) / (outer ** 2 - inner ** 2)
    return image / np.maximum(ring, EPS)


def detection_scores(image: np.ndarray, coords, guard: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """CFAR statistics at the targets (3 x 3 maximum) and at all pixels farther than guard pixels from any target."""
    stat = cfar_statistic(image)
    keep = np.ones(stat.shape, dtype=bool)
    target = np.empty(len(coords))
    for k, (py, px) in enumerate(coords):
        keep[max(py - guard, 0):py + guard + 1, max(px - guard, 0):px + guard + 1] = False
        target[k] = _peak3(stat, py, px)
    return target, stat[keep]


# ------------------------------------------------------------------------------------------------ one call
def evaluate_revision(noisy: np.ndarray, estimate: np.ndarray, clean: np.ndarray, coords, mask: np.ndarray) -> dict:
    """All revision columns of one estimate on one simulated scene (to be merged with ``evaluation.evaluate``)."""
    estimate = np.clip(np.asarray(estimate, dtype=np.float32), EPS, None)
    rec = {"total_log_rmse": total_log_rmse(estimate, clean), "radiometric_log_bias": radiometric_log_bias(estimate, clean)}
    rec.update(peak_summary(target_log_ratios(estimate, clean, coords)))
    rec.update(tdm_normalized(noisy, estimate, mask))
    table = proxy_target_table(noisy, estimate, clean, coords)
    rec["obs_scatterers"] = len(table)
    rec["obs_scatterers_true"] = int(sum(t["is_target"] for t in table))
    rec["obs_peak_abs_mean"] = float(np.mean([abs(t["proxy"]) for t in table])) if table else float("nan")
    rec["obs_peak_signed_mean"] = float(np.mean([t["proxy"] for t in table])) if table else float("nan")
    return rec
