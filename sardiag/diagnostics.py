"""No-reference despeckling diagnostics computed on the ratio image R = Z / X_hat.

All diagnostics are oriented so that a high value indicates a poor estimate, except the texture-divergence measure (TDM)
in its published orientation, for which a high value indicates a good estimate. The two-sided deviation D of the TDM
components from their pure-speckle value is oriented like the other diagnostics.
"""
from __future__ import annotations

import hashlib
import math

import numpy as np
from scipy.special import kl_div
from scipy.stats import gamma

EPS = 1e-8

# Settings of the Gamma JSD and the M-index (single-look intensity).
NOMINAL_LOOKS = 1.0
JSD_BINS = 64
JSD_TAIL = 0.0005
BLOCK_SIZE = 32
BLOCK_MIN_VALID = 0.9
BLOCK_FRACTION = 0.2
BLOCK_MIN_COUNT = 4
M_LEVELS = 32
M_CLIP = (0.5, 99.5)
M_DOWNSAMPLE = 4
M_OFFSETS = ((0, 1), (1, 0), (1, 1), (1, -1))
M_PERMUTATIONS = 4
M_SEED = 20269801

# Settings of the TDM (Vasquez-Salazar et al., Remote Sens. 2025, 17, 4048).
TDM_DISTANCE = 9
TDM_ALPHA = 0.5
TDM_JSD_CAP = 5.0
TDM_REFERENCE_SEED = 20251217


def ratio_image(noisy: np.ndarray, estimate: np.ndarray) -> np.ndarray:
    return np.asarray(noisy, dtype=np.float64) / np.maximum(np.asarray(estimate, dtype=np.float64), EPS)


# ------------------------------------------------------------------------------------------------ ratio-edge measure
def _gradient_magnitude(x: np.ndarray) -> np.ndarray:
    dx = np.zeros_like(x)
    dy = np.zeros_like(x)
    dx[:, :-1] = x[:, 1:] - x[:, :-1]
    dy[:-1, :] = x[1:, :] - x[:-1, :]
    return np.sqrt(dx * dx + dy * dy + np.float32(1e-8))


def ratio_edge(noisy: np.ndarray, estimate: np.ndarray) -> float:
    """r = |corr(|grad log R|, |grad log X_hat|)|, computed in single precision."""
    log_est = np.log(np.clip(estimate, EPS, None)).astype(np.float32)
    log_obs = np.log(np.clip(noisy, EPS, None)).astype(np.float32)
    g_ratio = _gradient_magnitude(log_obs - log_est)
    g_est = _gradient_magnitude(log_est)
    a = g_ratio - g_ratio.mean(dtype=np.float32)
    b = g_est - g_est.mean(dtype=np.float32)
    cov = (a * b).mean(dtype=np.float32)
    denom = np.sqrt((a * a).mean(dtype=np.float32) * (b * b).mean(dtype=np.float32) + np.float32(1e-8))
    return float(abs(cov / denom))


# ------------------------------------------------------------------------------------------------ Gamma JSD
def _finite(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values[np.isfinite(values)]


def gamma_jsd(ratio: np.ndarray, looks: float = NOMINAL_LOOKS) -> float:
    """Jensen-Shannon divergence between the ratio histogram and the unit-mean Gamma distribution."""
    values = _finite(ratio)
    if values.size < 64:
        return float("nan")
    upper = float(gamma.ppf(1.0 - JSD_TAIL, a=looks, scale=1.0 / looks))
    edges = np.linspace(0.0, upper, JSD_BINS + 1, dtype=np.float64)
    clipped = np.clip(values, 0.0, np.nextafter(upper, 0.0))
    empirical, _ = np.histogram(clipped, bins=edges)
    empirical = empirical.astype(np.float64) + EPS
    empirical /= empirical.sum()
    reference = np.diff(gamma.cdf(edges, a=looks, scale=1.0 / looks)) + EPS
    reference /= reference.sum()
    middle = 0.5 * (empirical + reference)
    jsd = 0.5 * np.sum(empirical * np.log(empirical / middle))
    jsd += 0.5 * np.sum(reference * np.log(reference / middle))
    return float(jsd)


# ------------------------------------------------------------------------------------------------ M-index
def _enl(values: np.ndarray) -> float:
    values = _finite(values)
    if values.size < 16:
        return float("nan")
    return float(values.mean() ** 2 / max(values.var(ddof=0), EPS))


def _nanmean(values) -> float:
    values = _finite(np.asarray(values))
    return float(values.mean()) if values.size else float("nan")


def homogeneous_blocks(noisy: np.ndarray, valid: np.ndarray) -> list:
    """The 20% of 32 x 32 blocks with the lowest coefficient of variation in the observation (at least four)."""
    candidates = []
    height, width = noisy.shape
    for row in range(0, height - BLOCK_SIZE + 1, BLOCK_SIZE):
        for col in range(0, width - BLOCK_SIZE + 1, BLOCK_SIZE):
            rs, cs = slice(row, row + BLOCK_SIZE), slice(col, col + BLOCK_SIZE)
            mask = valid[rs, cs]
            if float(mask.mean()) < BLOCK_MIN_VALID:
                continue
            values = noisy[rs, cs][mask]
            candidates.append((float(values.std() / max(values.mean(), EPS)), rs, cs, mask))
    if not candidates:
        return []
    candidates.sort(key=lambda item: (item[0], item[1].start, item[2].start))
    count = min(max(BLOCK_MIN_COUNT, int(math.ceil(BLOCK_FRACTION * len(candidates)))), len(candidates))
    return [(rs, cs, mask) for _, rs, cs, mask in candidates[:count]]


def _glcm_homogeneity(quantized: np.ndarray, valid: np.ndarray) -> float:
    matrix = np.zeros((M_LEVELS, M_LEVELS), dtype=np.float64)
    height, width = quantized.shape
    for dy, dx in M_OFFSETS:
        y0a, y1a = max(0, -dy), min(height, height - dy)
        x0a, x1a = max(0, -dx), min(width, width - dx)
        y0b, y1b, x0b, x1b = y0a + dy, y1a + dy, x0a + dx, x1a + dx
        pair = valid[y0a:y1a, x0a:x1a] & valid[y0b:y1b, x0b:x1b]
        if not pair.any():
            continue
        left = quantized[y0a:y1a, x0a:x1a][pair]
        right = quantized[y0b:y1b, x0b:x1b][pair]
        counts = np.bincount(left * M_LEVELS + right, minlength=M_LEVELS * M_LEVELS).reshape(M_LEVELS, M_LEVELS)
        matrix += counts
        matrix += counts.T
    total = matrix.sum()
    if total <= 0:
        return float("nan")
    idx = np.arange(M_LEVELS)
    return float(np.sum(matrix / total / (1.0 + (idx[:, None] - idx[None, :]) ** 2)))


def _structure_term(ratio: np.ndarray, valid: np.ndarray, seed_key: str) -> float:
    reduced = ratio[::M_DOWNSAMPLE, ::M_DOWNSAMPLE]
    ok = valid[::M_DOWNSAMPLE, ::M_DOWNSAMPLE] & np.isfinite(reduced)
    values = reduced[ok]
    if values.size < 256:
        return float("nan")
    low, high = np.percentile(values, M_CLIP)
    if not np.isfinite(low + high) or high <= low + EPS:
        return 0.0
    scaled = np.clip((reduced - low) / (high - low), 0.0, np.nextafter(1.0, 0.0))
    quantized = np.floor(scaled * M_LEVELS).astype(np.int32)
    original = _glcm_homogeneity(quantized, ok)
    seed = int.from_bytes(hashlib.sha256(f"{M_SEED}|{seed_key}".encode("utf-8")).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    coordinates = np.flatnonzero(ok)
    flat = quantized.ravel()
    original_values = flat[coordinates].copy()
    shuffled = []
    for _ in range(M_PERMUTATIONS):
        sample = flat.copy()
        sample[coordinates] = rng.permutation(original_values)
        shuffled.append(_glcm_homogeneity(sample.reshape(quantized.shape), ok))
    return 100.0 * abs(original - _nanmean(shuffled)) / max(abs(original), EPS)


def m_index(noisy: np.ndarray, estimate: np.ndarray, valid: np.ndarray | None = None, seed_key: str = "x|patch") -> float:
    """M-index: mean of a first-order term on homogeneous blocks and a structural term from co-occurrence homogeneity.

    The structural term compares the ratio image with random permutations of itself; seed_key makes the permutations
    reproducible (the study used 'x|<phantom>_<method>').
    """
    noisy = np.clip(np.asarray(noisy, dtype=np.float64), EPS, np.inf)
    estimate = np.clip(np.asarray(estimate, dtype=np.float64), EPS, np.inf)
    valid = np.ones(noisy.shape, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    ratio = noisy / estimate
    blocks = homogeneous_blocks(noisy, valid)
    if blocks:
        noisy_enl = np.asarray([_enl(noisy[rs, cs][m]) for rs, cs, m in blocks])
        ratio_enl = np.asarray([_enl(ratio[rs, cs][m]) for rs, cs, m in blocks])
        ratio_mean = np.asarray([_nanmean(ratio[rs, cs][m]) for rs, cs, m in blocks])
        relative_enl = np.abs((ratio_enl - noisy_enl) / np.maximum(noisy_enl, EPS))
        first_order = 50.0 * (_nanmean(relative_enl) + _nanmean(np.abs(ratio_mean - 1.0)))
    else:
        first_order = float("nan")
    return 0.5 * (first_order + _structure_term(ratio, valid, seed_key))


# ------------------------------------------------------------------------------------------------ TDM
def _quantize(image: np.ndarray, levels: int = 256) -> np.ndarray:
    top = float(np.percentile(image, 99.9))
    return np.round(np.clip(image / max(top, 1e-12), 0.0, 1.0) * (levels - 1)).astype(np.int64)


def _glcm_contrast_variance(q: np.ndarray, levels: int = 256, distance: int = TDM_DISTANCE) -> tuple[float, float]:
    h, w = q.shape
    d = distance
    idx = np.arange(levels, dtype=np.float64)
    contrasts, variances = [], []
    for dy, dx in ((0, d), (-d, d), (-d, 0), (-d, -d)):
        y0, y1 = max(0, -dy), h - max(0, dy)
        x0, x1 = max(0, -dx), w - max(0, dx)
        a = q[y0:y1, x0:x1]
        b = q[y0 + dy:y1 + dy, x0 + dx:x1 + dx]
        pairs = np.bincount((a * levels + b).ravel(), minlength=levels * levels).reshape(levels, levels).astype(np.float64)
        pairs = pairs + pairs.T
        pairs[0, :] = 0.0
        pairs[:, 0] = 0.0
        total = pairs.sum()
        if total <= 0:
            continue
        p = pairs / total
        i, j = np.meshgrid(idx, idx, indexing="ij")
        contrasts.append(float(np.sum((i - j) ** 2 * p)))
        px = p.sum(axis=1)
        mu = float(np.dot(px, idx))
        variances.append(float(np.dot(px, (idx - mu) ** 2)))
    return float(np.mean(contrasts)), float(np.mean(variances))


def _fitted_gamma_jsd(ratio_roi: np.ndarray, reference_roi: np.ndarray) -> float:
    finite = ratio_roi[np.isfinite(ratio_roi) & (ratio_roi > 0)]
    if finite.size < 16 or float(np.std(finite)) < 1e-9:
        return TDM_JSD_CAP
    fits = []
    for values in (ratio_roi, reference_roi):
        values = values[np.isfinite(values) & (values > 0)]
        fits.append(gamma.fit(values, floc=0))
    (s1, l1, c1), (s2, l2, c2) = fits
    mean = max(s1 * c1, s2 * c2)
    std = max(np.sqrt(s1) * c1, np.sqrt(s2) * c2)
    x = np.linspace(max(1e-3 * mean, mean - 4.0 * std), mean + 4.0 * std, 1000)
    p = gamma.pdf(x, s1, l1, c1)
    q = gamma.pdf(x, s2, l2, c2)
    m = 0.5 * (p + q)
    return float(0.5 * np.sum(kl_div(p, m)) + 0.5 * np.sum(kl_div(q, m)))


_TDM_REFERENCE: dict = {}


def tdm(noisy: np.ndarray, estimate: np.ndarray, homogeneous: np.ndarray) -> dict:
    """TDM components scaled by a pure-speckle reference, the published TDM and the two-sided deviation D."""
    ratio = np.asarray(noisy, dtype=np.float64) / np.maximum(np.asarray(estimate, dtype=np.float64), 1e-12)
    if ratio.shape not in _TDM_REFERENCE:
        ref = np.random.default_rng(TDM_REFERENCE_SEED).gamma(shape=1.0, scale=1.0, size=ratio.shape)
        _TDM_REFERENCE[ratio.shape] = (ref, _glcm_contrast_variance(_quantize(ref)))
    ref, (c_ref, v_ref) = _TDM_REFERENCE[ratio.shape]
    c, v = _glcm_contrast_variance(_quantize(ratio))
    roi = np.asarray(homogeneous, dtype=bool)
    j = min(_fitted_gamma_jsd(ratio[roi], ref[roi]), TDM_JSD_CAP) / TDM_JSD_CAP
    c_hat, v_hat = c / c_ref, v / v_ref
    published = 0.5 * TDM_ALPHA * (c_hat + v_hat) + (1.0 - TDM_ALPHA) * (1.0 - j)
    deviation = 0.25 * (abs(c_hat - 1.0) + abs(v_hat - 1.0)) + 0.5 * j
    return {"tdm": float(published), "tdm_deviation": float(deviation), "contrast": float(c_hat),
            "variance": float(v_hat), "jsd": float(j)}
