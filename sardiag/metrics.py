"""True error measures, available when the reference X is known."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter

EPS = 1e-8


def _norm(x: np.ndarray) -> np.ndarray:
    return (x / (float(np.mean(x)) + 1e-8)).astype(np.float32)


def true_log_rmse(estimate: np.ndarray, clean: np.ndarray) -> float:
    """Root-mean-square difference between the mean-normalized log estimate and the mean-normalized log reference."""
    a = np.log(np.clip(estimate / max(float(estimate.mean()), EPS), EPS, None))
    b = np.log(np.clip(clean / max(float(clean.mean()), EPS), EPS, None))
    return float(np.sqrt(np.mean((a - b) ** 2)))


def edge_preservation(estimate: np.ndarray, clean: np.ndarray) -> float:
    """Pearson correlation between the gradient magnitudes of the log estimate and the log reference."""
    le, lt = np.log(_norm(estimate) + 1e-3), np.log(_norm(clean) + 1e-3)

    def grad(a: np.ndarray) -> np.ndarray:
        gy = np.abs(np.diff(a, axis=0, append=a[-1:, :]))
        gx = np.abs(np.diff(a, axis=1, append=a[:, -1:]))
        return np.sqrt(gx ** 2 + gy ** 2)

    ge, gt = grad(le).ravel(), grad(lt).ravel()
    if ge.std() < 1e-8 or gt.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(ge, gt)[0, 1])


def point_target_ratio(estimate: np.ndarray, clean: np.ndarray, coords) -> float:
    """Median over targets of the largest estimate value in the 3 x 3 neighborhood divided by the reference peak."""
    est, tgt = _norm(estimate), _norm(clean)
    ratios = []
    for py, px in coords:
        y0, y1 = max(0, py - 1), min(est.shape[0], py + 2)
        x0, x1 = max(0, px - 1), min(est.shape[1], px + 2)
        peak_tgt = float(tgt[py, px])
        if peak_tgt > 1e-6:
            ratios.append(float(est[y0:y1, x0:x1].max()) / peak_tgt)
    return float(np.median(ratios)) if ratios else float("nan")


def peak_error(estimate: np.ndarray, clean: np.ndarray, coords) -> float:
    """Point-target error: absolute logarithm of point_target_ratio."""
    return float(abs(np.log(max(point_target_ratio(estimate, clean, coords), EPS))))


def psnr(estimate: np.ndarray, clean: np.ndarray) -> float:
    """PSNR on mean-normalized intensity with the maximum of the normalized reference as the peak value."""
    estimate, clean = _norm(estimate), _norm(clean)
    mse = float(np.mean((estimate - clean) ** 2))
    return 10.0 * np.log10(float(clean.max()) ** 2 / (mse + 1e-12))


def ssim(estimate: np.ndarray, clean: np.ndarray, window: int = 7) -> float:
    """SSIM on mean-normalized intensity with a uniform window."""
    x, y = _norm(estimate), _norm(clean)
    c1, c2 = (0.01 * y.max()) ** 2, (0.03 * y.max()) ** 2
    mu_x, mu_y = uniform_filter(x, window), uniform_filter(y, window)
    mu_x2, mu_y2, mu_xy = mu_x ** 2, mu_y ** 2, mu_x * mu_y
    sigma_x = uniform_filter(x ** 2, window) - mu_x2
    sigma_y = uniform_filter(y ** 2, window) - mu_y2
    sigma_xy = uniform_filter(x * y, window) - mu_xy
    smap = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / ((mu_x2 + mu_y2 + c1) * (sigma_x + sigma_y + c2) + 1e-12)
    return float(np.mean(smap))


def enl_homogeneous(estimate: np.ndarray, mask: np.ndarray) -> float:
    """Equivalent number of looks of the mean-normalized estimate in the homogeneous region."""
    values = _norm(estimate)[mask]
    mu, var = float(values.mean()), float(values.var())
    return mu ** 2 / (var + 1e-8)
