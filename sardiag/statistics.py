"""Statistics used in the study: AUC, rank correlation, resampling intervals, permutation tests, Holm correction and
thresholds. With the default seeds, the functions reproduce the random draws of the reported analyses."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

SEED_AUC = 20260928          # phantom bootstrap of the controlled-degradation AUCs
SEED_METHOD_LEVEL = 20261010  # phantom and method bootstrap and label permutations of the method-level correlations
SEED_REAL = 20261009          # permutations and record bootstrap of the pre-specified real-data test


def auc(score: np.ndarray, label: np.ndarray) -> float:
    """Mann-Whitney AUC; a high score is expected for positive (poor) outputs. Ties receive average ranks."""
    score, label = np.asarray(score, dtype=float), np.asarray(label, dtype=bool)
    pos, neg = int(label.sum()), int((~label).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[label].sum() - pos * (pos + 1) / 2) / (pos * neg))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman correlation; NaN when either variable is constant."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def holm(pvalues: dict) -> dict:
    """Holm step-down adjustment of a family of p-values."""
    keys = sorted(pvalues, key=lambda k: pvalues[k])
    m, out, running = len(keys), {}, 0.0
    for i, k in enumerate(keys):
        running = max(running, min(1.0, (m - i) * pvalues[k]))
        out[k] = running
    return out


def cluster_bootstrap(frame: pd.DataFrame, statistic, group: str = "phantom", draws: int = 2000,
                      seed: int = SEED_AUC) -> list[float]:
    """95% percentile interval of statistic(frame) when whole groups, for example phantoms, are resampled."""
    rng = np.random.default_rng(seed)
    groups = {k: g for k, g in frame.groupby(group)}
    keys = np.array(list(groups))
    values = []
    for _ in range(draws):
        pick = rng.choice(keys, len(keys), replace=True)
        v = statistic(pd.concat([groups[k] for k in pick], ignore_index=True))
        if np.isfinite(v):
            values.append(v)
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def method_level_correlations(cube: np.ndarray, columns: list[str], pairs: list[tuple[str, str]], draws: int = 2000,
                              permutations: int = 10000, seed: int = SEED_METHOD_LEVEL) -> dict:
    """Spearman correlation between method means of a diagnostic and of a true error.

    cube has the shape (phantoms, methods, columns). For each pair, the result holds rho, a 95% interval from
    resampling phantoms (method means recomputed in every draw), a 95% interval from resampling methods, a two-sided
    p-value from permuting the method labels and its Holm adjustment within the family of pairs.
    """
    idx = {c: i for i, c in enumerate(columns)}
    rng = np.random.default_rng(seed)
    n_ph, n_m = cube.shape[0], cube.shape[1]
    means = cube.mean(axis=0)
    boot_ph = rng.integers(0, n_ph, size=(draws, n_ph))
    boot_m = rng.integers(0, n_m, size=(draws, n_m))
    out, pvalues = {}, {}
    for d, t in pairs:
        a, b = means[:, idx[d]], means[:, idx[t]]
        rho = spearman(a, b)
        bp = [spearman(cube[s][:, :, idx[d]].mean(axis=0), cube[s][:, :, idx[t]].mean(axis=0)) for s in boot_ph]
        bm = [spearman(a[s], b[s]) for s in boot_m]
        perm = rng.permuted(np.tile(np.arange(n_m), (permutations, 1)), axis=1)
        null = np.array([spearman(a, b[p]) for p in perm])
        p = float((1 + np.sum(np.abs(null) >= abs(rho) - 1e-12)) / (1 + permutations))
        out[f"{d}|{t}"] = {"rho": rho, "ci_phantom": [float(np.nanpercentile(bp, 2.5)), float(np.nanpercentile(bp, 97.5))],
                           "ci_method": [float(np.nanpercentile(bm, 2.5)), float(np.nanpercentile(bm, 97.5))], "p": p}
        pvalues[f"{d}|{t}"] = p
    for k, v in holm(pvalues).items():
        out[k]["p_holm"] = v
    return out


def rank_correlation(x, y) -> float:
    """Spearman correlation over the pairs in which both values are finite; NaN for fewer than four pairs."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 4:
        return float("nan")
    return float(spearmanr(x[keep], y[keep])[0])


def one_sided_permutation_p(x, y, sign: int, permutations: int = 10000, seed: int = SEED_REAL) -> float:
    """One-sided permutation p-value for sign * rho being at least as large as observed."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    observed = sign * rank_correlation(x, y)
    rng = np.random.default_rng(seed)
    hits = sum(sign * rank_correlation(x, rng.permutation(y)) >= observed - 1e-12 for _ in range(permutations))
    return float((1 + hits) / (1 + permutations))


def youden_threshold(score: np.ndarray, label: np.ndarray) -> float:
    """Threshold that maximizes sensitivity + specificity - 1."""
    score, label = np.asarray(score, dtype=float), np.asarray(label, dtype=bool)
    best, best_j = None, -np.inf
    for t in np.unique(score):
        pred = score >= t
        j = (pred & label).sum() / max(label.sum(), 1) + (~pred & ~label).sum() / max((~label).sum(), 1) - 1
        if j > best_j:
            best, best_j = float(t), float(j)
    return best


def balanced_accuracy(score: np.ndarray, label: np.ndarray, threshold: float) -> float:
    score, label = np.asarray(score, dtype=float), np.asarray(label, dtype=bool)
    pred = score >= threshold
    return float(0.5 * ((pred & label).sum() / max(label.sum(), 1) + (~pred & ~label).sum() / max((~label).sum(), 1)))


def lin_ccc(x: np.ndarray, y: np.ndarray) -> float:
    """Lin's concordance correlation coefficient."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    cov = np.mean((x - x.mean()) * (y - y.mean()))
    return float(2 * cov / (x.var() + y.var() + (x.mean() - y.mean()) ** 2))
