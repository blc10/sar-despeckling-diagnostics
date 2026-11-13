"""Shared helpers of the revision analyses: fast phantom-cluster bootstrap of AUC and rank statistics."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sardiag import statistics as st  # noqa: E402

RESULTS = ROOT / "results"
OUT = ROOT / "revision" / "results"
SEED = 20261110
DIAG = {"r": "r", "jsd": "gamma_jsd", "m_index": "m_index", "tdm_dev": "tdm_deviation", "tdm_pub": "tdm_negated"}
UNDEGRADED = ("gauss_sigma_0.0", "edge_keep_1.0", "peak_keep_1.0", "radio_1.0")


def load(rel: str) -> pd.DataFrame:
    frame = pd.read_csv(RESULTS / rel, low_memory=False)
    if "tdm" in frame:
        frame["tdm_negated"] = -frame["tdm"]
    return frame


def auc(score, label) -> float:
    keep = np.isfinite(score)
    return st.auc(score[keep], label[keep])


def spearman(a, b) -> float:
    keep = np.isfinite(a) & np.isfinite(b)
    if keep.sum() < 4 or np.std(a[keep]) < 1e-12 or np.std(b[keep]) < 1e-12:
        return float("nan")
    return float(np.corrcoef(rankdata(a[keep]), rankdata(b[keep]))[0, 1])


def cluster_draws(groups: np.ndarray, draws: int, seed: int = SEED):
    """Yield row-index arrays for resampling whole clusters (phantoms, scenes, cities) with replacement."""
    rng = np.random.default_rng(seed)
    keys, inverse = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inverse == k) for k in range(len(keys))]
    for _ in range(draws):
        yield np.concatenate([members[k] for k in rng.integers(0, len(keys), len(keys))])


def boot(groups, draws, *stats, seed: int = SEED) -> np.ndarray:
    """Bootstrap several statistics f(index_array) on the same cluster draws; returns (draws, len(stats))."""
    return np.array([[f(idx) for f in stats] for idx in cluster_draws(np.asarray(groups), draws, seed)])


def ci(values) -> list[float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))] if values.size else [float("nan")] * 2
