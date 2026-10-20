"""Evaluation of one estimate against a simulated scene: true errors and no-reference diagnostics."""
from __future__ import annotations

import numpy as np

from . import diagnostics, metrics

EPS = 1e-8


def evaluate(noisy: np.ndarray, estimate: np.ndarray, clean: np.ndarray, coords, mask: np.ndarray,
             seed_key: str = "x|patch") -> dict:
    """Return the true error measures and the no-reference diagnostics of an estimate.

    noisy: observed intensity; estimate: despeckled intensity; clean: reference X; coords: point-target positions;
    mask: homogeneous region; seed_key: key that fixes the random permutations of the M-index.
    """
    estimate = np.clip(np.asarray(estimate, dtype=np.float32), EPS, None)
    rec = {
        "true_log_rmse": metrics.true_log_rmse(estimate, clean),
        "psnr": metrics.psnr(estimate, clean),
        "ssim": metrics.ssim(estimate, clean),
        "edge_preservation": metrics.edge_preservation(estimate, clean),
        "peak_error": metrics.peak_error(estimate, clean, coords),
        "enl_homogeneous": metrics.enl_homogeneous(estimate, mask),
        "r": diagnostics.ratio_edge(noisy, estimate),
        "gamma_jsd": diagnostics.gamma_jsd(diagnostics.ratio_image(np.clip(noisy, EPS, None), estimate)),
        "m_index": diagnostics.m_index(noisy, estimate, seed_key=seed_key),
    }
    rec["edge_error"] = 1.0 - rec["edge_preservation"]
    t = diagnostics.tdm(noisy, estimate, mask)
    rec.update({"tdm": t["tdm"], "tdm_deviation": t["tdm_deviation"], "tdm_contrast": t["contrast"],
                "tdm_variance": t["variance"], "tdm_jsd": t["jsd"]})
    return rec
