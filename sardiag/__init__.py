"""sardiag: validation of no-reference SAR despeckling diagnostics under known reflectivity.

Modules
-------
simulation    system-response-aware speckle simulation, phantoms and the temporal-median bias
diagnostics   ratio-edge measure r, Gamma JSD, M-index, TDM and its two-sided deviation D
metrics       true log error, edge preservation, point-target error, PSNR, SSIM, ENL
degradations  controlled degradations with known amounts of error
statistics    AUC, Spearman correlation, permutation tests, Holm correction, bootstrap, thresholds
evaluation    evaluate() combines the true errors and the diagnostics for one estimate
real_data     diagnostics and observation-referenced measures for real data
"""
from . import degradations, diagnostics, evaluation, metrics, real_data, simulation, statistics

__all__ = ["degradations", "diagnostics", "evaluation", "metrics", "real_data", "simulation", "statistics"]
__version__ = "1.0.0"
