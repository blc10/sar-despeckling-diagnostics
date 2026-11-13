"""Figure 4 of the revised manuscript: AUC in the pooled and structural series and within the peak-attenuation family.

    python revision/fig4_auc_series.py <output.png>
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

R = json.loads((Path(__file__).with_name("results") / "rev01_auc_sensitivity.json").read_text())
RESP = [("s1", "Sentinel-1 IW", "#2463ab"), ("umbra", "Umbra spotlight", "#d9832b"), ("sarlo", "SARLO-80 (80 cm)", "#3a9a5b")]
DIAG = [("r", "r"), ("jsd", "Gamma JSD"), ("m_index", "M-index"), ("tdm_dev", "D"), ("tdm_pub", "TDM\n(published)")]
plt.rcParams.update({"font.size": 8, "savefig.dpi": 600})


def panel(ax, getter, diags, title, ylabel=None):
    x = np.arange(len(diags))
    w = 0.26
    for k, (key, name, col) in enumerate(RESP):
        vals = [getter(key, d) for d, _ in diags]
        a = np.array([v["auc"] for v in vals])
        lo = a - np.array([v["ci"][0] for v in vals])
        hi = np.array([v["ci"][1] for v in vals]) - a
        ax.bar(x + (k - 1) * w, a, w, color=col, label=name, yerr=[lo, hi], capsize=2.5,
               error_kw={"elinewidth": 1, "capthick": 1})
    ax.axhline(0.5, ls="--", color="0.4", lw=1)
    ax.set_xticks(x, [n for _, n in diags])
    ax.set_ylim(0.3, 0.95)
    ax.set_title(title, loc="left", fontsize=11)
    if ylabel:
        ax.set_ylabel(ylabel)


def main(out):
    fig = plt.figure(figsize=(7.2, 5.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1])
    a, b, c = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0])
    panel(a, lambda r, d: R[r]["auc"]["log_rmse|pooled"][d], DIAG, "(a) Pooled series", "AUC")
    panel(b, lambda r, d: R[r]["auc"]["log_rmse|structural"][d], DIAG, "(b) Structural series")
    panel(c, lambda r, d: R[r]["peak_within_family"][d], DIAG[:4], "(c) Within peak attenuation", "AUC within family")
    leg = fig.add_subplot(gs[1, 1]); leg.axis("off")
    h, l = a.get_legend_handles_labels()
    leg.legend(h, l, loc="center", frameon=False, fontsize=9)
    for ax in (a, b, c):
        ax.tick_params(axis="x", labelsize=7.5)
    fig.tight_layout()
    fig.savefig(out)


if __name__ == "__main__":
    main(sys.argv[1])
