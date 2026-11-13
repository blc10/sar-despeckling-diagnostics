"""Render Supplementary Tables S7-S11 of the revision from the JSON results."""
import json
import sys
from pathlib import Path

R = Path(__file__).with_name("results")
J = lambda name: json.loads((R / name).read_text())
RESP = [("s1", "Sentinel-1 IW"), ("umbra", "Umbra spotlight"), ("sarlo", "SARLO-80")]
DIAG = [("r", "$r$"), ("jsd", "JSD"), ("m_index", "M"), ("tdm_dev", "D"), ("tdm_pub", "TDM")]
METHOD_NAMES = {
    "identity": "Observation (no filtering)",
    "lee_classical_7x7": "Lee 7 $\\times$ 7", "kuan_7x7": "Kuan 7 $\\times$ 7",
    "frost_7x7": "Frost 7 $\\times$ 7", "enhanced_lee_7x7": "Enhanced Lee 7 $\\times$ 7",
    "gamma_map_7x7": "Gamma-MAP 7 $\\times$ 7", "mean_filter_7x7": "Mean 7 $\\times$ 7",
    "lee_sigma_improved": "Improved Lee sigma", "mulog_bm3d": "MuLoG+BM3D",
    "merlin_indomain": "MERLIN-loss U-Net *", "temporal_unet_indomain": "Temporal-median U-Net *",
    "sds_sar_s1_reimpl": "SDS-SAR reimplementation *", "guarded_anchor": "PCGT-SAR anchor *",
    "point_passthrough": "PCGT-SAR", "no_point_passthrough": "PCGT-SAR without pass-through",
    "no_gamma_tangent": "PCGT-SAR without Gamma tangent", "no_heat": "PCGT-SAR without heat term",
    "no_selection": "PCGT-SAR without selection", "with_pareto": "PCGT-SAR with Pareto term",
    "merlin_s1tops_official": "MERLIN (Sentinel-1 TOPS)",
    "merlin_tsx_spotlight_official": "MERLIN (TerraSAR-X spotlight)",
    "sar2sar_official_direct": "SAR2SAR", "sar2sar_official_sub2": "SAR2SAR $\\times$2",
    "cl_sar_default_official": "CL-SAR (default)", "cl_sar_t2_official": "CL-SAR (T2)",
}


def f(x, d=3):
    s = f"{x:.{d}f}"
    return s.replace("-", "$-$") if x < 0 else s


def fc(v, d=3):
    return f"{f(v[0], d)} to {f(v[1], d)}"


def table(caption, label, colspec, header, rows, wide=False, size="\\footnotesize"):
    out = ["\\begin{table}[H]", f"\\caption{{{caption}\\label{{{label}}}}}"]
    if wide:
        out.append("\\begin{adjustwidth}{-\\extralength}{0cm}")
    out.append(size)
    out.append(f"\\begin{{tabularx}}{{{'\\fulllength' if wide else '\\textwidth'}}}{{{colspec}}}")
    out += ["\\toprule", header + "\\\\", "\\midrule"] + rows + ["\\bottomrule", "\\end{tabularx}"]
    if wide:
        out.append("\\end{adjustwidth}")
    out.append("\\end{table}")
    return "\n".join(out)


def s7():
    a1, a2 = J("rev01_auc_sensitivity.json"), J("rev02_normalized_tdm.json")
    series = [("pooled", "Pooled, five families"), ("pooled_dedup", "Pooled, one undegraded output"),
              ("structural", "Structural (no radiometric scaling)"),
              ("structural_dedup", "Structural, one undegraded output"),
              ("drop_gauss", "Pooled without Gaussian blur"), ("drop_mean", "Pooled without mean filter"),
              ("drop_edge", "Pooled without edge smearing"), ("drop_peak", "Pooled without peak attenuation"),
              ("drop_radio", "Pooled without radiometric scaling")]
    rows = []
    for key, name in RESP:
        rows.append(f"\\multicolumn{{7}}{{l}}{{\\textit{{{name} response}}}}\\\\")
        for s, label in series:
            b = a1[key]["auc"][f"log_rmse|{s}"]
            vals = " & ".join(f(b[d]["auc"]) for d, _ in DIAG)
            dd = b["jsd_minus_r"]
            rows.append(f"{label} & {vals} & {f(dd['diff'])} [{fc(dd['ci'])}]\\\\")
        b = a2[key]["auc_level_aware_pooled"]
        vals = " & ".join(f(b[d]["auc"]) for d, _ in DIAG)
        rows.append(f"Level-aware log error, pooled & {vals} & --\\\\")
        b = a1[key]["radiometric_axis"]
        vals = " & ".join(f(b[d]["auc"]) for d, _ in DIAG)
        rows.append(f"Within radiometric scaling & {vals} & --\\\\")
        if key != "sarlo":
            rows.append("\\midrule")
    cap = ("AUC of the diagnostics under controlled degradations on 80 phantoms for different definitions of the "
           "series and of the error. Except for the last two rows of each block, the label is a true log error "
           "(Equation (7)) above the median of the series. ``One undegraded output'' keeps one of the four "
           "undegraded outputs that the families share. The level-aware log error omits the mean normalization, "
           "so radiometric scaling counts as an error. The within-family row ranks the radiometric-scaling outputs "
           "by the size of the scaling. The last column gives the difference JSD $-$ $r$ with a 95\\% phantom-level "
           "bootstrap CI. M: M-index; D: two-sided deviation of TDM; TDM: published orientation.")
    hdr = ("\\textbf{Series} & \\textbf{$r$} & \\textbf{JSD} & \\textbf{M} & \\textbf{D} & \\textbf{TDM} & "
           "\\textbf{JSD $-$ $r$}")
    return table(cap, "tab:S7", "P{5.0cm}CCCCCP{3.3cm}", hdr, rows, wide=True)


def s8():
    a, m = J("rev02_normalized_tdm.json"), J("rev10_sim_methods.json")["method_level"]
    rows = []
    for key, name in RESP:
        b = a[key]
        dt = b["divergence_term"]
        cell = lambda blk, d: f(b[blk][d]["auc"] if isinstance(b[blk][d], dict) else b[blk][d])
        rows.append(f"{name} & {f(dt['median_ratio_released_over_normalized'], 1)} & "
                    f"{f(dt['spearman_released_vs_normalized'])} & "
                    f"{cell('auc_equation7_pooled', 'tdm_dev')} / {cell('auc_equation7_pooled', 'tdm_dev_norm')} & "
                    f"{cell('auc_equation7_structural', 'tdm_dev')} / {cell('auc_equation7_structural', 'tdm_dev_norm')} & "
                    f"{cell('auc_radiometric_within_family', 'tdm_dev')} / {cell('auc_radiometric_within_family', 'tdm_dev_norm')} & "
                    f"{cell('auc_equation7_pooled', 'tdm_pub')} / {cell('auc_equation7_pooled', 'tdm_pub_norm')}\\\\")
    cap = ("Released and normalized divergence term of TDM on the 80 degradation phantoms. Ratio: median of the "
           "released divergence term over the normalized one; $\\rho$: their Spearman correlation over outputs. "
           "AUC cells give the value with the released / the normalized term. The released term never reached its "
           "cap. At the method level (24 methods, 200 phantoms, SARLO-80 response), the Spearman correlation with "
           f"the true log error was {f(m['tdm|true_log_rmse'], 2)} for the published TDM and "
           f"{f(m['tdm_norm|true_log_rmse'], 2)} with the normalized term, and {f(m['tdm_dev|true_log_rmse'], 2)} "
           f"and {f(m['tdm_dev_norm|true_log_rmse'], 2)} for D.")
    hdr = ("\\textbf{Response} & \\textbf{Ratio} & \\textbf{$\\rho$} & \\textbf{D, pooled} & "
           "\\textbf{D, structural} & \\textbf{D, radiometric} & \\textbf{TDM, pooled}")
    return table(cap, "tab:S8", "P{2.6cm}CCCCCC", hdr, rows, wide=True)


def s9():
    t = J("rev13_threshold_transfer.json")
    nm = dict(RESP)
    short = {"s1": "S1", "umbra": "Umbra", "sarlo": "SARLO-80"}
    rows = []
    for d, dn in DIAG[:4]:
        for src, _ in RESP:
            for tgt, _ in RESP:
                v = t["diagnostics"][f"{d}|{src}->{tgt}"]
                loss = t["transfer_loss"].get(f"{d}|{src}->{tgt}")
                lc = f"{f(loss['loss'])} [{fc(loss['ci'])}]" if loss else "--"
                rows.append(f"{dn} & {short[src]} $\\rightarrow$ {short[tgt]} & {v['threshold']:.4g} & "
                            f"{f(v['balanced_accuracy'])} [{fc(v['ci_balanced_accuracy'])}] & {lc}\\\\")
        rows.append("\\midrule")
    rows.append("\\multicolumn{5}{l}{\\textit{Pass rate of the point-target criterion, Equation (9)}}\\\\")
    for src, _ in RESP:
        for tgt, _ in RESP:
            v = t["point_rule"][f"{src}->{tgt}"]
            rows.append(f"Peak & {short[src]} $\\rightarrow$ {short[tgt]} & {f(v['tau'])} & "
                        f"{f(v['pass_rate'])} [{fc(v['ci_pass_rate'])}] & --\\\\")
    cap = ("Uncertainty of the threshold transfer. For the diagnostics, the Youden threshold was set on the 40 "
           "even-numbered development phantoms of the source response and applied to the 40 odd-numbered test "
           "phantoms of the target response; the loss is the within-response balanced accuracy of the target "
           "minus the transferred one. For the point-target criterion, $\\tau$ was set on the 200 even-numbered "
           "evaluation phantoms and the pass rate of the unfiltered observation is given on the 200 odd-numbered "
           "phantoms (nominal 0.95). Intervals are 95\\% bootstrap CIs over development and test phantoms, with "
           "the threshold re-estimated in every draw. D: two-sided deviation of TDM.")
    hdr = ("\\textbf{Measure} & \\textbf{Source $\\rightarrow$ target} & \\textbf{Threshold} & "
           "\\textbf{Balanced accuracy} & \\textbf{Loss}")
    return table(cap, "tab:S9", "P{1.3cm}P{3.4cm}CCC", hdr, rows, wide=True, size="\\scriptsize")


def s10():
    s, m = J("rev08_site_sensitivity.json"), J("rev12_reflectivity_map.json")
    rows = []
    for key, name in RESP:
        rows.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{name} response}}}}\\\\")
        b = s[key]
        rows.append("All sites, MuLoG+BM3D map & " + " & ".join(
            f"{f(b['all'][d])} [{fc(b['site_bootstrap_ci'][d])}]" for d, _ in DIAG) + "\\\\")
        lo = b["leave_one_site_out"]
        rows.append("Leave one site out, range & " + " & ".join(
            f"{f(min(v[d] for v in lo.values()))} to {f(max(v[d] for v in lo.values()))}" for d, _ in DIAG) + "\\\\")
        for ser, sn in [("pooled", "pooled"), ("structural", "structural")]:
            c = m[f"{key}|{ser}"]
            rows.append(f"Temporal-mean map, {sn} & " + " & ".join(f(c["temporal_mean"][d]) for d, _ in DIAG) + "\\\\")
            rows.append(f"Difference to MuLoG+BM3D, {sn} & " + " & ".join(
                f"{f(c['temporal_mean'][d] - c['mulog_bm3d'][d])} [{fc(c['difference_ci'][d])}]" for d, _ in DIAG) + "\\\\")
        ml = b["method_level_all"]
        lml = b["method_level_leave_one_site_out"]
        rows.append(f"Method level, $\\rho$ (19 methods) & " + " & ".join(
            f"{f(ml[d])} ({f(min(v[d] for v in lml.values()), 2)} to {f(max(v[d] for v in lml.values()), 2)})"
            for d, _ in DIAG) + "\\\\")
        if key != "sarlo":
            rows.append("\\midrule")
    t = s["tiles"]
    tiles = ", ".join(f"{k} {v}/{t['evaluation'][k]}" for k, v in t["degradation"].items())
    cap = ("Site sensitivity and reflectivity maps of the semi-synthetic scenes. AUC under controlled degradations "
           "with the true log error as target; intervals are 95\\% bootstrap CIs over sites (first row) or tiles "
           "(differences). The temporal-mean map is the arithmetic mean of the same 10--12 dates without "
           "despeckling. The method-level row gives the Spearman correlation between the mean diagnostic and the "
           "mean true error on the evaluation tiles, with the range when one site is left out in parentheses. "
           f"Degradation/evaluation tiles per site: {tiles}. M: M-index; D: two-sided deviation of TDM; TDM: "
           "published orientation.")
    hdr = ("\\textbf{Analysis} & \\textbf{$r$} & \\textbf{JSD} & \\textbf{M} & \\textbf{D} & \\textbf{TDM}")
    return table(cap, "tab:S10", "P{3.6cm}CCCCC", hdr, rows, wide=True, size="\\scriptsize")


def s11():
    j = J("rev10_sim_methods.json")
    pm, ml = j["per_method"], j["method_level"]
    order = sorted(pm, key=lambda k: pm[k]["peak_targetwise"])
    rows = []
    for k in order:
        v = pm[k]
        rows.append(f"{METHOD_NAMES.get(k, k)} & {f(v['true_log_rmse'])} & {f(v['peak_targetwise'])} & "
                    f"{f(v['peak_signed'])} & {f(v['proxy_released'])} & {f(v['proxy_true_target_share'], 2)} & "
                    f"{f(v['proxy_ge14'])} & {f(v['proxy_ge14_true_share'], 2)} & {f(v['det_pd'], 2)} & "
                    f"{v['jsd']:.4f} & {f(v['log_enl'], 2)}\\\\")
    cap = ("Validation of the observation-referenced peak error and point-target detection on 200 phantoms under "
           "the SARLO-80 response, sorted by the target-wise peak error. log-RMSE: true log error; Peak: target-wise "
           "absolute peak error; Signed: target-wise signed peak error (negative: attenuation); Proxy: released "
           "observation-referenced peak error, with the share of selected scatterers that are simulated targets; "
           "Proxy 14: the same with selection restricted to an observed contrast of at least 14 dB; $P_{d}$: "
           "detection probability of the simulated targets with a cell-averaging CFAR detector at a false-alarm "
           "probability of $10^{-3}$; log ENL: logarithm of the ENL in homogeneous blocks. Spearman correlations "
           f"over the 24 filtering methods: Proxy vs Peak {f(ml['proxy_released|peak_targetwise'])} "
           f"({f(ml['proxy_released|peak_targetwise|given_log_enl'])} given log ENL), Proxy 14 vs Peak "
           f"{f(ml['proxy_ge14|peak_targetwise'])} ({f(ml['proxy_ge14|peak_targetwise|given_log_enl'])}), JSD vs Peak "
           f"{f(ml['jsd|peak_targetwise'])} ({f(ml['jsd|peak_targetwise|given_log_enl'])}), loss of $P_{{d}}$ vs Peak "
           f"{f(ml['peak_targetwise|det_pd_loss'])}, vs JSD {f(ml['jsd|det_pd_loss'])}, vs M-index "
           f"{f(ml['m_index|det_pd_loss'])} and vs D {f(ml['tdm_dev|det_pd_loss'])}. An asterisk marks networks "
           "trained by the authors on Sentinel-1 data.")
    hdr = ("\\textbf{Method} & \\textbf{log-RMSE} & \\textbf{Peak} & \\textbf{Signed} & \\textbf{Proxy} & "
           "\\textbf{Share} & \\textbf{Proxy 14} & \\textbf{Share 14} & \\textbf{$P_{d}$} & \\textbf{JSD} & "
           "\\textbf{log ENL}")
    return table(cap, "tab:S11", "P{4.2cm}CCCCCCCCCC", hdr, rows, wide=True, size="\\scriptsize")


SECTIONS = [
    ("Sensitivity of the AUC to the Error Definition and the Degradation Mixture",
     "Table~\\ref{tab:S7} complements Table 6 of the main text.", s7),
    ("Released and Normalized Divergence Term of TDM",
     "Table~\\ref{tab:S8} compares the divergence convention of the reference implementation of TDM with a "
     "normalized Jensen--Shannon divergence on probability masses.", s8),
    ("Uncertainty of the Threshold Transfer",
     "Table~\\ref{tab:S9} gives the confidence intervals of Table 8 of the main text and the losses from "
     "transferring thresholds between responses.", s9),
    ("Site Sensitivity and Reflectivity Maps of the Semi-Synthetic Scenes",
     "Table~\\ref{tab:S10} complements Table 9 of the main text.", s10),
    ("Validation of the Observation-Referenced Peak Error and Point-Target Detection",
     "Table~\\ref{tab:S11} gives the method-level values behind Section 3.3 of the main text.", s11),
]

if __name__ == "__main__":
    parts = []
    for title, intro, fn in SECTIONS:
        parts += [f"\\section{{{title}}}", "", intro, "", fn(), ""]
    out = "\n".join(parts)
    Path(sys.argv[1] if len(sys.argv) > 1 else "supplementary_s7_s11.tex").write_text(out)
    print(len(out))
