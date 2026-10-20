# Result tables

All tables are gzip-compressed CSV files with one row per patch, or per degraded output, and can be read directly with
`pandas.read_csv`. Values are stored with ten significant digits. The files in `summary/` are written by
`scripts/reproduce_results.py`.

| File | Rows | Content | Paper |
|---|---|---|---|
| `simulation/patch_metrics.csv.gz` | 72,800 | Method outputs on the evaluation phantoms under the Sentinel-1 IW, Umbra and SARLO-80 responses and the response fitted to Sentinel-1 data | Sections 3.3, 3.5, 3.6 |
| `simulation/controlled_degradations.csv.gz` | 9,280 | Controlled degradations of the 80 structural phantoms under the same responses | Sections 3.2, 3.5 |
| `semi_synthetic/patch_metrics.csv.gz` | 6,675 | Method outputs on the semi-synthetic urban scenes | Section 3.7 |
| `semi_synthetic/controlled_degradations.csv.gz` | 3,219 | Controlled degradations of the semi-synthetic scenes | Section 3.7 |
| `response_sweep/patch_metrics.csv.gz` | 50,000 | Method outputs over the grid of oversampling and window parameters | Section 3.4 |
| `response_sweep/controlled_degradations.csv.gz` | 29,000 | Controlled degradations over the same grid | Section 3.4 |
| `real_data/patch_measures.csv.gz` | 76,500 | No-reference and observation-referenced measures on the real SARLO-80 patches | Section 3.8 |
| `summary/*.json` | | AUCs, method-level correlations, response sweep and hypothesis tests recomputed from the tables above | Tables 6, 7, 9, 10; Figure 6 |

## Columns

| Column | Description |
|---|---|
| `response` | System response: `s1` (Sentinel-1 IW), `umbra` (Umbra spotlight), `sarlo` (SARLO-80), `s1_fitted` (window fitted to real Sentinel-1 correlation) or `sweep` |
| `oversampling`, `window_a` | Oversampling ratio and window parameter of the response sweep |
| `phantom`, `phantom_class`, `subswath` | Phantom or scene identifier, phantom class and the Sentinel-1 subswath drawn for the phantom |
| `texture_shape`, `n_lines` | Gamma texture shape parameter and number of linear structures of the phantom |
| `method` | Method identifier (see below) |
| `degradation`, `family`, `level` | Controlled degradation, its family (`gauss`, `mean`, `edge`, `peak`, `radio`) and its parameter |
| `true_log_rmse` | Root-mean-square difference between the mean-normalized log estimate and log reference |
| `edge_error` | One minus the correlation of the gradient magnitudes of the log estimate and log reference |
| `peak_error` | Absolute logarithm of `point_ratio`; on real data, the mean absolute log ratio of the 3 × 3 peaks of estimate and observation at strong scatterers |
| `radiometric_error` | Absolute relative error of the mean intensity |
| `point_ratio` | Median over the point targets of the largest estimate value in the 3 × 3 neighborhood divided by the reference peak |
| `psnr`, `ssim` | Peak signal-to-noise ratio and structural similarity on the normalized intensity |
| `edge_preservation`, `enl_homogeneous` | Gradient correlation with the reference and ENL in the homogeneous region |
| `r`, `gamma_jsd`, `m_index` | Ratio-edge measure, Gamma Jensen–Shannon divergence and M-index of the ratio image |
| `tdm`, `tdm_contrast`, `tdm_variance`, `tdm_jsd`, `tdm_deviation` | TDM in its published orientation, its three components and the two-sided deviation D |
| `obs_*` | Measures computed as on real data, with the observation as the only reference; available for all simulations except the Sentinel-1 IW and Umbra runs |
| `enl_gain`, `scatterers` | Real data: ENL gain over homogeneous blocks and number of strong scatterers used for `peak_error` |
| `record`, `patch`, `city`, `operation_sampling`, `primary` | Real data: SARLO-80 record, patch, scene location, resampling operation and membership in the primary set |
| `input_scale` | Input scale of the networks with published weights |

The M-index is not meaningful for the unfiltered observation, whose ratio image is constant.

## Methods

| Identifier | Method |
|---|---|
| `identity` | Unfiltered observation |
| `lee_classical_7x7`, `kuan_7x7`, `frost_7x7`, `enhanced_lee_7x7`, `gamma_map_7x7`, `mean_filter_7x7` | Classical local filters with 7 × 7 windows |
| `lee_sigma_improved` | Improved Lee sigma filter |
| `mulog_bm3d` | MuLoG+BM3D |
| `sar2sar_official_direct`, `sar2sar_official_sub2` | SAR2SAR with published weights, applied directly and with ×2 subsampling |
| `merlin_s1tops_official`, `merlin_tsx_spotlight_official` | MERLIN with its Sentinel-1 TOPS and TerraSAR-X spotlight weights |
| `cl_sar_default_official`, `cl_sar_t2_official` | CL-SAR with its default and T2 weights |
| `temporal_unet_indomain`, `merlin_indomain` | U-Nets trained by the authors on Sentinel-1 data with a temporal-median target and with the MERLIN loss |
| `sds_sar_s1_reimpl` | Reimplementation of the SDS-SAR strategy trained on Sentinel-1 data |
| `guarded_anchor` | Anchor network of PCGT-SAR |
| `point_passthrough` | PCGT-SAR |
| `no_point_passthrough` | PCGT-SAR without point-target pass-through |
| `no_gamma_tangent`, `no_heat`, `no_selection`, `with_pareto` | Further PCGT-SAR ablation conditions |
| `sweep_<filter>_w<size>` | Window-size variants of the classical filters; not analyzed in the paper |
| `rabasar_multitemporal` | RABASAR on the 11 additional dates; not analyzed in the paper |
| `sar_cnn_reimpl`, `id_cnn_reimpl`, `speckle2void_reimpl`, `overcomplete_cnn_reimpl`, `sar_ddpm_lite_reimpl`, `sar2sar_n2n_reimpl` | Further reimplementations of earlier networks; not analyzed in the paper |

## License

The tables are released under the Creative Commons Attribution 4.0 license. The real-data measures are derived from
SARLO-80 and are distributed under its Creative Commons Attribution-ShareAlike 4.0 license.
