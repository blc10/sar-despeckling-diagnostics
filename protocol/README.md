# Pre-specified protocol of the real-data test

`PROTOKOL.json` fixes the analysis plan of the real SARLO-80 test (Section 2.9 of the paper). It was written before the
data were downloaded, on 2026-10-03 at 22:46:03 UTC, and is released unchanged so that its hash can be verified. The
file was written in Turkish; this page summarizes its content in English. Field names and method identifiers are in
English and are the same as in the result tables.

## Verifying the hash

```bash
sha256sum -c PROTOKOL.json.sha256
```

or, on any platform,

```bash
python -c "import hashlib; print(hashlib.sha256(open('protocol/PROTOKOL.json', 'rb').read()).hexdigest())"
```

The expected value is `ed92802dd601901d1232eb7477f7dd97c3f35e1b8a54b3b19889d840cc8f9f86`.

## Content

| Field | Content |
|---|---|
| `data.untouched_chunks`, `data.previously_used_shards` | SARLO-80 training chunks that had never been downloaded and the four shards used in earlier work |
| `data.listing_sha256`, `data.selection_seed`, `data.selected_shards` | Hash of the shard listing, the seed 20261008 and the two shards selected with it, with their sizes and LFS hashes |
| `data.record_inclusion` | All records with a complex `sar.npy` of at least 1024 × 1024 pixels, `sicd.xml` and `meta.json`; no selection by image quality |
| `data.primary_operation_sampling` | The primary set consists of records whose metadata mark them as downsampled; all records form the sensitivity analysis |
| `data.patch_grid`, `data.max_zero_fraction`, `data.normalization` | Nine 256 × 256 patches per record on a fixed grid, at most 1% zero pixels, complex values divided by the square root of the mean power |
| `methods` | Method groups, input scales of the networks with published weights calibrated under the SARLO-80 response, and the input preparation |
| `measures` | Measures computed on every patch; values are averaged within records, the ENL gain on a logarithmic scale, and the value of a method is the mean over records |
| `hypotheses` | H1–H3 with their decision rules, α = 0.05, 10,000 permutations and the seed 20261009 |
| `predictions` | Method means of the simulation with the SARLO-80 response and the simulated signs of the H3 relations; ENL gains are natural logarithms |
| `code_sha256` | Hashes of the code files of the original study that were used for the test |

### Hypotheses

* **H1.** The official CL-SAR weights (`cl_sar_default_official`, `cl_sar_t2_official`) do not collapse under white
  X-band speckle: for each weight, the median record-level ENL gain is at least 10.
* **H2.** The method profiles of the simulation with the SARLO-80 response transfer to real data: for each of `r`,
  `rf_jsd`, `rf_m`, `rf_tdm_d`, `rf_enl_gain` and `rf_peak_log_err`, the Spearman correlation between simulated and
  real method means, over all methods except the observation, is positive. One-sided permutation tests with Holm
  correction; ρ ≥ 0.5 is reported as strong.
* **H3.** The no-reference diagnostics `rf_jsd` and `rf_tdm_d` keep the sign of their simulated method-level relation
  with the strong-scatterer peak error `rf_peak_log_err`. One-sided permutation tests with Holm correction.

All results are reported regardless of their direction and size, and the protocol and code were not changed after
the download.

### Measure names

| Protocol | Result table | Measure |
|---|---|---|
| `r` | `r` | Ratio-edge measure |
| `rf_jsd` | `gamma_jsd` | Gamma Jensen–Shannon divergence of the ratio image |
| `rf_m` | `m_index` | M-index |
| `rf_tdm`, `rf_tdm_d` | `tdm`, `tdm_deviation` | TDM in its published orientation and its two-sided deviation D |
| `rf_enl_gain` | `enl_gain` | ENL of the estimate divided by that of the observation in homogeneous blocks |
| `rf_peak_log_err` | `peak_error` | Mean absolute log ratio of the 3 × 3 peaks of estimate and observation at strong scatterers |

The test is recomputed from `results/real_data/patch_measures.csv.gz` by `scripts/reproduce_results.py`.
