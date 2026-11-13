# Revision analyses (electronics-4674775, round 1)

Everything in this folder was added for the first revision. The released package (`sardiag/diagnostics.py`,
`metrics.py`, `statistics.py`, `scripts/`, `results/`, `protocol/`) is unchanged, so the submitted tables remain
reproducible; new measures live in `sardiag/revision.py`.

| Script | Reviewer comment | Input | Needs the authors' machine |
|---|---|---|---|
| `rev01_auc_sensitivity.py` | R3-1, R3-7 (Table 9 intervals) | released tables | no |
| `rev01b_sweep.py` | R3-1 (response sweep, TDM direction) | released tables | no |
| `rev02_regenerate.py`, `rev02_analyse.py` | R3-1 (level-aware error), R3-4 (normalized TDM divergence) | simulator | no |
| `rev03_coherent_targets.py` | R3-2, R1-3 | simulator | no |
| `rev04_temporal_median.py` | R3-3 | simulator, Monte Carlo | no |
| `rev05_peak_proxy.py` | R3-5, R1-2, R4 (detection) | output of `rev02_regenerate.py`, released tables | no |
| `rev06_real_data_uncertainty.py` | R3-7 | released real-data table, protocol | no |
| `rev07_methods.py` | R3-4, R3-5, R3-6, R1-2, R4 at the method level (generic adapter interface) | simulator + method adapters | **yes** |
| `rev08_site_sensitivity.py` | R3-7, R1-4 (five urban sites) | released tables + `semi_scenes.json` | no |
| `rev09_real_scatterers.py` | R3-5, R1-2 (what the rule selects on real data) | SARLO-80 patches of the package | package |
| `rev10_sim_methods.py`, `rev10_analyse.py` | R3-4, R3-5, R1-2, R1-3, R4 with the authors' own adapters, SARLO-response simulation | package (`SAR/`) | package |
| `rev14_mindex_mechanism.py`, `rev14_analyse.py` | R1-4: decomposition of the M-index, phantoms vs urban tiles | sardiag + semi_scenes | sardiag |
| `rev15_panel_reference_floor.py` | R3-3: split-half check of the temporal-median floor on the S1 panel stacks | panel evalset | numpy |
| `rev16_detection.py`, `rev16_analyse.py` | R4-2: CFAR retention of strong scatterers, real SARLO-80 and simulation (post hoc) | package + SARLO-80 patches | package |
| `rev11_real_methods.py`, `rev11_analyse.py` | R3-5, R3-7, H2/H3 with a contrast-conditioned proxy (post hoc) | package + SARLO-80 patches | package |

Run order (about 15 minutes on two cores):

    for r in s1 umbra sarlo; do python revision/rev02_regenerate.py --response $r; done
    python revision/rev01_auc_sensitivity.py
    python revision/rev01b_sweep.py
    python revision/rev02_analyse.py
    python revision/rev03_coherent_targets.py
    python revision/rev04_temporal_median.py
    python revision/rev05_peak_proxy.py
    python revision/rev06_real_data_uncertainty.py
    python -m unittest discover -s tests

`rev07_methods.py --adapters <file>` adds the revision columns for the despeckling methods; see its docstring for
the adapter interface. Without adapters it runs the observation and a 7 x 7 mean filter as a self-test.

`rev10` and `rev11` run the authors' adapters (`SAR/kodlar/a159_methods.py`) on CPU. `winpath_shim.py` maps the
Windows project root hard-coded in the PCGT-SAR loading chain to the local copy of the package, in process only.
Each row is checked against the released tables: rev10 matched all 5000 rows of a160_extension/sarlo; rev11 reproduced
the released proxy in 99.8% of 6816 rows (within 0.01) and the released H2/H3 statistics on the subsample.

All real-data analyses here are post hoc. The pre-specified protocol and its tests are not modified.
