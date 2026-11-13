"""Summary of rev14_mindex_mechanism.py: AUC of the M-index terms and block-selection variants (R1-4)."""
import json, sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, auc  # noqa: E402
COLS = ["m_index", "m_first_order", "m_first_enl", "m_first_mean", "m_structure", "m_first_f005", "m_first_f020",
        "m_first_f050", "m_first_f100", "m_first_oracle", "m_index_oracle"]
f = pd.read_csv(OUT / "rev14_mindex_components.csv.gz")
out = {}
for st in ("phantom", "urban"):
    for ser in ("pooled", "structural"):
        for r in ("s1", "umbra", "sarlo"):
            b = f[(f.scene_type == st) & (f.response == r) & ((f.family != "radio") if ser == "structural" else True)]
            lab = (b.true_log_rmse > b.true_log_rmse.median()).to_numpy()
            out[f"{st}|{ser}|{r}"] = {c: auc(b[c].to_numpy(float), lab) for c in COLS}
out["clean_cv_in_selected_median"] = {f"{k[0]}|{k[1]}": v for k, v in f.groupby(["scene_type", "response"]).clean_cv_in_selected.median().items()}
bs = pd.read_csv(OUT / "rev14_block_statistics.csv")
out["block_statistics_median"] = {f"{k[0]}|{k[1]}": row for k, row in bs.groupby(["kind", "resp"]).median().round(4).to_dict("index").items()}
(OUT / "rev14_mindex_mechanism.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(out["clean_cv_in_selected_median"]))
