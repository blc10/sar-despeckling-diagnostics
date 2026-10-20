"""Regenerate the controlled-degradation series and the diagnostic AUCs from the simulator.

    python scripts/controlled_degradations.py --response s1 [--phantoms 80] [--output series_s1.csv]

The 80 structural phantoms of the controlled-degradation set are simulated under the chosen response (s1, umbra or
sarlo), the 29 degraded outputs of each phantom are evaluated, and the AUCs of Table 6 are computed. With all 80
phantoms, the rows equal those in results/simulation/controlled_degradations.csv.gz.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sardiag import degradations, evaluation, simulation as sim  # noqa: E402
from reproduce_results import degradation_aucs  # noqa: E402


def series(response: str, phantoms: int) -> pd.DataFrame:
    rows = []
    for item in sim.phantom_set("dv", response, n=phantoms):
        index = item["index"]
        outputs = degradations.controlled_degradations(item["clean"].astype(np.float32), item["noisy"], item["coords"])
        for name, family, level, estimate in outputs:
            rec = evaluation.evaluate(item["noisy"], estimate, item["clean"], item["coords"], item["mask"],
                                      seed_key=degradations.seed_key(name, index))
            rows.append({"response": response, "phantom": index, "degradation": name, "family": family, "level": level,
                         "subswath": item["subswath"], **rec})
    frame = pd.DataFrame(rows)
    frame["tdm_negated"] = -frame["tdm"]
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--response", choices=("s1", "umbra", "sarlo"), default="s1")
    parser.add_argument("--phantoms", type=int, default=80)
    parser.add_argument("--draws", type=int, default=2000, help="phantom bootstrap draws for the intervals")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    frame = series(args.response, args.phantoms)
    if args.output:
        frame.to_csv(args.output, index=False)
    for key, v in degradation_aucs(frame, args.draws).items():
        ci = f"  [{v['ci'][0]:.3f}, {v['ci'][1]:.3f}]" if "ci" in v else ""
        print(f"{key:<36}{v['auc']:.3f}{ci}")


if __name__ == "__main__":
    main()
