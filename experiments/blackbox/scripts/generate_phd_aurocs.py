import json
import os
from pathlib import Path

from sklearn.metrics import roc_auc_score

for f in os.listdir(Path(__file__).parent.parent / "results"):
    if "phd" in f:
        with open(Path(__file__).parent.parent / "results" / f, "r") as fl:
            results = json.load(fl)

            auroc = roc_auc_score(
                [1] * len(results["real_temptest_metric_lists"])
                + [0] * len(results["synthetic_temptest_metric_lists"]),
                results["real_temptest_metric_lists"]
                + results["synthetic_temptest_metric_lists"],
            )
            results["auroc"] = auroc

        with open(Path(__file__).parent.parent / "results" / f, "w") as fl:
            fl.write(json.dumps(results))
