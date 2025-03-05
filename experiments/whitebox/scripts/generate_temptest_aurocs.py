import json
import sys
from pathlib import Path

import numpy as np
from loguru import logger
from sklearn.metrics import roc_auc_score

OFFSET = 0

if __name__ == "__main__":
    dataset = sys.argv[1]
    model = sys.argv[2]
    gen_temp = float(sys.argv[3])
    score_temp = float(sys.argv[4])

    result_dir = Path(__file__).parent.parent / Path("results")
    result_dir.mkdir(parents=True, exist_ok=True)

    with open(
        result_dir
        / f"unagg_{dataset}_{model}_gen{gen_temp}_score{score_temp}_temptest_results.json",
        "r",
    ) as f:
        results = json.load(f)

    for num_tokens in [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300]:
        real_temptest_metrics = [
            np.mean(lst[OFFSET:num_tokens])
            for lst in results["real_temptest_metric_lists"]
        ]
        synthetic_temptest_metrics = [
            np.mean(lst[OFFSET:num_tokens])
            for lst in results["synthetic_temptest_metric_lists"]
        ]

        auroc = roc_auc_score(
            [1] * len(real_temptest_metrics) + [0] * len(synthetic_temptest_metrics),
            real_temptest_metrics + synthetic_temptest_metrics,
        )
        results["auroc"] = auroc

        logger.info(f"Num tokens: {num_tokens},AUROC: {auroc}")

        with open(
            result_dir
            / f"{dataset}_{model}_gen{gen_temp}_score{score_temp}_{num_tokens}_temptest_results.json",
            "w",
        ) as f:
            f.write(json.dumps(results))
