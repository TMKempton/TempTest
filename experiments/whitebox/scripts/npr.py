# Step 1. Load data
# Step 2. Compute temp statistic on data
# Step 3. Compute AUROC
# Step 4. Write results
import json
import sys
from pathlib import Path

from loguru import logger
from sklearn.metrics import roc_auc_score

from src.llm import LLM
from src.metrics import compute_npr_metric

if __name__ == "__main__":
    result_dir = Path(__file__).parent.parent / Path("results")
    result_dir.mkdir(parents=True, exist_ok=True)

    dataset = sys.argv[1]
    model = sys.argv[2]
    temp = float(sys.argv[3])
    num_tokens = int(sys.argv[4])

    with open(
        f"./data/perturbations/{dataset}_{model}_{temp}.perturbations.json", "r"
    ) as f:
        data = json.load(f)["perturbations"]

    llm = LLM("/data/llms/" + model)

    results = {
        "real_npr_statistics": [],
        "synthetic_npr_statistics": [],
    }
    for text in data:
        real_perturbations = text["perturbed_real"]
        real_completion = text["real_completion"]
        synthetic_perturbations = text["perturbed_synthetic"]
        synthetic_completion = text["synthetic_completion"]
        logger.info(f"Real perturbations: {real_perturbations}")
        logger.info(f"Synthetic perturbations: {synthetic_perturbations}")

        real_npr_statisic = compute_npr_metric(
            completion=real_completion,
            perturbed_completions=real_perturbations,
            scoring_llm=llm,
            num_tokens=num_tokens,
        )
        synthetic_npr_statistic = compute_npr_metric(
            completion=synthetic_completion,
            perturbed_completions=synthetic_perturbations,
            scoring_llm=llm,
            num_tokens=num_tokens,
        )
        results["real_npr_statistics"].append(real_npr_statisic)
        results["synthetic_npr_statistics"].append(synthetic_npr_statistic)

        logger.info(f"Real NPR statistic: {real_npr_statisic}")
        logger.info(f"Synthetic NPR statistic: {synthetic_npr_statistic }")

    auroc = roc_auc_score(
        [0] * len(results["real_npr_statistics"])
        + [1] * len(results["synthetic_npr_statistics"]),
        results["real_npr_statistics"] + results["synthetic_npr_statistics"],
    )
    results["auroc"] = auroc
    with open(
        result_dir
        / f"{dataset}_{model}_gen{temp}_scoreNULL_{num_tokens}_npr_results.json",
        "w",
    ) as f:
        f.write(json.dumps(results))

    logger.info(f"Completed experiment. NPR AUROC: {auroc}")
