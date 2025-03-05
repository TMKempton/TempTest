import json
import sys
from pathlib import Path

from loguru import logger
from sklearn.metrics import roc_auc_score

from src.llm import LLM
from src.metrics import (
    compute_baseline_metric,
    get_entropy,
    get_likelihood,
    get_logrank,
    get_rank,
)

CRITERION_MAP = {
    "entropy": get_entropy,
    "likelihood": get_likelihood,
    "logrank": get_logrank,
    "rank": get_rank,
}

if __name__ == "__main__":
    result_dir = Path(__file__).parent.parent / Path("results")
    result_dir.mkdir(parents=True, exist_ok=True)

    dataset = sys.argv[1]
    gen_model = sys.argv[2]
    score_model = sys.argv[3]
    temp = float(sys.argv[4])
    num_tokens = int(sys.argv[5])

    with open(
        f"./data/generated/{dataset}_{gen_model}_{temp}.completions.json", "r"
    ) as f:
        data = json.load(f)["completions"]

    llm = LLM("/data/llms/" + score_model)
    for baseline_criterion in ["rank", "logrank", "entropy", "likelihood"]:
        results = {
            f"real_{baseline_criterion}_statistics": [],
            f"synthetic_{baseline_criterion}_statistics": [],
        }
        for text in data:
            real_completion = text["real_completion"]
            synthetic_completion = text["synthetic_completion"]
            logger.info(f"Context: {text['context']}")
            logger.info(f"Real completion: {real_completion}")
            logger.info(f"Synthetic completion: {synthetic_completion}")

            real_baseline_statisic = compute_baseline_metric(
                real_completion,
                scoring_llm=llm,
                criterion=CRITERION_MAP[baseline_criterion],
                num_tokens=num_tokens,
            )
            synthetic_baseline_statistic = compute_baseline_metric(
                synthetic_completion,
                scoring_llm=llm,
                criterion=CRITERION_MAP[baseline_criterion],
                num_tokens=num_tokens,
            )
            results[f"real_{baseline_criterion}_statistics"].append(
                real_baseline_statisic
            )
            results[f"synthetic_{baseline_criterion}_statistics"].append(
                synthetic_baseline_statistic
            )

            logger.info(
                f"Real {baseline_criterion} statistic: {real_baseline_statisic}"
            )
            logger.info(
                f"Synthetic {baseline_criterion} statistic: {synthetic_baseline_statistic}"
            )

        auroc = roc_auc_score(
            [0] * len(results[f"real_{baseline_criterion}_statistics"])
            + [1] * len(results[f"synthetic_{baseline_criterion}_statistics"]),
            results[f"real_{baseline_criterion}_statistics"]
            + results[f"synthetic_{baseline_criterion}_statistics"],
        )
        results["auroc"] = auroc
        with open(
            result_dir
            / f"{dataset}_gen{gen_model}_score{score_model}_gen{temp}_scoreNULL_{num_tokens}_{baseline_criterion}_results.json",
            "w",
        ) as f:
            f.write(json.dumps(results))

        logger.info(f"Completed experiment. {baseline_criterion} AUROC: {auroc}")
