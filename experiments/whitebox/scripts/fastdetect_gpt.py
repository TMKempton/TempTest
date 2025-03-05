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
from src.metrics import compute_fastdetect_gpt_metric

if __name__ == "__main__":
    result_dir = Path(__file__).parent.parent / Path("results")
    result_dir.mkdir(parents=True, exist_ok=True)

    dataset = sys.argv[1]
    model = sys.argv[2]
    temp = float(sys.argv[3])
    num_tokens = int(sys.argv[4])

    with open(f"./data/generated/{dataset}_{model}_{temp}.completions.json", "r") as f:
        data = json.load(f)["completions"]

    llm = LLM("/data/llms/" + model)

    results = {
        "real_fastdetect_gpt_statistics": [],
        "synthetic_fastdetect_gpt_statistics": [],
    }
    for text in data:
        real_completion = text["real_completion"]
        synthetic_completion = text["synthetic_completion"]
        logger.info(f"Context: {text['context']}")
        logger.info(f"Real completion: {real_completion}")
        logger.info(f"Synthetic completion: {synthetic_completion}")

        real_fastdetect_gpt_statisic = compute_fastdetect_gpt_metric(
            real_completion, reference_llm=llm, scoring_llm=llm, num_tokens=num_tokens
        )
        synthetic_fastdetect_gpt_statistic = compute_fastdetect_gpt_metric(
            synthetic_completion,
            reference_llm=llm,
            scoring_llm=llm,
            num_tokens=num_tokens,
        )
        results["real_fastdetect_gpt_statistics"].append(real_fastdetect_gpt_statisic)
        results["synthetic_fastdetect_gpt_statistics"].append(
            synthetic_fastdetect_gpt_statistic
        )

        logger.info(f"Real FastDetectGPT statistic: {real_fastdetect_gpt_statisic}")
        logger.info(
            f"Synthetic FastDetectGPT statistic: {synthetic_fastdetect_gpt_statistic}"
        )

    auroc = roc_auc_score(
        [0] * len(results["real_fastdetect_gpt_statistics"])
        + [1] * len(results["synthetic_fastdetect_gpt_statistics"]),
        results["real_fastdetect_gpt_statistics"]
        + results["synthetic_fastdetect_gpt_statistics"],
    )
    results["auroc"] = auroc
    with open(
        result_dir
        / f"{dataset}_{model}_gen{temp}_scoreNULL_{num_tokens}_fastdetect_gpt_results.json",
        "w",
    ) as f:
        f.write(json.dumps(results))

    logger.info(f"Completed experiment. FastDetectGPT AUROC: {auroc}")
