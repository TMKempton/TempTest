import json
import sys
from pathlib import Path

from loguru import logger

from src.llm import LLM
from src.metrics import compute_temptest_metrics

if __name__ == "__main__":
    result_dir = Path(__file__).parent.parent / Path("results")
    result_dir.mkdir(parents=True, exist_ok=True)

    dataset = sys.argv[1]
    gen_model = sys.argv[2]
    score_model = sys.argv[3]
    gen_temp = float(sys.argv[4])
    score_temp = float(sys.argv[5])

    with open(
        f"./data/generated/{dataset}_{gen_model}_{gen_temp}.completions.json", "r"
    ) as f:
        data = json.load(f)["completions"]

    llm = LLM("/data/llms/" + score_model)

    results = {"real_temptest_metric_lists": [], "synthetic_temptest_metric_lists": []}
    for text in data:
        real_completion = text["real_completion"]
        synthetic_completion = text["synthetic_completion"]
        logger.info(f"Context: {text['context']}")
        logger.info(f"Real completion: {real_completion}")
        logger.info(f"Synthetic completion: {synthetic_completion}")

        real_temptest_metrics = compute_temptest_metrics(
            real_completion, llm=llm, temp=score_temp
        )
        synthetic_temptest_metrics = compute_temptest_metrics(
            synthetic_completion, llm=llm, temp=score_temp
        )

        results["real_temptest_metric_lists"].append(real_temptest_metrics)
        results["synthetic_temptest_metric_lists"].append(synthetic_temptest_metrics)

    with open(
        result_dir
        / f"unagg_{dataset}_gen{gen_model}_score{score_model}_gen{gen_temp}_score{score_temp}_temptest_results.json",
        "w",
    ) as f:
        f.write(json.dumps(results))

    logger.info("Completed experiment.")
