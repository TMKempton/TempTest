import json
import sys
from pathlib import Path

from loguru import logger

from src.dipper import DipperParaphraser

if __name__ == "__main__":
    data_dir = Path("./data/generated")
    output_dir = Path("./data/paraphrased")
    output_dir.mkdir(exist_ok=True, parents=True)
    gen_data = sys.argv[1]
    dp = DipperParaphraser()
    logger.info(f"Processing file {gen_data}")
    with open(data_dir / gen_data, "r") as f:
        data = json.load(f)

    for idx_inner, comp in enumerate(data["completions"]):
        logger.info(f"Processing completion {idx_inner} of {len(data['completions'])}")
        input_text = comp["synthetic_completion"]
        paraphrased = dp.paraphrase(
            input_text,
            lex_diversity=60,
            order_diversity=0,
            prefix="",
            do_sample=True,
            top_p=0.75,
            top_k=None,
            max_length=150,
        )
        comp["paraphrased_completion"] = paraphrased

    with open(output_dir / gen_data, "w") as f:
        f.write(json.dumps(data, indent=4))
