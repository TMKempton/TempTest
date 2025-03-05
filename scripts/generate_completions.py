# Generate 300 completions based on the first 30 tokens.
import json
import sys

from loguru import logger

from src.llm import LLM

CONTEXT_NUM_TOKENS = 30

if __name__ == "__main__":
    model = sys.argv[1]
    dataset = sys.argv[2]
    temp = float(sys.argv[3])

    logger.info(
        f"Processing dataset {dataset} and generation model for sampling {model} with temperature {temp}"
    )

    llm = LLM("/data/llms/" + model)

    with open(f"./data/raw_data/{dataset}.raw_data.json", "r") as f:
        data = json.load(f)

    results = {"completions": []}
    for text in data["original"]:
        encoded_text = llm.encode(text)
        logger.info(f"Encoded text shape: {encoded_text.shape}")
        context_tokens = encoded_text[:, :CONTEXT_NUM_TOKENS]
        real_completion_tokens = encoded_text[:, CONTEXT_NUM_TOKENS:]
        context = llm.tokenizer.decode(context_tokens[0])
        real_completion = llm.tokenizer.decode(real_completion_tokens[0])

        logger.info(f"Context: {context}")
        logger.info(f"Context token length: {llm.encode(context).shape[1]}")
        synthetic_completion = llm.fast_sampling(
            context,
            temp=temp,
            num_new_tokens=300,
            num_return_sequences=1,
        )[0]["generated_text"]

        results["completions"].append(
            {
                "context": context,
                "synthetic_completion": synthetic_completion,
                "real_completion": real_completion,
                "original_text": text,
            }
        )

    with open(
        f"./data/generated/{dataset}_{model}_{temp}.completions.json",
        "w",
    ) as f:
        f.write(
            json.dumps(
                results,
                indent=4,
            )
        )
