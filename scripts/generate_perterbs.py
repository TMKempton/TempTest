import argparse
import json
import re
from pathlib import Path

import numpy as np
import torch
import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.hardware import DEVICE

pattern = re.compile(r"<extra_id_\d+>")


def load_mask_model(model_path="/data/llms/t5-3b"):
    """
    From: https://github.com/baoguangsheng/fast-detect-gpt/blob/main/scripts/detect_gpt.py
    """
    mask_model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
    mask_model = mask_model.to(DEVICE)
    return mask_model


def load_mask_tokenizer(model_path="/data/llms/t5-3b", max_length=512):
    """
    From: https://github.com/baoguangsheng/fast-detect-gpt/blob/main/scripts/detect_gpt.py
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path, model_max_length=max_length)
    return tokenizer


def tokenize_and_mask(text, span_length, pct, ceil_pct=False):
    """
    From: https://github.com/baoguangsheng/fast-detect-gpt/blob/main/scripts/detect_gpt.py
    """
    buffer_size = 1
    tokens = text.split(" ")
    mask_string = "<<<mask>>>"

    n_spans = pct * len(tokens) / (span_length + buffer_size * 2)
    if ceil_pct:
        n_spans = np.ceil(n_spans)
    n_spans = int(n_spans)

    n_masks = 0
    while n_masks < n_spans:
        start = np.random.randint(0, len(tokens) - span_length)
        end = start + span_length
        search_start = max(0, start - buffer_size)
        search_end = min(len(tokens), end + buffer_size)
        if mask_string not in tokens[search_start:search_end]:
            tokens[start:end] = [mask_string]
            n_masks += 1

    # replace each occurrence of mask_string with <extra_id_NUM>, where NUM increments
    num_filled = 0
    for idx, token in enumerate(tokens):
        if token == mask_string:
            tokens[idx] = f"<extra_id_{num_filled}>"
            num_filled += 1
    assert num_filled == n_masks, f"num_filled {num_filled} != n_masks {n_masks}"
    text = " ".join(tokens)
    return text


def count_masks(texts):
    """
    From: https://github.com/baoguangsheng/fast-detect-gpt/blob/main/scripts/detect_gpt.py
    """
    return [
        len([x for x in text.split() if x.startswith("<extra_id_")]) for text in texts
    ]


# replace each masked span with a sample from T5 mask_model
def replace_masks(args, mask_model, mask_tokenizer, texts):
    """
    From: https://github.com/baoguangsheng/fast-detect-gpt/blob/main/scripts/detect_gpt.py
    """
    n_expected = count_masks(texts)
    stop_id = mask_tokenizer.encode(f"<extra_id_{max(n_expected)}>")[0]
    tokens = mask_tokenizer(texts, return_tensors="pt", padding=True).to(DEVICE)
    outputs = mask_model.generate(
        **tokens,
        max_length=150,
        do_sample=True,
        top_p=args.mask_top_p,
        num_return_sequences=1,
        eos_token_id=stop_id,
    )
    return mask_tokenizer.batch_decode(outputs, skip_special_tokens=False)


def extract_fills(texts):
    """
    From: https://github.com/baoguangsheng/fast-detect-gpt/blob/main/scripts/detect_gpt.py
    """
    # remove <pad> from beginning of each text
    texts = [x.replace("<pad>", "").replace("</s>", "").strip() for x in texts]

    # return the text in between each matched mask token
    extracted_fills = [pattern.split(x)[1:-1] for x in texts]

    # remove whitespace around each fill
    extracted_fills = [[y.strip() for y in x] for x in extracted_fills]

    return extracted_fills


def apply_extracted_fills(masked_texts, extracted_fills):
    """
    From: https://github.com/baoguangsheng/fast-detect-gpt/blob/main/scripts/detect_gpt.py
    """
    # split masked text into tokens, only splitting on spaces (not newlines)
    tokens = [x.split(" ") for x in masked_texts]

    n_expected = count_masks(masked_texts)

    # replace each mask token with the corresponding fill
    for idx, (text, fills, n) in enumerate(zip(tokens, extracted_fills, n_expected)):
        if len(fills) < n:
            tokens[idx] = []
        else:
            for fill_idx in range(n):
                text[text.index(f"<extra_id_{fill_idx}>")] = fills[fill_idx]

    # join tokens back into text
    texts = [" ".join(x) for x in tokens]
    return texts


def perturb_texts_(args, mask_model, mask_tokenizer, texts, ceil_pct=False):
    """
    From: https://github.com/baoguangsheng/fast-detect-gpt/blob/main/scripts/detect_gpt.py
    """
    span_length = args.span_length
    pct = args.pct_words_masked
    masked_texts = [tokenize_and_mask(x, span_length, pct, ceil_pct) for x in texts]
    raw_fills = replace_masks(args, mask_model, mask_tokenizer, masked_texts)
    extracted_fills = extract_fills(raw_fills)
    perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)

    # Handle the fact that sometimes the model doesn't generate the right number of fills and we have to try again
    attempts = 1
    while "" in perturbed_texts:
        idxs = [idx for idx, x in enumerate(perturbed_texts) if x == ""]
        print(
            f"WARNING: {len(idxs)} texts have no fills. Trying again [attempt {attempts}]."
        )
        masked_texts = [
            tokenize_and_mask(x, span_length, pct, ceil_pct)
            for idx, x in enumerate(texts)
            if idx in idxs
        ]
        raw_fills = replace_masks(args, mask_model, mask_tokenizer, masked_texts)
        extracted_fills = extract_fills(raw_fills)
        new_perturbed_texts = apply_extracted_fills(masked_texts, extracted_fills)
        for idx, x in zip(idxs, new_perturbed_texts):
            perturbed_texts[idx] = x
        attempts += 1
    return perturbed_texts


def perturb_texts(args, mask_model, mask_tokenizer, texts, ceil_pct=False):
    """
    From: https://github.com/baoguangsheng/fast-detect-gpt/blob/main/scripts/detect_gpt.py
    """
    chunk_size = 10
    outputs = []
    for i in range(0, len(texts), chunk_size):
        outputs.extend(
            perturb_texts_(
                args,
                mask_model,
                mask_tokenizer,
                texts[i : i + chunk_size],
                ceil_pct=ceil_pct,
            )
        )
    return outputs


def generate_perturbs(args, save_dir: Path):
    """
    From: https://github.com/baoguangsheng/fast-detect-gpt/blob/main/scripts/detect_gpt.py
    """
    n_perturbations = args.n_perturbations
    # load model
    mask_model = load_mask_model()
    mask_model.eval()
    try:
        n_positions = mask_model.config.n_positions
    except AttributeError:
        n_positions = 512
    mask_tokenizer = load_mask_tokenizer(max_length=n_positions)

    with open(
        f"./data/generated/{args.dataset}_{args.gen_model}_0.8.completions.json", "r"
    ) as f:
        data = json.load(f)["completions"]

    n_samples = len(data)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # generate perturb samples
    perturbs = {"perturbations": []}
    for idx in tqdm.tqdm(range(n_samples), desc="Perturb text"):
        original_text = data[idx]["real_completion"]
        sampled_text = data[idx]["synthetic_completion"]
        # perturb
        p_sampled_text = perturb_texts(
            args,
            mask_model,
            mask_tokenizer,
            [sampled_text for _ in range(n_perturbations)],
        )
        p_original_text = perturb_texts(
            args,
            mask_model,
            mask_tokenizer,
            [original_text for _ in range(n_perturbations)],
        )
        perturbs["perturbations"].append(
            {
                "real_completion": original_text,
                "synthetic_completion": sampled_text,
                "perturbed_synthetic": p_sampled_text,
                "perturbed_real": p_original_text,
            }
        )

    with open(
        save_dir / f"{args.dataset}_{args.gen_model}_0.8.perturbations.json", "w"
    ) as f:
        f.write(json.dumps(perturbs))


if __name__ == "__main__":
    perterb_dir = Path(__file__).parent.parent / "data" / Path("perturbations")
    perterb_dir.mkdir(parents=True, exist_ok=True)

    # Goal: for datasets xsum, writing, squad, generate perterb files in data/perterbed
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_file", type=str, default="./exp_test/results/xsum_gpt2"
    )
    parser.add_argument("--dataset", type=str)
    parser.add_argument(
        "--pct_words_masked", type=float, default=0.3
    )  # pct masked is actually pct_words_masked * (span_length / (span_length + 2 * buffer_size))
    parser.add_argument("--mask_top_p", type=float, default=1.0)
    parser.add_argument("--span_length", type=int, default=2)
    parser.add_argument("--n_perturbations", type=int, default=10)
    parser.add_argument("--scoring_model_name", type=str, default="gpt2")
    parser.add_argument("--gen-model", type=str)
    parser.add_argument("--mask_filling_model_name", type=str, default="t5-3b")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temp", type=float)
    args = parser.parse_args()

    generate_perturbs(args, save_dir=perterb_dir)
