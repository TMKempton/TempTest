from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from torch.nn import CrossEntropyLoss

from src.hardware import DEVICE
from src.llm import LLM


def compute_log_likelhood(tokens: torch.Tensor, llm: LLM) -> torch.Tensor:
    with torch.no_grad():
        logits = llm.model(tokens.to(DEVICE)).logits
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = tokens[..., 1:].contiguous().to(DEVICE)
    loss_fct = CrossEntropyLoss(reduction="none")
    neg_lls = loss_fct(shift_logits.transpose(1, 2), shift_labels)[0]
    return -neg_lls


def compute_perplexity(
    tokens: torch.Tensor,
    llm: LLM,
) -> Tuple[float, List[float]]:
    neg_lls = compute_log_likelhood(tokens, llm)
    avg_neg_ll = torch.mean(neg_lls)
    perplexity = -avg_neg_ll
    return {
        "perplexity": perplexity.item(),
        "negative_log_likelihoods": neg_lls.tolist(),
    }


def temp_norm(logits: torch.Tensor, temp: float):
    output = torch.softmax(logits, dim=1)
    output = torch.pow(output, 1 / temp)
    output = output.sum(1)
    return output


def compute_log_temp_norms(
    tokens: torch.Tensor,
    llm: LLM,
    temp: float = 0.7,
) -> torch.Tensor:
    with torch.no_grad():
        logits = llm.model(tokens.to(DEVICE)).logits
    shift_logits = logits[..., :-1, :].contiguous()
    temp_norms = temp_norm(shift_logits.transpose(1, 2), temp)
    log_temp_norms = torch.log(temp_norms)
    return log_temp_norms


def compute_temptest_metrics(completion: str, llm: LLM, temp: float) -> List[float]:
    completion_tokens = llm.encode(completion)
    log_temp_norms = compute_log_temp_norms(completion_tokens, llm, temp)
    log_likelhood = compute_log_likelhood(completion_tokens, llm)
    unaggregated_temtest_metrics = log_temp_norms - ((1 / temp) - 1) * log_likelhood
    logger.info(f"Average per token temp norm: {unaggregated_temtest_metrics.mean(-1)}")
    logger.info(f"Unaggregated metric shape: {unaggregated_temtest_metrics.shape}")
    return unaggregated_temtest_metrics[0].tolist()


def bayesian_decision_boundary(x, temperature: float, c: float):
    return np.log(1 / c - 1) / 1000 + x * (1 / temperature - 1)


def get_sampling_discrepancy_analytic(
    logits_ref: torch.Tensor, logits_score: torch.Tensor, labels: torch.Tensor
) -> float:
    """
    From: https://github.com/baoguangsheng/fast-detect-gpt/blob/main/scripts/fast_detect_gpt.py
    """

    if logits_ref.size(-1) != logits_score.size(-1):
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref = logits_ref[:, :, :vocab_size]
        logits_score = logits_score[:, :, :vocab_size]

    labels = labels.unsqueeze(-1) if labels.ndim == logits_score.ndim - 1 else labels
    lprobs_score = torch.log_softmax(logits_score, dim=-1)
    probs_ref = torch.softmax(logits_ref, dim=-1)
    log_likelihood = lprobs_score.gather(dim=-1, index=labels).squeeze(-1)
    mean_ref = (probs_ref * lprobs_score).sum(dim=-1)
    var_ref = (probs_ref * torch.square(lprobs_score)).sum(dim=-1) - torch.square(
        mean_ref
    )
    discrepancy = (log_likelihood.sum(dim=-1) - mean_ref.sum(dim=-1)) / var_ref.sum(
        dim=-1
    ).sqrt()
    discrepancy = discrepancy.mean()
    return discrepancy.item()


def compute_fastdetect_gpt_metric(
    completion: str, scoring_llm: LLM, reference_llm: LLM, num_tokens: int
):
    """
    Adapted from: https://github.com/baoguangsheng/fast-detect-gpt/blob/main/scripts/fast_detect_gpt.py
    """

    tokenized = (
        scoring_llm.tokenizer(
            completion, return_tensors="pt", return_token_type_ids=False
        )
        .input_ids[:, :num_tokens]
        .to(DEVICE)
    )
    logger.info(tokenized.shape)
    labels = tokenized[:, 1:]
    with torch.no_grad():
        logits_score = scoring_llm.model(tokenized).logits[:, :-1]
        if scoring_llm.model_name == reference_llm.model_name:
            logits_ref = logits_score
        else:
            tokenized = reference_llm.tokenizer(
                completion,
                return_tensors="pt",
                return_token_type_ids=False,
            )
            logits_ref = reference_llm(tokenized).logits[:, :-1]
        fastdetect_gpt_metric = get_sampling_discrepancy_analytic(
            logits_ref, logits_score, labels
        )
    return fastdetect_gpt_metric


def get_likelihood(logits, labels):
    """
    From: https://github.com/baoguangsheng/fast-detect-gpt/blob/main/scripts/baselines.py
    """
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1

    logits = logits.view(-1, logits.shape[-1])
    labels = labels.view(-1)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    log_likelihood = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return log_likelihood.mean().item()


def get_rank(logits, labels):
    """
    From: https://github.com/baoguangsheng/fast-detect-gpt/blob/main/scripts/baselines.py
    """
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1

    # get rank of each label token in the model's likelihood ordering
    matches = (logits.argsort(-1, descending=True) == labels.unsqueeze(-1)).nonzero()
    assert (
        matches.shape[1] == 3
    ), f"Expected 3 dimensions in matches tensor, got {matches.shape}"

    ranks, timesteps = matches[:, -1], matches[:, -2]

    # make sure we got exactly one match for each timestep in the sequence
    assert (
        timesteps == torch.arange(len(timesteps)).to(timesteps.device)
    ).all(), "Expected one match per timestep"

    ranks = ranks.float() + 1  # convert to 1-indexed rank
    return -ranks.mean().item()


def get_logrank(logits, labels):
    """
    From: https://github.com/baoguangsheng/fast-detect-gpt/blob/main/scripts/baselines.py
    """

    assert logits.shape[0] == 1
    assert labels.shape[0] == 1

    # get rank of each label token in the model's likelihood ordering
    matches = (logits.argsort(-1, descending=True) == labels.unsqueeze(-1)).nonzero()
    assert (
        matches.shape[1] == 3
    ), f"Expected 3 dimensions in matches tensor, got {matches.shape}"

    ranks, timesteps = matches[:, -1], matches[:, -2]

    # make sure we got exactly one match for each timestep in the sequence
    assert (
        timesteps == torch.arange(len(timesteps)).to(timesteps.device)
    ).all(), "Expected one match per timestep"

    ranks = ranks.float() + 1  # convert to 1-indexed rank
    ranks = torch.log(ranks)
    return -ranks.mean().item()


def get_entropy(logits, labels):
    """
    From: https://github.com/baoguangsheng/fast-detect-gpt/blob/main/scripts/baselines.py
    """

    assert logits.shape[0] == 1
    assert labels.shape[0] == 1

    entropy = F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)
    entropy = -entropy.sum(-1)
    return entropy.mean().item()


def compute_baseline_metric(
    completion: str, scoring_llm: LLM, criterion: Callable, num_tokens: int
):
    """
    Adapted from: https://github.com/baoguangsheng/fast-detect-gpt/blob/main/scripts/fast_detect_gpt.py
    """

    tokenized = (
        scoring_llm.tokenizer(
            completion, return_tensors="pt", return_token_type_ids=False
        )
        .input_ids[:, :num_tokens]
        .to(DEVICE)
    )
    labels = tokenized[:, 1:]
    with torch.no_grad():
        logits = scoring_llm.model(tokenized).logits[:, :-1]
        metric = criterion(logits, labels)
    return metric


# Get the log likelihood of each text under the base_model
def get_ll(scoring_model, scoring_tokenizer, text, num_tokens: int, fixed=True):
    with torch.no_grad():
        tokenized = (
            scoring_tokenizer(text, return_tensors="pt", return_token_type_ids=False)
            .to(DEVICE)
            .input_ids[:, :num_tokens]
        )
        labels = tokenized
        return -scoring_model(tokenized, labels=labels).loss.item()


def get_lls(scoring_model, scoring_tokenizer, texts, num_tokens):
    return [
        get_ll(scoring_model, scoring_tokenizer, text, num_tokens=num_tokens)
        for text in texts
    ]


def compute_detect_gpt_metric(
    completion: str,
    perturbed_completions: List[str],
    scoring_llm: LLM,
    num_tokens: int,
) -> Dict[str, float]:
    completion_ll = get_ll(
        scoring_model=scoring_llm.model,
        scoring_tokenizer=scoring_llm.tokenizer,
        text=completion,
        num_tokens=num_tokens,
    )
    p_ll = get_lls(
        scoring_model=scoring_llm.model,
        scoring_tokenizer=scoring_llm.tokenizer,
        texts=perturbed_completions,
        num_tokens=num_tokens,
    )
    perturbed_mean = np.mean(p_ll)
    perturbed_std = np.std(p_ll) if len(p_ll) > 1 else 1

    if perturbed_std == 0:
        perturbed_std = 1
        logger.warning("WARNING: std of perturbed list is 0, setting to 1")

    return (completion_ll - perturbed_mean) / perturbed_std


def compute_npr_metric(
    completion: str,
    perturbed_completions: List[str],
    scoring_llm: LLM,
    num_tokens: int,
) -> Dict[str, float]:
    with torch.no_grad():
        tokenized = (
            scoring_llm.tokenizer(
                completion, return_tensors="pt", return_token_type_ids=False
            )
            .input_ids[:, :num_tokens]
            .to(DEVICE)
        )
        labels = tokenized[:, 1:]
        logits = scoring_llm.model(tokenized).logits[:, :-1]
        logrank = get_logrank(logits, labels)
        # perturbations
        logranks = []
        for perturb in perturbed_completions:
            tokenized = (
                scoring_llm.tokenizer(
                    perturb, return_tensors="pt", return_token_type_ids=False
                )
                .input_ids[:, :num_tokens]
                .to(DEVICE)
            )
            labels = tokenized[:, 1:]
            logits = scoring_llm.model(tokenized).logits[:, :-1]
            logranks.append(get_logrank(logits, labels))
        # npr
        return np.mean(logranks) / logrank
