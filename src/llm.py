from typing import List, Optional, Tuple

import numpy as np
import torch
from loguru import logger
from numpy.random import choice
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from src.hardware import DEVICE


class LLM:
    def __init__(self, model_name):
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device_map="auto",
        )
        self.model_name = model_name

    def encode(
        self,
        text: str,
    ):
        encodings = self.tokenizer(
            text,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_tensors="pt",
            return_attention_mask=False,
        )
        return encodings["input_ids"]

    def get_predicted_next_token_logits(self, tokens: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.model(tokens)
        return outputs["logits"][0, -1, :]

    def sample_top_k_from_logits(
        self, logits: torch.Tensor, top_k: int = 5, local_normalization=False
    ) -> Tuple[List[str], List[float]]:
        top_k_token_indices = torch.topk(logits, top_k).indices
        all_token_probs = torch.nn.functional.softmax(logits, dim=-1)
        top_k_token_probs = all_token_probs[top_k_token_indices]
        if local_normalization:
            top_k_token_probs = torch.nn.functional.normalize(
                top_k_token_probs, dim=-1, p=1
            )
        top_k_tokens = [self.tokenizer.decode([idx]) for idx in top_k_token_indices]
        return {
            "tokens": top_k_tokens,
            "token_indices": top_k_token_indices.tolist(),
            "token_probs": top_k_token_probs.tolist(),
        }

    def sample_top_p_from_logits(
        self,
        logits: torch.Tensor,
        top_p: float,
        local_normalization=False,
    ):
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_all_token_probs = torch.nn.functional.softmax(sorted_logits, dim=-1)

        cumulative_probs = torch.cumsum(sorted_all_token_probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs >= top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        sorted_indices_to_keep = ~sorted_indices_to_remove

        top_p_token_indices = sorted_indices[sorted_indices_to_keep]
        top_p_tokens = [self.tokenizer.decode([idx]) for idx in top_p_token_indices]
        top_p_token_probs = sorted_all_token_probs[sorted_indices_to_keep]

        if local_normalization:
            top_p_token_probs = torch.nn.functional.normalize(
                top_p_token_probs, dim=-1, p=1
            )

        return {
            "tokens": top_p_tokens,
            "token_indices": top_p_token_indices.tolist(),
            "token_probs": top_p_token_probs.tolist(),
        }

    def sample_next_token(
        self,
        context_tokens: torch.Tensor,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temp: Optional[float] = None,
    ):
        logits = self.get_predicted_next_token_logits(context_tokens.to(DEVICE))

        if [top_k, top_p, temp].count(None) != 2:
            raise Exception("Only set one of top_k, top_p and temperature.")

        if top_k:
            top_k_sample = self.sample_top_k_from_logits(
                logits, local_normalization=True, top_k=top_k
            )
            next_token_index = choice(
                top_k_sample["token_indices"],
                1,
                p=np.array(top_k_sample["token_probs"])
                / np.array(top_k_sample["token_probs"]).sum(),
            )

        if top_p:
            top_p_sample = self.sample_top_p_from_logits(
                logits, local_normalization=True, top_p=top_p
            )
            next_token_index = choice(
                top_p_sample["token_indices"],
                1,
                p=np.array(top_p_sample["token_probs"])
                / np.array(top_p_sample["token_probs"]).sum(),
            )

        if temp:
            temp_weights = (
                torch.nn.functional.softmax(logits / temp, dim=-1)[..., :].cpu().numpy()
            )
            next_token_index = choice(
                range(logits.shape[-1]), 1, p=temp_weights / temp_weights.sum()
            )

        return torch.tensor(np.array([next_token_index]))

    def get_completion(
        self,
        context: str,
        num_return_sequences: int,
        num_new_tokens: int = 10,
        top_k: int = None,
        top_p: float = None,
        temperature: float = None,
    ) -> str:
        output = []
        initial_tokens = self.encode(context)
        for idx in range(num_return_sequences):
            logger.info(f"Generating {idx} of {num_return_sequences}")
            current_tokens = initial_tokens.clone()
            for _ in range(num_new_tokens):
                new_token = self.sample_next_token(
                    current_tokens, top_k=top_k, top_p=top_p, temp=temperature
                )
                current_tokens = torch.cat((current_tokens, new_token), dim=1)
            completion_tokens = current_tokens[:, initial_tokens.shape[1] :]
            generated_text = self.tokenizer.decode(completion_tokens[0])
            output.append(
                {
                    "context": context,
                    "full_tokens": current_tokens,
                    "context_tokens": initial_tokens,
                    "completion_tokens": completion_tokens,
                    "generated_text": generated_text,
                }
            )
        return output

    def fast_sampling(
        self,
        context: str,
        num_return_sequences: int = 10,
        num_new_tokens: int = 15,
        temp=1.0,
        top_p=1.0,
        top_k=None,
    ):
        """
        Accelerated version of get_completion() for pure sampling with top-p=1.0, temperature=1.0, and
        no top-k set. Leverages the huggingface library, which is avoided in non-trivial configurations
        of top_p/top_k/temperature due to rare oddities in sampling due to tokenisation
        choices that may lead to exceptions.
        """

        context_tokens = self.encode(context)
        sequences = self.pipe(
            context,
            do_sample=True,
            num_return_sequences=num_return_sequences,
            eos_token_id=self.tokenizer.eos_token_id,
            max_new_tokens=num_new_tokens,
            min_new_tokens=num_new_tokens,
            top_p=top_p,
            top_k=top_k,
            temperature=temp,
            return_full_text=False,
        )
        return [
            {
                "context": context,
                "full_tokens": self.encode(context + x["generated_text"]),
                "context_tokens": context_tokens,
                "completion_tokens": self.encode(context + x["generated_text"])[
                    :, context_tokens.shape[1] :
                ],
                "generated_text": x["generated_text"],
            }
            for x in sequences
        ]
