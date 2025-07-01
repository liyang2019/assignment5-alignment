from typing import Callable
from vllm import LLM, SamplingParams
import torch
from einops import einsum
import wandb
import pandas as pd
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedModel,
    GenerationConfig,
)
from vllm.model_executor import set_random_seed as vllm_set_random_seed
from unittest.mock import patch

from together import Together


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizer,
):
    sequences = []
    response_mask = []
    max_len = 0
    for prompt, output in zip(prompt_strs, output_strs):
        prompt_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.int64)
        output_ids = torch.tensor(tokenizer.encode(output), dtype=torch.int64)
        sequences.append(torch.cat([prompt_ids, output_ids]))
        response_mask.append(
            torch.cat(
                [
                    torch.zeros_like(prompt_ids, dtype=torch.bool),
                    torch.ones_like(output_ids, dtype=torch.bool),
                ]
            )
        )
        max_len = max(len(sequences[-1]), max_len)

    sequences = torch.stack(
        [
            torch.nn.functional.pad(
                x, [0, max_len - x.shape[0]], value=tokenizer.pad_token_id
            )
            for x in sequences
        ]
    )
    response_mask = torch.stack(
        [
            torch.nn.functional.pad(x, [0, max_len - x.shape[0]], value=False)
            for x in response_mask
        ]
    )[:, 1:]
    input_ids = sequences[:, :-1]
    labels = sequences[:, 1:]
    return {
        "input_ids": input_ids,
        "labels": labels,
        "response_mask": response_mask,
    }


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    logits -= logits.max(dim=-1, keepdim=True)[0]
    exp_logits = torch.exp(logits)
    probs = exp_logits / exp_logits.sum(dim=-1, keepdim=True)
    z = torch.logsumexp(logits, dim=-1, keepdim=True)
    return (probs * (z - logits)).sum(dim=-1)


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    logits = model(input_ids).logits
    log_probs = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    labels_one_hot = torch.nn.functional.one_hot(
        labels.to(torch.int64), log_probs.shape[-1]
    )
    log_probs = einsum(
        labels_one_hot.to(log_probs.dtype), log_probs, "b s v, b s v -> b s"
    )
    results = {"log_probs": log_probs}
    if return_token_entropy:
        results["token_entropy"] = compute_entropy(logits)

    return results


def masked_normalize(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    normalize_constant: float,
    dim: int | None = None,
) -> torch.Tensor:
    return (tensor * mask.to(tensor.dtype)).sum(dim=dim) / normalize_constant


def sft_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    normalize_constant: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss = -masked_normalize(
        policy_log_probs,
        response_mask,
        normalize_constant,
        dim=-1,
    ).mean()
    loss /= gradient_accumulation_steps
    loss.backward()
    return loss, {}


def log_generations(
    prompts: list[str],
    answers: list[str],
    tokenizer: PreTrainedTokenizer,
    model: PreTrainedModel,
    reward_fn: Callable[[str, str], dict[str, float]],
    step: int | None = None,
):
    config = GenerationConfig(
        num_beams=1,
        do_sample=False,
        max_new_tokens=1024,
        stop_strings=["</answer>"],
        return_dict_in_generate=True,
        output_logits=True,
    )
    records = []
    for prompt, answer in zip(prompts, answers):
        input_ids = tokenizer.encode(prompt)
        outputs = model.generate(
            input_ids,
            generation_config=config,
            tokenizer=tokenizer,
        )
        output_ids = outputs.sequences[0]
        output = tokenizer.decode(output_ids)
        reward = reward_fn(output, answer)

        logits = torch.concat(outputs.logits, dim=0)
        entropy = compute_entropy(logits)
        logs = {
            "prompt": prompt,
            "output": output,
            "answer": answer,
            "entropy": entropy,
            **reward,
        }
        print(" ".join(f"{k:<8} {v:<10}" for k, v in logs.items()))
        wandb.log(logs)
        records.append(
            {
                **reward,
                "length": len(output_ids),
            }
        )
    df = pd.DataFrame.from_records(records)
    wandb.log(
        {
            "avg_len": df["length"].mean(),
            "format_right_avg_len": df[df["format_reward"] == 1.0]["length"].mean(),
            "format_right_avg_len": df[df["format_reward"] == 0.0]["length"].mean(),
            "answer_right_avg_len": df[df["answer_reward"] == 1.0]["length"].mean(),
            "answer_right_avg_len": df[df["answer_reward"] == 0.0]["length"].mean(),
        },
        step=step,
    )


def init_vllm(
    model_id: str,
    device: str,
    seed: int,
    gpu_memory_utilization: float = 0.85,
):
    """
    Start the inference process, here we use vLLM to hold a model on
    a GPU separate from the policy.
    13
    """
    vllm_set_random_seed(seed)
    # Monkeypatch from TRL:
    # https://github.com/huggingface/trl/blob/22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py
    # Patch vLLM to make sure we can
    # (1) place the vLLM model on the desired device (world_size_patch) and
    # (2) avoid a test that is not designed for our setting (profiling_patch).
    world_size_patch = patch("torch.distributed.get_world_size", return_value=1)
    profiling_patch = patch(
        "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
        return_value=None,
    )
    with world_size_patch, profiling_patch:
        return LLM(
            model=model_id,
            device=device,
            dtype=torch.bfloat16,
            enable_prefix_caching=True,
            gpu_memory_utilization=gpu_memory_utilization,
        )


def load_policy_into_vllm_instance(policy: PreTrainedModel, llm: LLM):
    """
    Copied from https://github.com/huggingface/trl/blob/22759c820867c8659d00082ba8cf004e963873c1/trl/trainer/grpo_trainer.py#L670.
    """
    state_dict = policy.state_dict()
    llm_model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(state_dict.items())


if __name__ == "__main__":
    # Setup wandb metrics
    wandb.define_metric("train_step")  # the x‑axis for training
    wandb.define_metric("eval_step")  # the x‑axis for evaluation
    # everything that starts with train/ is tied to train_step
    wandb.define_metric("train/*", step_metric="train_step")
    # everything that starts with eval/ is tied to eval_step
    wandb.define_metric("eval/*", step_metric="eval_step")
