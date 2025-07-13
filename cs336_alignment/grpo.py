from typing import Callable, Literal
import torch
import wandb
import datetime
import os
import random
import numpy as np

from vllm import LLM, SamplingParams
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedModel,
    GenerationConfig,
)

from cs336_alignment import sft
from cs336_alignment import expert_iteration
from cs336_alignment import zero_shot
from cs336_alignment import drgrpo_grader

torch.set_float32_matmul_precision("high")


def compute_group_normalized_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
    group_size: int,
    advantage_eps: float,
    normalize_by_std: bool,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    rewards = []
    for response, ground_truth in zip(rollout_responses, repeated_ground_truths):
        rewards.append(reward_fn(response, ground_truth)["answer_reward"])
    rewards = torch.tensor(rewards).view(-1, group_size)

    rewards_mean = rewards.mean(dim=1, keepdim=True)
    advantages = rewards - rewards_mean
    if normalize_by_std:
        rewards_std = rewards.view(-1, group_size).std(dim=1, keepdim=True)
        advantages = advantages / (rewards_std + advantage_eps)
    return advantages.view(-1), rewards.view(-1), {}


def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
) -> torch.Tensor:
    return -raw_rewards_or_advantages * policy_log_probs


def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    ratios = torch.exp(policy_log_probs - old_log_probs)
    lhs = ratios * advantages
    rhs = torch.clip(ratios, 1 - cliprange, 1 + cliprange) * advantages
    loss = -torch.minimum(lhs, rhs)
    return loss, {"clipped": lhs < rhs}


def compute_grpo_no_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    ratios = torch.exp(policy_log_probs - old_log_probs)
    loss = -ratios * advantages
    return loss, {}


def compute_policy_gradient_loss(
    policy_log_probs: torch.Tensor,
    loss_type: Literal["no_baseline", "reinforce_with_baseline", "grpo_clip"],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if loss_type == "no_baseline":
        loss = compute_naive_policy_gradient_loss(raw_rewards, policy_log_probs)
        metadata = {}
    elif loss_type == "reinforce_with_baseline":
        loss = compute_naive_policy_gradient_loss(advantages, policy_log_probs)
        metadata = {}
    elif loss_type == "grpo_clip":
        loss, metadata = compute_grpo_clip_loss(
            advantages, policy_log_probs, old_log_probs, cliprange
        )
    elif loss_type == "grpo_no_clip":
        loss, metadata = compute_grpo_no_clip_loss(
            advantages, policy_log_probs, old_log_probs
        )
    return loss, metadata


def masked_mean(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    dim: int | None = None,
) -> torch.Tensor:
    return (tensor * mask).sum(dim=dim) / (mask.sum(dim=dim) + 1e-6)


def grpo_microbatch_train_step(
    policy_log_probs: torch.Tensor,
    response_mask: torch.Tensor,
    gradient_accumulation_steps: int,
    loss_type: Literal[
        "no_baseline",
        "reinforce_with_baseline",
        "grpo_clip",
        "grpo_no_clip",
    ],
    raw_rewards: torch.Tensor | None = None,
    advantages: torch.Tensor | None = None,
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    loss, metadata = compute_policy_gradient_loss(
        policy_log_probs,
        loss_type,
        raw_rewards,
        advantages,
        old_log_probs,
        cliprange,
    )
    loss = masked_mean(loss, response_mask) / gradient_accumulation_steps
    # loss = (
    #     sft.masked_normalize(
    #         loss,
    #         response_mask,
    #         dim=-1,
    #         normalize_constant=1024,
    #     ).mean()
    #     / gradient_accumulation_steps
    # )
    loss.backward()
    return loss, metadata


def generate_samples(
    vllm: LLM,
    prompts: list[str],
    answers: list[str],
    sampling_params: SamplingParams,
    reward_fn: Callable[[str, str], dict[str, float]],
):
    outputs = vllm.generate(prompts, sampling_params)
    samples = []
    for answer, prompt, output in zip(answers, prompts, outputs):
        for res in output.outputs:
            samples.append(
                {
                    "prompt": prompt,
                    "answer": answer,
                    "output": res.text,
                    "reward": reward_fn(res.text, answer)["answer_reward"],
                }
            )
    return samples


def run_grpo(
    n_grpo_steps: int = 200,
    learning_rate: float = 1e-5,
    advantage_eps: float = 1e-6,
    rollout_batch_size: int = 256,
    group_size: int = 8,
    sampling_temperature: float = 1.0,
    sampling_min_tokens: int = 4,
    sampling_max_tokens: int = 1024,
    epochs_per_rollout_batch: int = 2,
    train_batch_size: int = 128,
    gradient_accumulation_steps: int = 64,
    gpu_memory_utilization: float = 0.3,
    loss_type: Literal[
        "no_baseline",
        "reinforce_with_baseline",
        "grpo_clip",
        "grpo_no_clip",
    ] = "grpo_no_clip",
    use_std_normalization: bool = True,
    prompt_template: Literal[
        "r1_zero",
        "question_only",
    ] = "question_only",
):
    run = wandb.init(
        entity="liyang2029-meta",
        project="cs336-2025-assignment5-grpo",
        config=locals(),
        name=(
            f"{loss_type}_"
            f"{prompt_template}_"
            f"ep{epochs_per_rollout_batch}_"
            f"b{train_batch_size}_"
            f"lr{learning_rate}"
        ),
    )
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")

    assert (
        train_batch_size % gradient_accumulation_steps == 0
    ), "train_batch_size must be divisible by gradient_accumulation_steps"
    micro_train_batch_size = train_batch_size // gradient_accumulation_steps
    assert (
        rollout_batch_size % group_size == 0
    ), "rollout_batch_size must be divisible by group_size"
    n_prompts_per_rollout_batch = rollout_batch_size // group_size
    assert (
        train_batch_size >= group_size
    ), "train_batch_size must be greater than or equal to group_size"

    policy = AutoModelForCausalLM.from_pretrained(
        "./data/Qwen2.5-Math-1.5B",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to("cuda")
    policy.compile()

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=learning_rate,
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )

    tokenizer = AutoTokenizer.from_pretrained("./data/Qwen2.5-Math-1.5B")

    sampling_params = SamplingParams(
        n=group_size,
        temperature=sampling_temperature,
        top_p=1.0,
        min_tokens=sampling_min_tokens,
        max_tokens=sampling_max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    eval_sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=sampling_max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    vllm = sft.init_vllm(
        "./data/Qwen2.5-Math-1.5B",
        device="cuda",
        seed=123,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    vllm.sleep(level=1)

    if prompt_template == "r1_zero":
        prompt_template_file = "./cs336_alignment/prompts/r1_zero.prompt"
        reward_fn = drgrpo_grader.r1_zero_reward_fn
    elif prompt_template == "question_only":
        prompt_template_file = "./cs336_alignment/prompts/question_only.prompt"
        reward_fn = drgrpo_grader.question_only_reward_fn
    else:
        raise ValueError(f"Invalid prompt_template: {prompt_template!r}")
    train_examples = expert_iteration.load_examples(
        "./data/MATH/train.jsonl",
        prompt_template_file=prompt_template_file,
    )
    eval_examples = expert_iteration.load_examples(
        "./data/MATH/validation.jsonl",
        prompt_template_file=prompt_template_file,
    )

    output_dir = (
        f"./experiments/grpo/{datetime.datetime.now().strftime('%d-%m-%Y-%H-%M-%S')}"
    )
    os.makedirs(output_dir, exist_ok=True)

    get_response_log_probs = torch.compile(sft.get_response_log_probs)

    for step in range(n_grpo_steps):
        # Sampling stage
        examples = random.sample(train_examples, n_prompts_per_rollout_batch)
        vllm.wake_up()
        sft.load_policy_into_vllm_instance(policy, vllm)
        torch.cuda.empty_cache()
        if (step + 1) % 10 == 0:
            local_eval_examples = eval_examples[:1024]
            metrics = zero_shot.evaluate_vllm(
                vllm,
                [e["prompt"] for e in local_eval_examples],
                [e["solution"] for e in local_eval_examples],
                eval_sampling_params,
                reward_fn=reward_fn,
                result_file=f"{output_dir}/eval_results-{step}.jsonl",
            )
            run.log(metrics)
        samples = generate_samples(
            vllm,
            [e["prompt"] for e in examples],
            [e["solution"] for e in examples],
            sampling_params,
            reward_fn=reward_fn,
        )
        vllm.sleep(level=1)
        torch.cuda.empty_cache()

        rewards = [s["reward"] for s in samples]
        print(f"step {step} rewards: {int(sum(rewards))}/{len(rewards)}")
        run.log({"average_reward": sum(rewards) / len(rewards)})

        inputs = sft.tokenize_prompt_and_output(
            [s["prompt"] for s in samples],
            [s["output"] for s in samples],
            tokenizer,
            sequence_len=sampling_max_tokens,
        )
        run.log({"average_length": inputs["response_mask"].float().sum(dim=-1).mean()})

        advantages, raw_rewards, _ = compute_group_normalized_rewards(
            reward_fn,
            [s["output"] for s in samples],
            [s["answer"] for s in samples],
            group_size,
            advantage_eps,
            normalize_by_std=use_std_normalization,
        )
        advantages = advantages.to("cuda")
        raw_rewards = raw_rewards.to("cuda")

        # Compute old log probs
        optimizer.zero_grad()
        policy.eval()
        with torch.inference_mode():
            old_log_probs = []
            start = 0
            while start < len(samples):
                end = start + micro_train_batch_size
                batch_inputs = {k: v[start:end].to("cuda") for k, v in inputs.items()}
                batch_outputs = get_response_log_probs(
                    policy,
                    batch_inputs["input_ids"],
                    batch_inputs["labels"],
                    return_token_entropy=False,
                )
                old_log_probs.append(batch_outputs["log_probs"].detach())
                start = end
        old_log_probs = torch.concat(old_log_probs, dim=0)
        print(f"old_log_probs: {old_log_probs.shape}")
        torch.cuda.empty_cache()

        # Training stage
        policy.train()
        losses = []
        for epoch in range(epochs_per_rollout_batch):
            indices = torch.randperm(len(samples))
            start = 0
            while start < len(samples):
                end = start + micro_train_batch_size
                batch_idx = indices[start:end]
                batch_inputs = {k: v[batch_idx].to("cuda") for k, v in inputs.items()}
                batch_outputs = get_response_log_probs(
                    policy,
                    batch_inputs["input_ids"],
                    batch_inputs["labels"],
                    return_token_entropy=False,
                )
                batch_log_probs = batch_outputs["log_probs"]
                batch_response_mask = batch_inputs["response_mask"]
                batch_raw_rewards = raw_rewards[batch_idx].unsqueeze(1)
                batch_advantages = advantages[batch_idx].unsqueeze(1)
                batch_old_log_probs = old_log_probs[batch_idx]
                loss, metadata = grpo_microbatch_train_step(
                    batch_log_probs,
                    batch_response_mask,
                    gradient_accumulation_steps,
                    loss_type,
                    batch_raw_rewards,
                    batch_advantages,
                    batch_old_log_probs,
                    cliprange=0.2,
                )
                # average_entropy = outputs["token_entropy"][
                #     inputs["response_mask"]
                # ].mean()
                run.log(
                    {
                        "train/loss": loss,
                        "train/reward": batch_raw_rewards.mean(),
                        "train/advantage": batch_advantages.mean(),
                        # "train/average_entropy": average_entropy,
                    }
                )
                losses.append(loss.detach().cpu().float().numpy())
                if len(losses) == gradient_accumulation_steps:
                    loss = np.mean(losses)
                    print(f"step {step}-{epoch}-{end} loss {loss}")
                    losses.clear()
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), 1)

                    grad_norm = 0.0
                    for p in policy.parameters():
                        grad_norm += (p.grad**2.0).sum()
                    grad_norm = grad_norm**0.5
                    run.log({"train/grad_norm": grad_norm})

                    optimizer.step()
                    optimizer.zero_grad()
                start = end
        torch.cuda.empty_cache()

    vllm.wake_up()
    sft.load_policy_into_vllm_instance(policy, vllm)
    metrics = zero_shot.evaluate_vllm(
        vllm,
        [e["prompt"] for e in eval_examples],
        [e["solution"] for e in eval_examples],
        eval_sampling_params,
        reward_fn=reward_fn,
        result_file=f"{output_dir}/eval_results-{step}.jsonl",
    )
    run.log(metrics)

    policy.save_pretrained(save_directory=f"{output_dir}/ckpt-{step}")
    tokenizer.save_pretrained(save_directory=f"{output_dir}/ckpt-{step}")


def run_eval(
    sampling_max_tokens: int = 1024,
    prompt_template: Literal[
        "r1_zero",
        "question_only",
    ] = "question_only",
):
    output_dir = "./experiments/grpo/06-07-2025-19-19-30"
    run = wandb.init(
        entity="liyang2029-meta",
        project="cs336-2025-assignment5-grpo",
        config=locals(),
    )
    vllm = sft.init_vllm(
        f"{output_dir}/ckpt-199",
        device="cuda",
        seed=123,
        gpu_memory_utilization=0.8,
    )
    eval_sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=sampling_max_tokens,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    eval_examples = expert_iteration.load_examples("./data/MATH/validation.jsonl")
    if prompt_template == "r1_zero":
        prompt_template_file = "./cs336_alignment/prompts/r1_zero.prompt"
        reward_fn = drgrpo_grader.r1_zero_reward_fn
    elif prompt_template == "question_only":
        prompt_template_file = "./cs336_alignment/prompts/question_only.prompt"
        reward_fn = drgrpo_grader.question_only_reward_fn
    else:
        raise ValueError(f"Invalid prompt_template: {prompt_template!r}")
    metrics = zero_shot.evaluate_vllm(
        vllm,
        [e["prompt"] for e in eval_examples],
        [e["solution"] for e in eval_examples],
        eval_sampling_params,
        reward_fn=reward_fn,
        result_file=f"{output_dir}/eval_results-final.jsonl",
    )
    run.log(metrics)


if __name__ == "__main__":
    run_grpo()
    # run_eval()
