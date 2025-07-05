import json
import random
import pprint
import torch
from typing import Callable
from vllm import LLM, SamplingParams
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedModel,
    GenerationConfig,
)
import datetime
import os
import wandb
import numpy as np

from cs336_alignment import drgrpo_grader
from cs336_alignment import sft
from cs336_alignment import zero_shot


def load_examples(
    jsonl_path: str,
    prompt_template_file: str = "./cs336_alignment/prompts/r1_zero.prompt",
):
    with open(prompt_template_file, "r") as f:
        prompt_template = f.read()
    examples = []
    with open(jsonl_path) as f:
        for line in f.readlines():
            example = json.loads(line)
            example["prompt"] = prompt_template.format(question=example["problem"])
            examples.append(example)
    return examples


def generate_samples(
    vllm: LLM,
    prompts: list[str],
    answers: list[str],
    sampling_params: SamplingParams,
    reward_fn: Callable[[str, str], dict[str, float]] = drgrpo_grader.r1_zero_reward_fn,
):
    outputs = vllm.generate(prompts, sampling_params)
    samples = []
    for answer, prompt, output in zip(answers, prompts, outputs):
        correct_outputs = set()
        for res in output.outputs:
            reward = reward_fn(res.text, answer)
            if reward["answer_reward"]:
                correct_outputs.add(res.text)
        for correct_output in correct_outputs:
            samples.append(
                {
                    "prompt": prompt,
                    "answer": answer,
                    "output": correct_output,
                }
            )
    return samples


def run_expert_iteration(
    n_ei_steps: int = 10,
    D_b: int = 16,
    G: int = 32,
    n_sft_steps: int = 80,
    D_sft: int = 4,
    learning_rate: float = 0.0001,
    gradient_accumulation_steps: int = 4,
):
    run = wandb.init(
        entity="liyang2029-meta",
        project="cs336-2025-assignment5",
        config=locals(),
    )
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")

    model = AutoModelForCausalLM.from_pretrained(
        "./data/Qwen2.5-Math-1.5B",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to("cuda")
    model.compile()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    tokenizer = AutoTokenizer.from_pretrained("./data/Qwen2.5-Math-1.5B")
    vllm = sft.init_vllm(
        "./data/Qwen2.5-Math-1.5B",
        device="cuda",
        seed=123,
        gpu_memory_utilization=0.3,
    )
    sampling_params = SamplingParams(
        n=G,
        temperature=1.0,
        top_p=1.0,
        min_tokens=4,
        max_tokens=512,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )
    eval_sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    train_examples = load_examples("./data/MATH/train.jsonl")
    eval_examples = load_examples("./data/MATH/validation.jsonl")

    output_dir = (
        f"./experiments/{datetime.datetime.now().strftime('%d-%m-%Y-%H-%M-%S')}"
    )
    os.makedirs(output_dir, exist_ok=True)

    for step in range(n_ei_steps):
        # Sampling stage
        examples = random.sample(train_examples, D_b)
        samples = generate_samples(
            vllm,
            [e["prompt"] for e in examples],
            [e["solution"] for e in examples],
            sampling_params,
        )
        random.shuffle(samples)
        pprint.pprint(samples[-1])
        print(f"num samples obtained {len(samples)}/{D_b * G}")

        n_sft_steps = min((len(samples) * 5 - 1) // D_sft + 1, n_sft_steps)
        # Training stage
        losses = []
        for j in range(n_sft_steps):
            batch_samples = random.sample(samples, min(len(samples), D_sft))
            inputs = sft.tokenize_prompt_and_output(
                [s["prompt"] for s in batch_samples],
                [s["output"] for s in batch_samples],
                tokenizer,
            )
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            print(inputs["response_mask"].int().sum(dim=-1))
            outputs = torch.compile(sft.get_response_log_probs)(
                model,
                inputs["input_ids"],
                inputs["labels"],
                return_token_entropy=False,
            )
            loss, _ = sft.sft_microbatch_train_step(
                outputs["log_probs"],
                inputs["response_mask"],
                gradient_accumulation_steps=gradient_accumulation_steps,
            )
            losses.append(loss.detach().cpu().float().numpy())
            if (j + 1) % gradient_accumulation_steps == 0:
                # average_entropy = outputs["token_entropy"][
                #     inputs["response_mask"]
                # ].mean()
                loss = np.mean(losses)
                losses.clear()
                run.log(
                    {
                        "train/loss": loss,
                        # "train/average_entropy": average_entropy,
                    }
                )
                print(f"step {step}-{j + 1} loss {loss}")
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()

        sft.load_policy_into_vllm_instance(model, vllm)
        # local_eval_examples = eval_examples[:100]
        local_eval_examples = eval_examples
        metrics = zero_shot.evaluate_vllm(
            vllm,
            [e["prompt"] for e in local_eval_examples],
            [e["solution"] for e in local_eval_examples],
            eval_sampling_params,
            result_file=f"{output_dir}/eval_results-{step}.jsonl",
        )
        run.log(metrics)

    model.save_pretrained(save_directory=f"{output_dir}/ckpt-{step}")
    tokenizer.save_pretrained(save_directory=f"{output_dir}/ckpt-{step}")

    metrics = zero_shot.evaluate_vllm(
        vllm,
        [e["prompt"] for e in eval_examples],
        [e["solution"] for e in eval_examples],
        eval_sampling_params,
        result_file=f"{output_dir}/eval_results-{step}.jsonl",
    )
    run.log(metrics)


if __name__ == "__main__":
    run_expert_iteration()
