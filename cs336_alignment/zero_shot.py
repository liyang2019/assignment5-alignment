from typing import Callable
from vllm import LLM, SamplingParams
import json

from cs336_alignment import drgrpo_grader


def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: list[str],
    answers: list[str],
    eval_sampling_params: SamplingParams,
) -> None:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """
    outputs = vllm_model.generate(prompts, eval_sampling_params)
    with open("./eval_result.jsonl", "w") as f:
        for output, answer in zip(outputs, answers):
            metrics = reward_fn(output.outputs[0].text, answer)
            result = {
                "promot": output.prompt,
                "answer": answer,
                "output": output.outputs[0].text,
                **metrics,
            }
            f.write(json.dumps(result) + "\n")
            f.flush()


if __name__ == "__main__":
    with open("./cs336_alignment/prompts/r1_zero.prompt", "r") as f:
        prompt_template = f.read()

    prompts = []
    answers = []
    with open("./data/MATH/validation.jsonl", "r") as f:
        for line in f.readlines():
            example = json.loads(line)
            prompts.append(prompt_template.format(question=example["problem"]))
            answers.append(example["solution"])

    sampling_params = SamplingParams(
        temperature=1.0,
        top_p=1.0,
        max_tokens=1024,
        stop=["</answer>"],
        include_stop_str_in_output=True,
    )

    llm = LLM(model="data/Qwen2.5-Math-1.5B")

    evaluate_vllm(
        llm,
        drgrpo_grader.r1_zero_reward_fn,
        prompts,
        answers,
        sampling_params,
    )
