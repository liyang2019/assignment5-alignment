from typing import Callable
from vllm import LLM, SamplingParams
import json
import collections

from cs336_alignment import drgrpo_grader


def evaluate_vllm(
    vllm_model: LLM,
    prompts: list[str],
    answers: list[str],
    eval_sampling_params: SamplingParams,
    reward_fn: Callable[[str, str], dict[str, float]] = drgrpo_grader.r1_zero_reward_fn,
    result_file: str = "./eval_result.jsonl",
) -> None:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """
    outputs = vllm_model.generate(prompts, eval_sampling_params)
    with open(result_file, "w") as f:
        counter = collections.Counter()
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
            if result["format_reward"] > 0 and result["answer_reward"] > 0:
                counter["f1_a1"] += 1
            elif result["format_reward"] > 0:
                counter["f1_a0"] += 1
            elif result["answer_reward"] > 0:
                counter["f0_a1"] += 1
            else:
                counter["f0_a0"] += 1
    return {
        "eval/format_accuracy": (counter["f1_a0"] + counter["f1_a1"]) / len(outputs),
        "eval/answer_accuracy": (counter["f1_a1"]) / len(outputs),
    }


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
        prompts,
        answers,
        sampling_params,
    )
