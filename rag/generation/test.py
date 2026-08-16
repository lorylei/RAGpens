import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer


RAG_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAG_ROOT))
from data.datasets import Seq2SeqDataset
from metrics.eval_metrics import HeadlineEvaluation
from prompts.post_process import post_process


os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--model_path", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--CUDA_VISIBLE_DEVICES", default="0")
    parser.add_argument("--random_seed", type=int, default=2024)
    parser.add_argument("--begin_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=20600)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--cutoff_len", type=int, default=70000)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.70)
    parser.add_argument("--max_num_seqs", type=int, default=32)
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.CUDA_VISIBLE_DEVICES
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dataset = Seq2SeqDataset(
        args.input_file,
        llm_tokenizer=tokenizer,
        max_length=args.cutoff_len,
        begin_idx=args.begin_idx,
        end_idx=args.end_idx,
    )
    if not dataset:
        raise ValueError("The selected test slice is empty.")
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": dataset[index]["input"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for index in range(len(dataset))
    ]
    required_tokens = max(
        len(tokenizer.encode(prompt, add_special_tokens=False)) for prompt in prompts
    ) + args.max_new_tokens
    if required_tokens > args.cutoff_len:
        raise ValueError(
            f"Prompt requires {required_tokens} tokens, but cutoff_len={args.cutoff_len}."
        )
    print(f"samples={len(dataset)}, max_required_tokens={required_tokens}")

    model = LLM(
        model=args.model_path,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.cutoff_len,
        max_num_seqs=args.max_num_seqs,
    )
    sampling = SamplingParams(
        seed=args.random_seed,
        temperature=0,
        max_tokens=args.max_new_tokens,
    )
    outputs = model.generate(prompts, sampling)
    raw_predictions = [output.outputs[0].text for output in outputs]
    predictions = post_process(raw_predictions)

    references = {"target": [dataset[index]["output"] for index in range(len(dataset))]}
    for name in ("original", "rewritten"):
        values = [dataset[index][f"{name}_output"] for index in range(len(dataset))]
        if all(values):
            references[name] = values
    evaluation = HeadlineEvaluation()
    per_reference_scores = {
        name: evaluation.compute_metrics(predictions, labels, average=False)
        for name, labels in references.items()
    }

    results = []
    for index in tqdm(range(len(dataset))):
        item = dataset[index]
        result = {
            "user_id": item["user_id"],
            "input": item["input"],
            "output": raw_predictions[index],
            "predict": predictions[index],
            "label": item["output"],
            "original_label": item["original_output"],
            "rewritten_label": item["rewritten_output"],
        }
        for reference_name, scores in per_reference_scores.items():
            for metric_name, values in scores.items():
                result[f"{metric_name}-{reference_name}"] = values[index]
        results.append(result)

    mean_scores = {
        reference_name: {
            metric_name: float(np.mean(values))
            for metric_name, values in scores.items()
        }
        for reference_name, scores in per_reference_scores.items()
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_end = args.begin_idx + len(dataset)
    (output_dir / f"predictions_{args.begin_idx}-{result_end}.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / f"scores_{args.begin_idx}-{result_end}.json").write_text(
        json.dumps(mean_scores, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(mean_scores, indent=2))


if __name__ == "__main__":
    main()
