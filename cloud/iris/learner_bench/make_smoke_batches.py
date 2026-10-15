"""Write a small "short"-profile learner-benchmark ensemble from GSM8K reference solutions.

Each row is a GSM8K test question (prompt) followed by its reference solution (response), tokenized with the
Hero tokenizer and laid out as the trainer does: a left-padded prompt window then a right-padded response window.
Advantages are a fixed +/-1 per row. The batches carry real text, so policy log-probs on them are meaningful, but
the length distribution is GSM8K's, not an RL run's.
"""

import argparse

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from skyrl_train.benchmarks.learner.batch_artifact import (
    LengthProfile,
    batch_manifest,
    ensemble_batch_path,
    write_ensemble_batch,
)
from skyrl_train.training_batch import TrainingInputBatch

GSM8K_DATASET = "openai/gsm8k"
GSM8K_CONFIG = "main"
GSM8K_SPLIT = "test"


def build_batch(prompts: list[list[int]], responses: list[list[int]], pad_token_id: int) -> TrainingInputBatch:
    prompt_width = max(len(prompt) for prompt in prompts)
    response_width = max(len(response) for response in responses)
    sequences, attention, response_masks = [], [], []
    for prompt, response in zip(prompts, responses, strict=True):
        left = prompt_width - len(prompt)
        right = response_width - len(response)
        sequences.append([pad_token_id] * left + prompt + response + [pad_token_id] * right)
        attention.append([0] * left + [1] * (len(prompt) + len(response)) + [0] * right)
        response_masks.append([1] * len(response) + [0] * right)
    rows = len(prompts)
    response_mask = torch.tensor(response_masks, dtype=torch.long)
    zeros = torch.zeros(rows, response_width, dtype=torch.float32)
    sign = torch.tensor([1.0 if row % 2 == 0 else -1.0 for row in range(rows)]).unsqueeze(1)
    batch = TrainingInputBatch(
        {
            "sequences": torch.tensor(sequences, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
            "response_mask": response_mask,
            "loss_mask": response_mask.clone(),
            "advantages": sign * response_mask.float(),
            "returns": sign * response_mask.float(),
            "action_log_probs": zeros.clone(),
            "base_action_log_probs": zeros.clone(),
            "rollout_logprobs": zeros.clone(),
            "values": None,
        }
    )
    batch.metadata = {"response_length": response_width}
    return batch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--ensemble-version", required=True)
    parser.add_argument("--num-batches", type=int, required=True)
    parser.add_argument("--rows-per-batch", type=int, required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, revision=args.tokenizer_revision)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    dataset = load_dataset(GSM8K_DATASET, GSM8K_CONFIG, split=GSM8K_SPLIT)
    needed = args.num_batches * args.rows_per_batch
    if needed > len(dataset):
        raise ValueError(f"need {needed} GSM8K rows, the {GSM8K_SPLIT} split has {len(dataset)}")
    for batch_index in range(args.num_batches):
        start = batch_index * args.rows_per_batch
        rows = dataset.select(range(start, start + args.rows_per_batch))
        prompts = [tokenizer.encode(f"Question: {row['question']}\nAnswer:", add_special_tokens=True) for row in rows]
        responses = [
            tokenizer.encode(f" {row['answer']}", add_special_tokens=False) + [tokenizer.eos_token_id] for row in rows
        ]
        batch = build_batch(prompts, responses, pad_token_id)
        manifest = batch_manifest(
            batch,
            ensemble_version=args.ensemble_version,
            profile=LengthProfile.SHORT,
            batch_index=batch_index,
            source={
                "kind": "gsm8k_reference_solutions",
                "dataset": f"{GSM8K_DATASET}/{GSM8K_CONFIG}/{GSM8K_SPLIT}",
                "rows": [start, start + args.rows_per_batch],
            },
            tokenizer=f"{args.tokenizer}@{args.tokenizer_revision}",
        )
        path = ensemble_batch_path(args.output_root, args.ensemble_version, LengthProfile.SHORT, batch_index)
        write_ensemble_batch(path, batch, manifest)
        print(
            f"wrote {path}: rows={manifest.num_sequences} tokens={manifest.total_tokens} "
            f"response_tokens={manifest.response_tokens} width={tuple(batch['sequences'].shape)}",
            flush=True,
        )


if __name__ == "__main__":
    main()
