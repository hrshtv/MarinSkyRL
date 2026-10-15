"""Build learner-benchmark ensemble batches from a matched learner bank.

A matched learner bank is a JSON file of fixed ``(prompt_token_ids, response_ids)`` pairs, used to give model and
hardware comparisons identical token work::

    {"origin": {...}, "samples": [{"row_id": str, "prompt_token_ids": [int], "response_ids": [int]}, ...]}

Unlike a trainer ``dump_data`` pickle, a bank carries no rewards, advantages or log-probs, so this adapter fills
them in:

- rewards: a seeded Bernoulli(0.5) outcome reward per sequence on its final response token;
- advantages and returns: GRPO outcome advantages with the whole batch as one group (each bank prompt appears
  once), so every sequence carries ``+-1``-scale advantages over its response tokens;
- ``action_log_probs``: zeros on response tokens. They stand in for the behavior policy's log-probs; run the
  driver with ``--score-old-logprobs`` to train against the policy's own forward instead;
- ``rollout_logprobs``, ``base_action_log_probs`` and ``values``: ``None``, as for a run without rollout
  log-probs, a reference model or a critic.

Every response token is a loss token. Tensors are collated with the trainer's
``convert_prompts_responses_to_batch_tensors``, so the layout matches ``convert_dump`` output.
"""

import argparse
import dataclasses
import json
from collections.abc import Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from skyrl_train.benchmarks.learner.batch_artifact import (
    EnsembleBatchManifest,
    LengthProfile,
    batch_manifest,
    ensemble_batch_path,
    write_ensemble_batch,
)
from skyrl_train.dataset.preprocess import convert_prompts_responses_to_batch_tensors
from skyrl_train.io import io
from skyrl_train.tokenizer import create_tokenizer
from skyrl_train.training_batch import TrainingInputBatch
from skyrl_train.utils.advantage_estimators import compute_grpo_outcome_advantage


@dataclass(frozen=True)
class BankSample:
    row_id: str
    prompt_token_ids: tuple[int, ...]
    response_ids: tuple[int, ...]


def bank_samples(bank: dict[str, Any]) -> list[BankSample]:
    """Parse the ``samples`` of a matched learner bank, rejecting empty prompts or responses."""
    samples = []
    for entry in bank["samples"]:
        sample = BankSample(entry["row_id"], tuple(entry["prompt_token_ids"]), tuple(entry["response_ids"]))
        if not sample.prompt_token_ids or not sample.response_ids:
            raise ValueError(f"bank sample {sample.row_id} has an empty prompt or response")
        samples.append(sample)
    return samples


def synthetic_rewards(num_sequences: int, seed: int) -> list[float]:
    """Seeded Bernoulli(0.5) outcome rewards with at least one success and one failure."""
    if num_sequences < 2:
        raise ValueError(f"a batch needs at least two sequences for nonzero advantages, got {num_sequences}")
    generator = torch.Generator().manual_seed(seed)
    while True:
        rewards = torch.randint(0, 2, (num_sequences,), generator=generator).tolist()
        if 0 < sum(rewards) < num_sequences:
            return [float(reward) for reward in rewards]


def bank_batch(samples: Sequence[BankSample], *, pad_token_id: int, rewards: Sequence[float]) -> TrainingInputBatch:
    """Collate bank samples into a post-advantage ``TrainingInputBatch`` as the trainer would.

    Args:
        samples: Sequences of the batch, in row order.
        pad_token_id: The policy tokenizer's pad token id.
        rewards: One outcome reward per sample, placed on its final response token.
    """
    if len(rewards) != len(samples):
        raise ValueError(f"{len(rewards)} rewards for {len(samples)} samples")
    prompts = [list(sample.prompt_token_ids) for sample in samples]
    responses = [list(sample.response_ids) for sample in samples]
    token_rewards = [
        [0.0] * (len(response) - 1) + [reward] for response, reward in zip(responses, rewards, strict=True)
    ]
    loss_masks = [[1] * len(response) for response in responses]
    sequences, attention_mask, response_mask, rewards_tensor, loss_mask, _, _, _ = (
        convert_prompts_responses_to_batch_tensors(
            SimpleNamespace(pad_token_id=pad_token_id), prompts, responses, token_rewards, loss_masks, None
        )
    )
    advantages, returns = compute_grpo_outcome_advantage(
        rewards_tensor, response_mask, np.zeros(len(samples), dtype=np.int64)
    )
    batch = TrainingInputBatch(
        {
            "sequences": sequences,
            "attention_mask": attention_mask,
            "response_mask": response_mask,
            "rewards": rewards_tensor,
            "loss_mask": loss_mask,
            "rollout_logprobs": None,
            "rollout_staleness": torch.zeros(len(samples), dtype=torch.int32),
            "is_last_step": None,
            "advantages": advantages,
            "returns": returns,
            "action_log_probs": torch.zeros_like(response_mask, dtype=torch.float32),
            "base_action_log_probs": None,
            "values": None,
        }
    )
    batch.metadata = {"response_length": response_mask.shape[1]}
    return batch


def bank_ensemble(
    bank: dict[str, Any],
    *,
    batch_size: int,
    pad_token_id: int,
    ensemble_version: str,
    profile: LengthProfile,
    source: dict[str, Any],
    tokenizer: str,
    seed: int,
) -> list[tuple[TrainingInputBatch, EnsembleBatchManifest]]:
    """Split a bank into consecutive batches of ``batch_size`` sequences, in bank order.

    Batch ``i`` draws its rewards with seed ``seed + i``; its manifest ``source`` adds the batch's bank row ids.
    """
    samples = bank_samples(bank)
    if len(samples) % batch_size:
        raise ValueError(f"bank of {len(samples)} sequences does not split into batches of {batch_size}")
    batches = []
    for index, start in enumerate(range(0, len(samples), batch_size)):
        chunk = samples[start : start + batch_size]
        batch = bank_batch(chunk, pad_token_id=pad_token_id, rewards=synthetic_rewards(len(chunk), seed + index))
        manifest = batch_manifest(
            batch,
            ensemble_version=ensemble_version,
            profile=profile,
            batch_index=index,
            source={**source, "reward_seed": seed + index, "row_ids": [sample.row_id for sample in chunk]},
            tokenizer=tokenizer,
        )
        batches.append((batch, manifest))
    return batches


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m skyrl_train.benchmarks.learner.matched_bank")
    parser.add_argument("--bank", required=True, help="matched learner bank JSON path or URI")
    parser.add_argument("--out", required=True, help="ensemble root path or URI")
    parser.add_argument("--profile", required=True, choices=[profile.value for profile in LengthProfile])
    parser.add_argument("--ensemble-version", required=True)
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--tokenizer", required=True, help="policy tokenizer HF id")
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--model", required=True, help="model that generated the bank's responses")
    parser.add_argument("--seed", type=int, default=0, help="reward seed of batch 0")
    args = parser.parse_args(argv)

    bank = json.loads(io.read_bytes(args.bank))
    tokenizer = create_tokenizer(args.tokenizer, disable_fast_tokenizer=False, revision=args.tokenizer_revision)
    profile = LengthProfile(args.profile)
    batches = bank_ensemble(
        bank,
        batch_size=args.batch_size,
        pad_token_id=tokenizer.pad_token_id,
        ensemble_version=args.ensemble_version,
        profile=profile,
        source={"bank": args.bank, "origin_job": bank["origin"]["job"], "model": args.model},
        tokenizer=f"{args.tokenizer}@{args.tokenizer_revision}",
        seed=args.seed,
    )
    for batch, manifest in batches:
        path = ensemble_batch_path(args.out, args.ensemble_version, profile, manifest.batch_index)
        write_ensemble_batch(path, batch, manifest)
        summary = {key: value for key, value in dataclasses.asdict(manifest).items() if key != "source"}
        print(json.dumps({"path": path, **summary}))


if __name__ == "__main__":
    main()
