import json

import torch

from skyrl_train.benchmarks.learner.batch_artifact import LengthProfile, list_ensemble, read_ensemble_batch
from skyrl_train.benchmarks.learner.matched_bank import main
from skyrl_train.benchmarks.learner.route_pass import route_requests
from skyrl_train.training_batch import TrainingBatchIterator

PROMPTS = ([1, 2, 3], [4, 5], [6, 7, 8, 9], [10], [11, 12], [13, 14, 15], [16, 17], [18])
RESPONSES = ([20, 21], [22], [23, 24, 25, 26], [27, 28, 29], [30], [31, 32], [33, 34, 35, 36, 37], [38, 39])
BATCH_SIZE = 4
TOKENIZER = "marin-community/marin-tokenizer"
TOKENIZER_REVISION = "a5ca45f2feb6c959bd87b81689aa7279b5bdcaa2"
EOS_TOKEN_ID = 128001


def _bank_file(tmp_path) -> str:
    bank = {
        "origin": {"job": "/someone/bank-capture", "purpose": "test"},
        "samples": [
            {"row_id": f"train/{index}", "prompt_token_ids": prompt, "response_ids": response}
            for index, (prompt, response) in enumerate(zip(PROMPTS, RESPONSES, strict=True))
        ],
    }
    path = tmp_path / "bank.json"
    path.write_text(json.dumps(bank))
    return str(path)


def _write_ensemble(tmp_path, monkeypatch) -> str:
    monkeypatch.setattr(
        "skyrl_train.benchmarks.learner.matched_bank.create_tokenizer",
        lambda *args, **kwargs: type("Tokenizer", (), {"pad_token_id": EOS_TOKEN_ID}),
    )
    root = str(tmp_path / "ensemble")
    main(
        [
            "--bank",
            _bank_file(tmp_path),
            "--out",
            root,
            "--profile",
            "short",
            "--ensemble-version",
            "v0",
            "--batch-size",
            str(BATCH_SIZE),
            "--tokenizer",
            TOKENIZER,
            "--tokenizer-revision",
            TOKENIZER_REVISION,
            "--model",
            "grug-67b-sft",
        ]
    )
    return root


def test_bank_batches_replay_the_bank_token_work(tmp_path, monkeypatch):
    root = _write_ensemble(tmp_path, monkeypatch)

    batch_dirs = list_ensemble(root, "v0")
    assert len(batch_dirs) == len(PROMPTS) // BATCH_SIZE
    for index, batch_dir in enumerate(batch_dirs):
        batch, manifest = read_ensemble_batch(batch_dir, routes_checkpoint=None)
        prompts = PROMPTS[index * BATCH_SIZE : (index + 1) * BATCH_SIZE]
        responses = RESPONSES[index * BATCH_SIZE : (index + 1) * BATCH_SIZE]
        # The route pass strips padding with the trainer's layout rules, so it must recover each bank pair.
        requests = route_requests(batch)
        assert [request.token_ids for request in requests] == [
            tuple(prompt + response) for prompt, response in zip(prompts, responses, strict=True)
        ]
        assert [request.prompt_length for request in requests] == [len(prompt) for prompt in prompts]
        assert manifest.profile == LengthProfile.SHORT
        assert manifest.tokenizer == f"{TOKENIZER}@{TOKENIZER_REVISION}"
        assert manifest.source["row_ids"] == [f"train/{index * BATCH_SIZE + row}" for row in range(BATCH_SIZE)]
        assert manifest.response_tokens == sum(len(response) for response in responses)
        assert torch.equal(batch["loss_mask"], batch["response_mask"])
        assert torch.all(batch["sequences"][batch["attention_mask"] == 0] == EOS_TOKEN_ID)
        experiences = list(TrainingBatchIterator(batch, sample_batch_size=2))
        assert [experience.num_actions for experience in experiences] == [max(map(len, responses))] * 2


def test_bank_advantages_are_per_sequence_outcome_advantages(tmp_path, monkeypatch):
    root = _write_ensemble(tmp_path, monkeypatch)

    for batch_dir in list_ensemble(root, "v0"):
        batch, _ = read_ensemble_batch(batch_dir, routes_checkpoint=None)
        mask = batch["response_mask"].bool()
        rewards = batch["rewards"].sum(dim=1)
        assert set(rewards.tolist()) == {0.0, 1.0}
        assert torch.all(batch["advantages"][~mask] == 0)
        for row in range(batch.batch_size):
            advantage = batch["advantages"][row, mask[row]]
            # A constant advantage over the response whose sign says whether the sequence succeeded.
            assert torch.all(advantage == advantage[0])
            assert (advantage[0] > 0) == (rewards[row] == 1)
