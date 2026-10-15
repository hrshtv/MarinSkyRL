import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from skyrl_train.benchmarks.learner.batch_artifact import (
    LengthProfile,
    convert_dump,
    ensemble_batch_path,
    list_ensemble,
    main,
    read_ensemble_batch,
    write_ensemble_batch,
    write_routes,
)
from skyrl_train.dataset.preprocess import convert_prompts_responses_to_batch_tensors
from skyrl_train.dataset.routed_expert_batch import RoutedExpertRows
from skyrl_train.models.megatron_router_replay import dense_replay_targets
from skyrl_train.trainer import RayPPOTrainer
from skyrl_train.training_batch import TrainingBatchIterator, TrainingInputBatch

PROMPT_LENGTHS = (3, 5, 2, 4)
RESPONSE_LENGTHS = (4, 1, 6, 3)
DATA_PARALLEL_SIZE = 3
GLOBAL_STEP = 7
HERO_EXPERTS = 384
HERO_LAYERS = 3
TOP_K = 2


def _generated_batch() -> TrainingInputBatch:
    """A batch as the trainer collates it: left-padded prompts, right-padded responses, generator routes."""
    prompts = [list(range(10, 10 + n)) for n in PROMPT_LENGTHS]
    responses = [list(range(100, 100 + n)) for n in RESPONSE_LENGTHS]
    loss_masks = [[1] * (n - 1) + [0] for n in RESPONSE_LENGTHS]
    logprobs = [np.linspace(-1.0, -0.1, n) for n in RESPONSE_LENGTHS]
    rewards = [[0.0] * n for n in RESPONSE_LENGTHS]
    sequences, attention_mask, response_mask, _, loss_mask, rollout_logprobs, _, _ = (
        convert_prompts_responses_to_batch_tensors(
            SimpleNamespace(pad_token_id=0), prompts, responses, rewards, loss_masks, logprobs
        )
    )
    snowball_routes = RoutedExpertRows(
        tuple(np.full((n, 26, 8), 255, dtype=np.uint8) for n in RESPONSE_LENGTHS), response_mask.shape[1], 256
    )
    batch = TrainingInputBatch(
        {
            "sequences": sequences,
            "attention_mask": attention_mask,
            "response_mask": response_mask,
            "loss_mask": loss_mask,
            "rollout_logprobs": rollout_logprobs,
            "rollout_staleness": torch.tensor([0, 1, 0, 2], dtype=torch.int32),
        },
        routed_expert_rows=snowball_routes,
    )
    batch.metadata = {
        "uids": [f"uid{i}" for i in range(len(PROMPT_LENGTHS))],
        "response_length": response_mask.shape[1],
        "avg_response_length": sum(RESPONSE_LENGTHS) / len(RESPONSE_LENGTHS),
    }
    return batch


def _dumped_batch(tmp_path: Path) -> tuple[str, TrainingInputBatch]:
    """Pad, score and dump a batch with the production trainer methods, in the trainer's order."""
    rank = SimpleNamespace(dp_size=DATA_PARALLEL_SIZE)
    trainer = SimpleNamespace(
        policy_model=SimpleNamespace(actor_infos=[SimpleNamespace(rank=rank)]),
        critic_model=None,
        ref_model=None,
        cfg=SimpleNamespace(trainer=SimpleNamespace(export_path=str(tmp_path / "export"))),
    )
    batch = RayPPOTrainer.pad_batch(trainer, _generated_batch())
    assert batch.metadata["pad_size"] == 2
    # The forward pass and advantage estimation run on the padded batch; without a ref or critic model the
    # reference log-probs and values are None.
    advantages = torch.randn(batch["response_mask"].shape, generator=torch.Generator().manual_seed(0))
    batch["advantages"] = advantages * batch["response_mask"]
    batch["returns"] = batch["advantages"].clone()
    batch["action_log_probs"] = batch["rollout_logprobs"] + 0.01
    batch["base_action_log_probs"] = None
    batch["values"] = None
    RayPPOTrainer.dump_data(trainer, batch, file_name=f"global_step_{GLOBAL_STEP}_training_input")
    return str(tmp_path / "export" / "dumped_data" / f"global_step_{GLOBAL_STEP}_training_input.pkl"), batch


def _dump(tmp_path: Path) -> str:
    return _dumped_batch(tmp_path)[0]


def _store(tmp_path: Path, profile: LengthProfile = LengthProfile.SHORT, index: int = 0) -> str:
    batch, manifest = convert_dump(
        _dump(tmp_path),
        ensemble_version="v0",
        profile=profile,
        batch_index=index,
        source={"run": "snowball-test", "model": "snowball"},
        tokenizer="org/tokenizer@rev",
    )
    path = ensemble_batch_path(str(tmp_path / "ensemble"), "v0", profile, index)
    write_ensemble_batch(path, batch, manifest)
    return path


def _hero_routes(response_lengths: tuple[int, ...], response_len: int) -> RoutedExpertRows:
    rows = []
    for sequence, n in enumerate(response_lengths):
        token = np.arange(n).reshape(n, 1, 1)
        layer = np.arange(HERO_LAYERS).reshape(1, HERO_LAYERS, 1)
        choice = np.arange(TOP_K).reshape(1, 1, TOP_K)
        rows.append((1 + sequence * 50 + token * 7 + layer * 3 + choice * 100).astype(np.int32))
    return RoutedExpertRows(tuple(rows), response_len, HERO_EXPERTS)


def test_round_trip_keeps_real_rows_and_drops_generator_routes(tmp_path):
    dump_path, dumped = _dumped_batch(tmp_path)
    converted, manifest = convert_dump(
        dump_path, ensemble_version="v0", profile=LengthProfile.SHORT, batch_index=0, source={}, tokenizer="tok"
    )
    path = ensemble_batch_path(str(tmp_path / "ensemble"), "v0", LengthProfile.SHORT, 0)
    write_ensemble_batch(path, converted, manifest)

    batch, read_manifest = read_ensemble_batch(path, routes_checkpoint=None)

    assert read_manifest == manifest
    assert batch.batch_size == len(PROMPT_LENGTHS)
    assert batch.routed_expert_rows is None
    assert batch.metadata == {"response_length": max(RESPONSE_LENGTHS)}
    assert set(batch) == set(dumped)
    for key, value in dumped.items():
        if value is None:
            assert batch[key] is None
        else:
            assert torch.equal(batch[key], value[: len(PROMPT_LENGTHS)]), key
    assert manifest.num_sequences == len(PROMPT_LENGTHS)
    assert manifest.total_tokens == sum(PROMPT_LENGTHS) + sum(RESPONSE_LENGTHS)
    assert manifest.response_tokens == sum(RESPONSE_LENGTHS)
    assert manifest.source["global_step"] == GLOBAL_STEP
    experiences = list(TrainingBatchIterator(batch, sample_batch_size=2))
    assert [experience.num_actions for experience in experiences] == [max(RESPONSE_LENGTHS)] * 2


def test_attached_routes_land_on_response_prediction_positions(tmp_path):
    path = _store(tmp_path)
    routes = _hero_routes(RESPONSE_LENGTHS, max(RESPONSE_LENGTHS))
    write_routes(path, "hero-step1000", routes)

    batch, _ = read_ensemble_batch(path, routes_checkpoint="hero-step1000")

    seq_len = batch["sequences"].shape[1]
    num_actions = batch.metadata["response_length"]
    full, mask = dense_replay_targets(batch.routed_experts_tensor(), batch.batch_size, seq_len, num_actions)
    prompt_end = seq_len - num_actions
    for sequence, n in enumerate(RESPONSE_LENGTHS):
        for token in range(n):
            # The position predicting response token `token` is the one just before it.
            predictor = prompt_end + token - 1
            assert batch["sequences"][sequence, predictor + 1] == 100 + token
            assert np.array_equal(full[sequence, predictor].numpy(), routes.rows[sequence][token])
        assert mask[sequence].nonzero().flatten().tolist() == list(range(prompt_end - 1, prompt_end - 1 + n))


def test_routes_misaligned_with_response_tokens_are_rejected(tmp_path):
    path = _store(tmp_path)
    short_by_one = (RESPONSE_LENGTHS[0] - 1, *RESPONSE_LENGTHS[1:])
    write_routes(path, "hero", _hero_routes(short_by_one, max(RESPONSE_LENGTHS)))

    with pytest.raises(ValueError, match="route rows"):
        read_ensemble_batch(path, routes_checkpoint="hero")


def test_corrupted_tensors_are_detected(tmp_path):
    path = _store(tmp_path)
    tensors = Path(path) / "tensors.safetensors"
    payload = bytearray(tensors.read_bytes())
    payload[-1] ^= 0xFF
    tensors.write_bytes(bytes(payload))

    with pytest.raises(ValueError, match="sha256"):
        read_ensemble_batch(path, routes_checkpoint=None)


@pytest.mark.parametrize("field, value", [("schema_version", 2), ("profile", "medium")])
def test_unknown_schema_or_profile_is_rejected(tmp_path, field, value):
    path = _store(tmp_path)
    manifest_path = Path(path) / "manifest.json"
    fields = json.loads(manifest_path.read_text())
    fields[field] = value
    manifest_path.write_text(json.dumps(fields))

    with pytest.raises(ValueError):
        read_ensemble_batch(path, routes_checkpoint=None)


def test_list_ensemble_orders_by_profile_then_numeric_index(tmp_path):
    root = str(tmp_path / "ensemble")
    batch, manifest = convert_dump(
        _dump(tmp_path),
        ensemble_version="v0",
        profile=LengthProfile.SHORT,
        batch_index=0,
        source={},
        tokenizer="tok",
    )
    for profile, index in [
        (LengthProfile.AGENTIC, 0),
        (LengthProfile.SHORT, 10),
        (LengthProfile.LONG_REASONING, 0),
        (LengthProfile.SHORT, 2),
    ]:
        stored = dataclasses.replace(manifest, profile=profile, batch_index=index)
        write_ensemble_batch(ensemble_batch_path(root, "v0", profile, index), batch, stored)

    assert list_ensemble(root, "v0") == [
        f"{root}/v0/short/batch_2",
        f"{root}/v0/short/batch_10",
        f"{root}/v0/long_reasoning/batch_0",
        f"{root}/v0/agentic/batch_0",
    ]


def test_convert_cli_writes_a_readable_batch(tmp_path):
    root = str(tmp_path / "ensemble")
    main(
        [
            "convert",
            "--dump",
            _dump(tmp_path),
            "--out",
            root,
            "--profile",
            "agentic",
            "--ensemble-version",
            "v0",
            "--batch-index",
            "3",
            "--tokenizer",
            "org/tokenizer@rev",
            "--run",
            "snowball-test",
            "--revision",
            "abc123",
            "--model",
            "snowball",
        ]
    )

    [path] = list_ensemble(root, "v0")
    batch, manifest = read_ensemble_batch(path, routes_checkpoint=None)
    assert (manifest.profile, manifest.batch_index, manifest.source["revision"]) == (LengthProfile.AGENTIC, 3, "abc123")
    assert torch.equal(batch["sequences"], _generated_batch()["sequences"])
