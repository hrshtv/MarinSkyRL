import asyncio
import base64
import importlib.util
import io
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
import torch

# batch_artifact is built in parallel; until it lands, route_pass's file I/O imports resolve to an empty stub.
_BATCH_ARTIFACT = "skyrl_train.benchmarks.learner.batch_artifact"
if importlib.util.find_spec(_BATCH_ARTIFACT) is None:
    _stub = types.ModuleType(_BATCH_ARTIFACT)
    _stub.list_ensemble = _stub.read_ensemble_batch = _stub.write_routes = None
    sys.modules[_BATCH_ARTIFACT] = _stub

from skyrl_train.benchmarks.learner.route_pass import (  # noqa: E402
    RouteRequest,
    batch_routes,
    engine_assignments,
    route_requests,
    routed_expert_rows,
)
from skyrl_train.dataset.preprocess import convert_prompts_responses_to_batch_tensors  # noqa: E402
from skyrl_train.models.megatron_router_replay import dense_replay_targets  # noqa: E402
from skyrl_train.training_batch import TrainingInputBatch  # noqa: E402

PAD = 0
LAYERS = 3
TOP_K = 2
MODEL = "hero"


def expert_ids(token_id: int, position: int, num_experts: int) -> np.ndarray:
    """Synthetic [layers, top_k] route that identifies the token and absolute position it was computed at."""
    layer = np.arange(LAYERS)[:, None]
    choice = np.arange(TOP_K)[None, :]
    return (token_id * 31 + position * 7 + layer * 3 + choice) % num_experts


class FakeVllm:
    """Mimics the pinned vLLM fork's /completions route return for a max_tokens=1 prefill."""

    def __init__(self, num_experts: int, rows_delta: int = 0):
        self.num_experts = num_experts
        self.rows_delta = rows_delta
        self.prompts: list[list[int]] = []

    async def completion(self, request_payload):
        body = request_payload["json"]
        assert body["model"] == MODEL and body["max_tokens"] == 1
        tokens = body["prompt"]
        self.prompts.append(tokens)
        # Every prompt position is forwarded; the sampled token is not. A positive rows_delta appends rows as if
        # later positions had been forwarded too.
        forwarded = tokens + [PAD] * max(self.rows_delta, 0)
        routes = np.stack([expert_ids(t, p, self.num_experts) for p, t in enumerate(forwarded)]).astype(np.int16)
        routes = routes[body["routed_experts_prompt_start"] : len(tokens) + self.rows_delta]
        buffer = io.BytesIO()
        np.save(buffer, routes, allow_pickle=False)
        return {"choices": [{"routed_experts": base64.b64encode(buffer.getvalue()).decode("ascii")}]}


def make_batch(prompts, responses) -> TrainingInputBatch:
    sequences, attention_mask, response_mask, *_ = convert_prompts_responses_to_batch_tensors(
        SimpleNamespace(pad_token_id=PAD),
        prompts,
        responses,
        [[0.0] * len(r) for r in responses],
        [[1] * len(r) for r in responses],
    )
    batch = TrainingInputBatch(
        {"sequences": sequences, "attention_mask": attention_mask, "response_mask": response_mask}
    )
    batch.metadata = {"response_length": response_mask.shape[1]}
    return batch


def compute_routes(engines, batch, num_experts):
    return asyncio.run(
        batch_routes(
            engines,
            batch,
            served_model_name=MODEL,
            num_experts=num_experts,
            max_in_flight_per_engine=2,
            request_id_prefix="test",
        )
    )


@pytest.mark.parametrize(("num_experts", "dtype"), [(64, torch.uint8), (384, torch.int16)])
def test_replayed_routes_come_from_the_position_predicting_each_response_token(num_experts, dtype):
    # Mixed widths: longest prompt, longest response, a one-token prompt, and an empty response.
    prompts = [[11, 12, 13, 14], [21, 22], [31], [41, 42, 43]]
    responses = [[15, 16], [23, 24, 25, 26, 27], [32, 33], []]
    batch = make_batch(prompts, responses)
    engines = [FakeVllm(num_experts), FakeVllm(num_experts)]

    routes = compute_routes(engines, batch, num_experts)

    assert sorted(p for e in engines for p in e.prompts) == sorted(p + r for p, r in zip(prompts, responses))
    assert [len(rows) for rows in routes.rows] == [len(r) for r in responses]
    assert routes.dtype == dtype
    sequences = batch["sequences"]
    batch_size, seq_len = sequences.shape
    full, mask = dense_replay_targets(
        routes.materialize(), batch_size, seq_len, batch.metadata["response_length"]
    )
    prompt_width = seq_len - batch.metadata["response_length"]
    for row, (prompt, response) in enumerate(zip(prompts, responses)):
        left_pad = prompt_width - len(prompt)
        columns = torch.nonzero(mask[row]).flatten().tolist()
        assert columns == list(range(prompt_width - 1, prompt_width - 1 + len(response)))
        for column in columns:
            # The route at column c was computed on the token at c, and token c + 1 is the response token it predicts.
            expected = expert_ids(int(sequences[row, column]), column - left_pad, num_experts)
            np.testing.assert_array_equal(full[row, column].numpy(), expected)
            assert int(sequences[row, column + 1]) == response[column - prompt_width + 1]


@pytest.mark.parametrize("rows_delta", [-1, 1])
def test_capture_with_wrong_row_count_is_rejected(rows_delta):
    batch = make_batch([[1, 2, 3]], [[4, 5]])
    with pytest.raises(ValueError, match="token rows"):
        compute_routes([FakeVllm(64, rows_delta)], batch, 64)


def test_completion_error_is_raised():
    class FailingEngine:
        async def completion(self, request_payload):
            return {"error": {"message": "input tokens exceed context", "code": 400}}

    with pytest.raises(ValueError, match="exceed context"):
        compute_routes([FailingEngine()], make_batch([[1, 2]], [[3]]), 64)


def test_route_requests_strip_padding():
    batch = make_batch([[7, 8, 9], [5]], [[10], [6, 4, 3]])
    assert route_requests(batch) == [RouteRequest((7, 8, 9, 10), 3), RouteRequest((5, 6, 4, 3), 1)]


@pytest.mark.parametrize(
    "attention_row",
    [
        [1, 1, 0, 1, 1, 0],  # right-padded prompt: its final token is not next to the response window
        [0, 1, 1, 0, 1, 0],  # response does not start at the response window
        [1, 0, 1, 1, 1, 0],  # hole inside the prompt
    ],
)
def test_route_requests_reject_layouts_the_learner_cannot_replay(attention_row):
    attention_mask = torch.tensor([attention_row])
    batch = TrainingInputBatch(
        {
            "sequences": torch.arange(1, 7).unsqueeze(0),
            "attention_mask": attention_mask,
            "response_mask": attention_mask[:, 3:],
        }
    )
    batch.metadata = {"response_length": 3}
    with pytest.raises(ValueError):
        route_requests(batch)


def test_route_rows_reject_expert_ids_outside_the_model():
    rows = [np.full((2, LAYERS, TOP_K), 64, dtype=np.int32)]
    with pytest.raises(ValueError, match="outside"):
        routed_expert_rows(rows, 4, 64, layers=LAYERS, top_k=TOP_K)


def test_engine_assignments_balance_tokens():
    assignment = engine_assignments([10, 1, 9, 1, 2], 2)
    loads = [sum(n for n, e in zip([10, 1, 9, 1, 2], assignment) if e == engine) for engine in range(2)]
    assert sorted(loads) == [11, 12]
