import math

import pytest
import torch

from tests.cpu.util import stub_megatron_modules

stub_megatron_modules()

from skyrl_train.distributed.megatron import megatron_utils, parameter_fingerprint  # noqa: E402
from skyrl_train.distributed.megatron.parameter_fingerprint import (  # noqa: E402
    canonical_parameter_group,
    global_layer_indices,
    merge_parameter_statistics,
    parameter_delta_statistics,
    statistics_to_wire,
)


def _bf16_update(shape, seed: int, scale: float = 1e-2) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    before = torch.randn(shape, generator=generator).to(torch.bfloat16)
    after = (before.float() + scale * torch.randn(shape, generator=generator)).to(torch.bfloat16)
    return before, after


def _assert_statistics_close(actual, expected):
    assert actual.numel == expected.numel
    for got, want in zip(statistics_to_wire(actual), statistics_to_wire(expected), strict=True):
        assert math.isclose(got, want, rel_tol=1e-12, abs_tol=1e-12), (got, want)


def _sum(statistics):
    total = statistics[0]
    for item in statistics[1:]:
        total = total + item
    return total


def test_fingerprint_ignores_sharding_chunking_and_element_order(monkeypatch):
    before, after = _bf16_update((64, 48), seed=0)
    whole = parameter_delta_statistics(before, after)

    # Tensor-parallel slices along either dimension.
    for dim in (0, 1):
        shards = [
            parameter_delta_statistics(b, a)
            for b, a in zip(before.chunk(4, dim=dim), after.chunk(4, dim=dim), strict=True)
        ]
        _assert_statistics_close(_sum(shards), whole)

    # A layout that interleaves rows (fused GLU halves, QKV groups) permutes elements consistently.
    permutation = torch.randperm(before.shape[0], generator=torch.Generator().manual_seed(1))
    _assert_statistics_close(parameter_delta_statistics(before[permutation], after[permutation]), whole)

    monkeypatch.setattr(parameter_fingerprint, "STATISTICS_CHUNK_ELEMENTS", 100)
    _assert_statistics_close(parameter_delta_statistics(before, after), whole)


def test_canonical_groups_match_across_pipeline_and_expert_layouts():
    """Two pipeline stages x two expert ranks must fingerprint like one rank holding everything."""
    num_layers, num_experts = 4, 4
    updates = {
        (layer, expert): _bf16_update((8, 6), seed=10 * layer + expert)
        for layer in range(num_layers)
        for expert in range(num_experts)
    }
    attention = {layer: _bf16_update((6, 6), seed=100 + layer) for layer in range(num_layers)}

    def rank_wire(layers: range, experts: range) -> dict[str, list[float]]:
        modules = [(f"module.module.decoder.layers.{local}", _Layer(layer)) for local, layer in enumerate(layers)]
        global_by_local = global_layer_indices(modules)
        totals = {}
        for local, layer in enumerate(layers):
            # Expert ranks replicate the dense weight; only the first owns it.
            names_and_updates = (
                [(f"module.module.decoder.layers.{local}.self_attention.linear_proj.weight", attention[layer])]
                if experts.start == 0
                else []
            )
            names_and_updates += [
                (
                    f"module.module.decoder.layers.{local}.mlp.experts.linear_fc1.weight{local_expert}",
                    updates[layer, expert],
                )
                for local_expert, expert in enumerate(experts)
            ]
            for name, (before, after) in names_and_updates:
                group = canonical_parameter_group(name, global_by_local)
                statistics = parameter_delta_statistics(before, after)
                totals[group] = totals[group] + statistics if group in totals else statistics
        return {group: statistics_to_wire(statistics) for group, statistics in totals.items()}

    single = merge_parameter_statistics([rank_wire(range(4), range(4))])
    split = merge_parameter_statistics(
        [rank_wire(stage, experts) for stage in (range(0, 2), range(2, 4)) for experts in (range(0, 2), range(2, 4))]
    )
    assert single.keys() == split.keys()
    assert "decoder.layers.3.mlp.experts.linear_fc1.weight" in single
    for group in single:
        _assert_statistics_close(split[group], single[group])


class _Layer:
    def __init__(self, global_index: int):
        self.layer_number = global_index + 1


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        (
            "module.decoder.layers.1.mlp.experts.local_experts.5.linear_fc2.weight",
            "decoder.layers.7.mlp.experts.local_experts.linear_fc2.weight",
        ),
        (
            "decoder.layers.0.mlp.shared_experts.experts.1.linear_fc1.weight",
            "decoder.layers.6.mlp.shared_experts.experts.1.linear_fc1.weight",
        ),
        ("module.module.output_layer.weight", "output_layer.weight"),
    ],
)
def test_canonical_group_names(name, expected):
    assert canonical_parameter_group(name, {0: 6, 1: 7}) == expected


class _Chunk:
    def __init__(self, modules, parameters):
        self._modules = modules
        self._parameters = parameters

    def named_modules(self):
        return self._modules

    def named_parameters(self):
        return self._parameters


def _parameter(tensor: torch.Tensor, **attributes) -> torch.nn.Parameter:
    parameter = torch.nn.Parameter(tensor.clone(), requires_grad=False)
    for name, value in attributes.items():
        setattr(parameter, name, value)
    return parameter


def test_owned_shards_count_every_element_once(monkeypatch):
    """TP2 x DP2 with EP2 carved from DP (expert TP1): owned shards sum to the unsharded fingerprint."""
    dense_before, dense_after = _bf16_update((8, 4), seed=20)
    norm_before, norm_after = _bf16_update((4,), seed=21)
    experts = [_bf16_update((6, 4), seed=30 + expert) for expert in range(4)]
    prefix = "module.decoder.layers.0"
    layer_modules = [(prefix, _Layer(5))]

    def rank_chunk(tp: int, dp: int, which: int) -> _Chunk:
        parameters = [
            (
                f"{prefix}.self_attention.linear_proj.weight",
                _parameter(
                    dense_before.chunk(2)[tp] if which == 0 else dense_after.chunk(2)[tp], tensor_model_parallel=True
                ),
            ),
            (f"{prefix}.input_layernorm.weight", _parameter((norm_before, norm_after)[which])),
        ]
        parameters += [
            (
                f"{prefix}.mlp.experts.linear_fc1.weight{local}",
                _parameter(experts[2 * dp + local][which], allreduce=False),
            )
            for local in range(2)
        ]
        return _Chunk(layer_modules, parameters)

    merged = []
    for tp in range(2):
        for dp in range(2):
            ranks = {
                "get_data_parallel_rank": lambda with_context_parallel, dp=dp: dp,
                "get_expert_data_parallel_rank": lambda tp=tp: tp,
                "get_tensor_model_parallel_rank": lambda tp=tp: tp,
                "get_expert_tensor_parallel_rank": lambda: 0,
            }
            for name, function in ranks.items():
                monkeypatch.setattr(megatron_utils.mpu, name, function, raising=False)
            before = {
                key: parameter
                for key, _, parameter in megatron_utils.numerics_parameter_shards([rank_chunk(tp, dp, 0)])
            }
            totals = {}
            for key, group, parameter in megatron_utils.numerics_parameter_shards([rank_chunk(tp, dp, 1)]):
                statistics = parameter_delta_statistics(before[key], parameter)
                totals[group] = totals[group] + statistics if group in totals else statistics
            merged.append({group: statistics_to_wire(statistics) for group, statistics in totals.items()})

    observed = merge_parameter_statistics(merged)
    expected = {
        "decoder.layers.5.self_attention.linear_proj.weight": parameter_delta_statistics(dense_before, dense_after),
        "decoder.layers.5.input_layernorm.weight": parameter_delta_statistics(norm_before, norm_after),
        "decoder.layers.5.mlp.experts.linear_fc1.weight": _sum(
            [parameter_delta_statistics(*update) for update in experts]
        ),
    }
    assert observed.keys() == expected.keys()
    for group, statistics in expected.items():
        _assert_statistics_close(observed[group], statistics)
