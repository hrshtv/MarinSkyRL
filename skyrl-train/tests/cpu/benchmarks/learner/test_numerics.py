import math

from omegaconf import OmegaConf
import pytest
import torch

from tests.cpu.util import stub_megatron_modules

stub_megatron_modules()

from skyrl_train.benchmarks.learner import numerics  # noqa: E402
from skyrl_train.distributed.megatron import megatron_utils  # noqa: E402
from skyrl_train.benchmarks.learner.numerics import (  # noqa: E402
    CalibrationPolicy,
    NumericsProvenance,
    NumericsRecord,
    Signal,
    SignalKind,
    calibrate_reference,
    canonical_parameter_group,
    compare_to_reference,
    global_layer_indices,
    logprob_signals,
    merge_parameter_statistics,
    numerics_config_fingerprint,
    parameter_delta_statistics,
    parameter_signals,
    read_reference,
    statistics_to_wire,
    write_reference,
)

POLICY = CalibrationPolicy(
    multiplier=4.0,
    floors={SignalKind.RELATIVE: 1e-3, SignalKind.ABSOLUTE: 1e-4, SignalKind.RELATIVE_L2: 1e-3},
)


def _provenance(layout: str = "tp1-ep1") -> NumericsProvenance:
    return NumericsProvenance(
        checkpoint="hero-k4",
        batch_sha256="abc",
        config_fingerprint="cfg",
        num_updates=1,
        layout=layout,
        source_revision="rev",
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

    monkeypatch.setattr(numerics, "STATISTICS_CHUNK_ELEMENTS", 100)
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


def _record(statistics, layout: str = "tp1-ep1", extra: dict[str, Signal] | None = None) -> NumericsRecord:
    signals = parameter_signals(statistics)
    signals["train_status/update0/policy_loss"] = Signal(SignalKind.ABSOLUTE, (0.25,))
    signals.update(extra or {})
    return NumericsRecord(provenance=_provenance(layout), signals=signals)


def _groups(updates):
    return {group: parameter_delta_statistics(before, after) for group, (before, after) in updates.items()}


def test_single_layer_change_and_misplaced_update_fail_only_their_group():
    updates = {
        f"decoder.layers.{layer}.mlp.linear_fc1.weight": _bf16_update((32, 32), seed=layer) for layer in range(3)
    }
    reference = calibrate_reference([_record(_groups(updates)), _record(_groups(updates), layout="tp2-ep1")], POLICY)
    assert compare_to_reference(reference, _record(_groups(updates))).passed

    target = "decoder.layers.1.mlp.linear_fc1.weight"
    before, after = updates[target]
    # Zeroed update for one layer.
    zeroed = dict(updates, **{target: (before, before.clone())})
    failures = {check.name for check in compare_to_reference(reference, _record(_groups(zeroed))).failures}
    assert {f"params/{target}/delta_norm", f"params/{target}/sketch", "params/total_delta_norm"} <= failures
    assert all(target in name or name == "params/total_delta_norm" for name in failures)

    # The right update applied to the wrong rows keeps every norm but breaks the sketch.
    delta = after.float() - before.float()
    rolled = (before.float() + delta.roll(1, dims=0)).to(torch.bfloat16)
    misplaced = dict(updates, **{target: (before, rolled)})
    failures = {check.name for check in compare_to_reference(reference, _record(_groups(misplaced))).failures}
    assert f"params/{target}/sketch" in failures


def test_calibration_scales_largest_pairwise_spread_with_floor():
    def run(loss: float, layout: str) -> NumericsRecord:
        return NumericsRecord(
            provenance=_provenance(layout),
            signals={
                "loss": Signal(SignalKind.ABSOLUTE, (loss,)),
                "grad_norm": Signal(SignalKind.RELATIVE, (2.0,)),
            },
        )

    reference = calibrate_reference([run(1.0, "a"), run(1.01, "b"), run(0.995, "c")], POLICY)
    assert reference.spreads["loss"] == pytest.approx(0.015)
    assert reference.tolerances["loss"] == pytest.approx(0.06)
    # Identical runs leave only the floor.
    assert reference.tolerances["grad_norm"] == POLICY.floors[SignalKind.RELATIVE]

    assert compare_to_reference(reference, run(1.05, "d")).passed
    failures = compare_to_reference(reference, run(1.07, "d")).failures
    assert [check.name for check in failures] == ["loss"]


def test_comparison_fails_missing_extra_and_nonfinite_signals():
    base = {"loss": Signal(SignalKind.ABSOLUTE, (1.0,)), "norm": Signal(SignalKind.RELATIVE, (3.0,))}
    reference = calibrate_reference(
        [NumericsRecord(_provenance(), base), NumericsRecord(_provenance("b"), base)], POLICY
    )
    missing = NumericsRecord(_provenance(), {"loss": base["loss"]})
    extra = NumericsRecord(_provenance(), {**base, "new": Signal(SignalKind.ABSOLUTE, (0.0,))})
    nonfinite = NumericsRecord(_provenance(), {**base, "loss": Signal(SignalKind.ABSOLUTE, (math.nan,))})
    assert [check.name for check in compare_to_reference(reference, missing).failures] == ["norm"]
    assert [check.name for check in compare_to_reference(reference, extra).failures] == ["new"]
    assert [check.name for check in compare_to_reference(reference, nonfinite).failures] == ["loss"]

    other_batch = NumericsRecord(NumericsProvenance(**{**_provenance().__dict__, "batch_sha256": "other"}), base)
    with pytest.raises(ValueError, match="provenance"):
        compare_to_reference(reference, other_batch)


def test_reference_json_round_trip_preserves_verdicts(tmp_path):
    updates = {"decoder.layers.0.self_attention.linear_qkv.weight": _bf16_update((16, 8), seed=3)}
    logprobs = torch.log(torch.rand(4, 10, generator=torch.Generator().manual_seed(4)))
    mask = torch.ones(4, 10)
    extra = logprob_signals("logprobs/pre_update", logprobs, mask)
    reference = calibrate_reference(
        [_record(_groups(updates), extra=extra), _record(_groups(updates), layout="b", extra=extra)], POLICY
    )
    path = str(tmp_path / "reference.json")
    write_reference(path, reference)
    loaded = read_reference(path)
    assert loaded == reference

    shifted = logprob_signals("logprobs/pre_update", logprobs + 1e-2, mask)
    candidate = _record(_groups(updates), extra=shifted)
    assert compare_to_reference(loaded, candidate).failures == compare_to_reference(reference, candidate).failures
    assert not compare_to_reference(loaded, candidate).passed


def test_logprob_signals_follow_masked_tokens_only(monkeypatch):
    monkeypatch.setattr(numerics, "LOGPROB_SAMPLE_SIZE", 16)
    generator = torch.Generator().manual_seed(5)
    logprobs = -torch.rand(6, 20, generator=generator)
    mask = torch.zeros(6, 20)
    mask[:, :12] = 1
    reference = logprob_signals("lp", logprobs, mask)
    sampled = reference["lp/sampled_tokens"].values
    assert len(sampled) == 16
    assert set(sampled) <= set(logprobs[:, :12].reshape(-1).tolist())

    padding_changed = logprobs.clone()
    padding_changed[:, 12:] = 0.0
    assert logprob_signals("lp", padding_changed, mask) == reference

    token_changed = logprobs.clone()
    token_changed[2, 3] -= 0.5
    changed = logprob_signals("lp", token_changed, mask)
    assert changed["lp/sequence_mean"] != reference["lp/sequence_mean"]
    assert changed["lp/token_mean"] != reference["lp/token_mean"]


def test_config_fingerprint_ignores_layout_but_not_optimizer():
    cfg = OmegaConf.create(
        {
            "trainer": {
                "seed": 42,
                "algorithm": {"policy_loss_type": "regular"},
                "policy": {"optimizer_config": {"lr": 1e-6}, "megatron_config": {"tensor_model_parallel_size": 1}},
                "policy_mini_batch_size": 64,
                "update_epochs_per_batch": 1,
                "micro_train_batch_size_per_gpu": 1,
            },
            "generator": {"sampling_params": {"temperature": 1.0}},
        }
    )
    fingerprint = numerics_config_fingerprint(cfg)
    relayout = cfg.copy()
    relayout.trainer.policy.megatron_config.tensor_model_parallel_size = 4
    relayout.trainer.micro_train_batch_size_per_gpu = 2
    assert numerics_config_fingerprint(relayout) == fingerprint
    retuned = cfg.copy()
    retuned.trainer.policy.optimizer_config.lr = 2e-6
    assert numerics_config_fingerprint(retuned) != fingerprint


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
