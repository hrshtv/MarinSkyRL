import dataclasses

import pytest

from skyrl_train.benchmarks.learner.flops import (
    Accelerator,
    GrugFlopsModel,
    RecomputePolicy,
    accelerator_from_device_name,
    causal_attention_pairs,
    mfu,
)
from skyrl_train.models.grug_moe import GrugMoeConfig

HERO_SHAPE = dict(
    vocab_size=128256,
    hidden_size=6144,
    intermediate_size=3072,
    shared_expert_intermediate_size=3072,
    num_shared_experts=2,
    num_local_experts=384,
    num_experts_per_tok=8,
    num_hidden_layers=48,
    num_attention_heads=48,
    num_key_value_heads=12,
    local_kv_heads=12,
    global_kv_heads=6,
    head_dim=128,
    sliding_window=2048,
    rope_fused=True,
    latent_dim=3072,
    grugmoe_artifact_schema_version=2,
)
# Linear (non-attention-score) forward FLOPs per token that Marin's Hero pretraining MFU counts:
# experiments/grug/moe_hero_ep/train.py::_compute_flops (Levanter lm_flops_per_token plus the LatentMoE correction)
# at seq_len=16384, minus Levanter's attention-map terms for 12 global and 36 sliding-window layers.
MARIN_HERO_LINEAR_FLOPS_PER_TOKEN = 46_874_492_928
# Marin's whole training count at seq_len=16384 (3x forward), the harness's sanity target.
MARIN_HERO_TRAINING_FLOPS_PER_TOKEN_16K = 160.671596544e9


def _hero(**overrides) -> GrugFlopsModel:
    return GrugFlopsModel.from_config(GrugMoeConfig(**{**HERO_SHAPE, **overrides}))


@pytest.mark.parametrize(("seq_len", "window"), [(0, 4), (1, 4), (4, 4), (11, 4), (11, None), (7, 1)])
def test_causal_attention_pairs_matches_enumeration(seq_len, window):
    expected = sum(1 for query in range(seq_len) for key in range(query + 1) if window is None or query - key < window)
    assert causal_attention_pairs(seq_len, window) == expected


def test_hero_linear_flops_equal_marin_count_plus_the_terms_it_omits():
    model = _hero(sconv=True)
    d, rank = 6144, 128
    gated_norms = (2 * 48 + 2) * 2 * 2 * d * rank
    attention_gate = 48 * 2 * d * 48
    # ShortConv k (KV width), attn and mlp (hidden width) sites, kernel 4.
    sconv = 36 * 2 * 4 * (12 * 128 + 2 * d) + 12 * 2 * 4 * (6 * 128 + 2 * d)
    omitted = gated_norms + attention_gate + sconv
    assert model.stack_linear_flops_per_token() == MARIN_HERO_LINEAR_FLOPS_PER_TOKEN + omitted


def test_hero_training_flops_sit_below_marin_noncausal_count_at_16k():
    model = _hero(sconv=True)
    per_token = model.sequence_flops(16384, RecomputePolicy.NONE).training_model / 16384
    # Causal masking halves the long layers' attention map, which Marin prices as a full square.
    assert 0.95 * MARIN_HERO_TRAINING_FLOPS_PER_TOKEN_16K < per_token < MARIN_HERO_TRAINING_FLOPS_PER_TOKEN_16K


def test_long_layers_ignore_the_sliding_window():
    model = _hero()
    all_local = dataclasses.replace(model, long_layers=(False,) * model.num_layers)
    all_long = dataclasses.replace(model, long_layers=(True,) * model.num_layers)
    assert all_local.attention_score_flops(2048) == all_long.attention_score_flops(2048)
    # Past the window, sliding-window cost per added token is flat; full causal cost keeps growing.
    local_step = all_local.attention_score_flops(8193) - all_local.attention_score_flops(8192)
    assert local_step == all_local.attention_score_flops(4097) - all_local.attention_score_flops(4096)
    assert all_long.attention_score_flops(8193) - all_long.attention_score_flops(8192) > local_step


def test_hardware_flops_add_recompute_and_stored_kv_heads():
    uniform_kv = _hero(local_kv_heads=None, global_kv_heads=None)
    flops = uniform_kv.sequence_flops(4096, RecomputePolicy.NONE)
    assert flops.training_hardware == flops.training_model

    split_kv = _hero()
    flops = split_kv.sequence_flops(4096, RecomputePolicy.NONE)
    # Global layers project 12 stored KV heads and keep 6: 12 layers x K and V x 6 heads x 128 x hidden.
    assert flops.forward_hardware_excess == 4096 * 12 * 2 * (2 * 6 * 128) * 6144
    assert flops.training_hardware == flops.training_model + 3 * flops.forward_hardware_excess

    full = split_kv.sequence_flops(4096, RecomputePolicy.FULL_LAYERS)
    lm_head_and_outer_norms = 4096 * (2 * 6144 * 128256 + 2 * 2 * 2 * 6144 * 128)
    assert full.recompute == full.forward - lm_head_and_outer_norms
    core = split_kv.sequence_flops(4096, RecomputePolicy.CORE_ATTENTION)
    assert core.recompute == split_kv.attention_score_flops(4096)


def test_batch_flops_add_sequences():
    model = _hero()
    batch = model.batch_flops([100, 3000], RecomputePolicy.FULL_LAYERS)
    one = model.sequence_flops(100, RecomputePolicy.FULL_LAYERS)
    two = model.sequence_flops(3000, RecomputePolicy.FULL_LAYERS)
    assert batch.tokens == 3100
    assert batch.training_model == pytest.approx(one.training_model + two.training_model, rel=1e-12)


@pytest.mark.parametrize(
    ("device_name", "accelerator"),
    [("NVIDIA H100 80GB HBM3", Accelerator.H100), ("NVIDIA GB200", Accelerator.GB200)],
)
def test_accelerator_from_device_name(device_name, accelerator):
    assert accelerator_from_device_name(device_name) is accelerator


def test_unknown_device_has_no_peak():
    with pytest.raises(ValueError, match="no unique peak-FLOP entry"):
        accelerator_from_device_name("NVIDIA A100-SXM4-80GB")


def test_mfu_divides_by_cluster_peak():
    assert mfu(989e12, 2.0, 4, Accelerator.H100) == pytest.approx(0.125)
