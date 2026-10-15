import json
from pathlib import Path
from uuid import uuid4

import fsspec
import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer, models
from transformers import AutoTokenizer, PreTrainedTokenizerFast

from marinskyrl.model_manifest import (
    HF_WEIGHT_INDEX_FILENAME,
    MODEL_MANIFEST_FILENAME,
    ModelManifest,
    sha256_file,
    write_local_model_manifest,
)
from skyrl_train.benchmarks.learner.truncate_checkpoint import (
    read_truncation_provenance,
    slice_routes,
    truncate_checkpoint,
)
from skyrl_train.dataset.routed_expert_batch import RoutedExpertRows
from skyrl_train.models.grug_moe import GrugMoeConfig, GrugMoeForCausalLM

SOURCE_LAYERS = 8
KEPT_LAYERS = 4
VOCAB = 64


def _save_tokenizer(path: Path) -> None:
    vocab = {"<unk>": 0, **{f"t{index}": index for index in range(1, VOCAB)}}
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(models.WordLevel(vocab, unk_token="<unk>")))
    tokenizer.unk_token = "<unk>"
    tokenizer.save_pretrained(path)


def _snowball_config() -> GrugMoeConfig:
    return GrugMoeConfig(
        vocab_size=VOCAB,
        hidden_size=32,
        intermediate_size=16,
        shared_expert_intermediate_size=24,
        num_local_experts=8,
        num_experts_per_tok=2,
        num_hidden_layers=SOURCE_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        sliding_window=3,
        initializer_range=0.2,
    )


def _hero_config() -> GrugMoeConfig:
    return GrugMoeConfig(
        **{
            **_snowball_config().to_dict(),
            "latent_dim": 8,
            "num_shared_experts": 2,
            "local_kv_heads": 2,
            "global_kv_heads": 1,
            "rope_fused": True,
            "sconv": True,
            "grugmoe_artifact_schema_version": 2,
        }
    )


def _hero_state(config: GrugMoeConfig) -> dict[str, torch.Tensor]:
    """All Hero parameter families with split per-expert weights, as in the Megatron bridge."""
    generator = torch.Generator().manual_seed(5)

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator).to(torch.bfloat16)

    hidden, latent, ffn = config.hidden_size, config.latent_dim, config.intermediate_size
    state = {
        "model.embed_tokens.weight": randn(VOCAB, hidden),
        "model.embed_norm.weight": randn(hidden),
        "model.embed_gated_norm.down_proj.weight": randn(8, hidden),
        "model.embed_gated_norm.up_proj.weight": randn(hidden, 8),
        "model.norm.weight": randn(hidden),
        "model.final_gated_norm.down_proj.weight": randn(8, hidden),
        "model.final_gated_norm.up_proj.weight": randn(hidden, 8),
        "lm_head.weight": randn(VOCAB, hidden),
    }
    for layer in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer}"
        for name, shape in {
            "input_layernorm.weight": (hidden,),
            "post_attention_layernorm.weight": (hidden,),
            "attn_gated_norm.down_proj.weight": (8, hidden),
            "mlp_gated_norm.up_proj.weight": (hidden, 8),
            "self_attn.q_proj.weight": (32, hidden),
            "self_attn.k_proj.weight": (16, hidden),
            "self_attn.v_proj.weight": (16, hidden),
            "self_attn.o_proj.weight": (hidden, 32),
            "self_attn.attn_gate.weight": (4, hidden),
            "self_attn.sconv_k.weight": (4, 16),
            "sconv_attn.weight": (4, hidden),
            "sconv_mlp.weight": (4, hidden),
            "mlp.router.weight": (config.num_local_experts, hidden),
            "mlp.latent_down_proj.weight": (latent, hidden),
            "mlp.latent_up_proj.weight": (hidden, latent),
            "mlp.latent_norm.weight": (latent,),
        }.items():
            state[f"{prefix}.{name}"] = randn(*shape)
        state[f"{prefix}.mlp.router.bias"] = torch.randn(config.num_local_experts, generator=generator)
        for expert in range(config.num_local_experts):
            for projection, shape in (
                ("gate_proj", (ffn, latent)),
                ("up_proj", (ffn, latent)),
                ("down_proj", (latent, ffn)),
            ):
                state[f"{prefix}.mlp.experts.{expert}.{projection}.weight"] = randn(*shape)
        for expert in range(config.num_shared_experts):
            for projection in ("gate_proj", "up_proj", "down_proj"):
                state[f"{prefix}.shared_experts.{expert}.{projection}.weight"] = randn(24, hidden)
    return state


def _shard_of(key: str) -> str:
    """Three shards: kept-only, kept/dropped mixed (layer 3 | 4), and dropped layers plus head tensors."""
    if key.startswith("model.layers."):
        layer = int(key.split(".")[2])
        return f"model-0000{1 + (layer >= 3) + (layer >= 6)}-of-00003.safetensors"
    if key.startswith("model.embed"):
        return "model-00001-of-00003.safetensors"
    return "model-00003-of-00003.safetensors"


def _write_hero_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    config = _hero_config()
    state = _hero_state(config)
    path.mkdir()
    config.save_pretrained(path)
    _save_tokenizer(path)
    weight_map = {key: _shard_of(key) for key in state}
    for shard in sorted(set(weight_map.values())):
        save_file({key: state[key] for key, value in weight_map.items() if value == shard}, path / shard)
    (path / HF_WEIGHT_INDEX_FILENAME).write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    write_local_model_manifest(path)
    return state


def _output_location(tmp_path: Path, storage: str) -> tuple[str, Path]:
    """Return the truncation target and a local directory that will hold its contents."""
    local = tmp_path / "shallow"
    if storage == "local":
        return str(local), local
    return f"memory://truncate-{uuid4().hex}/shallow", local


def _materialize(output: str, local: Path) -> None:
    if output.startswith("memory://"):
        fsspec.filesystem("memory").get(output.removeprefix("memory:/") + "/", str(local), recursive=True)


@pytest.mark.parametrize("storage", ["local", "object_store"])
def test_hero_truncation_keeps_layer_prefix_and_all_non_layer_weights(tmp_path, storage):
    source = tmp_path / "hero"
    state = _write_hero_checkpoint(source)
    output, local = _output_location(tmp_path, storage)

    provenance = truncate_checkpoint(str(source), output, KEPT_LAYERS, max_concurrent_shards=2)
    _materialize(output, local)

    kept = {
        key: value
        for key, value in state.items()
        if not key.startswith("model.layers.") or int(key.split(".")[2]) < KEPT_LAYERS
    }
    index = json.loads((local / HF_WEIGHT_INDEX_FILENAME).read_text())["weight_map"]
    assert set(index) == set(kept)
    # The third shard mixes dropped layers with the LM head and final norm, so it is rewritten, not skipped.
    assert len(set(index.values())) == 3
    loaded = {}
    for shard in set(index.values()):
        shard_tensors = load_file(local / shard)
        assert {key for key, value in index.items() if value == shard} == set(shard_tensors)
        loaded.update(shard_tensors)
    assert loaded.keys() == kept.keys()
    for key, value in kept.items():
        assert torch.equal(loaded[key], value), key

    source_config = json.loads((source / "config.json").read_text())
    output_config = json.loads((local / "config.json").read_text())
    assert output_config["num_hidden_layers"] == output_config["num_layers"] == KEPT_LAYERS
    unchanged = {key for key in source_config if key not in ("num_hidden_layers", "num_layers")}
    assert {key: output_config[key] for key in unchanged} == {key: source_config[key] for key in unchanged}
    config = GrugMoeConfig.from_pretrained(local)
    assert config.num_hidden_layers == KEPT_LAYERS
    assert config.uses_hero_architecture

    assert AutoTokenizer.from_pretrained(local).convert_tokens_to_ids("t7") == 7
    assert read_truncation_provenance(output) == provenance
    assert provenance.source_path == str(source)
    assert provenance.source_config_sha256 == sha256_file(source / "config.json")
    assert (provenance.source_num_layers, provenance.num_layers) == (SOURCE_LAYERS, KEPT_LAYERS)
    manifest_path = local / MODEL_MANIFEST_FILENAME
    manifest = ModelManifest.from_mapping(json.loads(manifest_path.read_text()), str(manifest_path))
    assert {entry.path for entry in manifest.files} == {
        path.name for path in local.iterdir() if path.name != MODEL_MANIFEST_FILENAME
    }
    assert all(sha256_file(local / entry.path) == entry.sha256 for entry in manifest.files)


def test_truncation_reads_no_shard_whose_tensors_are_all_dropped(tmp_path):
    source = tmp_path / "hero"
    state = _write_hero_checkpoint(source)
    # Index every layer >= 6 tensor into a shard file that does not exist.
    weight_map = json.loads((source / HF_WEIGHT_INDEX_FILENAME).read_text())["weight_map"]
    for key in weight_map:
        if key.startswith("model.layers.") and int(key.split(".")[2]) >= 6:
            weight_map[key] = "dropped.safetensors"
    head = {key: state[key] for key, shard in weight_map.items() if shard == "model-00003-of-00003.safetensors"}
    save_file(head, source / "model-00003-of-00003.safetensors")
    (source / HF_WEIGHT_INDEX_FILENAME).write_text(json.dumps({"weight_map": weight_map}))
    (source / MODEL_MANIFEST_FILENAME).unlink()

    truncate_checkpoint(str(source), str(tmp_path / "shallow"), KEPT_LAYERS, max_concurrent_shards=1)

    index = json.loads((tmp_path / "shallow" / HF_WEIGHT_INDEX_FILENAME).read_text())["weight_map"]
    assert "lm_head.weight" in index and "model.layers.3.mlp.router.bias" in index


def test_truncated_model_reproduces_full_model_hidden_states(tmp_path):
    torch.manual_seed(11)
    model = GrugMoeForCausalLM(_snowball_config())
    with torch.no_grad():
        for layer in model.model.layers:
            layer.mlp.router.bias.normal_()
    source = tmp_path / "snowball"
    # Small shards put kept and dropped layers in the same files.
    model.save_pretrained(source, max_shard_size="40KB")
    _save_tokenizer(source)
    assert len(set(json.loads((source / HF_WEIGHT_INDEX_FILENAME).read_text())["weight_map"].values())) > 4

    truncate_checkpoint(str(source), str(tmp_path / "shallow"), KEPT_LAYERS, max_concurrent_shards=3)

    full = GrugMoeForCausalLM.from_pretrained(source, attn_implementation="eager").eval()
    shallow = GrugMoeForCausalLM.from_pretrained(tmp_path / "shallow", attn_implementation="eager").eval()
    tokens = torch.randint(0, VOCAB, (2, 12), generator=torch.Generator().manual_seed(3))
    mask = torch.ones_like(tokens)
    mask[1, :4] = 0
    with torch.no_grad():
        full_hidden = full(tokens, attention_mask=mask, output_hidden_states=True).hidden_states
        shallow_hidden = shallow(tokens, attention_mask=mask, output_hidden_states=True).hidden_states
    assert len(shallow_hidden) == KEPT_LAYERS + 2
    for layer in range(KEPT_LAYERS + 1):
        torch.testing.assert_close(shallow_hidden[layer], full_hidden[layer], rtol=0, atol=0)


def test_rejects_prefix_ending_on_sliding_window_layer(tmp_path):
    source = tmp_path / "hero"
    _write_hero_checkpoint(source)

    with pytest.raises(ValueError, match="global_every"):
        truncate_checkpoint(str(source), str(tmp_path / "shallow"), 3, max_concurrent_shards=1)
    assert not (tmp_path / "shallow").exists()


def test_rejects_per_layer_config_schedule(tmp_path):
    source = tmp_path / "hero"
    _write_hero_checkpoint(source)
    config = json.loads((source / "config.json").read_text())
    config["layer_types"] = ["sliding_attention"] * SOURCE_LAYERS
    (source / "config.json").write_text(json.dumps(config))

    with pytest.raises(ValueError, match="layer_types"):
        truncate_checkpoint(str(source), str(tmp_path / "shallow"), KEPT_LAYERS, max_concurrent_shards=1)


def test_slice_routes_keeps_leading_layers():
    rows = tuple(np.arange(n * SOURCE_LAYERS * 2, dtype=np.uint8).reshape(n, SOURCE_LAYERS, 2) for n in (3, 5))
    routes = RoutedExpertRows(rows, response_len=6, num_experts=128)

    sliced = slice_routes(routes, KEPT_LAYERS)

    assert sliced.shape == (2, 6, KEPT_LAYERS, 2)
    assert (sliced.response_len, sliced.num_experts) == (6, 128)
    for original, kept in zip(rows, sliced.rows, strict=True):
        np.testing.assert_array_equal(kept, original[:, :KEPT_LAYERS])
    with pytest.raises(ValueError, match="fewer than"):
        slice_routes(routes, SOURCE_LAYERS + 1)
