"""The policies and run settings the exact Grug trainer reproduces the decode-invariant engines on."""

from omegaconf import DictConfig
from transformers import PretrainedConfig

from skyrl_train.config.numerics import NumericsMode, NumericsResolution
from skyrl_train.config.weight_sync_pause import resolve_weight_sync_pause_policy
from skyrl_train.models.grug_exact.inductor_kernels import COMPILED_MODEL
from skyrl_train.models.grug_moe import GRUG_MOE_MODEL_TYPE, GrugMoeConfig

ALLTOALL_DISPATCHER = "alltoall"
FULL_RECOMPUTE = "full"
# The deepest pipeline the stage statistic hand-off is verified on.
MAX_PIPELINE_STAGES = 2
# Megatron settings the exact trainer takes from the model config, as the engines do.
MODEL_CONFIG_TRANSFORMER_SETTINGS = (
    "hidden_size",
    "num_attention_heads",
    "num_query_groups",
    "kv_channels",
    "layernorm_epsilon",
    "softmax_scale",
    "window_size",
    "window_attn_skip_freq",
    "no_rope_freq",
    "rotary_base",
    "rotary_interleaved",
    "ffn_hidden_size",
    "moe_ffn_hidden_size",
    "moe_shared_expert_intermediate_size",
    "num_moe_experts",
    "moe_router_topk",
    "grug_qk_mult",
    "grug_qk_mult_long_scale",
)


def policy_model_config(cfg: DictConfig) -> PretrainedConfig:
    """The policy's model config, read from ``trainer.policy.model.path`` without running model code."""
    model = cfg.trainer.policy.model
    values, _ = PretrainedConfig.get_config_dict(model.path, revision=model.get("revision"))
    config_class = GrugMoeConfig if values.get("model_type") == GRUG_MOE_MODEL_TYPE else PretrainedConfig
    return config_class.from_dict(values)


def exact_model_problems(model_config: PretrainedConfig) -> list[str]:
    """Why the exact trainer's kernels do not compute this policy: they compute Grug at the compiled shapes without
    Hero layers, ShortConv or latent experts."""
    if model_config.model_type != GRUG_MOE_MODEL_TYPE:
        return [f"the policy is {model_config.model_type}, not Grug"]
    problems = [
        f"the policy's {name} is {getattr(model_config, name)}, not the compiled kernels' {value}"
        for name, value in COMPILED_MODEL.items()
        if getattr(model_config, name) != value
    ]
    if model_config.uses_hero_architecture:
        problems.append("the policy uses Hero layers, ShortConv or latent experts")
    return problems


def resolve_exact_numerics(model_config: PretrainedConfig) -> NumericsResolution:
    """``exact`` for a policy the exact trainer computes, otherwise ``native`` with the reasons."""
    reasons = exact_model_problems(model_config)
    if reasons:
        return NumericsResolution(NumericsMode.NATIVE, tuple(reasons))
    return NumericsResolution(NumericsMode.EXACT)


def exact_trainer_problems(cfg: DictConfig) -> list[str]:
    """The settings the exact trainer runs on that ``cfg`` does not have."""
    generator, trainer = cfg.generator, cfg.trainer
    megatron = trainer.policy.megatron_config
    transformer = megatron.transformer_config_kwargs
    recompute = tuple(
        transformer.get(name) for name in ("recompute_granularity", "recompute_method", "recompute_num_layers")
    )
    overrides = [name for name in MODEL_CONFIG_TRANSFORMER_SETTINGS if name in transformer]
    requirements = {
        "generator.rope_scaling and generator.rope_theta unset": (
            not generator.get("rope_scaling") and generator.get("rope_theta") is None
        ),
        "generator.sampling_params.temperature=1": generator.sampling_params.temperature == 1.0,
        "generator.weight_sync_pause.clear_cache=true": resolve_weight_sync_pause_policy(generator).clear_cache,
        "trainer.use_sample_packing=false": not trainer.use_sample_packing,
        "trainer.flash_attn=false": not trainer.flash_attn,
        "tensor_model_parallel_size=1": megatron.tensor_model_parallel_size == 1,
        "context_parallel_size=1": megatron.context_parallel_size == 1,
        "expert_tensor_parallel_size=1": megatron.get("expert_tensor_parallel_size") in (None, 1),
        f"pipeline_model_parallel_size at most {MAX_PIPELINE_STAGES}": (
            megatron.pipeline_model_parallel_size <= MAX_PIPELINE_STAGES
        ),
        "ddp_config.overlap_param_gather=false": not megatron.ddp_config.overlap_param_gather,
        f"moe_token_dispatcher_type={ALLTOALL_DISPATCHER}": (
            transformer.get("moe_token_dispatcher_type", ALLTOALL_DISPATCHER) == ALLTOALL_DISPATCHER
        ),
        "activation recompute off or recompute_granularity=full, recompute_method=uniform, recompute_num_layers=1": (
            not trainer.gradient_checkpointing or recompute == (FULL_RECOMPUTE, "uniform", 1)
        ),
        f"transformer_config_kwargs without {', '.join(overrides)}": not overrides,
    }
    return [requirement for requirement, met in requirements.items() if not met]
