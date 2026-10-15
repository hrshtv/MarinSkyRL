from omegaconf import DictConfig

from marinskyrl.distillation import TeacherSource, compile_distillation_plan_from_config
from marinskyrl.runtime_options import WeightSyncTransport


def dummy_engine_weights_requested(load_format: str | None) -> bool:
    return load_format is not None and load_format.lower() == "dummy"


def validate_dummy_engine_weights(cfg: DictConfig) -> None:
    engine_kwargs = cfg.generator.engine_init_kwargs
    if cfg.generator.backend == "vllm" and engine_kwargs.get("load_format") is not None:
        engine_kwargs["load_format"] = engine_kwargs["load_format"].lower()
    if dummy_engine_weights_requested(engine_kwargs.get("load_format")):
        failures = []
        if cfg.generator.backend != "vllm":
            failures.append("generator.backend=vllm")
        if not cfg.generator.run_engines_locally:
            failures.append("generator.run_engines_locally=true")
        if cfg.trainer.placement.colocate_all:
            failures.append("trainer.placement.colocate_all=false")
        if cfg.generator.weight_sync_transport not in (WeightSyncTransport.BROADCAST, WeightSyncTransport.EXPERT_BLOCK):
            failures.append("generator.weight_sync_transport=broadcast or expert_block")
        if cfg.generator.fuse_weights:
            failures.append("generator.fuse_weights=false")
        if cfg.generator.get("speculative_decoding") or engine_kwargs.get("speculative_config"):
            failures.append("no speculative decoding")
        if engine_kwargs.get("quantization"):
            failures.append("no engine quantization")
        if engine_kwargs.get("kv_cache_dtype", "auto") != "auto":
            failures.append("kv_cache_dtype=auto")
        if engine_kwargs.get("enable_eplb", False):
            failures.append("no EPLB")
        if "model_loader_extra_config" in engine_kwargs:
            failures.append("no model_loader_extra_config")
        distillation_plan = compile_distillation_plan_from_config(cfg)
        if distillation_plan is not None and any(
            teacher.source is TeacherSource.LOCAL_INFERENCE for teacher in distillation_plan.teachers
        ):
            failures.append("no local-inference distillation teachers")
        if failures:
            raise ValueError("Dummy engine weights require: " + "; ".join(failures))
