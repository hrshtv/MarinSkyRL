"""How the trainer and the rollout engines compute, ``trainer.numerics``."""

from dataclasses import dataclass
from enum import StrEnum

from omegaconf import DictConfig, OmegaConf

NUMERICS_KEY = "trainer.numerics"
CHECK_KEY = f"{NUMERICS_KEY}.check"
# Engine kwargs a decode-invariant engine sets itself.
DECODE_INVARIANT_ENGINE_KWARGS = ("attention_backend", "compilation_config", "model_class_overrides")


class NumericsMode(StrEnum):
    """The kernels a run uses: ``native``, the trainer's and the engines' own; ``batch_invariant``, vLLM's
    batch-invariant kernels in both; ``exact``, decode-invariant vLLM engines and a trainer that reproduces their
    log-probabilities bit for bit."""

    NATIVE = "native"
    BATCH_INVARIANT = "batch_invariant"
    EXACT = "exact"

    @property
    def bit_identical(self) -> bool:
        """True for a mode that promises the trainer the engines' log-probabilities bit for bit."""
        return self is NumericsMode.EXACT


class CheckFailure(StrEnum):
    """What the exactness check does when a token's log-probability differs: log it and continue, or stop the run."""

    LOG = "log"
    STOP = "stop"


@dataclass(frozen=True)
class ExactnessCheckConfig:
    """When the online exactness check runs (``every_weight_syncs`` 0 = never) and what a mismatch does."""

    every_weight_syncs: int
    on_failure: CheckFailure

    @property
    def enabled(self) -> bool:
        return self.every_weight_syncs > 0


@dataclass(frozen=True)
class NumericsResolution:
    """The numerics mode a run uses, and why a configured ``exact`` resolved to ``native`` (empty otherwise)."""

    mode: NumericsMode
    fallback_reasons: tuple[str, ...] = ()


def read_numerics(cfg: DictConfig) -> NumericsResolution:
    """The numerics a run of ``cfg`` uses: its configured mode, with ``exact`` resolved for the policy's model config.
    Never writes to ``cfg``."""
    mode = NumericsMode(cfg.trainer.numerics.mode)
    if mode is not NumericsMode.EXACT:
        return NumericsResolution(mode)
    # Imported here: skyrl_train.models imports skyrl_train.utils, which imports this module.
    from skyrl_train.models.grug_exact.support import policy_model_config, resolve_exact_numerics  # noqa: PLC0415

    return resolve_exact_numerics(policy_model_config(cfg))


def exactness_check_config(cfg: DictConfig) -> ExactnessCheckConfig:
    """The exactness check settings of ``cfg``."""
    check = cfg.trainer.numerics.check
    return ExactnessCheckConfig(int(check.every_weight_syncs), CheckFailure(check.on_failure))


def validate_numerics_config(cfg: DictConfig) -> None:
    """Accept a known mode, the other modes than ``native`` only with local vLLM engines, and check settings the mode
    supports."""
    numerics = cfg.trainer.numerics
    try:
        NumericsMode(numerics.mode)
    except ValueError as error:
        choices = ", ".join(mode.value for mode in NumericsMode)
        raise ValueError(f"{NUMERICS_KEY}.mode must be one of {choices}; got {numerics.mode!r}") from error
    resolution = read_numerics(cfg)
    mode = resolution.mode
    if mode is not NumericsMode.NATIVE:
        if cfg.generator.backend != "vllm":
            raise ValueError(f"{NUMERICS_KEY}.mode={mode} requires generator.backend='vllm'")
        if not cfg.generator.run_engines_locally:
            raise ValueError(
                f"{NUMERICS_KEY}.mode={mode} cannot configure a remote inference server; run the vLLM "
                "engines locally so both rollout and trainer activation is guaranteed"
            )
    if mode is NumericsMode.EXACT:
        _validate_exact_numerics(cfg)
    if int(numerics.check.every_weight_syncs) < 0:
        raise ValueError(f"{CHECK_KEY}.every_weight_syncs must be at least 0; got {numerics.check.every_weight_syncs}")
    try:
        check = exactness_check_config(cfg)
    except ValueError as error:
        choices = ", ".join(failure.value for failure in CheckFailure)
        raise ValueError(
            f"{CHECK_KEY}.on_failure must be one of {choices}; got {numerics.check.on_failure!r}"
        ) from error
    if check.enabled and cfg.generator.backend != "vllm":
        raise ValueError(f"the exactness check requires generator.backend='vllm'; got {cfg.generator.backend!r}")
    if check.on_failure is CheckFailure.STOP and not mode.bit_identical:
        raise ValueError(
            f"{CHECK_KEY}.on_failure=stop requires a numerics mode that promises bit-identical log-probabilities; "
            f"{NUMERICS_KEY}.mode={mode} does not" + "".join(f"; {reason}" for reason in resolution.fallback_reasons)
        )


def _validate_exact_numerics(cfg: DictConfig) -> None:
    """Raise unless the rollout engines of ``cfg`` can be decode-invariant and hand rollouts to the trainer in memory,
    and the trainer runs on settings the exact trainer covers."""
    generator = cfg.generator
    kwargs = OmegaConf.to_container(generator.engine_init_kwargs, resolve=True)
    moe_backend = kwargs.get("moe_backend", (kwargs.get("kernel_config") or {}).get("moe_backend"))
    all2all_backend = kwargs.get("all2all_backend", "allgather_reducescatter")
    # vLLM's model runner V2, which a decode-invariant engine runs, refuses logits processors.
    refused_kwargs = [
        key for key in (*DECODE_INVARIANT_ENGINE_KWARGS, "speculative_config", "logits_processors") if kwargs.get(key)
    ]
    requirements = {
        "trainer.rollout_buffer.object_store_root=null": cfg.trainer.rollout_buffer.object_store_root is None,
        "generator.speculative_decoding=null": generator.get("speculative_decoding") is None,
        "generator.vllm_attention_backend null or FLASH_ATTN": generator.vllm_attention_backend in (None, "FLASH_ATTN"),
        "generator.enforce_eager=false": not generator.enforce_eager,
        "generator.inference_engine_tensor_parallel_size=1": generator.inference_engine_tensor_parallel_size == 1,
        "generator.inference_engine_decode_context_parallel_size=1": (
            generator.inference_engine_decode_context_parallel_size == 1
        ),
        "engine_init_kwargs.moe_backend=triton": moe_backend == "triton",
        "engine_init_kwargs.all2all_backend=allgather_reducescatter": all2all_backend == "allgather_reducescatter",
        f"engine_init_kwargs without {', '.join(refused_kwargs)}": not refused_kwargs,
    }
    # Imported here for the cycle read_numerics names.
    from skyrl_train.models.grug_exact.support import exact_trainer_problems  # noqa: PLC0415

    unmet = [requirement for requirement, met in requirements.items() if not met] + exact_trainer_problems(cfg)
    if unmet:
        raise ValueError(f"{NUMERICS_KEY}.mode=exact requires {'; '.join(unmet)}")
