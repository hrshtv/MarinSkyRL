import pytest
from omegaconf import OmegaConf

import skyrl_train.objective.losses  # noqa: F401
from skyrl_train.config.numerics import NumericsMode, resolve_numerics
from skyrl_train.config.utils import get_default_config
from skyrl_train.utils.utils import validate_cfg


def _config(**settings):
    cfg = get_default_config()
    cfg.trainer.logger = "console"
    for path, value in settings.items():
        # engine_init_kwargs holds arbitrary vLLM engine kwargs, so new keys may be added under it.
        OmegaConf.update(cfg, path, value, force_add=path == "generator.engine_init_kwargs")
    return cfg


def test_exact_numerics_resolve_with_snowball_engine_settings():
    # The Snowball engine settings an exact run keeps: FLASH_ATTN named in the config, Triton MoE kernels, TP 1
    # engines on their own GPUs.
    cfg = _config(
        **{
            "trainer.numerics.mode": "exact",
            "trainer.placement.colocate_all": False,
            "generator.vllm_attention_backend": "FLASH_ATTN",
            "generator.inference_engine_tensor_parallel_size": 1,
            "generator.engine_init_kwargs": {"moe_backend": "triton"},
        }
    )
    validate_cfg(cfg)

    assert resolve_numerics(cfg).mode is NumericsMode.EXACT


def test_validation_rejects_stop_without_a_bit_identical_mode():
    cfg = _config(**{"trainer.numerics.check.on_failure": "stop"})

    with pytest.raises(ValueError, match="on_failure=stop"):
        validate_cfg(cfg)
