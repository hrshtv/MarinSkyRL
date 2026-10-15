"""
uv run --isolated --group dev --extra cpu pytest tests/cpu/test_eval.py
"""

from unittest.mock import MagicMock

from torchdata.stateful_dataloader import StatefulDataLoader

import pytest
from omegaconf import OmegaConf

from skyrl_train.evaluate import evaluate
import skyrl_train.evaluate as evaluate_module
from skyrl_train.trajectory_runners.base import TrajectoryRunner, TrajectoryBatch
from tests.cpu.util import example_dummy_config

# A standalone evaluation starts the run's trajectory sink actor.
pytestmark = pytest.mark.usefixtures("ray_module")


@pytest.fixture
def dummy_config():
    return example_dummy_config()


def configure_eval(cfg, tmp_path):
    cfg.generator.backend = "vllm"
    cfg.generator.eval_sampling_params = OmegaConf.create(
        {
            "max_generate_length": 20,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "min_p": 0.0,
            "logprobs": None,
            "stop": None,
        }
    )
    cfg.generator.eval_n_samples_per_prompt = 1
    cfg.environment = OmegaConf.create({"env_class": "gsm8k"})
    cfg.trainer.pivot_pilot = None
    cfg.trainer.dump_eval_results = False
    cfg.trainer.export_path = str(tmp_path)
    return cfg


class DummyStatefulDataLoader:
    def __init__(self, batches):
        self._batches = batches

    def __len__(self):
        return len(self._batches)

    def __iter__(self):
        return iter(self._batches)


class DummyRunner(TrajectoryRunner):
    def __init__(self, output: TrajectoryBatch):
        self.output = output
        self.seen_inputs = []

    async def _run(self, input_batch, disable_tqdm: bool = False):
        self.seen_inputs.append(input_batch)
        return self.output


class FailingRunner(TrajectoryRunner):
    async def _run(self, input_batch, disable_tqdm: bool = False):
        raise RuntimeError("evaluation actor failed")


@pytest.mark.asyncio
async def test_evaluate_computes_expected_metrics(dummy_config, tmp_path):
    cfg = configure_eval(dummy_config, tmp_path)

    prompts_batch = [
        {
            "prompt": [{"role": "user", "content": "question-1"}],
            "env_class": None,
            "env_extras": {"data_source": "dataset/a"},
            "uid": "uid-1",
        },
        {
            "prompt": [{"role": "user", "content": "question-2"}],
            "env_class": "custom_env",
            "env_extras": {"data_source": "dataset/b"},
            "uid": "uid-2",
        },
    ]
    eval_dataloader = DummyStatefulDataLoader([prompts_batch])

    trajectory_batch: TrajectoryBatch = {
        "prompt_token_ids": [[101], [102]],
        "response_ids": [[201], [202]],
        "rewards": [1.0, 0.0],
        "loss_masks": [[1], [1]],
        "stop_reasons": ["stop", "stop"],
        "rollout_logprobs": None,
    }
    runner = DummyRunner(trajectory_batch)

    tokenizer = MagicMock()
    tokenizer.decode.side_effect = lambda tokens: "decoded"

    metrics = await evaluate(
        eval_dataloader=eval_dataloader,
        trajectory_runner=runner,
        cfg=cfg,
        global_step=5,
        tokenizer=tokenizer,
    )

    expected_metrics = {
        "eval/dataset_a/avg_score": 1.0,
        "eval/dataset_a/pass_at_1": 1.0,
        "eval/dataset_b/avg_score": 0.0,
        "eval/dataset_b/pass_at_1": 0.0,
        "eval/all/avg_score": 0.5,
        "eval/all/pass_at_1": 0.5,
    }

    for key, expected_value in expected_metrics.items():
        assert metrics[key] == pytest.approx(expected_value)

    assert len(runner.seen_inputs) == 1
    seen_batch = runner.seen_inputs[0]
    assert seen_batch["prompts"] == [prompt["prompt"] for prompt in prompts_batch]
    assert seen_batch["env_classes"] == ["gsm8k", "custom_env"]
    assert seen_batch["env_extras"] == [prompt["env_extras"] for prompt in prompts_batch]
    assert seen_batch["batch_metadata"].training_phase == "eval"


@pytest.mark.asyncio
async def test_evaluate_closes_progress_reporter_when_rollout_fails(dummy_config, tmp_path, monkeypatch):
    cfg = configure_eval(dummy_config, tmp_path)
    eval_dataloader = DummyStatefulDataLoader(
        [[{"prompt": [{"role": "user", "content": "question"}], "env_class": None, "env_extras": {}, "uid": "1"}]]
    )
    progress = MagicMock()
    monkeypatch.setattr(evaluate_module, "tqdm", lambda **_kwargs: progress)

    with pytest.raises(RuntimeError, match="evaluation actor failed"):
        await evaluate(
            eval_dataloader=eval_dataloader,
            trajectory_runner=FailingRunner(),
            cfg=cfg,
            global_step=5,
            tokenizer=MagicMock(),
        )

    progress.close.assert_called_once_with()


@pytest.mark.asyncio
@pytest.mark.parametrize("validation_rows", [256, 508])
async def test_pivot_evaluation_covers_the_configured_holdout(dummy_config, tmp_path, validation_rows):
    cfg = configure_eval(dummy_config, tmp_path)
    cfg.trainer.pivot_pilot = OmegaConf.create(
        {
            "arm": "rl_nemo",
            "baseline_cache": None,
            "split_hash": "validation",
            "quick_hash": "quick",
            "quick_source_ids": [str(i) for i in range(64)],
        }
    )
    cfg.trainer.policy.model.path = "test-model"
    cfg.generator.trajectory_retention.model_source_identity = "test-revision"
    prompts = [
        {
            "prompt": [{"role": "user", "content": "question"}],
            "env_class": None,
            "env_extras": {"data_source": "swe", "extra_info": {"source_id": str(i)}},
            "uid": str(i),
        }
        for i in range(validation_rows)
    ]
    loader = StatefulDataLoader(prompts, batch_size=validation_rows, collate_fn=list)
    runner = DummyRunner(
        {
            "prompt_token_ids": [[101]] * validation_rows,
            "response_ids": [[201]] * validation_rows,
            "rewards": [float(i % 2) for i in range(validation_rows)],
            "loss_masks": [[1]] * validation_rows,
            "stop_reasons": ["stop"] * validation_rows,
            "rollout_logprobs": None,
        }
    )
    tokenizer = MagicMock()
    tokenizer.decode.return_value = "decoded"
    metrics = await evaluate(loader, runner, cfg, 0, tokenizer)
    assert metrics["eval/all/avg_score"] == 0.5
    assert sum(len(batch["prompts"]) for batch in runner.seen_inputs) == validation_rows
