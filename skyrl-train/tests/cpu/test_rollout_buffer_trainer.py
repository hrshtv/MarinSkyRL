import asyncio
import io
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from skyrl_train.config.utils import get_default_config
from skyrl_train.rollouts.buffer import RolloutGroup
from skyrl_train.rollouts.context import TrainingContextState
from skyrl_train.rollouts.loader import PromptLoaderState
from skyrl_train.trainer import RayPPOTrainer
import skyrl_train.trainer as trainer_module
from skyrl_train.inference_engines.vllm.fixed_sampler_noise import FixedSamplerNoise
from skyrl_train.trajectory_runners.base import TrajectoryID
from skyrl_train.trajectory_selection import BestOfNTrajectorySelector


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "noise_scale,refresh_period,saved_step,sampler_step",
    [(0.0, 1, 5, 5), (0.05, 1, 5, 5), (0.0, 3, 2, 0), (0.0, 3, 5, 3), (0.0, 3, 6, 6), (0.05, 3, 5, 3)],
)
async def test_checkpoint_resume_preserves_original_noise_and_optimizer_state(
    tmp_path, monkeypatch, noise_scale, refresh_period, saved_step, sampler_step
):
    original = torch.arange(1, 4, dtype=torch.float32)
    restored_weights = original * (saved_step + 2)
    sampler_weights = original * (sampler_step + 2) if sampler_step else original
    moments = torch.tensor([0.3, 0.6, 0.9])
    checkpoint = tmp_path / "source" / f"global_step_{saved_step}"
    policy_checkpoint = checkpoint / "policy"
    policy_checkpoint.mkdir(parents=True)
    torch.save({"weights": restored_weights, "moments": moments}, policy_checkpoint / "state.pt")
    sampler_checkpoint = checkpoint.parent / f"global_step_{sampler_step}" / "policy" if sampler_step else None
    if sampler_checkpoint is not None and sampler_step != saved_step:
        sampler_checkpoint.mkdir(parents=True)
        torch.save({"weights": sampler_weights, "moments": moments * 100}, sampler_checkpoint / "state.pt")
        torch.save({"global_step": sampler_step}, sampler_checkpoint.parent / "trainer_state.pt")
    torch.save(
        {
            "global_step": saved_step,
            "last_evaluated_step": saved_step,
            "sampler_state": {
                "completed_updates": sampler_step,
                "checkpoint_path": str(sampler_checkpoint) if sampler_checkpoint is not None else None,
            },
        },
        checkpoint / "trainer_state.pt",
    )
    rollout_state = TrainingContextState(
        loader=PromptLoaderState(order={"epoch": 0, "position": 6}, retries=[]),
        ready=[],
        object_store_root="s3://fixture/groups",
    )
    torch.save(rollout_state, checkpoint / "data.pt")
    serving = original.clone()

    class Policy:
        def __init__(self):
            self.weights = original.clone()
            self.moments = torch.zeros_like(moments)

        def offload_to_cpu(self, *, offload_optimizer, offload_model):
            pass

        def async_run_ray_method(self, dispatch, method, **kwargs):
            if method == "barrier_all":
                return []
            if method == "save_checkpoint":
                destination = Path(kwargs["ckpt_dir"])
                destination.mkdir(parents=True, exist_ok=True)
                torch.save({"weights": self.weights, "moments": self.moments}, destination / "state.pt")
                return [None]
            assert method == "load_checkpoint"
            saved = torch.load(Path(kwargs["ckpt_dir"]) / "state.pt", weights_only=True)
            self.weights.copy_(saved["weights"])
            if kwargs["load_training_state"]:
                self.moments.copy_(saved["moments"])
            return [None]

        async def async_run_method(self, dispatch, method, client):
            assert method == "broadcast_to_inference_engines"
            serving.copy_(self.weights)

    class Engine:
        def __init__(self):
            self.noise = None

        async def pause_generation(self):
            pass

        async def resume_generation(self):
            pass

        async def sampler_noise(self, action, scale, seed):
            if action == "initialize":
                self.noise = FixedSamplerNoise([("weight", serving)], scale, seed)
            else:
                assert action == "refresh"
                self.noise.after_sync()
            return self.noise.evidence()

    class Context:
        restored = None

        async def load_state_dict(self, state):
            self.restored = state

    trainer = object.__new__(RayPPOTrainer)
    trainer.cfg = get_default_config()
    trainer.cfg.generator.sampler_weight_noise_scale = noise_scale
    trainer.cfg.generator.sampler_weight_noise_seed = 17
    trainer.cfg.generator.sampler_refresh_period = refresh_period
    trainer.cfg.trainer.resume_mode = "from_path"
    trainer.cfg.trainer.resume_path = str(checkpoint)
    trainer.cfg.trainer.ckpt_path = str(tmp_path / "resumed")
    trainer.cfg.trainer.restore_dataloader_state = True
    trainer.cfg.trainer.reset_global_step_on_resume = False
    trainer.resume_mode = trainer_module.ResumeMode.FROM_PATH
    trainer.colocate_all = False
    trainer.global_step = 0
    trainer.all_startup_timings = {}
    trainer.all_timings = {}
    trainer.all_metrics = {}
    trainer._domain_balancer = None
    trainer._restored_rollout_state = None
    trainer._resume_evaluation_progress = True
    trainer.policy_model = Policy()
    trainer.critic_model = None
    trainer.inference_engine_client = Engine()
    trainer.context = Context()
    trainer.speculative_decoding = None
    trainer.tokenizer = None
    monkeypatch.setattr(trainer_module.ray, "get", lambda references: references)

    # The reference sampler starts at the original model and then receives a later learner snapshot.
    expected = original.clone()
    reference_noise = FixedSamplerNoise([("weight", expected)], noise_scale, 17)
    expected.copy_(sampler_weights)
    reference_noise.after_sync()

    assert await trainer._initialize_training_state()

    torch.testing.assert_close(serving, expected, rtol=0, atol=0)
    torch.testing.assert_close(trainer.policy_model.weights, restored_weights, rtol=0, atol=0)
    torch.testing.assert_close(trainer.policy_model.moments, moments, rtol=0, atol=0)
    assert trainer.context.restored == rollout_state
    assert trainer.global_step == saved_step
    snapshot = trainer._snapshot_checkpoint(rollout_state)
    saved = torch.load(io.BytesIO(snapshot.trainer_state_payload), weights_only=False)
    assert saved["last_evaluated_step"] == saved_step
    if refresh_period > 1:
        assert saved["sampler_state"] == {
            "completed_updates": sampler_step,
            "checkpoint_path": str(sampler_checkpoint) if sampler_checkpoint is not None else None,
        }
    else:
        assert saved.get("sampler_state") is None


def _group(uid: str, policy_step: int, *, rewards: list[float] | None = None) -> RolloutGroup:
    batch = {
        "prompt_token_ids": [[1], [1]],
        "response_ids": [[2], [3]],
        "rewards": rewards or [0.0, 1.0],
        "loss_masks": [[1], [1]],
        "stop_reasons": ["stop", "stop"],
        "rollout_metrics": {},
        "rollout_logprobs": None,
        "trajectory_ids": [TrajectoryID(instance_id=uid, repetition_id=index) for index in range(2)],
    }
    return RolloutGroup(batch, uid, policy_step, {"uid": uid})


@pytest.mark.parametrize("reason", ["initial", "training_step"])
@pytest.mark.parametrize("offload_enabled", [False, True])
def test_weight_sync_respects_optimizer_offload_policy(reason, offload_enabled):
    trainer = object.__new__(RayPPOTrainer)
    trainer.cfg = SimpleNamespace(
        generator={}, trainer=SimpleNamespace(offload_optimizer_during_rollouts=offload_enabled)
    )
    trainer.colocate_all = False
    trainer.global_step = 0
    trainer.all_startup_timings = {}
    trainer.all_timings = {}
    trainer.all_metrics = {}
    events = []

    class Policy:
        optimizer_on_gpu = True

        def offload_to_cpu(self, *, offload_optimizer, offload_model):
            assert offload_optimizer and not offload_model
            self.optimizer_on_gpu = False
            events.append("offload")

    class Engine:
        async def pause_generation(self):
            events.append("pause")

        async def resume_generation(self):
            assert trainer.policy_model.optimizer_on_gpu != offload_enabled
            events.append("resume")

    async def sync_weights():
        events.append("sync")

    trainer.policy_model = Policy()
    trainer.inference_engine_client = Engine()
    trainer.sync_policy_weights_to_inference_engines = sync_weights

    asyncio.run(trainer._sync_policy_for_rollouts(reason=reason))

    assert trainer.policy_model.optimizer_on_gpu != offload_enabled
    paused = reason == "training_step"
    assert events == (["pause"] if paused else []) + (["offload"] if offload_enabled else []) + ["sync"] + (
        ["resume"] if paused else []
    )


@pytest.mark.asyncio
async def test_periodic_refresh_preserves_simulated_sampler_weights_across_evaluation(monkeypatch, tmp_path):
    trainer = object.__new__(RayPPOTrainer)
    trainer.cfg = OmegaConf.create(
        {
            "generator": {"sampler_refresh_period": 3, "sampler_weight_noise_scale": 0, "sampler_weight_noise_seed": 0},
            "trainer": {
                "offload_optimizer_during_rollouts": True,
                "step_wise_training": False,
                "ckpt_path": str(tmp_path),
            },
        }
    )
    trainer.colocate_all = False
    trainer.global_step = 0
    trainer.all_startup_timings = {}
    trainer.all_timings = {}
    trainer.all_metrics = {}
    serving = torch.zeros(3)

    class Policy:
        weights = torch.zeros(3)
        optimizer_on_gpu = True

        def offload_to_cpu(self, *, offload_optimizer, offload_model):
            assert offload_optimizer and not offload_model
            self.optimizer_on_gpu = False

        def async_run_ray_method(self, dispatch, method):
            assert dispatch == "pass_through" and method == "barrier_all"
            return []

        async def async_run_method(self, dispatch, method, client):
            assert dispatch == "pass_through" and method == "broadcast_to_inference_engines"
            serving.copy_(self.weights)

    class Engine:
        noise = None

        async def pause_generation(self):
            pass

        async def resume_generation(self):
            pass

        async def sampler_noise(self, action, scale, seed):
            assert scale == 0
            if action == "initialize":
                self.noise = FixedSamplerNoise([("weight", serving)], scale, seed)
            elif action == "refresh":
                self.noise.after_sync()
            elif action == "snapshot":
                self.noise.snapshot_for_evaluation()
            elif action == "restore":
                self.noise.restore_after_evaluation()
            elif action == "clean":
                self.noise.set_enabled(False)
            else:
                raise AssertionError(action)
            return {"policy_weights": serving.tolist()}

    trainer.policy_model = Policy()
    trainer.inference_engine_client = Engine()
    trainer.eval_dataloader = None
    trainer.trajectory_runner = None
    trainer.tokenizer = None
    trainer.trajectory_sink = None

    async def evaluate_current_policy(**kwargs):
        torch.testing.assert_close(serving, trainer.policy_model.weights, rtol=0, atol=0)
        return {"score": 1.0}

    monkeypatch.setattr(trainer_module, "evaluate", evaluate_current_policy)
    await trainer._sync_policy_for_rollouts(reason="initial")
    for step in range(1, 8):
        trainer.global_step = step
        trainer.policy_model.weights.fill_(step)
        trainer.policy_model.optimizer_on_gpu = True
        await trainer._sync_policy_for_rollouts(reason="training_step")
        expected = 3 * (step // 3)
        torch.testing.assert_close(serving, torch.full((3,), float(expected)), rtol=0, atol=0)
        assert trainer._sampler_refresh_step == expected
        assert not trainer.policy_model.optimizer_on_gpu
        if step == 2:
            assert await trainer.eval() == {"score": 1.0}
            torch.testing.assert_close(serving, torch.zeros(3), rtol=0, atol=0)
            assert trainer._sampler_refresh_step == 0


@pytest.mark.parametrize("step,policy_step", [(1, 1), (2, 1), (3, 1), (4, 4), (5, 4), (6, 4)])
def test_frozen_sampler_batch_conversion_accepts_only_the_current_refresh_cycle(step, policy_step):
    trainer = object.__new__(RayPPOTrainer)
    trainer.context = SimpleNamespace(config=SimpleNamespace(batch_size=1, max_staleness_steps=0))
    trainer.cfg = get_default_config()
    trainer.cfg.generator.sampler_refresh_period = 3
    trainer.cfg.trainer.algorithm.off_policy_correction = "none"
    trainer.global_step = step
    trainer.all_metrics = {}
    trainer.all_timings = {}
    trainer.tokenizer = SimpleNamespace(decode=str)
    trainer.postprocess_trajectory_batch = lambda batch, uids: batch
    trainer.select_trajectories = lambda batch, uids: (batch, uids)
    trainer.convert_to_training_input = lambda batch, uids, *, rollout_staleness: rollout_staleness

    assert trainer.convert_rollout_groups_to_training_input([_group("valid", policy_step)]) == [
        step - policy_step,
        step - policy_step,
    ]
    for wrong_step in [policy_step - 1, policy_step + 1]:
        with pytest.raises(AssertionError, match="frozen sampler batch must use policy step"):
            trainer.convert_rollout_groups_to_training_input([_group("wrong", wrong_step)])


def test_rollout_batch_conversion_reports_staleness_and_stage_timings(monkeypatch):
    trainer = object.__new__(RayPPOTrainer)
    trainer.context = SimpleNamespace(config=SimpleNamespace(batch_size=2, max_staleness_steps=2))
    trainer.cfg = SimpleNamespace(trainer=SimpleNamespace(algorithm=get_default_config().trainer.algorithm))
    trainer.cfg.trainer.algorithm.off_policy_correction = "none"
    trainer.global_step = 10
    trainer.all_metrics = {}
    trainer.all_timings = {}
    trainer.tokenizer = SimpleNamespace(decode=lambda response: str(response))
    now = [0.0]
    monkeypatch.setattr("skyrl_train.utils.utils.time", SimpleNamespace(monotonic=lambda: now[0]))

    def postprocess(batch, uids):
        now[0] += 7.0
        return batch

    def select(batch, uids):
        now[0] += 11.0
        return batch, uids

    def convert(batch, uids, *, rollout_staleness):
        now[0] += 3.0
        return {"rewards": batch["rewards"], "uids": uids, "rollout_staleness": rollout_staleness}

    trainer.postprocess_trajectory_batch = postprocess
    trainer.select_trajectories = select
    trainer.convert_to_training_input = convert

    result = trainer.convert_rollout_groups_to_training_input([_group("fresh", 10), _group("stale", 8)])

    assert result == {
        "rewards": [0.0, 1.0, 0.0, 1.0],
        "uids": ["fresh", "fresh", "stale", "stale"],
        "rollout_staleness": [0, 0, 2, 2],
    }
    assert trainer.all_metrics["async/staleness_max"] == 2
    assert trainer.all_metrics["async/staleness_ratio"] == 0.5
    assert trainer.all_timings == {
        "assemble_generation_group_mini_batch": 0.0,
        "postprocess_trajectory_batch": 18.0,
        "convert_to_training_input": 3.0,
    }


def test_rollout_batch_conversion_records_domain_reward_metrics():
    trainer = object.__new__(RayPPOTrainer)
    trainer.context = SimpleNamespace(config=SimpleNamespace(batch_size=3, max_staleness_steps=0))
    trainer.cfg = get_default_config()
    trainer.cfg.trainer.algorithm.policy_loss_type = "regular"
    trainer.cfg.trainer.algorithm.off_policy_correction = "none"
    trainer.cfg.generator.n_samples_per_prompt = 2
    trainer.global_step = 0
    trainer.all_metrics = {}
    trainer.all_timings = {}
    trainer.tokenizer = SimpleNamespace(decode=str)
    trainer.select_trajectories = lambda batch, uids: (batch, uids)
    trainer.convert_to_training_input = lambda batch, uids, *, rollout_staleness: batch
    math = _group("math", 0, rewards=[0.2, 0.6])
    tools = _group("tools", 0, rewards=[0.6, 1.0])
    missing = _group("missing", 0, rewards=[0.4, 0.6])
    math.trajectory_batch["data_sources"] = ["math", "math"]
    tools.trajectory_batch["data_sources"] = ["tools", "tools"]

    trainer.convert_rollout_groups_to_training_input([math, tools, missing])

    assert trainer.all_metrics["reward/domain/math/avg_raw_reward"] == pytest.approx(0.4)
    assert trainer.all_metrics["reward/domain/tools/avg_raw_reward"] == pytest.approx(0.8)
    assert trainer.all_metrics["reward/domain/_missing/avg_raw_reward"] == pytest.approx(0.5)


class _RecordingDistillationRuntime:
    def __init__(self):
        self.submitted = []

    async def submit_before_batch_assembly(self, trajectory_batch):
        self.submitted.append(trajectory_batch)
        return object()


def test_teacher_scores_only_the_rows_the_learner_selects():
    trainer = object.__new__(RayPPOTrainer)
    runtime = _RecordingDistillationRuntime()
    trainer._distillation_runtime = runtime
    trainer._distillation_tickets = {}
    trainer.trajectory_selector = BestOfNTrajectorySelector(2)

    asyncio.run(trainer._submit_admitted_groups_for_teacher_scoring([_group("best", 10, rewards=[0.25, 0.75])]))

    (submitted,) = runtime.submitted
    assert submitted["response_ids"] == [[3]]
    assert submitted["trajectory_ids"][0].repetition_id == 1
    assert set(trainer._distillation_tickets) == {"best"}
